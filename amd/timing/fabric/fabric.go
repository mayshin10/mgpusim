// Package fabric provides a bandwidth- and latency-aware connection that
// models an on-package interconnect (e.g., the MI300X Infinity Fabric between
// XCDs, IODs and HBM stacks).
//
// The Akita v5 switching networks are traffic-only (they deliver metadata, not
// the original messages), and directconnection models neither latency nor
// bandwidth. The fabric is a drop-in messaging.Connection that delivers the
// original messages, but holds each one for the time it takes to cross its
// route. Every port is attached to a node; nodes are joined by directed links.
// A message reserves each link on its shortest route first-come-first-served
// (cut-through): it starts on a link when both its head has arrived and the
// link is free, occupies the link for size/bandwidth, and its head reaches the
// next node one link latency after it started.
package fabric

import (
	"container/heap"
	"fmt"
	"math"

	"github.com/sarchlab/akita/v5/mem/memprotocol"
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/timing"
)

// Spec holds immutable configuration for the fabric.
type Spec struct {
	Freq timing.Freq `json:"freq"`

	// HeaderBytes is the size charged for a message's header (the whole size
	// of requests and write acknowledgements that carry no data).
	HeaderBytes int `json:"header_bytes"`

	// MaxInflightPerPort bounds the messages a single source port may have in
	// the fabric. It back-pressures the sender once its route is saturated.
	MaxInflightPerPort int `json:"max_inflight_per_port"`
}

// State holds mutable runtime state for the fabric.
type State struct {
	NextPortID int `json:"next_port_id"`
}

// LinkStats reports the traffic one directed link carried.
type LinkStats struct {
	From, To    string
	Bytes       uint64
	Msgs        uint64
	BusyPicoSec uint64
}

type link struct {
	from, to    int
	bytesPerSec float64
	latency     timing.VTimeInPicoSec
	nextFree    timing.VTimeInPicoSec

	bytes, msgs uint64
	busy        timing.VTimeInPicoSec
}

type flight struct {
	msg     messaging.Msg
	src     int // index of the source port
	arrival timing.VTimeInPicoSec
	seq     uint64
}

type flightHeap []*flight

func (h flightHeap) Len() int { return len(h) }
func (h flightHeap) Less(i, j int) bool {
	if h[i].arrival != h[j].arrival {
		return h[i].arrival < h[j].arrival
	}

	return h[i].seq < h[j].seq
}
func (h flightHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *flightHeap) Push(x any)   { *h = append(*h, x.(*flight)) }
func (h *flightHeap) Pop() any {
	old := *h
	f := old[len(old)-1]
	*h = old[:len(old)-1]

	return f
}

// Comp is the fabric connection.
type Comp struct {
	*modeling.Component[Spec, State, modeling.None]
}

func (c *Comp) mw() *middleware {
	return c.Middlewares()[0].(*middleware)
}

// AddNode adds a routing node and returns its ID.
func (c *Comp) AddNode(name string) int {
	m := c.mw()
	m.nodeNames = append(m.nodeNames, name)
	m.routes = nil

	return len(m.nodeNames) - 1
}

// AddLink adds a pair of directed links (one per direction) between two
// nodes. Bandwidth is per direction in bytes per second.
func (c *Comp) AddLink(a, b int, bytesPerSec float64, latency timing.VTimeInPicoSec) {
	m := c.mw()
	m.links = append(m.links,
		&link{from: a, to: b, bytesPerSec: bytesPerSec, latency: latency},
		&link{from: b, to: a, bytesPerSec: bytesPerSec, latency: latency})
	m.routes = nil
}

// PlugIn is not supported; ports must be attached to a node with PlugInAt.
func (c *Comp) PlugIn(port messaging.Port) {
	panic(fmt.Sprintf("fabric: port %s must be plugged in with PlugInAt",
		port.Name()))
}

// PlugInAt attaches a port to a node.
func (c *Comp) PlugInAt(port messaging.Port, node int) {
	c.Lock()
	defer c.Unlock()

	m := c.mw()
	if node < 0 || node >= len(m.nodeNames) {
		panic(fmt.Sprintf("fabric: node %d does not exist", node))
	}

	m.ports = append(m.ports, port)
	m.portNode = append(m.portNode, node)
	m.inflight = append(m.inflight, 0)
	m.portIndex[port.AsRemote()] = len(m.ports) - 1
	port.SetConnection(c)
}

// Unplug is not supported.
func (c *Comp) Unplug(_ messaging.Port) {
	panic("not implemented")
}

// NotifyAvailable is called by a port to notify the connection can deliver
// again.
func (c *Comp) NotifyAvailable(p messaging.Port) {
	for _, port := range c.mw().ports {
		if port == p {
			continue
		}
		port.NotifyAvailable()
	}
	c.TickNow()
}

// NotifySend is called by a port to notify the connection can start ticking.
func (c *Comp) NotifySend() {
	c.TickNow()
}

// HopCount returns the number of links on the route between two nodes.
func (c *Comp) HopCount(from, to int) int {
	return len(c.mw().route(from, to))
}

// Stats returns the traffic carried by every directed link that was used.
func (c *Comp) Stats() []LinkStats {
	m := c.mw()

	var stats []LinkStats
	for _, l := range m.links {
		if l.msgs == 0 {
			continue
		}

		stats = append(stats, LinkStats{
			From:        m.nodeNames[l.from],
			To:          m.nodeNames[l.to],
			Bytes:       l.bytes,
			Msgs:        l.msgs,
			BusyPicoSec: uint64(l.busy),
		})
	}

	return stats
}

type middleware struct {
	comp   *modeling.Component[Spec, State, modeling.None]
	engine timing.Engine

	nodeNames []string
	links     []*link
	routes    [][][]*link // [from][to] -> links on the shortest route

	ports     []messaging.Port
	portNode  []int
	inflight  []int
	portIndex map[messaging.RemotePort]int

	flights flightHeap
	stalled []*flight // due, but the destination could not accept them
	seq     uint64
}

func (m *middleware) Tick() bool {
	now := m.engine.CurrentTime()

	madeProgress := m.deliver(now)
	madeProgress = m.inject(now) || madeProgress

	// Messages still crossing the fabric arrive at a future time; keep ticking
	// until they land.
	return madeProgress || len(m.flights) > 0 || len(m.stalled) > 0
}

func (m *middleware) deliver(now timing.VTimeInPicoSec) bool {
	madeProgress := false

	for len(m.flights) > 0 && m.flights[0].arrival <= now {
		m.stalled = append(m.stalled, heap.Pop(&m.flights).(*flight))
	}

	remaining := m.stalled[:0]
	for _, f := range m.stalled {
		dst := m.ports[m.portIndexOf(f.msg.Meta().Dst)]
		if !dst.CanDeliver() {
			remaining = append(remaining, f)
			continue
		}

		dst.Deliver(f.msg)
		m.inflight[f.src]--
		madeProgress = true
	}
	m.stalled = remaining

	return madeProgress
}

func (m *middleware) inject(now timing.VTimeInPicoSec) bool {
	state := m.comp.State
	numPorts := len(m.ports)
	madeProgress := false
	maxInflight := m.comp.Spec().MaxInflightPerPort

	for i := range numPorts {
		src := (i + state.NextPortID) % numPorts
		port := m.ports[src]

		for m.inflight[src] < maxInflight {
			msg := port.PeekOutgoing()
			if msg == nil {
				break
			}

			dst := m.portIndexOf(msg.Meta().Dst)
			arrival := m.reserve(m.portNode[src], m.portNode[dst], m.msgBytes(msg), now)

			port.RetrieveOutgoing()
			m.inflight[src]++
			m.seq++
			heap.Push(&m.flights, &flight{msg: msg, src: src, arrival: arrival, seq: m.seq})
			madeProgress = true
		}
	}

	if numPorts > 0 {
		(&m.comp.State).NextPortID = (state.NextPortID + 1) % numPorts
	}

	return madeProgress
}

// reserve books every link on the route and returns the arrival time of the
// message's tail at the destination node.
func (m *middleware) reserve(
	from, to, bytes int,
	now timing.VTimeInPicoSec,
) timing.VTimeInPicoSec {
	head := now
	arrival := now

	for _, l := range m.route(from, to) {
		ser := timing.VTimeInPicoSec(math.Ceil(float64(bytes) / l.bytesPerSec * 1e12))
		start := max(head, l.nextFree)
		l.nextFree = start + ser
		l.bytes += uint64(bytes)
		l.msgs++
		l.busy += ser
		head = start + l.latency
		arrival = start + l.latency + ser
	}

	return arrival
}

func (m *middleware) msgBytes(msg messaging.Msg) int {
	header := m.comp.Spec().HeaderBytes

	switch msg := msg.(type) {
	case *memprotocol.WriteReq:
		return header + len(msg.Data)
	case memprotocol.WriteReq:
		return header + len(msg.Data)
	case *memprotocol.DataReadyRsp:
		return header + len(msg.Data)
	case memprotocol.DataReadyRsp:
		return header + len(msg.Data)
	}

	if b := msg.Meta().TrafficBytes; b > 0 {
		return b
	}

	return header
}

func (m *middleware) portIndexOf(p messaging.RemotePort) int {
	idx, found := m.portIndex[p]
	if !found {
		panic(fmt.Sprintf("fabric: port %s is not plugged in", p))
	}

	return idx
}

func (m *middleware) route(from, to int) []*link {
	if m.routes == nil {
		m.buildRoutes()
	}

	r := m.routes[from][to]
	if r == nil && from != to {
		panic(fmt.Sprintf("fabric: no route from %s to %s",
			m.nodeNames[from], m.nodeNames[to]))
	}

	return r
}

// buildRoutes computes deterministic shortest (fewest-hop) routes with a BFS
// from every node. Ties go to the link added first, so a 2D mesh whose X links
// are added before its Y links routes X-first.
func (m *middleware) buildRoutes() {
	n := len(m.nodeNames)
	out := make([][]*link, n)
	for _, l := range m.links {
		out[l.from] = append(out[l.from], l)
	}

	m.routes = make([][][]*link, n)
	for src := range n {
		m.routes[src] = make([][]*link, n)
		prev := make([]*link, n)
		visited := make([]bool, n)
		visited[src] = true
		queue := []int{src}

		for len(queue) > 0 {
			u := queue[0]
			queue = queue[1:]

			for _, l := range out[u] {
				if visited[l.to] {
					continue
				}
				visited[l.to] = true
				prev[l.to] = l
				queue = append(queue, l.to)
			}
		}

		for dst := range n {
			if dst == src || !visited[dst] {
				continue
			}

			var path []*link
			for v := dst; v != src; v = prev[v].from {
				path = append([]*link{prev[v]}, path...)
			}
			m.routes[src][dst] = path
		}
	}
}

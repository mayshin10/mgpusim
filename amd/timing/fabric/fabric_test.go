package fabric

import (
	"testing"

	"github.com/sarchlab/akita/v5/timing"
)

// newMesh returns the middleware of a 2x2 mesh (X links first) with one extra
// leaf node hanging off node 0.
func newMesh(bw float64, lat timing.VTimeInPicoSec) *middleware {
	m := &middleware{}
	m.nodeNames = []string{"n0", "n1", "n2", "n3", "leaf"}

	add := func(a, b int) {
		m.links = append(m.links,
			&link{from: a, to: b, bytesPerSec: bw, latency: lat},
			&link{from: b, to: a, bytesPerSec: bw, latency: lat})
	}
	add(0, 1)
	add(2, 3)
	add(0, 2)
	add(1, 3)
	add(4, 0)

	return m
}

func TestRoutesAreShortestAndXFirst(t *testing.T) {
	m := newMesh(1e12, 0)

	cases := []struct {
		from, to int
		path     []int
	}{
		{0, 0, nil},
		{0, 1, []int{1}},
		{0, 3, []int{1, 3}}, // X (0->1) before Y (1->3)
		{3, 0, []int{2, 0}},
		{4, 3, []int{0, 1, 3}},
	}

	for _, c := range cases {
		r := m.route(c.from, c.to)
		if len(r) != len(c.path) {
			t.Fatalf("%d->%d: got %d hops, want %d", c.from, c.to, len(r), len(c.path))
		}

		for i, l := range r {
			if l.to != c.path[i] {
				t.Fatalf("%d->%d: hop %d goes to %d, want %d", c.from, c.to, i, l.to, c.path[i])
			}
		}
	}
}

func TestReserveChargesLatencyAndSerialization(t *testing.T) {
	// 1 GB/s: 1000 bytes take 1 us = 1e6 ps per link.
	m := newMesh(1e9, 100)

	// Two hops, cut-through: the head reaches the 2nd link after one latency.
	if got, want := m.reserve(0, 3, 1000, 0), timing.VTimeInPicoSec(100+100+1e6); got != want {
		t.Fatalf("first message arrives at %d, want %d", got, want)
	}

	// A second message on the same route queues behind the first on link 0->1.
	if got, want := m.reserve(0, 3, 1000, 0), timing.VTimeInPicoSec(2e6+100+100); got != want {
		t.Fatalf("second message arrives at %d, want %d", got, want)
	}

	// A message on a disjoint route does not wait.
	if got, want := m.reserve(2, 0, 1000, 0), timing.VTimeInPicoSec(100+1e6); got != want {
		t.Fatalf("disjoint message arrives at %d, want %d", got, want)
	}
}

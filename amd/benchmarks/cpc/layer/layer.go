// Package layer runs one dense GQA transformer layer on the 8 chiplets of a
// CPX-partitioned MI300X model under the two configurations of cpc-analytic:
//
//   - TP8: tensor parallelism over all 8 chiplets (KV heads replicated).
//   - TP4xKVP2: 4-way TP x 2-way KV parallelism, with zigzag Ring Attention
//     for prefill and Helix (partial attention + all-to-all + LSE combine)
//     for decode. O-proj and the FFN stay 8-way TP in both.
//
// Each chiplet is one GPU (GPU IDs 1..8 = XCD 0..7). Collectives are kernels
// that read peer memory directly, one kernel per ring step, with a barrier
// between steps. Every stage is timed in simulated time.
package layer

import (
	_ "embed"
	"fmt"
	"log"
	"math"

	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/driver"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

// D is the head dimension (fixed by the kernels).
const D = 128

const (
	wgSize    = 256
	numChips  = 8
	log2e     = 1.4426950408889634
	copyMaxWG = 152 // work-groups of a copy kernel (4 per CU of an XCD)
)

// ring8 visits the 8 XCDs with 1-hop steps over the 2x2 IOD mesh (XCD x sits
// on IOD x/2; the IOD ring is 0-1-3-2).
var ring8 = []int{0, 1, 3, 2, 6, 7, 5, 4}

//go:embed kernels_gfx942.hsaco
var hsacoBytes []byte

var kernelNames = []string{
	"gemm_nt", "gemm_skinny", "reduce_splits", "swiglu", "copy16", "copy2d",
	"add_bf16", "attn_prefill", "attn_decode", "combine_splits", "combine_lse", "reduce_partials",
}

// Model holds the layer dimensions. H must equal NQ*D.
type Model struct {
	H, NQ, NKV, F int
}

// Config selects the parallel configuration and the workload.
type Config struct {
	Model

	TP, KVP int    // (8, 1) or (4, 2)
	Phase   string // "prefill" or "decode"

	// Prefill: S is the number of prompt tokens. Decode: S is the cached
	// context per request and Batch the number of requests.
	S, Batch int

	// DecodeSplits is the number of key splits of decode attention on each
	// chiplet (a multiple of 4). Zero picks enough splits to give the 38 CUs
	// of an XCD one 4-wave work-group each, with at least 16 keys per split.
	DecodeSplits int

	// Part selects the stages to run: "all", "head" (the stages that differ
	// between TP8 and TP4xKVP2, up to the O-proj input), or "tail" (O-proj
	// onward, identical in both configurations). Only "all" can be verified.
	Part string
}

func (c Config) runsHead() bool { return c.Part != "tail" }
func (c Config) runsTail() bool { return c.Part != "head" }

// Name returns the configuration name used by cpc-analytic.
func (c Config) Name() string {
	if c.KVP == 1 {
		return "TP8"
	}

	return fmt.Sprintf("TP%dxKVP%d", c.TP, c.KVP)
}

// Stage is one timed stage of the layer.
type Stage struct {
	Name    string
	Kind    string // "compute" or "net"
	Seconds float64
}

// Benchmark runs one layer.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	kernels map[string]*insts.KernelCodeObject

	Cfg Config
	// Now returns the simulated time.
	Now func() timing.VTimeInPicoSec

	Stages []Stage

	verify bool
	chips  []*chiplet
	host   *hostModel
}

type chiplet struct {
	index  int // 0..7, the XCD
	gpu    int // driver GPU ID
	t, r   int // TP group and KVP rank
	queues [2]*driver.CommandQueue
	buf    map[string]driver.Ptr
}

// NewBenchmark creates a benchmark on the given driver.
func NewBenchmark(d *driver.Driver) *Benchmark {
	b := &Benchmark{driver: d, context: d.Init(), kernels: map[string]*insts.KernelCodeObject{}}

	for _, name := range kernelNames {
		co := insts.LoadKernelCodeObjectFromBytes(hsacoBytes, name)
		if co == nil {
			log.Panicf("failed to load kernel %s", name)
		}

		b.kernels[name] = co
	}

	return b
}

// SelectGPU is ignored; the benchmark always uses GPUs 1..8.
func (b *Benchmark) SelectGPU(_ []int) {}

// SetUnifiedMemory is not supported.
func (b *Benchmark) SetUnifiedMemory() { panic("layer: unified memory is not supported") }

// EnableVerification makes Run keep the host data needed by Verify.
func (b *Benchmark) EnableVerification() { b.verify = true }

// Run sets up the weights and inputs, then runs and times the layer.
func (b *Benchmark) Run() {
	b.mustBeValid()
	b.host = newHostModel(b.Cfg)
	b.createChiplets()

	switch b.Cfg.Phase {
	case "prefill":
		b.setupPrefill()
		b.runPrefill()
	case "decode":
		b.setupDecode()
		b.runDecode()
	default:
		log.Panicf("unknown phase %q", b.Cfg.Phase)
	}

	total := 0.0
	for _, s := range b.Stages {
		total += s.Seconds
		fmt.Printf("stage config=%s phase=%s S=%d batch=%d name=%s kind=%s seconds=%.6e\n",
			b.Cfg.Name(), b.Cfg.Phase, b.Cfg.S, b.Cfg.Batch, s.Name, s.Kind, s.Seconds)
	}

	fmt.Printf("layer config=%s phase=%s S=%d batch=%d part=%s seconds=%.6e\n",
		b.Cfg.Name(), b.Cfg.Phase, b.Cfg.S, b.Cfg.Batch, b.Cfg.Part, total)
}

func (b *Benchmark) mustBeValid() {
	c := b.Cfg
	if c.H != c.NQ*D {
		log.Panicf("H (%d) must equal NQ*D (%d)", c.H, c.NQ*D)
	}

	if c.TP*c.KVP != numChips || (c.KVP != 1 && c.KVP != 2) {
		log.Panicf("unsupported configuration TP%d x KVP%d", c.TP, c.KVP)
	}

	if c.NQ%numChips != 0 || (c.NQ/c.TP)%c.KVP != 0 {
		log.Panicf("NQ (%d) must split into 8 O-proj shards", c.NQ)
	}

	if c.NKV > c.TP && c.NKV%c.TP != 0 || c.NKV < c.TP && c.TP%c.NKV != 0 {
		log.Panicf("NKV (%d) and TP (%d) must divide", c.NKV, c.TP)
	}

	if c.F%(numChips*128) != 0 {
		log.Panicf("F must be a multiple of %d", numChips*128)
	}

	if c.Phase == "prefill" && c.S%256 != 0 {
		log.Panicf("prefill S must be a multiple of 256")
	}

	if c.Part != "" && c.Part != "all" && c.Part != "head" && c.Part != "tail" {
		log.Panicf("unknown part %q", c.Part)
	}

	if c.Phase == "decode" && (c.Batch < 1 || c.Batch > 16 || c.DecodeSplits%4 != 0 || c.DecodeSplits < 0) {
		log.Panicf("decode needs 1 <= batch <= 16 and DecodeSplits %% 4 == 0")
	}
}

// autoDecodeSplits returns DecodeSplits, or when it is zero, the number of
// key splits that fills one XCD (38 CUs x 4 waves) without splits of fewer
// than 16 keys.
func (c Config) autoDecodeSplits() int {
	if c.DecodeSplits > 0 {
		return c.DecodeSplits
	}

	ctx := c.S / c.KVP
	splits := min(152, max(4, ctx/16)/c.Batch)

	return max(4, splits/4*4)
}

func (b *Benchmark) createChiplets() {
	for i := 0; i < numChips; i++ {
		c := &chiplet{index: i, gpu: i + 1, t: i, buf: map[string]driver.Ptr{}}
		if b.Cfg.KVP == 2 {
			c.t, c.r = i/2, i%2 // KVP partners share an IOD
		}

		b.driver.SelectGPU(b.context, c.gpu)
		c.queues[0] = b.driver.CreateCommandQueue(b.context)
		c.queues[1] = b.driver.CreateCommandQueue(b.context)
		b.chips = append(b.chips, c)

		// Load the code now so that the copies are not timed with the
		// first stage that uses each kernel.
		for _, co := range b.kernels {
			b.driver.LoadCodeObject(b.context, c.gpu, co)
		}
	}
}

// nq is the number of q heads a chiplet runs attention for, hkv its kv heads,
// and hO the heads of its O-proj shard.
func (b *Benchmark) nq() int  { return b.Cfg.NQ / b.Cfg.TP }
func (b *Benchmark) hkv() int { return max(1, b.Cfg.NKV/b.Cfg.TP) }
func (b *Benchmark) hO() int  { return b.Cfg.NQ / numChips }
func (b *Benchmark) f8() int  { return b.Cfg.F / numChips }

// firstKVHead is the first kv head of TP group t (TP > NKV replicates heads).
func (b *Benchmark) firstKVHead(t int) int {
	if b.Cfg.TP >= b.Cfg.NKV {
		return t * b.Cfg.NKV / b.Cfg.TP
	}

	return t * b.hkv()
}

// tpRing returns the members of chiplet c's TP group in 1-hop ring order.
func (b *Benchmark) tpRing(c *chiplet) []*chiplet {
	if b.Cfg.KVP == 1 {
		return b.ring8()
	}

	ring := make([]*chiplet, 0, 4)
	for _, t := range []int{0, 1, 3, 2} {
		ring = append(ring, b.chips[2*t+c.r])
	}

	return ring
}

func (b *Benchmark) ring8() []*chiplet {
	ring := make([]*chiplet, numChips)
	for p, i := range ring8 {
		ring[p] = b.chips[i]
	}

	return ring
}

// alloc allocates a named buffer on a chiplet, filled with data (or zeros).
func (b *Benchmark) alloc(c *chiplet, name string, bytes uint64, data any) driver.Ptr {
	b.driver.SelectGPU(b.context, c.gpu)
	p := b.driver.AllocateMemory(b.context, bytes)

	if data == nil {
		data = make([]byte, bytes)
	}

	b.driver.MemCopyH2D(b.context, p, data)
	c.buf[name] = p

	return p
}

// launch is one kernel launch of a stage.
type launch struct {
	chip   *chiplet
	queue  int
	kernel string
	grid   [3]int // work-groups
	args   any
}

// run launches every kernel of one barrier phase and waits for all of them.
// Launches on the same chiplet queue run in order; different queues overlap.
func (b *Benchmark) run(ls []launch) float64 {
	start := b.Now()

	used := map[*driver.CommandQueue]bool{}
	for _, l := range ls {
		q := l.chip.queues[l.queue]
		used[q] = true
		// The driver allocates the kernel arguments and the dispatch packet
		// on the context's current GPU, so select the launching GPU to keep
		// them local.
		b.driver.SelectGPU(b.context, l.chip.gpu)
		b.driver.EnqueueLaunchKernel(q, b.kernels[l.kernel],
			[3]uint32{uint32(l.grid[0] * wgSize), uint32(max(1, l.grid[1])), uint32(max(1, l.grid[2]))},
			[3]uint16{wgSize, 1, 1}, l.args)
	}

	for q := range used {
		b.driver.DrainCommandQueue(q)
	}

	return float64(b.Now()-start) * 1e-12
}

// stage runs a sequence of barrier phases and records their total time.
func (b *Benchmark) stage(name, kind string, phases ...[]launch) {
	total := 0.0
	for _, p := range phases {
		if len(p) > 0 {
			total += b.run(p)
		}
	}

	b.Stages = append(b.Stages, Stage{Name: name, Kind: kind, Seconds: total})
	log.Printf("stage %s done: %.3f us simulated", name, total*1e6)
}

func copyGrid(words16 int) (grid, stride int) {
	grid = min(copyMaxWG, max(1, (words16+wgSize-1)/wgSize))
	return grid, grid * wgSize
}

func (b *Benchmark) copyLaunch(c *chiplet, queue int, src, dst driver.Ptr, bytes uint64) launch {
	n16 := int(bytes / 16)
	grid, stride := copyGrid(n16)

	return launch{chip: c, queue: queue, kernel: "copy16", grid: [3]int{grid},
		args: &copyArgs{Src: src, Dst: dst, N16: int32(n16), Stride: int32(stride)}}
}

// copy2dLaunch copies rows x rowBytes from src (row stride srcStride bytes) to
// dst (row stride dstStride bytes). All sizes are multiples of 4 bytes.
func (b *Benchmark) copy2dLaunch(c *chiplet, queue int, src, dst driver.Ptr,
	rows, rowBytes, srcStride, dstStride int,
) launch {
	words := rows * rowBytes / 4
	grid, stride := copyGrid((words + 3) / 4)

	return launch{chip: c, queue: queue, kernel: "copy2d", grid: [3]int{grid},
		args: &copy2dArgs{Src: src, Dst: dst, Rows: int32(rows), Words: int32(rowBytes / 4),
			SrcStride: int32(srcStride / 4), DstStride: int32(dstStride / 4), Stride: int32(stride)}}
}

// ringAllGather gathers chunks in place: ring position p initially holds the
// chunk chunkOf(p) of its buffer; after n-1 steps every member holds all.
func (b *Benchmark) ringAllGather(ring []*chiplet, buf string, chunkOf func(p int) int, chunkBytes uint64) [][]launch {
	n := len(ring)
	phases := make([][]launch, 0, n-1)

	for s := 1; s < n; s++ {
		var ls []launch
		for p, c := range ring {
			left := ring[(p-1+n)%n]
			off := uint64(chunkOf((p-s+n)%n)) * chunkBytes
			ls = append(ls, b.copyLaunch(c, 0, left.buf[buf]+driver.Ptr(off), c.buf[buf]+driver.Ptr(off), chunkBytes))
		}
		phases = append(phases, ls)
	}

	return phases
}

// ringReduceScatter sums the members' buffers so that ring position p ends up
// holding the reduced chunk chunkOf(p) (in place). At step s, position p adds
// its left neighbour's partial of chunk chunkOf(p-s-2).
func (b *Benchmark) ringReduceScatter(ring []*chiplet, buf string, chunkOf func(p int) int, chunkBytes uint64) [][]launch {
	n := len(ring)
	phases := make([][]launch, 0, n-1)
	n8 := int(chunkBytes / 16)
	grid, stride := copyGrid(n8)

	for s := 0; s < n-1; s++ {
		var ls []launch
		for p, c := range ring {
			left := ring[(p-1+n)%n]
			off := driver.Ptr(uint64(chunkOf(((p-s-2)%n+2*n)%n)) * chunkBytes)
			ls = append(ls, launch{chip: c, kernel: "add_bf16", grid: [3]int{grid},
				args: &addArgs{Src: left.buf[buf] + off, Dst: c.buf[buf] + off, N8: int32(n8), Stride: int32(stride)}})
		}
		phases = append(phases, ls)
	}

	return phases
}

// gemmPartialBytes is the per-chiplet buffer for split-K partial sums.
const gemmPartialBytes = 16 << 20

// minGemmWGs is the work-group count below which a GEMM is split over K
// (twice the CUs of one XCD).
const minGemmWGs = 76

// gemm returns the launches of out[m][n] (row stride ldc) = a[m][k] *
// w[n][k]^T. GEMMs with fewer than minGemmWGs tiles are split over K (in
// slices that stay multiples of 64) and their fp32 partials reduced.
func (b *Benchmark) gemm(c *chiplet, a, w, out driver.Ptr, m, n, k, ldc int) []launch {
	tiles, splits := (n/128)*(m/128), 1
	for tiles*splits < minGemmWGs && (k/(2*splits))%64 == 0 && k/(2*splits) >= 64 {
		splits *= 2
	}

	args := &gemmArgs{A: a, B: w, C: out, K: int32(k / splits), Lda: int32(k), Ldb: int32(k),
		Ldc: int32(ldc), M: int32(m)}
	ls := []launch{{chip: c, kernel: "gemm_nt", grid: [3]int{n / 128, m / 128, splits}, args: args}}

	if splits > 1 {
		if splits*m*ldc*4 > gemmPartialBytes {
			log.Panicf("split-K partials need %d bytes", splits*m*ldc*4)
		}

		args.P = c.buf["gemmP"]
		ls = append(ls, launch{chip: c, kernel: "reduce_partials", grid: [3]int{(m*n/4 + wgSize - 1) / wgSize},
			args: &reducePartialsArgs{P: c.buf["gemmP"], C: out, M: int32(m), N: int32(n),
				Splits: int32(splits), Ldc: int32(ldc)}})
	}

	return ls
}

// skinnyGemm returns the launches of a decode GEMM: fp32 partial sums over
// K splits, then a reduction into out[m*ldc + n*ldn].
func (b *Benchmark) skinnyGemm(c *chiplet, a, w, out driver.Ptr, m, n, k, ldc, ldn int) []launch {
	splits := max(1, k/128)
	part := c.buf["partials"]

	return []launch{
		{chip: c, kernel: "gemm_skinny", grid: [3]int{n / 64, splits},
			args: &skinnyArgs{A: a, B: w, P: part, M: int32(m), N: int32(n), KPerSplit: int32(k / splits),
				Lda: int32(k), Ldb: int32(k)}},
		{chip: c, kernel: "reduce_splits", grid: [3]int{(m*n + wgSize - 1) / wgSize},
			args: &reduceArgs{P: part, C: out, M: int32(m), N: int32(n), Splits: int32(splits),
				Ldc: int32(ldc), Ldn: int32(ldn)}},
	}
}

func attnScale() float32 { return float32(log2e / math.Sqrt(D)) }

// readBF16 copies a device bf16 buffer to the host as float32.
func (b *Benchmark) readBF16(p driver.Ptr, n int) []float32 {
	raw := make([]uint16, n)
	b.driver.MemCopyD2H(b.context, raw, p)

	out := make([]float32, n)
	for i, h := range raw {
		out[i] = bf2f(h)
	}

	return out
}

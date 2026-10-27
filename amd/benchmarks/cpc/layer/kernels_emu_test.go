package layer

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/sarchlab/akita/v5/hooking"
	"github.com/sarchlab/akita/v5/simulation"
	"github.com/sarchlab/mgpusim/v5/amd/arch"
	"github.com/sarchlab/mgpusim/v5/amd/driver"
	"github.com/sarchlab/mgpusim/v5/amd/emu"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/emusystem"
)

// traceTarget, when set, is attached to every emulated compute unit.
var traceTarget hooking.Hook

// Kernel-level checks on MGPUSim's CDNA3 emulator against the CPU reference.
// The same kernels are checked on real hardware by kernels/test_mi300a.

type emuEnv struct {
	d   *driver.Driver
	ctx *driver.Context
	b   *Benchmark
	q   *driver.CommandQueue
}

func newEmuEnv(t *testing.T) *emuEnv {
	t.Helper()

	s := simulation.MakeBuilder().WithoutMonitoring().WithOutputFileName(t.TempDir() + "/sim").Build()
	t.Cleanup(s.Terminate)

	d := emusystem.MakeBuilder().WithSimulation(s).WithNumGPUs(1).WithArchitecture(arch.CDNA3).Build()
	if traceTarget != nil {
		for _, c := range s.Components() {
			if cu, ok := c.(*emu.Comp); ok {
				cu.AcceptHook(traceTarget)
			}
		}
	}
	d.Run()
	t.Cleanup(d.Terminate)

	if path := os.Getenv("CPC_HSACO"); path != "" { // debug variants of the kernels
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		hsacoBytes = data
	}

	b := NewBenchmark(d)
	b.driver.SelectGPU(b.context, 1)

	return &emuEnv{d: d, ctx: b.context, b: b, q: d.CreateCommandQueue(b.context)}
}

func (e *emuEnv) upload(v any, bytes int) driver.Ptr {
	p := e.d.AllocateMemory(e.ctx, uint64(bytes))
	if v == nil {
		v = make([]byte, bytes)
	}
	e.d.MemCopyH2D(e.ctx, p, v)

	return p
}

func (e *emuEnv) launch(kernel string, grid [3]int, args any) {
	e.d.EnqueueLaunchKernel(e.q, e.b.kernels[kernel],
		[3]uint32{uint32(grid[0] * wgSize), uint32(max(1, grid[1])), uint32(max(1, grid[2]))},
		[3]uint16{wgSize, 1, 1}, args)
	e.d.DrainCommandQueue(e.q)
}

func checkClose(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()

	if err := compare(got, want); err > tol || math.IsNaN(err) {
		t.Errorf("%s: relative error %.3e > %.1e", name, err, tol)
	}
}

func TestEmuGemm(t *testing.T) {
	e := newEmuEnv(t)
	rng := rand.New(rand.NewSource(3))
	m, n, k := 128, 256, 128
	a, w := randBF(rng, m*k, 1), randBF(rng, n*k, 1)
	want := matmulNT(toF32(a), w, m, n, k)

	pa, pw, pc := e.upload(a, m*k*2), e.upload(w, n*k*2), e.upload(nil, m*n*2)
	e.launch("gemm_nt", [3]int{n / 128, m / 128},
		&gemmArgs{A: pa, B: pw, C: pc, K: int32(k), Lda: int32(k), Ldb: int32(k), Ldc: int32(n), M: int32(m)})
	checkClose(t, "gemm_nt", e.b.readBF16(pc, m*n), want, 2e-2)

	pp, pc2 := e.upload(nil, 2*m*n*4), e.upload(nil, m*n*2)
	e.launch("gemm_nt", [3]int{n / 128, m / 128, 2},
		&gemmArgs{A: pa, B: pw, C: pc2, K: int32(k / 2), Lda: int32(k), Ldb: int32(k), Ldc: int32(n),
			P: pp, M: int32(m)})
	e.launch("reduce_partials", [3]int{(m*n/4 + wgSize - 1) / wgSize},
		&reducePartialsArgs{P: pp, C: pc2, M: int32(m), N: int32(n), Splits: 2, Ldc: int32(n)})
	checkClose(t, "gemm_nt split-K", e.b.readBF16(pc2, m*n), want, 2e-2)
}

func TestEmuPrefillAttention(t *testing.T) {
	t.Run("causal", func(t *testing.T) { testEmuPrefillAttention(t, true) })
	t.Run("full", func(t *testing.T) { testEmuPrefillAttention(t, false) })
}

func testEmuPrefillAttention(t *testing.T, causal bool) {
	e := newEmuEnv(t)
	rng := rand.New(rand.NewSource(4))
	s, nq := 64, 2
	q, k, v := randBF(rng, s*nq*D, 1), randBF(rng, s*D, 1), randBF(rng, s*D, 1)

	vt := make([]uint16, D*s)
	for j := 0; j < s; j++ {
		for x := 0; x < D; x++ {
			vt[x*s+j] = v[j*D+x]
		}
	}

	want := make([]float32, s*nq*D)
	identity := func(j int) int { return j }
	for i := 0; i < s*nq; i++ {
		res := attend(toF32(q[i*D:(i+1)*D]), toF32(k), toF32(v), s, D, nil, 0)
		if causal {
			res = attend(toF32(q[i*D:(i+1)*D]), toF32(k), toF32(v), i/nq+1, D, identity, i/nq)
		}
		copy(want[i*D:], res)
	}

	c := int32(0)
	if causal {
		c = 1
	}

	pq, pk, pvt := e.upload(q, s*nq*D*2), e.upload(k, s*D*2), e.upload(vt, s*D*2)
	po, pm, pl := e.upload(nil, s*nq*D*4), e.upload(nil, s*nq*4), e.upload(nil, s*nq*4)
	out := e.upload(nil, s*nq*D*2)

	e.launch("attn_prefill", [3]int{(s*nq + 63) / 64, 1}, &attnArgs{Q: pq, K: pk, Vt: pvt, O: po, M: pm, L: pl,
		NPos: int32(s), NQ: int32(nq), NKV: 1, NKeys: int32(s), VtStride: int32(s), Causal: c, Ranks: 1,
		Chunk: int32(s / 2), Scale: attnScale(), Splits: 1, RecSplitStride: int32(s * nq)})
	e.launch("combine_splits", [3]int{(s*nq*D + wgSize - 1) / wgSize}, &combineSplitsArgs{O: po, M: pm, L: pl,
		Out: out, NRows: int32(s * nq), Splits: 1, OutRowStride: D})

	checkClose(t, "attn_prefill", e.b.readBF16(out, s*nq*D), want, 2e-2)
}

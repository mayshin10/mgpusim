package layer

import (
	"fmt"
	"log"

	"github.com/sarchlab/mgpusim/v5/amd/driver"
)

// Token layout. Prefill activations are kept in "layout order": the
// concatenation of the KVP shards, where shard r holds zigzag chunks r and
// 2*KVP-1-r. The residual is split into 8 slices of S/8 tokens in layout order;
// chiplet c holds slice sigma(c), chosen so that a TP group's slices, gathered
// in TP order, form exactly its KVP shard.

func zigzag(u, rank, ranks, chunk int) int {
	if u < chunk {
		return rank*chunk + u
	}

	return (2*ranks-1-rank)*chunk + u - chunk
}

// layoutPos maps a layout-order token index to its sequence position.
func (b *Benchmark) layoutPos(i int) int {
	sLoc := b.Cfg.S / b.Cfg.KVP
	return zigzag(i%sLoc, i/sLoc, b.Cfg.KVP, b.Cfg.S/(2*b.Cfg.KVP))
}

// sigma is the residual slice held by a chiplet.
func (b *Benchmark) sigma(c *chiplet) int {
	if b.Cfg.KVP == 1 {
		return c.index
	}

	return c.r*b.Cfg.TP + c.t
}

func (b *Benchmark) setupPrefill() {
	cfg, m := b.Cfg, b.host
	h, sLoc, slice := cfg.H, cfg.S/cfg.KVP, cfg.S/numChips
	nq, hkv, hO, f8 := b.nq(), b.hkv(), b.hO(), b.f8()

	for _, c := range b.chips {
		// The gather buffer holds this chiplet's residual slice at its TP
		// position; the rest is filled by the all-gather.
		xg := make([]uint16, sLoc*h)
		own := b.sigma(c)
		gatherPos := c.index
		if cfg.KVP == 2 {
			gatherPos = c.t
		}

		for i := 0; i < slice; i++ {
			pos := b.layoutPos(own*slice + i)
			copy(xg[(gatherPos*slice+i)*h:], m.x[pos*h:(pos+1)*h])
		}

		kv0 := b.firstKVHead(c.t)
		b.alloc(c, "Xg", uint64(sLoc*h*2), xg)
		b.alloc(c, "Wq", uint64(nq*D*h*2), rows(m.wq, h, c.t*nq*D, (c.t+1)*nq*D))
		b.alloc(c, "Wk", uint64(hkv*D*h*2), rows(m.wk, h, kv0*D, (kv0+hkv)*D))
		b.alloc(c, "Wv", uint64(hkv*D*h*2), rows(m.wv, h, kv0*D, (kv0+hkv)*D))
		b.alloc(c, "Wo", uint64(h*hO*D*2), cols(m.wo, h, cfg.NQ*D, c.index*hO*D, (c.index+1)*hO*D))
		b.alloc(c, "Wgu", uint64(2*f8*h*2), concat(
			rows(m.wg, h, c.index*f8, (c.index+1)*f8), rows(m.wu, h, c.index*f8, (c.index+1)*f8)))
		b.alloc(c, "Wd", uint64(h*f8*2), cols(m.wd, h, cfg.F, c.index*f8, (c.index+1)*f8))

		b.alloc(c, "Q", uint64(sLoc*nq*D*2), nil)
		b.alloc(c, "K", uint64(sLoc*hkv*D*2), nil)
		b.alloc(c, "Vt", uint64(hkv*D*sLoc*2), nil)
		b.alloc(c, "Orec", uint64(sLoc*nq*D*4), nil)
		b.alloc(c, "Mrec", uint64(sLoc*nq*4), nil)
		b.alloc(c, "Lrec", uint64(sLoc*nq*4), nil)
		b.alloc(c, "Oloc", uint64(sLoc*nq*D*2), nil)

		if cfg.KVP == 2 {
			b.alloc(c, "recvK", uint64(sLoc*hkv*D*2), nil)
			b.alloc(c, "recvVt", uint64(hkv*D*sLoc*2), nil)
			b.alloc(c, "Oin", uint64(cfg.S*hO*D*2), nil)
		} else {
			c.buf["Oin"] = c.buf["Oloc"]
		}

		b.alloc(c, "Y", uint64(cfg.S*h*2), nil)
		b.alloc(c, "G", uint64(cfg.S*2*f8*2), nil)
		b.alloc(c, "A", uint64(cfg.S*f8*2), nil)
		b.alloc(c, "Y2", uint64(cfg.S*h*2), nil)
		b.alloc(c, "gemmP", gemmPartialBytes, nil)
	}
}

func (b *Benchmark) runPrefill() {
	cfg := b.Cfg
	h, slice := cfg.H, cfg.S/numChips
	hO, f8 := b.hO(), b.f8()

	if cfg.runsHead() {
		b.runPrefillHead()
	}

	if !cfg.runsTail() {
		return
	}

	// 5. 8-way O-proj, reduce-scatter to the residual slices, all-gather.
	var oproj []launch
	for _, c := range b.chips {
		oproj = append(oproj, b.gemm(c, c.buf["Oin"], c.buf["Wo"], c.buf["Y"], cfg.S, h, hO*D, h)...)
	}
	b.stage("oproj", "compute", oproj)

	ring := b.ring8()
	chunk := func(p int) int { return b.sigma(ring[p]) }
	b.stage("reducescatter_oproj", "net", b.ringReduceScatter(ring, "Y", chunk, uint64(slice*h*2))...)
	b.stage("allgather_ffn", "net", b.ringAllGather(ring, "Y", chunk, uint64(slice*h*2))...)

	// 6. 8-way FFN and reduce-scatter.
	var ffn []launch
	for _, c := range b.chips {
		ffn = append(ffn, b.gemm(c, c.buf["Y"], c.buf["Wgu"], c.buf["G"], cfg.S, 2*f8, h, 2*f8)...)
		ffn = append(ffn, launch{chip: c, kernel: "swiglu", grid: [3]int{(cfg.S*f8 + wgSize - 1) / wgSize},
			args: &swigluArgs{In: c.buf["G"], Out: c.buf["A"], T: int32(cfg.S), F: int32(f8)}})
		ffn = append(ffn, b.gemm(c, c.buf["A"], c.buf["Wd"], c.buf["Y2"], cfg.S, h, f8, h)...)
	}
	b.stage("ffn", "compute", ffn)
	b.stage("reducescatter_ffn", "net", b.ringReduceScatter(ring, "Y2", chunk, uint64(slice*h*2))...)
}

// runPrefillHead runs the stages that differ between the configurations:
// gathering the KVP shard, QKV, attention, and (KVP2) the all-to-all.
func (b *Benchmark) runPrefillHead() {
	cfg := b.Cfg
	h, sLoc, slice := cfg.H, cfg.S/cfg.KVP, cfg.S/numChips
	nq, hkv, hO := b.nq(), b.hkv(), b.hO()

	// 1. All-gather the TP group's residual slices into the KVP shard.
	var gather [][]launch
	for _, ring := range b.tpRings() {
		pos := func(p int) int { return ring[p].index }
		if cfg.KVP == 2 {
			pos = func(p int) int { return ring[p].t }
		}
		gather = mergePhases(gather, b.ringAllGather(ring, "Xg", pos, uint64(slice*h*2)))
	}
	b.stage("allgather_tp", "net", gather...)

	// 2. QKV projections; V^T = Wv * X^T fills the transposed V cache.
	var qkv []launch
	for _, c := range b.chips {
		qkv = append(qkv, b.gemm(c, c.buf["Xg"], c.buf["Wq"], c.buf["Q"], sLoc, nq*D, h, nq*D)...)
		qkv = append(qkv, b.gemm(c, c.buf["Xg"], c.buf["Wk"], c.buf["K"], sLoc, hkv*D, h, hkv*D)...)
		qkv = append(qkv, b.gemm(c, c.buf["Wv"], c.buf["Xg"], c.buf["Vt"], hkv*D, sLoc, h, sLoc)...)
	}
	b.stage("qkv", "compute", qkv)

	// 3. Attention: one causal pass for TP8; a 2-step zigzag ring for KVP2,
	// with the partner's KV block fetched while step 0 runs.
	attn := func(c *chiplet, k, vt driver.Ptr, kRank int, merge bool) launch {
		mg := int32(0)
		if merge {
			mg = 1
		}

		rowsN := sLoc * (nq / hkv)

		return launch{chip: c, kernel: "attn_prefill", grid: [3]int{(rowsN + 63) / 64, hkv},
			args: &attnArgs{Q: c.buf["Q"], K: k, Vt: vt, O: c.buf["Orec"], M: c.buf["Mrec"], L: c.buf["Lrec"],
				NPos: int32(sLoc), NQ: int32(nq), NKV: int32(hkv), NKeys: int32(sLoc), VtStride: int32(sLoc),
				Causal: 1, QRank: int32(c.r), KRank: int32(kRank), Ranks: int32(cfg.KVP),
				Chunk: int32(cfg.S / (2 * cfg.KVP)), Merge: mg, Scale: attnScale(), Splits: 1,
				RecSplitStride: int32(sLoc * nq)}}
	}

	var step0, step1, finalize []launch
	for _, c := range b.chips {
		step0 = append(step0, attn(c, c.buf["K"], c.buf["Vt"], c.r, false))

		if cfg.KVP == 2 {
			partner := b.chips[c.index^1]
			step0 = append(step0,
				b.copyLaunch(c, 1, partner.buf["K"], c.buf["recvK"], uint64(sLoc*hkv*D*2)),
				b.copyLaunch(c, 1, partner.buf["Vt"], c.buf["recvVt"], uint64(hkv*D*sLoc*2)))
			step1 = append(step1, attn(c, c.buf["recvK"], c.buf["recvVt"], partner.r, true))
		}

		rowsN := sLoc * nq
		finalize = append(finalize, launch{chip: c, kernel: "combine_splits",
			grid: [3]int{(rowsN*D + wgSize - 1) / wgSize},
			args: &combineSplitsArgs{O: c.buf["Orec"], M: c.buf["Mrec"], L: c.buf["Lrec"], Out: c.buf["Oloc"],
				NRows: int32(rowsN), Splits: 1, OutRowStride: D}})
	}

	if cfg.KVP == 1 {
		b.stage("attention", "compute", step0, finalize)
	} else {
		b.stage("ring_step0", "compute", step0)
		b.stage("ring_step1", "compute", step1, finalize)

		// 4. All-to-all over KVP: keep head half r of both shards.
		var a2a []launch
		for _, c := range b.chips {
			for src := 0; src < 2; src++ {
				s := b.chips[2*c.t+src]
				queue := 0
				if s != c {
					queue = 1
				}
				a2a = append(a2a, b.copy2dLaunch(c, queue,
					s.buf["Oloc"]+driver.Ptr(c.r*hO*D*2), c.buf["Oin"]+driver.Ptr(src*sLoc*hO*D*2),
					sLoc, hO*D*2, nq*D*2, hO*D*2))
			}
		}
		b.stage("alltoall_kvp", "net", a2a)
	}

}

// tpRings returns the TP groups (one for TP8, two for TP4xKVP2) as rings.
func (b *Benchmark) tpRings() [][]*chiplet {
	if b.Cfg.KVP == 1 {
		return [][]*chiplet{b.ring8()}
	}

	return [][]*chiplet{b.tpRing(b.chips[0]), b.tpRing(b.chips[1])}
}

// mergePhases runs the phases of independent groups side by side.
func mergePhases(a, c [][]launch) [][]launch {
	for i := range c {
		if i < len(a) {
			a[i] = append(a[i], c[i]...)
		} else {
			a = append(a, c[i])
		}
	}

	return a
}

func (b *Benchmark) verifyPrefill() {
	cfg := b.Cfg
	h, slice := cfg.H, cfg.S/numChips
	got := make([]float32, cfg.S*h)

	for _, c := range b.chips {
		s := b.sigma(c)
		part := b.readBF16(c.buf["Y2"]+driver.Ptr(s*slice*h*2), slice*h)

		for i := 0; i < slice; i++ {
			pos := b.layoutPos(s*slice + i)
			copy(got[pos*h:(pos+1)*h], part[i*h:(i+1)*h])
		}
	}

	b.checkResult(got, b.host.referencePrefill())
}

func (b *Benchmark) checkResult(got, want []float32) {
	err := compare(got, want)
	fmt.Printf("verify config=%s phase=%s rel_err=%.3e\n", b.Cfg.Name(), b.Cfg.Phase, err)

	if err > 0.05 {
		log.Fatalf("verification failed: relative error %.3e", err)
	}

	log.Printf("Passed!\n")
}

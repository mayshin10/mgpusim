package layer

import (
	"github.com/sarchlab/mgpusim/v5/amd/driver"
)

// Decode: every request has S cached tokens and generates one token. With
// KVP2, rank r caches tokens [r*S/2, (r+1)*S/2) and the new token's KV is
// appended on rank S%2 (round-robin append). TP8 caches all S tokens of its
// (replicated) kv heads.

func roundUp16(n int) int { return (n + 15) / 16 * 16 }

// ctxShard returns the first cached token and the number of cached tokens of a
// chiplet, and whether it appends the new token.
func (b *Benchmark) ctxShard(c *chiplet) (first, n int, owner bool) {
	if b.Cfg.KVP == 1 {
		return 0, b.Cfg.S, true
	}

	n = b.Cfg.S / 2

	return c.r * n, n, c.r == b.Cfg.S%2
}

// kStride is the per-request key capacity of a chiplet's cache, padded so the
// attention kernel's 16-key blocks never read past it.
func (b *Benchmark) kStride() int {
	_, n, _ := b.ctxShard(b.chips[0])
	return roundUp16(n+1) + 16
}

func (b *Benchmark) setupDecode() {
	cfg, m := b.Cfg, b.host
	h, bt, hk := cfg.H, cfg.Batch, cfg.NKV*D
	nq, hkv, hO, f8 := b.nq(), b.hkv(), b.hO(), b.f8()
	ks := b.kStride()
	splits := cfg.autoDecodeSplits()
	maxN := max(nq*D, h, 2*f8)

	for _, c := range b.chips {
		kv0 := b.firstKVHead(c.t)
		first, n, _ := b.ctxShard(c)

		// K cache [B][ks][hkv][D] and transposed V cache [B][hkv][D][ks].
		kc := make([]uint16, bt*ks*hkv*D)
		vt := make([]uint16, bt*hkv*D*ks)

		for r := 0; r < bt; r++ {
			for j := 0; j < n; j++ {
				src := (r*cfg.S + first + j) * hk
				for g := 0; g < hkv; g++ {
					for x := 0; x < D; x++ {
						kc[((r*ks+j)*hkv+g)*D+x] = m.kc[src+(kv0+g)*D+x]
						vt[((r*hkv+g)*D+x)*ks+j] = m.vc[src+(kv0+g)*D+x]
					}
				}
			}
		}

		b.alloc(c, "X", uint64(bt*h*2), m.x)
		b.alloc(c, "Wq", uint64(nq*D*h*2), rows(m.wq, h, c.t*nq*D, (c.t+1)*nq*D))
		b.alloc(c, "Wk", uint64(hkv*D*h*2), rows(m.wk, h, kv0*D, (kv0+hkv)*D))
		b.alloc(c, "Wv", uint64(hkv*D*h*2), rows(m.wv, h, kv0*D, (kv0+hkv)*D))
		b.alloc(c, "Wo", uint64(h*hO*D*2), cols(m.wo, h, cfg.NQ*D, c.index*hO*D, (c.index+1)*hO*D))
		b.alloc(c, "Wgu", uint64(2*f8*h*2), concat(
			rows(m.wg, h, c.index*f8, (c.index+1)*f8), rows(m.wu, h, c.index*f8, (c.index+1)*f8)))
		b.alloc(c, "Wd", uint64(h*f8*2), cols(m.wd, h, cfg.F, c.index*f8, (c.index+1)*f8))

		b.alloc(c, "Kc", uint64(len(kc)*2), kc)
		b.alloc(c, "Vtc", uint64(len(vt)*2), vt)
		b.alloc(c, "kvScratch", uint64(bt*hkv*D*2), nil)
		b.alloc(c, "partials", uint64((h/128)*16*maxN*4), nil)
		b.alloc(c, "q", uint64(bt*nq*D*2), nil)
		b.alloc(c, "Orec", uint64(splits*bt*nq*D*4), nil)
		b.alloc(c, "Mrec", uint64(splits*bt*nq*4), nil)
		b.alloc(c, "Lrec", uint64(splits*bt*nq*4), nil)
		b.alloc(c, "Opart", uint64(bt*nq*D*2), nil)
		b.alloc(c, "lse", uint64(bt*nq*4), nil)

		if cfg.KVP == 2 {
			b.alloc(c, "combO", uint64(2*bt*hO*D*2), nil)
			b.alloc(c, "combL", uint64(2*bt*hO*4), nil)
			b.alloc(c, "Oin", uint64(bt*hO*D*2), nil)
		} else {
			c.buf["Oin"] = c.buf["Opart"]
		}

		b.alloc(c, "Y", uint64(bt*h*2), nil)
		b.alloc(c, "G", uint64(bt*2*f8*2), nil)
		b.alloc(c, "A", uint64(bt*f8*2), nil)
		b.alloc(c, "Y2", uint64(bt*h*2), nil)
	}
}

func (b *Benchmark) runDecode() {
	cfg := b.Cfg
	h, bt := cfg.H, cfg.Batch
	hO, f8 := b.hO(), b.f8()

	if cfg.runsHead() {
		b.runDecodeHead()
	}

	if !cfg.runsTail() {
		return
	}

	// 4. 8-way O-proj and all-reduce.
	var oproj []launch
	for _, c := range b.chips {
		oproj = append(oproj, b.skinnyGemm(c, c.buf["Oin"], c.buf["Wo"], c.buf["Y"], bt, h, hO*D, h, 1)...)
	}
	b.stage("oproj", "compute", oproj)
	b.allReduce("allreduce_oproj", "Y")

	// 5. 8-way FFN and all-reduce.
	var ffn []launch
	for _, c := range b.chips {
		ffn = append(ffn, b.skinnyGemm(c, c.buf["Y"], c.buf["Wgu"], c.buf["G"], bt, 2*f8, h, 2*f8, 1)...)
		ffn = append(ffn, launch{chip: c, kernel: "swiglu", grid: [3]int{(bt*f8 + wgSize - 1) / wgSize},
			args: &swigluArgs{In: c.buf["G"], Out: c.buf["A"], T: int32(bt), F: int32(f8)}})
		ffn = append(ffn, b.skinnyGemm(c, c.buf["A"], c.buf["Wd"], c.buf["Y2"], bt, h, f8, h, 1)...)
	}
	b.stage("ffn", "compute", ffn)
	b.allReduce("allreduce_ffn", "Y2")
}

// runDecodeHead runs the stages that differ between the configurations: QKV
// with the KV append, attention over the local context, and (KVP2) Helix.
func (b *Benchmark) runDecodeHead() {
	cfg := b.Cfg
	h, bt := cfg.H, cfg.Batch
	nq, hkv, hO := b.nq(), b.hkv(), b.hO()
	ks, splits := b.kStride(), cfg.autoDecodeSplits()

	// 1. QKV (duplicated over KVP); the owner appends the new K/V.
	var qkv []launch
	for _, c := range b.chips {
		_, n, owner := b.ctxShard(c)
		kOut, kLdc := c.buf["kvScratch"], hkv*D
		vOut, vLdc, vLdn := c.buf["kvScratch"], hkv*D, 1

		if owner {
			kOut, kLdc = c.buf["Kc"]+driver.Ptr(n*hkv*D*2), ks*hkv*D
			vOut, vLdc, vLdn = c.buf["Vtc"]+driver.Ptr(n*2), hkv*D*ks, ks
		}

		qkv = append(qkv, b.skinnyGemm(c, c.buf["X"], c.buf["Wq"], c.buf["q"], bt, nq*D, h, nq*D, 1)...)
		qkv = append(qkv, b.skinnyGemm(c, c.buf["X"], c.buf["Wk"], kOut, bt, hkv*D, h, kLdc, 1)...)
		qkv = append(qkv, b.skinnyGemm(c, c.buf["X"], c.buf["Wv"], vOut, bt, hkv*D, h, vLdc, vLdn)...)
	}
	b.stage("qkv", "compute", qkv)

	// 2. Split attention over the local context, merged into (O, lse).
	var attn []launch
	for _, c := range b.chips {
		_, n, owner := b.ctxShard(c)
		if owner {
			n++
		}

		rowsN := bt * nq
		attn = append(attn,
			launch{chip: c, kernel: "attn_decode", grid: [3]int{1, hkv, bt * splits / 4},
				args: &attnArgs{Q: c.buf["q"], K: c.buf["Kc"], Vt: c.buf["Vtc"],
					O: c.buf["Orec"], M: c.buf["Mrec"], L: c.buf["Lrec"],
					NPos: 1, NQ: int32(nq), NKV: int32(hkv), NKeys: int32(n), VtStride: int32(ks),
					KeysPerSplit: int32(roundUp16((n + splits - 1) / splits)), Ranks: 1, Scale: attnScale(),
					Splits: int32(splits), QBatchStride: int32(nq * D), KBatchStride: int32(ks * hkv * D),
					VtBatchStride: int32(hkv * D * ks), RecBatchStride: int32(nq), RecSplitStride: int32(rowsN)}},
			launch{chip: c, kernel: "combine_splits", grid: [3]int{(rowsN*D + wgSize - 1) / wgSize},
				args: &combineSplitsArgs{O: c.buf["Orec"], M: c.buf["Mrec"], L: c.buf["Lrec"], Out: c.buf["Opart"],
					Lse: c.buf["lse"], NRows: int32(rowsN), Splits: int32(splits), OutRowStride: D}})
	}
	b.stage("attention", "compute", attn)

	// 3. Helix: all-to-all of (O, lse) for this chiplet's O-proj heads, then
	// the LSE-weighted combine.
	if cfg.KVP == 2 {
		var a2a, comb []launch
		for _, c := range b.chips {
			for src := 0; src < 2; src++ {
				s := b.chips[2*c.t+src]
				queue := 0
				if s != c {
					queue = 1
				}
				a2a = append(a2a,
					b.copy2dLaunch(c, queue, s.buf["Opart"]+driver.Ptr(c.r*hO*D*2),
						c.buf["combO"]+driver.Ptr(src*bt*hO*D*2), bt, hO*D*2, nq*D*2, hO*D*2),
					b.copy2dLaunch(c, queue, s.buf["lse"]+driver.Ptr(c.r*hO*4),
						c.buf["combL"]+driver.Ptr(src*bt*hO*4), bt, hO*4, nq*4, hO*4))
			}

			rowsN := bt * hO
			comb = append(comb, launch{chip: c, kernel: "combine_lse", grid: [3]int{(rowsN*D + wgSize - 1) / wgSize},
				args: &combineLseArgs{O: c.buf["combO"], Lse: c.buf["combL"], Out: c.buf["Oin"],
					NRows: int32(rowsN), Parts: 2, OutRowStride: D}})
		}
		b.stage("helix_alltoall", "net", a2a)
		b.stage("helix_combine", "compute", comb)
	}

}

// allReduce is a ring reduce-scatter followed by a ring all-gather over all 8
// chiplets of a [Batch][H] buffer.
func (b *Benchmark) allReduce(name, buf string) {
	ring := b.ring8()
	chunk := uint64(b.Cfg.Batch * b.Cfg.H * 2 / numChips)
	pos := func(p int) int { return p }

	phases := b.ringReduceScatter(ring, buf, pos, chunk)
	phases = append(phases, b.ringAllGather(ring, buf, pos, chunk)...)
	b.stage(name, "net", phases...)
}

func (b *Benchmark) verifyDecode() {
	got := b.readBF16(b.chips[0].buf["Y2"], b.Cfg.Batch*b.Cfg.H)
	b.checkResult(got, b.host.referenceDecode())
}

// Verify compares the layer output with the CPU reference.
func (b *Benchmark) Verify() {
	if b.Cfg.Phase == "prefill" {
		b.verifyPrefill()
	} else {
		b.verifyDecode()
	}
}

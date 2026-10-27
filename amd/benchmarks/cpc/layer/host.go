package layer

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// hostModel holds the full layer weights and inputs (bf16) and computes the
// CPU reference. Weights are stored [out][in].
type hostModel struct {
	cfg Config

	wq, wk, wv []uint16 // [NQ*D][H], [NKV*D][H], [NKV*D][H]
	wo         []uint16 // [H][NQ*D]
	wg, wu     []uint16 // [F][H]
	wd         []uint16 // [H][F]

	x []uint16 // prefill: [S][H] in position order; decode: [Batch][H]

	// decode caches: [Batch][S][NKV][D]
	kc, vc []uint16
}

func f2bf(f float32) uint16 {
	u := math.Float32bits(f)
	u += 0x7fff + ((u >> 16) & 1)

	return uint16(u >> 16)
}

func bf2f(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }

func roundBF(f float32) float32 { return bf2f(f2bf(f)) }

func randBF(rng *rand.Rand, n int, scale float64) []uint16 {
	v := make([]uint16, n)
	for i := range v {
		v[i] = f2bf(float32((rng.Float64()*2 - 1) * scale))
	}

	return v
}

func newHostModel(c Config) *hostModel {
	rng := rand.New(rand.NewSource(1))
	h, hq, hk := c.H, c.NQ*D, c.NKV*D
	w := func(out, in int) []uint16 { return randBF(rng, out*in, math.Sqrt(3.0/float64(in))) }

	m := &hostModel{
		cfg: c,
		wq:  w(hq, h), wk: w(hk, h), wv: w(hk, h),
		wo: w(h, hq),
		wg: w(c.F, h), wu: w(c.F, h),
		wd: w(h, c.F),
	}

	if c.Phase == "prefill" {
		m.x = randBF(rng, c.S*h, 1)
	} else {
		m.x = randBF(rng, c.Batch*h, 1)
		m.kc = randBF(rng, c.Batch*c.S*hk, 1)
		m.vc = randBF(rng, c.Batch*c.S*hk, 1)
	}

	return m
}

// rows returns rows [r0, r1) of a row-major [*][cols] matrix.
func rows(m []uint16, cols, r0, r1 int) []uint16 { return m[r0*cols : r1*cols] }

// cols returns columns [c0, c1) of a row-major [nRows][nCols] matrix.
func cols(m []uint16, nRows, nCols, c0, c1 int) []uint16 {
	out := make([]uint16, 0, nRows*(c1-c0))
	for r := 0; r < nRows; r++ {
		out = append(out, m[r*nCols+c0:r*nCols+c1]...)
	}

	return out
}

func concat(parts ...[]uint16) []uint16 {
	var out []uint16
	for _, p := range parts {
		out = append(out, p...)
	}

	return out
}

// matmulNT returns round_bf16(A[M][K] * B[N][K]^T) as float32.
func matmulNT(a []float32, b []uint16, m, n, k int) []float32 {
	bf := toF32(b)
	out := make([]float32, m*n)

	parallelFor(m, func(i int) {
		ar := a[i*k : (i+1)*k]
		for j := 0; j < n; j++ {
			br := bf[j*k : (j+1)*k]
			s := float32(0)
			for x := range ar {
				s += ar[x] * br[x]
			}
			out[i*n+j] = roundBF(s)
		}
	})

	return out
}

func toF32(v []uint16) []float32 {
	out := make([]float32, len(v))
	for i, h := range v {
		out[i] = bf2f(h)
	}

	return out
}

func parallelFor(n int, f func(i int)) {
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)
	next := make(chan int, n)

	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				f(i)
			}
		}()
	}
	wg.Wait()
}

// attend returns round_bf16(softmax(q k^T / sqrt(D)) v) for one query row
// over keys [0, n) of k and v ([n][D] each, row stride ldkv), with keys whose
// position exceeds qpos masked when keyPos is given.
func attend(q []float32, k, v []float32, n, ldkv int, keyPos func(int) int, qpos int) []float32 {
	scores := make([]float64, n)
	mx := math.Inf(-1)

	for j := 0; j < n; j++ {
		if keyPos != nil && keyPos(j) > qpos {
			scores[j] = math.Inf(-1)
			continue
		}

		s := 0.0
		for x := 0; x < D; x++ {
			s += float64(q[x]) * float64(k[j*ldkv+x])
		}
		scores[j] = s / math.Sqrt(D)
		mx = math.Max(mx, scores[j])
	}

	out := make([]float64, D)
	den := 0.0

	for j := 0; j < n; j++ {
		if math.IsInf(scores[j], -1) {
			continue
		}

		p := math.Exp(scores[j] - mx)
		den += p
		for x := 0; x < D; x++ {
			out[x] += p * float64(v[j*ldkv+x])
		}
	}

	res := make([]float32, D)
	for x := range res {
		res[x] = roundBF(float32(out[x] / den))
	}

	return res
}

// ffnAndOproj applies O-proj and the SwiGLU FFN to attention outputs o
// ([T][NQ*D]), returning [T][H].
func (m *hostModel) ffnAndOproj(o []float32, t int) []float32 {
	c := m.cfg
	y := matmulNT(o, m.wo, t, c.H, c.NQ*D)
	g := matmulNT(y, m.wg, t, c.F, c.H)
	u := matmulNT(y, m.wu, t, c.F, c.H)

	a := make([]float32, t*c.F)
	for i := range a {
		a[i] = roundBF(g[i] / (1 + float32(math.Exp(float64(-g[i])))) * u[i])
	}

	return matmulNT(a, m.wd, t, c.H, c.F)
}

// referencePrefill returns the layer output [S][H] in position order.
func (m *hostModel) referencePrefill() []float32 {
	c := m.cfg
	x := toF32(m.x)
	q := matmulNT(x, m.wq, c.S, c.NQ*D, c.H)
	k := matmulNT(x, m.wk, c.S, c.NKV*D, c.H)
	v := matmulNT(x, m.wv, c.S, c.NKV*D, c.H)

	group := c.NQ / c.NKV
	o := make([]float32, c.S*c.NQ*D)
	identity := func(j int) int { return j }

	parallelFor(c.S*c.NQ, func(i int) {
		pos, h := i/c.NQ, i%c.NQ
		kvh := h / group
		res := attend(q[i*D:(i+1)*D], k[kvh*D:], v[kvh*D:], pos+1, c.NKV*D, identity, pos)
		copy(o[i*D:], res)
	})

	return m.ffnAndOproj(o, c.S)
}

// referenceDecode returns the layer output [Batch][H] for one new token per
// request, attending over the S cached tokens and itself.
func (m *hostModel) referenceDecode() []float32 {
	c := m.cfg
	hk := c.NKV * D
	x := toF32(m.x)
	q := matmulNT(x, m.wq, c.Batch, c.NQ*D, c.H)
	kNew := matmulNT(x, m.wk, c.Batch, hk, c.H)
	vNew := matmulNT(x, m.wv, c.Batch, hk, c.H)

	group := c.NQ / c.NKV
	o := make([]float32, c.Batch*c.NQ*D)

	parallelFor(c.Batch*c.NQ, func(i int) {
		b, h := i/c.NQ, i%c.NQ
		kvh := h / group
		k := append(toF32(m.kc[b*c.S*hk:(b+1)*c.S*hk]), kNew[b*hk:(b+1)*hk]...)
		v := append(toF32(m.vc[b*c.S*hk:(b+1)*c.S*hk]), vNew[b*hk:(b+1)*hk]...)
		res := attend(q[i*D:(i+1)*D], k[kvh*D:], v[kvh*D:], c.S+1, hk, nil, 0)
		copy(o[i*D:], res)
	})

	return m.ffnAndOproj(o, c.Batch)
}

// compare returns the largest absolute error relative to the largest
// reference magnitude.
func compare(got, want []float32) (relErr float64) {
	maxErr, maxRef := 0.0, 0.0
	for i := range want {
		maxErr = math.Max(maxErr, math.Abs(float64(got[i]-want[i])))
		maxRef = math.Max(maxRef, math.Abs(float64(want[i])))
	}

	if math.IsNaN(maxErr) {
		return math.Inf(1)
	}

	return maxErr / math.Max(maxRef, 1e-6)
}

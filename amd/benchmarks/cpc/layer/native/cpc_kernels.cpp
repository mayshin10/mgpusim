/*
 * Kernels for one transformer layer under TP8 and TP4xKVP2 (cpc-analytic):
 * zigzag Ring Attention prefill and Helix decode, on gfx942 (CDNA3).
 *
 * Conventions
 *  - bf16 is stored as uint16_t; all accumulation is fp32.
 *  - Matrix multiplies use v_mfma_f32_16x16x16_bf16. With l = lane id,
 *    r = l % 16 and kq = 4 * (l / 16), a lane holds
 *      A[m][k]: m = r,       k = kq + j   (4 bf16, contiguous in k)
 *      B[k][n]: k = kq + j,  n = r        (4 bf16, contiguous in k)
 *      D[m][n]: m = kq + j,  n = r        (4 fp32)
 *  - GEMMs compute C = A * B^T with B stored [N][K] (weights as [out][in]),
 *    so both operands are 8-byte contiguous loads.
 *  - Attention computes S^T = K * Q^T so the softmax probabilities land in the
 *    A-operand layout of P * V. V is cached transposed, Vt[kv head][D][keys],
 *    so the B operand of P * V is also an 8-byte load.
 *  - Softmax runs in the log2 domain: scores are scaled by log2(e)/sqrt(D),
 *    and a partial result is the record (o = sum p v, m = running max,
 *    l = sum p) or, once normalized, (O, lse2 = m + log2 l).
 *  - No kernel reads blockDim/gridDim (no hidden kernel arguments); every
 *    launch uses 256-thread work-groups.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint16_t bf16;
typedef short bf16x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

#define WG 256
#define D 128
#define NEG_INF (-__builtin_inff())

__device__ inline float bf2f(bf16 h) { return __uint_as_float(((uint32_t)h) << 16); }

__device__ inline bf16 f2bf(float f) {  // round to nearest even
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

__device__ inline f32x4 mfma(bf16x4 a, bf16x4 b, f32x4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

__device__ inline bf16x4 ld4(const bf16* p) { return *(const bf16x4*)p; }

__device__ inline int lane_id() { return threadIdx.x & 63; }
__device__ inline int wave_id() { return threadIdx.x >> 6; }

/* ------------------------------------------------------------------ GEMM */

typedef short bf16x8 __attribute__((ext_vector_type(8)));

__device__ inline bf16x8 ld8(const bf16* p) { return *(const bf16x8*)p; }
__device__ inline bf16x4 lo4(bf16x8 v) { return (bf16x4){v[0], v[1], v[2], v[3]}; }
__device__ inline bf16x4 hi4(bf16x8 v) { return (bf16x4){v[4], v[5], v[6], v[7]}; }

// K is consumed in chunks of 32: lane group g = l / 16 loads elements
// [8g, 8g + 8) of the chunk with one 16-byte load and feeds the low and high
// halves to two MFMAs. A and B use the same permutation of k within the
// chunk, so the sum is unchanged, and every load reads whole 64-byte lines.

// C[M][N] (bf16, row stride ldc) = A[M][K] * B[N][K]^T. M, N multiples of 128,
// K multiple of 64. Grid (N/128, M/128); each wave owns a 64x64 tile. Two
// fragment buffers alternate (ping-pong), so one K chunk's loads are in flight
// while the other chunk's MFMAs run, without register copies that would make
// the compiler wait for every outstanding load.
__device__ inline void gemm_load(bf16x8 (&a)[4], bf16x8 (&b)[4], const bf16* pa, const bf16* pb,
                                 int lda, int ldb, int k) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        a[i] = ld8(pa + (size_t)16 * i * lda + k);
        b[i] = ld8(pb + (size_t)16 * i * ldb + k);
    }
}

__device__ inline void gemm_compute(f32x4 (&acc)[4][4], const bf16x8 (&a)[4], const bf16x8 (&b)[4]) {
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            acc[i][j] = mfma(lo4(a[i]), lo4(b[j]), acc[i][j]);
            acc[i][j] = mfma(hi4(a[i]), hi4(b[j]), acc[i][j]);
        }
}

// With P != nullptr the GEMM is split over K: work-group z of gridDim.z
// (blockIdx.z) sums K slice [z * K, (z + 1) * K) and writes fp32 partials to
// P[z][m][n] (row stride ldc); reduce_partials then adds them up. This gives
// small-M GEMMs enough work-groups to fill the compute units.
extern "C" __global__ void __launch_bounds__(WG, 1) gemm_nt(const bf16* A, const bf16* B, bf16* C,
                                   int K, int lda, int ldb, int ldc, float* P, int M) {
    const int l = lane_id(), w = wave_id(), r = l & 15, kg = (l >> 4) * 8, kq = (l >> 4) * 4;
    const int m0 = blockIdx.y * 128 + (w >> 1) * 64;
    const int n0 = blockIdx.x * 128 + (w & 1) * 64;
    const size_t k0 = (size_t)blockIdx.z * K;
    const bf16* pa = A + (size_t)(m0 + r) * lda + kg + k0;
    const bf16* pb = B + (size_t)(n0 + r) * ldb + kg + k0;

    f32x4 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = (f32x4){0, 0, 0, 0};

    bf16x8 a0[4], b0[4], a1[4], b1[4];
    gemm_load(a0, b0, pa, pb, lda, ldb, 0);
    for (int k = 0; k < K; k += 64) {
        gemm_load(a1, b1, pa, pb, lda, ldb, k + 32);
        gemm_compute(acc, a0, b0);
        if (k + 64 < K) gemm_load(a0, b0, pa, pb, lda, ldb, k + 64);
        gemm_compute(acc, a1, b1);
    }

    if (P) {
        float* pz = P + (size_t)blockIdx.z * M * ldc;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int e = 0; e < 4; e++)
                    pz[(size_t)(m0 + 16 * i + kq + e) * ldc + n0 + 16 * j + r] = acc[i][j][e];
        return;
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++)
                C[(size_t)(m0 + 16 * i + kq + e) * ldc + n0 + 16 * j + r] = f2bf(acc[i][j][e]);
}

// C[m][n] (bf16, row stride ldc) = sum over z of P[z][m][n] (fp32, row
// stride ldc); four elements per thread.
extern "C" __global__ void __launch_bounds__(WG, 1) reduce_partials(const float* P, bf16* C, int M, int N,
                                                                    int splits, int ldc) {
    int i = (blockIdx.x * WG + threadIdx.x) * 4;
    if (i >= M * N) return;
    int m = i / N, n = i % N;
    float4 s = {0, 0, 0, 0};
    for (int z = 0; z < splits; z++) {
        float4 v = *(const float4*)(P + ((size_t)z * M + m) * ldc + n);
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    bf16* c = C + (size_t)m * ldc + n;
    c[0] = f2bf(s.x); c[1] = f2bf(s.y); c[2] = f2bf(s.z); c[3] = f2bf(s.w);
}

// Skinny GEMM for M <= 16 (decode): fp32 partial sums over one K slice,
// P[split][16][N] = A[0:M][slice] * B[N][slice]^T. Grid (N/64, splits); each
// wave owns 16 columns and keeps 4 chunks of loads in flight. k_per_split is
// a multiple of 128. Rows >= M of A read as zero.
extern "C" __global__ void __launch_bounds__(WG, 1) gemm_skinny(const bf16* A, const bf16* B, float* P,
                                       int M, int N, int k_per_split, int lda, int ldb) {
    const int l = lane_id(), w = wave_id(), r = l & 15, kg = (l >> 4) * 8, kq = (l >> 4) * 4;
    const int n0 = blockIdx.x * 64 + w * 16;
    const int k_begin = blockIdx.y * k_per_split, k_end = k_begin + k_per_split;
    const bf16* pa = A + (size_t)r * lda + kg;
    const bf16* pb = B + (size_t)(n0 + r) * ldb + kg;
    const bf16x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};

    f32x4 acc0 = {0, 0, 0, 0}, acc1 = {0, 0, 0, 0};
    for (int k = k_begin; k < k_end; k += 128) {
        bf16x8 a[4], b[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            a[u] = r < M ? ld8(pa + k + 32 * u) : zero;
            b[u] = ld8(pb + k + 32 * u);
        }
#pragma unroll
        for (int u = 0; u < 4; u++) {
            acc0 = mfma(lo4(a[u]), lo4(b[u]), acc0);
            acc1 = mfma(hi4(a[u]), hi4(b[u]), acc1);
        }
    }

#pragma unroll
    for (int e = 0; e < 4; e++) {
        int m = kq + e;
        if (m < M) P[((size_t)blockIdx.y * 16 + m) * N + n0 + r] = acc0[e] + acc1[e];
    }
}

// C[m * ldc + n * ldn] (bf16) = sum over splits of P[split][m][n]; one
// thread per element. ldn > 1 writes columns strided (e.g., into a transposed
// V cache).
extern "C" __global__ void __launch_bounds__(WG, 1) reduce_splits(const float* P, bf16* C, int M, int N, int splits,
                                         int ldc, int ldn) {
    int i = blockIdx.x * WG + threadIdx.x;
    if (i >= M * N) return;
    int m = i / N, n = i % N;
    float s = 0;
    for (int z = 0; z < splits; z++) s += P[((size_t)z * 16 + m) * N + n];
    C[(size_t)m * ldc + (size_t)n * ldn] = f2bf(s);
}

// out[t][f] = silu(in[t][f]) * in[t][F + f]: the gate and up halves of one row.
extern "C" __global__ void __launch_bounds__(WG, 1) swiglu(const bf16* in, bf16* out, int T, int F) {
    int i = blockIdx.x * WG + threadIdx.x;
    if (i >= T * F) return;
    int t = i / F, f = i % F;
    float g = bf2f(in[(size_t)t * 2 * F + f]), u = bf2f(in[(size_t)t * 2 * F + F + f]);
    out[i] = f2bf(g / (1.0f + __expf(-g)) * u);
}

/* -------------------------------------------------------- collectives */

// dst[i] = src[i] for n16 16-byte words; either side may be a peer's memory.
// stride = total threads in the grid.
extern "C" __global__ void __launch_bounds__(WG, 1) copy16(const uint4* src, uint4* dst, int n16, int stride) {
    for (int i = blockIdx.x * WG + threadIdx.x; i < n16; i += stride) dst[i] = src[i];
}

// Strided 2D copy in 4-byte words: rows x words, with row strides in words.
extern "C" __global__ void __launch_bounds__(WG, 1) copy2d(const uint32_t* src, uint32_t* dst, int rows, int words,
                                  int src_stride, int dst_stride, int stride) {
    for (int i = blockIdx.x * WG + threadIdx.x; i < rows * words; i += stride) {
        int r = i / words, w = i % words;
        dst[(size_t)r * dst_stride + w] = src[(size_t)r * src_stride + w];
    }
}

// dst[i] += src[i] (bf16, fp32 math) for n8 groups of 8 elements.
extern "C" __global__ void __launch_bounds__(WG, 1) add_bf16(const uint4* src, uint4* dst, int n8, int stride) {
    for (int i = blockIdx.x * WG + threadIdx.x; i < n8; i += stride) {
        uint4 a = src[i], b = dst[i];
        uint32_t* pa = (uint32_t*)&a;
        uint32_t* pb = (uint32_t*)&b;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            float lo = bf2f(pa[k] & 0xffff) + bf2f(pb[k] & 0xffff);
            float hi = bf2f(pa[k] >> 16) + bf2f(pb[k] >> 16);
            pb[k] = (uint32_t)f2bf(lo) | ((uint32_t)f2bf(hi) << 16);
        }
        dst[i] = b;
    }
}

/* ---------------------------------------------------------- attention */

struct AttnArgs {
    const bf16* q;   // [pos][q heads][D]
    const bf16* k;   // [key][kv heads][D]
    const bf16* vt;  // [kv heads][D][key stride]
    float* o;        // records: [split][pos][q heads][D]
    float* m;        //          [split][pos][q heads]
    float* l;
    int n_pos;       // query positions (rows = n_pos * group)
    int nq;          // q heads on this chiplet
    int nkv;         // kv heads on this chiplet
    int n_keys;
    int vt_stride;   // keys per Vt row
    int keys_per_split;
    int causal;      // zigzag causal mask on
    int q_rank, k_rank, ranks, chunk;  // zigzag position mapping
    int merge;       // continue from the records already in o/m/l
    float scale;     // log2(e) / sqrt(D)
    // decode batch: blockIdx.z = batch * splits + split, strides per request
    int splits;
    int q_batch_stride, k_batch_stride, vt_batch_stride, rec_batch_stride;
    int rec_split_stride;  // records between consecutive splits
};

// Global position of local token t on rank `rank` of `ranks` (zigzag: rank r
// holds chunks r and 2*ranks-1-r, each `chunk` tokens).
__device__ inline int zz_pos(int t, int rank, int ranks, int chunk) {
    return t < chunk ? rank * chunk + t : (2 * ranks - 1 - rank) * chunk + (t - chunk);
}

// Running state of one wave's 16 query rows. m and l live in every lane of
// the row (l % 16 == row index); O rows follow the D layout (row kq + e).
struct AttnState {
    float m, l;
    f32x4 o[8];
};

// K and V fragments of one 16-key block: A operand K[key][d] for the 8
// d-steps of S^T = K Q^T, and B operand V^T[d][key] for the 8 d-tiles of P V.
struct KVFrag {
    bf16x4 k[8], v[8];
};

__device__ inline void attn_load(KVFrag& f, const AttnArgs& a, int kvh, int kb) {
    const int l = lane_id(), r = l & 15, kq = (l >> 4) * 4;
    const bf16* kp = a.k + ((size_t)(kb + r) * a.nkv + kvh) * D + kq;
    const bf16* vp = a.vt + ((size_t)kvh * D + r) * a.vt_stride + kb + kq;
#pragma unroll
    for (int t = 0; t < 8; t++) {
        f.k[t] = ld4(kp + 16 * t);
        f.v[t] = ld4(vp + (size_t)16 * t * a.vt_stride);
    }
}

// Online-softmax update of the wave's rows with one 16-key block.
__device__ inline void attn_block(AttnState& st, const KVFrag& f, const bf16x4 (&qf)[8], const AttnArgs& a,
                                  int kb, int k_end, int qpos) {
    const int l = lane_id(), kq = (l >> 4) * 4;

    // S^T[key][row] = K[key][:] . Q[row][:]
    f32x4 s = {0, 0, 0, 0};
#pragma unroll
    for (int t = 0; t < 8; t++) s = mfma(f.k[t], qf[t], s);

    float blk_max = NEG_INF;
#pragma unroll
    for (int e = 0; e < 4; e++) {
        int key = kb + kq + e;
        bool masked = key >= k_end || (a.causal && zz_pos(key, a.k_rank, a.ranks, a.chunk) > qpos);
        s[e] = masked ? NEG_INF : s[e] * a.scale;
        blk_max = fmaxf(blk_max, s[e]);
    }
    blk_max = fmaxf(blk_max, __shfl_xor(blk_max, 16));
    blk_max = fmaxf(blk_max, __shfl_xor(blk_max, 32));

    float m_new = fmaxf(st.m, blk_max);
    float m_use = m_new == NEG_INF ? 0.0f : m_new;  // all masked so far
    float alpha = exp2f(st.m - m_use);
    float psum = 0;
    bf16x4 pf;
#pragma unroll
    for (int e = 0; e < 4; e++) {
        float p = exp2f(s[e] - m_use);
        psum += p;
        pf[e] = (short)f2bf(p);
    }
    psum += __shfl_xor(psum, 16);
    psum += __shfl_xor(psum, 32);
    st.l = st.l * alpha + psum;
    st.m = m_new;

    // Rescale O rows (row kq + e takes the alpha of lane kq + e), then
    // O[row][d] += P[row][key] * V[key][d].
#pragma unroll
    for (int e = 0; e < 4; e++) {
        float ae = __shfl(alpha, kq + e);
#pragma unroll
        for (int t = 0; t < 8; t++) st.o[t][e] *= ae;
    }
#pragma unroll
    for (int t = 0; t < 8; t++) st.o[t] = mfma(pf, f.v[t], st.o[t]);
}

// Processes keys [kb, ke) in 16-key blocks with two fragment buffers, so the
// next block's K/V loads are in flight while the current block computes.
__device__ inline void attn_range(AttnState& st, const bf16x4 (&qf)[8], const AttnArgs& a, int kvh,
                                  int kb, int ke, int k_end, int qpos) {
    if (kb >= ke) return;
    KVFrag f0, f1;
    attn_load(f0, a, kvh, kb);
    for (; kb < ke; kb += 32) {
        bool second = kb + 16 < ke;
        if (second) attn_load(f1, a, kvh, kb + 16);
        attn_block(st, f0, qf, a, kb, k_end, qpos);
        if (kb + 32 < ke) attn_load(f0, a, kvh, kb + 32);
        if (second) attn_block(st, f1, qf, a, kb + 16, k_end, qpos);
    }
}

// Local keys of zigzag chunk `c` (0 or 1) that can be visible to a query at
// global position <= qmax: [c * chunk, c * chunk + n), rounded up to 16.
__device__ inline int visible_in_chunk(const AttnArgs& a, int c, int qmax) {
    int first = c == 0 ? a.k_rank * a.chunk : (2 * a.ranks - 1 - a.k_rank) * a.chunk;
    int n = min(max(qmax - first + 1, 0), a.chunk);
    return (n + 15) / 16 * 16;
}

// One wave: 16 query rows (row = pos * group + head-in-group) of kv head kvh
// against keys [k_begin, k_end), continuing/merging the rows' records.
__device__ void attn_wave(const AttnArgs& a, int row0, int kvh, int k_begin, int k_end, int split) {
    const int l = lane_id(), r = l & 15, kq = (l >> 4) * 4;
    const int group = a.nq / a.nkv, rows = a.n_pos * group;
    const int row = min(row0 + r, rows - 1);  // rows past the end recompute the last
    const int pos = row / group, hq = kvh * group + row % group;
    const int qpos = a.causal ? zz_pos(pos, a.q_rank, a.ranks, a.chunk) : 0;
    const size_t rec = (size_t)split * a.rec_split_stride + (size_t)pos * a.nq + hq;

    // Q fragments: B operand Q^T[d][row] for the 8 d-steps of 16.
    bf16x4 qf[8];
#pragma unroll
    for (int s = 0; s < 8; s++) qf[s] = ld4(a.q + ((size_t)pos * a.nq + hq) * D + 16 * s + kq);

    AttnState st;
    st.m = NEG_INF;
    st.l = 0;
#pragma unroll
    for (int t = 0; t < 8; t++) st.o[t] = (f32x4){0, 0, 0, 0};
    if (a.merge) {
        st.m = a.m[rec];
        st.l = a.l[rec];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int orow = min(row0 + kq + e, rows - 1);
            size_t orec = (size_t)split * a.rec_split_stride + (size_t)(orow / group) * a.nq +
                          kvh * group + orow % group;
#pragma unroll
            for (int t = 0; t < 8; t++) st.o[t][e] = a.o[orec * D + 16 * t + r];
        }
    }

    if (a.causal) {
        // Highest query position in this wave; keys beyond it are skipped.
        int qmax = qpos;
        qmax = max(qmax, __shfl_xor(qmax, 1));
        qmax = max(qmax, __shfl_xor(qmax, 2));
        qmax = max(qmax, __shfl_xor(qmax, 4));
        qmax = max(qmax, __shfl_xor(qmax, 8));
        attn_range(st, qf, a, kvh, 0, visible_in_chunk(a, 0, qmax), k_end, qpos);
        attn_range(st, qf, a, kvh, a.chunk, a.chunk + visible_in_chunk(a, 1, qmax), k_end, qpos);
    } else {
        attn_range(st, qf, a, kvh, k_begin, k_end, k_end, qpos);
    }

    if (row0 + r < rows && l < 16) {
        a.m[rec] = st.m;
        a.l[rec] = st.l;
    }
#pragma unroll
    for (int e = 0; e < 4; e++) {
        int orow = row0 + kq + e;
        if (orow >= rows) continue;
        size_t orec = (size_t)split * a.rec_split_stride + (size_t)(orow / group) * a.nq +
                      kvh * group + orow % group;
#pragma unroll
        for (int t = 0; t < 8; t++) a.o[orec * D + 16 * t + r] = st.o[t][e];
    }
}

__device__ inline AttnArgs batch_view(AttnArgs a, int b) {
    a.q += (size_t)b * a.q_batch_stride;
    a.k += (size_t)b * a.k_batch_stride;
    a.vt += (size_t)b * a.vt_batch_stride;
    a.o += (size_t)b * a.rec_batch_stride * D;
    a.m += (size_t)b * a.rec_batch_stride;
    a.l += (size_t)b * a.rec_batch_stride;
    return a;
}

// Prefill: grid (ceil(rows/64), kv heads); each wave takes 16 rows over all keys.
extern "C" __global__ void __launch_bounds__(WG, 1) attn_prefill(AttnArgs a) {
    int rows = a.n_pos * (a.nq / a.nkv);
    int row0 = (blockIdx.x * 4 + wave_id()) * 16;
    if (row0 >= rows) return;
    attn_wave(a, row0, blockIdx.y, 0, a.n_keys, 0);
}

// Decode: grid (1, kv heads, batch * splits/4); the 4 waves of a work-group
// take consecutive key splits of one request (all rows of the kv head's group).
extern "C" __global__ void __launch_bounds__(WG, 1) attn_decode(AttnArgs a) {
    int b = blockIdx.z / (a.splits / 4);
    int split = (blockIdx.z % (a.splits / 4)) * 4 + wave_id();
    AttnArgs v = batch_view(a, b);
    int k_begin = split * a.keys_per_split;
    int k_end = min(a.n_keys, k_begin + a.keys_per_split);
    attn_wave(v, 0, blockIdx.y, k_begin, k_end, split);
}

// Merge split records into (O bf16, lse2) per row. Rows = n_rows; one thread
// per (row, d) with d in [0, D).
extern "C" __global__ void __launch_bounds__(WG, 1) combine_splits(const float* o, const float* m, const float* l,
                                          bf16* out, float* lse, int n_rows, int splits,
                                          int out_row_stride) {
    int i = blockIdx.x * WG + threadIdx.x;
    if (i >= n_rows * D) return;
    int row = i / D, d = i % D;
    float mx = NEG_INF;
    for (int s = 0; s < splits; s++) mx = fmaxf(mx, m[(size_t)s * n_rows + row]);
    float mu = mx == NEG_INF ? 0.0f : mx, den = 0, num = 0;
    for (int s = 0; s < splits; s++) {
        float w = exp2f(m[(size_t)s * n_rows + row] - mu);
        den += w * l[(size_t)s * n_rows + row];
        num += w * o[((size_t)s * n_rows + row) * D + d];
    }
    out[(size_t)row * out_row_stride + d] = f2bf(den > 0 ? num / den : 0.0f);
    if (lse && d == 0) lse[row] = mu + log2f(den);
}

// Helix combine: rows hold `parts` normalized partials (O bf16, lse2) taken
// over disjoint key sets; out = sum_p O_p * 2^(lse_p - lse) / sum. Partial p of
// row lives at O[p * n_rows + row], lse[p * n_rows + row].
extern "C" __global__ void __launch_bounds__(WG, 1) combine_lse(const bf16* O, const float* lse, bf16* out,
                                       int n_rows, int parts, int out_row_stride) {
    int i = blockIdx.x * WG + threadIdx.x;
    if (i >= n_rows * D) return;
    int row = i / D, d = i % D;
    float mx = NEG_INF;
    for (int p = 0; p < parts; p++) mx = fmaxf(mx, lse[(size_t)p * n_rows + row]);
    float den = 0, num = 0;
    for (int p = 0; p < parts; p++) {
        float w = exp2f(lse[(size_t)p * n_rows + row] - mx);
        den += w;
        num += w * bf2f(O[((size_t)p * n_rows + row) * D + d]);
    }
    out[(size_t)row * out_row_stride + d] = f2bf(num / den);
}

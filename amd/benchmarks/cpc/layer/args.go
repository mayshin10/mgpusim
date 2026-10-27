package layer

import "github.com/sarchlab/mgpusim/v5/amd/driver"

// Kernel argument structs. The driver serializes them packed (binary.Write),
// matching the kernels' kernarg layouts: every pointer precedes the 4-byte
// fields, so the C layouts have no interior padding either.

type gemmArgs struct {
	A, B, C          driver.Ptr
	K, Lda, Ldb, Ldc int32
	P                driver.Ptr // split-K partials (0: write C directly)
	M                int32
}

type reducePartialsArgs struct {
	P, C              driver.Ptr
	M, N, Splits, Ldc int32
}

type skinnyArgs struct {
	A, B, P                   driver.Ptr
	M, N, KPerSplit, Lda, Ldb int32
}

type reduceArgs struct {
	P, C                   driver.Ptr
	M, N, Splits, Ldc, Ldn int32
}

type swigluArgs struct {
	In, Out driver.Ptr
	T, F    int32
}

type copyArgs struct {
	Src, Dst    driver.Ptr
	N16, Stride int32
}

type copy2dArgs struct {
	Src, Dst                                  driver.Ptr
	Rows, Words, SrcStride, DstStride, Stride int32
}

type addArgs struct {
	Src, Dst   driver.Ptr
	N8, Stride int32
}

type attnArgs struct {
	Q, K, Vt, O, M, L driver.Ptr

	NPos, NQ, NKV, NKeys, VtStride, KeysPerSplit int32
	Causal, QRank, KRank, Ranks, Chunk, Merge    int32
	Scale                                        float32

	Splits                                                    int32
	QBatchStride, KBatchStride, VtBatchStride, RecBatchStride int32
	RecSplitStride                                            int32
}

type combineSplitsArgs struct {
	O, M, L, Out, Lse           driver.Ptr
	NRows, Splits, OutRowStride int32
}

type combineLseArgs struct {
	O, Lse, Out                driver.Ptr
	NRows, Parts, OutRowStride int32
}

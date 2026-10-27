package mi300x

import (
	"github.com/sarchlab/akita/v5/mem"
	"github.com/sarchlab/akita/v5/mem/cache/writeback"
	"github.com/sarchlab/akita/v5/mem/cache/writethroughcache"
	"github.com/sarchlab/akita/v5/mem/simplebankedmemory"
	"github.com/sarchlab/akita/v5/mem/vm/tlb"
	"github.com/sarchlab/akita/v5/simulation"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/emu"
	"github.com/sarchlab/mgpusim/v5/amd/emu/cdna3"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/shaderarray"
	"github.com/sarchlab/mgpusim/v5/amd/timing/cp"
)

// The helpers below hold the calibrated MI300X component configurations, so
// that the monolithic builder in this package and the partitioned (CPX/NPS)
// builder in package mi300xpart build identical shader arrays, caches, and
// memory controllers.

// ShaderArrayBuilder returns a shader-array builder configured like an MI300X
// XCD's shader arrays. The caller still sets the L1 address mappers.
func ShaderArrayBuilder(
	sim *simulation.Simulation,
	freq timing.Freq,
	gpuID uint64,
	numCUPerShaderArray int,
	log2CacheLineSize, log2PageSize uint64,
) shaderarray.Builder {
	return shaderarray.MakeBuilder().
		WithSimulation(sim).
		WithFreq(freq).
		WithGPUID(gpuID).
		WithNumCUs(numCUPerShaderArray).
		WithLog2CacheLineSize(log2CacheLineSize).
		WithLog2PageSize(log2PageSize).
		// Use the CDNA3 ALU for MI300X timing simulation.
		WithALUBuilder(func() emu.ALU { return cdna3.NewALU(nil) }).
		// Decode with CDNA3 semantics so FLAT/GLOBAL SADDR addressing is
		// resolved correctly (the default disassembler decodes as GCN3, which
		// mis-handles SADDR=0 -> wrong address / "page not found").
		WithDecoderBuilder(func() emu.Decoder {
			d := insts.NewDisassembler()
			d.IsCDNA3 = true
			return d
		}).
		WithWfPoolSize(8).
		WithVGPRCount([]int{32768, 32768, 32768, 32768}).
		WithNumSinglePrecisionUnits(16).
		WithVecMemInstPipelineStages(2).
		WithVecMemTransPipelineStages(4).
		WithVecMemTransPipelineWidth(8).
		WithCUMemPipelineBufferSize(64).
		WithL1VCacheSize(32 * mem.KB).
		WithL1VBankLatency(9). // ~4.3 ns at 2.1 GHz (was 7 cyc / 1.7 GHz)
		WithMemPipelineBufferSize(64).
		WithMaxCoalescingPenalty(3).
		WithRegisterScoreboard(true)
}

// L2CacheSpec returns the spec of one L2 cache bank (channel).
func L2CacheSpec(
	freq timing.Freq,
	log2CacheLineSize, byteSize uint64,
	bankLatency int,
) writeback.Spec {
	spec := writeback.DefaultSpec()
	spec.Freq = freq
	spec.Log2BlockSize = log2CacheLineSize
	spec.WayAssociativity = 16
	spec.TotalByteSize = byteSize
	spec.NumMSHREntry = 512
	spec.NumReqPerCycle = 64
	spec.BankLatency = bankLatency
	spec.DirLatency = 2
	spec.MaxInflightFetch = 512
	spec.MaxInflightEviction = 512

	return spec
}

// MALLSpec returns the spec of one MALL (Infinity Cache) bank.
func MALLSpec(
	freq timing.Freq,
	log2CacheLineSize, byteSize uint64,
	bankLatency int,
) writethroughcache.Spec {
	spec := writethroughcache.DefaultSpec()
	spec.Freq = freq
	spec.WritePolicyType = "write-through"
	spec.Log2BlockSize = log2CacheLineSize
	spec.WayAssociativity = 16
	spec.TotalByteSize = byteSize
	spec.NumMSHREntry = 512
	spec.NumReqPerCycle = 64
	spec.NumBanks = 16
	spec.BankLatency = bankLatency
	spec.DirLatency = 2
	spec.MaxNumConcurrentTrans = 512

	return spec
}

// DRAMSpec returns the spec of one DRAM controller (memory channel). The
// controller is element `index` of `numElements` controllers that addresses
// are interleaved across at `interleavingSize` bytes; the bank-address
// conversion strips that interleave before bank selection.
func DRAMSpec(
	interleavingSize uint64,
	numElements, index int,
	capacity uint64,
) simplebankedmemory.Spec {
	spec := simplebankedmemory.DefaultSpec()
	spec.Freq = 1 * timing.GHz
	spec.NumBanks = 16
	spec.BankPipelineWidth = 1
	spec.BankPipelineDepth = 5
	spec.StageLatency = 1
	spec.BankSelectorKind = "interleaved"
	spec.BankSelectorLog2InterleaveSize = 6
	spec.PostPipelineBufSize = 128
	spec.BankAddrConvKind = "interleaving"
	spec.BankAddrInterleavingSize = interleavingSize
	spec.BankAddrTotalNumOfElements = numElements
	spec.BankAddrCurrentElementIndex = index
	spec.Capacity = capacity

	return spec
}

// CPSpec returns the spec of a command processor that dispatches across
// numDies XCDs with the given per-die algorithm ("per-die" or
// "per-die-interleaved").
func CPSpec(freq timing.Freq, alg string, numDies int) cp.Spec {
	spec := cp.DefaultSpec()
	spec.Freq = freq
	// No fixed kernel launch/teardown overhead is modeled. The empty_kernel floor
	// (~8.66 us at 1 WG) is launch_sync_us -- a host round-trip (CPU launch + API +
	// device sync) that is NOT GPU time. Modeling it as a fixed GPU launch latency
	// would fake host cost as device cost; the calibration metric anchors the floor
	// out (error is measured relative to the smallest-grid point), so only the
	// per-WG dispatch slope matters and it is unaffected. Leave all three overheads
	// at zero unless/until a GPU-side launch measurement exists.
	spec.ConstantKernelLaunchOverhead = 0
	spec.SubsequentKernelLaunchOverhead = 0
	spec.ConstantKernelOverhead = 0

	// Each XCD dispatches 1 wavefront / 2 cycles. For a 256-thread
	// (4-wavefront) WG that is 8 cycles/WG per die, so 8 dies aggregate to ~1
	// WG/cycle = ~0.48 ns/WG at 2.1 GHz, matching the empty_kernel sweep slope.
	spec.Alg = alg
	spec.NumDies = numDies
	spec.WavefrontDispatchCycles = 2

	return spec
}

// L2TLBSpec returns the spec of the L2 TLB.
func L2TLBSpec(freq timing.Freq, log2PageSize uint64) tlb.Spec {
	spec := tlb.DefaultSpec()
	spec.Freq = freq
	spec.NumWays = 64
	spec.NumSets = 64
	spec.MSHRSize = 64
	spec.NumReqPerCycle = 1024
	spec.Log2PageSize = log2PageSize

	return spec
}

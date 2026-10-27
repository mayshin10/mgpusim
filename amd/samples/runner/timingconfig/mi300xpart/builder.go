// Package mi300xpart builds a whole MI300X package (8 XCDs, 4 IODs, 8 HBM
// stacks) split into compute partitions (SPX/DPX/QPX/CPX) and memory
// partitions (NPS1/NPS2/NPS4/NPS8).
//
// Topology. The 4 IODs form a 2x2 mesh (IOD0-IOD1, IOD2-IOD3, IOD0-IOD2,
// IOD1-IOD3). XCD x sits on IOD x/2. Each HBM stack contributes 2 memory
// channels, and channel c (a MALL slice in front of a DRAM controller) sits on
// IOD c/4. All XCD<->memory and XCD<->XCD traffic crosses the fabric
// (amd/timing/fabric), which charges per-link bandwidth and per-hop latency.
//
// Compute partitions. A package with P compute partitions exposes P GPUs to the
// driver, each owning 8/P consecutive XCDs, its own command processor, DMA and
// RDMA engines, and L2 TLB. Every XCD has its own 4 MB L2 (16 channels), as on
// the real chip. Within a partition, work-groups go round-robin across its
// XCDs ("per-die-interleaved"), as MI300X does in SPX mode.
//
// Memory partitions. With N NUMA domains, domain d owns channels
// [d*16/N, (d+1)*16/N) and partition p allocates from domain p*N/P. A GPU's
// addresses are interleaved at 128 B over its domain's channels. NPS1 spreads
// every GPU over all 8 stacks; NPS4 confines a GPU to the 2 stacks on its IOD;
// NPS8 confines an XCD to the stack beside it. MI300X supports SPX/NPS1,
// CPX/NPS1 and CPX/NPS4 (and DPX/QPX with NPS1); NPS2 and NPS8 are
// hypothetical here (NPS8 matches cpc-analytic's "each XCD only reaches its
// own stack").
//
// Peer access. As in the rest of MGPUSim, a GPU's L1 sends addresses outside its
// own range to its RDMA engine, which forwards them over the fabric to the
// owner's RDMA engine and L2. The owner's L2 is therefore the point of
// coherence for its memory. The real chip routes a remote access straight to
// the memory's IOD; the two coincide under NPS4/NPS8, where the owner's XCD and
// memory share an IOD.
package mi300xpart

import (
	"fmt"

	"github.com/sarchlab/akita/v5/mem"
	"github.com/sarchlab/akita/v5/mem/cache/writeback"
	"github.com/sarchlab/akita/v5/mem/cache/writethroughcache"
	"github.com/sarchlab/akita/v5/mem/simplebankedmemory"
	"github.com/sarchlab/akita/v5/mem/vm/mmu"
	"github.com/sarchlab/akita/v5/mem/vm/tlb"
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/noc/directconnection"
	"github.com/sarchlab/akita/v5/simulation"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/gpubuilder"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/mi300x"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/shaderarray"
	"github.com/sarchlab/mgpusim/v5/amd/timing/cp"
	"github.com/sarchlab/mgpusim/v5/amd/timing/cu"
	"github.com/sarchlab/mgpusim/v5/amd/timing/fabric"
	"github.com/sarchlab/mgpusim/v5/amd/timing/rdma"
)

// Package-level hardware constants.
const (
	NumXCD             = 8
	NumIOD             = 4
	NumMemChannel      = 16 // 8 HBM3 stacks x 2 controllers
	NumL2ChannelPerXCD = 16
	XCDPerIOD          = NumXCD / NumIOD
	MemChannelPerIOD   = NumMemChannel / NumIOD
)

// Port buffer sizes, mirroring package mi300x.
const (
	cpPortBufSize      = 4096
	dmaToCPBufSize     = 4096
	dmaToMemBufSize    = 64
	rdmaPortBufSize    = 128
	dramTopPortBufSize = 1024
	l2TLBPortBufSize   = 1024
	ctrlPortBufSize    = 1
	l2CachePortBufSize = 128
)

// Builder builds a partitioned MI300X package.
type Builder struct {
	simulation *simulation.Simulation

	numComputePartition int
	numMemoryPartition  int

	freq                           timing.Freq
	l2CacheSize                    uint64 // per XCD
	l2BankLatency                  int
	mallCacheSize                  uint64 // whole package
	mallBankLatency                int
	log2CacheLineSize              uint64
	log2PageSize                   uint64
	log2MemoryBankInterleavingSize uint64
	gpuMemSize                     uint64

	// Fabric link parameters (per direction).
	xcdLinkBandwidth float64 // bytes/s, XCD <-> its IOD
	xcdLinkLatency   timing.VTimeInPicoSec
	iodLinkBandwidth float64 // bytes/s, IOD <-> IOD
	iodLinkLatency   timing.VTimeInPicoSec

	globalStorage     *mem.Storage
	mmu               *mmu.Comp
	rdmaAddressMapper mem.AddressToPortMapper
	driverPort        messaging.RemotePort

	fabric    *fabric.Comp
	iodNodes  []int
	xcdNodes  []int
	drams     []*simplebankedmemory.Comp
	malls     []*writethroughcache.Comp
	mallToMem *directconnection.Comp
	memCtrl   *directconnection.Comp
}

// MakeBuilder creates a builder for an SPX/NPS1 package with MI300X defaults.
func MakeBuilder() Builder {
	return Builder{
		numComputePartition:            1,
		numMemoryPartition:             1,
		freq:                           2100 * timing.MHz,
		l2CacheSize:                    4 * mem.MB,
		l2BankLatency:                  124,
		mallCacheSize:                  256 * mem.MB,
		mallBankLatency:                371,
		log2CacheLineSize:              6,
		log2PageSize:                   21,
		log2MemoryBankInterleavingSize: 7,
		gpuMemSize:                     4 * mem.GB,
		// ASSUMPTION: IOD-IOD links match cpc-analytic (1.2 TB/s per direction,
		// 20 ns per hop). XCD-IOD bandwidth is not published per direction;
		// 2.1 TB/s keeps it above one XCD's share of HBM in every NPS mode.
		// The XCD-IOD hop carries no extra latency because the calibrated MALL
		// bank latency already covers the local XCD->MALL path.
		xcdLinkBandwidth: 2.1e12,
		xcdLinkLatency:   0,
		iodLinkBandwidth: 1.2e12,
		iodLinkLatency:   20 * 1000,
	}
}

// WithSimulation sets the simulation.
func (b Builder) WithSimulation(sim *simulation.Simulation) Builder {
	b.simulation = sim
	return b
}

// WithComputePartitions sets the number of compute partitions: 1 (SPX),
// 2 (DPX), 4 (QPX), or 8 (CPX).
func (b Builder) WithComputePartitions(n int) Builder {
	b.numComputePartition = n
	return b
}

// WithMemoryPartitions sets the number of NUMA domains: 1 (NPS1), 2, 4 (NPS4),
// or 8.
func (b Builder) WithMemoryPartitions(n int) Builder {
	b.numMemoryPartition = n
	return b
}

// WithGPUMemSize sets the address range size of each GPU (partition).
func (b Builder) WithGPUMemSize(size uint64) Builder {
	b.gpuMemSize = size
	return b
}

// WithLog2PageSize sets the log2 page size.
func (b Builder) WithLog2PageSize(n uint64) Builder {
	b.log2PageSize = n
	return b
}

// WithIODLink sets the bandwidth (bytes/s per direction) and per-hop latency
// of the IOD-IOD links.
func (b Builder) WithIODLink(bytesPerSec float64, latency timing.VTimeInPicoSec) Builder {
	b.iodLinkBandwidth = bytesPerSec
	b.iodLinkLatency = latency
	return b
}

// WithXCDLink sets the bandwidth (bytes/s per direction) and latency of the
// XCD-IOD links.
func (b Builder) WithXCDLink(bytesPerSec float64, latency timing.VTimeInPicoSec) Builder {
	b.xcdLinkBandwidth = bytesPerSec
	b.xcdLinkLatency = latency
	return b
}

// WithMMU sets the MMU that provides address translation.
func (b Builder) WithMMU(m *mmu.Comp) Builder {
	b.mmu = m
	return b
}

// WithGlobalStorage sets the global storage.
func (b Builder) WithGlobalStorage(s *mem.Storage) Builder {
	b.globalStorage = s
	return b
}

// WithRDMAAddressMapper sets the mapper from an address to the RDMA engine of
// the GPU that owns it.
func (b Builder) WithRDMAAddressMapper(m mem.AddressToPortMapper) Builder {
	b.rdmaAddressMapper = m
	return b
}

// WithDriverPort sets the driver port the command processors respond to.
func (b Builder) WithDriverPort(p messaging.RemotePort) Builder {
	b.driverPort = p
	return b
}

// Package is the result of a build.
type Package struct {
	// GPUs are the compute partitions, GPU[1]..GPU[P]. The RDMA ports are
	// already on the fabric; only the command processor and translation
	// ports need to be connected to the driver and the MMU.
	GPUs []*gpubuilder.GPU

	// CUsPerGPU is the number of compute units in each partition.
	CUsPerGPU int

	// Fabric is the package interconnect, for reporting link statistics.
	Fabric *fabric.Comp
}

// Build builds the package.
func (b Builder) Build() *Package {
	b.mustBeValid()

	b.buildFabric()
	b.buildMemoryChannels()

	pkg := &Package{
		Fabric:    b.fabric,
		CUsPerGPU: NumXCD / b.numComputePartition * mi300x.NumShaderArrayPerXCD * mi300x.NumCUPerShaderArray,
	}

	for p := 0; p < b.numComputePartition; p++ {
		pb := partitionBuilder{Builder: &b, index: p}
		pkg.GPUs = append(pkg.GPUs, pb.build())
	}

	return pkg
}

func (b *Builder) mustBeValid() {
	validCounts := map[int]bool{1: true, 2: true, 4: true, 8: true}
	if !validCounts[b.numComputePartition] {
		panic(fmt.Sprintf("invalid number of compute partitions %d", b.numComputePartition))
	}

	if !validCounts[b.numMemoryPartition] {
		panic(fmt.Sprintf("invalid number of memory partitions %d", b.numMemoryPartition))
	}

	// A compute partition must live inside one NUMA domain.
	if b.numMemoryPartition > b.numComputePartition {
		panic(fmt.Sprintf("NPS%d needs at least %d compute partitions",
			b.numMemoryPartition, b.numMemoryPartition))
	}
}

func (b *Builder) buildFabric() {
	spec := fabric.DefaultSpec()
	spec.Freq = b.freq
	b.fabric = fabric.MakeBuilder().
		WithRegistrar(b.simulation).
		WithSpec(spec).
		Build("Fabric")

	for i := 0; i < NumIOD; i++ {
		b.iodNodes = append(b.iodNodes, b.fabric.AddNode(fmt.Sprintf("IOD[%d]", i)))
	}

	for x := 0; x < NumXCD; x++ {
		b.xcdNodes = append(b.xcdNodes, b.fabric.AddNode(fmt.Sprintf("XCD[%d]", x)))
	}

	// 2x2 mesh; X links first so routes go X-then-Y.
	for _, l := range [][2]int{{0, 1}, {2, 3}, {0, 2}, {1, 3}} {
		b.fabric.AddLink(b.iodNodes[l[0]], b.iodNodes[l[1]],
			b.iodLinkBandwidth, b.iodLinkLatency)
	}

	for x := 0; x < NumXCD; x++ {
		b.fabric.AddLink(b.xcdNodes[x], b.iodNodes[x/XCDPerIOD],
			b.xcdLinkBandwidth, b.xcdLinkLatency)
	}
}

// channelsPerDomain is the number of memory channels in one NUMA domain.
func (b *Builder) channelsPerDomain() int {
	return NumMemChannel / b.numMemoryPartition
}

// buildMemoryChannels builds the 16 MALL slices and DRAM controllers. The MALL
// tops sit on the fabric at their IOD; each MALL reaches its own DRAM
// controller over a package-level direct connection.
func (b *Builder) buildMemoryChannels() {
	perDomain := b.channelsPerDomain()
	interleave := uint64(1) << b.log2MemoryBankInterleavingSize

	b.mallToMem = directconnection.MakeBuilder().
		WithRegistrar(b.simulation).
		WithSpec(directconnection.Spec{Freq: b.freq}).
		Build("MALLToDRAM")

	// The CPs never send to the MALLs or DRAM controllers (see package mi300x);
	// their control ports only need to be connected.
	b.memCtrl = directconnection.MakeBuilder().
		WithRegistrar(b.simulation).
		WithSpec(directconnection.Spec{Freq: b.freq}).
		Build("MemCtrlConn")

	for c := 0; c < NumMemChannel; c++ {
		dramSpec := mi300x.DRAMSpec(interleave, perDomain, c%perDomain,
			b.gpuMemSize/uint64(perDomain))
		dram := simplebankedmemory.MakeBuilder().
			WithRegistrar(b.simulation).
			WithSpec(dramSpec).
			WithResources(simplebankedmemory.Resources{Storage: b.globalStorage}).
			Build(fmt.Sprintf("DRAM[%d]", c))
		buildPort(b.simulation, dram, "Top", dramTopPortBufSize)
		buildPort(b.simulation, dram, "Control", ctrlPortBufSize)
		b.drams = append(b.drams, dram)

		mallSpec := mi300x.MALLSpec(b.freq, b.log2CacheLineSize,
			b.mallCacheSize/NumMemChannel, b.mallBankLatency)
		mall := writethroughcache.MakeBuilder().
			WithRegistrar(b.simulation).
			WithSpec(mallSpec).
			WithResources(writethroughcache.Resources{
				AddressMapper: &mem.SinglePortMapper{
					Port: dram.GetPortByName("Top").AsRemote(),
				},
			}).
			Build(fmt.Sprintf("MALL[%d]", c))
		buildPort(b.simulation, mall, "Top", l2CachePortBufSize)
		buildPort(b.simulation, mall, "Bottom", l2CachePortBufSize)
		buildPort(b.simulation, mall, "Control", l2CachePortBufSize)
		b.malls = append(b.malls, mall)

		b.fabric.PlugInAt(mall.GetPortByName("Top"), b.iodNodes[c/MemChannelPerIOD])
		b.mallToMem.PlugIn(mall.GetPortByName("Bottom"))
		b.mallToMem.PlugIn(dram.GetPortByName("Top"))
		b.memCtrl.PlugIn(mall.GetPortByName("Control"))
		b.memCtrl.PlugIn(dram.GetPortByName("Control"))
	}
}

func buildPort(
	sim *simulation.Simulation,
	comp messaging.Component,
	name string,
	bufSize int,
) messaging.Port {
	port := modeling.MakePortBuilder().
		WithRegistrar(sim).
		WithComponent(comp).
		WithSpec(modeling.PortSpec{BufSize: bufSize}).
		Build(name)
	comp.AssignPort(name, port)

	return port
}

// partitionBuilder builds one compute partition (one GPU as the driver sees it).
type partitionBuilder struct {
	*Builder

	index   int // 0-based; the GPU ID is index+1
	name    string
	gpuID   uint64
	xcds    []int
	domain  int
	addrLow uint64

	cp           *cp.Comp
	dmaEngine    *cp.DMAComp
	rdmaEngine   *rdma.Comp
	l2TLB        *tlb.Comp
	sas          []*shaderarray.ShaderArray
	l2Caches     []*writeback.Comp
	l1Mappers    []*mem.InterleavedAddressPortMapper
	l1TLBMapper  *mem.SinglePortMapper
	internalConn *directconnection.Comp
	l1ToL2Conn   *directconnection.Comp
}

func (p *partitionBuilder) build() *gpubuilder.GPU {
	p.gpuID = uint64(p.index + 1)
	p.name = fmt.Sprintf("GPU[%d]", p.gpuID)
	p.addrLow = p.gpuID * p.gpuMemSize
	p.domain = p.index * p.numMemoryPartition / p.numComputePartition

	xcdsPerPartition := NumXCD / p.numComputePartition
	for k := 0; k < xcdsPerPartition; k++ {
		p.xcds = append(p.xcds, p.index*xcdsPerPartition+k)
	}

	p.buildCP()
	p.buildL2TLB()

	for k, x := range p.xcds {
		p.buildXCD(k, x)
	}

	p.buildDMA()
	p.buildRDMAEngine()
	p.buildSAs()
	p.connect()

	gpu := &gpubuilder.GPU{
		Name:                 p.name,
		CommandProcessor:     p.cp,
		CommandProcessorPort: p.cp.GetPortByName("ToDriver"),
		RDMARequestPort:      p.rdmaEngine.GetPortByName("RDMARequestOutside"),
		RDMADataPort:         p.rdmaEngine.GetPortByName("RDMADataOutside"),
		TranslationPorts:     []messaging.Port{p.l2TLB.GetPortByName("Bottom")},
	}

	return gpu
}

// domainChannels returns the memory channels of this partition's NUMA domain.
func (p *partitionBuilder) domainChannels() []int {
	perDomain := p.channelsPerDomain()

	channels := make([]int, perDomain)
	for i := range channels {
		channels[i] = p.domain*perDomain + i
	}

	return channels
}

func (p *partitionBuilder) buildCP() {
	spec := mi300x.CPSpec(p.freq, "per-die-interleaved", NumXCD/p.numComputePartition)
	// Host copies (buildDMA) and peer accesses (buildRDMAEngine) go through
	// the first XCD's L2. With one XCD that L2 sees every access to the
	// partition's memory, so a driver flush only needs to invalidate the L1s.
	spec.FlushL1Only = len(p.xcds) == 1

	p.cp = cp.MakeBuilder().
		WithRegistrar(p.simulation).
		WithSpec(spec).
		WithVisTracer(p.simulation.GetVisTracer()).
		WithMonitor(p.simulation.GetMonitor()).
		WithDriver(p.driverPort).
		Build(p.name + ".CommandProcessor")

	for _, name := range []string{"ToDriver", "ToDMA", "ToCUs", "ToTLBs",
		"ToAddressTranslators", "ToCaches", "ToRDMA"} {
		buildPort(p.simulation, p.cp, name, cpPortBufSize)
	}
}

// buildDMA builds the engine for host copies. They enter through the first
// XCD's L2, like the RDMA traffic from peers, so a kernel never reads an L2
// line that a host copy (e.g. the next kernel's arguments) has made stale.
// Partitions with several XCDs keep one L2 per XCD and those L2s are not
// coherent with each other; the CPX configurations have one XCD per GPU.
func (p *partitionBuilder) buildDMA() {
	p.dmaEngine = cp.MakeDMAEngineBuilder().
		WithRegistrar(p.simulation).
		WithSpec(cp.DefaultDMASpec()).
		WithResources(cp.DMAResources{LocalDataSource: p.l1Mappers[0]}).
		Build(p.name + ".DMA")
	buildPort(p.simulation, p.dmaEngine, "ToCP", dmaToCPBufSize)
	buildPort(p.simulation, p.dmaEngine, "ToMem", dmaToMemBufSize)
}

func (p *partitionBuilder) buildL2TLB() {
	p.l2TLB = tlb.MakeBuilder().
		WithRegistrar(p.simulation).
		WithSpec(mi300x.L2TLBSpec(p.freq, p.log2PageSize)).
		WithResources(tlb.Resources{
			TranslationProviderMapper: &mem.SinglePortMapper{
				Port: p.mmu.GetPortByName("Top").AsRemote(),
			},
		}).
		Build(p.name + ".L2TLB")
	buildPort(p.simulation, p.l2TLB, "Top", l2TLBPortBufSize)
	buildPort(p.simulation, p.l2TLB, "Bottom", l2TLBPortBufSize)
	buildPort(p.simulation, p.l2TLB, "Control", ctrlPortBufSize)

	p.l1TLBMapper = &mem.SinglePortMapper{Port: p.l2TLB.GetPortByName("Top").AsRemote()}
}

// buildXCD builds the k-th XCD of the partition (package XCD x): its 16 L2
// channels and the L1 mapper that spreads its shader arrays' misses over them.
// L2 channel j holds addresses with (addr/128)%16 == j, so its misses all go to
// the domain channel (addr/128)%perDomain == j%perDomain.
func (p *partitionBuilder) buildXCD(k, x int) {
	channels := p.domainChannels()
	interleave := uint64(1) << p.log2MemoryBankInterleavingSize

	l1Mapper := mem.NewInterleavedAddressPortMapper(interleave)
	l1Mapper.LowAddress = p.addrLow
	l1Mapper.HighAddress = p.addrLow + p.gpuMemSize
	l1Mapper.UseAddressSpaceLimitation = true

	spec := mi300x.L2CacheSpec(p.freq, p.log2CacheLineSize,
		p.l2CacheSize/NumL2ChannelPerXCD, p.l2BankLatency)

	for j := 0; j < NumL2ChannelPerXCD; j++ {
		mall := p.malls[channels[j%len(channels)]]
		l2 := writeback.MakeBuilder().
			WithRegistrar(p.simulation).
			WithSpec(spec).
			WithResources(writeback.Resources{
				AddressToPortMapper: &mem.SinglePortMapper{
					Port: mall.GetPortByName("Top").AsRemote(),
				},
			}).
			Build(fmt.Sprintf("%s.XCD[%d].L2Cache[%d]", p.name, k, j))
		buildPort(p.simulation, l2, "Top", l2CachePortBufSize)
		buildPort(p.simulation, l2, "Bottom", l2CachePortBufSize)
		buildPort(p.simulation, l2, "Control", l2CachePortBufSize)
		p.fabric.PlugInAt(l2.GetPortByName("Bottom"), p.xcdNodes[x])

		p.l2Caches = append(p.l2Caches, l2)
		l1Mapper.LowModules = append(l1Mapper.LowModules, l2.GetPortByName("Top").AsRemote())
	}

	p.l1Mappers = append(p.l1Mappers, l1Mapper)
}

// buildRDMAEngine builds the RDMA engine on the partition's first XCD. Remote
// requests it receives are served by that XCD's L2.
func (p *partitionBuilder) buildRDMAEngine() {
	spec := rdma.DefaultSpec()
	spec.Freq = 1 * timing.GHz

	p.rdmaEngine = rdma.MakeBuilder().
		WithRegistrar(p.simulation).
		WithSpec(spec).
		WithResources(rdma.Resources{
			LocalModules:           p.l1Mappers[0],
			RemoteRDMAAddressTable: p.rdmaAddressMapper,
		}).
		Build(p.name + ".RDMA")

	for _, name := range []string{"RDMARequestInside", "RDMARequestOutside",
		"RDMADataInside", "RDMADataOutside", "Ctrl"} {
		buildPort(p.simulation, p.rdmaEngine, name, rdmaPortBufSize)
	}

	for _, m := range p.l1Mappers {
		m.ModuleForOtherAddresses = p.rdmaEngine.GetPortByName("RDMARequestInside").AsRemote()
	}

	node := p.xcdNodes[p.xcds[0]]
	p.fabric.PlugInAt(p.rdmaEngine.GetPortByName("RDMARequestOutside"), node)
	p.fabric.PlugInAt(p.rdmaEngine.GetPortByName("RDMADataOutside"), node)
}

// buildSAs builds the shader arrays after the L1 mappers are complete, since
// the caches snapshot the mapper contents at build time.
func (p *partitionBuilder) buildSAs() {
	for k := range p.xcds {
		saBuilder := mi300x.ShaderArrayBuilder(p.simulation, p.freq, p.gpuID,
			mi300x.NumCUPerShaderArray, p.log2CacheLineSize, p.log2PageSize).
			WithL1AddressMapper(p.l1Mappers[k]).
			WithL1TLBAddressMapper(p.l1TLBMapper)

		for i := 0; i < mi300x.NumShaderArrayPerXCD; i++ {
			name := fmt.Sprintf("%s.XCD[%d].SA[%d]", p.name, k, i)
			p.sas = append(p.sas, saBuilder.Build(name))
		}
	}
}

func (p *partitionBuilder) connect() {
	p.internalConn = directconnection.MakeBuilder().
		WithRegistrar(p.simulation).
		WithSpec(directconnection.Spec{Freq: p.freq}).
		Build(p.name + ".InternalConn")

	for _, name := range []string{"ToDMA", "ToCaches", "ToCUs", "ToTLBs",
		"ToAddressTranslators", "ToRDMA"} {
		p.internalConn.PlugIn(p.cp.GetPortByName(name))
	}

	rdmaCtrl := p.rdmaEngine.GetPortByName("Ctrl")
	p.cp.State.RDMA = rdmaCtrl.AsRemote()
	p.internalConn.PlugIn(rdmaCtrl)

	dmaToCP := p.dmaEngine.GetPortByName("ToCP")
	p.cp.State.DMAEngine = dmaToCP.AsRemote()
	p.internalConn.PlugIn(dmaToCP)

	p.connectCPWithSAs()
	p.connectCPWithL2s()
	p.connectL1ToL2()
	p.connectL1TLBToL2TLB()
}

func (p *partitionBuilder) addCtrl(list *[]messaging.RemotePort, comp messaging.Component, port string) {
	ctrl := comp.GetPortByName(port)
	*list = append(*list, ctrl.AsRemote())
	p.internalConn.PlugIn(ctrl)
}

// connectCPWithSAs registers the CUs (in XCD order, as the per-die dispatcher
// expects) and the control ports of the L1 caches, TLBs, ATs, and ROBs.
func (p *partitionBuilder) connectCPWithSAs() {
	st := &p.cp.State

	for _, sa := range p.sas {
		for i, cuComp := range sa.CUs {
			cp.RegisterCU(p.cp, cu.DispatcherView{CU: cuComp})
			p.internalConn.PlugIn(cuComp.GetPortByName(cu.DispatchPortName))
			p.internalConn.PlugIn(cuComp.GetPortByName(cu.CtrlPortName))

			p.addCtrl(&st.AddressTranslators, sa.L1VATs[i], "Control")
			p.addCtrl(&st.ROBs, sa.L1VROBs[i], "Control")
			p.addCtrl(&st.TLBs, sa.L1VTLBs[i], "Control")
			p.addCtrl(&st.L1VCaches, sa.L1VCaches[i], "Control")
		}

		p.addCtrl(&st.AddressTranslators, sa.L1SAT, "Control")
		p.addCtrl(&st.ROBs, sa.L1SROB, "Control")
		p.addCtrl(&st.AddressTranslators, sa.L1IAT, "Control")
		p.addCtrl(&st.ROBs, sa.L1IROB, "Control")
		p.addCtrl(&st.TLBs, sa.L1STLB, "Control")
		p.addCtrl(&st.TLBs, sa.L1ITLB, "Control")
		p.addCtrl(&st.L1SCaches, sa.L1SCache, "Control")
		p.addCtrl(&st.L1ICaches, sa.L1ICache, "Control")
	}

	p.addCtrl(&st.TLBs, p.l2TLB, "Control")
}

func (p *partitionBuilder) connectCPWithL2s() {
	for _, l2 := range p.l2Caches {
		p.addCtrl(&p.cp.State.L2Caches, l2, "Control")
	}
}

// connectL1ToL2 joins the partition's L1s, L2 tops, and the RDMA inside ports.
// The connection is zero-latency and shared by the partition's XCDs, but each
// XCD's L1 mapper only targets that XCD's own L2 channels.
func (p *partitionBuilder) connectL1ToL2() {
	p.l1ToL2Conn = directconnection.MakeBuilder().
		WithRegistrar(p.simulation).
		WithSpec(directconnection.Spec{Freq: p.freq}).
		Build(p.name + ".L1ToL2")

	p.l1ToL2Conn.PlugIn(p.rdmaEngine.GetPortByName("RDMARequestInside"))
	p.l1ToL2Conn.PlugIn(p.rdmaEngine.GetPortByName("RDMADataInside"))
	p.l1ToL2Conn.PlugIn(p.dmaEngine.GetPortByName("ToMem"))

	for _, l2 := range p.l2Caches {
		p.l1ToL2Conn.PlugIn(l2.GetPortByName("Top"))
	}

	for _, sa := range p.sas {
		for i := range sa.CUs {
			p.l1ToL2Conn.PlugIn(sa.L1VCaches[i].GetPortByName("Bottom"))
		}

		p.l1ToL2Conn.PlugIn(sa.L1SCache.GetPortByName("Bottom"))
		p.l1ToL2Conn.PlugIn(sa.L1IAT.GetPortByName("Bottom"))
	}
}

func (p *partitionBuilder) connectL1TLBToL2TLB() {
	conn := directconnection.MakeBuilder().
		WithRegistrar(p.simulation).
		WithSpec(directconnection.Spec{Freq: p.freq}).
		Build(p.name + ".L1TLBToL2TLB")

	conn.PlugIn(p.l2TLB.GetPortByName("Top"))

	for _, sa := range p.sas {
		for i := range sa.CUs {
			conn.PlugIn(sa.L1VTLBs[i].GetPortByName("Bottom"))
		}

		conn.PlugIn(sa.L1STLB.GetPortByName("Bottom"))
		conn.PlugIn(sa.L1ITLB.GetPortByName("Bottom"))
	}
}

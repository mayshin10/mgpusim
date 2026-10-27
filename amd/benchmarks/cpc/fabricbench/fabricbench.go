// Package fabricbench measures local-HBM and peer bandwidth on a partitioned
// GPU package. Each transfer is a streaming copy kernel that runs on one GPU
// (partition), reads a buffer that lives on a second GPU, and writes a buffer
// that lives on a third. All transfers run concurrently.
package fabricbench

import (
	_ "embed"
	"fmt"
	"log"

	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/driver"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

const blockSize = 256

// KernelArgs defines the arguments of copy_f4.
type KernelArgs struct {
	Src    driver.Ptr
	Dst    driver.Ptr
	N4     int32
	Stride int32
}

// Transfer is one copy kernel: it runs on Exec and copies from Src to Dst.
type Transfer struct {
	Exec, Src, Dst int
}

// Result reports one transfer.
type Result struct {
	Transfer
	Bytes   uint64
	Seconds float64
}

// Benchmark runs a set of concurrent transfers.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	hsaco   *insts.KernelCodeObject

	// Transfers run concurrently.
	Transfers []Transfer
	// NumBytes copied by each transfer (rounded down to 16 B).
	NumBytes uint64
	// NumWG is the number of work-groups of each copy kernel.
	NumWG int
	// Now returns the simulated time; used to time the transfers.
	Now func() timing.VTimeInPicoSec

	Results []Result

	srcs, dsts []driver.Ptr
	data       [][]uint32
}

//go:embed kernels_gfx942.hsaco
var hsacoBytes []byte

// NewBenchmark creates a new benchmark.
func NewBenchmark(d *driver.Driver) *Benchmark {
	b := new(Benchmark)
	b.driver = d
	b.context = d.Init()
	b.NumWG = 304

	return b
}

// SelectGPU is ignored; the transfers name their GPUs.
func (b *Benchmark) SelectGPU(_ []int) {}

// SetUnifiedMemory is not supported.
func (b *Benchmark) SetUnifiedMemory() {
	panic("fabricbench does not support unified memory")
}

// Run runs the transfers.
func (b *Benchmark) Run() {
	b.hsaco = insts.LoadKernelCodeObjectFromBytes(hsacoBytes, "copy_f4")
	if b.hsaco == nil {
		log.Panic("failed to load copy_f4")
	}

	queues := b.setup()
	b.launch(queues)
}

// setup allocates and fills the buffers and creates one queue per transfer.
func (b *Benchmark) setup() []*driver.CommandQueue {
	n4 := int(b.NumBytes / 16)
	queues := make([]*driver.CommandQueue, len(b.Transfers))

	for i, t := range b.Transfers {
		words := make([]uint32, n4*4)
		for j := range words {
			words[j] = uint32(i<<24 + j)
		}
		b.data = append(b.data, words)

		b.driver.SelectGPU(b.context, t.Src)
		src := b.driver.AllocateMemory(b.context, uint64(n4*16))
		b.driver.MemCopyH2D(b.context, src, words)
		b.srcs = append(b.srcs, src)

		b.driver.SelectGPU(b.context, t.Dst)
		b.dsts = append(b.dsts, b.driver.AllocateMemory(b.context, uint64(n4*16)))

		b.driver.SelectGPU(b.context, t.Exec)
		queues[i] = b.driver.CreateCommandQueue(b.context)
	}

	return queues
}

// launch starts all copies at once and times each one until its queue drains.
func (b *Benchmark) launch(queues []*driver.CommandQueue) {
	n4 := int(b.NumBytes / 16)
	start := b.Now()

	for i := range b.Transfers {
		args := KernelArgs{
			Src:    b.srcs[i],
			Dst:    b.dsts[i],
			N4:     int32(n4),
			Stride: int32(b.NumWG * blockSize),
		}
		b.driver.EnqueueLaunchKernel(queues[i], b.hsaco,
			[3]uint32{uint32(b.NumWG * blockSize), 1, 1},
			[3]uint16{blockSize, 1, 1}, &args)
	}

	// The queues finish in any order; time each one when it drains.
	done := make(chan int, len(queues))
	for i, q := range queues {
		go func(i int, q *driver.CommandQueue) {
			b.driver.DrainCommandQueue(q)
			done <- i
		}(i, q)
	}

	b.Results = make([]Result, len(b.Transfers))
	for range queues {
		i := <-done
		b.Results[i] = Result{
			Transfer: b.Transfers[i],
			Bytes:    uint64(n4 * 16),
			Seconds:  float64(b.Now()-start) * 1e-12,
		}
	}

	for _, r := range b.Results {
		fmt.Printf("transfer exec=%d src=%d dst=%d bytes=%d time_s=%.6e copy_GBps=%.1f\n",
			r.Exec, r.Src, r.Dst, r.Bytes, r.Seconds, float64(r.Bytes)/r.Seconds/1e9)
	}
}

// Verify checks every destination buffer.
func (b *Benchmark) Verify() {
	for i := range b.Transfers {
		got := make([]uint32, len(b.data[i]))
		b.driver.MemCopyD2H(b.context, got, b.dsts[i])

		for j := range got {
			if got[j] != b.data[i][j] {
				log.Fatalf("transfer %d: word %d is %#x, want %#x", i, j, got[j], b.data[i][j])
			}
		}
	}

	log.Printf("Passed!\n")
}

// Command cpc_fabricbench measures local and peer bandwidth on the partitioned
// MI300X model, e.g.
//
//	cpc_fabricbench -timing -arch cdna3 -gpu mi300x-part \
//	  -compute-partition cpx -memory-partition nps4 -transfers 1:1:1,2:2:2
package main

import (
	"flag"
	"strconv"
	"strings"

	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/cpc/fabricbench"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner"
)

var transfersFlag = flag.String("transfers", "1:1:1",
	"Comma-separated exec:src:dst GPU triples; the copies run concurrently.")
var bytesFlag = flag.Uint64("bytes", 4<<20, "Bytes copied by each transfer.")
var numWGFlag = flag.Int("wg", 304, "Work-groups per copy kernel.")

func main() {
	flag.Parse()

	r := new(runner.Runner).Init()

	b := fabricbench.NewBenchmark(r.Driver())
	b.NumBytes = *bytesFlag
	b.NumWG = *numWGFlag
	b.Now = r.Engine().CurrentTime
	b.Transfers = parseTransfers(*transfersFlag)

	r.AddBenchmarkWithoutSettingGPUsToUse(b)
	r.Run()
}

func parseTransfers(s string) []fabricbench.Transfer {
	var ts []fabricbench.Transfer

	for _, tok := range strings.Split(s, ",") {
		f := strings.Split(tok, ":")
		if len(f) != 3 {
			panic("transfer must be exec:src:dst, got " + tok)
		}

		ids := make([]int, 3)
		for i := range f {
			id, err := strconv.Atoi(f[i])
			if err != nil {
				panic(err)
			}
			ids[i] = id
		}

		ts = append(ts, fabricbench.Transfer{Exec: ids[0], Src: ids[1], Dst: ids[2]})
	}

	return ts
}

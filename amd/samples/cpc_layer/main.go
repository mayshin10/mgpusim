// Command cpc_layer runs one transformer layer of cpc-analytic's TP8 or
// TP4xKVP2 configuration on the 8 chiplets of a CPX-partitioned MI300X, e.g.
//
//	cpc_layer -timing -arch cdna3 -gpu mi300x-part -compute-partition cpx \
//	  -memory-partition nps8 -magic-memory-copy -config tp4kvp2 -phase prefill -S 2048
package main

import (
	"flag"
	"log"

	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/cpc/layer"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner"
)

var configFlag = flag.String("config", "tp8", "Parallel configuration: tp8 or tp4kvp2.")
var phaseFlag = flag.String("phase", "prefill", "prefill or decode.")
var sFlag = flag.Int("S", 2048, "Prefill: prompt tokens. Decode: cached context per request.")
var batchFlag = flag.Int("batch", 1, "Decode batch (requests).")
var hFlag = flag.Int("H", 2048, "Hidden size (must equal NQ*128).")
var nqFlag = flag.Int("NQ", 16, "Query heads.")
var nkvFlag = flag.Int("NKV", 4, "KV heads.")
var fFlag = flag.Int("F", 4096, "SwiGLU FFN width.")
var partFlag = flag.String("part", "all",
	"Stages to run: all, head (configuration-specific, up to the O-proj input), or tail (O-proj onward).")
var splitsFlag = flag.Int("decode-splits", 0,
	"Decode attention key splits per chiplet (multiple of 4; 0 = automatic).")

func main() {
	flag.Parse()

	r := new(runner.Runner).Init()

	b := layer.NewBenchmark(r.Driver())
	b.Now = r.Engine().CurrentTime
	b.Cfg = layer.Config{
		Model: layer.Model{H: *hFlag, NQ: *nqFlag, NKV: *nkvFlag, F: *fFlag},
		Phase: *phaseFlag, S: *sFlag, Batch: *batchFlag, DecodeSplits: *splitsFlag, Part: *partFlag,
	}

	switch *configFlag {
	case "tp8":
		b.Cfg.TP, b.Cfg.KVP = 8, 1
	case "tp4kvp2":
		b.Cfg.TP, b.Cfg.KVP = 4, 2
	default:
		log.Fatalf("unknown config %q", *configFlag)
	}

	r.AddBenchmarkWithoutSettingGPUsToUse(b)
	r.Run()
}

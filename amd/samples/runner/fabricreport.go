package runner

import (
	"encoding/csv"
	"fmt"
	"os"

	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/timing/fabric"
)

// writeFabricReport writes the traffic of every used fabric link as CSV. The
// utilization is the link's busy time over the whole simulated time.
func writeFabricReport(path string, f *fabric.Comp, now timing.VTimeInPicoSec) {
	file, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	w := csv.NewWriter(file)
	defer w.Flush()

	_ = w.Write([]string{"from", "to", "bytes", "msgs", "busy_s", "utilization"})

	for _, s := range f.Stats() {
		util := 0.0
		if now > 0 {
			util = float64(s.BusyPicoSec) / float64(now)
		}

		_ = w.Write([]string{
			s.From, s.To,
			fmt.Sprint(s.Bytes), fmt.Sprint(s.Msgs),
			fmt.Sprintf("%.6e", float64(s.BusyPicoSec)*1e-12),
			fmt.Sprintf("%.4f", util),
		})
	}
}

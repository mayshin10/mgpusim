package layer

import (
	"testing"

	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

// TestKernelsDecode checks that every instruction of every kernel decodes with
// the CDNA3 decoder. An undecodable instruction panics in emulation but can
// stall a timing simulation silently.
func TestKernelsDecode(t *testing.T) {
	d := insts.NewDisassembler()
	d.IsCDNA3 = true

	for _, name := range kernelNames {
		co := insts.LoadKernelCodeObjectFromBytes(hsacoBytes, name)
		buf := co.Data

		for pc := 0; pc < len(buf); {
			inst, err := d.Decode(buf[pc:])
			if err != nil {
				t.Errorf("%s+%#x: %v", name, pc, err)
				break
			}

			pc += inst.ByteSize
		}
	}
}

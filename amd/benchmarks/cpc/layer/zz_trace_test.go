package layer

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/sarchlab/akita/v5/hooking"
	"github.com/sarchlab/mgpusim/v5/amd/emu"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

// traceHook logs every instruction of wave 0 of work-group (0,0,0): the
// instruction, EXEC, and the destination register for lanes 0..63.
type traceHook struct{ out *os.File }

func (h *traceHook) Func(ctx hooking.HookCtx) {
	wf, ok := ctx.Item.(*emu.Wavefront)
	if !ok || wf.WG.IDX != 0 || wf.WG.IDY != 0 || wf.FirstWiFlatID != 0 {
		return
	}
	inst := ctx.Detail.(*insts.Inst)
	line := fmt.Sprintf("%-60s exec=%016x vcc=%016x scc=%d", insts.NewInstPrinter(nil).Print(inst), wf.EXEC(), wf.VCC(), wf.SCC())
	if inst.Dst != nil && inst.Dst.OperandType == insts.RegOperand {
		if inst.Dst.Register.IsVReg() {
			vals := make([]string, 0, 64)
			for l := 0; l < 64; l++ {
				vals = append(vals, fmt.Sprintf("%x", wf.ReadOperand(inst.Dst, l)))
			}
			line += " dst=" + strings.Join(vals, ",")
		} else {
			line += fmt.Sprintf(" dst=%x", wf.ReadOperand(inst.Dst, 0))
		}
	}
	fmt.Fprintln(h.out, line)
}

func TestTraceCausal(t *testing.T) {
	path := os.Getenv("CPC_TRACE")
	if path == "" {
		t.Skip("set CPC_TRACE")
	}
	f, _ := os.Create(path)
	defer f.Close()
	traceTarget = &traceHook{out: f}
	testEmuPrefillAttention(t, true)
}

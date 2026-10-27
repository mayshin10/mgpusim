package cdna3

import (
	"encoding/binary"
	"log"
	"math"

	"github.com/sarchlab/mgpusim/v5/amd/emu"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

// runVMFMAF32x16x16x16BF16 implements v_mfma_f32_16x16x16_bf16:
// D[16x16] = A[16x16] * B[16x16] + C[16x16], with BF16 A/B and FP32 C/D.
//
// Lane layout (verified on an MI300A with the same instruction):
//
//	A[m][k]: lane l, element j holds m = l%16, k = 4*(l/16)+j  (2 VGPRs)
//	B[k][n]: lane l, element j holds k = 4*(l/16)+j, n = l%16  (2 VGPRs)
//	C/D[m][n]: lane l, element j holds m = 4*(l/16)+j, n = l%16 (4 VGPRs)
//
// MFMA is a cross-lane operation and ignores EXEC; all 64 lanes take part.
// Only the plain VGPR form is supported: the VOP3P-MAI fields that the
// generic VOP3a decoder files under Clamp (acc_cd), Abs (cbsz), OpSel (abid),
// Omod (AGPR A/B) and Neg (blgp) must all be zero.
func (u *ALU) runVMFMAF32x16x16x16BF16(state emu.InstEmuState) {
	inst := state.Inst()
	mustBePlainMFMA(inst)

	var a, b, c [16][16]float32

	for l := 0; l < 64; l++ {
		aBytes := state.ReadOperandBytes(inst.Src0, l, 8)
		bBytes := state.ReadOperandBytes(inst.Src1, l, 8)

		for j := 0; j < 4; j++ {
			k := 4*(l/16) + j
			a[l%16][k] = bf16ToF32(binary.LittleEndian.Uint16(aBytes[2*j:]))
			b[k][l%16] = bf16ToF32(binary.LittleEndian.Uint16(bBytes[2*j:]))
			c[k][l%16] = readMFMAAcc(state, inst.Src2, l, j)
		}
	}

	for l := 0; l < 64; l++ {
		out := make([]byte, 16)

		for j := 0; j < 4; j++ {
			m, n := 4*(l/16)+j, l%16
			acc := c[m][n]

			for k := 0; k < 16; k++ {
				acc += a[m][k] * b[k][n]
			}

			binary.LittleEndian.PutUint32(out[4*j:], math.Float32bits(acc))
		}

		state.WriteOperandBytes(inst.Dst, l, out)
	}
}

func mustBePlainMFMA(inst *insts.Inst) {
	if inst.Clamp || inst.Abs != 0 || inst.OpSel != 0 || inst.Omod != 0 || inst.Neg != 0 {
		log.Panicf("%s: AGPR operands and cbsz/abid/blgp are not supported",
			inst.InstName)
	}
}

// readMFMAAcc returns accumulator element j of lane l. An inline constant
// applies to every element.
func readMFMAAcc(state emu.InstEmuState, o *insts.Operand, l, j int) float32 {
	if o.OperandType != insts.RegOperand {
		return math.Float32frombits(binary.LittleEndian.Uint32(state.ReadOperandBytes(o, l, 4)))
	}

	accBytes := state.ReadOperandBytes(o, l, 16)

	return math.Float32frombits(binary.LittleEndian.Uint32(accBytes[4*j:]))
}

func bf16ToF32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

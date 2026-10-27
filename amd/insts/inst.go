package insts

// ExeUnit defines which execution unit should execute the instruction
type ExeUnit int

// Defines all possible execution units
const (
	ExeUnitVALU ExeUnit = iota
	ExeUnitScalar
	ExeUnitVMem
	ExeUnitBranch
	ExeUnitLDS
	ExeUnitGDS
	ExeUnitSpecial
)

// A InstType represents an instruction type. For example s_barrier instruction
// is a instruction type
type InstType struct {
	InstName  string
	Opcode    Opcode
	Format    *Format
	ID        int
	ExeUnit   ExeUnit
	DSTWidth  int
	SRC0Width int
	SRC1Width int
	SRC2Width int
	SDSTWidth int
}

// An Inst is a GCN3 instruction
type Inst struct {
	*Format
	*InstType
	ByteSize int
	PC       uint64

	Src0 *Operand
	Src1 *Operand
	Src2 *Operand
	Dst  *Operand
	SDst *Operand // For VOP3b

	Addr   *Operand
	Data   *Operand
	Data1  *Operand
	Base   *Operand
	Offset *Operand
	SImm16 *Operand
	SAddr  *Operand // FLAT scalar address (0x7F = OFF for global addressing)

	Abs                 int
	Omod                int
	Neg                 int
	OpSel               int // VOP3P: bits 11-14 for packed instructions
	OpSelHi             int // VOP3P: bits 59-60 for packed instructions
	Offset0             uint32
	Offset1             uint32
	SystemLevelCoherent bool
	GlobalLevelCoherent bool
	TextureFailEnable   bool
	Imm                 bool
	Clamp               bool
	GDS                 bool
	VMCNT               int
	LKGMCNT             int

	//Fields for SDWA extensions
	IsSdwa    bool
	DstSel    SDWASelect
	DstUnused SDWAUnused
	Src0Sel   SDWASelect
	Src0Sext  bool
	Src0Neg   bool
	Src0Abs   bool
	Src1Sel   SDWASelect
	Src1Sext  bool
	Src1Neg   bool
	Src1Abs   bool
	Src2Neg   bool
	Src2Abs   bool
}

// NewInst creates a zero-filled instruction
func NewInst() *Inst {
	i := new(Inst)
	i.Format = new(Format)
	i.InstType = new(InstType)
	return i
}


// FlatStoreBytes returns the bytes that a sub-dword FLAT/GLOBAL store
// (opcodes 24-27) writes from its 32-bit data register: store_byte and
// store_byte_d16_hi write bits [7:0] and [23:16]; store_short and
// store_short_d16_hi write bits [15:0] and [31:16]. Other opcodes return nil.
func FlatStoreBytes(opcode Opcode, data uint32) []byte {
	switch opcode {
	case 24:
		return []byte{byte(data)}
	case 25:
		return []byte{byte(data >> 16)}
	case 26:
		return []byte{byte(data), byte(data >> 8)}
	case 27:
		return []byte{byte(data >> 16), byte(data >> 24)}
	}

	return nil
}

// FlatLoadExtend returns the 32-bit register value of a sub-dword FLAT/GLOBAL
// load (opcodes 16-19: ubyte, sbyte, ushort, sshort) from the little-endian
// bytes at its address, or false for other opcodes.
func FlatLoadExtend(opcode Opcode, b []byte) (uint32, bool) {
	switch opcode {
	case 16:
		return uint32(b[0]), true
	case 17:
		return uint32(int32(int8(b[0]))), true
	case 18:
		return uint32(b[0]) | uint32(b[1])<<8, true
	case 19:
		return uint32(int32(int16(uint16(b[0]) | uint16(b[1])<<8))), true
	}

	return 0, false
}

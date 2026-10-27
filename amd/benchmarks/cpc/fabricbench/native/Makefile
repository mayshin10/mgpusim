# Compiles the fabricbench kernels with the local ROCm installation.
ROCM ?= /opt/rocm
GPU_TARGET = gfx942
SRC = fabricbench.cpp
GPU_OBJ = fabricbench-hip-amdgcn-amd-amdhsa-$(GPU_TARGET).o
HSACO = ../kernels_gfx942.hsaco

all: $(HSACO)

$(HSACO): $(SRC)
	$(ROCM)/bin/hipcc -O3 --save-temps -c --offload-arch=$(GPU_TARGET) $<
	cp $(GPU_OBJ) $(HSACO)
	$(ROCM)/llvm/bin/llvm-objdump --disassemble --mcpu=$(GPU_TARGET) $(HSACO) > fabricbench.disasm

clean:
	rm -f *.ll *.bc *.s *.o *.cui *.fatbin *.hipfb *.hipi *.out *.resolution.txt

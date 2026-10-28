# Compiles the cpc layer kernels with the local ROCm installation.
# -amdgpu-mfma-vgpr-form keeps MFMA accumulators in VGPRs (no AGPRs), which
# MGPUSim's CDNA3 model supports; launch bounds of (256, 1) avoid spills.
ROCM ?= /opt/rocm
GPU_TARGET = gfx942
HSACO = ../kernels_gfx942.hsaco

all: $(HSACO)

$(HSACO): cpc_kernels.cpp
	../../../../../../kernels/build_hsaco.sh $< $@ -mllvm -amdgpu-mfma-vgpr-form
	$(ROCM)/llvm/bin/llvm-objdump --disassemble --mcpu=$(GPU_TARGET) $@ > cpc_kernels.disasm

/*
 * Streaming copy used to measure local-HBM and peer (cross-XCD) bandwidth of
 * the partitioned MI300X model. Grid-stride, 16 B per thread per iteration.
 *
 * Block and grid geometry come in as explicit arguments (BLOCK_SIZE constant,
 * `stride` = total threads) so the compiler emits no hidden kernel arguments.
 */
#include "hip/hip_runtime.h"

#define BLOCK_SIZE 256

extern "C" __global__ void copy_f4(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    int n4,
    int stride)
{
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n4; i += stride) {
        dst[i] = src[i];
    }
}

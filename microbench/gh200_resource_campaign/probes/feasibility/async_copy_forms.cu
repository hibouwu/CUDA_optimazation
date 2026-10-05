// Compile-only S12 feasibility check. This is not a performance or correctness run.
#include <cuda_runtime.h>

#define COPY_FORM(NAME, CACHE, BYTES)                                           \
  extern "C" __global__ void NAME(const unsigned* input, unsigned* output) {    \
    __shared__ __align__(16) unsigned tile[128 * 4];                            \
    const unsigned index = threadIdx.x * 4;                                   \
    const unsigned address =                                                  \
        static_cast<unsigned>(__cvta_generic_to_shared(tile + index));         \
    asm volatile("cp.async." CACHE ".shared.global [%0], [%1], " #BYTES ";"  \
                 :: "r"(address), "l"(input + index) : "memory");             \
    asm volatile("cp.async.commit_group;" ::: "memory");                      \
    asm volatile("cp.async.wait_group 3;" ::: "memory");                      \
    asm volatile("cp.async.wait_group 2;" ::: "memory");                      \
    asm volatile("cp.async.wait_group 1;" ::: "memory");                      \
    asm volatile("cp.async.wait_group 0;" ::: "memory");                      \
    __syncthreads();                                                           \
    output[threadIdx.x] = tile[index];                                        \
  }

COPY_FORM(cp_ca4, "ca", 4)
COPY_FORM(cp_ca8, "ca", 8)
COPY_FORM(cp_ca16, "ca", 16)
COPY_FORM(cp_cg16, "cg", 16)

int main() { return 0; }

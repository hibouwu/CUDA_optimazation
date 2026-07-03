#include <cuda_runtime.h>
#include <stdio.h>

__global__ void simple_kernel(int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    for (int i = 0; i < 100; i++) {
      asm volatile("nop;");
    }
  }
}

int main() {
  simple_kernel<<<1, 32>>>(32);
  cudaDeviceSynchronize();
  return 0;
}

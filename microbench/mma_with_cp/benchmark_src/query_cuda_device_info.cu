
#include <cuda_runtime.h>
#include <cstdio>

int main() {
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  std::printf("%s|%d.%d|%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  return 0;
}

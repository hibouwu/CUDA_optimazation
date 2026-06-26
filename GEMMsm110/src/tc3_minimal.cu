#include "tc3_gemm_kernel.cuh"
#include "gemm_common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
  CHECK_CUDA(cudaFree(0));

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "GPU: " << prop.name << '\n';
  std::cout << "compute capability: " << prop.major << "." << prop.minor
            << '\n';

  if (prop.major != 11) {
    std::cout << "Not SM110-class device. Skip tcgen05 probe.\n";
    return EXIT_SUCCESS;
  }

  if (!tc3_sm110_tcgen05_available()) {
    std::cout << "This binary was not built with sm110a TCGen05 enabled. "
                 "Skip tcgen05 probe.\n";
    return EXIT_SUCCESS;
  }

  float* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_c, sizeof(float)));
  CHECK_CUDA(cudaMemset(d_c, 0, sizeof(float)));

  hgemm_tc3_sm110_tcgen05_tmem_probe<<<1, Tc3Sm110Shape::kThreads>>>(d_c, 1);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  float value = 0.0f;
  CHECK_CUDA(cudaMemcpy(&value, d_c, sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_c));

  const bool matched = std::abs(value - 1.0f) < 1e-6f;
  std::cout << "tc3 probe value = " << value << '\n';
  std::cout << "matched = " << matched << '\n';
  return matched ? EXIT_SUCCESS : EXIT_FAILURE;
}

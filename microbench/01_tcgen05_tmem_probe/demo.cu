#include "sm110_thor_components.cuh"

#include <cstdlib>
#include <iostream>

int main() {
  using namespace sm110_thor_components;

  if (!print_device_or_skip()) {
    return EXIT_SUCCESS;
  }

  float* d_out = nullptr;
  SM110_CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
  SM110_CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));

  tcgen05_tmem_probe_kernel<<<1, TmemProbeShape::kThreads>>>(d_out);
  SM110_CHECK_CUDA(cudaGetLastError());
  SM110_CHECK_CUDA(cudaDeviceSynchronize());

  float value = 0.0f;
  SM110_CHECK_CUDA(cudaMemcpy(&value, d_out, sizeof(float),
                              cudaMemcpyDeviceToHost));
  SM110_CHECK_CUDA(cudaFree(d_out));

  const bool matched = matched_one(value);
  std::cout << "tcgen05/tmem probe value = " << value << '\n';
  std::cout << "matched = " << matched << '\n';
  return matched ? EXIT_SUCCESS : EXIT_FAILURE;
}


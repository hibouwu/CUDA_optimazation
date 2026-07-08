#include "sm110_thor_components.cuh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  using namespace sm110_thor_components;

  int total_tiles = 128;
  int workers_per_sm = 1;
  if (argc > 1) total_tiles = std::atoi(argv[1]);
  if (argc > 2) workers_per_sm = std::atoi(argv[2]);
  if (argc > 3 || total_tiles <= 0 || workers_per_sm <= 0) {
    std::cerr << "usage: " << argv[0] << " [tiles] [workers_per_sm]\n";
    return EXIT_FAILURE;
  }
  workers_per_sm = std::min(workers_per_sm, 8);

  if (!print_device_or_skip()) {
    return EXIT_SUCCESS;
  }

  int device = 0;
  SM110_CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  SM110_CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  const int workers =
      std::min(total_tiles, prop.multiProcessorCount * workers_per_sm);
  std::cout << "total_tiles: " << total_tiles << '\n';
  std::cout << "workers: " << workers << '\n';
  std::cout << "workers_per_sm: " << workers_per_sm << '\n';

  float* d_out = nullptr;
  int* d_counter = nullptr;
  const size_t out_bytes = static_cast<size_t>(total_tiles) * sizeof(float);
  SM110_CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
  SM110_CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));

  dim3 grid(workers);
  dim3 block(TmemProbeShape::kThreads);

  SM110_CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));
  clc_static_tmem_probe_kernel<<<grid, block>>>(d_out, total_tiles);
  SM110_CHECK_CUDA(cudaGetLastError());
  SM110_CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> host(static_cast<size_t>(total_tiles));
  SM110_CHECK_CUDA(cudaMemcpy(host.data(), d_out, out_bytes,
                              cudaMemcpyDeviceToHost));
  int static_mismatches = 0;
  for (float value : host) {
    if (!matched_one(value)) ++static_mismatches;
  }
  std::cout << "static CLC mismatches = " << static_mismatches << '\n';

  SM110_CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));
  SM110_CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
  clc_dynamic_tmem_probe_kernel<<<grid, block>>>(d_out, total_tiles,
                                                 d_counter);
  SM110_CHECK_CUDA(cudaGetLastError());
  SM110_CHECK_CUDA(cudaDeviceSynchronize());

  SM110_CHECK_CUDA(cudaMemcpy(host.data(), d_out, out_bytes,
                              cudaMemcpyDeviceToHost));
  int dynamic_mismatches = 0;
  for (float value : host) {
    if (!matched_one(value)) ++dynamic_mismatches;
  }
  std::cout << "dynamic CLC mismatches = " << dynamic_mismatches << '\n';

  SM110_CHECK_CUDA(cudaFree(d_counter));
  SM110_CHECK_CUDA(cudaFree(d_out));
  return (static_mismatches == 0 && dynamic_mismatches == 0) ? EXIT_SUCCESS
                                                             : EXIT_FAILURE;
}


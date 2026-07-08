#include "sm110_thor_components.cuh"

#include <cstdlib>
#include <iostream>

int main() {
  using sm110_thor_components::report_cuda;

  int runtime_version = 0;
  if (!report_cuda("cudaRuntimeGetVersion",
                   cudaRuntimeGetVersion(&runtime_version))) {
    return EXIT_FAILURE;
  }
  std::cout << "runtime version: " << runtime_version << '\n';

  int driver_version = 0;
  if (!report_cuda("cudaDriverGetVersion",
                   cudaDriverGetVersion(&driver_version))) {
    return EXIT_FAILURE;
  }
  std::cout << "driver version: " << driver_version << '\n';

  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (!report_cuda("cudaGetDeviceCount", status)) {
    return EXIT_FAILURE;
  }
  std::cout << "devices: " << device_count << '\n';
  if (device_count <= 0) {
    std::cerr << "No CUDA devices found\n";
    return EXIT_FAILURE;
  }

  if (!report_cuda("cudaFree(0)", cudaFree(0))) {
    return EXIT_FAILURE;
  }
  if (!report_cuda("cudaSetDevice(0)", cudaSetDevice(0))) {
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  status = cudaGetDeviceProperties(&prop, 0);
  if (!report_cuda("cudaGetDeviceProperties(0)", status)) {
    return EXIT_FAILURE;
  }
  std::cout << "GPU: " << prop.name << '\n';
  std::cout << "compute: " << prop.major << "." << prop.minor << '\n';
  std::cout << "integrated: " << prop.integrated << '\n';
  std::cout << "unifiedAddressing: " << prop.unifiedAddressing << '\n';
  std::cout << "managedMemory: " << prop.managedMemory << '\n';

  void* ptr = nullptr;
  if (!report_cuda("cudaMalloc(4)", cudaMalloc(&ptr, 4))) {
    return EXIT_FAILURE;
  }
  if (!report_cuda("cudaFree(ptr)", cudaFree(ptr))) {
    return EXIT_FAILURE;
  }

  std::cout << "SM110 Thor component runtime sanity passed\n";
  return EXIT_SUCCESS;
}


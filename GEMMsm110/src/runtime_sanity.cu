#include "gemm_common.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace {

bool report_cuda(const char* label, cudaError_t status) {
  if (status == cudaSuccess) {
    std::cout << label << ": ok\n";
    return true;
  }
  std::cerr << label << ": code=" << static_cast<int>(status)
            << " name=" << cudaGetErrorName(status)
            << " msg=" << cudaGetErrorString(status) << '\n';
  return false;
}

}  // namespace

int main() {
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

  // CUDA 12+ defines cudaSetDevice as the explicit runtime/context
  // initialization API. The legacy cudaFree(0) idiom is not portable to all
  // Thor BSP/runtime combinations.
  if (!report_cuda("cudaSetDevice(0) / runtime init", cudaSetDevice(0))) {
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

  std::cout << "Runtime sanity passed\n";
  return EXIT_SUCCESS;
}

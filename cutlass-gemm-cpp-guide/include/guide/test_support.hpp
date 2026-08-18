// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "guide/case_runner.hpp"
#include "guide/device_buffer.hpp"

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

#include <sstream>

namespace guide {

inline std::string cutlass_status_message(cutlass::Status status) {
  return cutlassGetStatusString(status);
}

inline VerificationResult require_sm110_device() {
  VerificationResult result;
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    result.status = Status::skip;
    result.message = std::string("no usable CUDA device: ") + cudaGetErrorString(status);
    return result;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    result.status = Status::skip;
    result.message = std::string("cannot query CUDA device: ") + cudaGetErrorString(status);
    return result;
  }
  result.gpu_name = properties.name;
  result.compute_major = properties.major;
  result.compute_minor = properties.minor;
  if (properties.major != 11 || properties.minor != 0) {
    std::ostringstream message;
    message << "case requires SM110/CC 11.0, found " << properties.major << '.' << properties.minor;
    result.status = Status::skip;
    result.message = message.str();
    return result;
  }
  result.status = Status::not_run;
  return result;
}

}  // namespace guide

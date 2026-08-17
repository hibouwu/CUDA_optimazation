// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace guide {

inline void check_cuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_) check_cuda(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      DeviceBuffer tmp(std::move(other));
      swap(tmp);
    }
    return *this;
  }
  void swap(DeviceBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(count_, other.count_);
  }
  T* get() { return data_; }
  T const* get() const { return data_; }
  std::size_t size() const { return count_; }
  void copy_from_host(T const* source, std::size_t count) {
    if (count > count_) throw std::out_of_range("copy_from_host exceeds allocation");
    check_cuda(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  }
  void copy_to_host(T* destination, std::size_t count) const {
    if (count > count_) throw std::out_of_range("copy_to_host exceeds allocation");
    check_cuda(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace guide

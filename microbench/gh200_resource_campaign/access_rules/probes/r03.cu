#include "r00_common.hpp"
#include <fstream>
__device__ unsigned r03_smid() {
  unsigned x;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(x));
  return x;
}
__device__ uint64_t r03_ns() {
  uint64_t x;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(x));
  return x;
}
struct R03Stamp {
  uint64_t begin, end, begin_ns, end_ns;
  unsigned sm;
};
// Eight disjoint slots per warp. All lanes participate in every matrix instruction.
template <int Op, int Width>
__global__ void r03_probe(int iterations, unsigned seed, unsigned* output, R03Stamp* stamp) {
  constexpr int Words = Op >= 2 ? 128 : 32 * Width / 4;
  extern __shared__ __align__(128) unsigned tile[];
  unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int q = threadIdx.x; q < blockDim.x / 32 * 8 * Words; q += blockDim.x)
    tile[q] = 17u * q + seed;
  __syncthreads();
  unsigned sum = 0;
  if (threadIdx.x == 0) {
    stamp->sm = r03_smid();
    stamp->begin_ns = r03_ns();
    stamp->begin = clock64();
  }
  __syncthreads();
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      unsigned slot = (i + q) & 7, base = (warp * 8 + slot) * Words;
      unsigned index = base + (Op >= 2 ? lane * 4 : lane * (Width / 4));
      unsigned address = unsigned(__cvta_generic_to_shared(tile + index));
      unsigned a, b, c, d;
      if constexpr (Op == 0) {
        if constexpr (Width == 4) {
          asm volatile("ld.shared.u32 %0,[%1];" : "=r"(a) : "r"(address) : "memory");
          sum += a;
        } else {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];"
                       : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                       : "r"(address)
                       : "memory");
          sum += a + b + c + d;
        }
      } else if constexpr (Op == 1) {
        a = 17u * index + 31u * (i + 1) + seed;
        b = a + 17;
        c = b + 17;
        d = c + 17;
        if constexpr (Width == 4)
          asm volatile("st.shared.u32 [%0],%1;" ::"r"(address), "r"(a) : "memory");
        else
          asm volatile("st.shared.v4.u32 [%0],{%1,%2,%3,%4};" ::"r"(address), "r"(a), "r"(b),
                       "r"(c), "r"(d)
                       : "memory");
      } else if constexpr (Op == 2) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                     : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                     : "r"(address)
                     : "memory");
        sum += a + b + c + d;
      } else {
        unsigned word = base + (lane / 4) * 4 + lane % 4;
        a = 17u * word + 31u * (i + 1) + seed;
        b = a + 17u * 32;
        c = b + 17u * 32;
        d = c + 17u * 32;
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};" ::"r"(address),
                     "r"(a), "r"(b), "r"(c), "r"(d)
                     : "memory");
      }
    }
  }
  // Materialize load consumers before the end timestamp.
  __shared__ volatile unsigned sink[256];
  sink[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    stamp->end = clock64();
    stamp->end_ns = r03_ns();
  }
  if constexpr (Op == 0 || Op == 2)
    output[threadIdx.x] = sum;
  else
    for (int q = threadIdx.x; q < blockDim.x / 32 * 8 * Words; q += blockDim.x)
      output[q] = tile[q];
}
template <int Op, int Width>
int r03_run(int warps, int iterations, unsigned seed) {
  constexpr unsigned Words = Op >= 2 ? 128 : 32 * Width / 4;
  int threads = warps * 32, smem = warps * 8 * Words * 4;
  auto kernel = r03_probe<Op, Width>;
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  int occupancy;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, threads, smem));
  DeviceBuffer<unsigned> output(Op == 0 || Op == 2 ? threads : warps * 8 * Words);
  DeviceBuffer<R03Stamp> stamps(1);
  std::vector<unsigned> values(output.count);
  R03Stamp stamp;
  auto launch = [&](int count) {
    kernel<<<1, threads, smem>>>(count, seed, output.pointer, stamps.pointer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  };
  auto check = [&](int count) {
    CUDA_CHECK(
        cudaMemcpy(values.data(), output.pointer, values.size() * 4, cudaMemcpyDeviceToHost));
    unsigned errors = 0;
    for (unsigned j = 0; j < values.size(); ++j) {
      unsigned expected = 0;
      if constexpr (Op == 1 || Op == 3)
        expected = 17u * j + 31u * count + seed;
      else {
        unsigned warp = j / 32, lane = j % 32;
        for (int i = 0; i < count; ++i)
          for (int q = 0; q < 8; ++q) {
            unsigned base = (warp * 8 + ((i + q) & 7)) * Words;
            for (int r = 0; r < (Op == 2 ? 4 : Width / 4); ++r) {
              unsigned at = Op == 2 ? base + (lane / 4) * 4 + lane % 4 + r * 32
                                    : base + lane * (Width / 4) + r;
              expected += 17u * at + seed;
            }
          }
      }
      if (values[j] != expected)
        ++errors;
    }
    return errors;
  };
  for (int count : {1, 2, 9}) {
    launch(count);
    if (check(count))
      throw std::runtime_error("R03 nonuniform short check failed");
  }
  std::vector<float> warm;
  bool converged = false;
  for (int i = 0; i < 30; ++i) {
    launch(iterations);
    CUDA_CHECK(cudaMemcpy(&stamp, stamps.pointer, sizeof(stamp), cudaMemcpyDeviceToHost));
    warm.push_back(float(stamp.end - stamp.begin));
    if (i >= 7) {
      std::vector<float> last(warm.end() - 5, warm.end());
      if (coefficient_of_variation(last) <= .02) {
        converged = true;
        break;
      }
    }
  }
  launch(iterations);
  CUDA_CHECK(cudaMemcpy(&stamp, stamps.pointer, sizeof(stamp), cudaMemcpyDeviceToHost));
  unsigned errors = check(iterations);
  std::ofstream file("output.u32", std::ios::binary);
  file.write(reinterpret_cast<char*>(values.data()), values.size() * 4);
  file.close();
  uint64_t bytes = uint64_t(Words) * 4 * warps * 8 * iterations;
  std::cout
      << "{\"status\":\"" << (errors ? "numeric_error" : "measured") << "\",\"op\":" << Op
      << ",\"width\":" << Width << ",\"warps\":" << warps << ",\"iterations\":" << iterations
      << ",\"seed\":" << seed << ",\"accesses_per_round\":8,\"work_bytes\":" << bytes
      << ",\"warp_instructions\":" << uint64_t(warps) * 8 * iterations
      << ",\"elapsed\":" << stamp.end - stamp.begin
      << ",\"elapsed_ns\":" << stamp.end_ns - stamp.begin_ns << ",\"sm_id\":" << stamp.sm
      << ",\"unit\":\"clock64_cycle/CTA\",\"end_event\":\"consumer_sink_and_CTA_join\",\"errors\":"
      << errors << ",\"checked_elements\":" << values.size()
      << ",\"short_checks\":[1,2,9],\"registers_per_thread\":" << attr.numRegs
      << ",\"local_bytes_per_thread\":" << attr.localSizeBytes << ",\"dynamic_smem_bytes\":" << smem
      << ",\"static_smem_bytes\":" << attr.sharedSizeBytes
      << ",\"occupancy_limit_ctas_per_sm\":" << occupancy
      << ",\"warmup_converged\":" << (converged ? "true" : "false") << ",\"warmup\":[";
  for (size_t i = 0; i < warm.size(); ++i)
    std::cout << (i ? "," : "") << warm[i];
  std::cout << "]}\n";
  return errors ? 2 : 0;
}
int main(int argc, char** argv) try {
  int op = 0, width = 16, warps = 4, iterations = 4096;
  unsigned seed = 17;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc)
      throw std::runtime_error("missing value");
    std::string key = argv[i];
    int value = std::stoi(argv[i + 1]);
    if (key == "--op")
      op = value;
    else if (key == "--width")
      width = value;
    else if (key == "--warps")
      warps = value;
    else if (key == "--iterations")
      iterations = value;
    else if (key == "--seed")
      seed = value;
    else
      throw std::runtime_error("unknown option");
  }
  if (iterations < 1 || (warps != 1 && warps != 4 && warps != 8) || (width != 4 && width != 16))
    throw std::runtime_error("invalid coordinate");
  if (op == 0)
    return width == 4 ? r03_run<0, 4>(warps, iterations, seed)
                      : r03_run<0, 16>(warps, iterations, seed);
  if (op == 1)
    return width == 4 ? r03_run<1, 4>(warps, iterations, seed)
                      : r03_run<1, 16>(warps, iterations, seed);
  if (op == 2)
    return r03_run<2, 16>(warps, iterations, seed);
  if (op == 3)
    return r03_run<3, 16>(warps, iterations, seed);
  throw std::runtime_error("invalid op");
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}

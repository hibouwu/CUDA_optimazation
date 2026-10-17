// Narrow L2 requested-service probe. Build for sm_90a; GPU execution is explicit.
// J is the number of independent software streams, not a physical queue size.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using U64 = unsigned long long;
using U32 = unsigned;
constexpr int Threads = 128;
constexpr int ReservedSmem = 128 * 1024;
enum Mode { Read = 0, Write = 1, Mixed = 2 };

#define CUDA_CHECK(call) do { cudaError_t status_ = (call); if (status_ != cudaSuccess) \
  throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(status_)); } while (0)

struct Record {
  U64 begin_ns, end_ns, begin_cycle, end_cycle;
  U32 smid_start, smid_end, active;
};

__device__ __forceinline__ U64 cycle_stamp() {
  U64 value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
  return value;
}
__device__ __forceinline__ U64 ns_stamp() {
  U64 value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}
__device__ __forceinline__ U32 sm_stamp() {
  U32 value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value) :: "memory");
  return value;
}
__device__ __forceinline__ uint4 load_cg(const uint4* address) {
  uint4 value;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(address) : "memory");
  return value;
}
__device__ __forceinline__ void store_wb(uint4* address, uint4 value) {
  asm volatile("st.global.wb.v4.u32 [%0],{%1,%2,%3,%4};"
      :: "l"(address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
      : "memory");
}

template<int Chains, int AccessMode>
__global__ void l2_target(const uint4* input, uint4* output, U64 vectors,
    int active_ctas, int iterations, U32 seed, Record* records, U32* checksums) {
  __shared__ volatile U32 drain[Threads];
  const bool active = int(blockIdx.x) < active_ctas;
  U32 final_sum = 0;
  Record record{};
  if (threadIdx.x == 0) {
    record.active = active;
    record.smid_start = sm_stamp();
  }
  // Only participating CTAs are launched. One-CTA/SM capacity and the cooperative
  // gate give distinct SMs without idle CTAs polling a grid barrier during work.
  cooperative_groups::this_grid().sync();
  if (active) {
    const U64 total_threads = U64(active_ctas) * Threads;
    const U64 thread = U64(blockIdx.x) * Threads + threadIdx.x;
    const U64 batches = vectors / (total_threads * Chains);
    U32 sums[Chains]{};
    if (threadIdx.x == 0) {
      record.begin_ns = ns_stamp();
      record.begin_cycle = cycle_stamp();
    }
    __syncthreads();
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
      #pragma unroll 1
      for (U64 batch = 0; batch < batches; ++batch) {
        uint4 values[Chains];
        // Source/PTX groups J loads before consumption, with independent addresses
        // and checksum streams. ptxas may interleave consumers and reuse registers;
        // preserve that actual schedule in the SASS audit, rather than claiming J
        // simultaneous in-flight loads or J retained vector results.
        if constexpr (AccessMode != Write) {
          #pragma unroll
          for (int chain = 0; chain < Chains; ++chain) {
            U64 index = (batch * Chains + chain) * total_threads + thread;
            values[chain] = load_cg(input + index);
          }
          #pragma unroll
          for (int chain = Chains - 1; chain >= 0; --chain) {
            sums[chain] += values[chain].x + values[chain].y
                + values[chain].z + values[chain].w;
          }
        }
        if constexpr (AccessMode != Read) {
          #pragma unroll
          for (int chain = 0; chain < Chains; ++chain) {
            U64 index = (batch * Chains + chain) * total_threads + thread;
            U32 word = U32(index * 4);
            uint4 value = make_uint4(29u * word + seed, 29u * (word + 1) + seed,
                29u * (word + 2) + seed, 29u * (word + 3) + seed);
            store_wb(output + index, value);
          }
        }
      }
    }
    U32 sum = 0;
    #pragma unroll
    for (int chain = 0; chain < Chains; ++chain) sum += sums[chain];
    drain[threadIdx.x] = sum;
    final_sum = sum;
    if constexpr (AccessMode != Read) __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      record.end_cycle = cycle_stamp();
      record.end_ns = ns_stamp();
    }
  }
  if (threadIdx.x == 0) record.smid_end = sm_stamp();
  cooperative_groups::this_grid().sync();
  // Store diagnostic results only after every active timed interval has ended.
  if (active && AccessMode != Write)
    checksums[U64(blockIdx.x) * Threads + threadIdx.x] = final_sum;
  if (threadIdx.x == 0) records[blockIdx.x] = record;
}

std::string quote(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
U64 integer(const std::string& text, U64 minimum, U64 maximum) {
  if (text.empty() || text[0] == '-') throw std::runtime_error("invalid unsigned argument");
  errno = 0;
  char* end = nullptr;
  U64 value = std::strtoull(text.c_str(), &end, 10);
  if (errno || *end || value < minimum || value > maximum)
    throw std::runtime_error("argument out of range: " + text);
  return value;
}
struct Options {
  int active_ctas = 132, chains = 4, iterations = 16, samples = 1, warmup_limit = 30;
  U64 array_bytes = 17301504;
  U32 seed = 17;
  int mode = Read;
  std::string mode_name = "read";
  Options(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("missing option value");
      std::string key = argv[i], value = argv[i + 1];
      if (key == "--active-ctas") active_ctas = integer(value, 1, 4096);
      else if (key == "--chains") chains = integer(value, 1, 8);
      else if (key == "--array-bytes") array_bytes = integer(value, 16, U64(1) << 34);
      else if (key == "--iterations") iterations = integer(value, 1, 1048576);
      else if (key == "--samples") samples = integer(value, 1, 64);
      else if (key == "--warmup-limit") warmup_limit = integer(value, 8, 60);
      else if (key == "--seed") seed = integer(value, 0, std::numeric_limits<U32>::max());
      else if (key == "--mode") {
        mode_name = value;
        if (value == "read") mode = Read;
        else if (value == "write") mode = Write;
        else if (value == "mixed") mode = Mixed;
        else throw std::runtime_error("mode must be read, write, or mixed");
      } else throw std::runtime_error("unknown option: " + key);
    }
    if (chains != 1 && chains != 4 && chains != 8) throw std::runtime_error("chains must be 1, 4, or 8");
    if (array_bytes % (U64(active_ctas) * Threads * chains * 16))
      throw std::runtime_error("array_bytes must contain whole active_ctas*128*chains vector batches");
    if (array_bytes > std::numeric_limits<U64>::max() / (U64(iterations) * 2))
      throw std::runtime_error("logical work overflows uint64");
  }
};
struct Trial {
  int iterations = 0;
  U64 elapsed_ns = 0, read_checked = 0, write_checked = 0, errors = 0;
  float event_ms = 0;
  bool all_active_windows_overlap = false;
  std::vector<Record> records;
  std::vector<U32> read_checksums;
  std::string output_file;
  U64 output_bytes = 0;
};

// Independent modular arithmetic reference for every vector owned by this thread.
// S = sum_{k=0}^{g-1} sum_{j=0}^{3} (17*(4*(k*T+t)+j)+seed).
U32 expected_checksum(U64 array_bytes, U64 total_threads, U64 thread,
    int iterations, U32 seed) {
  U32 g = U32(array_bytes / 16 / total_threads), t = U32(thread), T = U32(total_threads);
  U32 indices = 8u * T * g * (g - 1u) + 16u * t * g + 6u * g;
  return (17u * indices + 4u * g * seed) * U32(iterations);
}
double tail_cv(const std::vector<U64>& windows) {
  if (windows.size() < 5) throw std::runtime_error("five windows required");
  double mean = 0, variance = 0;
  for (std::size_t i = windows.size() - 5; i < windows.size(); ++i) mean += double(windows[i]) / 5;
  if (mean <= 0) throw std::runtime_error("nonpositive warmup window");
  for (std::size_t i = windows.size() - 5; i < windows.size(); ++i) {
    double difference = double(windows[i]) - mean;
    variance += difference * difference / 4;
  }
  return std::sqrt(variance) / mean;
}

template<int Chains>
const void* mode_function(int mode) {
  if (mode == Read) return reinterpret_cast<const void*>(l2_target<Chains, Read>);
  if (mode == Write) return reinterpret_cast<const void*>(l2_target<Chains, Write>);
  return reinterpret_cast<const void*>(l2_target<Chains, Mixed>);
}
void print_trial(const Trial& trial) {
  std::cout << "{\"iterations\":" << trial.iterations << ",\"elapsed_ns\":" << trial.elapsed_ns
      << ",\"event_ms\":" << trial.event_ms
      << ",\"all_active_windows_overlap\":" << (trial.all_active_windows_overlap ? "true" : "false")
      << ",\"correctness\":{\"read_checked\":" << trial.read_checked
      << ",\"write_checked\":" << trial.write_checked << ",\"errors\":" << trial.errors << "}"
      << ",\"read_checksums\":[";
  for (std::size_t i = 0; i < trial.read_checksums.size(); ++i) std::cout << (i ? "," : "") << trial.read_checksums[i];
  std::cout << "],\"output_file\":" << (trial.output_file.empty() ? "null" : quote(trial.output_file))
      << ",\"output_bytes\":" << trial.output_bytes << ",\"blocks_detail\":[";
  for (std::size_t i = 0; i < trial.records.size(); ++i) {
    const auto& r = trial.records[i];
    std::cout << (i ? "," : "") << "{\"block_id\":" << i << ",\"active\":" << (r.active ? "true" : "false")
        << ",\"smid_start\":" << r.smid_start << ",\"smid_end\":" << r.smid_end
        << ",\"begin_ns\":" << r.begin_ns << ",\"end_ns\":" << r.end_ns
        << ",\"begin_cycle\":" << r.begin_cycle << ",\"end_cycle\":" << r.end_cycle << "}";
  }
  std::cout << "]}";
}

int main(int argc, char** argv) try {
  const Options options(argc, argv);
  const U32 endian_check = 1;
  if (*reinterpret_cast<const unsigned char*>(&endian_check) != 1) throw std::runtime_error("little endian witness required");
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count != 1) throw std::runtime_error("exactly one visible GPU required");
  cudaDeviceProp device{};
  CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
  if (device.major != 9 || device.minor != 0 || std::string(device.name).find("GH200") == std::string::npos)
    throw std::runtime_error("GH200 CC9.0 required");
  if (!device.cooperativeLaunch || options.active_ctas > device.multiProcessorCount)
    throw std::runtime_error("cooperative launch and active_ctas <= SM count required");
  const int grid_ctas = options.active_ctas;
  const U64 total_threads = U64(options.active_ctas) * Threads;
  const bool reads = options.mode != Write, writes = options.mode != Read;
  const U64 aggregate_bytes = options.array_bytes * (unsigned(reads) + unsigned(writes));
  if (!device.l2CacheSize || aggregate_bytes >= U64(device.l2CacheSize))
    throw std::runtime_error("active arrays must fit below queried L2 size");
  const void* function = options.chains == 1 ? mode_function<1>(options.mode)
      : options.chains == 4 ? mode_function<4>(options.mode) : mode_function<8>(options.mode);
  CUDA_CHECK(cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize, ReservedSmem));
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, function));
  int occupancy = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, function, Threads, ReservedSmem));
  if (occupancy != 1 || attributes.localSizeBytes != 0)
    throw std::runtime_error("one CTA/SM capacity without local/spill required");
  int driver_version = 0, runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  std::ostringstream uuid;
  uuid << "GPU-";
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) uuid << '-';
    uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(static_cast<unsigned char>(device.uuid.bytes[i]));
  }
  uint4 *input = nullptr, *output = nullptr;
  U32* sums = nullptr;
  Record* records = nullptr;
  std::vector<U32> host_input(reads ? options.array_bytes / 4 : 0);
  std::vector<U32> poison(writes ? options.array_bytes / 4 : 0), got_output(poison.size());
  for (U64 word = 0; word < host_input.size(); ++word) host_input[word] = 17u * U32(word) + options.seed;
  for (U64 word = 0; word < poison.size(); ++word) poison[word] = ~(29u * U32(word) + options.seed);
  if (reads) {
    CUDA_CHECK(cudaMalloc(&input, options.array_bytes));
    CUDA_CHECK(cudaMemcpy(input, host_input.data(), options.array_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&sums, total_threads * sizeof(U32)));
  }
  if (writes) CUDA_CHECK(cudaMalloc(&output, options.array_bytes));
  CUDA_CHECK(cudaMalloc(&records, grid_ctas * sizeof(Record)));
  cudaEvent_t event_begin, event_end;
  CUDA_CHECK(cudaEventCreate(&event_begin));
  CUDA_CHECK(cudaEventCreate(&event_end));
  auto execute = [&](int length, const std::string& witness_file) {
    if (writes) CUDA_CHECK(cudaMemcpy(output, poison.data(), options.array_bytes, cudaMemcpyHostToDevice));
    if (reads) CUDA_CHECK(cudaMemset(sums, 0xff, total_threads * sizeof(U32)));
    CUDA_CHECK(cudaMemset(records, 0xff, grid_ctas * sizeof(Record)));
    U64 vectors = options.array_bytes / 16;
    int active = options.active_ctas;
    U32 seed = options.seed;
    void* arguments[] = {&input, &output, &vectors, &active, &length, &seed, &records, &sums};
    CUDA_CHECK(cudaEventRecord(event_begin));
    CUDA_CHECK(cudaLaunchCooperativeKernel(function, dim3(grid_ctas), dim3(Threads), arguments, ReservedSmem));
    CUDA_CHECK(cudaEventRecord(event_end));
    CUDA_CHECK(cudaEventSynchronize(event_end));
    Trial trial;
    trial.iterations = length;
    trial.records.resize(grid_ctas);
    CUDA_CHECK(cudaEventElapsedTime(&trial.event_ms, event_begin, event_end));
    CUDA_CHECK(cudaMemcpy(trial.records.data(), records, grid_ctas * sizeof(Record), cudaMemcpyDeviceToHost));
    U64 first = std::numeric_limits<U64>::max(), last = 0, last_begin = 0, first_end = std::numeric_limits<U64>::max();
    std::set<U32> grid_sms, active_sms;
    for (int block = 0; block < grid_ctas; ++block) {
      const auto& r = trial.records[block];
      if (r.active != unsigned(block < active) || r.smid_start != r.smid_end || r.smid_start == ~U32(0))
        throw std::runtime_error("incomplete or migrated CTA record");
      grid_sms.insert(r.smid_start);
      if (r.active) {
        if (r.end_ns <= r.begin_ns || r.end_cycle <= r.begin_cycle) throw std::runtime_error("invalid active timer");
        first = std::min(first, r.begin_ns); last = std::max(last, r.end_ns);
        last_begin = std::max(last_begin, r.begin_ns); first_end = std::min(first_end, r.end_ns);
        active_sms.insert(r.smid_start);
      } else if (r.begin_ns || r.end_ns || r.begin_cycle || r.end_cycle) throw std::runtime_error("idle CTA entered timed work");
    }
    if (int(grid_sms.size()) != grid_ctas || int(active_sms.size()) != active)
      throw std::runtime_error("cooperative one-CTA/SM coverage not observed");
    trial.elapsed_ns = last - first;
    trial.all_active_windows_overlap = last_begin < first_end;
    if (!std::isfinite(trial.event_ms) || trial.event_ms <= 0 || double(trial.elapsed_ns) > double(trial.event_ms) * 1.05e6)
      throw std::runtime_error("event/globaltimer consistency failure");
    if (reads) {
      trial.read_checksums.resize(total_threads);
      CUDA_CHECK(cudaMemcpy(trial.read_checksums.data(), sums, total_threads * sizeof(U32), cudaMemcpyDeviceToHost));
      for (U64 thread = 0; thread < total_threads; ++thread) {
        if (trial.read_checksums[thread] != expected_checksum(options.array_bytes, total_threads, thread, length, options.seed)) ++trial.errors;
        ++trial.read_checked;
      }
    }
    if (writes) {
      CUDA_CHECK(cudaMemcpy(got_output.data(), output, options.array_bytes, cudaMemcpyDeviceToHost));
      for (U64 word = 0; word < got_output.size(); ++word) {
        if (got_output[word] != 29u * U32(word) + options.seed) ++trial.errors;
        ++trial.write_checked;
      }
      if (!witness_file.empty()) {
        std::ofstream file(witness_file, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(got_output.data()), options.array_bytes);
        file.close();
        if (!file) throw std::runtime_error("output witness write failed");
        trial.output_file = witness_file; trial.output_bytes = options.array_bytes;
      }
    }
    if (trial.errors) throw std::runtime_error("exact checksum/output errors: " + std::to_string(trial.errors));
    return trial;
  };
  const auto check_one = execute(1, ""), check_two = execute(2, "");
  std::vector<U64> warmup;
  double warmup_cv = 0;
  bool converged = false;
  for (int window = 0; window < options.warmup_limit; ++window) {
    warmup.push_back(execute(options.iterations, "").elapsed_ns);
    if (warmup.size() >= 5) warmup_cv = tail_cv(warmup);
    if (warmup.size() >= 8 && warmup_cv <= 0.02) { converged = true; break; }
  }
  std::vector<Trial> trials;
  for (int sample = 0; sample < options.samples; ++sample) {
    std::string witness = !writes ? "" : options.samples == 1 ? "output.u32" : "output-" + std::to_string(sample) + ".u32";
    trials.push_back(execute(options.iterations, witness));
  }
  CUDA_CHECK(cudaEventDestroy(event_begin));
  CUDA_CHECK(cudaEventDestroy(event_end));
  if (input) CUDA_CHECK(cudaFree(input));
  if (output) CUDA_CHECK(cudaFree(output));
  if (sums) CUDA_CHECK(cudaFree(sums));
  CUDA_CHECK(cudaFree(records));
  U64 read_bytes = reads ? options.array_bytes * options.iterations : 0;
  U64 write_bytes = writes ? options.array_bytes * options.iterations : 0;
  std::cout << std::setprecision(17) << "{\"schema\":\"gh200-l2-saturation/1\",\"status\":\"measured\""
      << ",\"active_ctas\":" << options.active_ctas << ",\"chains\":" << options.chains
      << ",\"mode\":" << quote(options.mode_name) << ",\"array_bytes\":" << options.array_bytes
      << ",\"iterations\":" << options.iterations << ",\"seed\":" << options.seed << ",\"samples\":" << options.samples
      << ",\"read_payload_bytes\":" << read_bytes << ",\"write_payload_bytes\":" << write_bytes
      << ",\"work_count\":" << read_bytes + write_bytes << ",\"aggregate_array_bytes\":" << aggregate_bytes
      << ",\"performance_eligible\":" << (converged ? "true" : "false")
      << ",\"cache_residency_proven\":false,\"physical_hbm_bytes_proven\":false"
      << ",\"device\":{\"uuid\":" << quote(uuid.str()) << ",\"name\":" << quote(device.name)
      << ",\"sms\":" << device.multiProcessorCount << ",\"l2_cache_bytes\":" << device.l2CacheSize
      << ",\"driver_version\":" << driver_version << ",\"runtime_version\":" << runtime_version
      << ",\"compiler_version\":" << quote(std::to_string(__CUDACC_VER_MAJOR__) + "." + std::to_string(__CUDACC_VER_MINOR__) + "." + std::to_string(__CUDACC_VER_BUILD__)) << "}"
      << ",\"resources\":{\"threads\":" << Threads << ",\"grid_ctas\":" << grid_ctas
      << ",\"registers_per_thread\":" << attributes.numRegs << ",\"dynamic_smem_bytes\":" << ReservedSmem
      << ",\"static_smem_bytes\":" << attributes.sharedSizeBytes << ",\"local_size_bytes\":" << attributes.localSizeBytes
      << ",\"occupancy_ctas_per_sm\":" << occupancy << "}"
      << ",\"conditions\":{\"vector_bytes\":16,\"read_cache_policy\":\"cg\",\"write_cache_policy\":\"wb\""
      << ",\"input_word\":\"uint32(17*word_index+seed)\",\"output_word\":\"uint32(29*word_index+seed)\""
      << ",\"source_destination_relation\":\"distinct allocations; independent write values\""
      << ",\"cache_preparation\":\"bounded target self warmup; output poisoned outside every timing window\""
      << ",\"endpoint\":\"CTA start gate through all read consumers, writing device fence, and CTA drain\""
      << ",\"chain_meaning\":\"independent software address/checksum streams; compiled loads and consumers may interleave; not a physical in-flight count\"}"
      << ",\"short_check\":{\"iterations\":[1,2],\"read_checked\":" << check_one.read_checked + check_two.read_checked
      << ",\"write_checked\":" << check_one.write_checked + check_two.write_checked << ",\"errors\":0}"
      << ",\"warmup\":{\"converged\":" << (converged ? "true" : "false")
      << ",\"cv\":" << warmup_cv << ",\"limit\":" << options.warmup_limit << ",\"elapsed_ns\":[";
  for (std::size_t i = 0; i < warmup.size(); ++i) std::cout << (i ? "," : "") << warmup[i];
  std::cout << "]},\"trials\":[";
  for (std::size_t i = 0; i < trials.size(); ++i) { if (i) std::cout << ','; print_trial(trials[i]); }
  std::cout << "]}\n";
  return 0;
} catch (const std::exception& error) {
  std::cout << "{\"schema\":\"gh200-l2-saturation/1\",\"status\":\"error\",\"error\":" << quote(error.what()) << "}\n";
  return 2;
}

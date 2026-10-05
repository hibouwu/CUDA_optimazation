#include "../common/probe_runtime.cuh"
#include "../common/synchronization_reference.hpp"

// Mode: 0 warp barrier, 1 CTA barrier, 2 mbarrier, 3 CTA fence,
//       4 GPU fence, 5 generic/async shared proxy fence.
struct SyncThread {
  unsigned value, errors, timeout, smid;
  gh::u64 completed, attempts, checked;
  gh::u64 arrival_cycle, departure_cycle;
};

__device__ __forceinline__ gh::u64 sync_ns() {
  gh::u64 value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
__device__ __forceinline__ gh::u64 sync_cycle() {
  gh::u64 value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value));
  return value;
}
__device__ __forceinline__ unsigned sync_smid() {
  unsigned value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}

template<bool Skew, int Mode, int Threads>
__device__ __forceinline__ unsigned fixed_work(unsigned value) {
  if constexpr (Skew) {
    const bool selected = Mode == 0 ? threadIdx.x % 32 >= 16
                                    : threadIdx.x / 32 == Threads / 32 - 1;
    if (selected) {
      #pragma unroll 256
      for (int step = 0; step < 256; ++step) {
        asm volatile("mad.lo.u32 %0, %0, 1664525, 1013904223;"
                     : "+r"(value));
      }
    }
  }
  return value;
}

template<int Mode>
__device__ __forceinline__ bool target_sync(unsigned address, unsigned* abort_flag,
                                           gh::u64& attempts, unsigned& timeout) {
  if constexpr (Mode == 0) {
    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
  } else if constexpr (Mode == 1) {
    asm volatile("bar.sync 1;" ::: "memory");
  } else if constexpr (Mode == 2) {
    // No per-phase CTA barrier is inserted. A thread may be one phase ahead;
    // every waiting/next-phase thread observes the same abort flag on failure.
    if (atomicAdd(abort_flag, 0u)) return false;
    gh::u64 state;
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];"
                 : "=l"(state) : "r"(address) : "memory");
    const gh::u64 start = sync_ns();
    for (;;) {
      unsigned done;
      asm volatile(
        "{ .reg .pred p; "
        "mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2, 64; "
        "selp.b32 %0, 1, 0, p; }"
        : "=r"(done) : "r"(address), "l"(state) : "memory");
      ++attempts;
      if (done) return true;
      if (atomicAdd(abort_flag, 0u)) return false;
      if (sync_ns() - start >= 1000000000ull) {
        timeout = 1;
        atomicExch(abort_flag, 1u);
        return false;
      }
    }
  } else if constexpr (Mode == 3) {
    asm volatile("fence.acq_rel.cta;" ::: "memory");
  } else if constexpr (Mode == 4) {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
  } else {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  return true;
}

template<int Mode, int Threads, bool Skew, int Purpose>
__device__ __forceinline__ void sync_body(int iterations, unsigned seed,
                                         gh::Stamp* stamps, SyncThread* outputs) {
  // Purpose 0: measured loop. 1: two-phase producer/consumer check.
  // Purpose 2: one-operation arrival diagnostic; never a performance sample.
  __shared__ __align__(8) unsigned long long barrier;
  __shared__ unsigned abort_flag;
  __shared__ volatile unsigned tokens[Threads];
  __shared__ volatile unsigned drain[Threads];
  __shared__ gh::u64 start_ns, start_cycle;
  const unsigned tid = threadIdx.x;
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
  if (tid == 0) {
    abort_flag = 0;
    if constexpr (Mode == 2) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                   :: "r"(address), "n"(Threads) : "memory");
    }
  }
  tokens[tid] = 0xdeadbeefu;
  SyncThread result{};
  result.value = seed + 17u * tid;
  __syncthreads();
  if (tid == 0) {
    start_ns = sync_ns();
    start_cycle = sync_cycle();
  }
  __syncthreads();

  constexpr int Batch = Purpose == 0 ? 8 : 1;
  bool active = true;
  #pragma unroll 1
  for (int iteration = 0; iteration < iterations && active; ++iteration) {
    #pragma unroll
    for (int position = 0; position < Batch; ++position) {
      const int phase = iteration * Batch + position;
      result.value = fixed_work<Skew, Mode, Threads>(result.value);
      if constexpr (Purpose == 1) {
        tokens[tid] = seed ^ (tid * 2654435761u) ^ (unsigned(phase) * 2246822519u);
      }
      if constexpr (Purpose == 2) result.arrival_cycle = sync_cycle();
      if (!target_sync<Mode>(address, &abort_flag, result.attempts, result.timeout)) {
        active = false;
        break;
      }
      if constexpr (Purpose == 2) result.departure_cycle = sync_cycle();
      if constexpr (Purpose == 1) {
        // Fence issue timing does not itself synchronize different participants.
        if constexpr (Mode >= 3) __syncthreads();
        const unsigned neighbor = Mode == 0 ? (tid + 1) % 32 : (tid + 1) % Threads;
        const unsigned expected = seed ^ (neighbor * 2654435761u)
                                  ^ (unsigned(phase) * 2246822519u);
        result.errors += tokens[neighbor] != expected;
        ++result.checked;
        // Prevent next-phase writers overtaking previous-phase consumers.
        if (!target_sync<Mode>(address, &abort_flag, result.attempts, result.timeout)) {
          active = false;
          break;
        }
        if constexpr (Mode >= 3) __syncthreads();
      }
      ++result.completed;
    }
  }

  drain[tid] = result.value ^ unsigned(result.completed) ^ unsigned(result.attempts);
  // Every mbarrier exit, including aborts, reaches this same CTA barrier.
  // Polling loops test abort independently; no thread returns early from the CTA.
  __syncthreads();
  if (tid == 0) {
    const auto stop_cycle = sync_cycle();
    const auto stop_ns = sync_ns();
    stamps[0] = {start_ns, stop_ns, start_cycle, stop_cycle, sync_smid()};
    if constexpr (Mode == 2) {
      // On failure no thread reuses the object; it is reclaimed with this CTA.
      // Do not invalidate an incompletely arrived phase.
      if (!abort_flag) {
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                     :: "r"(address) : "memory");
      }
    }
  }
  result.timeout |= abort_flag;
  result.smid = sync_smid();
  outputs[tid] = result;
}

template<int Mode, int Threads, bool Skew>
__global__ void sync_measured(int iterations, unsigned seed,
                               gh::Stamp* stamps, SyncThread* outputs) {
  sync_body<Mode, Threads, Skew, 0>(iterations, seed, stamps, outputs);
}
template<int Mode, int Threads, bool Skew>
__global__ void sync_correctness(int phases, unsigned seed,
                                  gh::Stamp* stamps, SyncThread* outputs) {
  sync_body<Mode, Threads, Skew, 1>(phases, seed, stamps, outputs);
}
template<int Mode, int Threads, bool Skew>
__global__ void sync_arrival(int phases, unsigned seed,
                              gh::Stamp* stamps, SyncThread* outputs) {
  sync_body<Mode, Threads, Skew, 2>(phases, seed, stamps, outputs);
}

using SyncKernel = void(*)(int, unsigned, gh::Stamp*, SyncThread*);
struct SyncCase {
  const char* id;
  int mode, threads, iterations;
  bool skew;
  SyncKernel measured, correctness, arrival;
};
template<int Mode, int Threads, bool Skew>
SyncCase make_case(const char* id) {
  return {id, Mode, Threads, Mode < 3 ? 2048 : 8192, Skew,
          sync_measured<Mode, Threads, Skew>,
          sync_correctness<Mode, Threads, Skew>, sync_arrival<Mode, Threads, Skew>};
}

inline std::string thread_json(const std::vector<SyncThread>& rows, bool diagnostic) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    if (i) out << ',';
    out << "{\"thread_id\":" << i << ",\"value\":" << r.value
        << ",\"completed\":" << r.completed << ",\"wait_attempts\":" << r.attempts
        << ",\"timeout\":" << r.timeout << ",\"errors\":" << r.errors
        << ",\"checked\":" << r.checked << ",\"smid\":" << r.smid;
    if (diagnostic) {
      out << ",\"arrival_cycle\":" << r.arrival_cycle
          << ",\"departure_cycle\":" << r.departure_cycle;
    }
    out << '}';
  }
  out << ']';
  return out.str();
}

int main(int argc, char** argv) try {
  auto device = gh::device();
  gh::emit_device(device);
  if (argc == 2 && std::string(argv[1]) == "device") return 0;
  if (argc != 4) throw std::runtime_error("CASE_ID ITERATIONS SEED required");
  int iterations = gh::integer(argv[2], 1, 8192);
  unsigned seed = gh::integer(argv[3], 0, 4294967295ull);
  const std::vector<SyncCase> cases = {
    make_case<0, 32, false>("warp_t32_aligned"),
    make_case<0, 32, true>("warp_t32_fixed_work_skew"),
    make_case<1, 128, false>("cta_t128_aligned"),
    make_case<1, 128, true>("cta_t128_fixed_work_skew"),
    make_case<1, 256, false>("cta_t256_aligned"),
    make_case<1, 256, true>("cta_t256_fixed_work_skew"),
    make_case<2, 128, false>("mbarrier_t128_aligned"),
    make_case<2, 128, true>("mbarrier_t128_fixed_work_skew"),
    make_case<2, 256, false>("mbarrier_t256_aligned"),
    make_case<2, 256, true>("mbarrier_t256_fixed_work_skew"),
    make_case<3, 32, false>("fence_cta_no_outstanding_work"),
    make_case<4, 32, false>("fence_gpu_no_outstanding_work"),
    make_case<5, 32, false>("proxy_async_no_outstanding_work")
  };
  auto found = std::find_if(cases.begin(), cases.end(), [&](const SyncCase& c) {
    return c.id == std::string(argv[1]);
  });
  if (found == cases.end()) throw std::runtime_error("unknown S11 case");
  const SyncCase c = *found;
  if (iterations != c.iterations) throw std::runtime_error("S11 frozen iteration mismatch");
  cudaFuncAttributes attributes{};
  GH_CUDA(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void*>(c.measured)));
  if (attributes.localSizeBytes) throw std::runtime_error("S11 local spill forbidden");
  gh::Stamp* stamps;
  SyncThread* output;
  GH_CUDA(cudaMalloc(&stamps, sizeof(gh::Stamp)));
  GH_CUDA(cudaMalloc(&output, c.threads * sizeof(SyncThread)));
  cudaEvent_t start, stop;
  GH_CUDA(cudaEventCreate(&start));
  GH_CUDA(cudaEventCreate(&stop));
  std::vector<SyncThread> got(c.threads);
  auto execute = [&](SyncKernel kernel, int count, int purpose) {
    GH_CUDA(cudaMemset(output, 0xff, c.threads * sizeof(SyncThread)));
    void* args[] = {&count, &seed, &stamps, &output};
    GH_CUDA(cudaEventRecord(start));
    GH_CUDA(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(1),
                            dim3(c.threads), args, 0, nullptr));
    GH_CUDA(cudaGetLastError());
    GH_CUDA(cudaEventRecord(stop));
    GH_CUDA(cudaEventSynchronize(stop));
    gh::Observation observation;
    observation.stamps.resize(1);
    float elapsed;
    GH_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    observation.event_ms = elapsed;
    GH_CUDA(cudaMemcpy(observation.stamps.data(), stamps, sizeof(gh::Stamp),
                       cudaMemcpyDeviceToHost));
    GH_CUDA(cudaMemcpy(got.data(), output, c.threads * sizeof(SyncThread),
                       cudaMemcpyDeviceToHost));
    const gh::u64 phases = purpose == 0 ? gh::u64(count) * 8 : count;
    for (unsigned t = 0; t < got.size(); ++t) {
      const auto& row = got[t];
      const auto expected = synchronization_reference::expected(
        c.mode, c.skew, t, c.threads, seed, phases);
      observation.errors += row.errors + row.timeout;
      observation.errors += row.value != expected || row.completed != phases;
      observation.errors += row.checked != (purpose == 1 ? phases : 0);
      const gh::u64 minimum_attempts = phases * (purpose == 1 ? 2 : 1);
      observation.errors += c.mode == 2 ? row.attempts < minimum_attempts : row.attempts != 0;
      observation.errors += row.smid != observation.stamps[0].smid;
      if (purpose == 2) {
        observation.errors += row.departure_cycle <= row.arrival_cycle;
      }
    }
    observation.checked_elements = c.threads;
    observation.method = "lcg_and_collective_phase_count_v1";
    observation.input_conditions = "x0=seed+17*thread modulo2^32; selected threads advance256 MADs per phase";
    if (observation.errors) {
      std::ostringstream error;
      error << "S11 numerical/timeout failure case=" << c.id << " purpose=" << purpose
            << " phases=" << phases << " errors=" << observation.errors
            << " thread_records=" << thread_json(got, purpose == 2);
      throw std::runtime_error(error.str());
    }
    return observation;
  };

  execute(c.correctness, 2, 1);
  const std::string correctness_rows = thread_json(got, false);
  execute(c.arrival, 1, 2);
  const std::string diagnostic_rows = thread_json(got, true);
  gh::u64 first_arrival = ~0ull, last_arrival = 0;
  for (const auto& row : got) {
    first_arrival = std::min(first_arrival, row.arrival_cycle);
    last_arrival = std::max(last_arrival, row.arrival_cycle);
  }
  auto measured = [&]() {
    auto observation = execute(c.measured, iterations, 0);
    gh::envelope(observation);
    return observation;
  };
  auto warmup = gh::warmup(measured);
  auto observation = measured();
  std::ostringstream extra;
  extra << "\"phase\":\"measure\",\"timing_model\":\"synchronization_phase_window_v1\""
        << ",\"registers_per_thread\":" << attributes.numRegs
        << ",\"static_smem_bytes\":" << attributes.sharedSizeBytes
        << ",\"local_size_bytes\":" << attributes.localSizeBytes
        << ",\"thread_results\":" << thread_json(got, false)
        << ",\"correctness_validation\":{\"phases\":2,\"errors\":0,\"threads\":"
        << correctness_rows << '}'
        << ",\"arrival_diagnostic\":{\"separate_launch\":true,\"phases\":1,"
        << "\"clock_basis\":\"same_sm_clock64\",\"arrival_spread_cycles\":"
        << last_arrival - first_arrival << ",\"threads\":" << diagnostic_rows << '}'
        << ",\"async_completion_proven\":false";
  gh::emit_trial(c.id, iterations, seed, c.threads, "one_cta", "operation",
                 gh::u64(iterations) * 8, 0, 0, observation, warmup, extra.str());
  GH_CUDA(cudaEventDestroy(start));
  GH_CUDA(cudaEventDestroy(stop));
  GH_CUDA(cudaFree(stamps));
  GH_CUDA(cudaFree(output));
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}

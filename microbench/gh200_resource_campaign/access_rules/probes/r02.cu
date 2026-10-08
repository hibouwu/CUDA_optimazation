// R02: legal source/pool/mixed-return probes. Timers distinguish producer progress
// from a consumer-dependent endpoint; neither claims completion of overwritten results.
#include "r01_support.hpp"
#include <cstring>

constexpr int kUnroll = 64;
constexpr int kOutputWords = 49;

struct R02Stamp {
  uint64_t start, progress, consumed;
  unsigned sm;
};

__device__ __forceinline__ uint64_t timer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
  return value;
}

// The clock predicate depends on the final consumer value. This witness always
// has a nonzero checksum. A zero checksum yields an invalid (zero) endpoint,
// rather than two identical clock branches that an optimizer might merge.
__device__ __forceinline__ uint64_t consumer_timer(uint64_t value) {
  uint64_t end;
  asm volatile("{ .reg .pred p; setp.ne.u64 p,%1,0; "
               "mov.u64 %0,0; @p mov.u64 %0, %%clock64; }"
               : "=l"(end) : "l"(value) : "memory");
  return end;
}

struct R02Options {
  std::string family = "source", source = "reuse_pair", pair = "fadd_imad_wide";
  int chains = 8, warps = 1, pool = 8, steps = 8192;
  int split = 0, control = 0, intermediate = 1, position = 0, banked = 0, witness = 0;
  R02Options(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("missing option value");
      std::string key = argv[i], val = argv[i + 1];
      if (key == "--family") family = val;
      else if (key == "--source") source = val;
      else if (key == "--pair") pair = val;
      else if (key == "--chains") chains = std::stoi(val);
      else if (key == "--warps") warps = std::stoi(val);
      else if (key == "--pool") pool = std::stoi(val);
      else if (key == "--steps") steps = std::stoi(val);
      else if (key == "--split") split = std::stoi(val);
      else if (key == "--control") control = std::stoi(val);
      else if (key == "--intermediate") intermediate = std::stoi(val);
      else if (key == "--position") position = std::stoi(val);
      else if (key == "--banked") banked = std::stoi(val);
      else if (key == "--witness") witness = std::stoi(val);
      else throw std::runtime_error("unknown option " + key);
    }
    if (steps < 64 || steps % 64 || (warps != 1 && warps != 4) ||
        (chains != 1 && chains != 8) || (pool != 8 && pool != 32) ||
        split < 0 || split > 1 || control < 0 || control > 1 ||
        intermediate < 0 || intermediate > 1 || position < 0 || position > 1 || banked < 0 || banked > 1 || witness < 0 || witness > 1)
      throw std::runtime_error("invalid R02 coordinate");
  }
};

// Family=0 source; 1 FADD pool; 2 u64 pool; 3 FADD+IMAD; 4 FADD+LDS.
// Source=0 constant multiplier; 1 one register pair; 2 eight pairs.
template <int Family, int Source, int Pool, bool Split, int Position>
__global__ void r02_probe(const float* input, int steps, bool preloaded,
                          bool intermediate, bool banked,
                          R02Stamp* stamps, uint64_t* output) {
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  constexpr bool mixed = Family >= 3;
  constexpr int float_count = Family == 2 ? 0 : mixed ? 8 : Pool;
  constexpr int int_count = Family == 2 ? Pool : Family == 3 ? 8 : 0;
  float src[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    // Independent volatile input loads retain separate runtime source registers.
    src[i] = reinterpret_cast<const volatile float*>(input)[i];
    asm volatile("" : "+f"(src[i]) :: "memory");
  }
  float f[32];
  uint64_t u[32];
  unsigned lds[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    f[j] = j < float_count ? (tid + 1) / 1024.f + j / 16.f : 0.f;
    u[j] = j < int_count ? (uint64_t((mixed ? lane : tid) + 1) << 32) + j + (Family == 3 ? 1 : 0) : 0;
    lds[j] = 0;
  }
  __shared__ __align__(16) unsigned memory[32 * 32];
  if constexpr (Family == 4) {
    for (int j = tid; j < 32 * 32; j += blockDim.x)
      memory[j] = banked ? unsigned(j / 32 + 1) * 256 + j % 32
                        : unsigned((j / 4) % 32 + 1) * 256 + (j / 128) * 4 + j % 4;
  }
  const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(memory)) + lane * (banked ? 128 : 16);
  const unsigned vector_step = banked ? 16 : 512;
  const bool active_f = !mixed || warp == 0;
  const bool active_b = !mixed || (Split ? warp == 1 : warp == 0);
  const unsigned a = (mixed ? lane : tid) + 17;
  const unsigned b = Family == 3 ? __float2uint_rn(src[0] * 4.f) + 1 : 3;
  if (preloaded) {
#pragma unroll
    for (int j = 0; j < float_count; ++j) {
      if (active_f) {
        if constexpr (Family == 0 && Source == 2 && Pool == 1) {
          float increment = 0;
#pragma unroll
          for (int q=0;q<8;++q) increment += src[q]*src[8+q];
          f[j] += (steps/8)*increment;
        } else if constexpr (Family == 0) {
          const int q=(Source==2||(Source==3&&banked))?j:0;
          f[j] += (steps/float_count)*(Source==0?1.25f:src[q])*src[8+q];
        } else f[j] += (steps/(float_count?float_count:1))*src[8];
      }
    }
#pragma unroll
    for (int j = 0; j < int_count; ++j) {
      if (active_b) {
        if constexpr (Family == 3) {
          for (int n = 0; n < steps / int_count; ++n) u[j] += uint64_t(unsigned(u[j])) * b;
        } else u[j] += uint64_t(steps / (int_count ? int_count : 1)) * a * b;
      }
    }
    if constexpr (Family == 4) {
#pragma unroll
      for (int j = 0; j < 32; ++j)
        if (active_b) lds[j] = unsigned(lane + 1) * 256 + j;
    }
  }
  // Opaque preparation prevents the preload expression from being propagated
  // into the measured consumer. Values remain identical to producer outputs.
#pragma unroll
  for (int j = 0; j < float_count; ++j) asm volatile("" : "+f"(f[j]) :: "memory");
#pragma unroll
  for (int j = 0; j < int_count; ++j) asm volatile("" : "+l"(u[j]) :: "memory");
  if constexpr (Family == 4) {
#pragma unroll
    for (int j = 0; j < 32; ++j) asm volatile("" : "+r"(lds[j]) :: "memory");
  }
  // Make the input-ready boundary real: every runtime source and retained
  // initial result is consumed before the store/barrier. A barrier alone
  // would not force independent global loads to become operand-ready.
  __shared__ volatile uint64_t ready_gate[128];
  float ready_float = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) ready_float += src[j];
#pragma unroll
  for (int j = 0; j < float_count; ++j) ready_float += f[j];
  uint64_t ready_integer = b;
#pragma unroll
  for (int j = 0; j < int_count; ++j) ready_integer += u[j];
  if constexpr (Family == 4) {
#pragma unroll
    for (int j = 0; j < 32; ++j) ready_integer += lds[j];
  }
  ready_gate[tid] = ready_integer ^ uint64_t(__float_as_uint(ready_float));
  __syncthreads();
  if constexpr (Position == 1) {
    // Position is an anchor control. It changes the generated code address, not
    // a claimed fixed byte displacement; actual addresses and the extra CTA
    // barrier preparation are audited in SASS. This is not a physical layout test.
#pragma unroll
    for (int i = 0; i < 24; ++i) asm volatile("bar.sync 0;" ::: "memory");
  }
  R02Stamp stamp{};
  stamp.sm = r_sm();
  // Read another warp's prepared gate after the CTA barrier and make the clock
  // depend on it. Otherwise register-only timestamps can precede the other
  // role's preparation completion and a CTA envelope includes untimed work.
  const uint64_t peer_ready = ready_gate[(tid + 32) % blockDim.x];
  stamp.start = consumer_timer(peer_ready);
  if (!preloaded) {
    if constexpr (Family == 0 && Source == 3) {
      // Both source forms share one kernel and therefore one allocated frame.
      // banked is the runtime selector only in this separate diagnostic mode.
      if (banked) {
#pragma unroll 1
        for (int block=0;block<steps/kUnroll;++block) {
#pragma unroll
          for (int j=0;j<kUnroll;++j)
            asm volatile("fma.rn.f32 %0,%1,%2,%0;" : "+f"(f[j%Pool])
                         : "f"(src[j%8]),"f"(src[8+j%8]) : "memory");
        }
      } else {
#pragma unroll 1
        for (int block=0;block<steps/kUnroll;++block) {
#pragma unroll
          for (int j=0;j<kUnroll;++j)
            asm volatile("fma.rn.f32 %0,%1,%2,%0;" : "+f"(f[j%Pool])
                         : "f"(src[0]),"f"(src[8]) : "memory");
        }
      }
    } else {
#pragma unroll 1
    for (int block = 0; block < steps / kUnroll; ++block) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        if constexpr (Family == 0) {
          const int q = Source == 2 ? j % 8 : 0;
          if constexpr (Source == 0)
            asm volatile("fma.rn.f32 %0,%1,0f3fa00000,%0;"
                         : "+f"(f[j % Pool]) : "f"(src[8]) : "memory");
          else
            asm volatile("fma.rn.f32 %0,%1,%2,%0;"
                         : "+f"(f[j % Pool]) : "f"(src[q]), "f"(src[8 + q]) : "memory");
        } else {
          if constexpr (Family == 1 || mixed) {
            if (active_f)
              asm volatile("add.rn.f32 %0,%0,%1;" : "+f"(f[j % float_count])
                           : "f"(src[8]) : "memory");
          }
          if constexpr (Family == 2 || Family == 3) {
            if (active_b) {
              if constexpr (Family == 3)
                asm volatile("mad.wide.u32 %0,%1,%2,%0;" : "+l"(u[j % int_count])
                             : "r"(unsigned(u[j % int_count])), "r"(b) : "memory");
              else
                asm volatile("add.u64 %0,%0,%1;" : "+l"(u[j % int_count])
                             : "l"(uint64_t(a) * b) : "memory");
            }
          }
          if constexpr (Family == 4) {
            if (active_b) {
              const int q = (j % 8) * 4;
              asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];"
                           : "=r"(lds[q]), "=r"(lds[q+1]), "=r"(lds[q+2]), "=r"(lds[q+3])
                           : "r"(base + (q / 4) * vector_step) : "memory");
            }
          }
        }
      }
    }
    }
  }
  if (intermediate) stamp.progress = timer();
  else stamp.progress = 0;
  // All retained results participate in dependent consumer chains. Integer
  // and FP consumers have separate costs, measured by the preloaded control.
  float fsum = 0;
  uint64_t usum = 0;
#pragma unroll
  for (int j = 0; j < float_count; ++j)
    asm volatile("add.rn.f32 %0,%0,%1;" : "+f"(fsum) : "f"(f[j]) : "memory");
#pragma unroll
  for (int j = 0; j < int_count; ++j)
    asm volatile("add.u64 %0,%0,%1;" : "+l"(usum) : "l"(u[j]) : "memory");
  if constexpr (Family == 4) {
#pragma unroll
    for (int j = 0; j < 32; ++j)
      asm volatile("add.u64 %0,%0,%1;" : "+l"(usum) : "l"(uint64_t(lds[j])) : "memory");
  }
  const uint64_t consumed = usum ^ uint64_t(__float_as_uint(fsum));
  stamp.consumed = consumer_timer(consumed);
  if (lane == 0) stamps[warp] = stamp;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    uint64_t value = 0;
    if constexpr (Family == 2) value = u[j];
    else if constexpr (Family == 3) {
      if (j < 8) value = __float_as_uint(f[j]);
      else if (j < 16) value = u[j - 8];
    } else if constexpr (Family == 4) {
      // The last LDS vector pool has 32 values; floats are checked separately.
      value = lds[j];
    } else value = __float_as_uint(f[j]);
    output[tid * kOutputWords + j] = value;
  }
  if constexpr (Family == 4) {
#pragma unroll
    for (int j = 0; j < 8; ++j) output[tid*kOutputWords + 32 + j] = __float_as_uint(f[j]);
    output[tid*kOutputWords + 48] = consumed;
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      // Keep all source values live throughout the timed interval, including
      // inputs not referenced by constant/reuse producer forms.
      asm volatile("" : "+f"(src[j]) :: "memory");
      output[tid*kOutputWords + 32 + j] = __float_as_uint(src[j]);
    }
  }
  // The checksum remains a checked witness even for non-LDS families.
  if constexpr (Family != 4) output[tid*kOutputWords + 48] = consumed;
}

static uint32_t float_bits(float f) {
  uint32_t value;
  std::memcpy(&value, &f, sizeof(value));
  return value;
}

template <int Family, int Source, int Pool, bool Split, int Position>
int measure(const R02Options& o) {
  constexpr bool mixed = Family >= 3;
  const int threads = mixed ? 64 : o.warps * 32;
  const int fc = Family == 2 ? 0 : mixed ? 8 : Pool;
  const int ic = Family == 2 ? Pool : Family == 3 ? 8 : 0;
  std::vector<float> input(16);
  for (int i = 0; i < 8; ++i) {
    input[i] = 1.25f + (o.witness ? i/64.f : 0.f);
    input[8+i] = 1/128.f + (o.witness ? (i%3)/1024.f : 0.f);
  }
  DeviceBuffer<float> device_input(16);
  CUDA_CHECK(cudaMemcpy(device_input.pointer,input.data(),64,cudaMemcpyHostToDevice));
  DeviceBuffer<uint64_t> output(threads*kOutputWords);
  DeviceBuffer<R02Stamp> stamps(threads/32);
  std::vector<R02Stamp> times(threads/32);
  auto kernel = r02_probe<Family,Source,Pool,Split,Position>;
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
  auto launch = [&]() {
    CUDA_CHECK(cudaMemset(output.pointer,0,output.count*sizeof(uint64_t)));
    kernel<<<1,threads>>>(device_input.pointer,o.steps,o.control==1,bool(o.intermediate),bool(o.banked),stamps.pointer,output.pointer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(times.data(),stamps.pointer,times.size()*sizeof(R02Stamp),cudaMemcpyHostToDevice));
    uint64_t first=UINT64_MAX,last=0;
    for (auto s:times) first=std::min(first,s.start),last=std::max(last,s.consumed);
    return last-first;
  };
  Windows warm=warm_up(launch);
  launch();
  std::vector<uint64_t> values(output.count);
  CUDA_CHECK(cudaMemcpy(values.data(),output.pointer,values.size()*8,cudaMemcpyDeviceToHost));
  size_t errors=0;
  for (int t=0;t<threads;++t) {
    const bool af=!mixed||t/32==0, ab=!mixed||(Split?t/32==1:t/32==0);
    float f[32]{};uint64_t u[32]{};unsigned lds[32]{};
    for (int j=0;j<fc;++j) {
      f[j]=(t+1)/1024.f+j/16.f;
      if constexpr (Family!=0) {
        if (af) for (int n=0;n<o.steps/(fc?fc:1);++n) f[j]+=input[8];
      }
    }
    if constexpr (Family==0) {
      // Replay each operation, independently of the GPU's preload formula.
      for (int n=0;n<o.steps;++n) {
        int j=n%fc,q=(Source==2||(Source==3&&o.banked))?n%8:0;
        f[j]=std::fma(Source==0?1.25f:input[q],input[8+q],f[j]);
      }
    }
    for (int j=0;j<ic;++j) {
      u[j]=(uint64_t((mixed ? t%32 : t)+1)<<32)+j+(Family==3?1:0);
      if (ab) {
        if constexpr (Family==3) {
          for (int n=0;n<o.steps/ic;++n) u[j]+=uint64_t(uint32_t(u[j]))*6;
        } else u[j]+=uint64_t(o.steps/(ic ? ic : 1))*(t+17)*3;
      }
    }
    if constexpr (Family==4)
      if (ab) for (int j=0;j<32;++j) lds[j]=(t%32+1)*256+j;
    float fsum=0;uint64_t usum=0;
    for (int j=0;j<fc;++j) fsum+=f[j];
    for (int j=0;j<ic;++j) usum+=u[j];
    if constexpr (Family==4) for (int j=0;j<32;++j) usum+=lds[j];
    const uint64_t consumed=usum^float_bits(fsum);
    errors+=consumed==0;
    for (int j=0;j<32;++j) {
      uint64_t expected=0;
      if constexpr (Family==2) expected=u[j];
      else if constexpr (Family==3) expected=j<8?float_bits(f[j]):j<16?u[j-8]:0;
      else if constexpr (Family==4) expected=lds[j];
      else expected=float_bits(f[j]);
      errors+=values[t*kOutputWords+j]!=expected;
    }
    if constexpr (Family==4) {
      for (int j=0;j<8;++j) errors+=values[t*kOutputWords+32+j]!=float_bits(f[j]);
      errors+=values[t*kOutputWords+48]!=consumed;
    } else {
      for (int j=0;j<16;++j) errors+=values[t*kOutputWords+32+j]!=float_bits(input[j]);
      errors+=values[t*kOutputWords+48]!=consumed;
    }
  }
  uint64_t first=UINT64_MAX,last_progress=0,last_consumed=0;
  for (auto s:times) {
    errors+=s.sm!=times[0].sm||s.consumed<s.start;
    if (o.intermediate) errors+=s.progress<s.start||s.consumed<s.progress;
    first=std::min(first,s.start);last_progress=std::max(last_progress,s.progress);
    last_consumed=std::max(last_consumed,s.consumed);
  }
  save_binary("output.u64",values);save_binary("input.f32",input);
  std::cout<<std::setprecision(17)<<"{\"family\":"<<Family<<",\"source\":"<<(Source==3?(o.banked?2:1):Source)
           <<",\"pool\":"<<Pool<<",\"split\":"<<(Split?"true":"false")
           <<",\"intermediate\":"<<(o.intermediate?"true":"false")<<",\"position\":"<<Position
           <<",\"witness\":"<<(o.witness?"true":"false")<<",\"banked\":"<<(o.banked?"true":"false")<<",\"threads\":"<<threads<<",\"steps\":"<<o.steps<<",\"unroll\":64"
           <<",\"preloaded\":"<<(o.control?"true":"false")<<",\"registers\":"<<attr.numRegs
           <<",\"local_bytes\":"<<attr.localSizeBytes<<",\"shared_bytes\":"<<attr.sharedSizeBytes
           <<",\"sm\":"<<times[0].sm<<",\"progress_cycles\":"
           <<(o.intermediate?last_progress-first:0)<<",\"consumed_cycles\":"<<last_consumed-first
           <<",\"check_errors\":"<<errors<<",\"warmup_converged\":"
           <<(warm.converged?"true":"false")<<",\"warmup\":";vector_json(warm.warmup);
  std::cout<<",\"stamps\":[";
  for (size_t i=0;i<times.size();++i) {
    auto s=times[i];std::cout<<(i?",":"")<<'['<<s.start<<','<<s.progress<<','<<s.consumed<<','<<s.sm<<']';
  }
  std::cout<<"],\"kernel\":\""<<Family<<'_'<<Source<<'_'<<Pool<<'_'<<Split<<'_'<<Position<<"\"}\n";
  return errors?2:warm.converged?0:3;
}

template <int Family,int Source,int Pool,bool Split>
int controls(const R02Options& o) {
  if (o.position) return measure<Family,Source,Pool,Split,1>(o);
  return measure<Family,Source,Pool,Split,0>(o);
}

int main(int argc,char** argv) {
  try {
    R02Options o(argc,argv);
    if (o.family=="source") {
#ifdef R02_FRAME_DIAG
      if(o.source=="fixed_frame") {
        if(o.chains!=8)throw std::runtime_error("fixed-frame diagnostic requires eight chains");
        return controls<0,3,8,false>(o);
      }
#endif
      const int s=o.source=="constant_multiplier"?0:o.source=="reuse_pair"?1:o.source=="rotate_eight_pairs"?2:-1;
      if (s<0) throw std::runtime_error("unknown source");
#define SOURCE_RUN(S) if(s==S){if(o.chains==1)return controls<0,S,1,false>(o);return controls<0,S,8,false>(o);}
      SOURCE_RUN(0) SOURCE_RUN(1) SOURCE_RUN(2)
#undef SOURCE_RUN
    }
    if (o.family=="fadd_f32") {
      if(o.pool==8)return controls<1,1,8,false>(o);return controls<1,1,32,false>(o);
    }
    if (o.family=="integer_u64") {
      if(o.pool==8)return controls<2,1,8,false>(o);return controls<2,1,32,false>(o);
    }
    if(o.family=="mixed_return") {
      if(o.pair=="fadd_imad_wide") {
        if(o.split)return controls<3,1,8,true>(o);return controls<3,1,8,false>(o);
      }
      if(o.pair=="fadd_lds128") {
        if(o.split)return controls<4,1,8,true>(o);return controls<4,1,8,false>(o);
      }
    }
    throw std::runtime_error("unknown family/pair");
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

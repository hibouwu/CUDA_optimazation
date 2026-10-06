#pragma once
// Shared host/device helpers for the R01 and R06 probes.
#include "r00_common.hpp"

#include <fstream>
#include <map>
#include <numeric>
#include <random>

// One CTA's timed window: globaltimer ns, clock64 cycles and first/last SM id.
struct RStamp {
  uint64_t begin_ns, end_ns, begin_cycle, end_cycle;
  unsigned first_sm, last_sm;
};

__device__ inline uint64_t r_ns() {
  uint64_t x;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(x));
  return x;
}

__device__ inline unsigned r_sm() {
  unsigned x;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(x));
  return x;
}

__device__ inline void stamp_begin(RStamp& s) {
  s.first_sm = r_sm();
  s.begin_ns = r_ns();
  s.begin_cycle = clock64();
}

__device__ inline void stamp_end(RStamp& s, RStamp* out) {
  s.end_cycle = clock64();
  s.end_ns = r_ns();
  s.last_sm = r_sm();
  *out = s;
}

// Single CTA: clock64 difference of CTA 0. Whole GPU: globaltimer envelope of all CTAs.
inline uint64_t elapsed_window(const std::vector<RStamp>& times, bool full = false) {
  if (!full) return times[0].end_cycle - times[0].begin_cycle;
  uint64_t first = UINT64_MAX, last = 0;
  for (const auto& s : times) {
    first = std::min(first, s.begin_ns);
    last = std::max(last, s.end_ns);
  }
  return last - first;
}

struct Windows {
  std::vector<float> warmup;
  bool converged = false;
};

// Warmup protocol shared by all R01/R06 points: 8-30 windows, stop once the last
// five have CV <= 2%; the caller then runs one formal window.
template <class Launch>
Windows warm_up(Launch launch) {
  Windows w;
  for (int i = 0; i < 30; ++i) {
    w.warmup.push_back(float(launch()));
    if (i >= 7 && coefficient_of_variation({w.warmup.end() - 5, w.warmup.end()}) <= 0.02) {
      w.converged = true;
      break;
    }
  }
  return w;
}

inline void stamp_json(const std::vector<RStamp>& times) {
  std::cout << '[';
  for (size_t i = 0; i < times.size(); ++i) {
    const auto& s = times[i];
    if (i) std::cout << ',';
    std::cout << '[' << s.begin_ns << ',' << s.end_ns << ',' << s.begin_cycle << ','
              << s.end_cycle << ',' << s.first_sm << ',' << s.last_sm << ']';
  }
  std::cout << ']';
}

inline void vector_json(const std::vector<float>& x) {
  std::cout << '[';
  for (size_t i = 0; i < x.size(); ++i) std::cout << (i ? "," : "") << x[i];
  std::cout << ']';
}

inline void windows_json(const Windows& w, const std::vector<RStamp>& times, bool full = false) {
  std::cout << ",\"elapsed\":" << elapsed_window(times, full)
            << ",\"warmup_converged\":" << (w.converged ? "true" : "false") << ",\"warmup\":";
  vector_json(w.warmup);
  std::cout << ",\"stamps\":";
  stamp_json(times);
}

template <class T>
inline void save_binary(const char* name, const std::vector<T>& x) {
  std::ofstream f(name, std::ios::binary);
  f.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(T));
}

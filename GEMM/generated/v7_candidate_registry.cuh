#pragma once

#include "sgemm_kernels.cuh"

#include <vector>

using V7KernelPtr = void (*)(int, int, int, float, const float*, const float*,
                             float, float*);

struct V7ProbeCandidate {
  const char* candidate_id;
  int BM;
  int BN;
  int BK;
  int WM;
  int WN;
  int TM;
  int TN;
  int WNITER;
  int WMITER;
  int WSUBM;
  int WSUBN;
  int NumThreads;
  int warps_per_block;
  int accumulators;
  int reg_m;
  int reg_n;
  int R_static;
  int shared_bytes_model;
  V7KernelPtr kernel;
};

inline std::vector<V7ProbeCandidate> v7_probe_candidates() {
  return {
    {
        "v7_baseline_128x128x16_warp64x64_micro8x4",
        128, 128, 16,
        64, 64, 8, 4,
        4, 1, 64, 16,
        128, 4,
        128, 8, 16,
        152, 32768,
        sgemm_v7_warp_tiling_double_buffer<
            128, 128, 16,
            64, 64, 4,
            8, 4, 128>
    },
    {
        "v7_bk8_128x128x8_warp64x64_micro8x4",
        128, 128, 8,
        64, 64, 8, 4,
        4, 1, 64, 16,
        128, 4,
        128, 8, 16,
        152, 16384,
        sgemm_v7_warp_tiling_double_buffer<
            128, 128, 8,
            64, 64, 4,
            8, 4, 128>
    },
    {
        "v7_acc64_threads256_warp32x64_micro4x4",
        128, 128, 16,
        32, 64, 4, 4,
        2, 2, 16, 32,
        256, 8,
        64, 8, 8,
        80, 32768,
        sgemm_v7_warp_tiling_double_buffer<
            128, 128, 16,
            32, 64, 2,
            4, 4, 256>
    },
    {
        "v7_m_wide_warp128x32_micro8x4",
        128, 128, 16,
        128, 32, 8, 4,
        2, 2, 64, 16,
        128, 4,
        128, 16, 8,
        152, 32768,
        sgemm_v7_warp_tiling_double_buffer<
            128, 128, 16,
            128, 32, 2,
            8, 4, 128>
    },
    {
        "v7_n_wide_warp32x128_micro4x8",
        128, 128, 16,
        32, 128, 4, 8,
        4, 1, 32, 32,
        128, 4,
        128, 4, 32,
        164, 32768,
        sgemm_v7_warp_tiling_double_buffer<
            128, 128, 16,
            32, 128, 4,
            4, 8, 128>
    },
    {
        "v7_micro_layout_4x4_warp64x64",
        128, 128, 16,
        64, 64, 4, 4,
        4, 2, 32, 16,
        128, 4,
        128, 8, 16,
        152, 32768,
        sgemm_v7_warp_tiling_double_buffer<
            128, 128, 16,
            64, 64, 4,
            4, 4, 128>
    }
  };
}

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_model import CandidateParams, derive_candidate


SELECTED = [
    ("v7_baseline_128x128x16_warp64x64_micro8x4", CandidateParams(128, 128, 16, 64, 64, 8, 4, 4)),
    ("v7_bk8_128x128x8_warp64x64_micro8x4", CandidateParams(128, 128, 8, 64, 64, 8, 4, 4)),
    ("v7_acc64_threads256_warp32x64_micro4x4", CandidateParams(128, 128, 16, 32, 64, 4, 4, 2)),
    ("v7_m_wide_warp128x32_micro8x4", CandidateParams(128, 128, 16, 128, 32, 8, 4, 2)),
    ("v7_n_wide_warp32x128_micro4x8", CandidateParams(128, 128, 16, 32, 128, 4, 8, 4)),
    ("v7_micro_layout_4x4_warp64x64", CandidateParams(128, 128, 16, 64, 64, 4, 4, 4)),
]


def registry_entry(name: str, candidate) -> str:
    return f"""    {{
        "{name}",
        {candidate.BM}, {candidate.BN}, {candidate.BK},
        {candidate.WM}, {candidate.WN}, {candidate.TM}, {candidate.TN},
        {candidate.WNITER}, {candidate.WMITER}, {candidate.WSUBM}, {candidate.WSUBN},
        {candidate.NumThreads}, {candidate.warps_per_block},
        {candidate.accumulators}, {candidate.reg_m}, {candidate.reg_n},
        {candidate.R_static}, {candidate.shared_bytes_model},
        sgemm_v7_warp_tiling_double_buffer<
            {candidate.BM}, {candidate.BN}, {candidate.BK},
            {candidate.WM}, {candidate.WN}, {candidate.WNITER},
            {candidate.TM}, {candidate.TN}, {candidate.NumThreads}>
    }}"""


def main() -> int:
    out = Path(__file__).resolve().parents[2] / "GEMM" / "generated" / "v7_candidate_registry.cuh"
    entries = []
    for name, params in SELECTED:
        candidate = derive_candidate(params)
        if not candidate.valid:
            raise RuntimeError(f"{name} invalid: {candidate.rejection_stage}:{candidate.rejection_reason}")
        entries.append(registry_entry(name, candidate))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        """#pragma once

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
"""
        + ",\n".join(entries)
        + """
  };
}
"""
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

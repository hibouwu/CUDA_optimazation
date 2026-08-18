#!/usr/bin/env python3
"""Bind guide claims to the pinned CUTLASS source, not to remembered APIs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def require(text: str, pattern: str, label: str) -> None:
    if not re.search(pattern, text, re.MULTILINE | re.DOTALL):
        raise AssertionError(f"missing {label}: {pattern}")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    cutlass = root / "third_party" / "cutlass"
    block_collective = (cutlass / "include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp").read_text()
    dense_collective = (cutlass / "include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp").read_text()
    copy_wrapper = (cutlass / "include/cute/arch/copy_sm100.hpp").read_text()
    mma_wrapper = (cutlass / "include/cute/arch/mma_sm100_umma.hpp").read_text()
    tma_wrapper = (cutlass / "include/cute/arch/copy_sm90_tma.hpp").read_text()

    require(block_collective, r"SM100_UTCCP_4x32dp128bit_1cta", "1CTA scale UTCCP selection")
    require(block_collective, r"copy\([^;]*SFA", "SFA SMEM-to-TMEM copy")
    require(block_collective, r"copy\([^;]*SFB", "SFB SMEM-to-TMEM copy")
    require(copy_wrapper, r"tcgen05\.cp\.cta_group::1\.32x128b\.warpx4", "tcgen05.cp wrapper")
    require(mma_wrapper, r"tcgen05\.mma\.cta_group::1\.kind::mxf8f6f4\.block_scale", "MXFP8 block-scale MMA wrapper")
    require(tma_wrapper, r"cp\.async\.bulk\.tensor", "TMA PTX wrapper")
    require(dense_collective, r"cute::gemm|gemm\(", "dense cute::gemm path")
    print("CUTLASS_SOURCE_CONTRACTS_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"CUTLASS_SOURCE_CONTRACTS_FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


WARP_SIZE = 32
FLOAT_SIZE_BYTES = 4
STATIC_SMEM_LIMIT_BYTES = 49152
MAX_THREADS_PER_BLOCK = 1024
FIRST_ROUND_THREAD_COUNTS = {128, 256}
FIRST_ROUND_MAX_ACCUMULATORS = 128

MAPPING_CORRECTNESS = "mapping_correctness"
LAUNCH_FEASIBILITY = "launch_feasibility"
IMPLEMENTATION_CONSTRAINT = "implementation_constraint"
RESOURCE_FEASIBILITY = "resource_feasibility"
SEARCH_SPACE_POLICY = "search_space_policy"
PERFORMANCE_HYPOTHESIS = "performance_hypothesis"


@dataclass(frozen=True)
class CandidateParams:
    BM: int
    BN: int
    BK: int
    WM: int
    WN: int
    TM: int
    TN: int
    WNITER: int


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    BM: int
    BN: int
    BK: int
    WM: int
    WN: int
    TM: int
    TN: int
    WNITER: int
    WMITER: int | None = None
    WSUBM: int | None = None
    WSUBN: int | None = None
    NumThreads: int | None = None
    warps_per_block: int | None = None
    accumulators: int | None = None
    reg_m: int | None = None
    reg_n: int | None = None
    R_static: int | None = None
    A_register_reuse: int | None = None
    B_register_reuse: int | None = None
    combined_register_reuse: float | None = None
    shared_bytes_model: int | None = None
    valid: bool = False
    rejection_stage: str = ""
    rejection_reason: str = ""

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def candidate_id(params: CandidateParams) -> str:
    return (
        f"bm{params.BM}_bn{params.BN}_bk{params.BK}_"
        f"wm{params.WM}_wn{params.WN}_tm{params.TM}_tn{params.TN}_"
        f"wniter{params.WNITER}"
    )


def reject(params: CandidateParams, stage: str, reason: str, **derived: object) -> Candidate:
    return Candidate(
        candidate_id=candidate_id(params),
        BM=params.BM,
        BN=params.BN,
        BK=params.BK,
        WM=params.WM,
        WN=params.WN,
        TM=params.TM,
        TN=params.TN,
        WNITER=params.WNITER,
        valid=False,
        rejection_stage=stage,
        rejection_reason=reason,
        **derived,
    )


def validate_lane_coverage(WSUBM: int, WSUBN: int, TM: int, TN: int) -> bool:
    if TM == 0 or TN == 0:
        return False
    if WSUBM % TM != 0 or WSUBN % TN != 0:
        return False
    return (WSUBM // TM) * (WSUBN // TN) == WARP_SIZE


def positive_template_params(params: CandidateParams) -> bool:
    return all(
        value > 0
        for value in (
            params.BM,
            params.BN,
            params.BK,
            params.WM,
            params.WN,
            params.TM,
            params.TN,
            params.WNITER,
        )
    )


def derive_candidate(
    params: CandidateParams,
    *,
    max_threads_per_block: int = MAX_THREADS_PER_BLOCK,
    static_smem_limit_bytes: int = STATIC_SMEM_LIMIT_BYTES,
    allowed_thread_counts: set[int] | None = FIRST_ROUND_THREAD_COUNTS,
    max_accumulators: int | None = FIRST_ROUND_MAX_ACCUMULATORS,
) -> Candidate:
    if not positive_template_params(params):
        return reject(params, MAPPING_CORRECTNESS, "non_positive_template_parameter")
    if params.BM % params.WM != 0:
        return reject(params, MAPPING_CORRECTNESS, "BM_not_divisible_by_WM")
    if params.BN % params.WN != 0:
        return reject(params, MAPPING_CORRECTNESS, "BN_not_divisible_by_WN")

    warps_per_block = (params.BM // params.WM) * (params.BN // params.WN)
    num_threads = WARP_SIZE * warps_per_block
    basic = {
        "NumThreads": num_threads,
        "warps_per_block": warps_per_block,
    }
    if num_threads % WARP_SIZE != 0:
        return reject(params, MAPPING_CORRECTNESS, "NumThreads_not_multiple_of_32", **basic)
    if num_threads > max_threads_per_block:
        return reject(params, LAUNCH_FEASIBILITY, "NumThreads_exceeds_max_threads_per_block", **basic)

    denom = WARP_SIZE * params.TM * params.TN * params.WNITER
    numerator = params.WM * params.WN
    if numerator % denom != 0:
        return reject(params, MAPPING_CORRECTNESS, "WMITER_not_integer", **basic)
    wmiter = numerator // denom
    if wmiter < 1:
        return reject(params, MAPPING_CORRECTNESS, "WMITER_less_than_1", WMITER=wmiter, **basic)
    if params.WM % wmiter != 0:
        return reject(params, MAPPING_CORRECTNESS, "WM_not_divisible_by_WMITER", WMITER=wmiter, **basic)
    if params.WN % params.WNITER != 0:
        return reject(params, MAPPING_CORRECTNESS, "WN_not_divisible_by_WNITER", WMITER=wmiter, **basic)

    wsubm = params.WM // wmiter
    wsubn = params.WN // params.WNITER
    layout = {"WMITER": wmiter, "WSUBM": wsubm, "WSUBN": wsubn, **basic}
    if wsubm % params.TM != 0:
        return reject(params, MAPPING_CORRECTNESS, "WSUBM_not_divisible_by_TM", **layout)
    if wsubn % params.TN != 0:
        return reject(params, MAPPING_CORRECTNESS, "WSUBN_not_divisible_by_TN", **layout)
    if not validate_lane_coverage(wsubm, wsubn, params.TM, params.TN):
        return reject(params, MAPPING_CORRECTNESS, "lane_coverage_not_32", **layout)

    accumulators = numerator // WARP_SIZE
    reg_m = wmiter * params.TM
    reg_n = params.WNITER * params.TN
    r_static = accumulators + reg_m + reg_n
    shared_bytes = 2 * params.BK * (params.BM + params.BN) * FLOAT_SIZE_BYTES
    combined_reuse = accumulators / (reg_m + reg_n)
    derived = {
        **layout,
        "accumulators": accumulators,
        "reg_m": reg_m,
        "reg_n": reg_n,
        "R_static": r_static,
        "A_register_reuse": reg_n,
        "B_register_reuse": reg_m,
        "combined_register_reuse": combined_reuse,
        "shared_bytes_model": shared_bytes,
    }

    if params.BK % 4 != 0:
        return reject(params, IMPLEMENTATION_CONSTRAINT, "BK_not_multiple_of_4", **derived)
    if params.BN % 4 != 0:
        return reject(params, IMPLEMENTATION_CONSTRAINT, "BN_not_multiple_of_4", **derived)
    if params.TN % 4 != 0:
        return reject(params, IMPLEMENTATION_CONSTRAINT, "TN_not_multiple_of_4", **derived)
    if (num_threads * 4) % params.BK != 0:
        return reject(params, IMPLEMENTATION_CONSTRAINT, "A_copy_row_stride_not_integer", **derived)
    if (num_threads * 4) % params.BN != 0:
        return reject(params, IMPLEMENTATION_CONSTRAINT, "B_copy_row_stride_not_integer", **derived)
    if (params.BM * params.BK) % (4 * num_threads) != 0:
        return reject(params, IMPLEMENTATION_CONSTRAINT, "A_copy_remainder", **derived)
    if (params.BN * params.BK) % (4 * num_threads) != 0:
        return reject(params, IMPLEMENTATION_CONSTRAINT, "B_copy_remainder", **derived)

    if allowed_thread_counts is not None and num_threads not in allowed_thread_counts:
        return reject(params, SEARCH_SPACE_POLICY, "NumThreads_not_in_first_round_set", **derived)
    if max_accumulators is not None and accumulators > max_accumulators:
        return reject(params, SEARCH_SPACE_POLICY, "accumulators_exceed_first_round_policy", **derived)
    if shared_bytes > static_smem_limit_bytes:
        return reject(params, RESOURCE_FEASIBILITY, "static_shared_memory_exceeds_49152", **derived)

    return Candidate(
        candidate_id=candidate_id(params),
        BM=params.BM,
        BN=params.BN,
        BK=params.BK,
        WM=params.WM,
        WN=params.WN,
        TM=params.TM,
        TN=params.TN,
        WNITER=params.WNITER,
        valid=True,
        **derived,
    )


def enumerate_first_round_params() -> Iterable[CandidateParams]:
    for BM in (64, 128, 256):
        for BN in (64, 128, 256):
            for BK in (8, 16, 32):
                for WM in (32, 64, 128):
                    for WN in (32, 64, 128):
                        for TM, TN in ((4, 4), (8, 4), (4, 8), (8, 8)):
                            for WNITER in (1, 2, 4, 8):
                                yield CandidateParams(BM, BN, BK, WM, WN, TM, TN, WNITER)


CSV_FIELDS = [
    "candidate_id",
    "valid",
    "rejection_stage",
    "rejection_reason",
    "BM",
    "BN",
    "BK",
    "WM",
    "WN",
    "TM",
    "TN",
    "WNITER",
    "WMITER",
    "WSUBM",
    "WSUBN",
    "NumThreads",
    "warps_per_block",
    "accumulators",
    "reg_m",
    "reg_n",
    "R_static",
    "A_register_reuse",
    "B_register_reuse",
    "combined_register_reuse",
    "shared_bytes_model",
]

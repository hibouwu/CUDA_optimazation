"""Unfitted, ideal FP16/BF16 -> FP32 GEMM baseline (alpha=1, beta=0)."""
import argparse
import json
import math
from pathlib import Path

PROFILES = Path(__file__).resolve().parents[3] / "Docs/ModelEvaluation/gemm/hardware/gh200_sm90/spec_baseline_profiles.json"


def calculate(m, n, k, compute_tflops, bandwidth_gbs):
    for name, value in (("m", m), ("n", n), ("k", k)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("compute_tflops", compute_tflops), ("bandwidth_gbs", bandwidth_gbs)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    flops = 2 * m * n * k
    input_bytes = 2 * (m * k + k * n)
    output_bytes = 4 * m * n
    compute_us = flops / (compute_tflops * 1e6)
    memory_us = (input_bytes + output_bytes) / (bandwidth_gbs * 1e3)
    return dict(m=m, n=n, k=k, flops=flops, input_bytes=input_bytes,
                output_bytes=output_bytes, logical_bytes=input_bytes + output_bytes,
                compute_us=compute_us, memory_us=memory_us,
                ideal_us=max(compute_us, memory_us),
                limiting_term="compute" if compute_us >= memory_us else "memory")


def profile_rates(profile):
    if "compute_tflops" in profile:
        compute = profile["compute_tflops"]
    else:
        compute = profile["sm_count"] * profile["flops_per_sm_cycle"] * profile["reference_sm_ghz"] / 1000
    return compute, profile["bandwidth_gbs"]


def main():
    profiles = json.loads(PROFILES.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    for dim in ("m", "n", "k"):
        parser.add_argument(f"--{dim}", type=int, required=True)
    parser.add_argument("--profile", choices=profiles, default="gh200_product")
    args = parser.parse_args()
    profile = profiles[args.profile]
    compute, bandwidth = profile_rates(profile)
    try:
        result = calculate(args.m, args.n, args.k, compute, bandwidth)
    except ValueError as error:
        parser.error(str(error))
    result.update(profile=args.profile, compute_tflops=compute, bandwidth_gbs=bandwidth,
                  provenance=profile, assumptions=[
                      "Dense FP16/BF16 inputs, FP32 output, alpha=1, beta=0",
                      "Each logical input read once and output written once through HBM",
                      "Perfect overlap, no launch, padding, wave or synchronization costs",
                      "Conditional ideal reference; not a strict bound for warm-L2 execution"])
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

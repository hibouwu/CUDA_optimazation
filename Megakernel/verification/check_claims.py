"""CPU checks of published table arithmetic and mathematical claim boundaries.

This does not execute either paper's GPU kernels. Table 2 was transcribed
from ForgeMegakernel v1, PDF p. 10, and visually checked against the PDF.
Run: python verification/check_claims.py
"""

import json
import math
from pathlib import Path


def geometric_mean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def main():
    # model, batch, context, generated_us, baseline_us, generated_acc, baseline_acc
    rows = [
        ("Qwen3-0.6B", 8, 2048, 1406, 1576, .560, .555),
        ("Qwen3-0.6B", 16, 1024, 1457, 1709, .555, .555),
        ("Qwen3-0.6B", 16, 4096, 1322, 1688, .545, .550),
        ("Qwen3-4B", 1, 128, 3387, 3804, .935, .930),
        ("Qwen3-4B", 8, 1024, 4217, 4459, .940, .930),
        ("Qwen3-8B", 1, 128, 5601, 6009, .965, .975),
        ("Qwen3-8B", 1, 512, 5757, 6116, .965, .970),
        ("Qwen3-8B", 1, 2048, 5847, 6219, .970, .970),
        ("Llama-3.1-8B", 4, 2048, 6130, 6378, .880, .865),
        ("Llama-2-13B-chat", 1, 128, 9498, 9761, .315, .315),
        ("MiniCPM5-1B", 1, 128, 1181, 1430, .550, .555),
        ("MiniCPM4-1B", 1, 128, 1718, 1970, .640, .650),
        ("MiniCPM4-3B", 1, 128, 3145, 3394, .620, .635),
    ]
    speedups = [r[4] / r[3] for r in rows]
    table_result = {
        "rows": len(rows),
        "all_listed_rows_faster": all(x > 1 for x in speedups),
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "speedup_geometric_mean": geometric_mean(speedups),
        "accuracy_equal_or_higher_count": sum(r[5] >= r[6] for r in rows),
        "maximum_absolute_accuracy_gap_percentage_points": max(abs(r[5]-r[6])*100 for r in rows),
        "sample_size": 200,
        "one_question_percentage_points": 100 / 200,
        "data": [dict(zip(["model", "batch", "context", "generated_us", "baseline_us", "generated_accuracy", "baseline_accuracy"], r)) for r in rows],
    }
    assert round(table_result["speedup_geometric_mean"], 2) == 1.11
    assert table_result["accuracy_equal_or_higher_count"] == 7

    # Forge Eq. 6 checks one projection, not all output error directions.
    # Construct an error orthogonal to the narrowed-reference direction.
    reference = [0.0, 0.0]
    narrowed = [1.0, 0.0]
    candidate = [0.0, 100.0]
    direction = [b-a for a, b in zip(reference, narrowed)]
    error = [k-a for k, a in zip(candidate, reference)]
    alpha = sum(e*d for e, d in zip(error, direction)) / sum(d*d for d in direction)
    projection_result = {
        "reference": reference, "narrowed_reference": narrowed,
        "candidate": candidate, "alpha": alpha,
        "passes_alpha_only": alpha <= .5,
        "l2_error": math.sqrt(sum(e*e for e in error)),
        "interpretation": "The projection test alone is not a norm bound or a proof of arithmetic width. This is not a bypass of the full oracle.",
    }
    assert alpha == 0 and projection_result["l2_error"] == 100

    # Same invented elapsed time and sequence length, different denominators.
    # This isolates the batch factor visible in the frozen artifact scripts.
    elapsed_ms, sequence_length, batch = 1088.0, 1088, 8
    accounting_result = {
        "kind": "synthetic arithmetic example, not GPU data",
        "elapsed_ms": elapsed_ms, "sequence_length": sequence_length, "batch": batch,
        "mpk_demo_formula_ms": elapsed_ms / sequence_length,
        "sglang_wrapper_formula_ms": elapsed_ms / (sequence_length * batch),
        "ratio_for_identical_elapsed_time": batch,
    }
    result = {
        "run_kind": "CPU arithmetic and counterexample checks",
        "gpu_kernel_executed": False,
        "forge_table2": table_result,
        "forge_precision_projection_boundary": projection_result,
        "mpk_artifact_denominator_example": accounting_result,
        "checks_completed": True,
    }
    dest = Path(__file__).with_name("results.json")
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k:v for k,v in table_result.items() if k != "data"}, indent=2))
    print("alpha-only counterexample: alpha=0, L2 error=100")
    print("synthetic identical-time denominator ratio: 8")
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()

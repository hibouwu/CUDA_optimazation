#!/usr/bin/env python3
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def shape_key(row):
    return (
        row.get("Category", ""),
        row.get("M", ""),
        row.get("N", ""),
        row.get("K", ""),
        row.get("Note", ""),
        row.get("Epilogue", "none"),
    )


def parse_backend_set(value):
    return {item.strip() for item in value.split(",") if item.strip()}


def write_analysis(rows, output_path, min_ratio, reference_backends):
    grouped = defaultdict(list)
    for row in rows:
        grouped[shape_key(row)].append(row)

    threshold_label = f"{min_ratio:.2f}"
    reference_label = ", ".join(sorted(reference_backends)) or "none"
    lines = [
        "# GEMM Shape Sweep Analysis",
        "",
        "本文件由 `GEMMShapeOpt/scripts/analyze_shape_sweep.py` 生成。",
        "RatioToReference 使用每次 benchmark CSV 中的 reference 字段；当前源码目标是 cuBLASLt Matmul heuristic。",
        f"90% gate 使用非 reference backend；reference backend: `{reference_label}`，min ratio: `{threshold_label}`。",
        "",
        "## Summary",
        "",
        "| Category | M | N | K | Epilogue | Best non-reference backend | GFLOP/s | Ratio | Gate | Candidate failed/unmatched | Note |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | --- |",
    ]

    gate_failures = []
    for key in sorted(grouped.keys(), key=lambda item: (item[0], int(item[1]), int(item[2]), int(item[3]), item[5])):
        rows_for_shape = grouped[key]
        candidate_rows = [
            row for row in rows_for_shape
            if row.get("BackendId", "") not in reference_backends
        ]
        matched_candidates = [
            row for row in candidate_rows
            if row.get("Matched") == "1" and row.get("Status") == "ok"
        ]
        failed_candidates = [
            row for row in candidate_rows
            if row.get("Status") != "ok" or row.get("Matched") != "1"
        ]
        if matched_candidates:
            best = max(matched_candidates, key=lambda row: to_float(row.get("GFLOPS")))
            best_backend = best.get("BackendId", "")
            best_gflops = f"{to_float(best.get('GFLOPS')):.1f}"
            best_ratio_value = to_float(best.get("RatioToReference"))
            best_ratio = f"{best_ratio_value:.3f}"
        else:
            best_backend = "none"
            best_gflops = "0.0"
            best_ratio_value = 0.0
            best_ratio = "0.000"
        gate = "PASS" if best_ratio_value >= min_ratio else "FAIL"
        category, m, n, k, note, epilogue = key
        if gate == "FAIL":
            gate_failures.append((category, m, n, k, epilogue, note, best_backend, best_ratio))
        lines.append(
            f"| {category} | {m} | {n} | {k} | {epilogue} | {best_backend} | "
            f"{best_gflops} | {best_ratio} | {gate} | "
            f"{len(failed_candidates)} | {note} |"
        )

    if gate_failures:
        lines.extend([
            "",
            f"## Gate Failures (< {threshold_label}x cuBLASLt)",
            "",
            "| Category | M | N | K | Epilogue | Best backend | Ratio | Note |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
        ])
        for category, m, n, k, epilogue, note, best_backend, best_ratio in gate_failures:
            lines.append(
                f"| {category} | {m} | {n} | {k} | {epilogue} | {best_backend} | "
                f"{best_ratio} | {note} |"
            )
    else:
        lines.extend([
            "",
            f"## Gate Failures (< {threshold_label}x cuBLASLt)",
            "",
            "None.",
        ])

    lines.extend([
        "",
        "## Next Actions",
        "",
        "- 如果 skinny/GEMV-like shape 的最佳 backend 仍是 cuBLASLt/CUTLASS，优先实现专用 vector/tile kernel，而不是复用 square GEMM tile。",
        "- 如果 ragged shape 主要输在 boundary cleanup，需要把 fast tile 和 tail tile 分开计时。",
        "- 如果 `tc5a/tc5b` 在非规则 shape 上返回 unavailable，不把 0 值当性能点；先补合法 tile/tail 路径。",
    ])

    output_path.write_text("\n".join(lines) + "\n")
    return gate_failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-ratio", default=0.90, type=float)
    parser.add_argument("--reference-backends", default="cublas_tc")
    parser.add_argument("--fail-under", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures = write_analysis(
        rows,
        args.out,
        args.min_ratio,
        parse_backend_set(args.reference_backends),
    )
    if args.fail_under and failures:
        print(
            f"{len(failures)} shape(s) failed the {args.min_ratio:.2f} "
            "non-reference performance gate.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

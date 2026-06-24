#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from candidate_model import CSV_FIELDS, derive_candidate, enumerate_first_round_params


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def shared_bucket(shared_bytes: int | None) -> str:
    if shared_bytes is None:
        return "unknown"
    if shared_bytes <= 24 * 1024:
        return "<=24KiB"
    if shared_bytes <= 32 * 1024:
        return "<=32KiB"
    if shared_bytes <= 48 * 1024:
        return "<=48KiB"
    return ">48KiB"


def write_summary(path: Path, candidates) -> None:
    valid = [c for c in candidates if c.valid]
    rejected = [c for c in candidates if not c.valid]
    reasons = Counter(c.rejection_reason for c in rejected)
    stages = Counter(c.rejection_stage for c in rejected)
    threads = Counter(c.NumThreads for c in valid)
    accumulators = Counter(c.accumulators for c in valid)
    smem_buckets = Counter(shared_bucket(c.shared_bytes_model) for c in valid)

    lines = [
        "# v7 Candidate Generation Summary",
        "",
        f"- Raw combinations: {len(candidates)}",
        f"- Valid candidates: {len(valid)}",
        f"- Rejected candidates: {len(rejected)}",
        "",
        "## Rejection Stages",
        "",
    ]
    for stage, count in sorted(stages.items()):
        lines.append(f"- {stage}: {count}")

    lines.extend(["", "## Rejection Reasons", ""])
    for reason, count in sorted(reasons.items()):
        lines.append(f"- {reason}: {count}")

    lines.extend(["", "## Valid Candidate Thread Counts", ""])
    for thread_count in (128, 256):
        lines.append(f"- {thread_count} threads: {threads.get(thread_count, 0)}")

    lines.extend(["", "## Valid Candidate Accumulator Counts", ""])
    for acc in sorted(k for k in accumulators if k is not None):
        lines.append(f"- {acc} accumulators: {accumulators[acc]}")

    lines.extend(["", "## Valid Candidate Shared Memory Buckets", ""])
    for bucket in ("<=24KiB", "<=32KiB", "<=48KiB", ">48KiB", "unknown"):
        if smem_buckets.get(bucket, 0):
            lines.append(f"- {bucket}: {smem_buckets[bucket]}")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    candidates = [derive_candidate(params) for params in enumerate_first_round_params()]
    all_rows = [c.to_row() for c in candidates]
    valid_rows = [c.to_row() for c in candidates if c.valid]
    rejected_rows = [c.to_row() for c in candidates if not c.valid]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "all_candidates.csv", all_rows)
    write_csv(args.out_dir / "valid_candidates.csv", valid_rows)
    write_csv(args.out_dir / "rejected_candidates.csv", rejected_rows)
    write_summary(args.out_dir / "candidate_generation_summary.md", candidates)

    print(f"raw={len(candidates)} valid={len(valid_rows)} rejected={len(rejected_rows)}")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

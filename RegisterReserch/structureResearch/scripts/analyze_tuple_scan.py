#!/usr/bin/env python3
import argparse
import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FAMILIES = {
    "lop3": ROOT / "results" / "tuple_scan_lop3",
    "imad": ROOT / "results" / "tuple_scan_imad",
}


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize_family(family, directory):
    manifest = directory / "manifest.csv"
    results = directory / "results.csv"
    if not manifest.exists() or not results.exists():
        print(f"{family}: missing manifest/results")
        return 1

    result_by_case = {row["case"]: row for row in load_rows(results)}
    rows = []
    for metadata in load_rows(manifest):
        result = result_by_case.get(metadata["case"])
        if result is None:
            raise SystemExit(f"{family}: missing result for {metadata['case']}")
        cycles = float(result["median_cycles_per_op"])
        rows.append({**metadata, **result, "cycles": cycles})

    fast = [row["cycles"] for row in rows if int(row["max_mod2"]) <= 2]
    slow = [row["cycles"] for row in rows if int(row["max_mod2"]) >= 3]
    if not fast or not slow:
        threshold = statistics.median(row["cycles"] for row in rows)
    else:
        threshold = (statistics.mean(fast) + statistics.mean(slow)) / 2.0

    print(f"\n{family}: {len(rows)} cases, slow threshold {threshold:.6f} cycles/op")
    for max_mod2 in (2, 3):
        values = [row["cycles"] for row in rows if int(row["max_mod2"]) == max_mod2]
        if values:
            print(
                f"  max_mod2={max_mod2}: n={len(values)}, "
                f"mean={statistics.mean(values):.6f}, "
                f"min={min(values):.6f}, max={max(values):.6f}"
            )

    for modulo in (2, 4, 8, 16):
        correct = 0
        for row in rows:
            predicted_slow = int(row[f"max_mod{modulo}"]) >= 3
            observed_slow = row["cycles"] >= threshold
            correct += int(predicted_slow == observed_slow)
        print(f"  mod{modulo} model accuracy: {correct}/{len(rows)} = {correct / len(rows) * 100:.1f}%")

    print("  designed cases:")
    for row in rows:
        if row["category"] == "designed":
            print(
                f"    {row['case']}: {row['tuple']} -> "
                f"{row['cycles']:.6f} c/op"
            )
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", nargs="?", choices=["all", *sorted(FAMILIES)], default="all")
    args = parser.parse_args()

    families = sorted(FAMILIES) if args.family == "all" else [args.family]
    status = 0
    for family in families:
        status |= summarize_family(family, FAMILIES[family])
    raise SystemExit(status)


if __name__ == "__main__":
    main()

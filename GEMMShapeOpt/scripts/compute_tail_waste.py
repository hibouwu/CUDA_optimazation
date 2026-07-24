#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def ceil_div(a, b):
    return (a + b - 1) // b


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Compute tile-tail waste for GEMM shapes.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tile-m", default=128, type=int)
    parser.add_argument("--tile-n", default=256, type=int)
    parser.add_argument("--tile-k", default=64, type=int)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Category",
            "M",
            "N",
            "K",
            "TileM",
            "TileN",
            "TileK",
            "PaddedM",
            "PaddedN",
            "PaddedK",
            "OutputElements",
            "PaddedOutputElements",
            "OutputWastePct",
            "ComputeVolume",
            "PaddedComputeVolume",
            "ComputeWastePct",
            "Mtail",
            "Ntail",
            "Ktail",
            "Note",
        ])
        for row in rows:
            m = int(row["M"])
            n = int(row["N"])
            k = int(row["K"])
            padded_m = ceil_div(m, args.tile_m) * args.tile_m
            padded_n = ceil_div(n, args.tile_n) * args.tile_n
            padded_k = ceil_div(k, args.tile_k) * args.tile_k
            output = m * n
            padded_output = padded_m * padded_n
            compute = m * n * k
            padded_compute = padded_m * padded_n * padded_k
            output_waste = (padded_output - output) / padded_output if padded_output else 0.0
            compute_waste = (padded_compute - compute) / padded_compute if padded_compute else 0.0
            writer.writerow([
                row.get("Category", ""),
                m,
                n,
                k,
                args.tile_m,
                args.tile_n,
                args.tile_k,
                padded_m,
                padded_n,
                padded_k,
                output,
                padded_output,
                f"{output_waste * 100:.2f}",
                compute,
                padded_compute,
                f"{compute_waste * 100:.2f}",
                m % args.tile_m,
                n % args.tile_n,
                k % args.tile_k,
                row.get("Note", ""),
            ])


if __name__ == "__main__":
    raise SystemExit(main())

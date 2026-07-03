#!/usr/bin/env python3
import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "register_bench"
RESULTS_DIR = ROOT / "results"
DETAIL_CSV = RESULTS_DIR / "sass_operands.csv"
SUMMARY_TXT = RESULTS_DIR / "sass_summary.txt"

CASE_BY_KIND = {
    0: "R0_imad_chain",
    1: "R1_imad_independent_x4",
    2: "R2_reuse_hot_x4",
    3: "R3_bank_dense_x4",
    4: "R4_bank_sparse_x4",
}

FUNCTION_RE = re.compile(r"Function\s+:\s+.*ProbeKindE(\d+)EEE")
INSTRUCTION_RE = re.compile(r"/\*([0-9a-f]+)\*/\s+(.+?)\s*;")
REGISTER_RE = re.compile(r"\bR(\d+)(\.reuse)?\b")
TARGET_OPCODES = {"IMAD"}
CANDIDATE_BANK_COUNTS = (2, 4, 8, 16)


def collision_pairs(registers, bank_count):
    residues = Counter(register % bank_count for register in registers)
    return sum(count * (count - 1) // 2 for count in residues.values())


def parse_sass(text):
    rows = []
    current_case = None
    in_timed_region = False

    for line in text.splitlines():
        function_match = FUNCTION_RE.search(line)
        if function_match:
            current_case = CASE_BY_KIND.get(int(function_match.group(1)))
            in_timed_region = False
            continue
        if current_case is None:
            continue

        instruction_match = INSTRUCTION_RE.search(line)
        if not instruction_match:
            continue
        address, instruction = instruction_match.groups()
        if "CS2R" in instruction and "SR_CLOCKLO" in instruction:
            in_timed_region = not in_timed_region
            continue
        if not in_timed_region:
            continue

        opcode = instruction.split()[0].lstrip("@!P01234567")
        if opcode not in TARGET_OPCODES:
            continue

        register_matches = list(REGISTER_RE.finditer(instruction))
        if len(register_matches) < 2:
            continue
        destination = int(register_matches[0].group(1))
        sources = [int(match.group(1)) for match in register_matches[1:]]
        source_reuse = [bool(match.group(2)) for match in register_matches[1:]]
        rf_sources = [
            register
            for register, reused in zip(sources, source_reuse)
            if not reused
        ]

        row = {
            "case": current_case,
            "address": address,
            "opcode": opcode,
            "destination": destination,
            "sources": " ".join(f"R{value}" for value in sources),
            "reuse_sources": " ".join(
                f"R{register}"
                for register, reused in zip(sources, source_reuse)
                if reused
            ),
            "rf_sources": " ".join(f"R{value}" for value in rf_sources),
            "instruction": instruction.strip(),
        }
        for bank_count in CANDIDATE_BANK_COUNTS:
            row[f"all_mod{bank_count}_pairs"] = collision_pairs(
                sources, bank_count
            )
            row[f"rf_mod{bank_count}_pairs"] = collision_pairs(
                rf_sources, bank_count
            )
        rows.append(row)
    return rows


def write_detail(rows):
    fieldnames = [
        "case",
        "address",
        "opcode",
        "destination",
        "sources",
        "reuse_sources",
        "rf_sources",
    ]
    for bank_count in CANDIDATE_BANK_COUNTS:
        fieldnames.extend(
            [f"all_mod{bank_count}_pairs", f"rf_mod{bank_count}_pairs"]
        )
    fieldnames.append("instruction")

    with DETAIL_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows):
    lines = [
        "SASS operand analysis",
        "=====================",
        "",
        "Only instructions between the two CS2R SR_CLOCKLO reads are included.",
        "rf_modN excludes source operands carrying the SASS .reuse modifier.",
        "Modulo collisions are hypotheses, not proof of physical RF banking.",
        "",
    ]
    for case_name in CASE_BY_KIND.values():
        case_rows = [row for row in rows if row["case"] == case_name]
        if not case_rows:
            continue
        reuse_count = sum(
            len(row["reuse_sources"].split())
            for row in case_rows
            if row["reuse_sources"]
        )
        lines.append(case_name)
        lines.append("-" * len(case_name))
        lines.append(
            f"target instructions: {len(case_rows)}, "
            f"reuse-marked sources: {reuse_count}"
        )
        for bank_count in CANDIDATE_BANK_COUNTS:
            all_pairs = sum(
                int(row[f"all_mod{bank_count}_pairs"]) for row in case_rows
            )
            rf_pairs = sum(
                int(row[f"rf_mod{bank_count}_pairs"]) for row in case_rows
            )
            lines.append(
                f"candidate mod {bank_count:>2}: "
                f"all-source pairs={all_pairs:>4}, "
                f"non-reuse RF pairs={rf_pairs:>4}"
            )
        signatures = []
        for row in case_rows:
            signature = (
                row["opcode"],
                row["sources"],
                row["reuse_sources"],
            )
            if signature not in signatures:
                signatures.append(signature)
        lines.append("unique operand signatures:")
        for opcode, sources, reuse_sources in signatures[:12]:
            reuse_text = reuse_sources or "none"
            lines.append(f"  {opcode:<8} {sources:<16} reuse={reuse_text}")
        if len(signatures) > 12:
            lines.append(f"  ... {len(signatures) - 12} more")
        lines.append("")
    return "\n".join(lines)


def main():
    if not BIN.exists():
        raise SystemExit(f"Missing {BIN}; run scripts/build.sh first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["cuobjdump", "--dump-sass", str(BIN)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = parse_sass(completed.stdout)
    if not rows:
        raise SystemExit("No timed arithmetic instructions found in SASS.")
    write_detail(rows)
    summary = build_summary(rows)
    SUMMARY_TXT.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote {DETAIL_CSV}")
    print(f"Wrote {SUMMARY_TXT}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr)
        raise

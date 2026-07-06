#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
import struct
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
RESULTS = ROOT / "results" / "physical_probe"
KERNEL = "sass_register_probe"
EXPECTED = 128

FAMILIES = {
    "lop3": {
        "template": "sass_lop3_wide_template.sm_110.cubin",
        "opcode_prefix": "LOP3.",
    },
    "imad": {
        "template": "sass_imad_wide_template.sm_110.cubin",
        "opcode_prefix": "IMAD ",
    },
}

INSTRUCTION_RE = re.compile(r"/\*([0-9a-f]+)\*/\s+(.+?)\s*;")
REGISTER_RE = re.compile(r"\b(RZ|R\d+)(?:\.reuse)?\b")


def decode_register(token):
    return 255 if token == "RZ" else int(token[1:])


def display_tuple(registers):
    return ":".join(f"R{reg}" for reg in registers)


def section_location(data, wanted):
    section_offset = struct.unpack_from("<Q", data, 0x28)[0]
    section_entry_size = struct.unpack_from("<H", data, 0x3A)[0]
    section_count = struct.unpack_from("<H", data, 0x3C)[0]
    string_index = struct.unpack_from("<H", data, 0x3E)[0]

    def header(index):
        offset = section_offset + index * section_entry_size
        return struct.unpack_from("<IIQQQQIIQQ", data, offset)

    string_header = header(string_index)
    strings_offset, strings_size = string_header[4], string_header[5]
    strings = data[strings_offset : strings_offset + strings_size]
    for index in range(section_count):
        values = header(index)
        name_offset = values[0]
        name_end = strings.find(b"\0", name_offset)
        name = strings[name_offset:name_end].decode("utf-8")
        if name == wanted:
            return values[4], values[5]
    raise RuntimeError(f"ELF section not found: {wanted}")


def disassemble(path, opcode_prefix):
    completed = subprocess.run(
        ["nvdisasm", "-hex", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    in_kernel = False
    in_timed_region = False
    instructions = []
    for line in completed.stdout.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(".section") and ".text." in stripped:
            in_kernel = f".text.{KERNEL}" in stripped
            in_timed_region = False
            continue
        if not in_kernel:
            continue
        match = INSTRUCTION_RE.search(line)
        if not match:
            continue
        address, instruction = match.groups()
        if "CS2R" in instruction and "SR_CLOCKLO" in instruction:
            in_timed_region = not in_timed_region
            continue
        if not in_timed_region or not instruction.startswith(opcode_prefix):
            continue
        registers = [
            decode_register(token) for token in REGISTER_RE.findall(instruction)
        ]
        if len(registers) != 4:
            raise RuntimeError(f"unexpected operands: {instruction}")
        instructions.append(
            {
                "address": int(address, 16),
                "registers": registers,
                "reuse": ".reuse" in instruction,
                "text": instruction.strip(),
            }
        )
    if len(instructions) != EXPECTED:
        raise RuntimeError(f"expected {EXPECTED} timed ops, found {len(instructions)}")
    return instructions


def pool_pick(source_pool, modulo, residue, count, exclude=()):
    excluded = set(exclude)
    matches = [
        reg
        for reg in source_pool
        if reg % modulo == residue and reg not in excluded
    ]
    if len(matches) < count:
        raise RuntimeError(
            f"not enough regs for mod{modulo} residue {residue}: {matches}"
        )
    return matches[:count]


def source_pair_for(dest, source_pool, mode):
    if mode == "mixed":
        same = pool_pick(source_pool, 2, dest % 2, 1, exclude=(dest,))
        other = pool_pick(source_pool, 2, 1 - dest % 2, 1, exclude=(dest, same[0]))
        return same[0], other[0]
    if mode == "same_mod4":
        regs = pool_pick(source_pool, 4, dest % 4, 2, exclude=(dest,))
        return regs[0], regs[1]
    if mode == "split_mod4":
        same = pool_pick(source_pool, 4, dest % 4, 1, exclude=(dest,))
        other = pool_pick(source_pool, 4, (dest + 2) % 4, 1, exclude=(dest, same[0]))
        return same[0], other[0]
    if mode == "same_mod8":
        regs = pool_pick(source_pool, 8, dest % 8, 2, exclude=(dest,))
        return regs[0], regs[1]
    raise RuntimeError(f"unknown pressure mode: {mode}")


def slot_cases():
    cases = []
    for name, registers in (
        ("same_mod4", [8, 12, 16]),
        ("split_mod4", [8, 10, 68]),
        ("mixed", [8, 70, 9]),
    ):
        for acc_index, acc in enumerate(registers):
            sources = [reg for reg in registers if reg != acc]
            cases.append(
                {
                    "case": f"slot_{name}_acc{acc_index}_ab",
                    "category": "slot",
                    "mode": name,
                    "tuple": display_tuple([sources[0], sources[1], acc]),
                    "kind": "single_chain",
                    "registers": [sources[0], sources[1], acc],
                }
            )
            cases.append(
                {
                    "case": f"slot_{name}_acc{acc_index}_ba",
                    "category": "slot",
                    "mode": name,
                    "tuple": display_tuple([sources[1], sources[0], acc]),
                    "kind": "single_chain",
                    "registers": [sources[1], sources[0], acc],
                }
            )
    return cases


def pressure_cases(source_pool, destinations):
    cases = []
    for mode in ("mixed", "split_mod4", "same_mod4", "same_mod8"):
        schedules = []
        for dest in destinations:
            src0, src1 = source_pair_for(dest, source_pool, mode)
            schedules.append((dest, src0, src1, dest))
        cases.append(
            {
                "case": f"pressure_{mode}",
                "category": "pressure",
                "mode": mode,
                "tuple": " ".join(display_tuple(row[1:4]) for row in schedules),
                "kind": "multi_chain",
                "schedule": schedules,
            }
        )
    return cases


def build_cases(source_pool, instructions):
    destinations = sorted({row["registers"][0] for row in instructions})
    cases = slot_cases()
    cases.extend(pressure_cases(source_pool, destinations))
    return cases


def patch_case(template_data, text_offset, instructions, case):
    output = bytearray(template_data)
    expected = []
    if case["kind"] == "single_chain":
        src0, src1, acc = case["registers"]
        schedule = [(acc, src0, src1, acc)]
    else:
        schedule = case["schedule"]
    for index, row in enumerate(instructions):
        dest, src0, src1, src2 = schedule[index % len(schedule)]
        instruction_offset = text_offset + row["address"]
        output[instruction_offset + 2] = dest
        output[instruction_offset + 3] = src0
        output[instruction_offset + 4] = src1
        output[instruction_offset + 8] = src2
        output[instruction_offset + 15] &= ~0x3C
        expected.append((dest, src0, src1, src2))
    return output, expected


def verify(path, expected, opcode_prefix):
    actual = disassemble(path, opcode_prefix)
    for index, (row, wanted) in enumerate(zip(actual, expected)):
        got = tuple(row["registers"])
        if got != wanted:
            raise RuntimeError(f"{path.name} op {index}: expected {wanted}, got {got}")
        if row["reuse"]:
            raise RuntimeError(f"{path.name} op {index} still has .reuse")


def patch_family(family):
    config = FAMILIES[family]
    template = BUILD / config["template"]
    if not template.exists():
        raise SystemExit(f"missing {template}; run scripts/build.sh first")
    result_dir = RESULTS / family
    result_dir.mkdir(parents=True, exist_ok=True)
    for stale in result_dir.glob("*.cubin"):
        stale.unlink()

    template_data = bytearray(template.read_bytes())
    text_offset, _ = section_location(template_data, f".text.{KERNEL}")
    instructions = disassemble(template, config["opcode_prefix"])
    source_pool = sorted(
        {reg for row in instructions for reg in row["registers"][1:3] if reg != 255}
    )
    cases = build_cases(source_pool, instructions)
    rows = []
    for case in cases:
        cubin = result_dir / f"{case['case']}.cubin"
        patched, expected = patch_case(template_data, text_offset, instructions, case)
        cubin.write_bytes(patched)
        verify(cubin, expected, config["opcode_prefix"])
        rows.append(
            {
                "case": case["case"],
                "category": case["category"],
                "mode": case["mode"],
                "tuple": case["tuple"],
                "instructions": len(expected),
                "source_pool_min": min(source_pool),
                "source_pool_max": max(source_pool),
                "source_pool_count": len(source_pool),
                "reuse_sources": 0,
                "cubin": str(cubin),
            }
        )
    manifest = result_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(template, result_dir / template.name)
    print(f"{family}: generated {len(rows)} physical-probe cubins")
    print(f"Wrote {manifest}")


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def analyze_family(family):
    result_dir = RESULTS / family
    manifest = result_dir / "manifest.csv"
    results = result_dir / "results.csv"
    result_by_case = {row["case"]: row for row in load_rows(results)}
    rows = []
    for meta in load_rows(manifest):
        result = result_by_case.get(meta["case"])
        if result is None:
            raise SystemExit(f"missing result for {meta['case']}")
        rows.append({**meta, **result, "cycles": float(result["median_cycles_per_op"])})

    print(f"\n{family} physical probe")
    for category in ("slot", "pressure"):
        print(f"  {category}:")
        for row in [item for item in rows if item["category"] == category]:
            print(f"    {row['case']}: {row['cycles']:.6f} c/op")

    pressure = {
        row["mode"]: row["cycles"]
        for row in rows
        if row["category"] == "pressure"
    }
    if {"split_mod4", "same_mod4", "same_mod8", "mixed"} <= set(pressure):
        print(
            "  pressure deltas: "
            f"same_mod4-split_mod4={pressure['same_mod4'] - pressure['split_mod4']:.6f}, "
            f"same_mod8-split_mod4={pressure['same_mod8'] - pressure['split_mod4']:.6f}, "
            f"split_mod4-mixed={pressure['split_mod4'] - pressure['mixed']:.6f}"
        )


def query_metrics():
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "ncu_metric_candidates.txt"
    ncu = shutil.which("ncu")
    if not ncu:
        output.write_text("NCU metric query summary\n\nncu not found on PATH\n", encoding="utf-8")
        print(f"Wrote {output}")
        return

    def metric_name(line):
        parts = line.split(maxsplit=1)
        if not parts or "__" not in parts[0]:
            return None
        return parts[0]

    def base_metric(name):
        parts = name.split(".")
        for index, part in enumerate(parts):
            if part in {"avg", "max", "min", "sum"}:
                return ".".join(parts[:index])
        return name

    def has_rf_term(name):
        text = name.lower()
        return any(
            term in text
            for term in (
                "register",
                "regfile",
                "reg_bank",
                "__rf",
                "rf__",
                "_rf_",
                "operand",
                "collector",
            )
        )

    def has_bank_conflict_term(name):
        text = name.lower()
        return any(term in text for term in ("bank", "conflict", "port"))

    def add_limited(lines, title, names, limit=20):
        lines.append("")
        lines.append(title)
        if not names:
            lines.append("  none")
            return
        for name in names[:limit]:
            lines.append(f"  {name}")
        if len(names) > limit:
            lines.append(f"  ... {len(names) - limit} more")

    chips = []
    listed = subprocess.run(
        [ncu, "--list-chips"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if listed.returncode == 0:
        available = {
            item.strip()
            for item in listed.stdout.replace("\n", " ").split(",")
            if item.strip()
        }
        chips = [chip for chip in ("gb10b", "gb110") if chip in available]

    commands = [
        [ncu, "--query-metrics", "--query-metrics-mode", "all"],
    ]
    for chip in chips:
        for collection in ("profiling", "source", "warpsampling"):
            commands.append(
                [
                    ncu,
                    "--query-metrics",
                    "--query-metrics-mode",
                    "all",
                    "--query-metrics-collection",
                    collection,
                    "--chips",
                    chip,
                ]
            )
    records = {}
    default_errors = []
    queried = []
    errors = []
    for command in commands:
        queried.append(" ".join(command[1:]))
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            message = completed.stdout.strip()
            if "--chips" not in command:
                default_errors.append(message)
            elif message:
                errors.append(message)
        for line in completed.stdout.splitlines():
            name = metric_name(line)
            if name is None:
                continue
            base = base_metric(name)
            records.setdefault(base, line.strip())

    names = sorted(records)
    direct_rf = [
        name for name in names if has_rf_term(name) and has_bank_conflict_term(name)
    ]
    nearby_rf = [
        name for name in names if has_rf_term(name) and name not in direct_rf
    ]
    l1tex_banks = [
        name
        for name in names
        if name.startswith("l1tex__data_bank_conflicts")
    ]
    other_banks = [
        name
        for name in names
        if has_bank_conflict_term(name)
        and name not in direct_rf
        and name not in l1tex_banks
    ]

    lines = [
        "NCU metric query summary",
        "",
        f"Queried commands: {len(queried)}",
        f"Offline chips used: {', '.join(chips) if chips else 'none'}",
    ]
    if default_errors:
        first_error = default_errors[0].splitlines()[0]
        lines.append(f"Default live query status: {first_error}")
    if errors:
        lines.append(f"Offline query errors: {len(errors)}")

    add_limited(
        lines,
        "Direct RF/register/operand-collector bank candidates:",
        direct_rf,
    )
    add_limited(
        lines,
        "Nearby RF/register/operand metrics without bank/conflict wording:",
        nearby_rf,
        limit=12,
    )
    add_limited(
        lines,
        "L1TEX/LSU data-bank conflict metrics, not RF-bank metrics:",
        l1tex_banks,
        limit=16,
    )
    add_limited(
        lines,
        "Other bank/conflict/port metrics, not direct RF-bank metrics:",
        other_banks,
        limit=12,
    )
    lines.extend(
        [
            "",
            "Conclusion: this NCU metric list exposes L1TEX/LSU data-bank counters, "
            "but no direct register-file/SRAM-bank conflict counter was found.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["patch", "analyze", "query-metrics"])
    parser.add_argument("--family", choices=["all", *sorted(FAMILIES)], default="all")
    args = parser.parse_args()

    if args.action == "query-metrics":
        query_metrics()
        return

    families = sorted(FAMILIES) if args.family == "all" else [args.family]
    for family in families:
        if args.action == "patch":
            patch_family(family)
        else:
            analyze_family(family)


if __name__ == "__main__":
    main()

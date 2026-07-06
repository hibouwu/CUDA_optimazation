#!/usr/bin/env python3
import argparse
import csv
import random
import re
import shutil
import struct
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
KERNEL = "sass_register_probe"
EXPECTED = 128

FAMILIES = {
    "lop3": {
        "template": "sass_lop3_wide_template.sm_110.cubin",
        "result_dir": "tuple_scan_lop3",
        "opcode_prefix": "LOP3.",
        "patch_source2": True,
    },
    "imad": {
        "template": "sass_imad_wide_template.sm_110.cubin",
        "result_dir": "tuple_scan_imad",
        "opcode_prefix": "IMAD ",
        "patch_source2": True,
    },
}

INSTRUCTION_RE = re.compile(r"/\*([0-9a-f]+)\*/\s+(.+?)\s*;")
REGISTER_RE = re.compile(r"\b(RZ|R\d+)(?:\.reuse)?\b")


def decode_register(token):
    return 255 if token == "RZ" else int(token[1:])


def display_register(register):
    return "RZ" if register == 255 else f"R{register}"


def section_location(data, wanted):
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise RuntimeError("expected a little-endian ELF64 cubin")
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


def max_count(registers, modulo):
    return max(Counter(register % modulo for register in registers).values())


def residue_pool(pool, modulo, residue, exclude=()):
    excluded = set(exclude)
    return [reg for reg in pool if reg % modulo == residue and reg not in excluded]


def add_case(rows, seen, name, category, registers):
    if len(set(registers)) != 3:
        return
    key = tuple(registers)
    if key in seen:
        return
    seen.add(key)
    src0, src1, src2 = registers
    row = {
        "case": name,
        "category": category,
        "src0": src0,
        "src1": src1,
        "src2": src2,
        "tuple": ":".join(display_register(reg) for reg in registers),
    }
    for modulo in (2, 4, 8, 16):
        row[f"max_mod{modulo}"] = max_count(registers, modulo)
    rows.append(row)


def build_cases(source_pool, target_count):
    rows = []
    seen = set()

    for parity_name, parity in (("even", 0), ("odd", 1)):
        regs = [reg for reg in source_pool if reg % 2 == parity]
        if len(regs) >= 3:
            add_case(rows, seen, f"S_{parity_name}_wide", "designed",
                     [regs[0], regs[len(regs) // 2], regs[-1]])

    for residue in range(4):
        regs = residue_pool(source_pool, 4, residue)
        if len(regs) >= 3:
            add_case(rows, seen, f"S_mod4_r{residue}", "designed", regs[:3])

    for residue in range(8):
        regs = residue_pool(source_pool, 8, residue)
        if len(regs) >= 3:
            add_case(rows, seen, f"S_mod8_r{residue}", "designed", regs[:3])

    for parity in (0, 1):
        left = residue_pool(source_pool, 4, parity)
        right = residue_pool(source_pool, 4, parity + 2)
        if len(left) >= 2 and right:
            add_case(rows, seen, f"S_parity{parity}_split_mod4", "designed",
                     [left[0], right[0], left[-1]])

    evens = [reg for reg in source_pool if reg % 2 == 0]
    odds = [reg for reg in source_pool if reg % 2 == 1]
    if len(evens) >= 2 and odds:
        add_case(rows, seen, "M_two_even_one_odd", "designed",
                 [evens[0], evens[-1], odds[0]])
    if len(odds) >= 2 and evens:
        add_case(rows, seen, "M_two_odd_one_even", "designed",
                 [odds[0], odds[-1], evens[0]])

    rng = random.Random(110)
    attempts = 0
    while len(rows) < target_count and attempts < target_count * 40:
        attempts += 1
        registers = rng.sample(source_pool, 3)
        add_case(rows, seen, f"R{len(rows):02d}", "random", registers)
    return rows


def patch_variant(template_data, text_offset, instructions, registers, patch_source2):
    output = bytearray(template_data)
    src0, src1, src2 = registers
    expected = []
    for row in instructions:
        instruction_offset = text_offset + row["address"]
        output[instruction_offset + 2] = src2
        output[instruction_offset + 3] = src0
        output[instruction_offset + 4] = src1
        if patch_source2:
            output[instruction_offset + 8] = src2
        output[instruction_offset + 15] &= ~0x3C
        expected.append((src2, src0, src1, src2))
    return output, expected


def verify(path, expected, opcode_prefix):
    actual = disassemble(path, opcode_prefix)
    for index, (row, wanted) in enumerate(zip(actual, expected)):
        got = tuple(row["registers"])
        if got != wanted:
            raise RuntimeError(f"{path.name} op {index}: expected {wanted}, got {got}")
        if row["reuse"]:
            raise RuntimeError(f"{path.name} op {index} still has .reuse: {row['text']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), default="lop3")
    parser.add_argument("--cases", type=int, default=40)
    args = parser.parse_args()

    config = FAMILIES[args.family]
    template = BUILD / config["template"]
    if not template.exists():
        raise SystemExit(f"missing {template}; run scripts/build.sh first")

    result_dir = ROOT / "results" / config["result_dir"]
    result_dir.mkdir(parents=True, exist_ok=True)
    for stale_cubin in result_dir.glob("*.cubin"):
        stale_cubin.unlink()

    template_data = bytearray(template.read_bytes())
    text_offset, text_size = section_location(template_data, f".text.{KERNEL}")
    instructions = disassemble(template, config["opcode_prefix"])
    if max(row["address"] for row in instructions) + 16 > text_size:
        raise RuntimeError("timed instruction lies outside kernel text section")

    source_pool = sorted(
        {reg for row in instructions for reg in row["registers"][1:3] if reg != 255}
    )
    if len(source_pool) < 6:
        raise RuntimeError(f"source pool too small: {source_pool}")

    rows = build_cases(source_pool, args.cases)
    for row in rows:
        registers = [int(row["src0"]), int(row["src1"]), int(row["src2"])]
        path = result_dir / f"{row['case']}.cubin"
        patched, expected = patch_variant(
            template_data,
            text_offset,
            instructions,
            registers,
            config["patch_source2"],
        )
        path.write_bytes(patched)
        verify(path, expected, config["opcode_prefix"])
        row["instructions"] = len(expected)
        row["source_pool_min"] = min(source_pool)
        row["source_pool_max"] = max(source_pool)
        row["source_pool_count"] = len(source_pool)
        row["reuse_sources"] = 0
        row["cubin"] = str(path)

    manifest = result_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(template, result_dir / template.name)
    print(
        f"{args.family}: generated {len(rows)} tuple cubins, "
        f"source pool R{min(source_pool)}..R{max(source_pool)} "
        f"({len(source_pool)} regs)"
    )
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()

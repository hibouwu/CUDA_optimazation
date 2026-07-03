#!/usr/bin/env python3
import csv
import re
import shutil
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
RESULTS = ROOT / "results" / "sass_patched"
TEMPLATE = BUILD / "sass_template.sm_110.cubin"
MANIFEST = RESULTS / "manifest.csv"
KERNEL = "sass_register_probe"
EXPECTED_IMADS = 128

INSTRUCTION_RE = re.compile(r"/\*([0-9a-f]+)\*/\s+(.+?)\s*;")
REGISTER_RE = re.compile(r"\bR(\d+)(?:\.reuse)?\b")


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


def disassemble(path):
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
        if not in_timed_region or instruction.split()[0] != "IMAD":
            continue
        registers = [int(value) for value in REGISTER_RE.findall(instruction)]
        if len(registers) != 4:
            raise RuntimeError(f"unexpected IMAD operands: {instruction}")
        instructions.append(
            {
                "address": int(address, 16),
                "registers": registers,
                "reuse": ".reuse" in instruction,
                "text": instruction.strip(),
            }
        )
    if len(instructions) != EXPECTED_IMADS:
        raise RuntimeError(
            f"expected {EXPECTED_IMADS} timed IMADs, found {len(instructions)}"
        )
    return instructions


def select_registers(template_instructions, modulus):
    destinations = {row["registers"][0] for row in template_instructions}
    source_registers = sorted(
        {
            register
            for row in template_instructions
            for register in row["registers"][1:3]
            if register not in destinations
        }
    )
    groups = defaultdict(list)
    for register in source_registers:
        groups[register % modulus].append(register)
    usable = [
        (residue, values[:2])
        for residue, values in sorted(groups.items())
        if len(values) >= 2
    ]
    if len(usable) < modulus:
        raise RuntimeError(
            f"need two source registers for every mod-{modulus} residue; "
            f"got {dict(groups)}"
        )
    usable = usable[:modulus]
    selected = [register for _, pair in usable for register in pair]
    return usable, selected


def make_pair_sets(groups):
    conflict = [tuple(pair) for _, pair in groups]
    control = []
    for member in range(2):
        values = [pair[member] for _, pair in groups]
        for index in range(0, len(values), 2):
            control.append((values[index], values[index + 1]))
    return control, conflict


def patch_variant(template_data, text_offset, template_instructions, pairs):
    output = bytearray(template_data)
    expected = []
    for index, row in enumerate(template_instructions):
        destination = row["registers"][0]
        source0, source1 = pairs[index % len(pairs)]
        instruction_offset = text_offset + row["address"]
        # sm_110 IMAD: byte 2=dst, byte 3=src0, byte 4=src1, byte 8=src2.
        output[instruction_offset + 2] = destination
        output[instruction_offset + 3] = source0
        output[instruction_offset + 4] = source1
        output[instruction_offset + 8] = destination
        # Reuse flags are bits 2..5 of byte 15 in the sm_110 control word.
        output[instruction_offset + 15] &= ~0x3C
        expected.append((destination, source0, source1, destination))
    return output, expected


def verify(path, expected):
    actual = disassemble(path)
    for index, (row, wanted) in enumerate(zip(actual, expected)):
        got = tuple(row["registers"])
        if got != wanted:
            raise RuntimeError(
                f"{path.name} IMAD {index}: expected {wanted}, got {got}"
            )
        if row["reuse"]:
            raise RuntimeError(
                f"{path.name} IMAD {index} still contains .reuse: {row['text']}"
            )


def main():
    if not TEMPLATE.exists():
        raise SystemExit(f"missing {TEMPLATE}; run scripts/build.sh first")
    RESULTS.mkdir(parents=True, exist_ok=True)
    template_data = bytearray(TEMPLATE.read_bytes())
    text_offset, text_size = section_location(
        template_data, f".text.{KERNEL}"
    )
    template_instructions = disassemble(TEMPLATE)
    if any(row["reuse"] for row in template_instructions):
        print(
            "Template contains .reuse; clearing and verifying all flags.",
            file=sys.stderr,
        )
    if max(row["address"] for row in template_instructions) + 16 > text_size:
        raise RuntimeError("timed instruction lies outside kernel text section")

    rows = []
    for modulus in (4, 8):
        groups, selected = select_registers(template_instructions, modulus)
        control, conflict = make_pair_sets(groups)
        for kind, pairs in (("control", control), ("conflict", conflict)):
            name = f"S{len(rows)}_mod{modulus}_{kind}_noreuse"
            path = RESULTS / f"{name}.cubin"
            patched, expected = patch_variant(
                template_data, text_offset, template_instructions, pairs
            )
            path.write_bytes(patched)
            verify(path, expected)
            rows.append(
                {
                    "case": name,
                    "modulus": modulus,
                    "pairing": kind,
                    "instructions": len(expected),
                    "source_registers": " ".join(
                        f"R{register}" for register in selected
                    ),
                    "pair_schedule": " ".join(
                        f"R{left}:R{right}" for left, right in pairs
                    ),
                    "reuse_sources": 0,
                    "cubin": str(path),
                }
            )

    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(TEMPLATE, RESULTS / TEMPLATE.name)
    print(f"Generated and verified {len(rows)} patched cubins")
    for row in rows:
        print(
            f"{row['case']}: {row['pair_schedule']} "
            f"(reuse={row['reuse_sources']})"
        )
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()

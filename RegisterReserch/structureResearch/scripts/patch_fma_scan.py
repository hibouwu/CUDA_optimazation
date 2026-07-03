#!/usr/bin/env python3
import csv
import re
import shutil
import struct
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
RESULTS = ROOT / "results" / "bank_scan_fma"
TEMPLATE = BUILD / "sass_fma_template.sm_110.cubin"
MANIFEST = RESULTS / "manifest.csv"
KERNEL = "sass_register_probe"
EXPECTED_LOP3 = 128
MAX_STRIDE = 16

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
        if not in_timed_region or not instruction.startswith("LOP3."):
            continue
        registers = [
            decode_register(token) for token in REGISTER_RE.findall(instruction)
        ]
        if len(registers) != 4:
            raise RuntimeError(f"unexpected LOP3 operands: {instruction}")
        instructions.append(
            {
                "address": int(address, 16),
                "registers": registers,
                "reuse": ".reuse" in instruction,
                "text": instruction.strip(),
            }
        )
    if len(instructions) != EXPECTED_LOP3:
        raise RuntimeError(
            f"expected {EXPECTED_LOP3} timed LOP3s, found {len(instructions)}"
        )
    return instructions


def patch_variant(template_data, text_offset, template_instructions, tuples):
    output = bytearray(template_data)
    expected = []
    for index, row in enumerate(template_instructions):
        destination, source0, source1, source2 = tuples[
            index % len(tuples)
        ]
        instruction_offset = text_offset + row["address"]
        output[instruction_offset + 2] = destination
        output[instruction_offset + 3] = source0
        output[instruction_offset + 4] = source1
        output[instruction_offset + 8] = source2
        output[instruction_offset + 15] &= ~0x3C
        expected.append((destination, source0, source1, source2))
    return output, expected


def verify(path, expected):
    actual = disassemble(path)
    for index, (row, wanted) in enumerate(zip(actual, expected)):
        got = tuple(row["registers"])
        if got != wanted:
            raise RuntimeError(
                f"{path.name} LOP3 {index}: expected {wanted}, got {got}"
            )
        if row["reuse"]:
            raise RuntimeError(
                f"{path.name} LOP3 {index} still has .reuse: {row['text']}"
            )


def tuple_text(tuples):
    return " ".join(
        ":".join(display_register(register) for register in registers)
        for registers in tuples
    )


def add_variant(
    rows,
    template_data,
    text_offset,
    template_instructions,
    name,
    category,
    tuples,
    rf_reads,
    base="",
    stride="",
    pattern="",
):
    path = RESULTS / f"{name}.cubin"
    patched, expected = patch_variant(
        template_data, text_offset, template_instructions, tuples
    )
    path.write_bytes(patched)
    verify(path, expected)
    rows.append(
        {
            "case": name,
            "category": category,
            "rf_reads": rf_reads,
            "base": base,
            "stride": stride,
            "pattern": pattern,
            "instructions": len(expected),
            "tuple_schedule": tuple_text(tuples),
            "reuse_sources": 0,
            "cubin": str(path),
        }
    )


def main():
    if not TEMPLATE.exists():
        raise SystemExit(f"missing {TEMPLATE}; run scripts/build.sh first")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for pattern in ("B*.cubin", "L*.cubin", "M*.cubin", "T*.cubin"):
        for stale_cubin in RESULTS.glob(pattern):
            stale_cubin.unlink()
    template_data = bytearray(TEMPLATE.read_bytes())
    text_offset, text_size = section_location(
        template_data, f".text.{KERNEL}"
    )
    template_instructions = disassemble(TEMPLATE)
    if max(row["address"] for row in template_instructions) + 16 > text_size:
        raise RuntimeError("timed instruction lies outside kernel text section")

    destinations = sorted(
        {row["registers"][0] for row in template_instructions}
    )
    if destinations != [4, 5, 6, 7]:
        raise RuntimeError(
            f"expected accumulators R4..R7, got "
            f"{[display_register(value) for value in destinations]}"
        )

    rows = []
    add_variant(
        rows,
        template_data,
        text_offset,
        template_instructions,
        "B0_chain_1rf",
        "source_count",
        [(4, 255, 255, 4)],
        1,
        pattern="accumulator only",
    )
    add_variant(
        rows,
        template_data,
        text_offset,
        template_instructions,
        "B1_chain_2rf_same_bank",
        "source_count",
        [(4, 8, 255, 4)],
        2,
        pattern="two even registers",
    )
    add_variant(
        rows,
        template_data,
        text_offset,
        template_instructions,
        "B2_chain_3rf_mixed",
        "source_count",
        [(4, 9, 8, 4)],
        3,
        pattern="two even, one odd",
    )
    add_variant(
        rows,
        template_data,
        text_offset,
        template_instructions,
        "B3_chain_3rf_same_bank",
        "source_count",
        [(4, 8, 10, 4)],
        3,
        pattern="three even registers",
    )

    slot_cases = [
        ("M0_same_pair_src1_src2", (4, 9, 8, 4), "same pair src1/src2"),
        ("M1_same_pair_src0_src2", (4, 8, 9, 4), "same pair src0/src2"),
        ("M2_same_pair_src0_src1", (4, 4, 8, 9), "same pair src0/src1"),
    ]
    for name, registers, pattern in slot_cases:
        add_variant(
            rows,
            template_data,
            text_offset,
            template_instructions,
            name,
            "slot_permutation",
            [registers],
            3,
            pattern=pattern,
        )

    for base in destinations:
        for stride in range(1, MAX_STRIDE + 1):
            source0 = base + stride
            source1 = base + 2 * stride
            if source1 > 39:
                raise RuntimeError(
                    f"R{source1} is outside initialized register range R4..R39"
                )
            add_variant(
                rows,
                template_data,
                text_offset,
                template_instructions,
                f"L_b{base:02d}_s{stride:02d}",
                "latency_stride",
                [(base, source0, source1, base)],
                3,
                base,
                stride,
            )

    for stride in range(1, MAX_STRIDE + 1):
        tuples = [
            (base, base + stride, base + 2 * stride, base)
            for base in destinations
        ]
        add_variant(
            rows,
            template_data,
            text_offset,
            template_instructions,
            f"T_s{stride:02d}",
            "throughput_stride",
            tuples,
            3,
            "",
            stride,
        )

    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(TEMPLATE, RESULTS / TEMPLATE.name)
    print(
        f"Generated and verified {len(rows)} cubins: "
        f"4 source-count + 3 slot permutations + "
        f"{len(destinations) * MAX_STRIDE} latency + "
        f"{MAX_STRIDE} throughput cases"
    )
    print("Physical scan range: accumulators R4..R7, initialized R4..R39")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()

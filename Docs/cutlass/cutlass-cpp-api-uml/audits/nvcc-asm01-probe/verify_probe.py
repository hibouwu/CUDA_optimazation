#!/usr/bin/env python3
"""Read-only verification of the NVCC ASM01 probe's per-entry evidence."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()
    directory = args.result_directory
    report = json.loads((directory / "report.json").read_text())
    for case in report["results"]:
        ptx = (directory / case["case"] / "probe.ptx").read_text()
        blocks = re.split(r"(?=\.visible \.entry )", ptx)[1:]
        entries = {re.search(r"\.entry\s+(\w+)", block)[1]: block for block in blocks}
        assert set(entries) == set(report["metadata"]["entry_mapping"])
        expected = int(case["forced_device_use"])
        counts = [len(re.findall(r"^\s*tcgen05\.cp", block, re.M)) for block in entries.values()]
        assert counts == [expected] * 12
        if expected:
            # This is the concrete emitted flow observed for NVCC 13.0.88,
            # not a register-allocation assumption for arbitrary compilers.
            for name, block in entries.items():
                assert re.search(r"ld\.param\.u64\s+%rd1, \[" + name + r"_param_0\]", block)
                assert re.search(r"ld\.param\.u32\s+%r1, \[" + name + r"_param_1\]", block)
                assert re.search(r"tcgen05\.cp[^\n]*\[%r1\], %rd1;", block)
        sass = (directory / case["case"] / "nvdisasm.stdout.txt").read_text()
        pieces = re.split(r"(?m)^\s*\.section\s+\.text\.(asm01_emit_\d+)[^\n]*", sass)
        sass_entries = {pieces[i]: pieces[i + 1] for i in range(1, len(pieces), 2)}
        assert set(sass_entries) == set(entries)
        assert all(len(re.findall(r"\bUTCCP(?:\.|\s)", block)) == expected for block in sass_entries.values())
        assert all(stage["returncode"] == 0 for stage in case["stages"])
        print(case["case"], "entries=12", "PTX_per_entry=" + str(expected), "SASS_per_entry=" + str(expected))
    for target in ("sm_100a", "sm_110a"):
        for emit in (False, True):
            pair = [case for case in report["results"] if case["target"] == target and case["forced_device_use"] == emit]
            assert len(pair) == 2
            assert len({case["ptx_sha256"] for case in pair}) == 1
            assert len({case["cubin_sha256"] for case in pair}) == 1
    print("ALL_12_WRAPPERS_TWO_TARGETS_EXACT_RAW_CONTROL_PTX_CUBIN_AND_OPERAND_FLOW_PASS")


if __name__ == "__main__":
    main()

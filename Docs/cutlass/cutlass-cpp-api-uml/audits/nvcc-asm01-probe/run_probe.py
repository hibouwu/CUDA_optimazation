#!/usr/bin/env python3
"""Compile-only NVCC dialect probe. Never load a module or launch a GPU kernel.

Creates a fresh result directory and preserves every command/stdout/stderr.
The raw 12-wrapper slice is byte-identical to the pinned snapshot; the comma
variant is a separately labelled counterfactual control, never a source edit.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

PROBE = Path(__file__).resolve().parent
ATLAS = PROBE.parents[1]
SNAPSHOT = ATLAS / "snapshot"
SOURCE_PATH = "include/cute/arch/copy_sm100.hpp"
SOURCE_SHA = "0fe82cbfc7c5a5e4629e16efa992d18801f1c136acb5fb63401f396e5965b305"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_command(command, directory, label):
    result = subprocess.run([str(x) for x in command], text=True, capture_output=True)
    (directory / (label + ".stdout.txt")).write_text(result.stdout)
    (directory / (label + ".stderr.txt")).write_text(result.stderr)
    record = {"command": [str(x) for x in command], "returncode": result.returncode,
              "stdout_file": str(directory / (label + ".stdout.txt")),
              "stderr_file": str(directory / (label + ".stderr.txt"))}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda-13.0/bin/nvcc"))
    parser.add_argument("--host-compiler", type=Path, default=Path("/usr/bin/g++-14"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        output = args.output.resolve()
    else:
        output = Path(tempfile.mkdtemp(prefix="run-", dir=PROBE))
    print("OUTPUT", output, flush=True)
    source = (SNAPSHOT / SOURCE_PATH).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    section = b"".join(source.splitlines(keepends=True)[365:599])
    names = re.findall(rb"struct\s+(SM100_UTCCP_\w+)\s*\{", section)
    assert len(names) == 12
    missing = b'"r"(dst_addr)  "l"(src_addr)'
    comma = b'"r"(dst_addr), "l"(src_addr)'
    assert section.count(missing) == 12
    license_text = source[:source.index(b"#pragma once")]
    prefix = license_text + b'''// Extracted compiler probe; never execute this kernel.
#include <cute/config.hpp>
#include <cute/arch/config.hpp>
#if defined(__CUDA_ARCH__)
#if !defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
#error "Expected original CUTLASS configuration to enable the actual TMEM branch"
#endif
#if __CUDA_ARCH__ != PROBE_EXPECT_ARCH
#error "Unexpected actual device architecture"
#endif
#endif
namespace cute {
#line 366 "snapshot/include/cute/arch/copy_sm100.hpp"
'''
    suffix = b'\n} // namespace cute\n#line 1 "asm01_probe_entry.cu"\n'
    # PTXAS rejects mixing .cta_group::1 and ::2 inside one entry. Give each
    # wrapper its own genuine entry, preserving the wrapper bodies verbatim.
    for index, name in enumerate(names):
        suffix += f'extern "C" __global__ void asm01_emit_{index:02}(uint64_t src_addr, uint32_t dst_addr) {{\n'.encode()
        suffix += b"#if defined(PROBE_EMIT)\n  cute::SM100::TMEM::UTCCP::" + name + b"::copy(src_addr, dst_addr);\n#endif\n}\n"
    generated = {}
    for variant, body in (("raw", section), ("comma_control", section.replace(missing, comma))):
        path = output / (variant + ".cu")
        path.write_bytes(prefix + body + suffix)
        generated[variant] = path
    (output / "raw_original_section.txt").write_bytes(section)
    metadata = {"nvcc_version": subprocess.check_output([str(args.nvcc), "--version"], text=True),
                "host_compiler_version": subprocess.check_output([str(args.host_compiler), "--version"], text=True),
                "snapshot_path": SOURCE_PATH, "snapshot_sha256": SOURCE_SHA,
                "source_lines": [366, 599], "source_section_sha256": hashlib.sha256(section).hexdigest(),
                "wrapper_names": [name.decode() for name in names],
                "entry_mapping": {f"asm01_emit_{i:02}": name.decode() for i, name in enumerate(names)},
                "generated_source_sha256": {name: sha(path) for name, path in generated.items()},
                "scope": "NVCC device parsing, PTX emission, PTXAS and SASS only; no runtime or GPU use",
                "source_condition": "defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)",
                "configuration_headers": {p: sha(SNAPSHOT / p) for p in
                                          ("include/cute/config.hpp", "include/cute/arch/config.hpp", "include/cutlass/arch/config.h")}}
    initial = output / "initial_host_compatibility"
    initial.mkdir()
    metadata["initial_host_compatibility"] = run_command(
        [args.nvcc, "-std=c++17", "-ccbin", args.host_compiler, "-arch=sm_100a", "--ptx", "-DPROBE_EMIT",
         PROBE / "minimal.cu", "-o", initial / "minimal.ptx"], initial, "nvcc_without_host_workaround")
    metadata["host_compatibility_flags"] = ["-Xcompiler", "-U_GNU_SOURCE", "-D_DEFAULT_SOURCE=1", "-D_POSIX_C_SOURCE=200809L"]
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def compile_case(target, variant, emit):
        case_name = target + "_" + variant + ("_used" if emit else "_unused")
        directory = output / case_name
        directory.mkdir()
        keep = directory / "keep"
        keep.mkdir()
        arch = "1100" if target == "sm_110a" else "1000"
        common = [args.nvcc, "-std=c++17", "-ccbin", args.host_compiler, "-Xcompiler", "-U_GNU_SOURCE",
                  "-D_DEFAULT_SOURCE=1", "-D_POSIX_C_SOURCE=200809L",
                  "-I", SNAPSHOT / "include", "-arch=" + target, "-DPROBE_EXPECT_ARCH=" + arch]
        if emit:
            common += ["-DPROBE_EMIT"]
        ptx = directory / "probe.ptx"
        cubin = directory / "probe.cubin"
        stages = []
        stages.append(run_command(common + ["--ptx", "--keep", "--keep-dir", keep, "-v", generated[variant], "-o", ptx], directory, "nvcc_ptx"))
        stages.append(run_command(common + ["--cubin", "-Xptxas=-v", generated[variant], "-o", cubin], directory, "nvcc_cubin"))
        ptx_text = ptx.read_text() if ptx.exists() else ""
        ptx_instructions = re.findall(r"^\s*(tcgen05\.cp[^;]*;)", ptx_text, re.M)
        report = {"case": case_name, "target": target, "variant": variant, "forced_device_use": emit,
                  "stages": stages, "ptx_instruction_count": len(ptx_instructions),
                  "ptx_instructions": ptx_instructions,
                  "ptx_sha256": sha(ptx) if ptx.exists() else None,
                  "cubin_sha256": sha(cubin) if cubin.exists() else None}
        if ptx.exists() and emit:
            stages.append(run_command([args.nvcc.parent / "ptxas", "-arch=" + target, "-v", ptx,
                                       "-o", directory / "direct_ptxas.cubin"], directory, "direct_ptxas"))
        if cubin.exists():
            disassembly = run_command([args.nvcc.parent / "nvdisasm", cubin], directory, "nvdisasm")
            stages.append(disassembly)
            sass = (directory / "nvdisasm.stdout.txt").read_text()
            report["sass_UTCCP_lines"] = [line.strip() for line in sass.splitlines() if re.search(r"\bUTCCP(?:\.|\s)", line)]
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(case_name, "returncodes", [s["returncode"] for s in stages], "PTX tcgen05.cp", len(ptx_instructions),
              "SASS UTCCP", len(report.get("sass_UTCCP_lines", [])), flush=True)
        return report

    cases = [(t, v, e) for t in ("sm_100a", "sm_110a") for v in generated for e in (False, True)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(compile_case, *case) for case in cases]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda item: item["case"])
    comparisons = []
    for target in ("sm_100a", "sm_110a"):
        for emit in (False, True):
            raw = next(r for r in results if r["target"] == target and r["variant"] == "raw" and r["forced_device_use"] == emit)
            control = next(r for r in results if r["target"] == target and r["variant"] == "comma_control" and r["forced_device_use"] == emit)
            comparisons.append({"target": target, "forced_device_use": emit,
                                "both_ptx_compilations_succeeded": raw["stages"][0]["returncode"] == control["stages"][0]["returncode"] == 0,
                                "expected_ptx_count": 12 if emit else 0,
                                "ptx_instructions_equal": (raw["stages"][0]["returncode"] == control["stages"][0]["returncode"] == 0
                                                           and len(raw["ptx_instructions"]) == len(control["ptx_instructions"]) == (12 if emit else 0)
                                                           and raw["ptx_instructions"] == control["ptx_instructions"]),
                                "cubin_hashes_equal": bool(raw["cubin_sha256"]) and raw["cubin_sha256"] == control["cubin_sha256"]})
    final = {"metadata": metadata, "results": results, "raw_control_comparisons": comparisons,
             "gpu_launched": False, "cuda_module_loaded": False}
    (output / "report.json").write_text(json.dumps(final, indent=2) + "\n")
    print("REPORT", output / "report.json", flush=True)
    print("COMPARISONS", json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()

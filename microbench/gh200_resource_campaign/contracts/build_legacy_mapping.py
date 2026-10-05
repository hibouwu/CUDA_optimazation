#!/usr/bin/env python3
"""Build an offline, source-checked mapping of the three immutable L0 archives.

This records legacy contracts; it neither qualifies old timing nor runs a GPU.
Only the requested output file is written. Paths in the result are repo-relative.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
ARCHIVES = Path("microbench/gh200_l0/results")
PROTOCOL = Path("microbench/gh200_resource_campaign/contracts/protocol.json")
OUTPUT = Path("microbench/gh200_resource_campaign/contracts/legacy_mapping.json")
BATCHES = {
    "initial": ("probe.cu", None, 32, 7, (128, 512, 2048)),
    "sustained": ("stress.cu", "stress_cases.json", 42, 5, None),
    "audit": ("audit.cu", "audit_cases.json", 7, 12, (8192, 32768, 65536)),
}
KEY_FIELDS = ("ptx", "instruction_template", "shape", "types", "sources", "layout",
              "threads", "chains", "batch", "wait", "input", "timer", "drain",
              "scope", "grid_policy", "iteration_policy", "phase")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def key_id(key):
    return sha256(encode(key).encode())


def extract_contract(case, source, batch):
    """Cross-check manifest fields against the archived kernel and host launch."""
    name, ptx = case["name"], case["ptx"]
    match = re.search(r"__global__ void " + re.escape(name) + r"\(.*?\n}\n", source, re.S)
    require(match is not None, f"{name}: archived function absent")
    body = match.group()
    assembly = re.findall(r'asm volatile\("([^"\n]+)"([^;\n]*);', body)
    arithmetic = [(a, b) for a, b in assembly if a.startswith(ptx + " ")]
    require(len(arithmetic) == 1, f"{name}: PTX form absent or ambiguous")
    template, binding = arithmetic[0]
    for var, field in (("q", "batch"), ("c", "chains")):
        require(re.search(r"for\(int " + var + r"=0;" + var + "<" + str(case[field]) + ";", body),
                f"{name}: {field} disagrees with source")
    require(f"drain[{case['threads']}]" in body, f"{name}: threads disagree with source")
    require("auto start=clock64();" in body and "auto stop=clock64();" in body,
            f"{name}: clock64 boundary absent")
    drain = "drain[threadIdx.x]=sum;\n  __syncthreads();\n  auto stop=clock64();"
    require(drain in body, f"{name}: result drain changed")
    wg, mma = ptx.startswith("wgmma."), ptx.startswith("mma.")
    matrix = wg or mma
    shape_match = re.search(r"\.m(\d+)n(\d+)k(\d+)\.", ptx)
    shape = list(map(int, shape_match.groups())) if shape_match else None
    require(matrix == (shape is not None), f"{name}: shape mismatch")
    dtype = case["kind"].removeprefix("wgmma_").removeprefix("mma_")
    lanes = 2 if dtype.endswith("x2") else 1
    width = 128 if wg else 32 if mma else 1
    require(case["threads"] % width == 0, f"{name}: partial collective")
    work = (2 * shape[0] * shape[1] * shape[2] * (case["threads"] // width)
            if matrix else 2 * case["threads"] * lanes)
    registers = 32 if wg else 2 if dtype == "f64" and mma else 4 if mma else 1
    require(case["work_per_collective"] == work, f"{name}: work mismatch")
    require(case["outputs"] == case["threads"] * case["chains"] * registers * lanes,
            f"{name}: output count mismatch")
    host_fields = ",".join(str(case[k]) for k in
                           ("threads", "outputs", "chains", "batch", "work_per_collective"))
    host_prefix = f'run({name},"{name}",' if batch == "initial" else (
        f'run_stress({name},"{name}","{case["kind"]}",' if batch == "sustained" else
        f'make_probe({name},"{name}","{case["kind"]}",')
    require(host_prefix + host_fields + "," in source, f"{name}: host launch mismatch")
    waits = [int(x) for x in re.findall(r"wgmma.wait_group.sync.aligned (\d+);", body)]
    pending = case.get("pending_after_wait", 0)
    require(waits == ([pending] + ([0] if batch != "initial" else []) if wg else []),
            f"{name}: wait/drain mismatch")
    if wg:
        require("wgmma.commit_group.sync.aligned;" in body, f"{name}: commit absent")
        require("fence.proxy.async.shared::cta;" in body and "wgmma.fence.sync.aligned;" in body,
                f"{name}: setup fence absent")
        require("(64ull<<16) | (8ull<<32)" in source, f"{name}: descriptor layout changed")
        bits = "0x2c00" if dtype == "f16" else "0x3d80"
        require(f"a[i]={bits};b[i]={bits};" in body, f"{name}: SMEM input changed")
        layout = {"A": "K-major", "B": "K-major", "swizzle": "none",
                  "descriptor_leading_offset_bytes": 1024,
                  "descriptor_stride_offset_bytes": 128,
                  "transpose_A": 0, "transpose_B": 0}
    else:
        layout = {"A": "row", "B": "col"} if mma else {"lanes": lanes}
    if batch != "initial":
        require("auto start_ns=timer_ns();\n  auto start=clock64();" in body and
                "auto stop=clock64();\n  auto stop_ns=timer_ns();" in body,
                f"{name}: globaltimer ordering changed")
    input_contract = {
        "distribution": "uniform_constant_finite", "accumulator_initial": 0,
        "arithmetic_inline_asm_bindings": binding.strip(),
        "A_B_value": 0.0625 if matrix else None,
        "recurrence": "D=A*B+D" if matrix else "D=D*0.5+0.25",
        "smem_bits": ("0x2c00" if dtype == "f16" else "0x3d80") if wg else None,
    }
    contract = {
        "ptx": ptx, "instruction_template": template,
        "shape": shape, "types": {"input": dtype, "accumulator": "f64" if dtype == "f64"
                                    else "f32" if matrix else dtype},
        "sources": {"A_B": "shared_memory_descriptors" if wg else "registers",
                    "accumulator": "registers"},
        "layout": layout, "threads": case["threads"], "chains": case["chains"],
        "batch": case["batch"], "wait": pending if wg else None, "input": input_contract,
    }
    evidence = {"kernel_first_line": source[:match.start()].count("\n") + 1,
                "kernel_sha256": sha256(body.encode()), "host_launch_prefix": host_prefix + host_fields,
                "work_flop_per_collective_per_cta": work,
                "collective_width_threads": width}
    return contract, evidence


def measurement_contract(compute, batch, scope, iterations, phase):
    wg = compute["ptx"].startswith("wgmma.")
    timer = {
        "start_after": "operand_setup_and_cta_barrier",
        "stop_after": "result_reduction_volatile_smem_store_and_cta_barrier",
        "read_order": ["clock64_start", "clock64_stop"] if batch == "initial" else
                      ["globaltimer_start", "clock64_start", "clock64_stop", "globaltimer_stop"],
        "primary_basis": "local_clock64_cycles" if scope == "single_cta" else "grid_globaltimer_ns",
        "grid_interval": None if scope == "single_cta" else "max(stop_ns)-min(start_ns)",
        "additional_cycle_basis": "sum_per_smid(max(stop_cycle)-min(start_cycle))"
                                  if batch == "audit" else None,
        "event_ms_role": "launch_envelope_and_calibration" if batch == "sustained" else
                         "launch_envelope_auxiliary",
    }
    policy = {"kind": "fixed", "iterations": iterations}
    if batch == "sustained":
        policy = {"kind": "dynamic_cuda_event_pilot", "pilot_iterations": 8192,
                  "target_event_ms": 100, "minimum_iterations": 8192,
                  "maximum_iterations": 65536 if compute["shape"] else 1048576,
                  "formula": "clamp(int(8192*100.0/max(pilot_ms,0.01)),8192,maximum_iterations)",
                  "observed_iterations": iterations}
    key = dict(compute, timer=timer,
               drain={"result": "sum_registers_to_volatile_double_smem_then_cta_barrier",
                      "per_iteration_wgmma": "commit_group_then_wait" if wg else None,
                      "post_loop_wgmma_wait": 0 if wg and batch != "initial" else None},
               scope=scope, grid_policy="1 CTA" if scope == "single_cta" else "sms*min(4,occupancy)",
               iteration_policy=policy, phase=phase)
    require(set(key) == set(KEY_FIELDS), "incomplete canonical key")
    return key


def validate_records(rows, case, repeats, iterations, scope, phase):
    require(len(rows) == repeats, f"{case['name']}: missing/extra {phase} repeats")
    ids = [r.get("repeat", r.get("round")) for _, r in rows]
    require(sorted(ids) == list(range(repeats)), f"{case['name']}: duplicate/missing repeat ID")
    for _, row in rows:
        require(row["iterations"] == iterations and row.get("scope", "single_cta") == scope,
                f"{case['name']}: iteration/scope mismatch")
        require(row.get("phase", "measure") == phase, f"{case['name']}: phase mismatch")
        require(row.get("threads", case["threads"]) == case["threads"], "raw threads mismatch")
        require(row["max_abs_error"] == 0, "legacy numerical error")
        expected = row.get("blocks", 1) * iterations * case["chains"] * case["batch"] * case["work_per_collective"]
        require(row["work_flop"] == expected, "legacy work mismatch")


def build(repo_root=ROOT):
    root = Path(repo_root)
    protocol = json.loads((root / PROTOCOL).read_text())
    result = {"schema_version": 2, "suite_id": protocol["suite_id"],
              "purpose": "legacy_contract_mapping_only; all destination sampling remains pending",
              "canonical_key_fields": list(KEY_FIELDS),
              "protocol_sha256": sha256((root / PROTOCOL).read_bytes()),
              "batches": {}, "configurations": [], "mappings": []}
    base_ids = set()
    for batch, (source_name, manifest_name, count, repeats, lengths) in BATCHES.items():
        relative = ARCHIVES / f"20260930-{batch}"
        directory = root / relative
        source = (directory / source_name).read_text()
        summary = json.loads((directory / "summary.json").read_text())
        manifest = json.loads((directory / manifest_name).read_text()) if manifest_name else summary
        cases = manifest["cases"]
        require(len(cases) == count and len({c['name'] for c in cases}) == count,
                f"{batch}: configuration count/identity mismatch")
        # Old manifests and summaries are independent of this generator.
        hashed = [source_name, "raw.jsonl"] + ([manifest_name] if manifest_name else [])
        archive_hashes = {line.split()[1]: line.split()[0]
                          for line in (directory / "SHA256SUMS").read_text().splitlines() if line.strip()}
        expected_hashes = (dict(archive_hashes, **{"raw.jsonl": summary["raw_sha256"]})
                           if batch == "audit" else summary["file_sha256"])
        for filename in hashed:
            require(expected_hashes[filename] == sha256((directory / filename).read_bytes()),
                    f"{batch}: archived {filename} hash mismatch")
        if batch == "sustained":
            require(manifest["base_source_sha256"] == sha256((root / ARCHIVES / "20260930-initial/probe.cu").read_bytes()),
                    "sustained: initial source linkage mismatch")
            require("int(8192*100.0/std::max(pilot_ms,0.01)),8192,max_iterations)" in source and
                    "int max_iterations=increment?65536:1048576;" in source,
                    "sustained: calibration source changed")
        if batch == "initial":
            require("for(int iterations:{128,512,2048})" in source, "initial: lengths changed")
        if batch == "audit":
            require(manifest["lengths"] == list(lengths) and manifest["rounds"] == repeats,
                    "audit: manifest protocol changed")
            require(manifest["source_sha256"] == sha256((root / ARCHIVES / "20260930-sustained/stress.cu").read_bytes()),
                    "audit: sustained source linkage mismatch")
        files = [source_name, "summary.json", "raw.jsonl", "SHA256SUMS"]
        files += [p.name for p in sorted(directory.glob("*.sh"))]
        files += [p.name for p in sorted(directory.glob("generate*.py"))]
        if manifest_name:
            files.append(manifest_name)
        grouped, auxiliary = defaultdict(list), defaultdict(list)
        known = {c["name"] for c in cases}
        for line, text in enumerate((directory / "raw.jsonl").read_text().splitlines(), 1):
            row = json.loads(text)
            if "case" not in row:
                require(line == 1 and "device" in row, "unexpected non-measurement row")
                continue
            require(row["case"] in known, "unknown archived case")
            phase = row.get("phase", "measure")
            require(phase in ("measure", "empty_control", "warmup", "transition_warmup"), "unknown phase")
            scope = row.get("scope", "single_cta")
            if phase in ("warmup", "transition_warmup"):
                require(batch == "audit", "unexpected auxiliary records")
                auxiliary[(row["case"], phase, scope, row["iterations"])].append(line)
            else:
                grouped[(row["case"], phase, scope, row["iterations"])].append((line, row))
        consumed = set()
        batch_counts = Counter()
        for case in cases:
            compute, evidence = extract_contract(case, source, batch)
            base_id = key_id(compute)
            base_ids.add(base_id)
            config_id = f"20260930-{batch}/{case['name']}"
            result["configurations"].append({"legacy_configuration_id": config_id,
                "basic_combination_id": base_id, "kind": case["kind"],
                "source_path": str(relative / source_name), "source_evidence": evidence,
                "manifest_path": str(relative / (manifest_name or "summary.json")),
                "manifest_row_index": cases.index(case),
                "manifest_role": "archived_cases" if manifest_name else "summary_cases_cross_checked_with_archived_source"})
            scopes = ["single_cta"] if batch == "initial" else ["single_cta", "full_gpu"]
            conditions = []
            for scope in scopes:
                selected_lengths = lengths
                if batch == "sustained":
                    selected_lengths = sorted({it for name, phase, sc, it in grouped
                                               if name == case["name"] and sc == scope and phase == "measure"})
                    require(len(selected_lengths) == 1, "sustained: missing or inconsistent calibrated length")
                    cap = 65536 if compute["shape"] else 1048576
                    require(8192 <= selected_lengths[0] <= cap, "sustained: length outside calibrated bounds")
                conditions.extend(("measure", scope, it) for it in selected_lengths)
            if batch == "audit":
                conditions.append(("empty_control", "single_cta", 0))
            for phase, scope, iterations in conditions:
                group = (case["name"], phase, scope, iterations)
                rows = grouped.get(group, [])
                validate_records(rows, case, repeats, iterations, scope, phase)
                consumed.add(group)
                key = measurement_contract(compute, batch, scope, iterations, phase)
                mapping_id = key_id(key)
                stage = "S07" if case["kind"].startswith("wgmma") else "S06" if case["kind"].startswith("mma") else "S05"
                result["mappings"].append({"mapping_id": mapping_id,
                    "legacy_configuration_id": config_id, "basic_combination_id": base_id,
                    "key": key, "legacy_raw_path": str(relative / "raw.jsonl"),
                    "legacy_raw_lines": [line for line, _ in rows],
                    "legacy_repeat_count": repeats,
                    "legacy_observed_blocks": sorted({r.get("blocks", 1) for _, r in rows}),
                    "destination": {"stage": stage, "case_id": "legacy-" + mapping_id,
                                    "status": "pending_remeasurement",
                                    "repetition_policy": "protocol.json external processes; legacy repeats are not independent processes"}})
                batch_counts[phase + "_conditions"] += 1
                batch_counts[phase + "_records"] += len(rows)
        require(consumed == set(grouped), f"{batch}: unrepresented measurement/control conditions")
        result["batches"][batch] = {
            "configuration_rows": count, **dict(batch_counts),
            "source_files": {str(relative / name): sha256((directory / name).read_bytes()) for name in sorted(set(files))},
            "gaps": ["standalone cases.json not archived; summary and source cross-checked"] if batch == "initial" else [],
            "auxiliary_records": [{"case": name, "phase": phase, "scope": scope,
                                   "iterations": it, "raw_lines": lines}
                                  for (name, phase, scope, it), lines in sorted(auxiliary.items())],
            "legacy_warmup_policy": {
                "initial": "two unrecorded launches per case/length before seven same-process repeats",
                "sustained": "unrecorded 4096 warmup, 8192 pilot, calibrated warmup before five same-process repeats",
                "audit": "retained full-GPU 65536 warmups: min8/max30, last5 event_ms sample CV<=1%; 4096 transition before each measure; 12 rounds",
            }[batch],
        }
    result["counts"] = {"legacy_configuration_rows": len(result["configurations"]),
                        "basic_instruction_launch_combinations": len(base_ids),
                        "mapped_conditions": len(result["mappings"]),
                        "formal_conditions": sum(x["key"]["phase"] == "measure" for x in result["mappings"]),
                        "empty_control_conditions": sum(x["key"]["phase"] == "empty_control" for x in result["mappings"])}
    require(result["counts"] == {"legacy_configuration_rows": 81, "basic_instruction_launch_combinations": 62,
            "mapped_conditions": 229, "formal_conditions": 222, "empty_control_conditions": 7}, "coverage counts changed")
    require(len({m["mapping_id"] for m in result["mappings"]}) == 229, "full contract collision")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Defaults to repo contracts/legacy_mapping.json")
    parser.add_argument("--check", action="store_true", help="Read-only byte-for-byte regeneration check")
    args = parser.parse_args()
    result = build(args.repo_root)
    output = args.output or args.repo_root / OUTPUT
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        require(output.read_text() == serialized, "mapping differs from deterministic regeneration")
    else:
        require(not output.resolve().is_relative_to((args.repo_root / ARCHIVES).resolve()),
                "refusing to overwrite legacy archives")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)
    print(json.dumps({"check_only": args.check, **result["counts"]}))


if __name__ == "__main__":
    main()

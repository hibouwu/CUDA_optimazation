#!/usr/bin/env python3
"""Validate the guide's case contracts without importing CUTLASS or CUDA."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_CASES = {
    "dense_f16_1sm_p128",
    "dense_bf16_1sm_p128",
    "dense_fp8_1sm_p128",
    "dense_f16_2sm_p256x128x128",
    "bs_mxfp8_1sm_p128",
    "bs_mxfp4_1sm_p128x128x256",
    "bs_nvfp4_1sm_p128x128x256",
    "sparse_bs_nvfp4_1sm_p128x128x256",
    "epilogue_bias_relu_f16_p128",
    "tail_dense_f16_p130x129x127",
}

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "case_id",
    "title",
    "track",
    "target",
    "problem",
    "kernel",
    "types",
    "layouts",
    "codegen",
    "numerical",
    "evidence",
}

REQUIRED_EVIDENCE = {
    "documented",
    "source_present",
    "compile_passed",
    "ptx_verified",
    "sass_verified",
    "runtime_correct",
    "performance_measured",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: cannot parse JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be an object")
    return value


def require_shape(errors: list[str], case_id: str, owner: dict[str, Any], key: str, rank: int) -> None:
    value = owner.get(key)
    if not isinstance(value, list) or len(value) != rank or not all(isinstance(x, int) and x > 0 for x in value):
        errors.append(f"{case_id}: {key} must be a positive rank-{rank} integer list")


def validate_case(path: Path, data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    case_id = data.get("case_id", path.parent.name)
    missing = REQUIRED_TOP_LEVEL - data.keys()
    if missing:
        errors.append(f"{case_id}: missing top-level fields {sorted(missing)}")
        return errors
    if data["schema_version"] != 1:
        errors.append(f"{case_id}: schema_version must be 1")
    if case_id != path.parent.name:
        errors.append(f"{case_id}: case_id must match directory {path.parent.name}")
    if not (path.parent / "case.cu").is_file():
        errors.append(f"{case_id}: case.cu is missing")
    if not (path.parent / "README.md").is_file():
        errors.append(f"{case_id}: README.md is missing")

    target = data.get("target", {})
    if target.get("arch") != "sm_110a" or target.get("virtual_arch") != "compute_110a":
        errors.append(f"{case_id}: target must be compute_110a -> sm_110a")
    if target.get("cutlass_arch_tag") != "cutlass::arch::Sm100":
        errors.append(f"{case_id}: CUTLASS builder tag must be cutlass::arch::Sm100")

    problem = data.get("problem", {})
    require_shape(errors, case_id, problem, "mnkl", 4)

    kernel = data.get("kernel", {})
    for key in ("mma_tile_mnk", "instruction_mnk", "cluster_mnk"):
        require_shape(errors, case_id, kernel, key, 3)
    if kernel.get("cta_group") not in (1, 2):
        errors.append(f"{case_id}: cta_group must be 1 or 2")
    schedule = kernel.get("mainloop_schedule", "")
    if not isinstance(schedule, str) or not schedule or schedule.endswith("Auto"):
        errors.append(f"{case_id}: instruction-contract cases require an explicit mainloop schedule")

    codegen = data.get("codegen", {})
    ptx = codegen.get("ptx", {})
    sass = codegen.get("sass", {})
    if not ptx.get("required"):
        errors.append(f"{case_id}: codegen.ptx.required cannot be empty")
    if not sass.get("required_families"):
        errors.append(f"{case_id}: codegen.sass.required_families cannot be empty")
    if codegen.get("attribution") != "single_target_function_block":
        errors.append(f"{case_id}: codegen attribution must be function-local")

    numerical = data.get("numerical", {})
    if numerical.get("reference") != "independent_cpu_reference":
        errors.append(f"{case_id}: the primary oracle must be independent_cpu_reference")
    if not isinstance(numerical.get("seed"), int):
        errors.append(f"{case_id}: numerical.seed must be an integer")
    for key in ("atol", "rtol"):
        if not isinstance(numerical.get(key), (int, float)) or numerical[key] < 0:
            errors.append(f"{case_id}: numerical.{key} must be nonnegative")

    evidence = data.get("evidence", {})
    if set(evidence) != REQUIRED_EVIDENCE:
        errors.append(f"{case_id}: evidence fields must be exactly {sorted(REQUIRED_EVIDENCE)}")
    if evidence.get("performance_measured") is not False:
        errors.append(f"{case_id}: performance_measured must remain false in v0.1")
    return errors


def validate_submodule(root: Path, lock: dict[str, Any], errors: list[str]) -> None:
    expected = lock.get("cutlass", {}).get("git_sha")
    cutlass = root / "third_party" / "cutlass"
    if not (cutlass / ".git").exists():
        errors.append("CUTLASS submodule is not initialized")
        return
    try:
        actual = subprocess.check_output(
            ["git", "-c", f"safe.directory={cutlass}", "-C", str(cutlass), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append(f"cannot inspect CUTLASS submodule: {error}")
        return
    if actual != expected:
        errors.append(f"CUTLASS submodule {actual} != locked {expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    errors: list[str] = []
    lock = load_json(root / "versions.lock.json")
    validate_submodule(root, lock, errors)

    manifests = sorted((root / "cases").glob("*/case.json"))
    actual = {path.parent.name for path in manifests}
    missing = EXPECTED_CASES - actual
    extra = actual - EXPECTED_CASES
    if missing:
        errors.append(f"missing core cases: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected cases not in v0.1 matrix: {sorted(extra)}")
    for path in manifests:
        try:
            errors.extend(validate_case(path, load_json(path)))
        except ValueError as error:
            errors.append(str(error))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"CASE_CONTRACTS_PASS cases={len(manifests)} cutlass={lock['cutlass']['git_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

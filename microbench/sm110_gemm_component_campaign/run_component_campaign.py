#!/usr/bin/env python3
"""Resumable Thor collection for TMA, TMEM readback, and NVFP4 epilogue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = Path(__file__).resolve().parent
RESULT_ROOT = REPO / "results" / "sm110_gemm_component_campaign"
EXPECTED_SMS = 20
TRIALS = 10
DEFAULT_TRIAL_TIMEOUT_SECONDS = 120
SOURCE_DEPENDENCIES = [
    "microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu",
    "microbench/12_tmem_readback_bandwidth/tmem_readback_bandwidth.cu",
    "microbench/13_tmem_scale_ingress_bandwidth/tmem_scale_ingress_bandwidth.cu",
    "microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu",
    "GEMMsm110/include/backends/tc5_persistent.cuh",
    "GEMMsm110/include/sm110_ptx_helpers.cuh",
    "GEMMsm110/tests/requant_epilogue_benchmark.cu",
    "GEMMsm110/include/gemm_common.cuh",
    "GEMMsm110/include/requant/requant_backend.cuh",
    "GEMMsm110/include/requant/e2m1_encode.cuh",
    "GEMMsm110/include/requant/pack_fp4.cuh",
    "GEMMsm110/include/requant/scale_policy.cuh",
    "GEMMsm110/include/requant/sm110_tcgen05_epilogue.cuh",
]


CASES = [
    {
        "id": "tma_l2_hit_32k",
        "resource": "tma.l2_hit_ingress.serial32k",
        "binary": "tma",
        "args": ["--mode", "l2-hit", "--bytes", str(16 << 20), "--tile-bytes", "32768", "--slots", "4", "--inflight", "1", "--iters", "4096", "--warmup-iters", "512", "--blocks", "1", "--blocks-per-sm", "1", "--threads", "128"],
        "sass": ["UTMALDG.3D"],
    },
    {
        "id": "tma_dram_stream_32k",
        "resource": "tma.dram_stream_ingress.serial32k",
        "binary": "tma",
        "args": ["--mode", "dram-stream", "--bytes", str(256 << 20), "--tile-bytes", "32768", "--slots", "4", "--inflight", "1", "--iters", "4096", "--warmup-iters", "32", "--blocks-per-sm", "1", "--threads", "128"],
        "sass": ["UTMALDG.3D"],
    },
    {
        "id": "tma_l2_hit_32k_inflight4",
        "resource": "tma.l2_hit_ingress.inflight4",
        "binary": "tma",
        "args": ["--mode", "l2-hit", "--bytes", str(16 << 20), "--tile-bytes", "32768", "--slots", "4", "--inflight", "4", "--iters", "4096", "--warmup-iters", "512", "--blocks", "1", "--blocks-per-sm", "1", "--threads", "128"],
        "sass": ["UTMALDG.3D"],
    },
    {
        "id": "tma_dram_stream_32k_inflight4",
        "resource": "tma.dram_stream_ingress.inflight4",
        "binary": "tma",
        "args": ["--mode", "dram-stream", "--bytes", str(256 << 20), "--tile-bytes", "32768", "--slots", "4", "--inflight", "4", "--iters", "4096", "--warmup-iters", "32", "--blocks-per-sm", "1", "--threads", "128"],
        "sass": ["UTMALDG.3D"],
    },
    {
        "id": "tma_l2_hit_tc5a_ab_inflight8",
        "resource": "tma.l2_hit_ingress",
        "binary": "tma",
        "args": ["--mode", "l2-hit", "--pattern", "tc5a-ab", "--bytes", str(16 << 20), "--tile-bytes", "16384", "--slots", "8", "--inflight", "8", "--iters", "4096", "--warmup-iters", "1024", "--blocks", "1", "--blocks-per-sm", "1", "--threads", "192"],
        "sass": ["UTMALDG.2D"],
    },
    {
        "id": "tma_dram_stream_tc5a_ab_inflight8",
        "resource": "tma.dram_stream_ingress",
        "binary": "tma",
        "args": ["--mode", "dram-stream", "--pattern", "tc5a-ab", "--bytes", str(256 << 20), "--tile-bytes", "16384", "--slots", "8", "--inflight", "8", "--iters", "4096", "--warmup-iters", "32", "--blocks-per-sm", "1", "--threads", "192"],
        "sass": ["UTMALDG.2D"],
    },
    {
        "id": "tmem_scale_ingress_32x128b_warpx4",
        "resource": "tmem.scale_ingress",
        "binary": "scale",
        "args": ["--iters", "16384", "--copies-per-commit", "32", "--blocks-per-sm", "1"],
        "sass": ["UTCCP.T.S.4x32dp128bit", "LDTM.x4"],
    },
    *[
        {
            "id": f"{residency}_{direction}_aggregate",
            "resource": f"{residency}.{direction}",
            "binary": "memory",
            "args": [
                "--residency", residency,
                "--direction", direction,
                "--bytes", str((16 if residency == "l2" else 256) << 20),
                "--iters", str(4096 if residency == "l2" else 1024),
                "--warmup-iters", "64",
                "--blocks-per-sm", "4",
                "--threads", "256",
            ],
            "sass": ["LDG.E.128" if direction == "read" else "STG.E.128"],
        }
        for residency in ("hbm", "l2") for direction in ("read", "write")
    ],
    *[
        {
            "id": f"tmem_ld_32x32b_x{registers}_warps{warps}",
            "resource": "tmem.accumulator_readback",
            "binary": "tmem",
            "args": ["--registers", str(registers), "--warps", str(warps), "--iters", "10000", "--blocks-per-sm", "1"],
            "sass": [f"LDTM.x{registers}"],
        }
        for registers in (8, 16) for warps in (1, 4)
    ],
    *[
        {
            "id": f"nvfp4_requant_4096x1024_{distribution}",
            "resource": "epilogue.nvfp4_requant",
            "binary": "epilogue",
            "args": ["--rows", "4096", "--cols", "1024", "--blocks-per-sm", "1", "--distribution", distribution, "--seed", "1234", "--warmup", "10", "--iterations", "100"],
            "sass": ["LDTM.x2", "STTM.x2"],
        }
        for distribution in ("normal", "outlier", "constant")
    ],
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(run_dir: Path, status: str, **extra: object) -> None:
    payload = {"status": status, "pid": os.getpid(), "hostname": platform.node(),
               "updated_at_utc": utc_now(), **extra}
    (run_dir / "campaign_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (run_dir / "progress.jsonl").open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result


def run_bounded_trial(command: list[str], timeout_seconds: int) -> dict[str, object]:
    """Run one GPU trial without any unbounded post-timeout wait."""
    proc = subprocess.Popen(
        command, cwd=REPO, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    termination_failed = False
    try:
        output, _ = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as initial_timeout:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired as term_timeout:
            os.killpg(proc.pid, signal.SIGKILL)
            try:
                output, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired as kill_timeout:
                termination_failed = True
                output = (kill_timeout.stdout or term_timeout.stdout
                          or initial_timeout.stdout or "")
                if isinstance(output, bytes):
                    output = output.decode(errors="backslashreplace")
    return {
        "returncode": proc.returncode, "stdout": output,
        "timed_out": timed_out,
        "termination_failed": termination_failed,
    }


def executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"required executable not found: {name}")
    return path


def environment() -> dict[str, object]:
    commands = {
        "gpu_identity": ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"],
        "gpu_state": ["nvidia-smi", "--query-gpu=pstate,clocks.current.graphics,power.limit", "--format=csv,noheader"],
        "nvcc": ["nvcc", "--version"],
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short"],
    }
    result: dict[str, object] = {
        "captured_at_utc": utc_now(), "hostname": platform.node(),
        "platform": platform.platform(), "python": sys.version,
    }
    for name, command in commands.items():
        proc = run(command, check=False)
        result[name] = {"returncode": proc.returncode, "output": proc.stdout}
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel:
        proc = run([nvpmodel, "-q"], check=False)
        result["power_mode"] = {"returncode": proc.returncode, "output": proc.stdout}
    else:
        result["power_mode"] = {"returncode": 127, "output": "nvpmodel not found"}
    return result


def parse_kv(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        for match in re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line):
            result[match.group(1)] = match.group(2)
    return result


def compile_binaries(run_dir: Path, host_compiler: str | None,
                     undef_gnu_source: bool) -> dict[str, dict[str, object]]:
    nvcc = executable("nvcc")
    cuobjdump = executable("cuobjdump")
    build = run_dir / "build"
    build.mkdir(exist_ok=True)
    specs = {
        "tma": {
            "source": REPO / "microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu",
            "include": [], "libs": ["-lcuda"],
        },
        "tmem": {
            "source": REPO / "microbench/12_tmem_readback_bandwidth/tmem_readback_bandwidth.cu",
            "include": [], "libs": [],
        },
        "scale": {
            "source": REPO / "microbench/13_tmem_scale_ingress_bandwidth/tmem_scale_ingress_bandwidth.cu",
            "include": [], "libs": [],
        },
        "memory": {
            "source": REPO / "microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu",
            "include": [], "libs": [],
        },
        "epilogue": {
            "source": REPO / "GEMMsm110/tests/requant_epilogue_benchmark.cu",
            "include": [f"-I{REPO / 'GEMMsm110/include'}"], "libs": [],
        },
    }
    artifacts: dict[str, dict[str, object]] = {}
    for name, spec in specs.items():
        binary = build / name
        command = [nvcc, "-O3", "-std=c++17"]
        if host_compiler:
            command += ["-ccbin", host_compiler]
        if undef_gnu_source:
            command += ["-Xcompiler=-U_GNU_SOURCE", "-D_DEFAULT_SOURCE",
                        "-D_POSIX_C_SOURCE=200809L", "-D_XOPEN_SOURCE=700",
                        "-D_XOPEN_SOURCE_EXTENDED=1", "-D_LARGEFILE64_SOURCE=1",
                        "-D_ATFILE_SOURCE=1"]
        command += ["-gencode", "arch=compute_110a,code=sm_110a"]
        command += list(spec["include"]) + [str(spec["source"])] + list(spec["libs"]) + ["-o", str(binary)]
        compile_result = run(command, check=False)
        (build / f"{name}.compile_command.json").write_text(json.dumps(command, indent=2) + "\n")
        (build / f"{name}.compile.log").write_text(compile_result.stdout)
        if compile_result.returncode:
            raise RuntimeError(f"{name} compile failed; see {build / f'{name}.compile.log'}")
        sass_result = run([cuobjdump, "--dump-sass", str(binary)], check=False)
        sass_path = build / f"{name}.sass.txt"
        sass_path.write_text(sass_result.stdout)
        if sass_result.returncode:
            raise RuntimeError(f"cuobjdump failed for {name}")
        artifacts[name] = {
            "binary": binary, "source": spec["source"],
            "source_sha256": sha256(spec["source"]),
            "binary_sha256": sha256(binary), "sass_sha256": sha256(sass_path),
            "sass_path": sass_path,
        }
        (build / f"{name}.binary.sha256").write_text(
            f"{artifacts[name]['binary_sha256']}  {name}\n")
    return artifacts


def validate_fields(case: dict[str, object], fields: dict[str, str]) -> float:
    if int(fields.get("sm_count", "0")) != EXPECTED_SMS:
        raise RuntimeError(f"{case['id']}: expected {EXPECTED_SMS} SMs")
    resource = str(case["resource"])
    if resource.startswith("tma."):
        expected_blocks = (1 if resource.startswith("tma.l2_hit_ingress")
                           else EXPECTED_SMS)
        expected_unique_sms = expected_blocks
        if int(fields.get("unique_smid_count", "0")) != expected_unique_sms:
            raise RuntimeError(f"{case['id']}: TMA SM-scope mismatch")
        requested = int(fields["requested_bytes"])
        elapsed_ns = int(fields["globaltimer_elapsed_ns"])
        reported = float(fields["globaltimer_gbytes_per_second"]) * 1e9
        recalculated = requested * 1e9 / elapsed_ns
        args = list(case["args"])
        expected_inflight = int(args[args.index("--inflight") + 1])
        expected_slots = int(args[args.index("--slots") + 1])
        expected_tile_bytes = int(args[args.index("--tile-bytes") + 1])
        expected_pattern = (
            str(args[args.index("--pattern") + 1])
            if "--pattern" in args else "uniform")
        expected_stage_count = 4 if expected_pattern == "tc5a-ab" else expected_slots
        expected_requests_per_stage = 2 if expected_pattern == "tc5a-ab" else 1
        expected_barrier_count = expected_stage_count
        expected_tensor_map = (
            "2d-sw128" if expected_pattern == "tc5a-ab" else "3d-none")
        expected_row_stride = 2048 if expected_pattern == "tc5a-ab" else 0
        expected_smem_bytes = (
            4 * 49152 if expected_pattern == "tc5a-ab"
            else expected_tile_bytes * expected_slots)
        expected_carveout = "max" if expected_pattern == "tc5a-ab" else "default"
        expected_mode = str(args[args.index("--mode") + 1])
        expected_backing_bytes = int(args[args.index("--bytes") + 1])
        expected_warmup_iters = int(
            args[args.index("--warmup-iters") + 1])
        expected_iters = int(args[args.index("--iters") + 1])
        expected_blocks_per_sm = int(
            args[args.index("--blocks-per-sm") + 1])
        expected_threads = int(args[args.index("--threads") + 1])
        if int(fields.get("inflight", "0")) != expected_inflight:
            raise RuntimeError(f"{case['id']}: TMA inflight contract mismatch")
        if int(fields.get("slots", "0")) != expected_slots:
            raise RuntimeError(f"{case['id']}: TMA slot contract mismatch")
        expected_launch = {
            "mode": expected_mode,
            "pattern": expected_pattern,
            "stage_count": str(expected_stage_count),
            "requests_per_stage": str(expected_requests_per_stage),
            "barrier_count": str(expected_barrier_count),
            "tensor_map": expected_tensor_map,
            "row_stride_elements": str(expected_row_stride),
            "smem_bytes": str(expected_smem_bytes),
            "preferred_smem_carveout": expected_carveout,
            "tile_bytes": str(expected_tile_bytes),
            "iters": str(expected_iters),
            "warmup_iters": str(expected_warmup_iters),
            "blocks_per_sm": str(expected_blocks_per_sm),
            "threads": str(expected_threads),
            "blocks": str(expected_blocks),
            "requested_blocks": str(
                1 if resource.startswith("tma.l2_hit_ingress") else 0),
        }
        launch_mismatches = [
            f"{name} expected={expected} "
            f"actual={fields.get(name, '<missing>')}"
            for name, expected in expected_launch.items()
            if fields.get(name) != expected
        ]
        if launch_mismatches:
            raise RuntimeError(
                f"{case['id']}: TMA launch/payload mismatch: "
                + "; ".join(launch_mismatches))
        if int(fields.get("occupancy_blocks_per_sm", "0")) <= 0:
            raise RuntimeError(
                f"{case['id']}: TMA occupancy is missing or nonpositive")
        warmup_bytes = expected_warmup_iters * (
            49152 if expected_pattern == "tc5a-ab"
            else expected_tile_bytes)
        if (resource.startswith("tma.l2_hit_ingress")
                and warmup_bytes < int(fields.get("working_set_bytes", "0"))):
            raise RuntimeError(
                f"{case['id']}: L2-hit warmup does not cover the backing set")
        total_tiles = int(fields.get("total_tiles", "0"))
        total_tiles_b = int(fields.get("total_tiles_b", "0"))
        working_set_bytes = int(fields.get("working_set_bytes", "0"))
        allocation_bytes = int(fields.get("allocation_bytes", "0"))
        if expected_pattern == "tc5a-ab":
            expected_total_tiles = max(
                1, expected_backing_bytes // 49152)
            if (total_tiles != expected_total_tiles
                    or total_tiles_b != total_tiles
                    or working_set_bytes != total_tiles * 49152
                    or allocation_bytes != total_tiles * 384 * 2048 * 2):
                raise RuntimeError(
                    f"{case['id']}: tc5a logical/physical backing mismatch")
        else:
            expected_total_tiles = max(
                expected_backing_bytes // expected_tile_bytes,
                expected_blocks,
            )
            if expected_mode == "dram-stream":
                expected_total_tiles = max(
                    expected_total_tiles,
                    expected_blocks
                    * (expected_warmup_iters
                       + int(args[args.index("--iters") + 1])),
                )
            if (total_tiles != expected_total_tiles
                    or total_tiles_b != total_tiles
                    or working_set_bytes
                    != total_tiles * expected_tile_bytes
                    or allocation_bytes != working_set_bytes):
                raise RuntimeError(
                    f"{case['id']}: uniform TMA backing mismatch")
    elif resource == "tmem.accumulator_readback":
        if int(fields.get("unique_smid_count", "0")) != EXPECTED_SMS:
            raise RuntimeError(f"{case['id']}: incomplete SM coverage")
        issued = int(fields["issued_bytes"])
        elapsed_ns = int(fields["globaltimer_elapsed_ns"])
        reported = float(fields["bytes_per_second"])
        recalculated = issued * 1e9 / elapsed_ns
    elif resource == "tmem.scale_ingress":
        if fields.get("case_id") != case["id"]:
            raise RuntimeError(f"{case['id']}: scale case identity mismatch")
        if int(fields.get("unique_smid_count", "0")) != EXPECTED_SMS:
            raise RuntimeError(f"{case['id']}: incomplete SM coverage")
        if int(fields.get("source_bytes_per_instruction", "0")) != 512:
            raise RuntimeError(f"{case['id']}: scale source atom is not 512 B")
        if int(fields.get("multicast_partitions", "0")) != 4:
            raise RuntimeError(f"{case['id']}: scale multicast contract mismatch")
        if (int(fields.get("destination_slots", "0")) != 32
                or int(fields.get("destination_columns_per_copy", "0")) != 4):
            raise RuntimeError(
                f"{case['id']}: scale destination slots do not match 32x4 contract")
        if int(fields.get("value_mismatches", "-1")) != 0:
            raise RuntimeError(f"{case['id']}: scale S2T value mismatch")
        issued = int(fields["issued_source_bytes"])
        elapsed_ns = int(fields["globaltimer_elapsed_ns"])
        reported = float(fields["bytes_per_second"])
        recalculated = issued * 1e9 / elapsed_ns
    elif resource in {"hbm.read", "hbm.write", "l2.read", "l2.write"}:
        if fields.get("case_id") != case["id"]:
            raise RuntimeError(f"{case['id']}: memory-path case identity mismatch")
        if int(fields.get("unique_smid_count", "0")) != EXPECTED_SMS:
            raise RuntimeError(f"{case['id']}: incomplete SM coverage")
        expected_residency, expected_direction = resource.split(".")
        if (fields.get("residency") != expected_residency
                or fields.get("direction") != expected_direction):
            raise RuntimeError(f"{case['id']}: memory-path mode mismatch")
        if int(fields.get("blocks_per_sm", "0")) != 4:
            raise RuntimeError(f"{case['id']}: memory-path launch mismatch")
        expected_bytes = (16 if expected_residency == "l2" else 256) << 20
        if int(fields.get("working_set_bytes", "0")) != expected_bytes:
            raise RuntimeError(f"{case['id']}: working-set contract mismatch")
        requested = int(fields["requested_bytes"])
        elapsed_ns = int(fields["globaltimer_elapsed_ns"])
        reported = float(fields["bytes_per_second"])
        recalculated = requested * 1e9 / elapsed_ns
    else:
        if int(fields.get("unique_smid_count", "0")) != EXPECTED_SMS:
            raise RuntimeError(f"{case['id']}: incomplete epilogue SM coverage")
        if int(fields.get("value_mismatches", "-1")) != 0 or int(fields.get("scale_mismatches", "-1")) != 0:
            raise RuntimeError(f"{case['id']}: NVFP4 reference mismatch")
        gelements = float(fields["gelements_per_second"])
        if gelements <= 0:
            raise RuntimeError(f"{case['id']}: nonpositive epilogue rate")
        return gelements * 1e9
    if elapsed_ns <= 0 or abs(reported - recalculated) > max(1.0, recalculated * 2e-6):
        raise RuntimeError(f"{case['id']}: invalid rate arithmetic")
    return recalculated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--host-compiler")
    parser.add_argument("--nvcc-host-undef-gnu-source", action="store_true")
    parser.add_argument("--trial-timeout-seconds", type=int,
                        default=DEFAULT_TRIAL_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("invalid run-id")
    if args.trial_timeout_seconds <= 0:
        parser.error("--trial-timeout-seconds must be positive")
    run_dir = RESULT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    global_lock = (REPO / "results" / ".sm110_gpu_campaign.lock").open("w")
    try:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another SM110 GPU campaign holds the global lock") from error
    spec = {
        "schema_version": 1, "run_id": args.run_id,
        "campaign": "sm110_gemm_component_closure",
        "expected_sm_count": EXPECTED_SMS, "trials": TRIALS,
        "trial_timeout_seconds": args.trial_timeout_seconds,
        "static_only": args.static_only,
        "host_compiler": args.host_compiler,
        "nvcc_host_undef_gnu_source": args.nvcc_host_undef_gnu_source,
        "timing": "earliest CTA globaltimer start to latest CTA stop for TMA/TMEM/HBM/L2; CUDA events for epilogue",
        "generator": str(Path(__file__).relative_to(REPO)),
        "generator_sha256": sha256(Path(__file__)),
        "source_dependencies": {
            path: sha256(REPO / path) for path in SOURCE_DEPENDENCIES
        },
        "cases": CASES,
    }
    spec_text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    spec_path = run_dir / "run_spec.json"
    if spec_path.exists() and spec_path.read_text() != spec_text:
        raise RuntimeError("run-id exists with a different run contract")
    spec_path.write_text(spec_text)
    complete_marker = run_dir / "COMPLETE"
    if complete_marker.exists():
        complete_marker.unlink()
    write_status(run_dir, "running", current_case=None, completed_cases=0,
                 total_cases=len(CASES))
    if not args.static_only:
        snapshot = environment()
        if "MAXN" not in str(snapshot["power_mode"]).upper():
            raise RuntimeError("MAXN power mode is not proven")
        env_path = run_dir / "environment.json"
        if not env_path.exists():
            env_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        with (run_dir / "environment_snapshots.jsonl").open("a") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
    artifacts = compile_binaries(run_dir, args.host_compiler, args.nvcc_host_undef_gnu_source)
    results = []
    for case in CASES:
        write_status(run_dir, "running", current_case=case["id"],
                     completed_cases=len(results), total_cases=len(CASES))
        name = str(case["binary"])
        sass = Path(artifacts[name]["sass_path"]).read_text()
        missing_sass = [token for token in case["sass"] if token not in sass]
        if missing_sass:
            raise RuntimeError(f"{case['id']}: SASS tokens absent: {missing_sass}")
        base = {
            "case_id": case["id"], "resource": case["resource"],
            "source_path": str(Path(artifacts[name]["source"]).relative_to(REPO)),
            "source_sha256": artifacts[name]["source_sha256"],
            "binary_sha256": artifacts[name]["binary_sha256"],
            "binary_hash_path": f"build/{name}.binary.sha256",
            "sass_sha256": artifacts[name]["sass_sha256"],
            "sass_tokens": case["sass"],
        }
        if args.static_only:
            results.append({**base, "status": "static_ok"})
            continue
        case_dir = run_dir / "cases" / str(case["id"])
        case_dir.mkdir(parents=True, exist_ok=True)
        prior_path = case_dir / "result.json"
        trials_path = case_dir / "trials.jsonl"
        if prior_path.is_file() and trials_path.is_file():
            prior = json.loads(prior_path.read_text())
            trial_count = len([line for line in trials_path.read_text().splitlines() if line])
            if (prior.get("status") == "ok" and prior.get("trial_count") == TRIALS
                    and trial_count == TRIALS
                    and prior.get("source_sha256") == base["source_sha256"]
                    and prior.get("binary_sha256") == base["binary_sha256"]
                    and prior.get("sass_sha256") == base["sass_sha256"]):
                results.append(prior)
                print(f"SKIP {case['id']}: complete artifact match", flush=True)
                continue
        rows, rates = [], []
        for trial in range(1, TRIALS + 1):
            command = [str(artifacts[name]["binary"]), *case["args"]]
            outcome = run_bounded_trial(
                command, args.trial_timeout_seconds)
            if outcome["timed_out"]:
                timeout_record = {
                    "case_id": case["id"], "trial": trial,
                    "captured_at_utc": utc_now(), "command": command,
                    "timeout_seconds": args.trial_timeout_seconds,
                    "termination_failed": outcome["termination_failed"],
                    "returncode": outcome["returncode"],
                    "stdout": outcome["stdout"],
                }
                (case_dir / "timeout.json").write_text(
                    json.dumps(timeout_record, indent=2, sort_keys=True) + "\n")
                raise RuntimeError(
                    f"{case['id']} trial {trial} exceeded "
                    f"{args.trial_timeout_seconds}s "
                    f"(termination_failed={outcome['termination_failed']}); "
                    f"see {case_dir / 'timeout.json'}")
            proc = subprocess.CompletedProcess(
                command, outcome["returncode"], outcome["stdout"], None)
            fields = parse_kv(proc.stdout)
            if proc.returncode:
                raise RuntimeError(f"{case['id']} trial {trial} failed: {proc.stdout}")
            rate = validate_fields(case, fields)
            rates.append(rate)
            rows.append({"trial": trial, "captured_at_utc": utc_now(),
                         "returncode": proc.returncode, "raw_stdout": proc.stdout,
                         "fields": fields, "audited_rate_per_second": rate})
        trials_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        result = {**base, "status": "ok", "trial_count": len(rates),
                  "rate_unit": "element/s" if str(case["resource"]).startswith("epilogue.") else "B/s",
                  "rate_per_second_median": statistics.median(rates),
                  "rate_per_second_min": min(rates), "rate_per_second_max": max(rates)}
        (case_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
    status = "static_complete" if args.static_only else "complete"
    summary = {"schema_version": 1, "run_id": args.run_id, "status": status,
               "case_count": len(results), "results": results,
               "updated_at_utc": utc_now()}
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not args.static_only:
        complete_marker.write_text(f"run_id={args.run_id}\nsummary_sha256={sha256(summary_path)}\n")
        for artifact in artifacts.values():
            Path(artifact["binary"]).unlink()
    write_status(run_dir, status, current_case=None, completed_cases=len(results),
                 total_cases=len(CASES))
    print(json.dumps({"run_dir": str(run_dir), "status": status,
                      "case_count": len(results)}, indent=2))
    return 0


def mark_failed_from_argv(message: str) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id")
    args, _ = parser.parse_known_args()
    if not args.run_id or not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        return
    run_dir = RESULT_ROOT / args.run_id
    if run_dir.is_dir():
        write_status(run_dir, "failed", error=message)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        mark_failed_from_argv(str(exc))
        raise

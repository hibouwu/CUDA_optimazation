from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from .model import (
    Capacity, EvidenceKind, ModelError, audit_inputs, precision_specs,
)


THOR_T5000_HARDWARE_ID = "thor_t5000_sm110_20sm"
THOR_T5000_OPERATING_MODE = "MAXN"


def _audited_bundle(
    run_dir: Path,
    *,
    repo_root: Path,
    auditor_relative: str,
    extra_audit_args: tuple[str, ...] = (),
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    repo_root = repo_root.resolve()
    run_dir = run_dir.resolve()
    try:
        relative_run = run_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ModelError(f"campaign directory is outside repo: {run_dir}") from exc
    auditor = repo_root / auditor_relative
    if not auditor.is_file():
        raise ModelError(f"canonical campaign auditor is missing: {auditor}")
    audit = subprocess.run(
        [sys.executable, str(auditor), str(run_dir), *extra_audit_args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if audit.returncode:
        raise ModelError(f"campaign audit failed for {relative_run}:\n{audit.stdout}")
    spec = json.loads((run_dir / "run_spec.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    return repo_root, relative_run, spec, summary


def _common_artifacts(relative_run: Path, case_id: str) -> list[str]:
    return [
        str(relative_run / "run_spec.json"),
        str(relative_run / "environment.json"),
        str(relative_run / "environment_snapshots.jsonl"),
        str(relative_run / "campaign_status.json"),
        str(relative_run / "progress.jsonl"),
        str(relative_run / "launcher.log"),
        str(relative_run / "summary.json"),
        str(relative_run / "COMPLETE"),
        str(relative_run / "cases" / case_id / "trials.jsonl"),
        str(relative_run / "cases" / case_id / "result.json"),
    ]


def _finalize_capacities(
    capacities: list[Capacity], *, repo_root: Path,
) -> list[Capacity]:
    findings = audit_inputs(capacities, repo_root=repo_root.resolve())
    errors = [row for row in findings if row["severity"] == "error"]
    if errors:
        raise ModelError(
            "imported campaign capacity provenance failed: "
            + json.dumps(errors, sort_keys=True)
        )
    return sorted(capacities, key=lambda row: row.capacity_id)


def import_compute_campaign(
    run_dir: Path,
    *,
    repo_root: Path,
    require_ncu: bool = False,
) -> list[Capacity]:
    _, relative_run, spec, summary = _audited_bundle(
        run_dir,
        repo_root=repo_root,
        auditor_relative="microbench/sm110_gemm_campaign/audit_campaign.py",
        extra_audit_args=("--require-ncu",) if require_ncu else (),
    )
    manifest = {row["case_id"]: row for row in spec["manifest"]}
    known = precision_specs()
    capacities: list[Capacity] = []
    for result in summary["results"]:
        case_id = str(result["case_id"])
        entry = manifest[case_id]
        precision_id = str(result["precision_id"])
        if precision_id not in known:
            raise ModelError(f"{case_id}: unknown precision {precision_id}")
        if entry["launch"] != "full_sm_4warp_block":
            # Single-block latency/throughput is preserved in the raw bundle,
            # but is not a full-GPU service-rate capacity.
            continue
        m = int(entry["m"])
        n = int(entry["n"])
        k = int(entry["precision"]["k"])
        artifacts = _common_artifacts(relative_run, case_id) + [
            str(relative_run / "cases" / case_id / "source.cu"),
            str(relative_run / "cases" / case_id / "descriptor.json"),
            str(relative_run / "cases" / case_id / "compile_command.json"),
            str(relative_run / "cases" / case_id / "compile.log"),
            str(relative_run / "cases" / case_id / "sass.txt"),
            str(relative_run / "cases" / case_id / "binary.sha256"),
        ]
        ncu = result.get("ncu", {})
        if ncu.get("selected") is True and ncu.get("returncode") == 0:
            artifacts.extend(
                [
                    str(relative_run / "cases" / case_id / "ncu/profile.ncu-rep"),
                    str(relative_run / "cases" / case_id / "ncu/profile.log"),
                ]
            )
        capacity = Capacity(
            capacity_id=f"{case_id}_closure_median",
            resource=(
                f"{known[precision_id].compute_resource}.m{m}n{n}"
            ),
            rate_per_second=float(result["rate_per_second_median"]),
            work_unit=str(result["work_unit"]),
            evidence_kind=EvidenceKind.MEASURED_SUSTAINED,
            source_id=f"sm110_compute_campaign:{spec['run_id']}:{case_id}",
            source_path=str(relative_run / "cases" / case_id / "result.json"),
            source_locator="rate_per_second_median",
            original_value=float(result["rate_per_second_median"]),
            original_unit=f"{result['work_unit']}/s",
            condition=(
                f"{spec['timed_scope']}; {spec['residency']}; full GPU; "
                f"MMA m{m}n{n}k{k}; CTA group 1; {spec['expected_sm_count']} SM"
            ),
            qualification="closure_qualified",
            trial_count=int(result["trial_count"]),
            artifact_paths=tuple(artifacts),
            applicable_precision_ids=(precision_id,),
            applicable_mma_shapes=(f"m{m}n{n}k{k}",),
            applicable_cta_groups=(1,),
            applicable_sm_counts=(int(spec["expected_sm_count"]),),
            applicable_hardware_ids=(THOR_T5000_HARDWARE_ID,),
            applicable_operating_modes=(THOR_T5000_OPERATING_MODE,),
            applicable_threads_per_cta=(128,),
            applicable_resident_ctas_per_sm=(1,),
            timed_scope=str(spec["timed_scope"]),
            measurement_operand_residency="smem_operands",
        )
        capacity.validate()
        capacities.append(capacity)
    return _finalize_capacities(capacities, repo_root=repo_root)


def import_component_campaign(
    run_dir: Path,
    *,
    repo_root: Path,
) -> list[Capacity]:
    _, relative_run, spec, summary = _audited_bundle(
        run_dir,
        repo_root=repo_root,
        auditor_relative=(
            "microbench/sm110_gemm_component_campaign/audit_campaign.py"
        ),
    )
    cases = {row["id"]: row for row in spec["cases"]}
    capacities: list[Capacity] = []
    for result in summary["results"]:
        case_id = str(result["case_id"])
        case = cases[case_id]
        raw_resource = str(result["resource"])
        if raw_resource == "tmem.accumulator_readback":
            resource = (
                "tmem.readback"
                if case_id == "tmem_ld_32x32b_x16_warps4"
                else case_id.replace(
                    "tmem_ld_32x32b_", "tmem.readback."
                ).replace("_warps", ".warps")
            )
        else:
            resource = {
                "tma.l2_hit_ingress": "tma.smem_ingress.per_sm",
                "tma.dram_stream_ingress": "tma.hbm",
                "tma.l2_hit_ingress.serial32k":
                    "tma.smem_ingress.diagnostic.serial32k.per_sm",
                "tma.dram_stream_ingress.serial32k":
                    "tma.hbm.diagnostic.serial32k",
                "tma.l2_hit_ingress.inflight4":
                    "tma.smem_ingress.per_sm.inflight4",
                "tma.dram_stream_ingress.inflight4": "tma.hbm.inflight4",
                "tmem.scale_ingress": "tmem.scale_ingress",
                "hbm.read": "hbm.read",
                "hbm.write": "hbm.write",
                "l2.read": "l2.read",
                "l2.write": "l2.write",
                "epilogue.nvfp4_requant": "epilogue.nvfp4_requant",
            }.get(raw_resource)
        if resource is None:
            raise ModelError(f"{case_id}: unknown component resource {raw_resource}")
        unit = "element" if str(result["rate_unit"]) == "element/s" else "byte"
        artifacts = _common_artifacts(relative_run, case_id) + [
            str(relative_run / str(result["binary_hash_path"])),
            str(relative_run / "build" / f"{case['binary']}.compile_command.json"),
            str(relative_run / "build" / f"{case['binary']}.compile.log"),
            str(relative_run / "build" / f"{case['binary']}.sass.txt"),
            str(result["source_path"]),
        ]
        tma_tile_bytes: tuple[int, ...] = ()
        tma_destination_slots: tuple[int, ...] = ()
        ldtm_registers: tuple[int, ...] = ()
        readback_warps: tuple[int, ...] = ()
        threads_per_cta: tuple[int, ...] = ()
        resident_ctas_per_sm: tuple[int, ...] = ()
        measurement_residency = "unspecified"
        access_patterns: tuple[str, ...] = ()
        if resource.startswith("tma."):
            args = list(case["args"])
            tile_index = args.index("--tile-bytes") + 1
            tile_bytes = int(args[tile_index])
            tma_tile_bytes = (
                (tile_bytes, tile_bytes * 2)
                if "--pattern" in args
                and args[args.index("--pattern") + 1] == "tc5a-ab"
                else (tile_bytes,)
            )
            tma_destination_slots = (int(args[args.index("--slots") + 1]),)
            threads_per_cta = (int(args[args.index("--threads") + 1]),)
            resident_ctas_per_sm = (
                int(args[args.index("--blocks-per-sm") + 1]),
            )
            measurement_residency = (
                "l2_hit_requests"
                if "--mode" in args
                and args[args.index("--mode") + 1] == "l2-hit"
                else "dram_stream_requests"
            )
        elif resource == "tmem.readback":
            match = re.fullmatch(r"tmem_ld_32x32b_x(\d+)_warps(\d+)", case_id)
            if not match:
                raise ModelError(f"{case_id}: cannot decode TMEM applicability")
            ldtm_registers = (int(match.group(1)),)
            readback_warps = (int(match.group(2)),)
            # The canonical kernel always launches 128 threads and the frozen
            # campaign requires exactly one CTA per SM.
            threads_per_cta = (128,)
            resident_ctas_per_sm = (1,)
            measurement_residency = "tmem_operands"
        elif resource.startswith("tmem.readback."):
            match = re.fullmatch(r"tmem_ld_32x32b_x(\d+)_warps(\d+)", case_id)
            if not match:
                raise ModelError(f"{case_id}: cannot decode TMEM applicability")
            ldtm_registers = (int(match.group(1)),)
            readback_warps = (int(match.group(2)),)
            threads_per_cta = (128,)
            resident_ctas_per_sm = (1,)
            measurement_residency = "tmem_operands"
        elif resource in {"hbm.read", "hbm.write", "l2.read", "l2.write"}:
            args = list(case["args"])
            threads_per_cta = (int(args[args.index("--threads") + 1]),)
            resident_ctas_per_sm = (
                int(args[args.index("--blocks-per-sm") + 1]),
            )
            measurement_residency = (
                "dram_stream_requests"
                if resource.startswith("hbm.")
                else "l2_hit_requests"
            )
            access_patterns = ("coalesced_load_store",)
        capacity = Capacity(
            capacity_id=f"{case_id}_closure_median",
            resource=resource,
            rate_per_second=float(result["rate_per_second_median"]),
            work_unit=unit,
            evidence_kind=EvidenceKind.MEASURED_SUSTAINED,
            source_id=f"sm110_component_campaign:{spec['run_id']}:{case_id}",
            source_path=str(relative_run / "cases" / case_id / "result.json"),
            source_locator="rate_per_second_median",
            original_value=float(result["rate_per_second_median"]),
            original_unit=str(result["rate_unit"]),
            condition=(
                f"{spec['timing']}; full GPU; {spec['expected_sm_count']} SM; "
                f"component contract {raw_resource}"
            ),
            qualification="closure_qualified",
            trial_count=int(result["trial_count"]),
            artifact_paths=tuple(artifacts),
            applicable_sm_counts=(int(spec["expected_sm_count"]),),
            applicable_hardware_ids=(THOR_T5000_HARDWARE_ID,),
            applicable_operating_modes=(THOR_T5000_OPERATING_MODE,),
            measurement_operand_residency=measurement_residency,
            applicable_tma_tile_bytes=tma_tile_bytes,
            applicable_tma_destination_slots=tma_destination_slots,
            applicable_tmem_load_registers=ldtm_registers,
            applicable_readback_warps=readback_warps,
            applicable_threads_per_cta=threads_per_cta,
            applicable_resident_ctas_per_sm=resident_ctas_per_sm,
            applicable_access_patterns=access_patterns,
            timed_scope=str(spec["timing"]),
        )
        capacity.validate()
        capacities.append(capacity)
    return _finalize_capacities(capacities, repo_root=repo_root)


def import_tma_payload_campaign(
    run_dir: Path,
    *,
    repo_root: Path,
) -> list[Capacity]:
    _, relative_run, spec, summary = _audited_bundle(
        run_dir,
        repo_root=repo_root,
        auditor_relative="microbench/sm110_tma_payload_campaign/audit_campaign.py",
    )
    cases = {row["id"]: row for row in spec["cases"]}
    capacities: list[Capacity] = []
    for result in summary["results"]:
        case_id = str(result["case_id"])
        cases[case_id]  # Fail if the audited summary is not in the immutable manifest.
        ncu = result["ncu"]
        artifacts = _common_artifacts(relative_run, case_id) + [
            str(relative_run / str(result["binary_hash_path"])),
            str(relative_run / "build" / "compile_command.json"),
            str(relative_run / "build" / "compile.log"),
            str(relative_run / str(result["sass_path"])),
            str(result["source_path"]),
            str(relative_run / "cases" / case_id / str(ncu["report_path"])),
            str(relative_run / "cases" / case_id / str(ncu["raw_path"])),
            str(relative_run / "cases" / case_id / str(ncu["stderr_path"])),
            str(relative_run / "cases" / case_id / "ncu" / "summary.json"),
        ]
        capacity = Capacity(
            capacity_id=f"{case_id}_closure_median",
            resource=str(result["resource"]),
            rate_per_second=float(result["rate_per_second_median"]),
            work_unit="byte",
            evidence_kind=EvidenceKind.MEASURED_SUSTAINED,
            source_id=f"sm110_tma_payload_campaign:{spec['run_id']}:{case_id}",
            source_path=str(relative_run / "cases" / case_id / "result.json"),
            source_locator="rate_per_second_median",
            original_value=float(result["rate_per_second_median"]),
            original_unit="B/s",
            condition=(
                f"{spec['timed_scope']}; NCU-confirmed {result['residency']}; "
                f"{spec['expected_sm_count']} SM"
            ),
            qualification="closure_qualified",
            trial_count=int(result["trial_count"]),
            artifact_paths=tuple(artifacts),
            applicable_sm_counts=(int(spec["expected_sm_count"]),),
            applicable_hardware_ids=(THOR_T5000_HARDWARE_ID,),
            applicable_operating_modes=(THOR_T5000_OPERATING_MODE,),
            measurement_operand_residency=(
                "l2_hit_requests"
                if str(result["residency"]) == "hot_l2"
                else "dram_stream_requests"
            ),
            applicable_tma_tile_bytes=(int(result["tile_bytes"]),),
            applicable_tma_destination_slots=(
                int(result["destination_slots"]),
            ),
            applicable_threads_per_cta=(int(result["threads_per_cta"]),),
            applicable_resident_ctas_per_sm=(
                int(result["resident_ctas_per_sm"]),
            ),
            timed_scope=str(spec["timed_scope"]),
        )
        capacity.validate()
        capacities.append(capacity)
    return _finalize_capacities(capacities, repo_root=repo_root)


def import_memory_duplex_campaign(
    run_dir: Path,
    *,
    repo_root: Path,
) -> list[Capacity]:
    """Import ratio-qualified joint memory service without upgrading proxies.

    Hot-L2 cases become ``l2.duplex`` capacities scoped to one exact issued-byte
    read:write ratio.  Thor's cold cases deliberately remain
    ``hbm.duplex.proxy`` because their NCU contract proves external read misses
    and L2 write-path issue, not physical external write bytes.  Consequently
    they cannot satisfy the model's physical ``hbm.duplex`` demand.
    """
    _, relative_run, spec, summary = _audited_bundle(
        run_dir,
        repo_root=repo_root,
        auditor_relative=(
            "microbench/sm110_memory_duplex_campaign/audit_campaign.py"
        ),
    )
    cases = {str(row["id"]): row for row in spec["cases"]}
    capacities: list[Capacity] = []
    for result in summary["results"]:
        case_id = str(result["case_id"])
        case = cases[case_id]
        read_ops = int(case["read_operations"])
        write_ops = int(case["write_operations"])
        residency = str(result["residency"])
        is_cold_proxy = residency == "cold_hbm"
        if is_cold_proxy and result.get("external_write_bytes_proven") is not False:
            raise ModelError(
                f"{case_id}: cold proxy must not claim external write bytes"
            )
        if not is_cold_proxy and residency != "hot_l2":
            raise ModelError(f"{case_id}: unsupported duplex residency")
        ncu = result["ncu"]
        artifacts = _common_artifacts(relative_run, case_id) + [
            str(relative_run / "build" / "compile_command.json"),
            str(relative_run / "build" / "compile.log"),
            str(relative_run / "build" / "sass.txt"),
            str(relative_run / str(result["function_sass_path"])),
            str(result["source_path"]),
            str(relative_run / "cases" / case_id / str(ncu["report_path"])),
            str(relative_run / "cases" / case_id / str(ncu["raw_path"])),
            str(relative_run / "cases" / case_id / str(ncu["stderr_path"])),
            str(relative_run / "cases" / case_id / "ncu" / "summary.json"),
            str(relative_run / "plots" / "manifest.json"),
        ]
        args = list(case["args"])
        capacity = Capacity(
            capacity_id=f"{case_id}_closure_median",
            resource="hbm.duplex.proxy" if is_cold_proxy else "l2.duplex",
            rate_per_second=float(result["rate_per_second_median"]),
            work_unit="byte",
            evidence_kind=EvidenceKind.MEASURED_JOINT,
            source_id=(
                f"sm110_memory_duplex_campaign:{spec['run_id']}:{case_id}"
            ),
            source_path=str(relative_run / "cases" / case_id / "result.json"),
            source_locator="rate_per_second_median",
            original_value=float(result["rate_per_second_median"]),
            original_unit="B/s",
            condition=(
                "NCU-qualified cold external-read/L2-write-path proxy; "
                "external_write_bytes_proven=false"
                if is_cold_proxy
                else "NCU-qualified hot-L2 simultaneous read/write service"
            ),
            qualification="closure_qualified",
            trial_count=int(result["trial_count"]),
            artifact_paths=tuple(artifacts),
            applicable_sm_counts=(int(spec["expected_sm_count"]),),
            applicable_hardware_ids=(THOR_T5000_HARDWARE_ID,),
            applicable_operating_modes=(THOR_T5000_OPERATING_MODE,),
            applicable_residencies=(residency,),
            applicable_read_write_ratios=(f"{read_ops}:{write_ops}",),
            applicable_threads_per_cta=(
                int(args[args.index("--threads") + 1]),
            ),
            applicable_resident_ctas_per_sm=(
                int(args[args.index("--blocks-per-sm") + 1]),
            ),
            timed_scope="full_grid_globaltimer_span",
            measurement_operand_residency=(
                "dram_stream_requests" if is_cold_proxy else "l2_hit_requests"
            ),
            residency_evidence_qualification="ncu_proven",
        )
        capacity.validate()
        capacities.append(capacity)
    return _finalize_capacities(capacities, repo_root=repo_root)

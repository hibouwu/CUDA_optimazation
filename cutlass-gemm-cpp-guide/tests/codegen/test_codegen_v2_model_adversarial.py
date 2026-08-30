#!/usr/bin/env python3
"""Adversarial tests for v2 schema, fingerprint, artifact, journal, and result closure."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from codegen_v2.model import (  # noqa: E402
    ContractError,
    explicit_subject_inventory_sha256,
    canonical_json_bytes,
    contract_sha256,
    expected_execution_plan,
    expected_actual_docker_argv,
    fingerprint_payload,
    hash_input_tree,
    journal_event_hash,
    load_strict_json,
    sha256_bytes,
    sha256_file,
    validate_artifact_manifest,
    validate_fingerprint,
    validate_instance,
    validate_device_kernel_symbol,
    validate_result,
    validate_static_pass_evidence,
)
from codegen_v2.attribution import parse_ptx_entries  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def reference(identifier: str, root: Path, path: Path) -> dict[str, str]:
    return {
        "id": identifier,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def materialize_actual_docker_argv(
    normalized: list[str], root: Path, archive: Path
) -> list[str]:
    """Turn the portable expected argv into the raw argv recorded by Docker."""
    origin_user = f"{os.getuid()}:{os.getgid()}"
    replacements = {
        "<host-user>": origin_user,
        "<guide-root>:/workspace:ro": f"{root}:/workspace:ro",
        "<attempt-root>:/out": f"{archive}:/out",
        "<attempt-root>:/out:ro": f"{archive}:/out:ro",
    }
    return [replacements.get(argument, argument) for argument in normalized]


def event(seq: int, previous: str | None, event_type: str, attempt: str, instance: str, fp: str):
    value = {
        "seq": seq,
        "event_type": event_type,
        "attempt_id": attempt,
        "instance_id": instance,
        "fingerprint_id": fp,
        "timestamp_utc": f"2026-08-30T00:00:0{seq}Z",
        "previous_event_sha256": previous,
        "payload": {},
        "event_sha256": "0" * 64,
    }
    value["event_sha256"] = journal_event_hash(value)
    return value


def make_fixture(contract_mutator=None) -> tuple[Path, dict[str, Path]]:
    root = Path(tempfile.mkdtemp(prefix="codegen-v2-model-"))
    (root / "tests/codegen/schemas").mkdir(parents=True)
    shutil.copy2(
        ROOT / "tests/codegen/static_codegen_contract.json",
        root / "tests/codegen/static_codegen_contract.json",
    )
    if contract_mutator is not None:
        contract_value = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
        contract_mutator(contract_value)
        write_json(root / "tests/codegen/static_codegen_contract.json", contract_value)
    for schema in (ROOT / "tests/codegen/schemas").glob("*.json"):
        shutil.copy2(schema, root / "tests/codegen/schemas" / schema.name)
    contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
    for component in contract["harness_components"]:
        target = root / component
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / component, target)
    for relative in (
        "versions.lock.json",
        "tests/codegen/sm110a_tensor_schedule_tags.json",
        "tests/codegen/sm110a_schedule_reference_inventory.json",
        "tests/codegen/sm110a_auto_control_inventory.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    instance_id = "fixture"
    attempt_id = "fixture.fixture.a001"
    instance_dir = root / "tests/codegen/instances/fixture"
    instance_dir.mkdir(parents=True)
    config_path = instance_dir / "config.hpp"
    kernel_path = instance_dir / "kernel.cu"
    witness_path = instance_dir / "type_witness.cu"
    config_path.write_text(
        "namespace guide::codegen::fixture {\n"
        "using Config = Fixture<cutlass::gemm::KernelFixtureSm100>;\n"
        "}\n",
        encoding="utf-8",
    )
    kernel_path.write_text(
        '#include "config.hpp"\n'
        "using GemmKernel = guide::codegen::fixture::Config::GemmKernel;\n"
        "template __global__ void cutlass::device_kernel<GemmKernel>(\n"
        "    CUTLASS_GRID_CONSTANT GemmKernel::Params const params);\n",
        encoding="utf-8",
    )
    witness_path.write_text(
        '#include "config.hpp"\n'
        "int main(){ guide::write_codegen_type_report<guide::codegen::fixture::Config>(\n"
        "    std::cout, \"fixture\"); }\n",
        encoding="utf-8",
    )
    anchor_path = root / "third_party/cutlass/reference.cpp"
    anchor_path.parent.mkdir(parents=True)
    anchor_path.write_text("KernelFixtureSm100\n", encoding="utf-8")
    inventory_path = root / "tests/codegen/sm110a_schedule_reference_inventory.json"
    write_json(
        inventory_path,
        {
            "entries": [
                {
                    "tag": "KernelFixtureSm100",
                    "reference_class": "official_cpp_explicit_or_conditional",
                    "path": "reference.cpp",
                    "line": 1,
                    "anchor_line_sha256": sha256_bytes(b"KernelFixtureSm100"),
                }
            ]
        },
    )
    tag_manifest_path = root / "tests/codegen/sm110a_tensor_schedule_tags.json"
    tag_manifest = load_strict_json(tag_manifest_path)
    tag_manifest["entries"].append(
        {
            "tag": "KernelFixtureSm100",
            "group": "dense",
            "status": "NOT_CHECKED",
            "current_case_ids": [],
            "fresh_replay_required": True,
        }
    )
    write_json(tag_manifest_path, tag_manifest)
    contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
    contract["inventory_contract_sha256"]["explicit_tags"] = (
        explicit_subject_inventory_sha256(root)
    )
    write_json(root / "tests/codegen/static_codegen_contract.json", contract)
    contract_hash = contract_sha256(root)
    instance = {
        "schema_version": 1,
        "contract_sha256": contract_hash,
        "objective_sha256": contract["objective_sha256"],
        "freshness_epoch": contract["freshness_epoch"],
        "instance_id": instance_id,
        "subject": {"kind": "explicit_schedule_tag", "id": "KernelFixtureSm100", "group": "dense"},
        "provenance": {
            "reference_inventory_sha256": sha256_file(inventory_path),
            "reference_class": "official_cpp_explicit_or_conditional",
            "source_anchors": [
                {
                    "path": "reference.cpp",
                    "line_start": 1,
                    "line_end": 1,
                    "sha256": sha256_bytes(b"KernelFixtureSm100"),
                }
            ],
            "parent_instance_id": None,
            "derivation_axis": "fixture",
        },
        "target": contract["target"],
        "translation_units": {
            "config": {"path": config_path.relative_to(root).as_posix(), "sha256": sha256_file(config_path)},
            "kernel": {"path": kernel_path.relative_to(root).as_posix(), "sha256": sha256_file(kernel_path)},
            "type_witness": {"path": witness_path.relative_to(root).as_posix(), "sha256": sha256_file(witness_path)},
        },
        "declared_config": {
            "family": "fixture",
            "operator_class": "cutlass::arch::OpClassTensorOp",
            "elements": {"a": "half", "b": "half", "accumulator": "float", "d": "float"},
            "layouts": {"a": "RowMajor", "b": "ColumnMajor", "d": "RowMajor"},
            "alignments": {"a": 8, "b": 8, "d": 4},
            "mma_tile_mnk": [128, 128, 64],
            "cluster_mnk": [1, 1, 1],
            "stage_policy": "Auto",
            "mainloop_schedule": "cutlass::gemm::KernelFixtureSm100",
            "epilogue_schedule": "NoSmem",
            "kernel_problem_shape": "cute::Shape<int,int,int,int>",
            "tile_scheduler": "void",
            "cta_group": 1,
            "operand_source": "SS",
            "mechanism": {
                "pointer_mode": "single",
                "transforms": {"a": "identity", "b": "identity"},
                "block_scaled": {"enabled": False, "scale_a": None, "scale_b": None, "vector_size_a": None, "vector_size_b": None},
                "sparse": {"enabled": False, "metadata": None},
                "fast_fp32": {"enabled": False, "atom_model": None},
                "complex": {"enabled": False, "representation": None},
            },
        },
        "static_contract": {
            "required_layers": contract["layer_order"],
            "target_entity": contract["function_selector"]["target_cpp_entity"],
            "function_selector": contract["function_selector"]["policy"],
            "ptx": {
                "required": [
                    {"id": "tma", "regex": "cp\\.async\\.bulk\\.tensor(?:\\..*)?", "min_count": 1, "max_count": None},
                    {"id": "mma", "regex": "tcgen05\\.mma\\.cta_group::1(?:\\..*)?", "min_count": 1, "max_count": None},
                    {"id": "load", "regex": "tcgen05\\.ld(?:\\..*)?", "min_count": 1, "max_count": None},
                ],
                "forbidden": [
                    {"id": "mma2", "regex": "tcgen05\\.mma\\.cta_group::2(?:\\..*)?", "min_count": 0, "max_count": 0}
                ],
            },
            "sass": {
                "required": [
                    {"id": "tma", "regex": "UTMALDG(?:\\..*)?", "min_count": 1, "max_count": None},
                    {"id": "mma", "regex": "UTCHMMA(?:\\..*)?", "min_count": 1, "max_count": None},
                    {"id": "load", "regex": "LDTM(?:\\..*)?", "min_count": 1, "max_count": None},
                ],
                "forbidden": [
                    {"id": "mma2", "regex": "UTCHMMA\\.2CTA(?:\\..*)?", "min_count": 0, "max_count": 0}
                ],
            },
        },
        "hypothesis": {
            "expected_outcome": "STATIC_PASS",
            "failure_domain": None,
            "failure_layer": None,
            "diagnostic_patterns": [],
            "control_instance_id": None,
        },
    }
    instance_path = root / "tests/codegen/instances/fixture/instance.json"
    write_json(instance_path, instance)
    harness_paths = [root / "tests/codegen/static_codegen_contract.json"]
    harness_paths.extend(root / value for value in contract["record_schemas"].values())
    harness_paths.extend(root / value for value in contract["harness_components"])
    input_paths = harness_paths + [instance_path, config_path, kernel_path, witness_path]
    input_paths.extend(
        root / relative
        for relative in (
            "versions.lock.json",
            "tests/codegen/sm110a_schedule_reference_inventory.json",
        )
    )
    input_records = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(set(input_paths))
    ]
    input_tree = sha256_bytes(
        canonical_json_bytes(sorted((item["path"], item["sha256"]) for item in input_records))
    )
    execution_plan = expected_execution_plan(instance)
    tool_versions = {
        "nvcc": "V13.0.88",
        "ptxas": "V13.0.88",
        "cuobjdump": "V13.0.85",
        "nvdisasm": "V13.0.85",
        "host_compiler": "g++ 13.3.0",
    }
    toolchain = {
        "container_digest": contract["toolchain"]["container_digest"],
        "versions_lock_sha256": contract["toolchain"]["versions_lock_file_sha256"],
        "executables": {
            name: {
                "path": f"/tool/{name}",
                "binary_sha256": contract["toolchain"]["tool_binary_sha256"][name],
                "version": version,
                "version_sha256": sha256_bytes((version + "\n").encode()),
            }
            for name, version in tool_versions.items()
        },
    }
    fingerprint = {
        "schema_version": 1,
        "contract_sha256": contract_hash,
        "objective_sha256": contract["objective_sha256"],
        "freshness_epoch": contract["freshness_epoch"],
        "fingerprint_id": "pending",
        "instance_ref": reference(instance_id, root, instance_path),
        "source_closure": {
            "cutlass_git_sha": contract["target"]["cutlass_git_sha"],
            "cutlass_clean": True,
            "inputs": input_records,
            "input_tree_sha256": input_tree,
        },
        "toolchain": toolchain,
        "execution_plan": execution_plan,
        "environment": {"allowed": {"LC_ALL": "C"}, "asserted_absent": contract["compile_plan"]["asserted_absent_environment"]},
        "component_hashes": {
            "instance": sha256_file(instance_path),
            "source": input_tree,
            "toolchain": sha256_bytes(canonical_json_bytes(toolchain)),
            "harness": sha256_bytes(canonical_json_bytes(sorted((path.relative_to(root).as_posix(), sha256_file(path)) for path in harness_paths))),
            "plan": sha256_bytes(canonical_json_bytes(execution_plan)),
        },
        "overall_sha256": "0" * 64,
    }
    fingerprint_without = dict(fingerprint)
    fingerprint_without.pop("overall_sha256")
    fingerprint_without.pop("fingerprint_id")
    fingerprint["overall_sha256"] = sha256_bytes(canonical_json_bytes(fingerprint_without))
    fingerprint_id = f"fp-{instance_id}-{fingerprint['overall_sha256'][:12]}"
    fingerprint["fingerprint_id"] = fingerprint_id
    fingerprint_path = root / "evidence/codegen-sm110a-v2/fingerprints" / f"{fingerprint_id}.json"
    write_json(fingerprint_path, fingerprint)

    archive = root / "artifacts/codegen-sm110a-v2/fixture/attempts/fixture.fixture.a001"
    archive.mkdir(parents=True)
    target_symbol = "_ZN7cutlass13device_kernelI13FixtureKernelEEvNT_6ParamsE"
    ptx_text = (
        ".version 9.0\n.target sm_110a\n.address_size 64\n"
        f".entry {target_symbol}()\n{{\n  cp.async.bulk.tensor.3d %r1, %r2;\n"
        "  tcgen05.mma.cta_group::1.kind::f16 %r1, %r2;\n"
        "  tcgen05.ld.sync.aligned{%r1},[%r2];\n}\n"
    )
    ptx_block = parse_ptx_entries(ptx_text)[0].text + "\n"
    sass_function = {
        "function-name": target_symbol,
        "start": 0,
        "length": 48,
        "sass-instructions": [
            {"opcode": "UTMALDG.3D", "operands": "R1,R2"},
            {"opcode": "UTCHMMA", "operands": "R1,R2"},
            {"opcode": "LDTM.X64", "operands": "R1,R2"},
        ],
    }
    nvjson = json.dumps([
        {"SM": {"version": {"major": 11, "minor": 0}}, ".note.nv.tkinfo": {"tki_toolOptions": "-arch sm_110a -m 64 "}},
        [sass_function],
    ], separators=(",", ":"))
    type_output_value = {
        "schema_version": 1,
        "instance_id": instance_id,
        "resolved_types": {
            "collective_mainloop": "collective_mainloop",
            "mainloop_builder": "mainloop_builder",
            "mainloop_builder_collective_op": "collective_mainloop",
            "epilogue_builder": "epilogue_builder",
            "epilogue_builder_collective_op": "collective_epilogue",
            "dispatch_policy": "dispatch_policy",
            "dispatch_schedule": "dispatch_schedule",
            "tiled_mma": "tiled_mma",
            "mma_atom": "mma_atom_SS",
            "mainloop_dispatch_policy": "dispatch_policy",
            "mainloop_tiled_mma": "tiled_mma",
            "tiled_mma_atom": "mma_atom_SS",
            "mma_value_type_a": "half",
            "mma_value_type_b": "half",
            "mma_value_type_c": "float",
            "gemm_kernel": "cutlass::gemm::kernel::GemmUniversal<Fixture>",
            "gmem_tiled_copy_a": "copy_a",
            "gmem_tiled_copy_b": "copy_b",
            "smem_layout_atom_a": "layout_a",
            "smem_layout_atom_b": "layout_b",
            "smem_copy_atom_a": "void",
            "smem_copy_atom_b": "void",
            "collective_epilogue": "collective_epilogue",
            "kernel_collective_mainloop": "collective_mainloop",
            "kernel_collective_epilogue": "collective_epilogue",
            "config_arch_tag": "cutlass::arch::Sm100",
            "config_operator_class": "cutlass::arch::OpClassTensorOp",
            "config_element_a": "half",
            "config_element_b": "half",
            "config_element_accumulator": "float",
            "config_element_d": "float",
            "config_layout_a": "cutlass::layout::RowMajor",
            "config_layout_b": "cutlass::layout::ColumnMajor",
            "config_layout_d": "cutlass::layout::RowMajor",
            "config_mainloop_schedule": "cutlass::gemm::KernelFixtureSm100",
            "config_epilogue_schedule": "NoSmem",
            "config_stage_policy": "Auto",
            "config_problem_shape": "cute::Shape<int,int,int,int>",
            "config_tile_scheduler": "void"
        },
        "resolved_optional_types": {},
        "resolved_values": {"mainloop_stages": 2, "scheduler_stages": 1, "accumulator_stages": 1, "atom_shape_mnk": [128,128,16], "mma_tile_mnk": [128,128,64], "cluster_mnk": [1,1,1], "scale_vector_size": 0, "scale_vector_size_a": 0, "scale_vector_size_b": 0, "element_a_sparsity": 0, "alignment_a": 8, "alignment_b": 8, "alignment_d": 4, "mainloop_shared_storage_bytes": 1024, "epilogue_shared_storage_bytes": 0, "kernel_shared_storage_bytes": 1024},
    }
    ptx_result_values = [
        {"id":"tma","required":True,"match_count":1,"matched_opcodes":["cp.async.bulk.tensor.3d"]},
        {"id":"mma","required":True,"match_count":1,"matched_opcodes":["tcgen05.mma.cta_group::1.kind::f16"]},
        {"id":"load","required":True,"match_count":1,"matched_opcodes":["tcgen05.ld.sync.aligned"]},
        {"id":"mma2","required":False,"match_count":0,"matched_opcodes":[]}
    ]
    sass_result_values = [
        {"id":"tma","required":True,"match_count":1,"matched_opcodes":["UTMALDG.3D"]},
        {"id":"mma","required":True,"match_count":1,"matched_opcodes":["UTCHMMA"]},
        {"id":"load","required":True,"match_count":1,"matched_opcodes":["LDTM.X64"]},
        {"id":"mma2","required":False,"match_count":0,"matched_opcodes":[]}
    ]
    artifacts = []
    for artifact_id, role, source_file, parents in (
        ("config_source", "KERNEL_SOURCE_SNAPSHOT", config_path, []),
        ("kernel_source", "KERNEL_SOURCE_SNAPSHOT", kernel_path, ["config_source"]),
        ("type_source", "TYPE_WITNESS_SOURCE_SNAPSHOT", witness_path, ["config_source"]),
    ):
        path = archive / artifact_id
        shutil.copy2(source_file, path)
        artifacts.append({"artifact_id":artifact_id,"role":role,"path":path.relative_to(root).as_posix(),"storage":"ignored_archive","media_type":"text/plain","sha256":sha256_file(path),"size_bytes":path.stat().st_size,"producer_step_id":"snapshot_sources","parent_artifact_ids":parents,"sealed":True})
    full_files = {
        "type_executable": ("TYPE_WITNESS_EXECUTABLE", b"\x7fELFtype-witness", ["type_source", "config_source"], "compile_type_witness"),
        "fatbin": ("FATBIN", bytes.fromhex("50ed55ba") + b"fixture", ["kernel_source", "config_source"], "compile_fatbin"),
        "ptx_full": ("PTX_FULL", ptx_text.encode(), ["fatbin"], "extract_ptx"),
        "cubin": ("CUBIN", b"\x7fELF" + b"fixture", ["fatbin"], "extract_elf"),
        "elf_symbols": ("ELF_SYMBOL_TABLE", f"STT_FUNC STB_LOCAL STO_ENTRY {target_symbol}\n".encode(), ["cubin"], "elf_symbols"),
        "code_metadata": ("CODE_OBJECT_METADATA", b"arch = sm_110a\ncode for sm_110a\n.target sm_110a\n", ["cubin"], "code_object_metadata"),
        "nvdisasm_json": ("NVDISASM_JSON_FULL", nvjson.encode(), ["cubin"], "nvdisasm_json"),
        "nvdisasm_text": ("NVDISASM_TEXT_FULL", b"nvdisasm text\n", ["cubin"], "nvdisasm_text"),
    }
    for artifact_id, (role, content, parents, producer) in full_files.items():
        path = archive / artifact_id
        path.write_bytes(content)
        artifacts.append({"artifact_id":artifact_id,"role":role,"path":path.relative_to(root).as_posix(),"storage":"ignored_archive","media_type":"application/octet-stream","sha256":sha256_file(path),"size_bytes":path.stat().st_size,"producer_step_id":producer,"parent_artifact_ids":parents,"sealed":True})
    excerpt = root / "evidence/codegen-sm110a-v2/excerpts/fixture.fixture.a001"
    excerpt.mkdir(parents=True)
    excerpt_values = {
        "type_output": ("TYPE_WITNESS_OUTPUT", json.dumps(type_output_value).encode()+b"\n", ["type_executable"]),
        "ptx_target": ("PTX_TARGET_FUNCTION", ptx_block.encode(), ["ptx_full"]),
        "sass_target": ("SASS_TARGET_FUNCTION", (json.dumps(sass_function,separators=(",", ":"))+"\n").encode(), ["nvdisasm_json"]),
        "contract_report": ("CONTRACT_CHECK_REPORT", (json.dumps({"schema_version":1,"instance_id":instance_id,"symbol":target_symbol,"ptx_target":"sm_110a","sass_arch":"sm_110a","nvdisasm_total_function_count":1,"ptx_contract_results":ptx_result_values,"sass_contract_results":sass_result_values,"cta_group":{"declared":1,"ptx_mma_opcodes":["tcgen05.mma.cta_group::1.kind::f16"],"sass_mma_opcodes":["UTCHMMA"],"errors":[]}},indent=2)+"\n").encode(), ["ptx_target","sass_target","type_output"]),
    }
    for artifact_id,(role,content,parents) in excerpt_values.items():
        path=excerpt/artifact_id; path.write_bytes(content)
        producer = "run_type_witness" if artifact_id == "type_output" else "function_contract"
        artifacts.append({"artifact_id":artifact_id,"role":role,"path":path.relative_to(root).as_posix(),"storage":"git_evidence","media_type":"application/octet-stream","sha256":sha256_file(path),"size_bytes":path.stat().st_size,"producer_step_id":producer,"parent_artifact_ids":parents,"sealed":True})
    log_paths = {}
    logs_dir = archive / "logs"
    logs_dir.mkdir()
    for step in execution_plan:
        for stream_name, role in (("stdout", "COMPILE_STDOUT"), ("stderr", "COMPILE_STDERR")):
            path = logs_dir / f"{step['step_id']}.{stream_name}"
            stdout_contents = {
                "run_type_witness": json.dumps(type_output_value) + "\n",
                "elf_symbols": "STT_FUNC STB_LOCAL STO_ENTRY " + target_symbol + "\n",
                "code_object_metadata": "arch = sm_110a\ncode for sm_110a\n.target sm_110a\n",
                "nvdisasm_json": nvjson,
                "nvdisasm_text": "nvdisasm text\n",
            }
            path.write_text(
                stdout_contents.get(step["step_id"], "") if stream_name == "stdout" else "",
                encoding="utf-8",
            )
            artifact_id = f"{step['step_id']}_{stream_name}"
            log_paths[artifact_id] = path
            artifacts.append({"artifact_id":artifact_id,"role":role,"path":path.relative_to(root).as_posix(),"storage":"ignored_archive","media_type":"text/plain","sha256":sha256_file(path),"size_bytes":path.stat().st_size,"producer_step_id":step["step_id"],"parent_artifact_ids":[],"sealed":True})
    bundle_values = [
        (item["artifact_id"], item["path"], item["sha256"], item["size_bytes"])
        for item in artifacts
        if item["storage"] == "ignored_archive"
    ]
    manifest = {
        "schema_version": 1,
        "contract_sha256": contract_hash,
        "artifact_manifest_id": "am-fixture.fixture.a001",
        "attempt_id": attempt_id,
        "instance_ref": reference(instance_id, root, instance_path),
        "fingerprint_ref": reference(fingerprint_id, root, fingerprint_path),
        "items": artifacts,
        "archive_bundle": {
            "path": archive.relative_to(root).as_posix(),
            "format": "directory-manifest",
            "sha256": sha256_bytes(canonical_json_bytes(bundle_values)),
            "size_bytes": sum(
                item["size_bytes"] for item in artifacts if item["storage"] == "ignored_archive"
            ),
        },
    }
    manifest_path = root / "evidence/codegen-sm110a-v2/artifact-manifests/am-fixture.fixture.a001.json"
    write_json(manifest_path, manifest)

    events = []
    previous = None
    for event_type in ("ATTEMPT_CREATED", "INSTANCE_VALIDATED", "FINGERPRINT_SEALED"):
        item = event(len(events) + 1, previous, event_type, attempt_id, instance_id, fingerprint_id)
        if event_type == "ATTEMPT_CREATED":
            item["payload"] = {
                "run_id": "fixture",
                "origin_guide_root": str(root),
                "origin_attempt_root": str(archive),
                "origin_user": f"{os.getuid()}:{os.getgid()}",
            }
            item["event_sha256"] = journal_event_hash(item)
        elif event_type == "INSTANCE_VALIDATED":
            item["payload"] = {"instance_sha256": sha256_file(instance_path)}
            item["event_sha256"] = journal_event_hash(item)
        elif event_type == "FINGERPRINT_SEALED":
            item["payload"] = {"overall_sha256": fingerprint["overall_sha256"]}
            item["event_sha256"] = journal_event_hash(item)
        events.append(item)
        previous = item["event_sha256"]
    for step in execution_plan:
        started = event(len(events) + 1, previous, "STEP_STARTED", attempt_id, instance_id, fingerprint_id)
        started["payload"] = {
            "step_id": step["step_id"],
            "argv": materialize_actual_docker_argv(
                expected_actual_docker_argv(root, archive, step, manifest),
                root,
                archive,
            ),
        }
        started["event_sha256"] = journal_event_hash(started)
        events.append(started)
        previous = started["event_sha256"]
        stdout_path = log_paths[f"{step['step_id']}_stdout"]
        stderr_path = log_paths[f"{step['step_id']}_stderr"]
        finished = event(len(events) + 1, previous, "COMMAND_FINISHED", attempt_id, instance_id, fingerprint_id)
        finished["payload"] = {
            "step_id": step["step_id"],
            "returncode": 0,
            "stdout_path": stdout_path.relative_to(root).as_posix(),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_path": stderr_path.relative_to(root).as_posix(),
            "stderr_sha256": sha256_file(stderr_path),
        }
        finished["event_sha256"] = journal_event_hash(finished)
        events.append(finished)
        previous = finished["event_sha256"]
    for artifact in artifacts:
        sealed = event(len(events) + 1, previous, "ARTIFACT_SEALED", attempt_id, instance_id, fingerprint_id)
        sealed["payload"] = {
            "artifact_id": artifact["artifact_id"],
            "path": artifact["path"],
            "sha256": artifact["sha256"],
            "producer_step_id": artifact["producer_step_id"],
        }
        sealed["event_sha256"] = journal_event_hash(sealed)
        events.append(sealed)
        previous = sealed["event_sha256"]
    manifest_sealed = event(len(events) + 1, previous, "ARTIFACT_MANIFEST_SEALED", attempt_id, instance_id, fingerprint_id)
    manifest_sealed["payload"] = {"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256_file(manifest_path)}
    manifest_sealed["event_sha256"] = journal_event_hash(manifest_sealed)
    events.append(manifest_sealed)
    previous = manifest_sealed["event_sha256"]
    terminal = event(len(events) + 1, previous, "ATTEMPT_SEALED", attempt_id, instance_id, fingerprint_id)
    terminal["payload"] = {
        "terminal_pipeline_state": "STATIC_PASS",
        "artifact_manifest_path": manifest_path.relative_to(root).as_posix(),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "last_completed_step_id": "function_contract",
    }
    terminal["event_sha256"] = journal_event_hash(terminal)
    events.append(terminal)
    journal_path = root / "evidence/codegen-sm110a-v2/journals/fixture.fixture.a001.jsonl"
    journal_path.parent.mkdir(parents=True)
    journal_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in events) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "contract_sha256": contract_hash,
        "objective_sha256": contract["objective_sha256"],
        "freshness_epoch": contract["freshness_epoch"],
        "result_id": attempt_id,
        "scope": "STATIC_CODEGEN_ONLY",
        "subject": instance["subject"],
        "instance_ref": reference(instance_id, root, instance_path),
        "fingerprint_ref": reference(fingerprint_id, root, fingerprint_path),
        "attempt_id": attempt_id,
        "journal_ref": {
            "path": journal_path.relative_to(root).as_posix(),
            "sha256": sha256_file(journal_path),
            "last_seq": len(events),
            "last_event_sha256": terminal["event_sha256"],
        },
        "artifact_manifest_ref": reference("am-fixture.fixture.a001", root, manifest_path),
        "status": "STATIC_PASS",
        "layer_outcomes": [
            {"layer": layer, "state": "RESOLVED", "witness_artifact_id": contract["layer_witness_artifact_ids"][layer]}
            for layer in contract["layer_order"]
        ],
        "evidence": {
            "kind": "STATIC_PASS",
            "type_witness_artifact_id": "type_output",
            "resolved_stage": {"declared_policy_cpp": "Auto", "resolved_policy_cpp": "Auto", "stage_count": 2, "scheduler_stages": 1, "accumulator_stages": 1},
            "function_binding": {
                "target_cpp_entity": "cutlass::device_kernel<GemmKernel>",
                "selection_policy": "sole_ptx_entry_equals_sole_elf_sto_entry_equals_unique_nvdisasm_function",
                "symbol": target_symbol,
                "ptx_entry_count": 1,
                "elf_sto_entry_count": 1,
                "nvdisasm_symbol_match_count": 1,
                "nvdisasm_total_function_count": 1,
                "ptx_function_artifact_id": "ptx_target",
                "sass_function_artifact_id": "sass_target",
                "same_symbol": True,
            },
            "ptx_contract_results": ptx_result_values,
            "sass_contract_results": sass_result_values,
            "cta_group_contract_pass": True,
        },
    }
    result_path = root / "evidence/codegen-sm110a-v2/results/fixture.fixture.a001.json"
    write_json(result_path, result)
    return root, {
        "instance": instance_path,
        "source": root / "tools/codegen_v2/attribution.py",
        "fingerprint": fingerprint_path,
        "manifest": manifest_path,
        "journal": journal_path,
        "result": result_path,
        "archive": archive,
        "type_output": excerpt / "type_output",
        "ptx_target": excerpt / "ptx_target",
        "sass_target": excerpt / "sass_target",
    }


def require_rejected(name: str, expected: str, mutate) -> None:
    root, paths = make_fixture()
    try:
        mutate(root, paths)
        try:
            validate_result(root, paths["result"])
        except ContractError as error:
            if expected not in str(error):
                raise AssertionError(f"{name}: wrong rejection: {error}") from error
        else:
            raise AssertionError(f"{name}: deliberate corruption was accepted")
    finally:
        shutil.rmtree(root)

def require_direct_rejected(name: str, expected: str, callback) -> None:
    root, paths = make_fixture()
    try:
        try:
            callback(root, paths)
        except ContractError as error:
            if expected not in str(error):
                raise AssertionError(f"{name}: wrong direct rejection: {error}") from error
        else:
            raise AssertionError(f"{name}: deliberate direct corruption was accepted")
    finally:
        shutil.rmtree(root)


def require_offline_rejected(name: str, expected: str, mutate) -> None:
    root, paths = make_fixture()
    try:
        mutate(root, paths)
        shutil.rmtree(paths["archive"])
        try:
            validate_result(root, paths["result"], require_archive=False)
        except ContractError as error:
            if expected not in str(error):
                raise AssertionError(f"{name}: wrong offline rejection: {error}") from error
        else:
            raise AssertionError(f"{name}: deliberate offline corruption was accepted")
    finally:
        shutil.rmtree(root)


def mutate_json(path: Path, callback) -> None:
    value = load_strict_json(path)
    callback(value)
    write_json(path, value)


def mutate_fingerprint(paths: dict[str, Path], callback) -> None:
    mutate_json(paths["fingerprint"], callback)
    fingerprint_sha = sha256_file(paths["fingerprint"])
    mutate_json(
        paths["manifest"],
        lambda value: value["fingerprint_ref"].__setitem__("sha256", fingerprint_sha),
    )
    manifest_sha = sha256_file(paths["manifest"])
    result = load_strict_json(paths["result"])
    result["fingerprint_ref"]["sha256"] = fingerprint_sha
    result["artifact_manifest_ref"]["sha256"] = manifest_sha
    write_json(paths["result"], result)


def mutate_manifest(paths: dict[str, Path], callback) -> None:
    mutate_json(paths["manifest"], callback)
    mutate_json(
        paths["result"],
        lambda value: value["artifact_manifest_ref"].__setitem__(
            "sha256", sha256_file(paths["manifest"])
        ),
    )


def reseal_journal(paths: dict[str, Path], callback) -> None:
    events = [json.loads(line) for line in paths["journal"].read_text().splitlines()]
    callback(events)
    previous = None
    for seq, item in enumerate(events, start=1):
        item["seq"] = seq
        item["previous_event_sha256"] = previous
        item["event_sha256"] = journal_event_hash(item)
        previous = item["event_sha256"]
    paths["journal"].write_text(
        "\n".join(json.dumps(item, separators=(",", ":")) for item in events) + "\n",
        encoding="utf-8",
    )
    result = load_strict_json(paths["result"])
    result["journal_ref"]["sha256"] = sha256_file(paths["journal"])
    result["journal_ref"]["last_seq"] = len(events)
    result["journal_ref"]["last_event_sha256"] = previous
    write_json(paths["result"], result)


def reseal_manifest_and_journal(paths: dict[str, Path], callback) -> None:
    manifest = load_strict_json(paths["manifest"])
    callback(manifest)
    archive_items = [item for item in manifest["items"] if item["storage"] == "ignored_archive"]
    manifest["archive_bundle"]["sha256"] = sha256_bytes(
        canonical_json_bytes(
            [
                (item["artifact_id"], item["path"], item["sha256"], item["size_bytes"])
                for item in archive_items
            ]
        )
    )
    manifest["archive_bundle"]["size_bytes"] = sum(
        item["size_bytes"] for item in archive_items
    )
    write_json(paths["manifest"], manifest)
    result = load_strict_json(paths["result"])
    result["artifact_manifest_ref"]["sha256"] = sha256_file(paths["manifest"])
    write_json(paths["result"], result)

    def synchronize(events):
        items_by_id = {item["artifact_id"]: item for item in manifest["items"]}
        for item in events:
            if item["event_type"] == "COMMAND_FINISHED":
                step_id = item["payload"]["step_id"]
                for stream_name in ("stdout", "stderr"):
                    artifact = items_by_id[f"{step_id}_{stream_name}"]
                    item["payload"][f"{stream_name}_path"] = artifact["path"]
                    item["payload"][f"{stream_name}_sha256"] = artifact["sha256"]
            elif item["event_type"] == "ARTIFACT_SEALED":
                artifact = items_by_id[item["payload"]["artifact_id"]]
                item["payload"] = {
                    "artifact_id": artifact["artifact_id"],
                    "path": artifact["path"],
                    "sha256": artifact["sha256"],
                    "producer_step_id": artifact["producer_step_id"],
                }
            elif item["event_type"] == "ARTIFACT_MANIFEST_SEALED":
                item["payload"]["sha256"] = sha256_file(paths["manifest"])
            elif item["event_type"] == "ATTEMPT_SEALED":
                item["payload"]["artifact_manifest_sha256"] = sha256_file(paths["manifest"])

    reseal_journal(paths, synchronize)


def main() -> int:
    root, paths = make_fixture()
    try:
        validate_fingerprint(root, paths["fingerprint"])
        validate_artifact_manifest(root, paths["manifest"])
        validate_result(root, paths["result"])
    finally:
        shutil.rmtree(root)

    offline_root, offline_paths = make_fixture()
    try:
        shutil.rmtree(offline_paths["archive"])
        validate_result(offline_root, offline_paths["result"], require_archive=False)
    finally:
        shutil.rmtree(offline_root)

    projection_root, projection_paths = make_fixture()
    try:
        tags_path = projection_root / "tests/codegen/sm110a_tensor_schedule_tags.json"
        tags = load_strict_json(tags_path)
        fixture_tag = next(entry for entry in tags["entries"] if entry["tag"] == "KernelFixtureSm100")
        fixture_tag["status"] = "STATIC_PASS"
        fixture_tag["current_case_ids"] = ["projection-only"]
        write_json(tags_path, tags)
        auto_path = projection_root / "tests/codegen/sm110a_auto_control_inventory.json"
        auto = load_strict_json(auto_path)
        auto["entries"][0]["status"] = "STATIC_PASS"
        auto["entries"][0]["resolved_builder_specialization"] = "projection-only"
        auto["entries"][0]["resolved_dispatch_policy"] = "projection-only"
        auto["entries"][0]["result_id"] = "projection-only"
        write_json(auto_path, auto)
        validate_result(projection_root, projection_paths["result"])
    finally:
        shutil.rmtree(projection_root)

    validate_device_kernel_symbol(
        "_ZN7cutlass13device_kernelI13FixtureKernelEEvNT_6ParamsE",
        "abi-mangled:N7cutlass4gemm6kernel13GemmUniversalI7FixtureEE",
    )

    strict_root = Path(tempfile.mkdtemp(prefix="codegen-v2-strict-"))
    try:
        duplicate = strict_root / "duplicate.json"
        duplicate.write_text('{"status":"A","status":"B"}\n', encoding="utf-8")
        try:
            load_strict_json(duplicate)
        except ContractError as error:
            if "duplicate JSON key" not in str(error):
                raise
        else:
            raise AssertionError("duplicate JSON key was accepted")
        nonfinite = strict_root / "nonfinite.json"
        nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
        try:
            load_strict_json(nonfinite)
        except ContractError as error:
            if "non-finite" not in str(error):
                raise
        else:
            raise AssertionError("NaN was accepted")
    finally:
        shutil.rmtree(strict_root)

    require_rejected("fingerprint_overall", "overall_sha256", lambda r, p: mutate_fingerprint(p, lambda v: v.__setitem__("overall_sha256", "0" * 64)))
    require_rejected("fingerprint_tree", "input_tree_sha256", lambda r, p: mutate_fingerprint(p, lambda v: v["source_closure"].__setitem__("input_tree_sha256", "0" * 64)))
    require_rejected("source_drift", "source_closure.inputs", lambda r, p: p["source"].write_text("drift\n", encoding="utf-8"))
    require_rejected("instance_ref_hash", "instance_ref SHA-256", lambda r, p: mutate_json(p["result"], lambda v: v["instance_ref"].__setitem__("sha256", "0" * 64)))
    require_rejected("fingerprint_ref_hash", "fingerprint_ref SHA-256", lambda r, p: mutate_json(p["result"], lambda v: v["fingerprint_ref"].__setitem__("sha256", "0" * 64)))
    require_rejected("manifest_ref_hash", "artifact manifest SHA-256", lambda r, p: mutate_json(p["result"], lambda v: v["artifact_manifest_ref"].__setitem__("sha256", "0" * 64)))
    require_rejected("journal_ref_hash", "journal SHA-256", lambda r, p: mutate_json(p["result"], lambda v: v["journal_ref"].__setitem__("sha256", "0" * 64)))
    require_rejected("nonresolved_pass", "every layer RESOLVED", lambda r, p: mutate_json(p["result"], lambda v: v["layer_outcomes"][3].__setitem__("state", "NOT_REACHED")))
    require_rejected("unknown_witness", "wrong witness artifact", lambda r, p: mutate_json(p["result"], lambda v: v["layer_outcomes"][0].__setitem__("witness_artifact_id", "unknown")))
    require_rejected("required_zero", "required opcode did not match", lambda r, p: mutate_json(p["result"], lambda v: v["evidence"]["ptx_contract_results"][0].__setitem__("match_count", 0)))
    require_rejected("forbidden_match", "forbidden opcode matched", lambda r, p: mutate_json(p["result"], lambda v: v["evidence"]["ptx_contract_results"].append({"id":"forbidden","required":False,"match_count":1,"matched_opcodes":["bad"]})))
    require_rejected("runtime_field", "Additional properties", lambda r, p: mutate_json(p["result"], lambda v: v.__setitem__("runtime_correct", False)))
    require_rejected("evidence_kind", "not valid under any", lambda r, p: mutate_json(p["result"], lambda v: v["evidence"].__setitem__("kind", "EXPECTED_STATIC_REJECT")))
    require_rejected("artifact_hash", "artifact size/hash", lambda r, p: p["type_output"].write_text("drift\n", encoding="utf-8"))
    require_rejected("artifact_parent", "topologically ordered", lambda r, p: mutate_manifest(p, lambda v: v["items"][0]["parent_artifact_ids"].append("sass_target")))
    require_rejected("bundle_hash", "bundle checksum", lambda r, p: mutate_manifest(p, lambda v: v["archive_bundle"].__setitem__("sha256", "0" * 64)))
    require_rejected("archive_missing", "file is missing", lambda r, p: shutil.rmtree(p["archive"]))
    require_rejected("path_traversal", "does not match", lambda r, p: mutate_manifest(p, lambda v: v["items"][0].__setitem__("path", "../escape")))

    def change_parent(artifact_id: str, parents: list[str]):
        def mutate(root: Path, paths: dict[str, Path]) -> None:
            reseal_manifest_and_journal(
                paths,
                lambda manifest: next(
                    item for item in manifest["items"] if item["artifact_id"] == artifact_id
                ).__setitem__("parent_artifact_ids", parents),
            )

        return mutate

    require_rejected(
        "type_output_lineage",
        "artifact lineage mismatch for type_output",
        change_parent("type_output", ["type_source"]),
    )
    require_rejected(
        "fatbin_lineage",
        "artifact lineage mismatch for fatbin",
        change_parent("fatbin", ["type_source", "config_source"]),
    )

    def source_snapshot_hash_drift(root: Path, paths: dict[str, Path]) -> None:
        def mutate(manifest):
            item = next(
                value for value in manifest["items"] if value["artifact_id"] == "config_source"
            )
            item["sha256"] = "0" * 64

        reseal_manifest_and_journal(paths, mutate)

    require_offline_rejected(
        "source_snapshot_hash_drift",
        "differs from the referenced translation unit",
        source_snapshot_hash_drift,
    )

    def tracked_ptx_opcode_tamper(root: Path, paths: dict[str, Path]) -> None:
        paths["ptx_target"].write_text(
            paths["ptx_target"].read_text(encoding="utf-8").replace(
                "cp.async.bulk.tensor.3d", "mov.u32"
            ),
            encoding="utf-8",
        )

        def mutate(manifest):
            item = next(
                value for value in manifest["items"] if value["artifact_id"] == "ptx_target"
            )
            item["sha256"] = sha256_file(paths["ptx_target"])
            item["size_bytes"] = paths["ptx_target"].stat().st_size

        reseal_manifest_and_journal(paths, mutate)

    require_offline_rejected(
        "tracked_ptx_opcode_tamper",
        "tracked function excerpts do not satisfy",
        tracked_ptx_opcode_tamper,
    )

    require_offline_rejected(
        "tracked_function_artifact_id",
        "function binding differs from tracked excerpts",
        lambda r, p: mutate_json(
            p["result"],
            lambda value: value["evidence"]["function_binding"].__setitem__(
                "ptx_function_artifact_id", "contract_report"
            ),
        ),
    )

    def symlink_artifact(root: Path, paths: dict[str, Path]) -> None:
        target = paths["type_output"]
        moved = target.with_name("real_type_output")
        target.rename(moved)
        target.symlink_to(moved)

    require_rejected("artifact_symlink", "symlinks are forbidden", symlink_artifact)

    def mutate_journal(root: Path, paths: dict[str, Path], callback) -> None:
        events = [json.loads(line) for line in paths["journal"].read_text().splitlines()]
        callback(events)
        paths["journal"].write_text("\n".join(json.dumps(item, separators=(",", ":")) for item in events) + "\n", encoding="utf-8")
        mutate_json(paths["result"], lambda value: value["journal_ref"].__setitem__("sha256", sha256_file(paths["journal"])))

    require_rejected("journal_seq", "non-contiguous seq", lambda r, p: mutate_journal(r, p, lambda e: e[1].__setitem__("seq", 3)))
    require_rejected("journal_chain", "hash chain mismatch", lambda r, p: mutate_journal(r, p, lambda e: e[1].__setitem__("previous_event_sha256", "0" * 64)))
    require_rejected("journal_event_hash", "event SHA-256 mismatch", lambda r, p: mutate_journal(r, p, lambda e: e[0].__setitem__("event_sha256", "0" * 64)))
    require_rejected("journal_unsealed", "journal is not sealed", lambda r, p: mutate_journal(r, p, lambda e: e.pop()))

    def after_terminal(root: Path, paths: dict[str, Path]) -> None:
        def append(events):
            events.append(event(len(events) + 1, events[-1]["event_sha256"], "STEP_STARTED", "fixture.fixture.a001", "fixture", events[-1]["fingerprint_id"]))
        mutate_journal(root, paths, append)

    require_rejected("journal_after_terminal", "events after ATTEMPT_SEALED", after_terminal)
    require_rejected("journal_instance", "instance mismatch", lambda r, p: mutate_journal(r, p, lambda e: e[0].__setitem__("instance_id", "other")))
    require_rejected("journal_event_type", "unknown event type", lambda r, p: mutate_journal(r, p, lambda e: e[0].__setitem__("event_type", "INVENTED")))
    require_rejected("attempt_mismatch", "result_id, attempt_id", lambda r, p: mutate_json(p["result"], lambda v: v.__setitem__("attempt_id", "other.a001")))
    require_rejected("subject_mismatch", "subject does not match", lambda r, p: mutate_json(p["result"], lambda v: v["subject"].__setitem__("id", "OtherTag")))

    require_rejected(
        "result_identity",
        "result_id, attempt_id",
        lambda r, p: mutate_json(p["result"], lambda v: v.__setitem__("result_id", "unrelated.run.a999")),
    )

    def forge_reject(root: Path, paths: dict[str, Path]) -> None:
        def mutate(value):
            value["status"] = "EXPECTED_STATIC_REJECT"
            value["layer_outcomes"][0]["state"] = "REJECTED"
            for layer in value["layer_outcomes"][1:]:
                layer["state"] = "NOT_REACHED"
            value["evidence"] = {
                "kind": "EXPECTED_STATIC_REJECT",
                "failure": {"layer": "DECLARED_BUILDER_CONFIG", "domain": "CONFIGURATION_LEGALITY", "step_id": "compile_fatbin", "returncode": 1, "stderr_artifact_id": "compile_fatbin_stderr"},
                "diagnostic_results": [{"id":"diag","required":True,"match_count":1,"matched_opcodes":["fake"]}],
                "source_constraints": [value["instance_ref"]],
                "legal_control": {"result_ref": value["fingerprint_ref"], "relation": "sibling", "controlled_delta": ["fake"]}
            }
        mutate_json(paths["result"], mutate)

    require_rejected("forged_reject", "not implemented", forge_reject)

    require_rejected(
        "resolved_stage_tamper",
        "resolved_stage differs",
        lambda r, p: mutate_json(
            p["result"], lambda v: v["evidence"]["resolved_stage"].__setitem__("stage_count", 99)
        ),
    )

    require_rejected(
        "argv_tamper",
        "actual argv differs",
        lambda r, p: reseal_journal(
            p,
            lambda events: next(
                item for item in events if item["event_type"] == "STEP_STARTED"
            )["payload"].__setitem__("argv", ["/bin/false", "--tampered"]),
        ),
    )

    def mount_tamper(events):
        started = next(item for item in events if item["event_type"] == "STEP_STARTED")
        argv = started["payload"]["argv"]
        mount_index = argv.index("-v") + 1
        argv[mount_index] = "/tmp/attacker-source:/workspace:ro"

    require_rejected(
        "workspace_mount_tamper",
        "workspace mount source differs",
        lambda r, p: reseal_journal(p, mount_tamper),
    )

    require_rejected(
        "instance_validation_seal",
        "INSTANCE_VALIDATED does not seal",
        lambda r, p: reseal_journal(
            p,
            lambda events: events[1].__setitem__(
                "payload", {"instance_sha256": "0" * 64}
            ),
        ),
    )
    require_rejected(
        "fingerprint_validation_seal",
        "FINGERPRINT_SEALED does not seal",
        lambda r, p: reseal_journal(
            p,
            lambda events: events[2].__setitem__(
                "payload", {"overall_sha256": "1" * 64}
            ),
        ),
    )
    require_rejected(
        "terminal_manifest_binding",
        "ATTEMPT_SEALED payload",
        lambda r, p: reseal_journal(
            p,
            lambda events: events[-1]["payload"].__setitem__(
                "last_completed_step_id", "compile_fatbin"
            ),
        ),
    )

    def cross_attempt_excerpt(root: Path, paths: dict[str, Path]) -> None:
        other = root / "evidence/codegen-sm110a-v2/excerpts/other.attempt/target.ptx"
        other.parent.mkdir(parents=True)
        shutil.copy2(paths["ptx_target"], other)

        def mutate(manifest):
            item = next(value for value in manifest["items"] if value["artifact_id"] == "ptx_target")
            item["path"] = other.relative_to(root).as_posix()

        reseal_manifest_and_journal(paths, mutate)

    require_rejected(
        "cross_attempt_excerpt",
        "outside the attempt excerpt namespace",
        cross_attempt_excerpt,
    )

    def direct_min_count(root: Path, paths: dict[str, Path]) -> None:
        result = load_strict_json(paths["result"])
        instance = load_strict_json(paths["instance"])
        instance["static_contract"]["ptx"]["required"][0]["min_count"] = 2
        artifacts = load_strict_json(paths["manifest"])
        validate_static_pass_evidence(
            root, result, instance, artifacts, require_archive=True
        )

    require_direct_rejected("contract_min_count", "outside [2", direct_min_count)

    def direct_provenance(root: Path, paths: dict[str, Path]) -> None:
        mutate_json(
            paths["instance"],
            lambda value: value["provenance"].__setitem__(
                "reference_inventory_sha256", "0" * 64
            ),
        )
        validate_instance(root, paths["instance"])

    require_direct_rejected("provenance_inventory", "reference inventory SHA-256", direct_provenance)

    def direct_wrong_group(root: Path, paths: dict[str, Path]) -> None:
        mutate_json(
            paths["instance"],
            lambda value: value["subject"].__setitem__("group", "wrong_group"),
        )
        validate_instance(root, paths["instance"])

    require_direct_rejected("subject_group", "59-Tag manifest", direct_wrong_group)

    def direct_tag_inventory_drift(root: Path, paths: dict[str, Path]) -> None:
        tag_path = root / "tests/codegen/sm110a_tensor_schedule_tags.json"
        tags = load_strict_json(tag_path)
        next(entry for entry in tags["entries"] if entry["tag"] == "KernelFixtureSm100")[
            "group"
        ] = "forged_group"
        write_json(tag_path, tags)
        validate_instance(root, paths["instance"])

    require_direct_rejected(
        "immutable_tag_inventory", "immutable 59-Tag subject inventory drifted", direct_tag_inventory_drift
    )

    def direct_auto_inventory_drift(root: Path, paths: dict[str, Path]) -> None:
        auto_path = root / "tests/codegen/sm110a_auto_control_inventory.json"
        auto = load_strict_json(auto_path)
        auto["entries"][0]["expected_outcome_hypothesis"] = "EXPECTED_STATIC_REJECT"
        write_json(auto_path, auto)
        validate_instance(root, paths["instance"])

    require_direct_rejected(
        "immutable_auto_inventory", "immutable Auto control inventory drifted", direct_auto_inventory_drift
    )

    def direct_bogus_auto(root: Path, paths: dict[str, Path]) -> None:
        instance = load_strict_json(paths["instance"])
        auto_path = root / "tests/codegen/sm110a_auto_control_inventory.json"
        instance["subject"] = {
            "kind": "kernel_schedule_auto_control",
            "id": "nonexistent_auto",
            "group": "bogus",
        }
        instance["provenance"]["reference_inventory_sha256"] = load_strict_json(
            root / "tests/codegen/static_codegen_contract.json"
        )["inventory_contract_sha256"]["auto_controls"]
        instance["provenance"]["reference_class"] = "auto_control"
        instance["declared_config"]["mainloop_schedule"] = (
            "cutlass::gemm::collective::KernelScheduleAuto"
        )
        write_json(paths["instance"], instance)
        validate_instance(root, paths["instance"])

    require_direct_rejected("bogus_auto_control", "no unique inventory entry", direct_bogus_auto)

    def direct_self_parent(root: Path, paths: dict[str, Path]) -> None:
        mutate_json(
            paths["instance"],
            lambda value: value["provenance"].__setitem__(
                "parent_instance_id", value["instance_id"]
            ),
        )
        validate_instance(root, paths["instance"])

    require_direct_rejected(
        "self_parent", "non-derived explicit instance must not declare a parent", direct_self_parent
    )

    def direct_cross_layer_type(root: Path, paths: dict[str, Path]) -> None:
        result = load_strict_json(paths["result"])
        instance = load_strict_json(paths["instance"])
        artifacts = load_strict_json(paths["manifest"])
        value = load_strict_json(paths["type_output"])
        value["resolved_types"]["dispatch_policy"] = "FORGED_DISPATCH"
        write_json(paths["type_output"], value)
        validate_static_pass_evidence(root, result, instance, artifacts, require_archive=True)

    require_direct_rejected(
        "cross_layer_type_binding", "cross-layer binding differs", direct_cross_layer_type
    )

    def direct_atom_shape(root: Path, paths: dict[str, Path]) -> None:
        result = load_strict_json(paths["result"])
        instance = load_strict_json(paths["instance"])
        artifacts = load_strict_json(paths["manifest"])
        value = load_strict_json(paths["type_output"])
        value["resolved_values"].pop("atom_shape_mnk")
        write_json(paths["type_output"], value)
        validate_static_pass_evidence(root, result, instance, artifacts, require_archive=True)

    require_direct_rejected("atom_shape", "positive atom_shape_mnk", direct_atom_shape)

    def direct_declared_config(root: Path, paths: dict[str, Path]) -> None:
        result = load_strict_json(paths["result"])
        instance = load_strict_json(paths["instance"])
        artifacts = load_strict_json(paths["manifest"])
        instance["declared_config"]["operator_class"] = "forged::OperatorClass"
        validate_static_pass_evidence(root, result, instance, artifacts, require_archive=True)

    require_direct_rejected(
        "declared_config_binding", "config_operator_class differs", direct_declared_config
    )

    def direct_stage_carveout(root: Path, paths: dict[str, Path]) -> None:
        result = load_strict_json(paths["result"])
        instance = load_strict_json(paths["instance"])
        artifacts = load_strict_json(paths["manifest"])
        instance["declared_config"]["stage_policy"] = (
            "cutlass::gemm::collective::StageCountAutoCarveout<EpilogueSharedStorage>"
        )
        value = load_strict_json(paths["type_output"])
        value["resolved_types"]["config_stage_policy"] = (
            "cutlass::gemm::collective::StageCountAutoCarveout<64>"
        )
        value["resolved_values"]["epilogue_shared_storage_bytes"] = 32
        write_json(paths["type_output"], value)
        validate_static_pass_evidence(root, result, instance, artifacts, require_archive=True)

    require_direct_rejected(
        "stage_carveout_bytes",
        "StageCountAutoCarveout bytes differ",
        direct_stage_carveout,
    )

    def direct_config_comment(root: Path, paths: dict[str, Path]) -> None:
        instance = load_strict_json(paths["instance"])
        config = root / instance["translation_units"]["config"]["path"]
        config.write_text("// cutlass::gemm::KernelFixtureSm100\n", encoding="utf-8")
        instance["translation_units"]["config"]["sha256"] = sha256_file(config)
        write_json(paths["instance"], instance)
        validate_instance(root, paths["instance"])

    require_direct_rejected("config_tag_only_comment", "does not instantiate", direct_config_comment)

    def direct_witness_config(root: Path, paths: dict[str, Path]) -> None:
        instance = load_strict_json(paths["instance"])
        witness = root / instance["translation_units"]["type_witness"]["path"]
        witness.write_text("int main(){ write_codegen_type_report(); }\n", encoding="utf-8")
        instance["translation_units"]["type_witness"]["sha256"] = sha256_file(witness)
        write_json(paths["instance"], instance)
        validate_instance(root, paths["instance"])

    require_direct_rejected("witness_config_split", "does not include", direct_witness_config)

    def direct_type_kernel(root: Path, paths: dict[str, Path]) -> None:
        result = load_strict_json(paths["result"])
        instance = load_strict_json(paths["instance"])
        artifacts = load_strict_json(paths["manifest"])
        type_output = paths["type_output"]
        value = load_strict_json(type_output)
        value["resolved_types"]["gemm_kernel"] = "float"
        write_json(type_output, value)
        validate_static_pass_evidence(root, result, instance, artifacts, require_archive=True)

    require_direct_rejected("type_kernel_split", "GemmKernel differs", direct_type_kernel)

    def direct_report(root: Path, paths: dict[str, Path]) -> None:
        result = load_strict_json(paths["result"])
        instance = load_strict_json(paths["instance"])
        artifacts = load_strict_json(paths["manifest"])
        report_item = next(item for item in artifacts["items"] if item["role"] == "CONTRACT_CHECK_REPORT")
        report_path = root / report_item["path"]
        value = load_strict_json(report_path)
        value["symbol"] = "forged"
        write_json(report_path, value)
        validate_static_pass_evidence(root, result, instance, artifacts, require_archive=True)

    require_direct_rejected("contract_report_tamper", "contract report differs", direct_report)

    def reseal_fingerprint_value(value):
        value["component_hashes"]["toolchain"] = sha256_bytes(canonical_json_bytes(value["toolchain"]))
        value["component_hashes"]["plan"] = sha256_bytes(canonical_json_bytes(value["execution_plan"]))
        value["overall_sha256"] = sha256_bytes(canonical_json_bytes(fingerprint_payload(value)))
        value["fingerprint_id"] = f"fp-fixture-{value['overall_sha256'][:12]}"

    def direct_plan(root: Path, paths: dict[str, Path]) -> None:
        value = load_strict_json(paths["fingerprint"])
        value["execution_plan"][0]["argv"].append("--fmad=false")
        reseal_fingerprint_value(value)
        write_json(paths["fingerprint"], value)
        validate_fingerprint(root, paths["fingerprint"])

    require_direct_rejected("extra_compile_flag", "frozen instance plan", direct_plan)

    def direct_extra_fingerprint_input(root: Path, paths: dict[str, Path]) -> None:
        value = load_strict_json(paths["fingerprint"])
        tag_path = root / "tests/codegen/sm110a_tensor_schedule_tags.json"
        value["source_closure"]["inputs"].append(
            {
                "path": tag_path.relative_to(root).as_posix(),
                "sha256": sha256_file(tag_path),
            }
        )
        tree_hash = hash_input_tree(value["source_closure"]["inputs"])
        value["source_closure"]["input_tree_sha256"] = tree_hash
        value["component_hashes"]["source"] = tree_hash
        reseal_fingerprint_value(value)
        write_json(paths["fingerprint"], value)
        validate_fingerprint(root, paths["fingerprint"])

    require_direct_rejected(
        "extra_fingerprint_input",
        "exact frozen input set",
        direct_extra_fingerprint_input,
    )

    def direct_tool_hash(root: Path, paths: dict[str, Path]) -> None:
        value = load_strict_json(paths["fingerprint"])
        value["toolchain"]["executables"]["nvcc"]["binary_sha256"] = "1" * 64
        reseal_fingerprint_value(value)
        write_json(paths["fingerprint"], value)
        validate_fingerprint(root, paths["fingerprint"])

    require_direct_rejected("tool_binary_hash", "binary SHA-256 mismatch", direct_tool_hash)

    def direct_fatbin_magic(root: Path, paths: dict[str, Path]) -> None:
        result = load_strict_json(paths["result"])
        instance = load_strict_json(paths["instance"])
        artifacts = load_strict_json(paths["manifest"])
        fatbin_item = next(item for item in artifacts["items"] if item["role"] == "FATBIN")
        (root / fatbin_item["path"]).write_bytes(b"not-fatbin")
        validate_static_pass_evidence(root, result, instance, artifacts, require_archive=True)

    require_direct_rejected("fatbin_magic", "invalid magic", direct_fatbin_magic)

    def direct_bad_abi_fallback(root: Path, paths: dict[str, Path]) -> None:
        validate_device_kernel_symbol(
            "_ZN7cutlass13device_kernelI13FixtureKernelEEvT_",
            "abi-mangled:N7cutlass4gemm6kernel13GemmUniversalI7FixtureEE",
        )

    require_direct_rejected(
        "abi_fallback_symbol_shape", "target symbol is not", direct_bad_abi_fallback
    )

    print("CODEGEN_V2_MODEL_ADVERSARIAL_PASS mutations=64 positive=6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

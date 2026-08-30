"""Strict static-codegen records, hashes, and cross-file semantic validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .attribution import (
    AttributionError,
    FunctionBlock,
    attribute_codegen,
    evaluate_contract,
    evaluate_cta_group,
    parse_ptx_entries,
    validate_patterns,
)


class ContractError(ValueError):
    """A v2 record violates its schema or semantic evidence contract."""


def strip_cpp_comments_and_strings(text: str) -> str:
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    return text


def normalize_cpp_type(value: str) -> str:
    """Normalize a deliberately small set of compiler-reported C++ aliases."""
    normalized = re.sub(r"\s+", "", value)
    normalized = normalized.replace("cute::Shape<", "cute::tuple<")
    normalized = re.sub(r"cute::_([0-9]+)(?![A-Za-z0-9_])", r"cute::C<\1>", normalized)
    normalized = normalized.replace(
        "cutlass::int4b_t", "cutlass::integer_subbyte<4,true>"
    )
    normalized = normalized.replace(
        "cute::UMMA::Major::K", "(cute::UMMA::Major)0"
    ).replace("cute::UMMA::Major::MN", "(cute::UMMA::Major)1")
    normalized = normalized.replace(
        "cutlass::FloatRoundStyle::round_to_nearest",
        "(cutlass::FloatRoundStyle)2",
    )
    aliases = {
        "int8_t": "signedchar",
        "uint8_t": "unsignedchar",
        "int32_t": "int",
        "uint32_t": "unsignedint",
    }
    for alias, canonical in aliases.items():
        normalized = re.sub(
            rf"(?<![A-Za-z0-9_])(?:std::)?{alias}(?![A-Za-z0-9_])",
            canonical,
            normalized,
        )
    return normalized


def split_cpp_template(value: str) -> tuple[str, list[str]] | None:
    """Split one normalized C++ template-id without pretending to parse C++."""
    opening = value.find("<")
    if opening < 0 or not value.endswith(">"):
        return None
    base = value[:opening]
    body = value[opening + 1 : -1]
    arguments: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(body):
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
            if depth < 0:
                return None
        elif character == "," and depth == 0:
            arguments.append(body[start:index])
            start = index + 1
    if depth != 0:
        return None
    arguments.append(body[start:])
    return base, arguments


def cpp_type_equivalent(
    actual: str, declared: str, *, allow_trailing_default_arguments: bool = False
) -> bool:
    actual_normalized = normalize_cpp_type(actual)
    declared_normalized = normalize_cpp_type(declared)
    if actual_normalized == declared_normalized:
        return True
    actual_template = split_cpp_template(actual_normalized)
    declared_template = split_cpp_template(declared_normalized)
    if actual_template is None or declared_template is None:
        return False
    actual_base, actual_arguments = actual_template
    declared_base, declared_arguments = declared_template
    if actual_base != declared_base or len(actual_arguments) < len(declared_arguments):
        return False
    if len(actual_arguments) != len(declared_arguments):
        if not allow_trailing_default_arguments:
            return False
        if declared_base == "cutlass::epilogue::fusion::LinearCombination":
            if not 2 <= len(declared_arguments) <= 5 or len(actual_arguments) != 5:
                return False
            expanded_declared = list(declared_arguments)
            if len(expanded_declared) < 3:
                expanded_declared.append(expanded_declared[0])
            if len(expanded_declared) < 4:
                expanded_declared.append(expanded_declared[1])
            if len(expanded_declared) < 5:
                expanded_declared.append("(cutlass::FloatRoundStyle)2")
            declared_arguments = expanded_declared
        else:
            return False
    return all(
        cpp_type_equivalent(actual_arg, declared_arg)
        for actual_arg, declared_arg in zip(actual_arguments, declared_arguments, strict=False)
    )


def classify_mma_fragment_type(value: str) -> str:
    if "sparse_smem_desc" in value:
        return "SPARSE_SMEM_DESCRIPTOR"
    if "smem_desc" in value:
        return "SMEM_DESCRIPTOR"
    if "tmem_frg" in value:
        return "TMEM_FRAGMENT"
    return "UNKNOWN"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON number is forbidden: {value}")


def load_strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{path}: cannot parse strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path}: top level must be an object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(root: Path, path: Path, *, identifier: str | None = None) -> dict[str, str]:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    result = {"path": relative, "sha256": sha256_file(path)}
    if identifier is not None:
        return {"id": identifier, **result}
    return result


def safe_path(root: Path, relative: object, field: str, *, require_file: bool = True) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ContractError(f"{field}: path must be a nonempty string")
    path_value = Path(relative)
    if path_value.is_absolute() or ".." in path_value.parts:
        raise ContractError(f"{field}: path must be safe and relative")
    path = root / path_value
    if require_file and not path.is_file():
        raise ContractError(f"{field}: file is missing: {relative}")
    current = root.resolve()
    for part in path_value.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{field}: symlinks are forbidden: {relative}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{field}: path escapes root: {relative}") from error
    return path


def validate_with_schema(record: dict[str, Any], schema_path: Path, label: str) -> None:
    schema = load_strict_json(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(str(value) for value in error.absolute_path) or "<root>"
        raise ContractError(f"{label}.{location}: {error.message}")


def contract_sha256(root: Path) -> str:
    return sha256_file(root / "tests/codegen/static_codegen_contract.json")


def explicit_subject_inventory_sha256(root: Path) -> str:
    manifest = load_strict_json(root / "tests/codegen/sm110a_tensor_schedule_tags.json")
    payload = {
        "schema_version": manifest["schema_version"],
        "objective_sha256": manifest["objective_sha256"],
        "target_contract": manifest["target_contract"],
        "groups": [
            {"id": group["id"], "expected_count": group["expected_count"]}
            for group in manifest["groups"]
        ],
        "subjects": [
            {
                "tag": entry["tag"],
                "group": entry["group"],
                "fresh_replay_required": entry["fresh_replay_required"],
            }
            for entry in manifest["entries"]
        ],
    }
    return sha256_bytes(canonical_json_bytes(payload))


def auto_control_contract_sha256(root: Path) -> str:
    inventory = load_strict_json(root / "tests/codegen/sm110a_auto_control_inventory.json")
    payload = {
        key: inventory[key]
        for key in (
            "schema_version",
            "objective_sha256",
            "cutlass_git_sha",
            "target",
            "schedule_input",
            "coverage_policy",
            "group_count",
        )
    }
    payload["controls"] = [
        {
            key: entry[key]
            for key in (
                "control_id",
                "group",
                "seed_tag",
                "expected_outcome_hypothesis",
                "hypothesis_failure_layer",
                "hypothesis_evidence",
                "fresh_replay_required",
            )
        }
        for entry in inventory["entries"]
    ]
    return sha256_bytes(canonical_json_bytes(payload))


def validate_source_anchor(root: Path, anchor: dict[str, Any], field: str) -> None:
    cutlass_root = root / "third_party/cutlass"
    path = safe_path(cutlass_root, anchor.get("path"), f"{field}.path")
    lines = path.read_text(encoding="utf-8").splitlines()
    start = anchor.get("line_start")
    end = anchor.get("line_end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start or end > len(lines):
        raise ContractError(f"{field}: invalid source line range {start!r}:{end!r}")
    value = "\n".join(lines[start - 1 : end])
    if sha256_bytes(value.encode("utf-8")) != anchor.get("sha256"):
        raise ContractError(f"{field}: source range SHA-256 mismatch")


def validate_instance(root: Path, path: Path) -> dict[str, Any]:
    instance = load_strict_json(path)
    validate_with_schema(instance, root / "tests/codegen/schemas/instance.schema.json", path.name)
    contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
    if instance["contract_sha256"] != contract_sha256(root):
        raise ContractError(f"{path.name}: contract_sha256 mismatch")
    for key in ("objective_sha256", "freshness_epoch", "target"):
        if instance[key] != contract[key]:
            raise ContractError(f"{path.name}: {key} differs from static contract")
    if instance["static_contract"]["required_layers"] != contract["layer_order"]:
        raise ContractError(f"{path.name}: required_layers differ from the frozen layer order")
    canonical_instance_path = (
        root / "tests/codegen/instances" / instance["instance_id"] / "instance.json"
    )
    if path.resolve() != canonical_instance_path.resolve():
        raise ContractError(f"{path.name}: instance path does not match instance_id")
    canonical_tu_paths = {
        "config": canonical_instance_path.parent / "config.hpp",
        "kernel": canonical_instance_path.parent / "kernel.cu",
        "type_witness": canonical_instance_path.parent / "type_witness.cu",
    }
    for role, expected_path in canonical_tu_paths.items():
        if instance["translation_units"][role]["path"] != expected_path.relative_to(root).as_posix():
            raise ContractError(f"{path.name}: {role} translation unit path is not canonical")
    subject_id = instance["subject"]["id"]
    declared_schedule = instance["declared_config"]["mainloop_schedule"].removeprefix(
        "cutlass::gemm::"
    )
    tags_path = root / "tests/codegen/sm110a_tensor_schedule_tags.json"
    tag_manifest = load_strict_json(tags_path)
    if explicit_subject_inventory_sha256(root) != contract["inventory_contract_sha256"][
        "explicit_tags"
    ]:
        raise ContractError(f"{path.name}: immutable 59-Tag subject inventory drifted")
    if auto_control_contract_sha256(root) != contract["inventory_contract_sha256"][
        "auto_controls"
    ]:
        raise ContractError(f"{path.name}: immutable Auto control inventory drifted")
    if instance["subject"]["kind"] == "explicit_schedule_tag":
        inventory_path = root / "tests/codegen/sm110a_schedule_reference_inventory.json"
        inventory = load_strict_json(inventory_path)
        if instance["provenance"]["reference_inventory_sha256"] != sha256_file(inventory_path):
            raise ContractError(f"{path.name}: reference inventory SHA-256 mismatch")
        if declared_schedule != subject_id:
            raise ContractError(f"{path.name}: subject Tag differs from declared Mainloop Schedule")
        tag_entries = [entry for entry in tag_manifest["entries"] if entry.get("tag") == subject_id]
        if len(tag_entries) != 1 or tag_entries[0].get("group") != instance["subject"]["group"]:
            raise ContractError(f"{path.name}: subject Tag/group differs from the 59-Tag manifest")
        references = [entry for entry in inventory["entries"] if entry.get("tag") == subject_id]
        if len(references) != 1:
            raise ContractError(f"{path.name}: subject Tag has no unique reference inventory entry")
        reference = references[0]
        if instance["provenance"]["reference_class"] != reference["reference_class"]:
            raise ContractError(f"{path.name}: reference class differs from inventory")
        expected_anchor = {
            "path": reference["path"],
            "line_start": reference["line"],
            "line_end": reference["line"],
            "sha256": reference["anchor_line_sha256"],
        }
        if instance["provenance"]["source_anchors"] != [expected_anchor]:
            raise ContractError(f"{path.name}: provenance anchor differs from inventory")
        if reference["reference_class"] == "source_derived":
            parent_id = instance["provenance"]["parent_instance_id"]
            if parent_id is None or parent_id == instance["instance_id"]:
                raise ContractError(f"{path.name}: source-derived instance lacks a distinct parent")
            parent_path = root / "tests/codegen/instances" / parent_id / "instance.json"
            parent = load_strict_json(
                safe_path(root, parent_path.relative_to(root).as_posix(), "parent_instance_id")
            )
            if (
                parent.get("instance_id") != parent_id
                or parent.get("subject", {}).get("id") != reference.get("parent_tag")
                or instance["provenance"]["derivation_axis"] != reference.get("derivation_axis")
            ):
                raise ContractError(f"{path.name}: source-derived parent/axis differs from inventory")
        elif instance["provenance"]["parent_instance_id"] is not None:
            raise ContractError(f"{path.name}: non-derived explicit instance must not declare a parent")
    else:
        auto_path = root / "tests/codegen/sm110a_auto_control_inventory.json"
        auto_inventory = load_strict_json(auto_path)
        if instance["provenance"]["reference_inventory_sha256"] != contract[
            "inventory_contract_sha256"
        ]["auto_controls"]:
            raise ContractError(f"{path.name}: Auto inventory contract SHA-256 mismatch")
        controls = [entry for entry in auto_inventory["entries"] if entry["control_id"] == subject_id]
        if len(controls) != 1:
            raise ContractError(f"{path.name}: Auto control has no unique inventory entry")
        control = controls[0]
        if instance["subject"]["group"] != control["group"]:
            raise ContractError(f"{path.name}: Auto control group differs from inventory")
        if instance["provenance"]["reference_class"] != "auto_control":
            raise ContractError(f"{path.name}: Auto control reference_class mismatch")
        if instance["provenance"]["source_anchors"] != control["hypothesis_evidence"]:
            raise ContractError(f"{path.name}: Auto control anchors differ from inventory")
        if instance["declared_config"]["mainloop_schedule"] != auto_inventory["schedule_input"]:
            raise ContractError(f"{path.name}: Auto control does not declare KernelScheduleAuto")
        parent_id = instance["provenance"]["parent_instance_id"]
        if parent_id is None or parent_id == instance["instance_id"]:
            raise ContractError(f"{path.name}: Auto control lacks a distinct seed instance")
        parent_path = root / "tests/codegen/instances" / parent_id / "instance.json"
        parent = load_strict_json(
            safe_path(root, parent_path.relative_to(root).as_posix(), "parent_instance_id")
        )
        if parent.get("subject", {}).get("id") != control["seed_tag"]:
            raise ContractError(f"{path.name}: Auto seed instance differs from inventory")
        if instance["provenance"]["derivation_axis"] != "mainloop_schedule_to_auto":
            raise ContractError(f"{path.name}: Auto derivation axis mismatch")
        hypothesis = instance["hypothesis"]
        if hypothesis["expected_outcome"] != control["expected_outcome_hypothesis"]:
            raise ContractError(f"{path.name}: Auto outcome hypothesis differs from inventory")
        expected_failure = control["hypothesis_failure_layer"]
        observed_failure = hypothesis["failure_layer"]
        if (expected_failure or "").upper() != (observed_failure or "").upper():
            raise ContractError(f"{path.name}: Auto failure-layer hypothesis differs from inventory")
    for role, reference in instance["translation_units"].items():
        source = safe_path(root, reference["path"], f"translation_units.{role}.path")
        if sha256_file(source) != reference["sha256"]:
            raise ContractError(f"translation_units.{role}: SHA-256 mismatch")
    config_path = root / instance["translation_units"]["config"]["path"]
    kernel_path = root / instance["translation_units"]["kernel"]["path"]
    witness_path = root / instance["translation_units"]["type_witness"]["path"]
    config_text = strip_cpp_comments_and_strings(config_path.read_text(encoding="utf-8"))
    if instance["subject"]["kind"] == "explicit_schedule_tag" and re.search(
        rf"(?<![A-Za-z0-9_])cutlass::gemm::{re.escape(subject_id)}(?![A-Za-z0-9_])",
        config_text,
    ) is None:
        raise ContractError(f"{path.name}: config TU does not instantiate the declared Schedule Tag")
    if instance["subject"]["kind"] == "kernel_schedule_auto_control" and re.search(
        r"(?<![A-Za-z0-9_])cutlass::gemm::collective::KernelScheduleAuto(?![A-Za-z0-9_])",
        config_text,
    ) is None:
        raise ContractError(f"{path.name}: Auto config TU does not instantiate KernelScheduleAuto")
    for role, translation_path in (("kernel", kernel_path), ("type_witness", witness_path)):
        if '#include "config.hpp"' not in translation_path.read_text(encoding="utf-8"):
            raise ContractError(f"{path.name}: {role} TU does not include the shared config.hpp")
    kernel_text = kernel_path.read_text(encoding="utf-8")
    expected_config_type = f"guide::codegen::{instance['instance_id']}::Config"
    if re.search(
        rf"using\s+GemmKernel\s*=\s*{re.escape(expected_config_type)}::GemmKernel\s*;",
        strip_cpp_comments_and_strings(kernel_text),
    ) is None:
        raise ContractError(f"{path.name}: kernel TU does not bind GemmKernel to its Config")
    if "template __global__ void cutlass::device_kernel<GemmKernel>" not in kernel_text:
        raise ContractError(f"{path.name}: kernel TU lacks the actual device_kernel instantiation")
    if re.search(r"\bint\s+main\s*\(", kernel_text) or 'extern "C"' in kernel_text:
        raise ContractError(f"{path.name}: kernel TU contains a wrapper or host entry point")
    witness_text = witness_path.read_text(encoding="utf-8")
    if (
        f"write_codegen_type_report<{expected_config_type}>" not in witness_text
        or f'"{instance["instance_id"]}"' not in witness_text
        or re.search(r"\bint\s+main\s*\(", witness_text) is None
        or "device_kernel<" in witness_text
    ):
        raise ContractError(f"{path.name}: type witness TU is not bound to the same Config/instance")
    for index, anchor in enumerate(instance["provenance"]["source_anchors"]):
        validate_source_anchor(root, anchor, f"provenance.source_anchors[{index}]")
    for artifact_kind in ("ptx", "sass"):
        item = instance["static_contract"][artifact_kind]
        for list_name in ("required", "forbidden"):
            patterns = item[list_name]
            try:
                validate_patterns(
                    [pattern["regex"] for pattern in patterns],
                    f"static_contract.{artifact_kind}.{list_name}",
                )
            except AttributionError as error:
                raise ContractError(str(error)) from error
            for pattern in patterns:
                maximum = pattern["max_count"]
                if maximum is not None and maximum < pattern["min_count"]:
                    raise ContractError(
                        f"{artifact_kind}.{pattern['id']}: max_count is smaller than min_count"
                    )
                if list_name == "required" and pattern["min_count"] < 1:
                    raise ContractError(
                        f"{artifact_kind}.{pattern['id']}: required min_count must be positive"
                    )
                if list_name == "forbidden" and (
                    pattern["min_count"] != 0 or pattern["max_count"] != 0
                ):
                    raise ContractError(
                        f"{artifact_kind}.{pattern['id']}: forbidden contract must require exactly zero matches"
                    )
    hypothesis = instance["hypothesis"]
    if hypothesis["expected_outcome"] == "STATIC_PASS" and any(
        hypothesis[key] is not None for key in ("failure_domain", "failure_layer")
    ):
        raise ContractError(f"{path.name}: STATIC_PASS hypothesis must not declare a failure")
    if hypothesis["expected_outcome"] == "EXPECTED_STATIC_REJECT" and (
        not hypothesis["failure_domain"] or not hypothesis["failure_layer"]
    ):
        raise ContractError(f"{path.name}: reject hypothesis requires failure domain/layer")
    control_id = hypothesis["control_instance_id"]
    if hypothesis["expected_outcome"] == "STATIC_PASS" and control_id is not None:
        raise ContractError(f"{path.name}: STATIC_PASS hypothesis must not declare a control")
    if hypothesis["expected_outcome"] in {"EXPECTED_STATIC_REJECT", "UNSUPPORTED_SM110A"}:
        if control_id is None or control_id == instance["instance_id"]:
            raise ContractError(f"{path.name}: rejection hypothesis lacks a distinct legal control")
    if control_id is not None:
        control_path = root / "tests/codegen/instances" / control_id / "instance.json"
        control_instance = load_strict_json(
            safe_path(root, control_path.relative_to(root).as_posix(), "control_instance_id")
        )
        if control_instance.get("instance_id") != control_id:
            raise ContractError(f"{path.name}: control instance ID mismatch")
    mechanism = instance["declared_config"]["mechanism"]
    block_scaled = mechanism["block_scaled"]
    scale_values = [
        block_scaled["scale_a"],
        block_scaled["scale_b"],
        block_scaled["vector_size_a"],
        block_scaled["vector_size_b"],
    ]
    if block_scaled["enabled"] != all(value is not None for value in scale_values) or (
        not block_scaled["enabled"] and any(value is not None for value in scale_values)
    ):
        raise ContractError(f"{path.name}: block-scaled mechanism fields are inconsistent")
    if block_scaled["enabled"] and block_scaled["scale_a"] != block_scaled["scale_b"]:
        raise ContractError(f"{path.name}: the current block-scale witness requires one shared scale type")
    blockwise = mechanism["blockwise"]
    blockwise_values = [
        blockwise["granularity_m"], blockwise["granularity_n"],
        blockwise["granularity_k"], blockwise["major_a"], blockwise["major_b"],
        blockwise["element_sfa"], blockwise["element_sfb"],
    ]
    if blockwise["enabled"] != all(value is not None for value in blockwise_values) or (
        not blockwise["enabled"] and any(value is not None for value in blockwise_values)
    ):
        raise ContractError(f"{path.name}: blockwise mechanism fields are inconsistent")
    sparse = mechanism["sparse"]
    sparse_values = [
        sparse["metadata_element"], sparse["a_sparsity"], sparse["e_sparsity"]
    ]
    if sparse["enabled"] != all(value is not None for value in sparse_values) or (
        not sparse["enabled"] and any(value is not None for value in sparse_values)
    ):
        raise ContractError(f"{path.name}: sparse mechanism fields are inconsistent")
    mixed = mechanism["mixed_input"]
    mixed_core = [
        mixed["mode"], mixed["operands_swapped"], mixed["transformed_operand"],
        mixed["tuple_arity"], mixed["narrow_type"], mixed["wide_type"],
    ]
    if mixed["enabled"] != all(value is not None for value in mixed_core) or (
        not mixed["enabled"] and any(value is not None for value in mixed_core)
    ):
        raise ContractError(f"{path.name}: mixed-input mechanism fields are inconsistent")
    if mixed["enabled"]:
        expected_arity = {
            "convert_only": 1,
            "scale_only": 2,
            "scale_zero": 3,
        }.get(mixed["mode"])
        if expected_arity != mixed["tuple_arity"]:
            raise ContractError(f"{path.name}: mixed-input tuple arity differs from its mode")
        if (mixed["scale_type"] is not None) != (mixed["tuple_arity"] >= 2) or (
            (mixed["zero_type"] is not None) != (mixed["tuple_arity"] >= 3)
        ):
            raise ContractError(f"{path.name}: mixed-input scale/zero fields differ from tuple arity")
    elif mixed["scale_type"] is not None or mixed["zero_type"] is not None:
        raise ContractError(f"{path.name}: disabled mixed-input carries scale/zero fields")
    fast = mechanism["fast_fp32"]
    fast_values = [
        fast["atom_model"], fast["num_compute_matrices"], fast["num_bands"],
        fast["scaling_factor"], fast["acc_promotion_interval"],
    ]
    if fast["enabled"] != all(value is not None for value in fast_values) or (
        not fast["enabled"] and any(value is not None for value in fast_values)
    ):
        raise ContractError(f"{path.name}: FastFP32 mechanism fields are inconsistent")
    if fast["enabled"]:
        expected_atom_model = (
            "9xBF16-smem" if "FastFP32Smem" in subject_id else "9xBF16-no-smem"
        )
        if fast["atom_model"] != expected_atom_model:
            raise ContractError(f"{path.name}: FastFP32 atom model differs from the Schedule Tag")
    complex_mechanism = mechanism["complex"]
    if complex_mechanism["enabled"] != (complex_mechanism["representation"] is not None):
        raise ContractError(f"{path.name}: complex mechanism fields are inconsistent")
    expected_pointer_mode = "array" if "PtrArray" in subject_id else "single"
    if mechanism["pointer_mode"] != expected_pointer_mode:
        raise ContractError(f"{path.name}: pointer mode differs from the Schedule Tag")
    problem_mode = instance["declared_config"]["builder_contract"]["problem_mode"]
    if expected_pointer_mode == "array" and problem_mode not in {"array", "grouped"}:
        raise ContractError(f"{path.name}: pointer-array Tag requires array/grouped ProblemShape")
    if expected_pointer_mode == "single" and problem_mode in {"array", "grouped"}:
        raise ContractError(f"{path.name}: non-pointer Tag uses an array/grouped ProblemShape")
    group = instance["subject"]["group"]
    expected_mechanisms = {
        "block_scaled": "block_scaled" in group,
        "blockwise": group == "blockwise",
        "sparse": group in {"sparse", "sparse_block_scaled"},
        "mixed_input": group == "mixed_input",
        "fast_fp32": group == "fast_fp32",
        "complex": group in {"planar_complex", "interleaved_complex_tf32"},
    }
    for mechanism_name, expected_enabled in expected_mechanisms.items():
        if mechanism[mechanism_name]["enabled"] is not expected_enabled:
            raise ContractError(
                f"{path.name}: group {group} requires {mechanism_name}.enabled={expected_enabled}"
            )
    expected_complex_representation = {
        "planar_complex": "planar",
        "interleaved_complex_tf32": "interleaved",
    }.get(group)
    if complex_mechanism["representation"] != expected_complex_representation:
        raise ContractError(
            f"{path.name}: group {group} requires complex representation "
            f"{expected_complex_representation!r}"
        )
    if group == "mixed_input":
        expected_transforms = {"a": "swap+transpose", "b": "swap+transpose"}
    elif group in {"sparse", "sparse_block_scaled"}:
        expected_transforms = {"a": "sparse-2:4-compression", "b": "identity"}
    else:
        expected_transforms = {"a": "identity", "b": "identity"}
    if mechanism["transforms"] != expected_transforms:
        raise ContractError(
            f"{path.name}: group {group} requires {expected_transforms} operand transforms"
        )
    required_ptx_patterns = [
        item["regex"] for item in instance["static_contract"]["ptx"]["required"]
    ]
    required_sass_patterns = [
        item["regex"] for item in instance["static_contract"]["sass"]["required"]
    ]
    if not any("mma" in pattern for pattern in required_ptx_patterns) or not any(
        "MMA" in pattern for pattern in required_sass_patterns
    ):
        raise ContractError(f"{path.name}: static contract lacks a required MMA family")
    if block_scaled["enabled"] and not any(
        pattern.lstrip("^").startswith(r"tcgen05\.mma") and "block_scale" in pattern
        for pattern in required_ptx_patterns
    ):
        raise ContractError(f"{path.name}: block-scaled contract lacks a block_scale MMA")
    if sparse["enabled"] and not any(
        pattern.lstrip("^").startswith(r"tcgen05\.mma\.sp")
        for pattern in required_ptx_patterns
    ):
        raise ContractError(f"{path.name}: sparse contract lacks a tcgen05.mma.sp family")
    return instance


def hash_input_tree(items: list[dict[str, str]]) -> str:
    normalized = sorted((item["path"], item["sha256"]) for item in items)
    return sha256_bytes(canonical_json_bytes(normalized))


def expected_execution_plan(instance: dict[str, Any]) -> list[dict[str, Any]]:
    instance_id = instance["instance_id"]
    common = [
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "--expt-relaxed-constexpr",
        f"--frandom-seed={instance_id}",
        "-Iinclude",
        "-Ithird_party/cutlass/include",
        "-Ithird_party/cutlass/tools/util/include",
    ]
    return [
        {
            "step_id": "compile_type_witness",
            "tool": "nvcc",
            "argv": common
            + [
                "--generate-code=arch=compute_110a,code=sm_110a",
                instance["translation_units"]["type_witness"]["path"],
                "-o",
                "<attempt>/full/type_witness",
            ],
        },
        {"step_id": "run_type_witness", "tool": "type_witness", "argv": ["<attempt>/full/type_witness"]},
        {
            "step_id": "compile_fatbin",
            "tool": "nvcc",
            "argv": common
            + [
                "--fatbin",
                "--generate-code=arch=compute_110a,code=compute_110a",
                "--generate-code=arch=compute_110a,code=sm_110a",
                "--ptxas-options=-v",
                instance["translation_units"]["kernel"]["path"],
                "-o",
                "<attempt>/full/kernel.fatbin",
            ],
        },
        {"step_id": "list_ptx", "tool": "cuobjdump", "argv": ["--list-ptx", "<fatbin>"]},
        {"step_id": "list_elf", "tool": "cuobjdump", "argv": ["--list-elf", "<fatbin>"]},
        {"step_id": "extract_ptx", "tool": "cuobjdump", "argv": ["--extract-ptx", "all", "<fatbin>"]},
        {"step_id": "extract_elf", "tool": "cuobjdump", "argv": ["--extract-elf", "all", "<fatbin>"]},
        {"step_id": "elf_symbols", "tool": "cuobjdump", "argv": ["--dump-elf-symbols", "<cubin>"]},
        {"step_id": "code_object_metadata", "tool": "cuobjdump", "argv": ["--dump-sass", "<cubin>"]},
        {"step_id": "nvdisasm_json", "tool": "nvdisasm", "argv": ["--emit-json", "-c", "<cubin>"]},
        {"step_id": "nvdisasm_text", "tool": "nvdisasm", "argv": ["-c", "-sf", "<cubin>"]},
    ]


def record_without_hash(record: dict[str, Any], field: str) -> dict[str, Any]:
    value = dict(record)
    value.pop(field, None)
    return value


def fingerprint_payload(fingerprint: dict[str, Any]) -> dict[str, Any]:
    value = dict(fingerprint)
    value.pop("overall_sha256", None)
    value.pop("fingerprint_id", None)
    return value


def validate_fingerprint(root: Path, path: Path) -> dict[str, Any]:
    fingerprint = load_strict_json(path)
    validate_with_schema(
        fingerprint, root / "tests/codegen/schemas/fingerprint.schema.json", path.name
    )
    if fingerprint["contract_sha256"] != contract_sha256(root):
        raise ContractError("fingerprint contract SHA-256 mismatch")
    contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
    if fingerprint["objective_sha256"] != contract["objective_sha256"]:
        raise ContractError("fingerprint objective SHA-256 mismatch")
    if fingerprint["freshness_epoch"] != contract["freshness_epoch"]:
        raise ContractError("fingerprint freshness epoch mismatch")
    instance_path = safe_path(root, fingerprint["instance_ref"]["path"], "instance_ref.path")
    if sha256_file(instance_path) != fingerprint["instance_ref"]["sha256"]:
        raise ContractError("fingerprint instance_ref SHA-256 mismatch")
    instance = load_strict_json(instance_path)
    if fingerprint["execution_plan"] != expected_execution_plan(instance):
        raise ContractError("fingerprint execution plan differs from the frozen instance plan")
    if fingerprint["environment"]["allowed"] != {"LC_ALL": "C"}:
        raise ContractError("fingerprint allowed environment differs from the frozen environment")
    if fingerprint["source_closure"]["cutlass_git_sha"] != contract["target"]["cutlass_git_sha"]:
        raise ContractError("fingerprint CUTLASS SHA mismatch")
    cutlass_root = root / "third_party/cutlass"
    if (cutlass_root / ".git").exists():
        head = subprocess.check_output(
            ["git", "-C", str(cutlass_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(cutlass_root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).strip()
        if head != fingerprint["source_closure"]["cutlass_git_sha"] or dirty:
            raise ContractError("current CUTLASS checkout does not match the fingerprint")
    inputs = fingerprint["source_closure"]["inputs"]
    if hash_input_tree(inputs) != fingerprint["source_closure"]["input_tree_sha256"]:
        raise ContractError("fingerprint input_tree_sha256 mismatch")
    for index, item in enumerate(inputs):
        source = safe_path(root, item["path"], f"source_closure.inputs[{index}].path")
        if sha256_file(source) != item["sha256"]:
            raise ContractError(f"source_closure.inputs[{index}]: SHA-256 mismatch")
    component_hashes = fingerprint["component_hashes"]
    if component_hashes["instance"] != fingerprint["instance_ref"]["sha256"]:
        raise ContractError("fingerprint instance component hash mismatch")
    if component_hashes["source"] != fingerprint["source_closure"]["input_tree_sha256"]:
        raise ContractError("fingerprint source component hash mismatch")
    if component_hashes["toolchain"] != sha256_bytes(
        canonical_json_bytes(fingerprint["toolchain"])
    ):
        raise ContractError("fingerprint toolchain component hash mismatch")
    if component_hashes["plan"] != sha256_bytes(
        canonical_json_bytes(fingerprint["execution_plan"])
    ):
        raise ContractError("fingerprint plan component hash mismatch")
    inputs_by_path = {item["path"]: item for item in inputs}
    harness_paths = ["tests/codegen/static_codegen_contract.json"]
    harness_paths.extend(contract["record_schemas"].values())
    harness_paths.extend(contract["harness_components"])
    if any(path_value not in inputs_by_path for path_value in harness_paths):
        raise ContractError("fingerprint source closure omits a harness component")
    required_input_paths = {
        fingerprint["instance_ref"]["path"],
        "versions.lock.json",
        "tests/codegen/sm110a_schedule_reference_inventory.json",
    }
    required_input_paths.update(instance["translation_units"][role]["path"] for role in instance["translation_units"])
    required_input_paths.update(
        path_value.relative_to(root).as_posix()
        for path_value in (root / "include/guide").glob("*.hpp")
    )
    expected_input_paths = required_input_paths | set(harness_paths)
    actual_input_paths = set(inputs_by_path)
    if actual_input_paths != expected_input_paths:
        raise ContractError(
            "fingerprint source closure differs from the exact frozen input set: "
            f"missing={sorted(expected_input_paths - actual_input_paths)} "
            f"extra={sorted(actual_input_paths - expected_input_paths)}"
        )
    expected_harness_hash = hash_input_tree([inputs_by_path[path_value] for path_value in harness_paths])
    if component_hashes["harness"] != expected_harness_hash:
        raise ContractError("fingerprint harness component hash mismatch")
    expected_toolchain = contract["toolchain"]
    if fingerprint["toolchain"]["container_digest"] != expected_toolchain["container_digest"]:
        raise ContractError("fingerprint container digest mismatch")
    if fingerprint["toolchain"]["versions_lock_sha256"] != expected_toolchain[
        "versions_lock_file_sha256"
    ]:
        raise ContractError("fingerprint versions lock SHA-256 mismatch")
    expected_tools = {"nvcc", "ptxas", "cuobjdump", "nvdisasm", "host_compiler"}
    if set(fingerprint["toolchain"]["executables"]) != expected_tools:
        raise ContractError("fingerprint toolchain executable set mismatch")
    version_fragments = {
        "nvcc": expected_toolchain["nvcc"],
        "ptxas": expected_toolchain["ptxas"],
        "cuobjdump": expected_toolchain["cuobjdump"],
        "nvdisasm": expected_toolchain["nvdisasm"],
        "host_compiler": "13.3.0",
    }
    for tool, fragment in version_fragments.items():
        tool_record = fingerprint["toolchain"]["executables"][tool]
        if fragment not in tool_record["version"]:
            raise ContractError(f"fingerprint {tool} version mismatch")
        if tool_record["version_sha256"] != sha256_bytes(
            (tool_record["version"] + "\n").encode("utf-8")
        ):
            raise ContractError(f"fingerprint {tool} version SHA-256 mismatch")
        if tool_record["binary_sha256"] != expected_toolchain["tool_binary_sha256"][tool]:
            raise ContractError(f"fingerprint {tool} binary SHA-256 mismatch")
    if fingerprint["environment"]["asserted_absent"] != contract["compile_plan"][
        "asserted_absent_environment"
    ]:
        raise ContractError("fingerprint asserted-absent environment mismatch")
    expected = sha256_bytes(canonical_json_bytes(fingerprint_payload(fingerprint)))
    if expected != fingerprint["overall_sha256"]:
        raise ContractError("fingerprint overall_sha256 mismatch")
    expected_id = f"fp-{fingerprint['instance_ref']['id']}-{expected[:12]}"
    if fingerprint["fingerprint_id"] != expected_id:
        raise ContractError("fingerprint_id is not derived from the overall fingerprint")
    return fingerprint


def journal_event_hash(event: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(record_without_hash(event, "event_sha256")))


def validate_journal(
    root: Path,
    path: Path,
    attempt_id: str,
    fingerprint_id: str,
    instance_id: str | None = None,
) -> tuple[int, str, list[dict[str, Any]]]:
    previous: str | None = None
    last_seq = 0
    seen_terminal = False
    events: list[dict[str, Any]] = []
    allowed_events = set(load_strict_json(root / "tests/codegen/static_codegen_contract.json")["journal_events"])
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ContractError(f"journal line {line_number}: blank lines are forbidden")
        try:
            event = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ContractError) as error:
            raise ContractError(f"journal line {line_number}: {error}") from error
        validate_with_schema(
            event,
            root / "tests/codegen/schemas/journal_event.schema.json",
            f"journal[{line_number}]",
        )
        if event["seq"] != line_number or event["seq"] != last_seq + 1:
            raise ContractError(f"journal line {line_number}: non-contiguous seq")
        if event["previous_event_sha256"] != previous:
            raise ContractError(f"journal line {line_number}: hash chain mismatch")
        if event["attempt_id"] != attempt_id or event["fingerprint_id"] != fingerprint_id:
            raise ContractError(f"journal line {line_number}: attempt/fingerprint mismatch")
        if instance_id is not None and event["instance_id"] != instance_id:
            raise ContractError(f"journal line {line_number}: instance mismatch")
        if event["event_type"] not in allowed_events:
            raise ContractError(f"journal line {line_number}: unknown event type")
        expected = journal_event_hash(event)
        if event["event_sha256"] != expected:
            raise ContractError(f"journal line {line_number}: event SHA-256 mismatch")
        if seen_terminal:
            raise ContractError("journal contains events after ATTEMPT_SEALED")
        seen_terminal = event["event_type"] == "ATTEMPT_SEALED"
        previous = expected
        last_seq = event["seq"]
        events.append(event)
    if not seen_terminal or previous is None:
        raise ContractError("journal is not sealed")
    return last_seq, previous, events


def validate_artifact_manifest(
    root: Path, path: Path, *, require_archive: bool = True
) -> dict[str, Any]:
    manifest = load_strict_json(path)
    validate_with_schema(
        manifest,
        root / "tests/codegen/schemas/artifact_manifest.schema.json",
        path.name,
    )
    if manifest["contract_sha256"] != contract_sha256(root):
        raise ContractError("artifact manifest contract SHA-256 mismatch")
    ids = [item["artifact_id"] for item in manifest["items"]]
    if len(ids) != len(set(ids)):
        raise ContractError("artifact IDs must be unique")
    known: set[str] = set()
    allowed_roles = set(load_strict_json(root / "tests/codegen/static_codegen_contract.json")["artifact_roles"])
    expected_excerpt_dir = Path("evidence/codegen-sm110a-v2/excerpts") / manifest["attempt_id"]
    for index, item in enumerate(manifest["items"]):
        if item["role"] not in allowed_roles:
            raise ContractError(f"items[{index}]: unknown artifact role {item['role']!r}")
        artifact = safe_path(
            root, item["path"], f"items[{index}].path", require_file=False
        )
        must_exist = item["storage"] == "git_evidence" or require_archive
        if not artifact.is_file():
            if must_exist:
                raise ContractError(f"items[{index}]: artifact file is missing")
        elif sha256_file(artifact) != item["sha256"] or artifact.stat().st_size != item["size_bytes"]:
            raise ContractError(f"items[{index}]: artifact size/hash mismatch")
        if any(parent not in known for parent in item["parent_artifact_ids"]):
            raise ContractError(f"items[{index}]: parent lineage is not topologically ordered")
        if item["storage"] == "ignored_archive" and not Path(item["path"]).is_relative_to(
            Path(manifest["archive_bundle"]["path"])
        ):
            raise ContractError(f"items[{index}]: ignored artifact is outside the attempt archive")
        if item["storage"] == "git_evidence" and Path(item["path"]).parent != expected_excerpt_dir:
            raise ContractError(f"items[{index}]: Git evidence is outside the attempt excerpt namespace")
        known.add(item["artifact_id"])
    bundle = manifest["archive_bundle"]
    bundle_relative = Path(bundle["path"])
    if bundle_relative.name != manifest["attempt_id"] or bundle_relative.parent.name != "attempts":
        raise ContractError("archive bundle path does not match the attempt namespace")
    bundle_path = safe_path(root, bundle["path"], "archive_bundle.path", require_file=False)
    if require_archive and not bundle_path.is_dir():
        raise ContractError("archive_bundle.path must be a directory")
    archive_items = [item for item in manifest["items"] if item["storage"] == "ignored_archive"]
    canonical_items = [
        (item["artifact_id"], item["path"], item["sha256"], item["size_bytes"])
        for item in archive_items
    ]
    if sha256_bytes(canonical_json_bytes(canonical_items)) != bundle["sha256"]:
        raise ContractError("archive bundle checksum mismatch")
    if sum(item["size_bytes"] for item in archive_items) != bundle["size_bytes"]:
        raise ContractError("archive bundle size mismatch")
    return manifest


def validate_journal_semantics(
    root: Path,
    events: list[dict[str, Any]],
    fingerprint: dict[str, Any],
    artifacts: dict[str, Any],
    manifest_path: Path,
) -> None:
    event_types = [event["event_type"] for event in events]
    if event_types[:3] != ["ATTEMPT_CREATED", "INSTANCE_VALIDATED", "FINGERPRINT_SEALED"]:
        raise ContractError("journal does not start with the required validation/fingerprint sequence")
    origin = events[0]["payload"]
    origin_guide_root = origin.get("origin_guide_root")
    origin_attempt_root = origin.get("origin_attempt_root")
    origin_user = origin.get("origin_user")
    run_id = origin.get("run_id")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", run_id) is None
        or not isinstance(origin_guide_root, str)
        or not Path(origin_guide_root).is_absolute()
        or not isinstance(origin_attempt_root, str)
        or not Path(origin_attempt_root).is_absolute()
        or not isinstance(origin_user, str)
        or re.fullmatch(r"[0-9]+:[0-9]+", origin_user) is None
    ):
        raise ContractError("ATTEMPT_CREATED lacks a valid immutable origin context")
    attempt_id = events[0]["attempt_id"]
    instance_id = events[0]["instance_id"]
    if re.fullmatch(
        re.escape(f"{run_id}.{instance_id}.a") + r"[0-9]{3}", attempt_id
    ) is None:
        raise ContractError("attempt_id does not match run_id and instance_id")
    expected_bundle_path = (
        Path("artifacts/codegen-sm110a-v2") / run_id / "attempts" / attempt_id
    ).as_posix()
    if artifacts["archive_bundle"]["path"] != expected_bundle_path:
        raise ContractError("archive bundle path does not match ATTEMPT_CREATED run_id")
    origin_attempt = Path(origin_attempt_root)
    if (
        origin_attempt.name != attempt_id
        or origin_attempt.parent.name != "attempts"
        or origin_attempt.parent.parent.name != run_id
    ):
        raise ContractError("ATTEMPT_CREATED attempt root does not match its run namespace")
    if events[1]["payload"] != {
        "instance_sha256": artifacts["instance_ref"]["sha256"]
    }:
        raise ContractError("INSTANCE_VALIDATED does not seal the referenced instance")
    if events[2]["payload"] != {
        "overall_sha256": fingerprint["overall_sha256"]
    }:
        raise ContractError("FINGERPRINT_SEALED does not seal the referenced fingerprint")
    index = 3
    command_events: dict[str, dict[str, Any]] = {}
    attempt_root = root / artifacts["archive_bundle"]["path"]
    for step in fingerprint["execution_plan"]:
        if index + 1 >= len(events):
            raise ContractError("journal ended before all execution-plan commands")
        started, finished = events[index], events[index + 1]
        if started["event_type"] != "STEP_STARTED" or finished["event_type"] != "COMMAND_FINISHED":
            raise ContractError("journal command events are not STARTED -> FINISHED pairs")
        if started["payload"].get("step_id") != step["step_id"] or finished["payload"].get(
            "step_id"
        ) != step["step_id"]:
            raise ContractError("journal command step order differs from the fingerprint plan")
        expected_argv = expected_actual_docker_argv(root, attempt_root, step, artifacts)
        actual_argv = started["payload"].get("argv")
        if not isinstance(actual_argv, list) or normalize_actual_docker_argv(
            actual_argv,
            origin_guide_root=origin_guide_root,
            origin_attempt_root=origin_attempt_root,
            origin_user=origin_user,
        ) != expected_argv:
            raise ContractError(f"journal actual argv differs from frozen plan for {step['step_id']}")
        if finished["payload"].get("returncode") != 0:
            raise ContractError("STATIC_PASS journal contains a failed command")
        command_events[step["step_id"]] = finished
        index += 2
    sealed_events: dict[str, dict[str, Any]] = {}
    while index < len(events) and events[index]["event_type"] == "ARTIFACT_SEALED":
        payload = events[index]["payload"]
        artifact_id = payload.get("artifact_id")
        if not isinstance(artifact_id, str) or artifact_id in sealed_events:
            raise ContractError("journal contains an invalid/duplicate ARTIFACT_SEALED event")
        sealed_events[artifact_id] = payload
        index += 1
    if index + 2 != len(events):
        raise ContractError("journal terminal sequence must be manifest-sealed -> attempt-sealed")
    manifest_event, terminal_event = events[index], events[index + 1]
    if manifest_event["event_type"] != "ARTIFACT_MANIFEST_SEALED" or terminal_event[
        "event_type"
    ] != "ATTEMPT_SEALED":
        raise ContractError("journal terminal sequence is invalid")
    if manifest_event["payload"].get("path") != manifest_path.as_posix() and manifest_event[
        "payload"
    ].get("path") != manifest_path.name:
        # The caller passes an absolute path; stored evidence uses root-relative form and is checked below.
        stored_path = manifest_event["payload"].get("path")
        if not isinstance(stored_path, str) or not str(manifest_path).endswith(stored_path):
            raise ContractError("journal manifest-sealed path mismatch")
    if manifest_event["payload"].get("sha256") != sha256_file(manifest_path):
        raise ContractError("journal manifest-sealed SHA-256 mismatch")
    if terminal_event["payload"].get("terminal_pipeline_state") != "STATIC_PASS":
        raise ContractError("STATIC_PASS result lacks a STATIC_PASS terminal journal event")
    expected_terminal_payload = {
        "terminal_pipeline_state": "STATIC_PASS",
        "artifact_manifest_path": manifest_path.relative_to(root).as_posix(),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "last_completed_step_id": "function_contract",
    }
    if terminal_event["payload"] != expected_terminal_payload:
        raise ContractError("ATTEMPT_SEALED payload does not bind the final manifest/state")

    items_by_id = {item["artifact_id"]: item for item in artifacts["items"]}
    if set(sealed_events) != set(items_by_id):
        raise ContractError("journal ARTIFACT_SEALED set differs from artifact manifest")
    for artifact_id, item in items_by_id.items():
        payload = sealed_events[artifact_id]
        for key in ("path", "sha256", "producer_step_id"):
            if payload.get(key) != item[key]:
                raise ContractError(f"journal artifact seal differs for {artifact_id}.{key}")
        producer = item["producer_step_id"]
        if producer in command_events:
            finished = command_events[producer]["payload"]
            if artifact_id == f"{producer}_stdout" and (
                finished.get("stdout_path") != item["path"]
                or finished.get("stdout_sha256") != item["sha256"]
            ):
                raise ContractError(f"journal stdout binding mismatch for {producer}")
            if artifact_id == f"{producer}_stderr" and (
                finished.get("stderr_path") != item["path"]
                or finished.get("stderr_sha256") != item["sha256"]
            ):
                raise ContractError(f"journal stderr binding mismatch for {producer}")
        elif producer not in {"snapshot_sources", "function_contract"}:
            raise ContractError(f"artifact {artifact_id} has an unknown producer step {producer}")
    for step_id in command_events:
        for stream_name in ("stdout", "stderr"):
            artifact_id = f"{step_id}_{stream_name}"
            if artifact_id not in items_by_id:
                raise ContractError(f"journal command {step_id} lacks {stream_name} artifact")


def expected_actual_docker_argv(
    root: Path,
    attempt_root: Path,
    step: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
) -> list[str]:
    contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
    step_id = step["step_id"]
    tool = step["tool"]
    entrypoints = {
        "nvcc": "/usr/local/cuda/bin/nvcc",
        "cuobjdump": "/usr/local/cuda/bin/cuobjdump",
        "nvdisasm": "/usr/local/cuda/bin/nvdisasm",
        "type_witness": "/out/full/type_witness",
    }
    if tool not in entrypoints:
        raise ContractError(f"execution plan uses unknown tool {tool!r}")
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--security-opt",
        "label=disable",
        "--user",
        "<host-user>",
        "--env",
        "LC_ALL=C",
        "--entrypoint",
        entrypoints[tool],
    ]
    compile_step = step_id in {"compile_type_witness", "compile_fatbin"}
    if compile_step:
        command += ["-v", "<guide-root>:/workspace:ro"]
    mount_suffix = ":ro" if step_id == "run_type_witness" else ""
    command += ["-v", f"<attempt-root>:/out{mount_suffix}"]
    if compile_step:
        command += ["-w", "/workspace"]
    elif step_id in {"extract_ptx", "extract_elf"}:
        command += ["-w", "/out/full/extracted"]
    command.append(contract["toolchain"]["container_reference"])
    if step_id == "run_type_witness":
        return command
    translated: list[str] = []
    for argument in step["argv"]:
        if argument == "<fatbin>":
            if artifacts is None:
                translated.append("/out/full/kernel.fatbin")
            else:
                item = next(
                    (value for value in artifacts["items"] if value["role"] == "FATBIN"), None
                )
                if item is None:
                    raise ContractError("cannot resolve FATBIN artifact in actual command validation")
                translated.append(
                    "/out/" + (root / item["path"]).relative_to(attempt_root).as_posix()
                )
        elif argument == "<cubin>":
            if artifacts is None:
                cubins = sorted((attempt_root / "full/extracted").glob("*.cubin"))
                if len(cubins) != 1:
                    raise ContractError("cannot resolve <cubin> in actual command validation")
                translated.append(f"/out/full/extracted/{cubins[0].name}")
            else:
                item = next(
                    (value for value in artifacts["items"] if value["role"] == "CUBIN"), None
                )
                if item is None:
                    raise ContractError("cannot resolve CUBIN artifact in actual command validation")
                translated.append(
                    "/out/" + (root / item["path"]).relative_to(attempt_root).as_posix()
                )
        elif argument.startswith("<attempt>"):
            translated.append(argument.replace("<attempt>", "/out", 1))
        else:
            translated.append(argument)
    return command + translated


def normalize_actual_docker_argv(
    command: list[Any],
    *,
    origin_guide_root: str,
    origin_attempt_root: str,
    origin_user: str,
) -> list[Any]:
    normalized = list(command)
    for index, value in enumerate(normalized[:-1]):
        if value == "--user":
            if normalized[index + 1] != origin_user:
                raise ContractError("Docker --user differs from ATTEMPT_CREATED origin")
            normalized[index + 1] = "<host-user>"
        elif value == "-v" and isinstance(normalized[index + 1], str):
            mount = normalized[index + 1]
            if mount.endswith(":/workspace:ro"):
                if mount[: -len(":/workspace:ro")] != origin_guide_root:
                    raise ContractError("workspace mount source differs from ATTEMPT_CREATED origin")
                normalized[index + 1] = "<guide-root>:/workspace:ro"
            elif mount.endswith(":/out:ro"):
                if mount[: -len(":/out:ro")] != origin_attempt_root:
                    raise ContractError("read-only attempt mount differs from ATTEMPT_CREATED origin")
                normalized[index + 1] = "<attempt-root>:/out:ro"
            elif mount.endswith(":/out"):
                if mount[: -len(":/out")] != origin_attempt_root:
                    raise ContractError("attempt mount source differs from ATTEMPT_CREATED origin")
                normalized[index + 1] = "<attempt-root>:/out"
    return normalized


def _single_artifact_by_role(
    root: Path, artifacts: dict[str, Any], role: str
) -> tuple[dict[str, Any], Path]:
    matches = [item for item in artifacts["items"] if item["role"] == role]
    if len(matches) != 1:
        raise ContractError(f"artifact manifest requires exactly one {role}, found {len(matches)}")
    item = matches[0]
    return item, safe_path(root, item["path"], f"artifact role {role}")


def _expected_pattern_results(
    contracts: list[dict[str, Any]],
    matches: dict[str, tuple[str, ...]],
    required: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "id": contract["id"],
            "required": required,
            "match_count": len(matches.get(contract["regex"], ())),
            "matched_opcodes": list(matches.get(contract["regex"], ())),
        }
        for contract in contracts
    ]


def validate_device_kernel_symbol(symbol: str, gemm_kernel_type: str) -> None:
    normal_kernel = gemm_kernel_type.startswith("cutlass::gemm::kernel::GemmUniversal<")
    abi_kernel = gemm_kernel_type.startswith(
        "abi-mangled:N7cutlass4gemm6kernel13GemmUniversalI"
    ) and re.fullmatch(
        r"abi-mangled:[A-Za-z0-9_]+", gemm_kernel_type
    ) is not None
    if not normal_kernel and not abi_kernel:
        raise ContractError("type witness GemmKernel differs from the target device_kernel symbol")
    if not symbol.startswith("_ZN7cutlass13device_kernelI") or not symbol.endswith(
        "EvNT_6ParamsE"
    ):
        raise ContractError(
            "target symbol is not a cutlass::device_kernel<GemmKernel>(Params) specialization"
        )


def validate_static_pass_evidence(
    root: Path,
    result: dict[str, Any],
    instance: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    require_archive: bool,
) -> None:
    evidence = result["evidence"]
    items_by_id = {item["artifact_id"]: item for item in artifacts["items"]}
    canonical_artifacts = {
        "config_source": (
            "KERNEL_SOURCE_SNAPSHOT",
            "ignored_archive",
            "snapshot_sources",
            [],
        ),
        "kernel_source": (
            "KERNEL_SOURCE_SNAPSHOT",
            "ignored_archive",
            "snapshot_sources",
            ["config_source"],
        ),
        "type_source": (
            "TYPE_WITNESS_SOURCE_SNAPSHOT",
            "ignored_archive",
            "snapshot_sources",
            ["config_source"],
        ),
        "type_executable": (
            "TYPE_WITNESS_EXECUTABLE",
            "ignored_archive",
            "compile_type_witness",
            ["type_source", "config_source"],
        ),
        "type_output": (
            "TYPE_WITNESS_OUTPUT",
            "git_evidence",
            "run_type_witness",
            ["type_executable"],
        ),
        "fatbin": (
            "FATBIN",
            "ignored_archive",
            "compile_fatbin",
            ["kernel_source", "config_source"],
        ),
        "ptx_full": ("PTX_FULL", "ignored_archive", "extract_ptx", ["fatbin"]),
        "cubin": ("CUBIN", "ignored_archive", "extract_elf", ["fatbin"]),
        "elf_symbols": (
            "ELF_SYMBOL_TABLE",
            "ignored_archive",
            "elf_symbols",
            ["cubin"],
        ),
        "code_metadata": (
            "CODE_OBJECT_METADATA",
            "ignored_archive",
            "code_object_metadata",
            ["cubin"],
        ),
        "nvdisasm_json": (
            "NVDISASM_JSON_FULL",
            "ignored_archive",
            "nvdisasm_json",
            ["cubin"],
        ),
        "nvdisasm_text": (
            "NVDISASM_TEXT_FULL",
            "ignored_archive",
            "nvdisasm_text",
            ["cubin"],
        ),
        "ptx_target": (
            "PTX_TARGET_FUNCTION",
            "git_evidence",
            "function_contract",
            ["ptx_full"],
        ),
        "sass_target": (
            "SASS_TARGET_FUNCTION",
            "git_evidence",
            "function_contract",
            ["nvdisasm_json"],
        ),
        "contract_report": (
            "CONTRACT_CHECK_REPORT",
            "git_evidence",
            "function_contract",
            ["ptx_target", "sass_target", "type_output"],
        ),
    }
    for artifact_id, (role, storage, producer, parents) in canonical_artifacts.items():
        item = items_by_id.get(artifact_id)
        if item is None:
            raise ContractError(f"artifact manifest lacks canonical artifact {artifact_id}")
        if item["role"] != role:
            raise ContractError(f"canonical artifact {artifact_id} has the wrong role")
        if item["storage"] != storage:
            raise ContractError(f"canonical artifact {artifact_id} has the wrong storage class")
        if item["producer_step_id"] != producer:
            raise ContractError(f"canonical artifact {artifact_id} has the wrong producer")
        if item["parent_artifact_ids"] != parents:
            raise ContractError(f"artifact lineage mismatch for {artifact_id}")
    source_hash_bindings = {
        "config_source": instance["translation_units"]["config"]["sha256"],
        "kernel_source": instance["translation_units"]["kernel"]["sha256"],
        "type_source": instance["translation_units"]["type_witness"]["sha256"],
    }
    for artifact_id, expected_sha256 in source_hash_bindings.items():
        if items_by_id[artifact_id]["sha256"] != expected_sha256:
            raise ContractError(
                f"source snapshot {artifact_id} differs from the referenced translation unit"
            )
    role_items: dict[str, dict[str, Any]] = {}
    role_paths: dict[str, Path] = {}
    tracked_roles = (
        "TYPE_WITNESS_OUTPUT",
        "PTX_TARGET_FUNCTION",
        "SASS_TARGET_FUNCTION",
        "CONTRACT_CHECK_REPORT",
    )
    archive_roles = (
        "FATBIN",
        "PTX_FULL",
        "CUBIN",
        "ELF_SYMBOL_TABLE",
        "CODE_OBJECT_METADATA",
        "NVDISASM_JSON_FULL",
        "NVDISASM_TEXT_FULL",
    )
    for role in tracked_roles + (archive_roles if require_archive else ()):
        role_items[role], role_paths[role] = _single_artifact_by_role(root, artifacts, role)
    if require_archive:
        copy_bindings = {
            "TYPE_WITNESS_OUTPUT": "run_type_witness_stdout",
            "ELF_SYMBOL_TABLE": "elf_symbols_stdout",
            "CODE_OBJECT_METADATA": "code_object_metadata_stdout",
            "NVDISASM_JSON_FULL": "nvdisasm_json_stdout",
            "NVDISASM_TEXT_FULL": "nvdisasm_text_stdout",
        }
        for role, stdout_id in copy_bindings.items():
            stdout_item = items_by_id.get(stdout_id)
            if stdout_item is None or stdout_item["sha256"] != role_items[role]["sha256"]:
                raise ContractError(f"artifact role {role} differs from its command stdout")

    type_witness = load_strict_json(role_paths["TYPE_WITNESS_OUTPUT"])
    if evidence["type_witness_artifact_id"] != role_items["TYPE_WITNESS_OUTPUT"]["artifact_id"]:
        raise ContractError("result type_witness_artifact_id is not the unique TYPE_WITNESS_OUTPUT")
    if type_witness.get("instance_id") != instance["instance_id"]:
        raise ContractError("type witness instance_id mismatch")
    resolved_types = type_witness.get("resolved_types")
    resolved_values = type_witness.get("resolved_values")
    if not isinstance(resolved_types, dict) or not isinstance(resolved_values, dict):
        raise ContractError("type witness lacks resolved types/values")
    for key in (
        "collective_mainloop",
        "mainloop_builder",
        "mainloop_builder_collective_op",
        "epilogue_builder",
        "epilogue_builder_collective_op",
        "dispatch_policy",
        "dispatch_schedule",
        "tiled_mma",
        "mma_atom",
        "mainloop_dispatch_policy",
        "mainloop_tiled_mma",
        "tiled_mma_atom",
        "mma_value_type_a",
        "mma_value_type_b",
        "mma_value_type_c",
        "mma_fragment_type_a",
        "mma_fragment_type_b",
        "mma_operand_source_a",
        "mma_operand_source_b",
        "gemm_kernel",
        "gmem_tiled_copy_a",
        "gmem_tiled_copy_b",
        "smem_layout_atom_a",
        "smem_layout_atom_b",
        "smem_copy_atom_a",
        "smem_copy_atom_b",
        "collective_epilogue",
        "kernel_collective_mainloop",
        "kernel_collective_epilogue",
        "config_arch_tag",
        "config_operator_class",
        "mainloop_operator_class",
        "epilogue_operator_class",
        "config_element_a",
        "config_element_b",
        "config_element_c",
        "config_element_compute",
        "config_element_accumulator",
        "config_element_d",
        "config_layout_a",
        "config_layout_b",
        "config_layout_c",
        "config_layout_d",
        "builder_element_a",
        "builder_element_b",
        "builder_layout_a",
        "builder_layout_b",
        "epilogue_element_c",
        "epilogue_element_d",
        "epilogue_layout_c",
        "epilogue_layout_d",
        "epilogue_tile",
        "fusion_operation",
        "config_mainloop_schedule",
        "config_epilogue_schedule",
        "config_stage_policy",
        "config_problem_shape",
        "config_cluster_shape",
        "config_cluster_default_shape",
        "config_tile_scheduler",
    ):
        if not isinstance(resolved_types.get(key), str) or not resolved_types[key]:
            raise ContractError(f"type witness lacks {key}")
    cross_layer_type_bindings = (
        ("collective_mainloop", "mainloop_builder_collective_op"),
        ("collective_epilogue", "epilogue_builder_collective_op"),
        ("dispatch_policy", "mainloop_dispatch_policy"),
        ("tiled_mma", "mainloop_tiled_mma"),
        ("mma_atom", "tiled_mma_atom"),
        ("collective_mainloop", "kernel_collective_mainloop"),
        ("collective_epilogue", "kernel_collective_epilogue"),
    )
    for canonical_key, consumer_view_key in cross_layer_type_bindings:
        if resolved_types[canonical_key] != resolved_types[consumer_view_key]:
            raise ContractError(
                f"type witness cross-layer binding differs for {canonical_key}"
            )
    for axis in ("a", "b"):
        fragment_type = resolved_types[f"mma_fragment_type_{axis}"]
        reported_source = resolved_types[f"mma_operand_source_{axis}"]
        classified_source = classify_mma_fragment_type(fragment_type)
        if classified_source == "UNKNOWN" or reported_source != classified_source:
            raise ContractError(
                f"type witness cannot classify MMA operand {axis.upper()} source"
            )
    dispatch_template = split_cpp_template(
        normalize_cpp_type(resolved_types["dispatch_policy"])
    )
    dispatch_base = dispatch_template[0] if dispatch_template is not None else ""
    allowed_dispatch_bases = {
        "dense": {
            "cutlass::gemm::MainloopSm100UmmaCpAsyncWarpSpecialized",
            "cutlass::gemm::MainloopSm100UmmaMixedTmaCpAsyncWarpSpecialized",
            "cutlass::gemm::MainloopSm100TmaUmmaWarpSpecialized",
        },
        "ptr_array_dense": {
            "cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecialized",
            "cutlass::gemm::MainloopSm100RCGroupGemmTmaUmmaWarpSpecialized",
        },
        "blockwise": {
            "cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling",
            "cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedBlockwiseScaling",
        },
        "planar_complex": {
            "cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedPlanarComplex",
            "cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedPlanarComplex",
        },
        "fast_fp32": {
            "cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedFastF32",
            "cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedFastF32",
        },
        "mixed_input": {
            "cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedMixedInput",
        },
        "interleaved_complex_tf32": {
            "cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedInterleavedComplexTF32",
            "cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedInterleavedComplexTF32",
        },
        "sparse": {
            "cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedSparse",
        },
        "dense_block_scaled": {
            "cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled",
            "cutlass::gemm::MainloopSm100UmmaMixedTmaCpAsyncWarpSpecializedBlockScaled",
        },
        "ptr_array_block_scaled": {
            "cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedBlockScaled",
            "cutlass::gemm::MainloopSm100RCGroupGemmTmaUmmaWarpSpecializedBlockScaled",
        },
        "sparse_block_scaled": {
            "cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaledSparse",
        },
    }
    subject_group = instance["subject"]["group"]
    if dispatch_base not in allowed_dispatch_bases.get(subject_group, set()):
        raise ContractError(
            f"type witness DispatchPolicy family differs from subject group {subject_group}"
        )
    declared = instance["declared_config"]
    exact_config_types = {
        "config_arch_tag": instance["target"]["cutlass_arch_tag"],
        "config_operator_class": declared["operator_class"],
        "config_element_a": declared["elements"]["a"],
        "config_element_b": declared["elements"]["b"],
        "config_element_c": declared["elements"]["c"],
        "config_element_compute": declared["elements"]["compute"],
        "config_element_accumulator": declared["elements"]["accumulator"],
        "config_element_d": declared["elements"]["d"],
        "mainloop_operator_class": declared["builder_contract"]["mainloop_operator_class"],
        "epilogue_operator_class": declared["builder_contract"]["epilogue_operator_class"],
        "builder_element_a": declared["builder_contract"]["element_a"],
        "builder_element_b": declared["builder_contract"]["element_b"],
        "builder_layout_a": declared["builder_contract"]["layout_a"],
        "builder_layout_b": declared["builder_contract"]["layout_b"],
        "epilogue_element_c": declared["builder_contract"]["epilogue_element_c"],
        "epilogue_element_d": declared["builder_contract"]["epilogue_element_d"],
        "epilogue_layout_c": declared["builder_contract"]["epilogue_layout_c"],
        "epilogue_layout_d": declared["builder_contract"]["epilogue_layout_d"],
        "epilogue_tile": declared["builder_contract"]["epilogue_tile"],
        "fusion_operation": declared["builder_contract"]["fusion_operation"],
        "config_mainloop_schedule": declared["mainloop_schedule"],
        "config_epilogue_schedule": declared["epilogue_schedule"],
        "config_problem_shape": declared["kernel_problem_shape"],
        "config_cluster_shape": declared["builder_contract"]["cluster_type_cpp"],
    }
    for witness_key, declared_value in exact_config_types.items():
        equivalent = cpp_type_equivalent(
            resolved_types[witness_key],
            declared_value,
            allow_trailing_default_arguments=(witness_key == "fusion_operation"),
        )
        if not equivalent:
            raise ContractError(f"type witness {witness_key} differs from declared_config")
    for axis in ("a", "b", "c", "d"):
        declared_layout = declared["layouts"][axis]
        layout_name = next(
            (name for name in ("RowMajor", "ColumnMajor") if name in declared_layout),
            None,
        )
        normalized_layout = normalize_cpp_type(resolved_types[f"config_layout_{axis}"])
        if layout_name is None or not normalized_layout.rstrip("*").endswith(layout_name):
            raise ContractError(f"type witness config_layout_{axis} differs from declared_config")
    default_cluster_type = resolved_types["config_cluster_default_shape"]
    if resolved_values.get("cluster_mnk") != declared["builder_contract"]["cluster_default_mnk"]:
        raise ContractError("type witness cluster default differs from declared_config")
    if declared["builder_contract"]["cluster_is_dynamic"] is False and (
        normalize_cpp_type(default_cluster_type)
        != normalize_cpp_type(resolved_types["config_cluster_shape"])
    ):
        raise ContractError("static cluster type differs from its default cluster type")
    problem_type = resolved_types["config_problem_shape"]
    problem_mode = declared["builder_contract"]["problem_mode"]
    problem_markers = {
        "array": "ArrayProblemShape",
        "grouped": "GroupProblemShape",
        "moe": "MoEProblemShape",
    }
    for mode, marker in problem_markers.items():
        if (marker in problem_type) is not (problem_mode == mode):
            raise ContractError("type witness ProblemShape family differs from declared problem_mode")
    stage_policy = declared["stage_policy"]
    stage_family = stage_policy.split("<", 1)[0].rsplit("::", 1)[-1]
    resolved_stage_policy = resolved_types["config_stage_policy"]
    if stage_family not in resolved_stage_policy:
        raise ContractError("type witness config_stage_policy differs from declared_config")
    epilogue_storage_bytes = resolved_values.get("epilogue_shared_storage_bytes")
    if not isinstance(epilogue_storage_bytes, int) or epilogue_storage_bytes < 0:
        raise ContractError("type witness lacks epilogue_shared_storage_bytes")
    if stage_family == "StageCountAutoCarveout":
        expected_policy = (
            "cutlass::gemm::collective::StageCountAutoCarveout<"
            f"{epilogue_storage_bytes}>"
        )
        if normalize_cpp_type(resolved_stage_policy) != normalize_cpp_type(expected_policy):
            raise ContractError("resolved StageCountAutoCarveout bytes differ from Epilogue storage")
    elif stage_family == "StageCountAutoCarveoutEpi":
        if resolved_types["collective_epilogue"] not in resolved_stage_policy:
            raise ContractError("resolved StageCountAutoCarveoutEpi does not bind the Epilogue type")
    elif normalize_cpp_type(resolved_stage_policy) != normalize_cpp_type(stage_policy):
        raise ContractError("resolved stage policy differs from the declared policy")
    declared_scheduler = declared["tile_scheduler"].split(" ", 1)[0]
    if normalize_cpp_type(resolved_types["config_tile_scheduler"]) != normalize_cpp_type(
        declared_scheduler
    ):
        raise ContractError("type witness config_tile_scheduler differs from declared_config")
    optional_types = type_witness.get("resolved_optional_types")
    if not isinstance(optional_types, dict):
        raise ContractError("type witness lacks resolved_optional_types")
    expected_stage = {
        "declared_policy_cpp": instance["declared_config"]["stage_policy"],
        "resolved_policy_cpp": resolved_types["config_stage_policy"],
        "stage_count": resolved_values.get("mainloop_stages"),
        "scheduler_stages": resolved_values.get("scheduler_stages"),
        "accumulator_stages": resolved_values.get("accumulator_stages"),
        "load_to_transform_stages": resolved_values.get("load_to_transform_stages"),
        "transform_to_mma_stages": resolved_values.get("transform_to_mma_stages"),
        "computation_stages": resolved_values.get("computation_stages"),
        "transformation_stages": resolved_values.get("transformation_stages"),
    }
    if evidence["resolved_stage"] != expected_stage:
        raise ContractError("result resolved_stage differs from type witness")
    if expected_stage["stage_count"] is None or expected_stage["stage_count"] <= 0:
        raise ContractError("type witness mainloop stage count must be positive")
    atom_shape = resolved_values.get("atom_shape_mnk")
    if (
        not isinstance(atom_shape, list)
        or len(atom_shape) != 3
        or any(not isinstance(value, int) or value <= 0 for value in atom_shape)
    ):
        raise ContractError("type witness lacks a positive atom_shape_mnk")
    for axis in ("a", "b"):
        tuple_arity = resolved_values.get(f"builder_tuple_arity_{axis}")
        if not isinstance(tuple_arity, int) or tuple_arity < 0:
            raise ContractError(f"type witness lacks builder_tuple_arity_{axis}")
    for witness_key, declared_key in (
        ("mma_tile_mnk", "mma_tile_mnk"),
        ("cluster_mnk", "cluster_mnk"),
    ):
        if resolved_values.get(witness_key) != instance["declared_config"][declared_key]:
            raise ContractError(f"type witness {witness_key} differs from the instance")
    for axis in ("a", "b", "c", "d"):
        if resolved_values.get(f"alignment_{axis}") != declared["alignments"][axis]:
            raise ContractError(f"type witness alignment_{axis} differs from declared_config")
    mechanism = instance["declared_config"]["mechanism"]
    scale_optional_keys = {
        "scale_element",
        "gmem_tiled_copy_sfa",
        "gmem_tiled_copy_sfb",
        "smem_layout_atom_sfa",
        "smem_layout_atom_sfb",
    }
    sparse_optional_keys = {
        "sparse_config",
        "smem_layout_atom_e",
        "element_e",
        "gmem_copy_atom_e",
        "smem_layout_e",
    }
    input_compute_optional_keys = {
        "smem_layout_atoms_a",
        "smem_layout_atoms_b",
        "input_copy_atom_a",
        "input_copy_atom_b",
        "compute_copy_atom_a",
        "compute_copy_atom_b",
    }
    planar_optional_keys = {
        "planar_tiled_mma_pair",
        "planar_tiled_mma_a_negative",
    }
    scale_factor_optional_keys = {
        "scale_factor_tiled_mma",
        "scale_factor_mma_atom",
    }
    blockwise_optional_keys = {
        "blockwise_scale_config",
        "blockwise_element_sfa",
        "blockwise_element_sfb",
        "blockwise_layout_sfa",
        "blockwise_layout_sfb",
        "blockwise_major_a",
        "blockwise_major_b",
    }
    mixed_optional_keys = {
        "mixed_element_scale",
        "mixed_element_zero",
        "mixed_layout_scale",
        "mixed_gmem_tiled_copy_scale",
        "mixed_smem_layout_atom_scale",
    }
    allowed_optional_keys: set[str] = set()
    if mechanism["block_scaled"]["enabled"]:
        allowed_optional_keys.update(scale_optional_keys)
        for key in scale_optional_keys:
            if not isinstance(optional_types.get(key), str) or not optional_types[key]:
                raise ContractError(f"block-scaled type witness lacks {key}")
        scale_a = mechanism["block_scaled"]["scale_a"]
        scale_b = mechanism["block_scaled"]["scale_b"]
        if scale_a == scale_b and scale_a not in optional_types["scale_element"]:
            raise ContractError("type witness scale element differs from declared block-scale type")
        resolved_vector_size = resolved_values.get("scale_vector_size")
        if (
            resolved_values.get("scale_vector_size_a") != resolved_vector_size
            or resolved_values.get("scale_vector_size_b") != resolved_vector_size
            or mechanism["block_scaled"]["vector_size_a"] != resolved_vector_size
            or mechanism["block_scaled"]["vector_size_b"] != resolved_vector_size
        ):
            raise ContractError("type witness scale vector size differs from the declared mechanism")
    elif (
        any(
            resolved_values.get(key) != 0
            for key in ("scale_vector_size", "scale_vector_size_a", "scale_vector_size_b")
        )
        or scale_optional_keys & set(optional_types)
    ):
        raise ContractError("non-block-scaled type witness contains block-scale details")
    if mechanism["sparse"]["enabled"]:
        allowed_optional_keys.update(sparse_optional_keys)
        for key in sparse_optional_keys:
            if not isinstance(optional_types.get(key), str) or not optional_types[key]:
                raise ContractError(f"sparse type witness lacks {key}")
        sparse_declared = mechanism["sparse"]
        if resolved_values.get("element_a_sparsity") != sparse_declared["a_sparsity"]:
            raise ContractError("sparse type witness does not expose the declared A sparsity")
        if resolved_values.get("element_e_sparsity") != sparse_declared["e_sparsity"]:
            raise ContractError("sparse type witness metadata sparsity differs from the declaration")
        if not cpp_type_equivalent(
            optional_types["element_e"], sparse_declared["metadata_element"]
        ):
            raise ContractError("sparse metadata element type differs from the declaration")
        sparse_config = split_cpp_template(
            normalize_cpp_type(optional_types["sparse_config"])
        )
        if sparse_config is None or sparse_config[0] != "cutlass::Sm1xxGemmSparseConfig":
            raise ContractError("sparse type witness has an unrelated SparseConfig")
    elif sparse_optional_keys & set(optional_types):
        raise ContractError("dense type witness unexpectedly contains sparse metadata details")
    elif resolved_values.get("element_a_sparsity") != 0 or resolved_values.get(
        "element_e_sparsity"
    ) != 0:
        raise ContractError("dense type witness unexpectedly contains sparse ratio details")
    if mechanism["blockwise"]["enabled"]:
        allowed_optional_keys.update(blockwise_optional_keys)
        for key in blockwise_optional_keys:
            if not isinstance(optional_types.get(key), str) or not optional_types[key]:
                raise ContractError(f"blockwise type witness lacks {key}")
        for key, declared_key in (
            ("blockwise_granularity_m", "granularity_m"),
            ("blockwise_granularity_n", "granularity_n"),
            ("blockwise_granularity_k", "granularity_k"),
        ):
            if resolved_values.get(key) != mechanism["blockwise"][declared_key]:
                raise ContractError(f"blockwise type witness {key} differs from declared mechanism")
        for axis in ("a", "b"):
            if optional_types[f"blockwise_major_{axis}"] != mechanism["blockwise"][f"major_{axis}"]:
                raise ContractError(
                    f"blockwise type witness major_{axis} differs from declared mechanism"
                )
            if not cpp_type_equivalent(
                optional_types[f"blockwise_element_sf{axis}"],
                mechanism["blockwise"][f"element_sf{axis}"],
            ):
                raise ContractError(
                    f"blockwise type witness ElementSF{axis.upper()} differs from declared mechanism"
                )
        scale_config = split_cpp_template(
            normalize_cpp_type(optional_types["blockwise_scale_config"])
        )
        major_value = {"K": 0, "MN": 1}
        expected_scale_arguments = [
            str(mechanism["blockwise"]["granularity_m"]),
            str(mechanism["blockwise"]["granularity_n"]),
            str(mechanism["blockwise"]["granularity_k"]),
            f"(cute::UMMA::Major){major_value[mechanism['blockwise']['major_a']]}",
            f"(cute::UMMA::Major){major_value[mechanism['blockwise']['major_b']]}",
        ]
        if (
            scale_config is None
            or scale_config[0] != "cutlass::detail::Sm1xxBlockwiseScaleConfig"
            or scale_config[1] != expected_scale_arguments
        ):
            raise ContractError("blockwise ScaleConfig differs from the declared mechanism")
    elif blockwise_optional_keys & set(optional_types) or any(
        resolved_values.get(key) != 0
        for key in ("blockwise_granularity_m", "blockwise_granularity_n", "blockwise_granularity_k")
    ):
        raise ContractError("non-blockwise type witness contains blockwise details")
    needs_input_compute = (
        mechanism["fast_fp32"]["enabled"]
        or mechanism["mixed_input"]["enabled"]
        or mechanism["complex"]["representation"] == "interleaved"
    )
    if needs_input_compute:
        allowed_optional_keys.update(input_compute_optional_keys)
        for key in input_compute_optional_keys:
            if not isinstance(optional_types.get(key), str) or not optional_types[key]:
                raise ContractError(f"transform-pipeline type witness lacks {key}")
    elif input_compute_optional_keys & set(optional_types):
        raise ContractError("type witness contains undeclared input/compute copy roles")
    if mechanism["complex"]["enabled"]:
        for axis in ("a", "b"):
            builder_complex = split_cpp_template(
                normalize_cpp_type(resolved_types[f"builder_element_{axis}"])
            )
            if (
                builder_complex is None
                or builder_complex[0] != "cute::tuple"
                or len(builder_complex[1]) != 2
                or resolved_values.get(f"builder_tuple_arity_{axis}") != 2
                or not cpp_type_equivalent(
                    builder_complex[1][0], resolved_types[f"config_element_{axis}"]
                )
                or not cpp_type_equivalent(builder_complex[1][1], "cute::identity")
            ):
                raise ContractError(
                    f"complex type witness Builder operand {axis.upper()} lacks value/transform pair"
                )
    if mechanism["complex"]["representation"] == "planar":
        allowed_optional_keys.update(planar_optional_keys)
        for key in planar_optional_keys:
            if not isinstance(optional_types.get(key), str) or not optional_types[key]:
                raise ContractError(f"planar-complex type witness lacks {key}")
        planar_pair = normalize_cpp_type(optional_types["planar_tiled_mma_pair"])
        planar_negative = normalize_cpp_type(
            optional_types["planar_tiled_mma_a_negative"]
        )
        planar_pair_template = split_cpp_template(planar_pair)
        if (
            planar_pair_template is None
            or planar_pair_template[0]
            != "cutlass::gemm::collective::detail::Sm100CollectiveMmaPlanarComplexTiledMmaType"
            or normalize_cpp_type(resolved_types["tiled_mma"]) not in planar_pair
            or planar_negative not in planar_pair
            or planar_negative == normalize_cpp_type(resolved_types["tiled_mma"])
        ):
            raise ContractError("planar-complex TiledMMA pair does not bind positive and negative roles")
    elif planar_optional_keys & set(optional_types):
        raise ContractError("type witness contains undeclared planar Atom roles")
    needs_scale_factor_atom = (
        mechanism["block_scaled"]["enabled"]
        and "MixedTmaCpAsync" in instance["subject"]["id"]
    )
    if needs_scale_factor_atom:
        allowed_optional_keys.update(scale_factor_optional_keys)
        for key in scale_factor_optional_keys:
            if not isinstance(optional_types.get(key), str) or not optional_types[key]:
                raise ContractError(f"mixed block-scaled type witness lacks {key}")
        scale_factor_tiled_mma = normalize_cpp_type(
            optional_types["scale_factor_tiled_mma"]
        )
        scale_factor_atom = normalize_cpp_type(optional_types["scale_factor_mma_atom"])
        scale_factor_template = split_cpp_template(scale_factor_tiled_mma)
        if (
            scale_factor_template is None
            or scale_factor_template[0] != "cute::TiledMMA"
            or scale_factor_atom not in scale_factor_tiled_mma
            or scale_factor_atom == normalize_cpp_type(resolved_types["mma_atom"])
        ):
            raise ContractError("mixed block-scaled scale-factor TiledMMA does not bind its Atom")
    elif scale_factor_optional_keys & set(optional_types):
        raise ContractError("type witness contains an undeclared scale-factor Atom")
    if mechanism["mixed_input"]["enabled"]:
        allowed_optional_keys.update(mixed_optional_keys)
        for key in mixed_optional_keys:
            if not isinstance(optional_types.get(key), str) or not optional_types[key]:
                raise ContractError(f"mixed-input type witness lacks {key}")
        scale_type = optional_types["mixed_element_scale"]
        zero_type = optional_types["mixed_element_zero"]
        if ("void" not in scale_type) is not (mechanism["mixed_input"]["scale_type"] is not None):
            raise ContractError("mixed-input scale witness differs from declared tuple")
        if ("void" not in zero_type) is not (mechanism["mixed_input"]["zero_type"] is not None):
            raise ContractError("mixed-input zero witness differs from declared tuple")
        if mechanism["mixed_input"]["scale_type"] is not None and not cpp_type_equivalent(
            scale_type, mechanism["mixed_input"]["scale_type"]
        ):
            raise ContractError("mixed-input scale type differs from declared tuple")
        if mechanism["mixed_input"]["zero_type"] is not None and not cpp_type_equivalent(
            zero_type, mechanism["mixed_input"]["zero_type"]
        ):
            raise ContractError("mixed-input zero type differs from declared tuple")
        mixed_declared = mechanism["mixed_input"]
        transformed_axis = mixed_declared["transformed_operand"]
        other_axis = "b" if transformed_axis == "a" else "a"
        builder_transformed = resolved_types[f"builder_element_{transformed_axis}"]
        builder_other = resolved_types[f"builder_element_{other_axis}"]
        builder_tuple = split_cpp_template(normalize_cpp_type(builder_transformed))
        expected_tuple_types = [mixed_declared["narrow_type"]]
        if mixed_declared["scale_type"] is not None:
            expected_tuple_types.append(mixed_declared["scale_type"])
        if mixed_declared["zero_type"] is not None:
            expected_tuple_types.append(mixed_declared["zero_type"])
        if (
            builder_tuple is None
            or builder_tuple[0] != "cute::tuple"
            or len(builder_tuple[1]) != mixed_declared["tuple_arity"]
            or resolved_values.get(f"builder_tuple_arity_{transformed_axis}")
            != mixed_declared["tuple_arity"]
            or not all(
                cpp_type_equivalent(actual, declared)
                for actual, declared in zip(
                    builder_tuple[1], expected_tuple_types, strict=True
                )
            )
        ):
            raise ContractError("mixed-input transformed Builder tuple differs from the declaration")
        if resolved_values.get(f"builder_tuple_arity_{other_axis}") != 0:
            raise ContractError("mixed-input wide Builder operand must not be a tuple")
        if not cpp_type_equivalent(builder_other, mixed_declared["wide_type"]):
            raise ContractError("mixed-input untransformed Builder operand differs from the wide type")
        logical_narrow_axis = other_axis if mixed_declared["operands_swapped"] else transformed_axis
        logical_wide_axis = transformed_axis if mixed_declared["operands_swapped"] else other_axis
        if not cpp_type_equivalent(
            resolved_types[f"config_element_{logical_narrow_axis}"],
            mixed_declared["narrow_type"],
        ):
            raise ContractError("mixed-input logical narrow operand differs from the declared swap relation")
        if not cpp_type_equivalent(
            resolved_types[f"config_element_{logical_wide_axis}"],
            mixed_declared["wide_type"],
        ):
            raise ContractError("mixed-input logical wide operand differs from the declared swap relation")
    elif mixed_optional_keys & set(optional_types):
        raise ContractError("type witness contains undeclared mixed-input roles")
    if mechanism["fast_fp32"]["enabled"]:
        fast_expected = {
            "num_compute_matrices": "num_compute_matrices",
            "num_bands_to_compute": "num_bands",
            "fast_scaling_factor": "scaling_factor",
            "acc_promotion_interval": "acc_promotion_interval",
        }
        for witness_key, declared_key in fast_expected.items():
            if resolved_values.get(witness_key) != mechanism["fast_fp32"][declared_key]:
                raise ContractError(f"FastFP32 {witness_key} differs from declared mechanism")
    elif any(
        resolved_values.get(key) != 0
        for key in (
            "num_compute_matrices",
            "num_bands_to_compute",
            "fast_scaling_factor",
            "acc_promotion_interval",
        )
    ):
        raise ContractError("non-FastFP32 type witness contains emulation algorithm details")
    unexpected_optional = set(optional_types) - allowed_optional_keys
    if unexpected_optional:
        raise ContractError(f"type witness contains undeclared optional mechanisms: {unexpected_optional}")

    static_contract = instance["static_contract"]
    if not require_archive:
        tracked_ptx = role_paths["PTX_TARGET_FUNCTION"].read_text(encoding="utf-8")
        ptx_entries = parse_ptx_entries(tracked_ptx)
        if len(ptx_entries) != 1:
            raise ContractError("tracked PTX target excerpt must contain exactly one entry")
        ptx_function = ptx_entries[0]
        sass_value = load_strict_json(role_paths["SASS_TARGET_FUNCTION"])
        sass_symbol = sass_value.get("function-name")
        sass_instructions = sass_value.get("sass-instructions")
        if not isinstance(sass_symbol, str) or not isinstance(sass_instructions, list):
            raise ContractError("tracked SASS target excerpt has an invalid function object")
        sass_opcodes: list[str] = []
        for index, instruction in enumerate(sass_instructions):
            if not isinstance(instruction, dict) or not isinstance(instruction.get("opcode"), str):
                raise ContractError(f"tracked SASS instruction {index} lacks opcode")
            sass_opcodes.append(instruction["opcode"].upper())
        sass_function = FunctionBlock(
            sass_symbol,
            json.dumps(sass_value, separators=(",", ":")),
            tuple(sass_opcodes),
        )
        if ptx_function.symbol != sass_function.symbol:
            raise ContractError("tracked PTX/SASS excerpts have different symbols")
        ptx_contract = evaluate_contract(
            ptx_function.opcodes,
            [item["regex"] for item in static_contract["ptx"]["required"]],
            [item["regex"] for item in static_contract["ptx"]["forbidden"]],
            "ptx",
        )
        sass_contract = evaluate_contract(
            sass_function.opcodes,
            [item["regex"] for item in static_contract["sass"]["required"]],
            [item["regex"] for item in static_contract["sass"]["forbidden"]],
            "sass",
        )
        cta_contract = evaluate_cta_group(
            ptx_function.opcodes,
            sass_function.opcodes,
            instance["declared_config"]["cta_group"],
        )
        if not ptx_contract.passed or not sass_contract.passed or not cta_contract.passed:
            raise ContractError("tracked function excerpts do not satisfy the static contract")
        expected_ptx_results = _expected_pattern_results(
            static_contract["ptx"]["required"], ptx_contract.required_matches, True
        ) + _expected_pattern_results(
            static_contract["ptx"]["forbidden"], ptx_contract.forbidden_matches, False
        )
        expected_sass_results = _expected_pattern_results(
            static_contract["sass"]["required"], sass_contract.required_matches, True
        ) + _expected_pattern_results(
            static_contract["sass"]["forbidden"], sass_contract.forbidden_matches, False
        )
        for artifact_kind, contracts, expected_results in (
            ("PTX", static_contract["ptx"]["required"] + static_contract["ptx"]["forbidden"], expected_ptx_results),
            ("SASS", static_contract["sass"]["required"] + static_contract["sass"]["forbidden"], expected_sass_results),
        ):
            for contract_item, result_item in zip(contracts, expected_results, strict=True):
                maximum = contract_item["max_count"]
                count = result_item["match_count"]
                if count < contract_item["min_count"] or (
                    maximum is not None and count > maximum
                ):
                    raise ContractError(
                        f"tracked {artifact_kind} contract {contract_item['id']} count {count} "
                        f"outside [{contract_item['min_count']}, {maximum}]"
                    )
        report = load_strict_json(role_paths["CONTRACT_CHECK_REPORT"])
        binding = evidence["function_binding"]
        expected_binding = {
            "target_cpp_entity": static_contract["target_entity"],
            "selection_policy": static_contract["function_selector"],
            "symbol": ptx_function.symbol,
            "ptx_entry_count": 1,
            "elf_sto_entry_count": 1,
            "nvdisasm_symbol_match_count": 1,
            "nvdisasm_total_function_count": report.get("nvdisasm_total_function_count"),
            "ptx_function_artifact_id": role_items["PTX_TARGET_FUNCTION"]["artifact_id"],
            "sass_function_artifact_id": role_items["SASS_TARGET_FUNCTION"]["artifact_id"],
            "same_symbol": True,
        }
        if binding != expected_binding:
            raise ContractError("result function binding differs from tracked excerpts")
        validate_device_kernel_symbol(binding["symbol"], resolved_types["gemm_kernel"])
        expected_report = {
            "schema_version": 1,
            "instance_id": instance["instance_id"],
            "symbol": ptx_function.symbol,
            "ptx_target": instance["target"]["binary_arch"],
            "sass_arch": instance["target"]["binary_arch"],
            "nvdisasm_total_function_count": binding["nvdisasm_total_function_count"],
            "ptx_contract_results": expected_ptx_results,
            "sass_contract_results": expected_sass_results,
            "cta_group": {
                "declared": cta_contract.declared_cta_group,
                "ptx_mma_opcodes": list(cta_contract.ptx_mma_opcodes),
                "sass_mma_opcodes": list(cta_contract.sass_mma_opcodes),
                "errors": list(cta_contract.errors),
            },
        }
        if report != expected_report:
            raise ContractError("tracked contract report differs from function excerpts")
        if evidence["ptx_contract_results"] != expected_ptx_results:
            raise ContractError("result PTX contract differs from tracked excerpts")
        if evidence["sass_contract_results"] != expected_sass_results:
            raise ContractError("result SASS contract differs from tracked excerpts")
        if evidence["cta_group_contract_pass"] is not cta_contract.passed:
            raise ContractError("result CTA-group contract differs from tracked excerpts")
        return

    if role_paths["FATBIN"].read_bytes()[:4] != bytes.fromhex("50ed55ba"):
        raise ContractError("FATBIN artifact has an invalid magic value")
    if role_paths["CUBIN"].read_bytes()[:4] != b"\x7fELF":
        raise ContractError("CUBIN artifact is not an ELF file")

    ptx_text = role_paths["PTX_FULL"].read_text(encoding="utf-8")
    entries = parse_ptx_entries(ptx_text)
    if len(entries) != 1:
        raise ContractError("STATIC_PASS PTX must contain exactly one .entry")
    symbol = entries[0].symbol
    try:
        attribution = attribute_codegen(
            ptx_text=ptx_text,
            nvdisasm_json_text=role_paths["NVDISASM_JSON_FULL"].read_text(encoding="utf-8"),
            elf_symbols_text=role_paths["ELF_SYMBOL_TABLE"].read_text(encoding="utf-8"),
            code_object_metadata_text=role_paths["CODE_OBJECT_METADATA"].read_text(
                encoding="utf-8"
            ),
            expected_symbol=symbol,
            declared_cta_group=instance["declared_config"]["cta_group"],
            expected_ptx_target=instance["target"]["binary_arch"],
            expected_sass_arch=instance["target"]["binary_arch"],
            required_ptx=[item["regex"] for item in static_contract["ptx"]["required"]],
            forbidden_ptx=[item["regex"] for item in static_contract["ptx"]["forbidden"]],
            required_sass=[item["regex"] for item in static_contract["sass"]["required"]],
            forbidden_sass=[item["regex"] for item in static_contract["sass"]["forbidden"]],
        )
    except AttributionError as error:
        raise ContractError(f"cannot recompute function attribution: {error}") from error
    if not attribution.passed:
        raise ContractError("recomputed function-local contract does not pass")
    validate_device_kernel_symbol(symbol, resolved_types["gemm_kernel"])
    expected_binding = {
        "target_cpp_entity": static_contract["target_entity"],
        "selection_policy": static_contract["function_selector"],
        "symbol": symbol,
        "ptx_entry_count": 1,
        "elf_sto_entry_count": 1,
        "nvdisasm_symbol_match_count": 1,
        "nvdisasm_total_function_count": attribution.sass_total_function_count,
        "ptx_function_artifact_id": role_items["PTX_TARGET_FUNCTION"]["artifact_id"],
        "sass_function_artifact_id": role_items["SASS_TARGET_FUNCTION"]["artifact_id"],
        "same_symbol": True,
    }
    if evidence["function_binding"] != expected_binding:
        raise ContractError("result function_binding differs from recomputed attribution")
    expected_ptx_results = _expected_pattern_results(
        static_contract["ptx"]["required"], attribution.ptx_contract.required_matches, True
    ) + _expected_pattern_results(
        static_contract["ptx"]["forbidden"], attribution.ptx_contract.forbidden_matches, False
    )
    expected_sass_results = _expected_pattern_results(
        static_contract["sass"]["required"], attribution.sass_contract.required_matches, True
    ) + _expected_pattern_results(
        static_contract["sass"]["forbidden"], attribution.sass_contract.forbidden_matches, False
    )
    for artifact_kind, contracts, expected_results in (
        ("PTX", static_contract["ptx"]["required"] + static_contract["ptx"]["forbidden"], expected_ptx_results),
        ("SASS", static_contract["sass"]["required"] + static_contract["sass"]["forbidden"], expected_sass_results),
    ):
        for contract_item, result_item in zip(contracts, expected_results, strict=True):
            count = result_item["match_count"]
            maximum = contract_item["max_count"]
            if count < contract_item["min_count"] or (
                maximum is not None and count > maximum
            ):
                raise ContractError(
                    f"recomputed {artifact_kind} contract {contract_item['id']} count {count} "
                    f"outside [{contract_item['min_count']}, {maximum}]"
                )
    if evidence["ptx_contract_results"] != expected_ptx_results:
        raise ContractError("result PTX contract results differ from recomputation")
    if evidence["sass_contract_results"] != expected_sass_results:
        raise ContractError("result SASS contract results differ from recomputation")
    if evidence["cta_group_contract_pass"] is not attribution.cta_group_contract.passed:
        raise ContractError("result CTA-group contract differs from recomputation")
    if role_paths["PTX_TARGET_FUNCTION"].read_text(encoding="utf-8") != (
        attribution.ptx_function.text + "\n"
    ):
        raise ContractError("tracked PTX target excerpt differs from recomputation")
    if role_paths["SASS_TARGET_FUNCTION"].read_text(encoding="utf-8") != (
        attribution.sass_function.text + "\n"
    ):
        raise ContractError("tracked SASS target excerpt differs from recomputation")
    expected_report = {
        "schema_version": 1,
        "instance_id": instance["instance_id"],
        "symbol": symbol,
        "ptx_target": attribution.ptx_target,
        "sass_arch": attribution.sass_arch,
        "nvdisasm_total_function_count": attribution.sass_total_function_count,
        "ptx_contract_results": expected_ptx_results,
        "sass_contract_results": expected_sass_results,
        "cta_group": {
            "declared": attribution.cta_group_contract.declared_cta_group,
            "ptx_mma_opcodes": list(attribution.cta_group_contract.ptx_mma_opcodes),
            "sass_mma_opcodes": list(attribution.cta_group_contract.sass_mma_opcodes),
            "errors": list(attribution.cta_group_contract.errors),
        },
    }
    if load_strict_json(role_paths["CONTRACT_CHECK_REPORT"]) != expected_report:
        raise ContractError("tracked contract report differs from recomputation")


def validate_result(root: Path, path: Path, *, require_archive: bool = True) -> dict[str, Any]:
    result = load_strict_json(path)
    validate_with_schema(result, root / "tests/codegen/schemas/result.schema.json", path.name)
    if result["contract_sha256"] != contract_sha256(root):
        raise ContractError("result contract SHA-256 mismatch")
    if result["result_id"] != result["attempt_id"] or path.stem != result["result_id"]:
        raise ContractError("result_id, attempt_id, and result filename must be identical")
    contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
    if result["objective_sha256"] != contract["objective_sha256"] or result["freshness_epoch"] != contract[
        "freshness_epoch"
    ]:
        raise ContractError("result objective/freshness mismatch")
    if result["status"] not in contract["implemented_terminal_statuses"]:
        raise ContractError(
            f"result status {result['status']} is not implemented by the current semantic validator"
        )
    layers = result["layer_outcomes"]
    if [item["layer"] for item in layers] != contract["layer_order"]:
        raise ContractError("result layer order mismatch")
    states = [item["state"] for item in layers]
    if result["status"] == "STATIC_PASS":
        if any(state != "RESOLVED" for state in states):
            raise ContractError("STATIC_PASS requires every layer RESOLVED")
        expected_witnesses = contract["layer_witness_artifact_ids"]
        for item in layers:
            if item["witness_artifact_id"] != expected_witnesses[item["layer"]]:
                raise ContractError(
                    f"layer {item['layer']} uses the wrong witness artifact"
                )
    elif result["status"] in {"EXPECTED_STATIC_REJECT", "UNSUPPORTED_SM110A"}:
        if states.count("REJECTED") != 1:
            raise ContractError("reject terminal result requires exactly one REJECTED layer")
        rejected = states.index("REJECTED")
        if any(state != "RESOLVED" for state in states[:rejected]) or any(
            state != "NOT_REACHED" for state in states[rejected + 1 :]
        ):
            raise ContractError("reject result violates stop-at-first-failure ordering")
    instance_ref = result["instance_ref"]
    instance_path = safe_path(root, instance_ref["path"], "instance_ref.path")
    if sha256_file(instance_path) != instance_ref["sha256"]:
        raise ContractError("result instance_ref SHA-256 mismatch")
    instance = validate_instance(root, instance_path)
    if instance.get("instance_id") != instance_ref["id"] or instance.get("subject") != result["subject"]:
        raise ContractError("result subject does not match the referenced instance")
    if instance["hypothesis"]["expected_outcome"] != result["status"]:
        raise ContractError("result status differs from the frozen instance hypothesis")
    fingerprint_ref = result["fingerprint_ref"]
    fingerprint_path = safe_path(root, fingerprint_ref["path"], "fingerprint_ref.path")
    if sha256_file(fingerprint_path) != fingerprint_ref["sha256"]:
        raise ContractError("result fingerprint_ref SHA-256 mismatch")
    fingerprint = validate_fingerprint(root, fingerprint_path)
    if fingerprint["fingerprint_id"] != fingerprint_ref["id"]:
        raise ContractError("result fingerprint_ref ID mismatch")
    if fingerprint_path.stem != fingerprint_ref["id"]:
        raise ContractError("fingerprint filename differs from fingerprint ID")
    if fingerprint["instance_ref"] != instance_ref:
        raise ContractError("fingerprint and result reference different instances")
    manifest_ref = result["artifact_manifest_ref"]
    manifest_path = safe_path(root, manifest_ref["path"], "artifact_manifest_ref.path")
    if sha256_file(manifest_path) != manifest_ref["sha256"]:
        raise ContractError("result artifact manifest SHA-256 mismatch")
    artifacts = validate_artifact_manifest(root, manifest_path, require_archive=require_archive)
    expected_manifest_id = f"am-{result['attempt_id']}"
    if (
        manifest_ref["id"] != expected_manifest_id
        or artifacts["artifact_manifest_id"] != expected_manifest_id
        or manifest_path.stem != expected_manifest_id
    ):
        raise ContractError("artifact manifest ID/path does not match the attempt")
    if artifacts["attempt_id"] != result["attempt_id"]:
        raise ContractError("artifact manifest attempt_id mismatch")
    if artifacts["instance_ref"] != instance_ref or artifacts["fingerprint_ref"] != fingerprint_ref:
        raise ContractError("artifact manifest refs differ from result refs")
    artifact_ids = {item["artifact_id"] for item in artifacts["items"]}
    journal_ref = result["journal_ref"]
    journal_path = safe_path(root, journal_ref["path"], "journal_ref.path")
    if journal_path.stem != result["attempt_id"]:
        raise ContractError("journal filename does not match the attempt")
    if sha256_file(journal_path) != journal_ref["sha256"]:
        raise ContractError("result journal SHA-256 mismatch")
    last_seq, last_hash, journal_events = validate_journal(
        root,
        journal_path,
        result["attempt_id"],
        fingerprint["fingerprint_id"],
        instance_ref["id"],
    )
    if last_seq != journal_ref["last_seq"] or last_hash != journal_ref["last_event_sha256"]:
        raise ContractError("result journal terminal reference mismatch")
    validate_journal_semantics(root, journal_events, fingerprint, artifacts, manifest_path)
    if result["status"] == "STATIC_PASS":
        evidence = result["evidence"]
        referenced = {
            evidence["type_witness_artifact_id"],
            evidence["function_binding"]["ptx_function_artifact_id"],
            evidence["function_binding"]["sass_function_artifact_id"],
        }
        if not referenced <= artifact_ids:
            raise ContractError("STATIC_PASS evidence references unknown artifacts")
        layer_witnesses = {
            item["witness_artifact_id"]
            for item in layers
            if item["witness_artifact_id"] is not None
        }
        if not layer_witnesses <= artifact_ids:
            raise ContractError("layer outcomes reference unknown artifacts")
        for results_name in ("ptx_contract_results", "sass_contract_results"):
            for item in evidence[results_name]:
                if item["required"] and item["match_count"] < 1:
                    raise ContractError(f"{results_name}: required opcode did not match")
                if not item["required"] and item["match_count"] != 0:
                    raise ContractError(f"{results_name}: forbidden opcode matched")
        validate_static_pass_evidence(
            root, result, instance, artifacts, require_archive=require_archive
        )
    return result


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
    os.replace(temporary, path)

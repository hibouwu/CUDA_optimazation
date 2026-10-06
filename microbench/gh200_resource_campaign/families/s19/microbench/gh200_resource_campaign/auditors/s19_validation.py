"""S19 short ABI1 metadata and independent full-value replay.

The ABI exports are pure: they neither read files nor launch work. Passing the
metadata ABI is separate from replaying every artifact value. No SASS stream is
accepted until its normalized identity has an independent target-code review.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import struct

from common.suite_io import require, relative
from auditors import s19_reference as reference

ADAPTER_ID = "s19_validation_v1"
ADAPTER_ABI_VERSION = 1
REFERENCE_MODEL = "s19_integer_gemm_v1"
REFERENCE_PATH = "microbench/gh200_resource_campaign/common/s19_reference.hpp"
# Frozen from the actual header; updated only together with its reviewed bundle.
REFERENCE_SHA256 = "e2e53c1bfd6e786a09074461f0fb452da7b28747fffcd9db42f482ba7c458b1e"
PROFILE_HASHES = {'periodic_1': '0a906eda1109f2cb0bc3b391440939cce6ac888ef5f125715254b844917694fd', 'periodic_3': 'caabfc37ef1ccfa43c5a54483f2dae870d1064e2e031f9b28353ed88f42bde44', 'tagged_1': '5de2462b289215d7504d03175ff5effec4846bd3bc7b819a12d14a85c2547a98', 'tagged_3': 'f096ddb5251cb8c25eb65773c36cac4e297304122a34718fb154264a7e65c13a'}
MODES = ("compute", "transport", "serial", "overlap", "output")
STAGES = (1, 2, 4)
K_TILES = (1, 2, 4, 8, 16, 32, 64)
FIELDS = {
    "schema_version", "validation_schema_version", "type", "case_id",
    "profile_id", "seed", "scope", "threads", "blocks", "errors",
    "target_launches", "checks", "resource_identity", "performance_eligible",
    "warmup_executed", "pilot_executed",
}
CHECK_FIELDS = {
    "launch_index", "reference_model", "reference_sha256", "comparison",
    "tolerance_id", "checked_elements", "expected_elements", "errors",
    "completed", "verified_CTA_ids", "output_artifacts",
}
RESOURCE_FIELDS = {
    "kernel_symbol", "registers_per_thread", "static_smem_bytes",
    "dynamic_smem_bytes", "local_size_bytes", "occupancy_limit_ctas_per_sm",
    "extensions",
}
ARTIFACT_FIELDS = {"path", "sha256", "dtype", "shape", "evidence_kind"}
PAYLOAD_NAMES = {
    "input.u32le": "input_bits", "output.u32le": "output_bits",
    "digest.u32le": "digest", "slots.u32le": "shared_bits",
    "trace_input.u32le": "input_trace_bits", "trace_c.u32le": "c_trace_bits",
}


def relative_path(value):
    require(isinstance(value, str) and value, "nonempty relative path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in value
            and path.as_posix() == value and value != ".", "relative canonical path")
    return value


def case_identity(case):
    require(isinstance(case, dict) and isinstance(case.get("parameters"), dict),
            "S19 case and parameters")
    parameters = case["parameters"]
    require(set(parameters) == {"mode", "stages", "k_tiles", "tile", "dynamic_smem_bytes", "input_profile", "holdout"}, "finite S19 parameter fields")
    mode, stages, tiles = (parameters[k] for k in ("mode", "stages", "k_tiles"))
    require(mode in MODES and type(stages) is int and stages in STAGES
            and type(tiles) is int and tiles in K_TILES, "finite S19 coordinate")
    require(parameters["tile"] == [32, 32, 32]
            and isinstance(parameters["tile"], list)
            and all(type(value) is int for value in parameters["tile"])
            and type(parameters["dynamic_smem_bytes"]) is int
            and parameters["dynamic_smem_bytes"] == 32768
            and parameters["input_profile"] == "periodic"
            and type(parameters["holdout"]) is bool and parameters["holdout"] == (tiles == 64),
            "fixed tile, common resource, base input and holdout roles")
    require(case.get("id") == f"{mode}_s{stages}_k{tiles}", "S19 case identity")
    require(case.get("scope") == "one_cta" and type(case.get("threads")) is int
            and case["threads"] == 128, "S19 one CTA with 128 threads")
    require(case.get("launch") == {"kind": "one_cta"}, "S19 launch contract")
    require(case.get("capabilities") == {"cc": "9.0", "device_name_contains": "GH200"},
            "S19 target capabilities")
    require(type(case.get("iterations")) is int and case["iterations"] == 1
            and case.get("work_model") == "s19_controlled_sequence_v1"
            and case.get("work_unit") == "operation" and case.get("coverage_policy") == "one_sm"
            and case.get("exportable") is False, "short candidate has no formal export qualification")
    return parameters


def validate_device(device):
    require(isinstance(device, dict), "device object")
    require(type(device.get("schema_version")) is int and device["schema_version"] == 2
            and device.get("type") == "device", "device schema")
    require(device.get("cc") == "9.0" and isinstance(device.get("name"), str)
            and "GH200" in device["name"], "GH200 SM90 device")
    require(isinstance(device.get("uuid"), str) and re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", device["uuid"]),
        "complete GPU UUID")
    for field in ("sms", "driver_version", "registers_per_sm", "smem_per_sm_bytes",
                  "smem_per_cta_optin_bytes"):
        require(type(device.get(field)) is int and device[field] > 0,
                "positive device capacity: " + field)
    require(device["sms"] <= 512 and type(device.get("runtime_version")) is int
            and device["runtime_version"] == 12090, "bounded GH200 and CUDA 12.9")
    return device


def profile_check(profile):
    require(isinstance(profile, dict), "S19 profile object")
    name = profile.get("id")
    require(isinstance(name, str) and name in PROFILE_HASHES, "S19 frozen profile required")
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    require(digest == PROFILE_HASHES[name], "frozen executable S19 profile")
    match = re.fullmatch(r"(periodic|tagged)_(1|3)", name)
    require(match is not None, "finite short input and length")
    inputs, iterations = match[1], int(match[2])
    require(profile.get("target_iterations") == [iterations]
            and profile.get("input_profiles") == [inputs], "single short coordinate")
    require(type(profile.get("maximum_target_launches")) is int
            and profile["maximum_target_launches"] == 1
            and type(profile.get("maximum_explicit_auxiliary_launches")) is int
            and profile["maximum_explicit_auxiliary_launches"] == 0, "one target, no auxiliary launches")
    require(profile.get("reference_identity") == {
        "model": REFERENCE_MODEL, "path": REFERENCE_PATH, "sha256": REFERENCE_SHA256},
        "actual frozen reference header identity")
    require(profile.get("output_evidence_kind") == "full_values"
            and profile.get("resource_extensions", {}) == {}, "full output and resource policy")
    return inputs, iterations


def coordinate(case, profile, seed):
    parameters = case_identity(case)
    inputs, iterations = profile_check(profile)
    require(type(seed) is int and 0 <= seed <= 0xffffffff, "uint32 seed")
    require(parameters["mode"] != "compute" or inputs == "periodic",
            "compute-only cannot consume tagged tiles")
    return parameters, inputs, iterations


def artifact_layouts(case, profile, seed):
    parameters, _, iterations = coordinate(case, profile, seed)
    tiles = parameters["k_tiles"]
    return {
        "input.u32le": [tiles * 2048], "output.u32le": [1024],
        "digest.u32le": [128], "slots.u32le": [8192],
        "trace_input.u32le": [iterations * tiles * 2048],
        "trace_c.u32le": [iterations * tiles * 1024],
        "guards.u32le": [96], "stamp.u32le": [10],
    }


def expected_elements(case, profile, seed):
    return sum(shape[0] for shape in artifact_layouts(case, profile, seed).values())


def validate_resources(resource, device):
    require(isinstance(resource, dict) and set(resource) == RESOURCE_FIELDS,
            "complete resource identity")
    require(resource["kernel_symbol"] == "s19_pipeline" and resource["extensions"] == {},
            "shared S19 kernel and no undeclared extensions")
    for field in ("registers_per_thread", "static_smem_bytes", "dynamic_smem_bytes",
                  "local_size_bytes", "occupancy_limit_ctas_per_sm"):
        require(type(resource[field]) is int and resource[field] >= 0,
                "nonnegative resource integer: " + field)
    registers = resource["registers_per_thread"]
    occupancy = resource["occupancy_limit_ctas_per_sm"]
    require(1 <= registers <= 255 and 1 <= occupancy <= 32
            and resource["static_smem_bytes"] == resource["local_size_bytes"] == 0
            and resource["dynamic_smem_bytes"] == 32768, "fixed nonspilling S19 resources")
    require(32768 <= device["smem_per_cta_optin_bytes"]
            and occupancy * 32768 <= device["smem_per_sm_bytes"]
            and occupancy * 128 * registers <= device["registers_per_sm"]
            and occupancy * 128 <= 2048, "necessary residency capacity bounds")


def validation_argv(binary_relative, case, profile, seed):
    coordinate(case, profile, seed)
    return [relative_path(binary_relative), "validate-only", case["id"], profile["id"], str(seed)]


def validate_validation(device, row, case, profile, seed):
    _, inputs, iterations = coordinate(case, profile, seed)
    validate_device(device)
    require(isinstance(row, dict) and set(row) == FIELDS, "complete ABI1 validation row")
    for field, expected in (("schema_version", 2), ("validation_schema_version", 1),
                            ("seed", seed), ("threads", 128), ("blocks", 1), ("errors", 0)):
        require(type(row[field]) is int and row[field] == expected, "fixed integer " + field)
    require(row["type"] == "validation" and row["case_id"] == case["id"]
            and row["profile_id"] == profile["id"] and row["scope"] == "one_cta", "S19 identity")
    require(all(row[field] is False for field in ("performance_eligible", "warmup_executed", "pilot_executed")),
            "short validation cannot grant performance qualification")
    validate_resources(row["resource_identity"], device)
    launches, checks = row["target_launches"], row["checks"]
    require(isinstance(launches, list) and isinstance(checks, list)
            and len(launches) == len(checks) == 1, "one target/check pair")
    launch = launches[0]
    require(isinstance(launch, dict) and all(type(launch.get(field)) is int for field in
            ("launch_index", "iterations", "threads", "blocks"))
            and launch == {"launch_index": 0, "iterations": iterations, "input_profile": inputs,
                           "threads": 128, "blocks": 1}, "fixed target launch")
    check = checks[0]
    require(isinstance(check, dict) and set(check) == CHECK_FIELDS, "complete check fields")
    require(type(check["launch_index"]) is int and check["launch_index"] == 0
            and check["reference_model"] == REFERENCE_MODEL and check["reference_sha256"] == REFERENCE_SHA256
            and check["comparison"] == "exact" and check["tolerance_id"] is None, "exact frozen reference")
    require(type(check["errors"]) is int and check["errors"] == 0 and check["completed"] is True,
            "numerical/completion failure")
    count = expected_elements(case, profile, seed)
    require(type(check["checked_elements"]) is int and type(check["expected_elements"]) is int
            and check["checked_elements"] == check["expected_elements"] == count, "all output and lifecycle words")
    require(isinstance(check["verified_CTA_ids"], list) and len(check["verified_CTA_ids"]) == 1
            and type(check["verified_CTA_ids"][0]) is int and check["verified_CTA_ids"] == [0], "complete CTA identity")
    layouts = artifact_layouts(case, profile, seed)
    artifacts = check["output_artifacts"]
    require(isinstance(artifacts, list) and len(artifacts) == len(layouts), "complete artifact set")
    seen = set()
    for item in artifacts:
        require(isinstance(item, dict) and set(item) == ARTIFACT_FIELDS, "fixed artifact fields")
        name = relative_path(item["path"])
        require(name in layouts and name not in seen, "unique required artifact role")
        seen.add(name)
        require(item["dtype"] == "uint32" and item["evidence_kind"] == "full_values"
                and isinstance(item["shape"], list) and all(type(n) is int for n in item["shape"])
                and item["shape"] == layouts[name], "lossless flat uint32 layout")
        require(isinstance(item["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
                "complete SHA256 identity")
    return {"status": "pass", "case_id": case["id"], "profile_id": profile["id"],
            "target_launches": 1, "checked_elements": count,
            "output_evidence_kind": "full_values", "performance_eligible": False}


def audit_artifact_values(case, profile, seed, arrays, device):
    """Recompute every payload and guard word; inspect the recorded clock pair."""
    parameters, inputs, iterations = coordinate(case, profile, seed)
    validate_device(device)
    layouts = artifact_layouts(case, profile, seed)
    require(isinstance(arrays, dict) and set(arrays) == set(layouts), "exact full-value set")
    for name, shape in layouts.items():
        values = arrays[name]
        require(isinstance(values, (list, tuple)) and len(values) == shape[0], "full array length: " + name)
        require(all(type(value) is int and 0 <= value <= 0xffffffff for value in values),
                "uint32 values: " + name)
    reference_case = reference.Case(parameters["mode"], parameters["stages"], parameters["k_tiles"],
                                    iterations, seed, inputs)
    payload = {field: list(arrays[name]) for name, field in PAYLOAD_NAMES.items()}
    compared = reference.verify_payload(reference_case, payload)
    guards = ([0xd1900000 + i for i in range(8)] + [0xd1910000 + i for i in range(8)]) * 6
    require(list(arrays["guards.u32le"]) == guards, "every input/output/trace guard word")
    stamp = arrays["stamp.u32le"]
    times = [stamp[i] | (stamp[i + 1] << 32) for i in range(0, 8, 2)]
    require(times[0] <= times[1] and times[2] < times[3], "ordered same-CTA clocks; positive cycle interval")
    event_ms = struct.unpack("<f", struct.pack("<I", stamp[9]))[0]
    require(math.isfinite(event_ms) and event_ms > 0, "positive finite CUDA event time")
    require(times[1] - times[0] <= event_ms * 1.05e6, "CUDA event covers the internal nanosecond interval")
    # Physical SM IDs may be sparse; multiprocessor count is not an ID bound.
    require(stamp[8] != 0xffffffff, "recorded SM ID cannot be the missing-value sentinel")
    return {"status": "pass", "case_id": case["id"], "profile_id": profile["id"],
            "checked_elements": expected_elements(case, profile, seed),
            "reference_values": compared, "guard_words": 96, "stamp_words": 10,
            "GPU_execution_verified": False, "performance_eligible": False}


def audit_saved_values(directory, device, row, case, profile, seed):
    """Filesystem entry: validate metadata, rehash bytes, then replay full values.

    This check does not authenticate process receipts or assign a family gate.
    The caller must separately use the frozen public archive reader for those.
    """
    validate_validation(device, row, case, profile, seed)
    arrays = {}
    identities = {}
    for item in row["checks"][0]["output_artifacts"]:
        path = relative(directory, item["path"])
        data = path.read_bytes()
        require(len(data) == item["shape"][0] * 4
                and hashlib.sha256(data).hexdigest() == item["sha256"],
                "artifact bytes/shape/SHA256: " + item["path"])
        arrays[item["path"]] = list(struct.unpack("<" + "I" * item["shape"][0], data))
        identities[item["path"]] = item["sha256"]
    result = audit_artifact_values(case, profile, seed, arrays, device)
    result["artifacts_sha256"] = identities
    return result


def validate_prior_evidence(evidence, request, frozen_refs):
    require(isinstance(evidence, dict) and isinstance(request, dict) and isinstance(frozen_refs, dict),
            "prior evidence/request/reference dictionaries")
    case_identity(request.get("case"))
    profile_check(request.get("profile"))
    declared = evidence.get("source_artifacts_sha256", {})
    require(isinstance(declared, dict) and (not evidence or declared), "prior artifacts must be bound")
    for name, digest in declared.items():
        relative_path(name)
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                and frozen_refs.get(name) == digest, "corrupt prior artifact identity")
    return {"status": "insufficient", "case_id": request["case"]["id"],
            "missing_requirements": ["no independently approved S19 prior numerical reuse mapping"]}


def full_encoding(text):
    """Return exact instruction bytes, including both 64-bit words and padding."""
    require(isinstance(text, str), "SASS text")
    functions = re.split(r"Function\s*:\s*", text)[1:]
    selected = [part for part in functions if part.splitlines()
                and part.splitlines()[0].strip() == "s19_pipeline"]
    require(len(selected) == 1, "one exact S19 kernel symbol")
    lines = selected[0].splitlines()[1:]
    words = bytearray()
    index = 0
    while index < len(lines):
        line = lines[index]
        if "/*" not in line:
            stripped = line.strip()
            require(not stripped or stripped.startswith(".headerflags") or stripped == "..........",
                    "unknown SASS stream line")
            index += 1
            continue
        first = re.fullmatch(r"\s*/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*;\s*/\* 0x([0-9a-fA-F]{16}) \*/\s*", line)
        require(first is not None, "complete PC/text/first encoding word")
        require(int(first[1], 16) == len(words), "contiguous 16-byte instruction PCs from zero")
        require(index + 1 < len(lines), "missing second encoding word")
        second = re.fullmatch(r"\s*/\* 0x([0-9a-fA-F]{16}) \*/\s*", lines[index + 1])
        require(second is not None, "complete second encoding word")
        words.extend(int(first[3], 16).to_bytes(8, "little"))
        words.extend(int(second[1], 16).to_bytes(8, "little"))
        index += 2
    require(words, "nonempty full machine-code stream")
    return bytes(words)


def audit_sass(text, contract):
    """Bind a complete reviewed 128-bit stream; upstream gates authorize its use.

    The target review separately compares these bytes to the cubin text section
    and judges loops, synchronization, dependencies, timing and resources.
    A review_path string alone never grants execution or numerical qualification.
    """
    require(isinstance(contract, dict) and contract.get("stage") == "S19"
            and contract.get("family") == "simt_pipeline", "S19 SASS contract")
    require(isinstance(contract.get("cases"), list) and len(contract["cases"]) == 105,
            "complete S19 case set")
    require(len({case.get("id") for case in contract["cases"]}) == 105, "unique S19 coordinates")
    for case in contract["cases"]:
        case_identity(case)
    baseline = contract.get("sass_baseline")
    require(isinstance(baseline, dict) and set(baseline) == {
        "kernel_symbol", "full_encoding_sha256", "instruction_count", "encoding_bits",
        "review_path", "cubin_sha256", "binary_sha256"}, "reviewed full-encoding baseline required")
    require(baseline["kernel_symbol"] == "s19_pipeline"
            and type(baseline["encoding_bits"]) is int and baseline["encoding_bits"] == 128
            and type(baseline["instruction_count"]) is int and baseline["instruction_count"] > 0,
            "exact full-code shape")
    for field in ("full_encoding_sha256", "cubin_sha256", "binary_sha256"):
        require(isinstance(baseline[field], str) and re.fullmatch(r"[0-9a-f]{64}", baseline[field]),
                "reviewed target identity: " + field)
    relative_path(baseline["review_path"])
    code = full_encoding(text)
    digest = hashlib.sha256(code).hexdigest()
    require(len(code) == 16 * baseline["instruction_count"]
            and digest == baseline["full_encoding_sha256"], "full 128-bit machine code changed")
    return [{"case_id": case["id"], "symbol": "s19_pipeline", "full_encoding_sha256": digest,
             "instruction_count": len(code) // 16, "encoding_bits": 128,
             "baseline_review": baseline["review_path"]} for case in contract["cases"]]

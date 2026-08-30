#!/usr/bin/env python3
"""Generate the five canonical instances translated from CUTLASS generator.py."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_phase2_official_instances as shared  # noqa: E402
from codegen_v2.model import load_strict_json  # noqa: E402


SPEC_PATH = ROOT / "tests/codegen/phase3_generator_specs.json"


def load_phase3_specs() -> list[dict]:
    document = load_strict_json(SPEC_PATH)
    raw_specs = document.get("specs", [])
    if (
        document.get("schema_version") != 1
        or document.get("scope") != "PHASE3_GENERATOR_CONFIG_CANONICAL_INSTANCES"
        or document.get("cutlass_git_sha") != shared.TARGET["cutlass_git_sha"]
        or document.get("expected_count") != 5
        or len(raw_specs) != 5
    ):
        raise ValueError("invalid Phase 3 generator spec metadata")
    for index, entry in enumerate(raw_specs):
        missing = (shared.REQUIRED_SPEC_KEYS | {"factory_id"}) - set(entry)
        if missing:
            raise ValueError(
                f"Phase 3 spec {index} lacks required fields: {sorted(missing)}"
            )
        for field in ("align", "tile", "cluster"):
            values = entry[field]
            if len(values) != (4 if field == "align" else 3) or any(
                not isinstance(value, int) or value <= 0 for value in values
            ):
                raise ValueError(f"Phase 3 spec {index} has invalid {field}")
        if entry["cta"] not in {1, 2}:
            raise ValueError(f"Phase 3 spec {index} has invalid CTA group")

    items = [shared.spec(**entry) for entry in raw_specs]
    tags = [item["tag"] for item in items]
    instance_ids = [item["instance_id"] for item in items]
    if len(set(tags)) != 5 or len(set(instance_ids)) != 5:
        raise ValueError("Phase 3 specs must contain five unique Tags and instance IDs")

    inventory = load_strict_json(
        ROOT / "tests/codegen/sm110a_schedule_reference_inventory.json"
    )
    expected_tags = {
        entry["tag"]
        for entry in inventory["entries"]
        if entry["reference_class"] == "generator_config"
    }
    if set(tags) != expected_tags:
        raise ValueError(
            f"Phase 3 Tag set mismatch: missing={sorted(expected_tags - set(tags))} "
            f"extra={sorted(set(tags) - expected_tags)}"
        )
    references = {entry["tag"]: entry for entry in inventory["entries"]}
    for item in items:
        reference = references[item["tag"]]
        if (
            reference["reference_class"] != "generator_config"
            or item["factory_id"] != reference["factory_id"]
        ):
            raise ValueError(f"{item['tag']}: generator factory differs from inventory")
        if item["factory_id"] == "fast_fp32_complex":
            if (
                item["builder_a"] != "cutlass::complex<float>"
                or item["builder_b"] != "cutlass::complex<float>"
                or item["mechanism_patch"].get("complex")
                != {"enabled": True, "representation": "interleaved"}
                or item["expected_outcome"] != "UNSUPPORTED_SM110A"
                or not isinstance(item["guard_profile_id"], str)
                or item["tmem_store_count"] != 6
            ):
                raise ValueError(f"{item['tag']}: Fast complex compatibility contract drifted")
        elif item["factory_id"] == "interleaved_complex_tf32":
            expected_builder = "cute::tuple<cutlass::complex<float>, cute::identity>"
            if (
                item["builder_a"] != expected_builder
                or item["builder_b"] != expected_builder
                or item["mma_kind"] != "tf32"
                or item["mma_count"] != 8
                or item["tmem_store_count"] != 2
                or item["expected_outcome"] != "STATIC_PASS"
            ):
                raise ValueError(f"{item['tag']}: InterleavedComplexTF32 contract drifted")
        elif item["factory_id"] == "blockscaled_dense":
            block_scaled = item["mechanism_patch"].get("block_scaled", {})
            if (
                item["mma_kind"] != "mxf4"
                or item["builder_a"]
                != "cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue8m0_t>"
                or block_scaled.get("vector_size_a") != 32
                or block_scaled.get("vector_size_b") != 32
            ):
                raise ValueError(f"{item['tag']}: Mxf4 generator contract drifted")
    return items


def main() -> int:
    for item in load_phase3_specs():
        shared.write_instance(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

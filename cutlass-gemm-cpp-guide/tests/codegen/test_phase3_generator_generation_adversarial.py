#!/usr/bin/env python3
"""Positive and deliberate-breakage checks for the five generator specs."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_phase2_official_instances as shared  # noqa: E402
import generate_phase3_generator_instances as generator  # noqa: E402
from codegen_v2.model import ContractError, load_strict_json, validate_instance  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def require_rejected(name: str, expected: str, mutate) -> None:
    temporary = Path(tempfile.mkdtemp(prefix="phase3-generator-adversarial-"))
    original = generator.SPEC_PATH
    try:
        target = temporary / original.name
        shutil.copy2(original, target)
        document = load_strict_json(target)
        mutate(document)
        write_json(target, document)
        generator.SPEC_PATH = target
        try:
            generator.load_phase3_specs()
        except (ValueError, KeyError, ContractError) as error:
            if expected not in str(error):
                raise AssertionError(f"{name}: wrong rejection: {error}") from error
        else:
            raise AssertionError(f"{name}: deliberate spec corruption was accepted")
    finally:
        generator.SPEC_PATH = original
        shutil.rmtree(temporary)


def main() -> int:
    items = generator.load_phase3_specs()
    if len(items) != 5:
        raise AssertionError(f"expected five Phase 3 specs, found {len(items)}")
    for item in items:
        directory = ROOT / "tests/codegen/instances" / item["instance_id"]
        config = directory / "config.hpp"
        kernel = directory / "kernel.cu"
        witness = directory / "type_witness.cu"
        instance = directory / "instance.json"
        if config.read_text(encoding="utf-8") != shared.render_config(item):
            raise AssertionError(f"{item['instance_id']}: config.hpp drifted from spec")
        if kernel.read_text(encoding="utf-8") != shared.render_kernel(item):
            raise AssertionError(f"{item['instance_id']}: kernel.cu drifted from spec")
        if witness.read_text(encoding="utf-8") != shared.render_type_witness(item):
            raise AssertionError(f"{item['instance_id']}: type_witness.cu drifted from spec")
        if load_strict_json(instance) != shared.render_instance(
            item, config, kernel, witness
        ):
            raise AssertionError(f"{item['instance_id']}: instance.json drifted from spec")
        validate_instance(ROOT, instance)

    require_rejected(
        "missing_spec",
        "invalid Phase 3 generator spec metadata",
        lambda document: document["specs"].pop(),
    )
    require_rejected(
        "duplicate_tag",
        "five unique",
        lambda document: document["specs"][1].__setitem__(
            "tag", document["specs"][0]["tag"]
        ),
    )
    require_rejected(
        "duplicate_instance",
        "five unique",
        lambda document: document["specs"][1].__setitem__(
            "instance_id", document["specs"][0]["instance_id"]
        ),
    )
    require_rejected(
        "invented_tag",
        "Tag set mismatch",
        lambda document: document["specs"][0].__setitem__(
            "tag", "KernelInventedSm100"
        ),
    )
    require_rejected(
        "official_tag_reused",
        "Tag set mismatch",
        lambda document: document["specs"][0].__setitem__(
            "tag", "KernelTmaWarpSpecialized1SmSm100"
        ),
    )
    require_rejected(
        "missing_source_note",
        "lacks required fields",
        lambda document: document["specs"][0].pop("source_note"),
    )
    require_rejected(
        "invalid_tile",
        "invalid tile",
        lambda document: document["specs"][0].__setitem__("tile", [128, 0, 16]),
    )
    require_rejected(
        "wrong_cutlass_sha",
        "invalid Phase 3 generator spec metadata",
        lambda document: document.__setitem__("cutlass_git_sha", "0" * 40),
    )

    require_rejected(
        "wrong_factory",
        "generator factory differs",
        lambda document: document["specs"][0].__setitem__("factory_id", "invented"),
    )
    require_rejected(
        "fast_tuple_bypasses_compatibility",
        "Fast complex compatibility contract drifted",
        lambda document: document["specs"][1].__setitem__(
            "builder_a", "cute::tuple<cutlass::complex<float>, cute::identity>"
        ),
    )
    require_rejected(
        "fast_complex_disabled",
        "Fast complex compatibility contract drifted",
        lambda document: document["specs"][1]["mechanism_patch"].__setitem__(
            "complex", {"enabled": False, "representation": None}
        ),
    )
    require_rejected(
        "interleaved_mma_count",
        "InterleavedComplexTF32 contract drifted",
        lambda document: document["specs"][0].__setitem__("mma_count", 1),
    )
    require_rejected(
        "mxf4_kind",
        "Mxf4 generator contract drifted",
        lambda document: document["specs"][-1].__setitem__("mma_kind", "mxf4nvf4"),
    )

    print("PHASE3_GENERATOR_GENERATION_ADVERSARIAL_PASS mutations=13 positive=5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

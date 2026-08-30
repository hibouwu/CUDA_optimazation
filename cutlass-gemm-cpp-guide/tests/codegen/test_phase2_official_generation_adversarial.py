#!/usr/bin/env python3
"""Positive and deliberate-breakage checks for the 34 official-C++ specs."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_phase2_official_instances as generator  # noqa: E402
from codegen_v2.model import ContractError, load_strict_json, validate_instance  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def require_rejected(name: str, expected: str, mutate) -> None:
    temporary = Path(tempfile.mkdtemp(prefix="phase2-spec-adversarial-"))
    original_paths = generator.SPEC_PATHS
    try:
        paths = []
        documents = []
        for source in original_paths:
            target = temporary / source.name
            shutil.copy2(source, target)
            paths.append(target)
            documents.append(load_strict_json(target))
        mutate(documents)
        for path, document in zip(paths, documents, strict=True):
            write_json(path, document)
        generator.SPEC_PATHS = tuple(paths)
        try:
            generator.load_phase2_specs()
        except (ValueError, KeyError, ContractError) as error:
            if expected not in str(error):
                raise AssertionError(f"{name}: wrong rejection: {error}") from error
        else:
            raise AssertionError(f"{name}: deliberate spec corruption was accepted")
    finally:
        generator.SPEC_PATHS = original_paths
        shutil.rmtree(temporary)


def main() -> int:
    items = generator.load_phase2_specs()
    if len(items) != 34:
        raise AssertionError(f"expected 34 Phase 2 specs, found {len(items)}")

    for item in items:
        directory = ROOT / "tests/codegen/instances" / item["instance_id"]
        config = directory / "config.hpp"
        kernel = directory / "kernel.cu"
        witness = directory / "type_witness.cu"
        instance = directory / "instance.json"
        if config.read_text(encoding="utf-8") != generator.render_config(item):
            raise AssertionError(f"{item['instance_id']}: config.hpp drifted from the spec")
        if kernel.read_text(encoding="utf-8") != generator.render_kernel(item):
            raise AssertionError(f"{item['instance_id']}: kernel.cu drifted from the spec")
        if witness.read_text(encoding="utf-8") != generator.render_type_witness(item):
            raise AssertionError(f"{item['instance_id']}: type_witness.cu drifted from the spec")
        if load_strict_json(instance) != generator.render_instance(item, config, kernel, witness):
            raise AssertionError(f"{item['instance_id']}: instance.json drifted from the spec")
        validate_instance(ROOT, instance)

    require_rejected(
        "missing_spec",
        "invalid Phase 2 spec document metadata",
        lambda documents: documents[0]["specs"].pop(),
    )
    require_rejected(
        "duplicate_tag",
        "34 unique",
        lambda documents: documents[0]["specs"][1].__setitem__(
            "tag", documents[0]["specs"][0]["tag"]
        ),
    )
    require_rejected(
        "duplicate_instance_id",
        "34 unique",
        lambda documents: documents[0]["specs"][1].__setitem__(
            "instance_id", documents[0]["specs"][0]["instance_id"]
        ),
    )
    require_rejected(
        "invented_tag",
        "Tag set mismatch",
        lambda documents: documents[0]["specs"][0].__setitem__("tag", "KernelInventedSm100"),
    )
    require_rejected(
        "phase1_tag_reused",
        "Tag set mismatch",
        lambda documents: documents[0]["specs"][0].__setitem__(
            "tag", "KernelTmaWarpSpecialized1SmSm100"
        ),
    )
    require_rejected(
        "missing_required_field",
        "lacks required fields",
        lambda documents: documents[0]["specs"][0].pop("source_note"),
    )
    require_rejected(
        "invalid_shape",
        "invalid tile",
        lambda documents: documents[0]["specs"][0].__setitem__("tile", [128, 0, 64]),
    )
    require_rejected(
        "wrong_cutlass_sha",
        "invalid Phase 2 spec document metadata",
        lambda documents: documents[0].__setitem__("cutlass_git_sha", "0" * 40),
    )

    print("PHASE2_OFFICIAL_GENERATION_ADVERSARIAL_PASS mutations=8 positive=34")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Adversarial checks for current-summary selection and evidence namespaces."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codegen_v2.model import file_ref, load_strict_json  # noqa: E402
from test_codegen_v2_model_adversarial import make_fixture, write_json  # noqa: E402


def make_campaign_fixture() -> tuple[Path, dict[str, Path]]:
    def freeze_fixture_campaign(contract):
        contract["phase1_fresh_replay_instances"] = ["fixture"]
        contract["phase1_run_id"] = "fixture"
        contract["phase1_current_summary"] = (
            "evidence/codegen-sm110a-v2/summary-fixture.json"
        )

    root, paths = make_fixture(freeze_fixture_campaign)
    summary = root / "evidence/codegen-sm110a-v2/summary-fixture.json"
    write_json(
        summary,
        {
            "schema_version": 1,
            "run_id": "fixture",
            "scope": "STATIC_CODEGEN_ONLY",
            "result_refs": [file_ref(root, paths["result"], identifier="fixture.fixture.a001")],
            "history_result_refs": [],
        },
    )
    paths["summary"] = summary
    return root, paths


def run_gate(root: Path, *, require_results: bool = True) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(root / "tools/validate_codegen_v2.py"), "--root", str(root)]
    if require_results:
        command.append("--require-results")
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def require_rejected(name: str, expected: str, mutate) -> None:
    root, paths = make_campaign_fixture()
    try:
        mutate(root, paths)
        result = run_gate(root)
        if result.returncode == 0:
            raise AssertionError(f"{name}: campaign corruption was accepted")
        if expected not in result.stderr:
            raise AssertionError(
                f"{name}: wrong rejection; expected {expected!r}, stderr={result.stderr!r}"
            )
    finally:
        shutil.rmtree(root)


def main() -> int:
    root, _ = make_campaign_fixture()
    try:
        result = run_gate(root)
        if result.returncode != 0 or "phase1_results=1" not in result.stdout:
            raise AssertionError(f"clean campaign fixture failed: {result.stdout}\n{result.stderr}")
    finally:
        shutil.rmtree(root)

    require_rejected(
        "summary_result_hash",
        "result ref identity/hash mismatch",
        lambda r, p: (
            lambda value: (
                value["result_refs"][0].__setitem__("sha256", "0" * 64),
                write_json(p["summary"], value),
            )
        )(load_strict_json(p["summary"])),
    )
    require_rejected(
        "summary_filename_run_id",
        "summary filename differs from run_id",
        lambda r, p: (
            lambda value: (
                value.__setitem__("run_id", "other"),
                write_json(p["summary"], value),
            )
        )(load_strict_json(p["summary"])),
    )

    def cross_run_summary(root: Path, paths: dict[str, Path]) -> None:
        value = load_strict_json(paths["summary"])
        value["run_id"] = "other"
        replacement = paths["summary"].with_name("summary-other.json")
        write_json(replacement, value)
        paths["summary"].unlink()

    require_rejected(
        "cross_run_summary",
        "result attempt belongs to a different run_id",
        cross_run_summary,
    )

    def orphan_fingerprint(root: Path, paths: dict[str, Path]) -> None:
        orphan = root / "evidence/codegen-sm110a-v2/fingerprints/orphan.json"
        shutil.copy2(paths["fingerprint"], orphan)

    def orphan_manifest(root: Path, paths: dict[str, Path]) -> None:
        orphan = root / "evidence/codegen-sm110a-v2/artifact-manifests/orphan.json"
        shutil.copy2(paths["manifest"], orphan)

    def orphan_journal(root: Path, paths: dict[str, Path]) -> None:
        orphan = root / "evidence/codegen-sm110a-v2/journals/orphan.jsonl"
        shutil.copy2(paths["journal"], orphan)

    def orphan_excerpt(root: Path, paths: dict[str, Path]) -> None:
        orphan = root / "evidence/codegen-sm110a-v2/excerpts/orphan.attempt/unreferenced.txt"
        orphan.parent.mkdir(parents=True)
        orphan.write_text("orphan\n", encoding="utf-8")

    unsealed_root, unsealed_paths = make_campaign_fixture()
    try:
        orphan_fingerprint(unsealed_root, unsealed_paths)
        orphan_manifest(unsealed_root, unsealed_paths)
        orphan_journal(unsealed_root, unsealed_paths)
        orphan_excerpt(unsealed_root, unsealed_paths)
        unsealed_result = unsealed_paths["result"].with_name("unsealed.json")
        shutil.copy2(unsealed_paths["result"], unsealed_result)
        result = run_gate(unsealed_root)
        if result.returncode != 0 or "unsealed_results=1" not in result.stdout:
            raise AssertionError(
                f"unsealed staging files affected the committed campaign: {result.stdout}\n{result.stderr}"
            )
    finally:
        shutil.rmtree(unsealed_root)

    require_rejected(
        "missing_current_summary",
        "phase1_current_summary",
        lambda r, p: p["summary"].unlink(),
    )

    print("CODEGEN_V2_CAMPAIGN_ADVERSARIAL_PASS mutations=4 positive=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

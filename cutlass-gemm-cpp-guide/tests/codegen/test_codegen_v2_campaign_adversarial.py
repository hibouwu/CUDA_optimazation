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
    root, paths = make_fixture()
    campaign = root / "tests/codegen/run-campaigns/fixture.json"
    campaign.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        campaign,
        {
            "schema_version": 1,
            "objective_sha256": "463fa7015808acd883b28d115fa33708f66064aaceed96f728996a01ca0e4091",
            "scope": "STATIC_CODEGEN_ONLY",
            "phase_id": 99,
            "campaign_id": "fixture",
            "run_id": "fixture",
            "current_summary": "evidence/codegen-sm110a-v2/summary-fixture.json",
            "allowed_terminal_statuses": ["STATIC_PASS"],
            "ordered_instances": ["fixture"],
        },
    )
    summary = root / "evidence/codegen-sm110a-v2/summary-fixture.json"
    write_json(
        summary,
        {
            "schema_version": 1,
            "run_id": "fixture",
            "scope": "STATIC_CODEGEN_ONLY",
            "campaign_ref": file_ref(root, campaign, identifier="fixture"),
            "result_refs": [file_ref(root, paths["result"], identifier="fixture.fixture.a001")],
            "history_result_refs": [],
        },
    )
    paths["summary"] = summary
    paths["campaign"] = campaign
    return root, paths


def run_gate(root: Path, *, require_results: bool = True) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(root / "tools/validate_codegen_v2.py"), "--root", str(root)]
    if require_results:
        command += ["--require-campaign", "fixture", "--require-archive"]
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
        if result.returncode != 0 or "campaign_results=fixture:1" not in result.stdout:
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
        "campaign_ref does not bind this summary",
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

    require_rejected(
        "orphan_fingerprint",
        "fingerprint namespace differs from published results",
        orphan_fingerprint,
    )
    require_rejected(
        "orphan_manifest",
        "artifact manifest namespace differs from published results",
        orphan_manifest,
    )
    require_rejected(
        "orphan_journal",
        "journal namespace differs from published results",
        orphan_journal,
    )
    require_rejected(
        "orphan_excerpt",
        "excerpt namespace differs from published manifests",
        orphan_excerpt,
    )
    require_rejected(
        "unsealed_result",
        "result namespace contains unsealed entries",
        lambda r, p: shutil.copy2(
            p["result"], p["result"].with_name("unsealed.json")
        ),
    )

    def orphan_attempt(root: Path, paths: dict[str, Path]) -> None:
        orphan = root / "artifacts/codegen-sm110a-v2/orphan/attempts/orphan.a001"
        orphan.mkdir(parents=True)
        (orphan / "unsealed.txt").write_text("orphan\n", encoding="utf-8")

    require_rejected(
        "orphan_attempt",
        "attempt archive namespace contains orphan entries",
        orphan_attempt,
    )

    def non_directory_attempt(root: Path, paths: dict[str, Path]) -> None:
        invalid = root / "artifacts/codegen-sm110a-v2/orphan/attempts/not-a-directory"
        invalid.parent.mkdir(parents=True)
        invalid.write_text("not an attempt directory\n", encoding="utf-8")

    require_rejected(
        "non_directory_attempt",
        "attempt archive namespace contains a non-directory entry",
        non_directory_attempt,
    )

    require_rejected(
        "missing_current_summary",
        "current_summary: file is missing",
        lambda r, p: p["summary"].unlink(),
    )

    require_rejected(
        "campaign_ref_hash",
        "campaign_ref does not bind this summary",
        lambda r, p: (
            lambda value: (
                value["campaign_ref"].__setitem__("sha256", "0" * 64),
                write_json(p["summary"], value),
            )
        )(load_strict_json(p["summary"])),
    )

    def campaign_member_status(root: Path, paths: dict[str, Path]) -> None:
        campaign = load_strict_json(paths["campaign"])
        campaign["allowed_terminal_statuses"] = ["UNSUPPORTED_SM110A"]
        write_json(paths["campaign"], campaign)
        summary = load_strict_json(paths["summary"])
        summary["campaign_ref"] = file_ref(root, paths["campaign"], identifier="fixture")
        write_json(paths["summary"], summary)

    require_rejected(
        "campaign_member_status",
        "invalid terminal statuses",
        campaign_member_status,
    )

    print("CODEGEN_V2_CAMPAIGN_ADVERSARIAL_PASS mutations=13 positive=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

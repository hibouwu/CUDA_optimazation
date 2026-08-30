#!/usr/bin/env python3
"""Validate committed machine-readable deep-replay attestations and their sealed inputs."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from codegen_v2.model import (  # noqa: E402
    ContractError,
    load_strict_json,
    safe_path,
    sha256_bytes,
    sha256_file,
)


def validate_attestation(path: Path) -> None:
    value = load_strict_json(path)
    expected_fields = {
        "schema_version",
        "objective_sha256",
        "scope",
        "target",
        "result_contract_bundle_sha256",
        "command",
        "command_sha256",
        "exit_code",
        "stdout",
        "stdout_sha256",
        "replayed_results",
        "status_counts",
        "campaigns",
        "evidence_boundary",
    }
    if set(value) != expected_fields:
        raise ContractError(f"{path.name}: attestation fields differ from the frozen shape")
    if (
        value["schema_version"] != 1
        or value["objective_sha256"]
        != "463fa7015808acd883b28d115fa33708f66064aaceed96f728996a01ca0e4091"
        or value["scope"] != "STATIC_CODEGEN_ONLY"
        or value["target"] != {"virtual_arch": "compute_110a", "binary_arch": "sm_110a"}
        or value["exit_code"] != 0
        or any(value["evidence_boundary"].values())
    ):
        raise ContractError(f"{path.name}: invalid identity, exit code, or evidence boundary")
    if sha256_bytes((value["command"] + "\n").encode()) != value["command_sha256"]:
        raise ContractError(f"{path.name}: command SHA-256 mismatch")
    if sha256_bytes(value["stdout"].encode()) != value["stdout_sha256"]:
        raise ContractError(f"{path.name}: stdout SHA-256 mismatch")
    bundle_match = re.search(
        r"result_contract_bundle_sha256=([0-9a-f]{64})\n$", value["stdout"]
    )
    if bundle_match is None or bundle_match.group(1) != value["result_contract_bundle_sha256"]:
        raise ContractError(f"{path.name}: stdout does not bind the result-contract bundle")
    unsealed_match = re.search(r"\bunsealed_results=([0-9]+)\b", value["stdout"])
    if unsealed_match is None or unsealed_match.group(1) != "0":
        raise ContractError(f"{path.name}: stdout does not prove zero unsealed results")

    observed_statuses: Counter[str] = Counter()
    observed_results = 0
    command_campaigns: list[str] = []
    tokens = value["command"].split()
    for index, token in enumerate(tokens[:-1]):
        if token == "--require-campaign":
            command_campaigns.append(tokens[index + 1])
    campaign_ids = [entry["id"] for entry in value["campaigns"]]
    if command_campaigns != campaign_ids or len(campaign_ids) != len(set(campaign_ids)):
        raise ContractError(f"{path.name}: command and campaign list differ")
    for index, entry in enumerate(value["campaigns"]):
        if set(entry) != {"id", "path", "sha256", "summary_path", "summary_sha256"}:
            raise ContractError(f"{path.name}: campaigns[{index}] shape differs")
        campaign_path = safe_path(ROOT, entry["path"], f"campaigns[{index}].path")
        summary_path = safe_path(ROOT, entry["summary_path"], f"campaigns[{index}].summary")
        if sha256_file(campaign_path) != entry["sha256"] or sha256_file(summary_path) != entry[
            "summary_sha256"
        ]:
            raise ContractError(f"{path.name}: campaign/summary hash drift")
        campaign = load_strict_json(campaign_path)
        summary = load_strict_json(summary_path)
        if (
            campaign["campaign_id"] != entry["id"]
            or summary["run_id"] != campaign["run_id"]
            or len(summary["result_refs"]) != len(campaign["ordered_instances"])
        ):
            raise ContractError(f"{path.name}: campaign/summary membership differs")
        for ref in summary["result_refs"]:
            result_path = safe_path(ROOT, ref["path"], "attested result")
            if sha256_file(result_path) != ref["sha256"]:
                raise ContractError(f"{path.name}: attested result hash drift")
            result = load_strict_json(result_path)
            observed_statuses[result["status"]] += 1
            observed_results += 1
    if observed_results != value["replayed_results"] or dict(observed_statuses) != value[
        "status_counts"
    ]:
        raise ContractError(f"{path.name}: replayed result/status counts differ")


def main() -> int:
    paths = sorted(
        (ROOT / "evidence/codegen-sm110a-v2/replay-attestations").glob("*.json")
    )
    if not paths:
        raise ContractError("no replay attestations found")
    for path in paths:
        validate_attestation(path)
    print(f"REPLAY_ATTESTATION_PASS records={len(paths)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, KeyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

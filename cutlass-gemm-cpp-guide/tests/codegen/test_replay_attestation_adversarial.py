#!/usr/bin/env python3
"""Adversarial checks for the committed deep-replay attestation."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from validate_replay_attestations import ContractError, sha256_bytes, validate_attestation


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "evidence/codegen-sm110a-v2/replay-attestations/phase3-final-deep-replay.json"
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def require_rejected(name: str, expected: str, mutate) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=f"replay-attestation-{name}-"))
    try:
        candidate = temporary / SOURCE.name
        shutil.copy2(SOURCE, candidate)
        value = json.loads(candidate.read_text(encoding="utf-8"))
        mutate(value)
        write_json(candidate, value)
        try:
            validate_attestation(candidate)
        except ContractError as error:
            if expected not in str(error):
                raise AssertionError(f"{name}: wrong rejection: {error}") from error
        else:
            raise AssertionError(f"{name}: corrupted replay attestation was accepted")
    finally:
        shutil.rmtree(temporary)


def main() -> int:
    validate_attestation(SOURCE)

    def unsealed_result(value: dict) -> None:
        value["stdout"] = value["stdout"].replace(
            "unsealed_results=0", "unsealed_results=1"
        )
        value["stdout_sha256"] = sha256_bytes(value["stdout"].encode())

    require_rejected(
        "unsealed_result",
        "zero unsealed results",
        unsealed_result,
    )
    require_rejected(
        "stdout_hash",
        "stdout SHA-256 mismatch",
        lambda value: value.__setitem__("stdout_sha256", "0" * 64),
    )
    require_rejected(
        "summary_hash",
        "campaign/summary hash drift",
        lambda value: value["campaigns"][0].__setitem__(
            "summary_sha256", "0" * 64
        ),
    )
    print("REPLAY_ATTESTATION_ADVERSARIAL_PASS mutations=3 positive=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

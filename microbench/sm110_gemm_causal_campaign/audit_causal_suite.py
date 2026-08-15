#!/usr/bin/env python3
"""Audit the Thor platform interval around the tc5a causal campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "codex/sm110-all-precision-closure"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
EXPECTED_PLATFORM_DEPENDENCIES = {
    "microbench/sm110_causal_suite.sh",
    "microbench/run_sm110_causal_suite.sh",
    "microbench/sm110_gemm_causal_campaign/run_causal_campaign.py",
    "microbench/sm110_gemm_causal_campaign/audit_campaign.py",
    "microbench/sm110_gemm_causal_campaign/audit_causal_suite.py",
}


def load_campaign_auditor() -> Any:
    path = Path(__file__).with_name("audit_campaign.py")
    spec = importlib.util.spec_from_file_location(
        "sm110_causal_campaign_auditor", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load causal campaign auditor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def git_blob(commit: str, relative: str) -> bytes | None:
    path = Path(relative)
    if (
        COMMIT_RE.fullmatch(commit) is None
        or path.is_absolute()
        or not path.parts
        or ".." in path.parts
    ):
        return None
    try:
        completed = subprocess.run(
            ["git", "show", "--no-ext-diff", f"{commit}:{path.as_posix()}"],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def parse_counter_tsv(path: Path) -> dict[str, int] | None:
    counters: dict[str, int] = {}
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or fields[0] in counters:
            return None
        try:
            value = int(fields[1])
        except ValueError:
            return None
        if value < 0:
            return None
        counters[fields[0]] = value
    return counters or None


def audit_preflight(
    text: str, *, expected_branch: str, expected_commit: str,
) -> list[str]:
    errors: list[str] = []
    try:
        git_section = text.split("=== git ===", 1)[1].split(
            "=== nvpmodel ===", 1
        )[0]
    except IndexError:
        return ["preflight Git section is malformed"]
    git_lines = [line.strip() for line in git_section.splitlines() if line.strip()]
    add(
        errors,
        git_lines == [expected_branch, expected_commit],
        "preflight does not prove the clean expected checkout",
    )
    add(errors, "NV Power Mode: MAXN" in text, "preflight does not prove MAXN")
    for token in (
        "min_freq=1575000000",
        "max_freq=1575000000",
        "cur_freq=1575000000",
        "governor=performance",
    ):
        add(errors, token in text, f"preflight lacks clock token:{token}")
    return errors


def audit_suite(
    suite_dir: Path, *, expected_commit: str | None = None,
) -> dict[str, Any]:
    suite_dir = suite_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    for relative in (
        "run_contract.json",
        "preflight.txt",
        "oc_before.tsv",
        "oc_after.tsv",
        "suite_launcher.log",
    ):
        add(errors, (suite_dir / relative).is_file(), f"missing:{relative}")
    if errors:
        return {
            "pass": False,
            "suite_dir": str(suite_dir),
            "errors": errors,
            "warnings": warnings,
        }

    try:
        contract = json.loads((suite_dir / "run_contract.json").read_text())
    except json.JSONDecodeError:
        contract = {}
        errors.append("run contract is not valid JSON")
    suite_id = suite_dir.name
    commit = contract.get("expected_commit")
    add(
        errors,
        isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None,
        "run contract has no valid commit",
    )
    if expected_commit is not None:
        add(errors, commit == expected_commit, "expected commit mismatch")
    required_contract = {
        "schema_version": 1,
        "kind": "exact_tc5a_causal_pipeline_suite",
        "suite_id": suite_id,
        "causal_run_id": f"{suite_id}-causal",
        "expected_branch": EXPECTED_BRANCH,
        "expected_commit": commit,
        "ncu_required": True,
    }
    for name, value in required_contract.items():
        add(errors, contract.get(name) == value, f"run contract mismatch:{name}")

    dependencies = contract.get("platform_dependencies")
    add(errors, isinstance(dependencies, dict), "platform dependency hashes are missing")
    if isinstance(dependencies, dict):
        add(
            errors,
            set(dependencies) == EXPECTED_PLATFORM_DEPENDENCIES,
            "platform dependency path set mismatch",
        )
        if isinstance(commit, str):
            for relative in EXPECTED_PLATFORM_DEPENDENCIES:
                blob = git_blob(commit, relative)
                add(errors, blob is not None, f"platform dependency unavailable:{relative}")
                if blob is not None:
                    add(
                        errors,
                        hashlib.sha256(blob).hexdigest() == dependencies.get(relative),
                        f"platform dependency hash mismatch:{relative}",
                    )

    if isinstance(commit, str):
        errors.extend(audit_preflight(
            (suite_dir / "preflight.txt").read_text(errors="replace"),
            expected_branch=EXPECTED_BRANCH,
            expected_commit=commit,
        ))
    log_lines = (suite_dir / "suite_launcher.log").read_text(
        errors="replace"
    ).splitlines()
    add(
        errors,
        "CAUSAL_CAMPAIGN_COMPLETE" in log_lines,
        "suite launcher log has no campaign-complete marker",
    )

    before = parse_counter_tsv(suite_dir / "oc_before.tsv")
    after = parse_counter_tsv(suite_dir / "oc_after.tsv")
    add(errors, before is not None, "pre-run OC counters are malformed")
    add(errors, after is not None, "post-run OC counters are malformed")
    deltas: dict[str, int] = {}
    if before is not None and after is not None:
        add(errors, set(before) == set(after), "OC counter path set changed")
        if set(before) == set(after):
            deltas = {name: after[name] - value for name, value in before.items()}
            add(
                errors,
                all(value >= 0 for value in deltas.values()),
                "OC counters reset during the evidence interval",
            )
            warnings.extend(
                f"overcurrent_delta:{name}:{value}"
                for name, value in sorted(deltas.items())
                if value > 0
            )

    resume_preflights = sorted(suite_dir.glob("resume_preflight.*.txt"))
    resume_counters = sorted(suite_dir.glob("oc_resume.*.tsv"))
    preflight_tokens = {
        path.name.removeprefix("resume_preflight.").removesuffix(".txt")
        for path in resume_preflights
    }
    counter_tokens = {
        path.name.removeprefix("oc_resume.").removesuffix(".tsv")
        for path in resume_counters
    }
    add(
        errors,
        len(resume_preflights) == len(resume_counters),
        "resume preflight/counter snapshot counts differ",
    )
    add(errors, preflight_tokens == counter_tokens, "resume snapshot identities differ")
    if isinstance(commit, str):
        for path in resume_preflights:
            errors.extend(
                f"{path.name}:{message}"
                for message in audit_preflight(
                    path.read_text(errors="replace"),
                    expected_branch=EXPECTED_BRANCH,
                    expected_commit=commit,
                )
            )
    if before is not None and after is not None:
        for path in resume_counters:
            current = parse_counter_tsv(path)
            add(errors, current is not None, f"resume OC snapshot is malformed:{path.name}")
            if current is not None:
                add(errors, set(current) == set(before), f"resume OC counter set changed:{path.name}")
                if set(current) == set(before):
                    add(
                        errors,
                        all(
                            before[name] <= current[name] <= after[name]
                            for name in before
                        ),
                        f"resume OC counters are outside interval:{path.name}",
                    )

    campaign: dict[str, Any] = {}
    run_id = contract.get("causal_run_id")
    if isinstance(run_id, str) and isinstance(commit, str):
        run_dir = REPO / "results/sm110_gemm_causal_campaign" / run_id
        campaign = load_campaign_auditor().audit(
            run_dir,
            require_ncu=True,
            expected_commit=commit,
        )
        add(errors, campaign.get("pass") is True, "causal campaign independent audit failed")
        if campaign.get("pass") is not True:
            errors.extend(
                f"campaign:{message}" for message in campaign.get("errors", [])
            )
        warnings.extend(
            f"campaign:{message}" for message in campaign.get("warnings", [])
        )
    else:
        errors.append("causal campaign cannot be located")

    return {
        "pass": not errors,
        "suite_dir": str(suite_dir),
        "expected_commit": commit,
        "overcurrent_deltas": deltas,
        "profile_qualified": bool(campaign.get("profile_qualified")),
        "errors": errors,
        "warnings": sorted(set(warnings)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    if args.expected_commit is not None and COMMIT_RE.fullmatch(
        args.expected_commit
    ) is None:
        parser.error("--expected-commit must be 40 lowercase hex digits")
    result = audit_suite(args.suite_dir, expected_commit=args.expected_commit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

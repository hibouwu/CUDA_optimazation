from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

from .io import pipeline_profiles_from_rows
from .model import ModelError, audit_pipeline_profiles


RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _load_independent_auditor(repo_root: Path) -> Any:
    path = (
        repo_root
        / "microbench/sm110_gemm_causal_campaign/audit_campaign.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sm110_causal_campaign_independent_auditor", path
    )
    if spec is None or spec.loader is None:
        raise ModelError(f"cannot load causal campaign auditor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def import_causal_profile(
    *,
    repo_root: Path,
    run_id: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Audit one returned Thor causal campaign and emit model profile input.

    The campaign auditor independently reconstructs the frozen 91-case matrix,
    every timestamp-derived metric, the joint fit, SASS/NCU contracts, and the
    artifact manifest from Git blobs at ``expected_commit``.  A fit that misses
    its predeclared gates is imported as ``quarantined`` rather than promoted.
    """

    repo_root = repo_root.resolve()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ModelError("causal run ID must be a stable path-safe identifier")
    if not COMMIT_RE.fullmatch(expected_commit):
        raise ModelError("causal expected commit must be 40 lowercase hex digits")
    run_dir = repo_root / "results/sm110_gemm_causal_campaign" / run_id
    auditor = _load_independent_auditor(repo_root)
    audit = auditor.audit(
        run_dir,
        require_ncu=True,
        expected_commit=expected_commit,
    )
    if audit.get("pass") is not True:
        raise ModelError(
            "causal campaign independent audit failed: "
            f"{audit.get('errors', [])}"
        )
    profile_path = run_dir / "pipeline_profile.json"
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(f"cannot read causal pipeline profile: {error}") from error
    profiles = pipeline_profiles_from_rows([raw])
    if len(profiles) != 1:
        raise ModelError("causal campaign must emit exactly one pipeline profile")
    profile = profiles[0]
    if profile.source_id != run_id or profile.expected_commit != expected_commit:
        raise ModelError("causal profile identity differs from import contract")
    provenance_findings = audit_pipeline_profiles(
        profiles, repo_root=repo_root
    )
    errors = [
        finding for finding in provenance_findings
        if finding["severity"] == "error"
    ]
    if errors:
        raise ModelError(f"causal profile provenance audit failed: {errors}")
    return {
        "schema_version": 1,
        "kind": "sm110_causal_pipeline_profile_import",
        "run_id": run_id,
        "expected_commit": expected_commit,
        "qualification": profile.qualification,
        "profile_qualified": profile.closure_qualified,
        "audit": audit,
        "pipeline_profiles": [profile.to_dict()],
    }

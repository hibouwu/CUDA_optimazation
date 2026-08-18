from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .model import ModelError


@dataclass(frozen=True)
class SuiteLinkage:
    suite_id: str
    expected_commit: str
    hostname: str
    gpu_identity: str
    compute_run_id: str
    component_run_id: str
    full_gemm_run_id: str
    ncu_required: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteDeclaredProvenance:
    source_paths: tuple[str, ...]
    source_urls: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read suite artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelError(f"suite artifact is not a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read suite artifact {path}: {exc}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ModelError(f"suite JSONL has no valid environment snapshots: {path}")
    return rows


def _inside_repo(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ModelError(f"suite run directory is outside repo: {path}") from exc
    return resolved


def _successful_output(
    environment: dict[str, object], key: str, *, run_id: str
) -> str:
    record = environment.get(key)
    if not isinstance(record, dict):
        raise ModelError(f"{run_id}: environment field {key} is missing")
    if record.get("returncode") != 0:
        raise ModelError(f"{run_id}: environment command {key} did not succeed")
    output = str(record.get("output", "")).strip()
    if not output:
        raise ModelError(f"{run_id}: environment command {key} has empty output")
    return output


def audit_suite_linkage(
    compute_run_dir: Path,
    component_run_dir: Path,
    full_gemm_run_dir: Path,
    *,
    repo_root: Path,
    expected_commit: str,
    require_ncu: bool = False,
) -> SuiteLinkage:
    """Prove that three independently audited bundles form one frozen suite."""
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ModelError("expected_commit must be exactly 40 lowercase hex digits")
    run_dirs = {
        "compute": _inside_repo(compute_run_dir, repo_root),
        "component": _inside_repo(component_run_dir, repo_root),
        "full_gemm": _inside_repo(full_gemm_run_dir, repo_root),
    }
    specs = {
        label: _read_json(path / "run_spec.json")
        for label, path in run_dirs.items()
    }
    environments = {
        label: _read_json(path / "environment.json")
        for label, path in run_dirs.items()
    }
    environment_snapshots = {
        label: _read_jsonl(path / "environment_snapshots.jsonl")
        for label, path in run_dirs.items()
    }

    expected_campaigns = {
        "compute": "sm110_dense_tcgen05_compute_only",
        "component": "sm110_gemm_component_closure",
        "full_gemm": "sm110_full_gemm_closure",
    }
    suffixes = {
        "compute": "-compute",
        "component": "-components",
        "full_gemm": "-full",
    }
    suite_ids: set[str] = set()
    run_ids: dict[str, str] = {}
    for label, spec in specs.items():
        run_id = str(spec.get("run_id", ""))
        run_ids[label] = run_id
        if run_id != run_dirs[label].name:
            raise ModelError(
                f"{label}: run_spec run_id {run_id!r} does not match directory "
                f"{run_dirs[label].name!r}"
            )
        suffix = suffixes[label]
        if not run_id.endswith(suffix) or len(run_id) == len(suffix):
            raise ModelError(f"{label}: run_id does not end with {suffix!r}")
        suite_ids.add(run_id[: -len(suffix)])
        if spec.get("campaign") != expected_campaigns[label]:
            raise ModelError(f"{run_id}: unexpected campaign contract")
        if spec.get("static_only") is not False:
            raise ModelError(f"{run_id}: static-only evidence cannot enter a suite")
    if len(suite_ids) != 1:
        raise ModelError(f"suite run IDs do not share one prefix: {run_ids}")

    all_environments = [
        environment
        for label in run_dirs
        for environment in [
            environments[label],
            *environment_snapshots[label],
        ]
    ]
    hostnames = {
        str(env.get("hostname", "")).strip() for env in all_environments
    }
    if "" in hostnames or len(hostnames) != 1:
        raise ModelError(f"suite hostname mismatch: {sorted(hostnames)}")
    identity_keys = {
        "compute": "nvidia_smi_identity_csv",
        "component": "gpu_identity",
        "full_gemm": "gpu_identity",
    }
    identities: dict[str, str] = {}
    all_identities: list[str] = []
    for label in run_dirs:
        for snapshot_index, environment in enumerate(
            [environments[label], *environment_snapshots[label]]
        ):
            identity = _successful_output(
                environment, identity_keys[label], run_id=run_ids[label]
            )
            if snapshot_index == 0:
                identities[label] = identity
            all_identities.append(identity)
    if len(set(all_identities)) != 1:
        raise ModelError(f"suite GPU identity mismatch: {identities}")
    for label in run_dirs:
        for snapshot_index, environment in enumerate(
            [environments[label], *environment_snapshots[label]]
        ):
            git_head = _successful_output(
                environment, "git_head", run_id=run_ids[label]
            )
            if git_head != expected_commit:
                raise ModelError(
                    f"{run_ids[label]}: environment snapshot {snapshot_index} "
                    f"Git HEAD {git_head!r} does not match frozen commit "
                    f"{expected_commit}"
                )
    if require_ncu and specs["full_gemm"].get("ncu_requested") is not True:
        raise ModelError("full-GEMM run did not request NCU evidence")

    return SuiteLinkage(
        suite_id=next(iter(suite_ids)),
        expected_commit=expected_commit,
        hostname=next(iter(hostnames)),
        gpu_identity=identities["compute"],
        compute_run_id=run_ids["compute"],
        component_run_id=run_ids["component"],
        full_gemm_run_id=run_ids["full_gemm"],
        ncu_required=require_ncu,
    )


def collect_suite_declared_provenance(
    run_dirs: tuple[Path, ...], *, repo_root: Path
) -> SuiteDeclaredProvenance:
    """Collect the source graph frozen into audited campaign run specs."""
    repo_root = repo_root.resolve()
    source_paths: set[str] = set()
    source_urls: set[str] = set()

    def collect_nested_sources(value: object, *, key: str = "") -> None:
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                collect_nested_sources(nested_value, key=str(nested_key))
            return
        if isinstance(value, list):
            for nested_value in value:
                collect_nested_sources(nested_value, key=key)
            return
        if not isinstance(value, str) or not value:
            return
        if value.startswith(("https://", "http://")):
            source_urls.add(value)
        elif key in {"source_path", "source_paths"}:
            source_paths.add(value)

    for run_dir in run_dirs:
        resolved = _inside_repo(run_dir, repo_root)
        spec = _read_json(resolved / "run_spec.json")
        for key in ("generator", "generator_path", "support_manifest"):
            value = spec.get(key)
            if isinstance(value, str) and value:
                source_paths.add(value)
        dependencies = spec.get("source_dependencies", {})
        if not isinstance(dependencies, dict):
            raise ModelError(
                f"{resolved.name}: source_dependencies is not a path/hash object"
            )
        source_paths.update(str(path) for path in dependencies)
        for key, value in spec.items():
            if (
                isinstance(value, str)
                and value.startswith(("https://", "http://"))
                and (key.endswith("_source") or key.endswith("_url"))
            ):
                source_urls.add(value)
        support_manifest = spec.get("support_manifest")
        if isinstance(support_manifest, str) and support_manifest:
            manifest_path = repo_root / support_manifest
            manifest = _read_json(manifest_path)
            collect_nested_sources(manifest)
    for source_path in sorted(source_paths):
        path = Path(source_path)
        if path.is_absolute() or ".." in path.parts:
            raise ModelError(f"suite declares non-repo-relative source: {source_path}")
        if not (repo_root / path).is_file():
            raise ModelError(f"suite declared source is missing: {source_path}")
    return SuiteDeclaredProvenance(
        source_paths=tuple(sorted(source_paths)),
        source_urls=tuple(sorted(source_urls)),
    )


def collect_suite_artifact_paths(
    run_dirs: tuple[Path, ...], *, repo_root: Path
) -> tuple[str, ...]:
    """Enumerate every file contained in the three audited result bundles."""
    repo_root = repo_root.resolve()
    artifacts: set[str] = set()
    for run_dir in run_dirs:
        resolved = _inside_repo(run_dir, repo_root)
        for path in resolved.rglob("*"):
            if not path.is_file():
                continue
            resolved_path = path.resolve()
            try:
                relative = resolved_path.relative_to(repo_root)
            except ValueError as exc:
                raise ModelError(f"suite artifact escapes repo: {path}") from exc
            artifacts.add(str(relative))
    if not artifacts:
        raise ModelError("suite contains no result artifacts")
    return tuple(sorted(artifacts))

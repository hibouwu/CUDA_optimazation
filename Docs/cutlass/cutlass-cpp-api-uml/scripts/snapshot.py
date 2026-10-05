#!/usr/bin/env python3
"""Export the approved immutable Git blobs, never the source worktree."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

PIN = "8f50b052e1099fb982392a622caab69b97b63128"
PREFIXES = ("include/cutlass/", "include/cute/")
EXPECTED = {"include/cutlass": 712, "include/cute": 112}
ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def domain(path):
    """Mechanical navigation only; NOT an API-generation classification."""
    parts = PurePosixPath(path).parts
    return "/".join(parts[1:3]) if len(parts) > 3 else parts[1] + "/foundation"


def write_checked(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"Existing immutable snapshot differs: {path}")
    if not path.exists():
        path.write_bytes(content)


def build(repo, output=ROOT):
    resolved = git(repo, "rev-parse", PIN + "^{commit}").decode().strip()
    if resolved != PIN:
        raise RuntimeError("Commit mismatch")
    tree = git(repo, "ls-tree", "-rz", PIN, "--", "include/cutlass", "include/cute")
    entries = {}
    for item in tree.split(b"\0"):
        if not item:
            continue
        attrs, raw_path = item.split(b"\t", 1)
        mode, kind, oid = attrs.decode().split()
        path = raw_path.decode()
        if kind != "blob" or mode not in ("100644", "100755"):
            raise RuntimeError(f"Unexpected Git entry: {mode} {kind} {path}")
        if not path.startswith(PREFIXES) or ".." in PurePosixPath(path).parts:
            raise RuntimeError(f"Unexpected scope path: {path}")
        entries[path] = {"path": path, "git_blob": oid, "git_mode": mode}
    counts = Counter("/".join(PurePosixPath(p).parts[:2]) for p in entries)
    if dict(counts) != EXPECTED:
        raise RuntimeError(f"Scope changed: {dict(counts)} != {EXPECTED}")
    archive = git(repo, "archive", "--format=tar", PIN, "include/cutlass", "include/cute", "LICENSE.txt")
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.isdir():
                continue
            path = member.name
            if not member.isfile() or (path not in entries and path != "LICENSE.txt"):
                raise RuntimeError(f"Unexpected archive member: {path}")
            data = tar.extractfile(member).read()
            write_checked(output / "snapshot" / path, data)
            (output / "snapshot" / path).chmod(int(entries[path]["git_mode"], 8) & 0o777 if path in entries else 0o644)
            if path == "LICENSE.txt":
                continue
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blob != entries[path]["git_blob"]:
                raise RuntimeError(f"Git blob hash mismatch: {path}")
            source = data.decode("utf-8")
            entries[path].update({
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                "lines": len(source.splitlines()), "navigation_domain": domain(path),
                "declaration_status": "pending_phase_1", "api_generation": "pending_entity_review",
                "architecture_name_hints": sorted(set(re.findall(r"(?i)sm_?\d+[a-z]?", path))),
                "source_url": f"https://github.com/NVIDIA/cutlass/blob/{PIN}/{path}",
            })
            seen.add(path)
    if seen != set(entries):
        raise RuntimeError("Archive and tree differ")
    version_text = (output / "snapshot/include/cutlass/version.h").read_text()
    version = ".".join(re.search(rf"#define CUTLASS_{part}\s+(\d+)", version_text)[1]
                       for part in ("MAJOR", "MINOR", "PATCH"))
    manifest = {
        "schema_version": 1, "commit": PIN, "library_version_macros": version,
        "api_reading_mainline": "3.x", "scope_roots": list(EXPECTED),
        "file_count": len(entries), "file_counts": dict(sorted(counts.items())),
        "source_origin": "git_objects_not_worktree", "license": "snapshot/LICENSE.txt",
        "classification_note": "Directory and architecture hints are navigation only; never exclusion rules or API-generation proof.",
        "files": [entries[p] for p in sorted(entries)],
    }
    dest = output / "data/scope.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"commit": PIN, "version": version, "files": len(entries),
                      "bytes": sum(f["bytes"] for f in entries.values()), "counts": dict(counts)}))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT)
    a = p.parse_args()
    build(a.repo, a.output)

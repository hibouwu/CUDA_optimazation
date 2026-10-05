#!/usr/bin/env python3
"""Independent scope and content check, including deliberate corruptions."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PIN = "8f50b052e1099fb982392a622caab69b97b63128"


def validate(manifest, repo, snapshot):
    assert manifest["commit"] == PIN
    records = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-rz", PIN]).split(b"\0")
    expected = {}
    for record in records:
        if not record: continue
        attrs, path = record.split(b"\t", 1)
        path = path.decode()
        if path.startswith(("include/cutlass/", "include/cute/")):
            mode, kind, blob = attrs.decode().split()
            assert kind == "blob", (path, kind)
            expected[path] = {"mode": mode, "blob": blob}
    listed = [f["path"] for f in manifest["files"]]
    assert len(listed) == len(set(listed)), "duplicate file"
    assert set(listed) == set(expected), "scope membership mismatch"
    assert len(expected) == manifest["file_count"] == 824
    assert Counter(p.split('/')[1] for p in listed) == {"cutlass": 712, "cute": 112}
    assert manifest["schema_version"] == 1
    assert manifest["scope_roots"] == ["include/cutlass", "include/cute"]
    assert manifest["file_counts"] == {"include/cutlass": 712, "include/cute": 112}
    assert manifest["source_origin"] == "git_objects_not_worktree"
    assert manifest["api_reading_mainline"] == "3.x"
    assert manifest["license"] == "snapshot/LICENSE.txt"
    assert not snapshot.is_symlink(), "snapshot root is a symlink"
    actual_entries = list(snapshot.rglob('*'))
    assert not any(p.is_symlink() for p in actual_entries), "symlink in immutable snapshot"
    assert all(p.is_dir() or p.is_file() for p in actual_entries), "special filesystem object in snapshot"
    actual_files = {p.relative_to(snapshot).as_posix() for p in actual_entries if p.is_file()}
    assert actual_files == set(expected) | {"LICENSE.txt"}, "actual snapshot file set differs from fixed scope"
    for f in manifest["files"]:
        original = expected[f["path"]]
        assert f["git_blob"] == original["blob"], "blob identity differs from fixed Git tree"
        assert f["git_mode"] == original["mode"], "file mode differs from fixed Git tree"
        assert (snapshot / f["path"]).stat().st_mode & 0o777 == int(original["mode"], 8) & 0o777
        data = (snapshot / f["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == f["sha256"], f["path"]
        assert len(data) == f["bytes"]
        assert len(data.decode("utf-8").splitlines()) == f["lines"]
        parts = f["path"].split('/')
        assert f["navigation_domain"] == ('/'.join(parts[1:3]) if len(parts) > 3 else parts[1]+'/foundation')
        assert f["architecture_name_hints"] == sorted(set(re.findall(r'(?i)sm_?\d+[a-z]?', f['path'])))
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == original["blob"]
        assert f["source_url"] == f"https://github.com/NVIDIA/cutlass/blob/{PIN}/{f['path']}"
    assert (snapshot / "LICENSE.txt").read_bytes() == subprocess.check_output(["git", "-C", str(repo), "show", PIN+":LICENSE.txt"])
    version_text = (snapshot / "include/cutlass/version.h").read_text()
    assert manifest["library_version_macros"] == '.'.join(re.search(rf'#define CUTLASS_{p}\s+(\d+)', version_text)[1] for p in ('MAJOR','MINOR','PATCH'))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    a = p.parse_args()
    m = json.loads((ROOT / "data/scope.json").read_text())
    validate(m, a.repo, ROOT / "snapshot")
    mutations = {}
    for name in ("missing_file", "duplicate_file", "wrong_commit", "corrupt_hash", "forged_blob", "wrong_mode", "forged_version", "wrong_scope_roots", "wrong_file_counts", "wrong_line_count", "wrong_navigation", "wrong_architecture_hint", "wrong_license"):
        bad = copy.deepcopy(m)
        if name == "missing_file": bad["files"].pop()
        elif name == "duplicate_file": bad["files"].append(copy.deepcopy(bad["files"][0]))
        elif name == "wrong_commit": bad["commit"] = "0"*40
        elif name == "forged_blob": bad["files"][0]["git_blob"] = "0"*40
        elif name == "wrong_mode": bad["files"][0]["git_mode"] = "100755" if bad["files"][0]["git_mode"] == "100644" else "100644"
        elif name == "forged_version": bad["library_version_macros"] = "9.9.9"
        elif name == "wrong_scope_roots": bad["scope_roots"] = ["include/cute"]
        elif name == "wrong_file_counts": bad["file_counts"]["include/cute"] = 0
        elif name == "wrong_line_count": bad["files"][0]["lines"] += 1
        elif name == "wrong_navigation": bad["files"][0]["navigation_domain"] = "fake"
        elif name == "wrong_architecture_hint": bad["files"][0]["architecture_name_hints"] = ["sm999"]
        elif name == "wrong_license": bad["license"] = "elsewhere/LICENSE.txt"
        else: bad["files"][0]["sha256"] = "0"*64
        try: validate(bad, a.repo, ROOT / "snapshot")
        except AssertionError: mutations[name] = "detected"
        else: raise AssertionError("mutation escaped: " + name)
    bad = copy.deepcopy(m)
    victim = ROOT / "snapshot" / bad["files"][0]["path"]
    corrupted = victim.read_bytes() + b"\n// adversarial coordinated mutation\n"
    bad["files"][0].update({"sha256": hashlib.sha256(corrupted).hexdigest(), "bytes": len(corrupted),
        "git_blob": hashlib.sha1(b"blob " + str(len(corrupted)).encode() + b"\0" + corrupted).hexdigest()})
    original_read = Path.read_bytes
    with patch.object(Path, "read_bytes", lambda p: corrupted if p == victim else original_read(p)):
        try: validate(bad, a.repo, ROOT / "snapshot")
        except AssertionError: mutations["coordinated_snapshot_and_manifest_tampering"] = "detected"
        else: raise AssertionError("coordinated mutation escaped")
    original_is_symlink = Path.is_symlink
    with patch.object(Path, 'is_symlink', lambda p: True if p == victim else original_is_symlink(p)):
        try: validate(m, a.repo, ROOT / "snapshot")
        except AssertionError: mutations["symlink_snapshot_entry"] = "detected"
        else: raise AssertionError("symlink mutation escaped")
    original_rglob = Path.rglob
    original_is_file = Path.is_file
    original_is_dir = Path.is_dir
    fake = ROOT / "snapshot/unapproved-extra.hpp"
    with patch.object(Path, 'rglob', lambda p,*a,**kw: iter(list(original_rglob(p,*a,**kw))+([fake] if p == ROOT / 'snapshot' else []))), \
         patch.object(Path, 'is_file', lambda p: True if p == fake else original_is_file(p)):
        try: validate(m, a.repo, ROOT / "snapshot")
        except AssertionError: mutations["extra_snapshot_file"] = "detected"
        else: raise AssertionError("extra file mutation escaped")
    with patch.object(Path, 'rglob', lambda p,*a,**kw: iter(list(original_rglob(p,*a,**kw))+([fake] if p == ROOT / 'snapshot' else []))), \
         patch.object(Path, 'is_file', lambda p: False if p == fake else original_is_file(p)), \
         patch.object(Path, 'is_dir', lambda p: False if p == fake else original_is_dir(p)):
        try: validate(m, a.repo, ROOT / "snapshot")
        except AssertionError: mutations["special_snapshot_object"] = "detected"
        else: raise AssertionError("special object mutation escaped")
    result = {"stage": 0, "scope_content_check": "pass", "files": 824,
              "declaration_coverage_claimed": False, "relationship_coverage_claimed": False,
              "mutation_tests": mutations}
    out = ROOT / "data/phase-0-checks.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

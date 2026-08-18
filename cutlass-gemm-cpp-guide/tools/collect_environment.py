#!/usr/bin/env python3
"""Capture toolchain/device provenance without turning absence of a GPU into PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def command_output(command: list[str], cwd: Path | None = None) -> dict[str, object]:
    try:
        result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=30, check=False)
        return {"command": command, "returncode": result.returncode, "output": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "returncode": None, "output": str(error)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    payload = {
        "schema_version": 1,
        "platform": {"machine": platform.machine(), "system": platform.system(), "release": platform.release()},
        "lock_sha256": sha256(root / "versions.lock.json"),
        "cutlass_head": command_output(["git", "-C", "third_party/cutlass", "rev-parse", "HEAD"], cwd=root),
        "nvcc": command_output(["nvcc", "--version"]),
        "ptxas": command_output(["ptxas", "--version"]),
        "nvdisasm": command_output(["nvdisasm", "--version"]),
        "host_compiler": command_output(["g++", "--version"]),
        "cmake": command_output(["cmake", "--version"]),
        "ninja": command_output(["ninja", "--version"]),
        "gpu": command_output([
            "nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"
        ]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Function-local PTX/SASS inspection for one standalone guide case."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout


def ptx_function_blocks(text: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^\s*(?:\.visible\s+)?\.entry\s+", text)]
    if not starts:
        return []
    starts.append(len(text))
    return [text[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def sass_function_blocks(text: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^\s*Function\s*:\s*.+$", text))
    if not matches:
        return []
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[match.start():end])
    return blocks


def block_with_all(blocks: Iterable[str], patterns: list[str]) -> str | None:
    for block in blocks:
        if all(re.search(pattern, block, re.IGNORECASE) for pattern in patterns):
            return block
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--case", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--ptx", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-ptx", action="store_true")
    args = parser.parse_args()

    manifest_path = args.root / "cases" / args.case / "case.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_ptx = manifest["codegen"]["ptx"]["required"]
    forbidden_ptx = manifest["codegen"]["ptx"].get("forbidden", [])
    required_sass = manifest["codegen"]["sass"]["required_families"]

    errors: list[str] = []
    ptx_block = None
    if args.ptx and args.ptx.is_file():
        ptx_text = args.ptx.read_text(encoding="utf-8", errors="replace")
        ptx_block = block_with_all(ptx_function_blocks(ptx_text), required_ptx)
        if ptx_block is None:
            errors.append("no single PTX function block contains every required pattern")
        elif any(re.search(pattern, ptx_block, re.IGNORECASE) for pattern in forbidden_ptx):
            errors.append("target PTX function block contains a forbidden pattern")
    elif not args.allow_missing_ptx:
        errors.append("PTX artifact is required")

    sass_text = run(["cuobjdump", "--dump-sass", str(args.binary)])
    sass_block = block_with_all(sass_function_blocks(sass_text), required_sass)
    if sass_block is None:
        errors.append("no single SASS function block contains every required opcode family")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sass.full.txt").write_text(sass_text, encoding="utf-8")
    if sass_block:
        (args.output / "sass.target.txt").write_text(sass_block, encoding="utf-8")
    if ptx_block:
        (args.output / "ptx.target.ptx").write_text(ptx_block, encoding="utf-8")

    evidence = {
        "schema_version": 1,
        "case_id": args.case,
        "attribution": "single_target_function_block",
        "binary_sha256": sha256(args.binary),
        "ptx_verified": ptx_block is not None,
        "sass_verified": sass_block is not None,
        "errors": errors,
    }
    (args.output / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"CODEGEN_CONTRACT_PASS case={args.case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

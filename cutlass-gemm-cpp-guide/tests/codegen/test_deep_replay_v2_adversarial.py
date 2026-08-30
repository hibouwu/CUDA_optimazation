#!/usr/bin/env python3
"""Adversarial checks for independent source/binary derivation replay."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codegen_v2.deep_replay import replay_result_derivations  # noqa: E402
from codegen_v2.model import (  # noqa: E402
    ContractError,
    load_strict_json,
    sha256_file,
    validate_result,
)
from test_codegen_v2_model_adversarial import (  # noqa: E402
    make_fixture,
    reseal_manifest_and_journal,
    write_json,
)


class FixtureReplay:
    """Simulate the pinned tools while preserving an immutable clean baseline."""

    def __init__(self, root: Path, paths: dict[str, Path]):
        self.root = root
        manifest = load_strict_json(paths["manifest"])
        self.baseline = {
            item["artifact_id"]: (root / item["path"]).read_bytes()
            for item in manifest["items"]
            if item["artifact_id"]
            in {
                "type_output",
                "fatbin",
                "ptx_full",
                "cubin",
                "elf_symbols",
                "code_metadata",
                "nvdisasm_json",
                "nvdisasm_text",
            }
        }

    @staticmethod
    def _mount_source(command: list[str], container_path: str) -> Path:
        for index, value in enumerate(command[:-1]):
            if value != "-v":
                continue
            host, _, target = command[index + 1].partition(":")
            if target.removesuffix(":ro") == container_path:
                return Path(host)
        raise AssertionError(f"fixture command lacks mount for {container_path}: {command}")

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[bytes]:
        entrypoint = command[command.index("--entrypoint") + 1]
        replay_root = self._mount_source(command, "/out") if ":/out" in " ".join(command) else None
        stdout = b""
        if entrypoint == "/usr/local/cuda/bin/nvcc":
            assert replay_root is not None
            if "--fatbin" in command:
                (replay_root / "full/kernel.fatbin").write_bytes(self.baseline["fatbin"])
            else:
                (replay_root / "full/type_witness").write_bytes(b"\x7fELFfixture")
        elif entrypoint == "/out/full/type_witness":
            stdout = self.baseline["type_output"]
        elif entrypoint == "/usr/local/cuda/bin/cuobjdump":
            assert replay_root is not None
            if "--list-ptx" in command:
                stdout = b"PTX file    1: replay.sm_110a.ptx\n"
            elif "--list-elf" in command:
                stdout = b"ELF file    1: replay.sm_110a.cubin\n"
            elif "--extract-ptx" in command:
                (replay_root / "full/extracted/replay.sm_110a.ptx").write_bytes(
                    self.baseline["ptx_full"]
                )
            elif "--extract-elf" in command:
                (replay_root / "full/extracted/replay.sm_110a.cubin").write_bytes(
                    self.baseline["cubin"]
                )
            elif "--dump-elf-symbols" in command:
                stdout = self.baseline["elf_symbols"]
            elif "--dump-sass" in command:
                stdout = self.baseline["code_metadata"]
        elif entrypoint == "/usr/local/cuda/bin/nvdisasm":
            stdout = (
                self.baseline["nvdisasm_json"]
                if "--emit-json" in command
                else self.baseline["nvdisasm_text"]
            )
        else:
            raise AssertionError(f"unexpected fixture entrypoint: {entrypoint}")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")


def refresh_item(manifest: dict, root: Path, artifact_id: str) -> None:
    item = next(value for value in manifest["items"] if value["artifact_id"] == artifact_id)
    path = root / item["path"]
    item["sha256"] = sha256_file(path)
    item["size_bytes"] = path.stat().st_size


def require_deep_rejected(name: str, expected: str, mutate) -> None:
    root, paths = make_fixture()
    try:
        replay = FixtureReplay(root, paths)
        mutate(root, paths)
        validate_result(root, paths["result"], require_archive=True)
        try:
            replay_result_derivations(
                root,
                paths["result"],
                run_command=replay,
                environment_verified=True,
            )
        except ContractError as error:
            if expected not in str(error):
                raise AssertionError(f"{name}: wrong deep-replay rejection: {error}") from error
        else:
            raise AssertionError(f"{name}: derivation splice was accepted")
    finally:
        shutil.rmtree(root)


def main() -> int:
    root, paths = make_fixture()
    try:
        replay_result_derivations(
            root,
            paths["result"],
            run_command=FixtureReplay(root, paths),
            environment_verified=True,
        )
    finally:
        shutil.rmtree(root)

    def forge_type_chain(root: Path, paths: dict[str, Path]) -> None:
        value = load_strict_json(paths["type_output"])
        for canonical, consumer in (
            ("dispatch_policy", "mainloop_dispatch_policy"),
            ("tiled_mma", "mainloop_tiled_mma"),
            ("mma_atom", "tiled_mma_atom"),
        ):
            value["resolved_types"][canonical] = f"FORGED_{canonical}_SS"
            value["resolved_types"][consumer] = value["resolved_types"][canonical]
        payload = (json.dumps(value, indent=2) + "\n").encode()
        paths["type_output"].write_bytes(payload)
        manifest = load_strict_json(paths["manifest"])
        stdout_item = next(
            item for item in manifest["items"] if item["artifact_id"] == "run_type_witness_stdout"
        )
        stdout_path = root / stdout_item["path"]
        stdout_path.write_bytes(payload)

        def mutate(manifest_value):
            refresh_item(manifest_value, root, "type_output")
            refresh_item(manifest_value, root, "run_type_witness_stdout")

        reseal_manifest_and_journal(paths, mutate)

    require_deep_rejected(
        "forged_type_chain",
        "type witness output differs",
        forge_type_chain,
    )

    def replace_binary(artifact_id: str, payload: bytes):
        def mutate(root: Path, paths: dict[str, Path]) -> None:
            manifest = load_strict_json(paths["manifest"])
            item = next(value for value in manifest["items"] if value["artifact_id"] == artifact_id)
            (root / item["path"]).write_bytes(payload)
            reseal_manifest_and_journal(
                paths, lambda manifest_value: refresh_item(manifest_value, root, artifact_id)
            )

        return mutate

    require_deep_rejected(
        "unrelated_fatbin",
        "FATBIN differs",
        replace_binary("fatbin", bytes.fromhex("50ed55ba") + b"unrelated-fatbin"),
    )
    require_deep_rejected(
        "unrelated_cubin",
        "CUBIN differs",
        replace_binary("cubin", b"\x7fELFunrelated-cubin"),
    )

    print("DEEP_REPLAY_V2_ADVERSARIAL_PASS mutations=3 positive=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

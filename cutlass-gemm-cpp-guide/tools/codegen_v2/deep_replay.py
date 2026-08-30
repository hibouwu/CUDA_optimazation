"""Independent pinned-container replay of static-codegen artifact derivations."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from .model import (
    ContractError,
    _reject_diagnostic_value,
    load_strict_json,
    safe_path,
    sha256_file,
    sha256_bytes,
    validate_result,
)


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[bytes]]


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ContractError(
            f"deep replay command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout={result.stdout.decode(errors='replace')}\n"
            f"stderr={result.stderr.decode(errors='replace')}"
        )
    return result


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _docker_command(
    image: str,
    *,
    entrypoint: str,
    guide_root: Path | None = None,
    replay_root: Path | None = None,
    replay_read_only: bool = False,
    workdir: str | None = None,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--security-opt",
        "label=disable",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "LC_ALL=C",
        "--entrypoint",
        entrypoint,
    ]
    if guide_root is not None:
        command += ["-v", f"{guide_root}:/workspace:ro"]
    if replay_root is not None:
        suffix = ":ro" if replay_read_only else ""
        command += ["-v", f"{replay_root}:/out{suffix}"]
    if workdir is not None:
        command += ["-w", workdir]
    return command + [image]


def verify_deep_replay_environment(
    root: Path, *, run_command: CommandRunner = _run_checked
) -> None:
    contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
    image = contract["toolchain"]["container_reference"]
    inspected = run_command(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    ).stdout.decode("utf-8", errors="strict").strip()
    if inspected != contract["toolchain"]["container_digest"]:
        raise ContractError("deep replay Docker image differs from the pinned digest")
    environment = run_command(
        _docker_command(image, entrypoint="/usr/bin/env")
    ).stdout.decode("utf-8", errors="strict")
    names = {line.partition("=")[0] for line in environment.splitlines() if "=" in line}
    forbidden = set(contract["compile_plan"]["asserted_absent_environment"]) & names
    if forbidden:
        raise ContractError(f"deep replay container contains forbidden environment: {sorted(forbidden)}")
    tool_paths = {
        "nvcc": "/usr/local/cuda/bin/nvcc",
        "ptxas": "/usr/local/cuda/bin/ptxas",
        "cuobjdump": "/usr/local/cuda/bin/cuobjdump",
        "nvdisasm": "/usr/local/cuda/bin/nvdisasm",
        "host_compiler": "/usr/bin/g++",
    }
    for tool, executable in tool_paths.items():
        output = run_command(
            _docker_command(image, entrypoint="/usr/bin/sha256sum") + [executable]
        ).stdout.decode("utf-8", errors="strict")
        observed = output.split()[0] if output.split() else ""
        if observed != contract["toolchain"]["tool_binary_sha256"][tool]:
            raise ContractError(f"deep replay {tool} binary hash mismatch")


def _artifact_path(root: Path, manifest: dict[str, Any], artifact_id: str) -> Path:
    matches = [item for item in manifest["items"] if item["artifact_id"] == artifact_id]
    if len(matches) != 1:
        raise ContractError(f"deep replay requires one artifact {artifact_id}")
    return safe_path(root, matches[0]["path"], f"deep_replay.{artifact_id}")


def _replace_plan_paths(argv: list[str], cubin_name: str | None = None) -> list[str]:
    translated: list[str] = []
    for argument in argv:
        if argument == "<fatbin>":
            translated.append("/out/full/kernel.fatbin")
        elif argument == "<cubin>":
            if cubin_name is None:
                raise ContractError("deep replay cannot resolve <cubin> before extraction")
            translated.append(f"/out/full/extracted/{cubin_name}")
        else:
            translated.append(argument.replace("<attempt>", "/out"))
    return translated


def replay_result_derivations(
    root: Path,
    result_path: Path,
    *,
    run_command: CommandRunner = _run_checked,
    capture_command: CommandRunner = _run_capture,
    environment_verified: bool = False,
) -> None:
    """Rebuild and re-derive every decisive artifact from the current frozen inputs."""
    result = validate_result(root, result_path, require_archive=True)
    if not environment_verified:
        verify_deep_replay_environment(root, run_command=run_command)
    contract = load_strict_json(root / "tests/codegen/static_codegen_contract.json")
    image = contract["toolchain"]["container_reference"]
    instance = load_strict_json(root / result["instance_ref"]["path"])
    fingerprint = load_strict_json(root / result["fingerprint_ref"]["path"])
    manifest = load_strict_json(root / result["artifact_manifest_ref"]["path"])
    plan = {step["step_id"]: step for step in fingerprint["execution_plan"]}

    with tempfile.TemporaryDirectory(prefix=f"codegen-v2-replay-{instance['instance_id']}-") as value:
        replay_root = Path(value)
        full = replay_root / "full"
        extracted = full / "extracted"
        extracted.mkdir(parents=True)

        if result["status"] == "EXPECTED_STATIC_REJECT":
            prefix_reference = instance["translation_units"]["collective_prefix_witness"]
            if prefix_reference is not None:
                prefix_compile = _docker_command(
                    image,
                    entrypoint="/usr/local/cuda/bin/nvcc",
                    guide_root=root,
                    replay_root=replay_root,
                    workdir="/workspace",
                ) + _replace_plan_paths(
                    plan["compile_collective_prefix_witness"]["argv"]
                )
                run_command(prefix_compile)
                prefix_output = run_command(
                    _docker_command(
                        image,
                        entrypoint="/out/full/collective_prefix_witness",
                        replay_root=replay_root,
                        replay_read_only=True,
                    )
                ).stdout
                if prefix_output != _artifact_path(
                    root, manifest, "collective_prefix_output"
                ).read_bytes():
                    raise ContractError("deep replay collective prefix output differs")
            compile_type = _docker_command(
                image,
                entrypoint="/usr/local/cuda/bin/nvcc",
                guide_root=root,
                replay_root=replay_root,
                workdir="/workspace",
            ) + _replace_plan_paths(plan["compile_type_witness"]["argv"])
            rejected = capture_command(compile_type)
            reject_contract = instance["hypothesis"]["reject_contract"]
            if reject_contract is None or rejected.returncode != reject_contract[
                "expected_returncode"
            ]:
                raise ContractError("deep replay rejection return code differs")
            if (full / "type_witness").exists():
                raise ContractError("deep replay rejection produced a type executable")
            try:
                stderr = rejected.stderr.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ContractError("deep replay rejection stderr is not UTF-8") from error
            diagnostic, results, error_count = _reject_diagnostic_value(
                root,
                instance,
                stderr,
                sha256_bytes(rejected.stderr),
                len(rejected.stderr),
            )
            evidence = result["evidence"]
            if (
                results != evidence["diagnostic_results"]
                or error_count != evidence["observed_error_count"]
                or diagnostic["selected_lines"]
                != load_strict_json(
                    _artifact_path(root, manifest, "diagnostic_excerpt")
                )["selected_lines"]
            ):
                raise ContractError("deep replay rejection diagnostic differs")
            return

        compile_type = _docker_command(
            image,
            entrypoint="/usr/local/cuda/bin/nvcc",
            guide_root=root,
            replay_root=replay_root,
            workdir="/workspace",
        ) + _replace_plan_paths(plan["compile_type_witness"]["argv"])
        run_command(compile_type)
        type_output = run_command(
            _docker_command(
                image,
                entrypoint="/out/full/type_witness",
                replay_root=replay_root,
                replay_read_only=True,
            )
        ).stdout
        if type_output != _artifact_path(root, manifest, "type_output").read_bytes():
            raise ContractError("deep replay type witness output differs from tracked evidence")

        compile_fatbin = _docker_command(
            image,
            entrypoint="/usr/local/cuda/bin/nvcc",
            guide_root=root,
            replay_root=replay_root,
            workdir="/workspace",
        ) + _replace_plan_paths(plan["compile_fatbin"]["argv"])
        run_command(compile_fatbin)
        replay_fatbin = full / "kernel.fatbin"
        if sha256_file(replay_fatbin) != sha256_file(_artifact_path(root, manifest, "fatbin")):
            raise ContractError("deep replay FATBIN differs from the sealed source compilation")

        list_outputs: dict[str, str] = {}
        for step_id in ("list_ptx", "list_elf"):
            output = run_command(
                _docker_command(
                    image,
                    entrypoint="/usr/local/cuda/bin/cuobjdump",
                    replay_root=replay_root,
                )
                + _replace_plan_paths(plan[step_id]["argv"])
            ).stdout.decode("utf-8", errors="strict")
            list_outputs[step_id] = output
        ptx_names = re.findall(r"(?m)^PTX file\s+\d+:\s+(\S+)\s*$", list_outputs["list_ptx"])
        elf_names = re.findall(r"(?m)^ELF file\s+\d+:\s+(\S+)\s*$", list_outputs["list_elf"])
        if len(ptx_names) != 1 or len(elf_names) != 1:
            raise ContractError("deep replay FATBIN does not list one PTX and one ELF")

        for step_id in ("extract_ptx", "extract_elf"):
            run_command(
                _docker_command(
                    image,
                    entrypoint="/usr/local/cuda/bin/cuobjdump",
                    replay_root=replay_root,
                    workdir="/out/full/extracted",
                )
                + _replace_plan_paths(plan[step_id]["argv"])
            )
        ptx_files = sorted(extracted.glob("*.ptx"))
        cubin_files = sorted(extracted.glob("*.cubin"))
        if len(ptx_files) != 1 or len(cubin_files) != 1:
            raise ContractError("deep replay extraction did not produce one PTX and one CUBIN")
        if sha256_file(ptx_files[0]) != sha256_file(_artifact_path(root, manifest, "ptx_full")):
            raise ContractError("deep replay PTX differs from the sealed FATBIN extraction")
        if sha256_file(cubin_files[0]) != sha256_file(_artifact_path(root, manifest, "cubin")):
            raise ContractError("deep replay CUBIN differs from the sealed FATBIN extraction")

        derived_steps = {
            "elf_symbols": ("/usr/local/cuda/bin/cuobjdump", "elf_symbols"),
            "code_object_metadata": ("/usr/local/cuda/bin/cuobjdump", "code_metadata"),
            "nvdisasm_json": ("/usr/local/cuda/bin/nvdisasm", "nvdisasm_json"),
            "nvdisasm_text": ("/usr/local/cuda/bin/nvdisasm", "nvdisasm_text"),
        }
        for step_id, (entrypoint, artifact_id) in derived_steps.items():
            output = run_command(
                _docker_command(
                    image,
                    entrypoint=entrypoint,
                    replay_root=replay_root,
                )
                + _replace_plan_paths(plan[step_id]["argv"], cubin_files[0].name)
            ).stdout
            if output != _artifact_path(root, manifest, artifact_id).read_bytes():
                raise ContractError(
                    f"deep replay {artifact_id} differs from the sealed CUBIN derivation"
                )

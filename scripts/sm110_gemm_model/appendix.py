from __future__ import annotations

import json
import os
from pathlib import Path
from .model import ModelError


CATEGORY_ORDER = (
    "源码与生成器",
    "编译与运行命令",
    "原始 trial 与程序输出",
    "聚合结果与审计状态",
    "SASS 与 binary hash",
    "NCU 证据",
    "环境、运行合同与完整性标记",
    "其他已审计 artifact",
)


def _category(path: str) -> str:
    name = Path(path).name.lower()
    lower = path.lower()
    suffix = Path(path).suffix.lower()
    if suffix in {".cu", ".cuh", ".h", ".hpp", ".cc", ".cpp", ".py", ".sh"}:
        return "源码与生成器"
    if "command" in name or name in {"launcher.log"}:
        return "编译与运行命令"
    if (
        "trial_" in lower
        or name == "trials.jsonl"
        or name in {"stdout.log", "stderr.log"}
        or (suffix == ".csv" and "/ncu/" not in lower)
    ):
        return "原始 trial 与程序输出"
    if name in {"result.json", "summary.json", "campaign_status.json", "progress.jsonl"}:
        return "聚合结果与审计状态"
    if "sass" in name or "binary.sha256" in name:
        return "SASS 与 binary hash"
    if "/ncu/" in lower or suffix == ".ncu-rep" or name.startswith("ncu"):
        return "NCU 证据"
    if name in {
        "environment.json",
        "environment_snapshots.jsonl",
        "run_spec.json",
        "complete",
    }:
        return "环境、运行合同与完整性标记"
    return "其他已审计 artifact"


def _markdown_link(path: str, repo_link_prefix: str) -> str:
    escaped_label = path.replace("[", "\\[").replace("]", "\\]")
    target = str(Path(repo_link_prefix) / path) if repo_link_prefix else path
    return f"[{escaped_label}]({target})"


def render_suite_appendix(
    payload: dict[str, object], *, repo_link_prefix: str = ""
) -> str:
    linkage = payload.get("suite_linkage")
    if not isinstance(linkage, dict):
        raise ModelError("suite report is missing suite_linkage")
    artifact_paths = payload.get("artifact_paths")
    source_paths = payload.get("source_paths")
    source_urls = payload.get("source_urls")
    if not isinstance(artifact_paths, list) or not all(
        isinstance(path, str) for path in artifact_paths
    ):
        raise ModelError("suite report artifact_paths must be a string list")
    if not isinstance(source_paths, list) or not all(
        isinstance(path, str) for path in source_paths
    ):
        raise ModelError("suite report source_paths must be a string list")
    if not isinstance(source_urls, list) or not all(
        isinstance(url, str) for url in source_urls
    ):
        raise ModelError("suite report source_urls must be a string list")

    lines = [
        "## Microbenchmark 来源与证据附录",
        "",
        "本附录由 `render-suite-appendix` 从通过交叉审计的 suite report 确定性生成；",
        "路径集合来自三个 canonical auditor 接受的完整 result bundle，不是人工摘录。",
        "链接目标由输出文件位置机械换算到仓库根目录，显示文字仍为仓库相对路径。",
        "",
        "| Suite 参数 | 值 |",
        "| --- | --- |",
    ]
    for key in (
        "suite_id",
        "expected_commit",
        "hostname",
        "gpu_identity",
        "compute_run_id",
        "component_run_id",
        "full_gemm_run_id",
        "ncu_required",
    ):
        if key not in linkage:
            raise ModelError(f"suite_linkage is missing {key}")
        value = str(linkage[key]).replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "### 仓库源码与一手规范来源", ""])
    for path in sorted(set(source_paths)):
        lines.append(f"- {_markdown_link(path, repo_link_prefix)}")
    for url in sorted(set(source_urls)):
        lines.append(f"- [{url}]({url})")

    grouped: dict[str, list[str]] = {category: [] for category in CATEGORY_ORDER}
    for path in sorted(set(artifact_paths)):
        grouped[_category(path)].append(path)
    for category in CATEGORY_ORDER:
        lines.extend(["", f"### {category}", ""])
        if not grouped[category]:
            lines.append("- 无（suite report 未声明此类 artifact）")
            continue
        for path in grouped[category]:
            lines.append(f"- {_markdown_link(path, repo_link_prefix)}")
    return "\n".join(lines) + "\n"


def load_and_render_suite_appendix(
    path: Path, *, repo_root: Path, output_path: Path
) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read suite report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelError("suite report must be a JSON object")
    repo_root = repo_root.resolve()
    output_parent = output_path.resolve().parent
    repo_link_prefix = os.path.relpath(repo_root, output_parent)
    if repo_link_prefix == ".":
        repo_link_prefix = ""
    return render_suite_appendix(
        payload, repo_link_prefix=Path(repo_link_prefix).as_posix()
    )

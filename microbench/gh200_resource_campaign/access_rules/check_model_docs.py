"""Check GH200 model/evidence navigation; never rewrite scores or frozen files."""
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
GEMM = ROOT / "Docs/ModelEvaluation/gemm"
EXP = GEMM / "experiments/gh200_sm90"
ACCESS = EXP / "access_rules"
MODEL = GEMM / "model"
RULES_DIR = MODEL / "mechanism/hardware_rules"
CATEGORIES = {"实测", "规格", "半经验", "机制-硬件", "机制-程序"}


def targets(path):
    for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        link = link.removeprefix("<").removesuffix(">")
        if re.match(r"[a-zA-Z]+:", link):
            continue
        name, _, anchor = unquote(link).partition("#")
        yield (path.parent / name).resolve() if name else path, anchor


def anchors(path):
    text = path.read_text()
    found = set(re.findall(r'<a\s+id="([^"]+)"', text))
    counts = {}
    for heading in re.findall(r"^#{1,6} +(.+)$", text, re.M):
        heading = heading.lower().replace("`", "")
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        count = counts.get(slug, 0)
        found.add(slug if count == 0 else f"{slug}-{count}")
        counts[slug] = count + 1
    return found


def main():
    errors = []
    frozen = re.findall(r"\| `([^`]+\.md)`", (EXP / "HISTORICAL-PATHS.md").read_text())
    for name in frozen:
        current = EXP / name
        original = EXP.parent / "gh200_sm90_archive" / name
        if not current.exists() or not original.exists() or current.read_bytes() != original.read_bytes():
            errors.append(f"Frozen compatibility copy differs: {name}")

    for directory, excluded in ((EXP, set(frozen)), (ACCESS, set())):
        index = directory / "README.md"
        linked = {path for path, _ in targets(index)}
        for path in directory.glob("*.md"):
            if path.name != "README.md" and path.name not in excluded and path.resolve() not in linked:
                errors.append(f"Not indexed: {path.relative_to(ROOT)}")
        lines = index.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith("|") and " 类别 |" in line:
                column = line.split("|").index(" 类别 ")
                for row in lines[i + 2:]:
                    if not row.startswith("|"):
                        break
                    value = row.split("|")[column].strip()
                    if value not in CATEGORIES:
                        errors.append(f"Unknown category {value}: {index}")

    documents = [GEMM / "GOALS.md", MODEL / "README.md", MODEL / "spec/spec_baseline.md",
                 MODEL / "mechanism/program/cutlass_program_graph.md", EXP / "README.md", ACCESS / "README.md",
                 MODEL / "semi_empirical/MODEL-CANDIDATES.md", MODEL / "semi_empirical/MAINLINE-LONG-K.md",
                 MODEL / "common/inputs.md", MODEL / "common/interfaces.md", MODEL / "mechanism/composition.md",
                 GEMM / "experiments/README.md"]
    documents.extend(sorted(RULES_DIR.glob("*.md")))

    # 按模型分的文件夹：每份文档都要能从模型入口或硬件路径目录到达
    reachable = {path for index in (MODEL / "README.md", RULES_DIR / "README.md") for path, _ in targets(index)}
    for folder in ("common", "spec", "semi_empirical", "mechanism"):
        for path in (MODEL / folder).rglob("*.md"):
            if path.resolve() not in reachable and path != RULES_DIR / "README.md":
                errors.append(f"Not reachable from model index: {path.relative_to(ROOT)}")
    for path in documents:
        for target, anchor in targets(path):
            if not target.exists():
                errors.append(f"Missing target in {path.name}: {target}")
            elif anchor and target.suffix == ".md" and anchor not in anchors(target):
                errors.append(f"Missing anchor in {path.name}: {target.name}#{anchor}")

    profile_path = GEMM / "hardware/gh200_sm90/spec_baseline_profiles.json"
    for name, profile in json.loads(profile_path.read_text()).items():
        for source in profile["sources"]:
            filename, _, anchor = source.partition("#")
            target = (profile_path.parent / filename).resolve()
            if not target.exists() or (anchor and anchor not in anchors(target)):
                errors.append(f"Missing specification source for {name}: {source}")

    # Raw runs are local artifacts; a source-only checkout can check navigation
    # only when these linked artifacts have been restored.
    m11 = (RULES_DIR / "M11-SMEM.md").read_text()
    for name, median_key, max_key in (("20261012-m11-holdout", "median_abs", "max_abs"),
                                      ("20261012-m11-holdout-v2", "median_abs_error", "max_abs_error")):
        score = ROOT / "results/gh200_resource_campaign/access_rules" / name / "score.json"
        if not score.exists():
            errors.append(f"Cannot verify local frozen score: {score}")
            continue
        verdict = json.loads(score.read_text())["verdict"]
        values = f"{100 * verdict[median_key]:.2f}% / {100 * verdict[max_key]:.2f}%"
        if verdict["passed"] or values not in m11:
            errors.append(f"m11 status or rounded score out of sync: {name}")

    input_errors, input_summary = check_inputs()
    errors += input_errors + check_composition()
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: two indexes, {len(documents)} entry pages, {len(frozen)} frozen copies, two m11 scores")
    print("inputs: " + ", ".join(f"{k} {v}" for k, v in sorted(input_summary.items())))


CASE_PATTERNS = ("*/cases.json", "*/*/cases.json", "*/*-cases.json", "*/*/*-cases.json")


def check_inputs():
    """共同输入：probe 参数与解析字段都已登记；受检清单内的 GEMM case 全部可解析。

    返回 (错误, 摘要)。文件损坏或混有非 GEMM 行时报错；其他 JSON 格式按“非目标格式”排除并计数；
    “可解析”与“某读取方输入齐备”分别统计，无 setup 的历史记录不算 V09 或程序图可运行。
    """
    import sys
    from collections import Counter
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import case_inputs as CI
    errors = []
    documented = set(re.findall(r"`([a-z_0-9]+)`", (MODEL / "common/inputs.md").read_text()))
    probe = (Path(__file__).resolve().parent / "probes/r18.cu").read_text()
    options = {o.replace("-", "_") for o in re.findall(r'key\s*==\s*"--([a-z0-9-]+)"', probe)}
    fields = (set(CI.PROBE_DEFAULTS) | {d for d, _ in CI.DERIVED} | set(CI.PROBLEM) | set(CI.PROGRAM)
              | set(CI.BOOKKEEPING) | set(CI.ANNOTATIONS) | set(CI.SETUP_PROGRAM) | set(CI.SETUP_ECHO))
    for name in sorted((options | fields) - documented):
        errors.append(f"Input field not documented in inputs.md: {name}")
    base = ROOT / "results/gh200_resource_campaign/access_rules"
    files = sorted({p for pattern in CASE_PATTERNS for p in base.glob(pattern)})
    summary = Counter()
    for path in files:
        rel = path.relative_to(base)
        try:
            cases = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            errors.append(f"Corrupt case file {rel}: {error}")
            continue
        gemm = [isinstance(r, dict) and "config" in r and "m" in r for r in cases] if isinstance(cases, list) else []
        if not any(gemm):
            summary["excluded: not GEMM case rows"] += 1
            continue
        if not all(gemm):
            errors.append(f"Mixed rows in {rel}: {gemm.count(False)} rows lack config/m")
            continue
        setups = {}
        setup_path = path.with_name("static_setup.json")
        if path.name == "cases.json" and setup_path.exists():
            try:
                for item in json.loads(setup_path.read_text()):
                    setup = item.get("setup", item)
                    if not isinstance(setup, dict):
                        raise ValueError("setup must be an object")
                    setups[item.get("case", item.get("case_id"))] = setup
            except (OSError, ValueError, AttributeError) as error:
                errors.append(f"Corrupt static_setup for {rel}: {error}")
                continue
        environment_path = path.with_name("environment.json")
        summary["files"] += 1
        for row in cases:
            summary["rows"] += 1
            setup = setups.get(row.get("id"))
            summary["rows with setup"] += setup is not None
            gpu_uuid = None
            if not (setup or {}).get("gpu_uuid") and environment_path.exists():
                environment = json.loads(environment_path.read_text())
                gpu_uuid = environment.get("gpu", "").split(",")[0].strip() or None
            eff, problems = CI.resolve(row, setup, gpu_uuid=gpu_uuid)
            errors += [f"{rel}/{row.get('id')}: {x}" for x in problems]
            if not problems:
                for model in CI.REQUIRED:
                    summary[f"ready {model}"] += not CI.missing_for(eff, model)
    if not summary["rows"]:
        errors.append("Cannot verify inputs: no local GEMM case files")
    return errors, summary


def check_composition():
    """组合首版：模块表的每一行都给出预先规定的资格。"""
    text = (MODEL / "mechanism/composition.md").read_text()
    allowed = ("已核对", "已测", "待核对", "假设", "经验替代")
    table = text[text.index("| 模块 | 首版来源 | 资格 |"):].split("\n\n")[0].splitlines()[2:]
    errors = [f"Composition module without allowed qualification: {line.split('|')[1].strip()}"
              for line in table if not line.split("|")[3].strip().startswith(allowed)]
    for heading in ("## 1. 范围", "## 2. 模块与资格", "## 3. 连接方式", "## 4. 输出", "## 5. 完成判据"):
        if heading not in text:
            errors.append(f"Composition section missing: {heading}")
    return errors


if __name__ == "__main__":
    main()

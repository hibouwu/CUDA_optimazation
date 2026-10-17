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
NUMBERED_EXPERIMENT = r"EXP-\d{2}|[RVB]\d{2}|[A-Z][A-Z0-9]*-\d{2}"
QUESTION_DIR = re.compile(r"Q(0[1-9]|1[0-6])-[a-z0-9-]+$")
EXP_TOP = {"README.md", "HISTORICAL-PATHS.md", "MOVED-PATHS.md"}
ACCESS_TOP = {"README.md", "PLAN.md", "EXECUTION.md", "RESEARCH-GUIDE.md", "RULES.md"}


def question_dirs():
    return sorted(path for path in EXP.iterdir() if path.is_dir() and path.name.startswith("Q"))


def page_dirs():
    return {EXP, ACCESS, *question_dirs()}


def links(text, path):
    for label, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
        link = link.removeprefix("<").removesuffix(">")
        if re.match(r"[a-zA-Z]+:", link):
            continue
        name, _, anchor = unquote(link).partition("#")
        yield label, (path.parent / name).resolve() if name else path, anchor


def targets(path, text=None):
    for link in re.findall(r"\]\(([^)]+)\)", path.read_text() if text is None else text):
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


def numbered_experiment_targets():
    """逻辑编号来自现有入口与实验页，兼容副本和草稿不另计项。"""
    frozen = set(re.findall(r"\| `([^`]+\.md)`", (EXP / "HISTORICAL-PATHS.md").read_text()))
    pages = {path for directory in page_dirs() for path in directory.glob("*.md")
             if path.name not in frozen and ".draft." not in path.name
             and re.match(f"({NUMBERED_EXPERIMENT})-", path.name)}
    # EXP-01/03仍是现有编号入口；其历史正文虽为冻结副本，逻辑实验各计一次。
    for index in (EXP / "README.md", ACCESS / "README.md"):
        for target, _ in targets(index):
            if target.parent in page_dirs() and re.match(f"({NUMBERED_EXPERIMENT})-", target.name):
                pages.add(target)
    expected = {}
    for path in sorted(pages):
        name = re.match(f"({NUMBERED_EXPERIMENT})-", path.name).group(1)
        expected.setdefault(name, set()).add((path, None))
        if "已并入" in path.read_text().splitlines()[0]:
            expected[name].update((target, anchor) for target, anchor in targets(path)
                                  if target.parent in page_dirs() and anchor)
    # SYNC-02等页内子实验沿实际编号链接与所在节定位，不要求另建实验页。
    for path in sorted(pages):
        section = ""
        for line in path.read_text().splitlines():
            heading = re.match(r"^#{1,6} +(.+)$", line)
            if heading:
                section = re.sub(r"[^\w\- ]", "", heading[1].lower().replace("`", "")).replace(" ", "-")
            for label, _, _ in links(line, path):
                for name in re.findall(f"(?<![A-Za-z0-9])({NUMBERED_EXPERIMENT})(?![A-Za-z0-9-])", label):
                    if name not in expected:
                        expected[name] = {(path, section)}
    return expected


def check_research_questions(text=None):
    """每个编号须在16问题主表内有带实际目标的显式编号链接。"""
    index = EXP / "README.md"
    text = index.read_text() if text is None else text
    _, start, section = text.partition('<a id="research-questions"></a>')
    section, end, _ = section.partition("\n## 测量口径")
    if not start or not end:
        return ["Research question table missing: research-questions through 测量口径"]
    rows = [line for line in section.splitlines()
            if line.startswith("|") and re.search(r'<a id="question-\d{2}"', line)]
    question_ids = re.findall(r'<a id="(question-\d{2})"', "\n".join(rows))
    errors = []
    if len(rows) != 16 or sorted(question_ids) != [f"question-{i:02}" for i in range(1, 17)]:
        errors.append("Research question table must contain 16 rows with question-01 through question-16")
    expected = numbered_experiment_targets()
    covered = set()
    for row in rows:
        for label, target, anchor in links(row, index):
            for name in re.findall(f"(?<![A-Za-z0-9])({NUMBERED_EXPERIMENT})(?![A-Za-z0-9-])", label):
                if any(target == path and (section_anchor is None or anchor == section_anchor)
                       for path, section_anchor in expected.get(name, ())):
                    covered.add(name)
    errors += [f"Numbered experiment not covered by research question table: {name}"
               for name in sorted(expected.keys() - covered)]
    # 实验页按主归属问题存放：Qnn 目录中的页面须出现在第 nn 行
    row_targets = {int(re.search(r'question-(\d{2})', row)[1]): {target for _, target, _ in links(row, index)}
                   for row in rows}
    for directory in question_dirs():
        match = QUESTION_DIR.match(directory.name)
        if not match:
            errors.append(f"Question folder name must be Qnn-slug with nn in 01-16: {directory.name}")
            continue
        for page in sorted(directory.glob("*.md")):
            listed = row_targets.get(int(match[1]), set())
            # 已并入的页随合并目标存放，主表链接合并目标即可
            if "已并入" in page.read_text().splitlines()[0]:
                listed_here = any(target in listed for target, anchor in targets(page) if anchor)
            else:
                listed_here = page.resolve() in listed
            if not listed_here:
                errors.append(f"Page not listed under its folder's question row: {page.relative_to(ROOT)}")
    for directory, allowed in ((EXP, EXP_TOP | set(re.findall(r"\| `([^`]+\.md)`", (EXP / "HISTORICAL-PATHS.md").read_text()))),
                               (ACCESS, ACCESS_TOP)):
        errors += [f"Experiment page outside question folders: {page.relative_to(ROOT)}"
                   for page in sorted(directory.glob("*.md")) if page.name not in allowed]
    return errors


def result_coverage(docs_root=None, results_root=None, exception_index=None):
    """从 Docs 链接与可达结果 README 追踪日期轮次，例外表单独计数。"""
    docs_root = (docs_root or ROOT / "Docs").resolve()
    results_root = (results_root or ROOT / "results/gh200_resource_campaign/access_rules").resolve()
    exception_index = (exception_index or ACCESS / "README.md").resolve()
    rounds = {path.name: path for path in results_root.iterdir()
              if path.is_dir() and re.match(r"^\d{8}-", path.name)}
    exceptions = set()
    index_text = exception_index.read_text()
    before, marker, section = index_text.partition('<a id="non-experiment-results"></a>')
    section_parts = re.split(r"(?m)^## ", section, maxsplit=1)
    if marker:
        table = "\n".join(line for line in section_parts[0].splitlines() if line.startswith("|"))
        exceptions = {path.name for path, _ in targets(exception_index, table)
                      if path.parent == results_root and path.name in rounds}
        index_text = before + (section_parts[1] if len(section_parts) == 2 else "")

    reached, direct, readmes, visited = set(), set(), [], set()

    def visit(target, is_direct=False):
        if not target.exists() or not target.is_relative_to(results_root):
            return
        relative = target.relative_to(results_root).parts
        if not relative or relative[0] not in rounds:
            return
        name = relative[0]
        reached.add(name)
        if is_direct:
            direct.add(name)
        root_readme = rounds[name] / "README.md"
        if root_readme.is_file() and root_readme not in visited:
            readmes.append(root_readme)
        linked_readme = target / "README.md" if target.is_dir() else target
        if linked_readme.name == "README.md" and linked_readme.is_file() and linked_readme not in visited:
            readmes.append(linked_readme)

    for path in sorted(docs_root.rglob("*.md")):
        text = index_text if path.resolve() == exception_index else path.read_text()
        for target, _ in targets(path, text):
            visit(target, is_direct=True)
    while readmes:
        path = readmes.pop()
        if path in visited:
            continue
        visited.add(path)
        for target, _ in targets(path):
            visit(target)
    groups = {
        "direct": sorted(direct - exceptions),
        "indirect": sorted(reached - direct - exceptions),
        "non_experiment": sorted(exceptions),
        "uncovered": sorted(rounds.keys() - reached - exceptions),
    }
    return {"results_root": str(results_root), "counts": {"total": len(rounds),
            **{key: len(value) for key, value in groups.items()}}, **groups}


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
    linked = {path for index in (EXP / "README.md", ACCESS / "README.md") for path, _ in targets(index)}
    for directory in question_dirs():
        for path in directory.glob("*.md"):
            if path.resolve() not in linked:
                errors.append(f"Not indexed: {path.relative_to(ROOT)}")

    documents = [GEMM / "GOALS.md", MODEL / "README.md", MODEL / "spec/spec_baseline.md",
                 MODEL / "mechanism/program/cutlass_program_graph.md", EXP / "README.md", ACCESS / "README.md",
                 MODEL / "semi_empirical/MODEL-CANDIDATES.md", MODEL / "semi_empirical/MAINLINE-LONG-K.md",
                 MODEL / "common/inputs.md", MODEL / "common/interfaces.md", MODEL / "mechanism/composition.md",
                 MODEL / "mechanism/README.md",
                 GEMM / "experiments/README.md"]
    documents.extend(sorted(RULES_DIR.glob("*.md")))
    documents.extend(sorted(path for directory in question_dirs() for path in directory.rglob("*.md")))
    documents.append(EXP / "MOVED-PATHS.md")

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
    errors += input_errors + check_composition() + check_research_questions()
    coverage = result_coverage()
    errors += [f"Result round not linked from Docs or reachable README and not listed as non-experiment: {name}"
               for name in coverage["uncovered"]]
    print("results: " + ", ".join(f"{key} {value}" for key, value in coverage["counts"].items()))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: two indexes, 16 research questions, {len(documents)} entry pages, {len(frozen)} frozen copies, two m11 scores")
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
              | set(CI.PROTOCOL_FIELDS) | set(CI.BOOKKEEPING) | set(CI.ANNOTATIONS) | set(CI.SETUP_PROGRAM) | set(CI.SETUP_ECHO))
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

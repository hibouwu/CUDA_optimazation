#!/usr/bin/env python3
"""固定回放 m11 两次冻结留出与 lane 对照；不拟合，不启动 GPU。"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys


BASE = Path("gh200_resource_campaign/access_rules")
V1 = BASE / "20261012-m11-holdout"
V2 = BASE / "20261012-m11-holdout-v2"
LANE = BASE / "20261012-m11-lane-contrast/run-20261012T110219Z"
GPU = "GPU-009a8880-f309-a55d-d256-748fb9adad74"
MANIFEST_SHA = {
    1: "f117c1d2c356e34091f59b24e0c65916ad24cf7a702997a846751880d1c0aa20",
    2: "de37020d415bf2ee382b75dff5565e9956af191fe3f7d712bd7fce262c43970f",
}
SCORE_SHA = {
    1: "59248fca424c9804900a819408748969a4c652c16ae3cb63d4b4be60418037e9",
    2: "e9efd57d6c89450e19960ed3f42016c51b925ab2e38718eb309d32c6ad9dfdcb",
}
LANE_MANIFEST_SHA = "281274ca1a43c68abec6b2b87d4de9917218f470c001f061dd66663019e77bab"
V2_TRANSPORT_SHA = "b9b04febc240c0a03d7684e3fbb64b9d483040f8053e1c923def0649f084422a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def bind(root, name, expected):
    path = root / name
    require(path.resolve().is_relative_to(root), "输入路径逃离归档: " + name)
    actual = sha(path)
    require(actual == expected, "SHA不符: " + str(path))
    return actual


def package_identity(root):
    package = root / "package"
    identity = read(package / "identity.json")
    for name, field in (("historical-build/smem", "binary_sha256"),
                        ("historical-build/smem.sass", "historical_sass_sha256"),
                        ("protocol.json", "protocol_sha256")):
        bind(package, name, identity[field])
    for filename, directory in (("source-sha256.json", "source"),
                                ("execution-sha256.json", "execution-source")):
        for name, expected in read(package / filename).items():
            bind(package / directory, name, expected)
    environment = read(root / "measurements/environment.json")
    require(environment["gpu_uuid"] == identity["gpu_uuid"] == GPU and
            environment["CUDA_VISIBLE_DEVICES"] == GPU and environment["host"] == "romeo-a048" and
            environment["binary_sha256"] == identity["binary_sha256"] and
            environment["protocol_sha256"] == identity["protocol_sha256"], "设备/二进制/协议身份不符")
    return dict(original_identity=identity, environment=environment,
                source_sha256=sha(package / "source/probes/r03_lds_sts.cu"),
                source_manifest_sha256=sha(package / "source-sha256.json"),
                execution_manifest_sha256=sha(package / "execution-sha256.json"))


def pure_functions(path, names):
    """只编译已验 SHA 文件的指定纯函数，跳过历史 fit 库导入和主入口。"""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in selected} == set(names), "冻结纯函数缺失")
    namespace = {"defaultdict": defaultdict}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def check_process(root, case, trial):
    folder = root / "measurements/samples" / case["id"] / f"trial-{trial:02}"
    completion = read(folder / "completion.json")
    require(completion["status"] == "measured" and completion["attempt"] == 0 and
            [p.name for p in folder.iterdir() if p.is_dir()] == ["attempt-00"], "进程缺失或发生重试")
    attempt = folder / "attempt-00"
    record = read(attempt / "result.json")
    process = read(attempt / "measure.process.json")
    require(process["returncode"] == 0 and (attempt / "measure.stderr").read_text() == "", "旧进程失败")
    args = [str(value) for value in (case["op"], case["width"], case["warps"],
                                    int(case["dependent"]), int(case["native"]),
                                    case["iterations"], case["seed"])]
    require(process["command"][1:8] == args and process["command"][8].endswith(
        f"/samples/{case['id']}/trial-{trial:02}/attempt-00/addresses.map"), "命令与冻结条件不符")
    expected_map = "\n".join(str(value) for row in case["addresses_words"] for value in row) + "\n"
    require((attempt / "addresses.map").read_text() == expected_map, "地址映射不符")
    hashes = {name: sha(attempt / name) for name in (
        "result.json", "measure.stdout", "measure.process.json", "addresses.map", "output.u32")}
    require(record["map_sha256"] == hashes["addresses.map"] and
            record["output_sha256"] == hashes["output.u32"], "地址/输出SHA不符")
    require(read(attempt / "measure.stdout") == {key: value for key, value in record.items()
            if key not in ("output_sha256", "map_sha256")}, "stdout与result不符")
    require(record["status"] == "measured" and record["errors"] == record["local_bytes"] == 0 and
            record["checked_elements"] > 0 and record["warmup_converged"] is True,
            "原正确性/暖机/无spill资格未通过")
    require((record["width"], record["warps"], record["dependent"], record["iterations"],
             record["native_cutlass"]) == (case["width"], case["warps"], case["dependent"],
             case["iterations"], case["native"]), "样本形状不符")
    require(record["target_warp_instructions"] == 8 * case["iterations"] * case["warps"] and
            record["elapsed_cycles"] > 0 and record["smid"] == 124, "工作量/计时/SMID不符")
    return record, dict(case_id=case["id"], trial=trial, status="verified",
                        elapsed_cycles=record["elapsed_cycles"], warmup_windows=len(record["warmup_cycles"]),
                        hashes=hashes)


def replay_holdout(results, version):
    root = results / (V1 if version == 1 else V2)
    frozen = root / "freeze"
    bind(frozen, "manifest.sha256", MANIFEST_SHA[version])
    manifest = [line.split(maxsplit=1) for line in (frozen / "manifest.sha256").read_text().splitlines()]
    for expected, name in manifest:
        bind(frozen, name, expected)
        bind(root / "package/freeze", name, expected)
    for name in ("manifest.sha256", "frozen-at.txt"):
        bind(root / "package/freeze", name, sha(frozen / name))
    bind(root, "score.json", SCORE_SHA[version])
    identity = package_identity(root)
    require(identity["environment"]["freeze_manifest_sha256"] == MANIFEST_SHA[version], "运行冻结身份不符")
    require(read(root / "measurements/completion.json") == dict(
        cases=60 if version == 1 else 66, processes=180 if version == 1 else 198, failures=0) and
        read(root / "measurements/failures.json") == [], "旧整批运行未完成")
    transport_count = 0
    if version == 2:
        bind(root, "measurements-sha256.txt", V2_TRANSPORT_SHA)
        for line in (root / "measurements-sha256.txt").read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            bind(root, name.strip(), expected)
            transport_count += 1
        require(transport_count == 1396, "v2固定回传清单数量不符")
    cases = read(frozen / "holdout-cases.json")
    predictions = {row["id"]: row for row in read(frozen / "holdout-predictions.json")}
    parameters = read(frozen / "frozen-parameters.json")
    rule = read(frozen / "scoring-rule.json")
    old_score = read(root / "score.json")
    old_rows = {row["id"]: row for row in old_score["rows"]}
    require(len(cases) == (60 if version == 1 else 66) and
            {c["id"] for c in cases} == set(predictions) == set(old_rows) ==
            {p.name for p in (root / "measurements/samples").iterdir()}, "冻结条件/样本集合不符")
    if version == 1:
        functions = pure_functions(frozen / "source/m11_phase_rule.py", ("phases", "d_bank"))
        demand = lambda case: functions["d_bank"](case, True)
    else:
        functions = pure_functions(frozen / "source/m11_disjoint.py", ("d_disjoint",))
        demand = functions["d_disjoint"]
    rows, raw_checks = [], []
    for case in cases:
        p = predictions[case["id"]]
        group = f"op{case['op']}_b{case['width']}_d{int(case['dependent'])}"
        D = demand(case) / (8 * case["warps"])
        q = parameters["params"][group]
        F, c0, slope, floor = [abs(value) for value in q[:4]]
        P1 = max(F, c0 + slope * D)
        knee = (P1 / case["warps"]) / D
        predicted = q[4] + (case["iterations"] * 8 * case["warps"]) * max(D, P1 / case["warps"], floor)
        reasons = []
        if not parameters["demand_spans"][group][0] <= D <= parameters["demand_spans"][group][1]:
            reasons.append("demand_outside_training_span")
        if version == 1 and case["warps"] == 4 and .67 <= knee <= 1.5:
            reasons.append("knee_band_W4")
        if version == 2 and case["warps"] == 4 and not case["dependent"]:
            reasons.append("W4_independent_unsupported")
        require(case["iterations"] == 2048 and case["warps"] in (1, 4, 8) and not case["native"], "留出范围不符")
        require(group == p["group"] and D == p["per_instruction_demand"] and
                predicted == p["predicted_cycles"] and knee == p["knee_ratio"] and
                reasons == p["unsupported_reasons"] and (not reasons) == p["supported"],
                "冻结需求/预测/支持标签未逐位复现: " + case["id"])
        folder = root / "measurements/samples" / case["id"]
        require(sorted(p.name for p in folder.iterdir()) == [f"trial-{t:02}" for t in range(3)], "trial集合不符")
        records = []
        for trial in range(3):
            record, checked = check_process(root, case, trial)
            records.append(record)
            raw_checks.append(checked)
        cycles = [record["elapsed_cycles"] for record in records]
        observed = statistics.median(cycles)
        error = predicted / observed - 1
        old = old_rows[case["id"]]
        values = dict(D=D, predicted=predicted, observed=observed, cycles=cycles,
                      cv=statistics.pstdev(cycles) / statistics.mean(cycles), signed_error=error,
                      abs_cycle_error=abs(predicted - observed), smid=sorted({r["smid"] for r in records}))
        require(all(old[key] == value for key, value in values.items()) and
                old["supported"] == p["supported"], "原评分字段不符: " + case["id"])
        rows.append(dict(id=case["id"], group=group, op=case["op"], width=case["width"],
                         warps=case["warps"], dependent=case["dependent"], layout=case["layout"],
                         supported=p["supported"], unsupported_reasons=reasons, knee_ratio=knee,
                         **values))
    supported = [row for row in rows if row["supported"]]
    errors = [abs(row["signed_error"]) for row in supported]
    median_error, maximum_error = statistics.median(errors), max(errors)
    passed = maximum_error <= .10 and median_error <= .05
    original_verdict = old_score["verdict"]
    require(len(supported) == (50 if version == 1 else 47) and passed is False and
            original_verdict["passed"] is False and median_error == original_verdict[
                "median_abs" if version == 1 else "median_abs_error"] and
            maximum_error == original_verdict["max_abs" if version == 1 else "max_abs_error"], "原失败判定不符")
    over = [row["id"] for row in supported if abs(row["signed_error"]) > .10]
    require(len(over) == (15 if version == 1 else 1), "原超标数不符")
    return dict(schema="gh200-smem-frozen-holdout-replay/1", version=version, status="verified",
                rows=rows, verdict=original_verdict, original_rule=rule,
                reproduced_verdict=dict(passed=passed, supported=len(supported),
                    median_absolute_relative_error=median_error, maximum_absolute_relative_error=maximum_error,
                    over_10pct=over), raw_checks=raw_checks, identity=identity,
                frozen_prediction_float_equality=True, frozen_demand_and_support_equality=True,
                frozen_manifest_sha256=MANIFEST_SHA[version], original_score_sha256=SCORE_SHA[version],
                frozen_files_verified=len(manifest), transport_files_verified=transport_count,
                frozen_at=(frozen / "frozen-at.txt").read_text().strip(),
                source_method="冻结文件AST仅抽phases/d_bank或d_disjoint；时间按冻结hard-max纯公式复算，逐float严格相等",
                qualification="两次原失败均保留；没有正式可迁移SMEM机制资格；留出已见只能转开发",
                warmup_qualification="承接旧进程的warmup_converged与保存窗口；本入口不重算暖机CV",
                read_evidence_gap="回放保存输出的SHA和旧零错误状态，不把SHA一致冒充重新执行read数学oracle",
                physical_wavefronts_measured=False, full_GEMM_claim=False, new_fits=0, new_GPU_executions=0)


def replay_lane(results, output):
    root = results / LANE
    bind(root, "delivery-manifest.json", LANE_MANIFEST_SHA)
    delivery = read(root / "delivery-manifest.json")
    for name, record in delivery["files"].items():
        bind(root, name, record["sha256"])
    identity = package_identity(root)
    cases = read(root / "package/minimal-measurement-cases.json")
    require(len(cases) == 48 and len({c["id"] for c in cases}) == 48, "lane固定48条件不符")
    raw_checks = []
    for case in cases:
        for trial in range(3):
            _, checked = check_process(root, case, trial)
            raw_checks.append(checked)
    execution = root / "package/execution-source"
    previous_path = list(sys.path)
    previous_model = sys.modules.pop("r03_lane_contrast_model", None)
    try:
        sys.path.insert(0, str(execution))
        spec = importlib.util.spec_from_file_location("frozen_lane_analysis", execution / "analyze_r03_lane_contrast.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.analyze(root / "package", root / "measurements", output / "lane-analysis")
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("r03_lane_contrast_model", None)
        if previous_model is not None:
            sys.modules["r03_lane_contrast_model"] = previous_model
    files = ("conditions.json", "layout-pairs.json", "bridges.json", "F1_fullwarp_cycles.json",
             "old_form_nonbroadcast_refit_cycles.json", "decision.json")
    cv_differences = []
    for name in files:
        replayed, original = read(output / "lane-analysis" / name), read(root / "analysis" / name)
        if name != "conditions.json":
            require(replayed == original, "原lane分析不符: " + name)
            continue
        require(len(replayed) == len(original) == 48, "lane条件分析数量不符")
        for new, old in zip(replayed, original):
            require({k: v for k, v in new.items() if k != "cv"} ==
                    {k: v for k, v in old.items() if k != "cv"}, "lane非CV字段不符")
            require(new["cv"] in (old["cv"], math.nextafter(old["cv"], -math.inf),
                                   math.nextafter(old["cv"], math.inf)) and
                    (new["cv"] <= .02) == (old["cv"] <= .02), "lane CV超过1ULP或改变原门槛")
            if new["cv"] != old["cv"]:
                cv_differences.append(dict(case_id=old["case"], original=old["cv"], replayed=new["cv"], ulps=1))
    original_csv = root / "analysis/conditions.csv"
    replayed_csv = output / "lane-analysis/conditions.csv"
    with original_csv.open() as stream:
        original_csv_rows = list(csv.DictReader(stream))
    with replayed_csv.open() as stream:
        replayed_csv_rows = list(csv.DictReader(stream))
    require(len(original_csv_rows) == len(replayed_csv_rows) == 48, "lane条件CSV数量不符")
    for new, old in zip(replayed_csv_rows, original_csv_rows):
        require({k: v for k, v in new.items() if k != "cv"} ==
                {k: v for k, v in old.items() if k != "cv"} and
                float(new["cv"]) in (float(old["cv"]), math.nextafter(float(old["cv"]), -math.inf),
                                      math.nextafter(float(old["cv"]), math.inf)), "原lane条件CSV不符")
    pairs = read(output / "lane-analysis/layout-pairs.json")
    exact_pairs = []
    for pair in pairs:
        a, b, tolerance = pair["contiguous_cycles"], pair["interleaved_cycles"], pair["absolute_cycle_tolerance"]
        lower = max(a - tolerance, b - tolerance, Fraction(9, 10) * a, Fraction(9, 10) * b)
        upper = min(a + tolerance, b + tolerance, Fraction(11, 10) * a, Fraction(11, 10) * b)
        require((lower <= upper) == pair["common_prediction_joint_feasible"] and
                float(Fraction(abs(a - b), a + b)) == pair["minimum_common_max_relative_error"], "lane有理数区间不符")
        exact_pairs.append(dict(contiguous=pair["contiguous"], interleaved=pair["interleaved"],
                                joint_feasible=lower <= upper, minimum_common_max_absolute_error_exact=str(Fraction(abs(a-b), 2)),
                                minimum_common_max_relative_error_exact=str(Fraction(abs(a-b), a+b))))
    require(len(pairs) == 14 and sum(not p["joint_feasible"] for p in exact_pairs) == 12 and
            all(p["common_prediction_joint_feasible"] for p in pairs if p["width"] == 4), "原lane反例判定不符")
    return dict(schema="gh200-smem-lane-replay/1", status="verified", identity=identity,
                delivery_manifest_sha256=LANE_MANIFEST_SHA, bound_files_verified=len(delivery["files"]),
                conditions=48, processes=144, raw_checks=raw_checks,
                original_analysis_non_CV_fields_equal=True, CV_one_ULP_differences=cv_differences,
                original_conditions_csv_byte_equal=original_csv.read_bytes() == replayed_csv.read_bytes(),
                exact_pairs=exact_pairs, incompatible_pairs=12, compatible_4B_negative_pairs=2,
                decision=read(output / "lane-analysis/decision.json"),
                qualification="开发lane排列反例；12/14对不相容，两个4B负对照相容；原失败形式不恢复资格",
                warmup_qualification="承接原暖机状态；不把源码bar.warp.sync当作实际SASS WARPSYNC",
                new_fits=0, new_GPU_executions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    results = args.results_root.resolve(strict=True)
    output = args.output.resolve()
    for relative in (V1, V2, LANE):
        require(not output.is_relative_to(results / relative), "输出不能位于所读旧归档内")
    output.mkdir(parents=True, exist_ok=False)
    try:
        for version in (1, 2):
            save(output / f"holdout-v{version}.json", replay_holdout(results, version))
        save(output / "lane.json", replay_lane(results, output))
        artifacts = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
        save(output / "summary.json", dict(
            schema="gh200-smem-evidence-summary/1", status="verified", results_root=str(results),
            replay_source_sha256=sha(Path(__file__)), frozen_holdout_conditions=126,
            frozen_holdout_processes=378, lane_conditions=48, lane_processes=144,
            total_conditions=174, total_processes=522, both_original_holdouts_failed=True,
            incompatible_lane_pairs=12, frozen_predictions_float_equal=True,
            native_CUTLASS_runtime_cases=0, new_fits=0, new_GPU_executions=0, artifacts=artifacts,
            endpoint="固定旧证据回放完成；两冻结失败及lane反例保留，不接入合格SMEM通用服务"))
    except Exception as error:
        save(output / "failure.json", dict(status="failed", error=str(error)))
        raise
    print(f"已回放174条件、522旧进程：{output / 'summary.json'}")


if __name__ == "__main__":
    main()

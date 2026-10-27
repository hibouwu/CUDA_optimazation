#!/usr/bin/env python3
"""用原12开发点冻结L1条件服务；原8留出只评分，不重拟合。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics as st
import sys

SEEDS = (3, 22, 41)
TIME_LIMIT = .05
RESPONSE_LIMIT = .05


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def runner_at(root):
    sys.path.insert(0, str(root/'source'))
    spec = importlib.util.spec_from_file_location('q13_original_l1_runner', root/'source/l1_lsu_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solve3(matrix, values):
    rows = [list(row)+[value] for row, value in zip(matrix, values)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda i: abs(rows[i][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        require(rows[column][column] != 0, '开发参数不可辨识')
        divisor = rows[column][column]
        rows[column] = [value/divisor for value in rows[column]]
        for index in range(3):
            if index != column:
                factor = rows[index][column]
                rows[index] = [a-factor*b for a, b in zip(rows[index], rows[column])]
    return [row[-1] for row in rows]


def fit_service(points):
    """每J三参数，最小化Σ(predicted/observed−1)^2；全部12点参加。"""
    parameters = {}
    for chains in (1, 4):
        matrix, values = [[0.]*3 for _ in range(3)], [0.]*3
        for point in points:
            case, observed = point['case'], point['median_cycle']
            if case['chains'] != chains:
                continue
            groups = case['rounds']*case['work']['groups_per_sweep']
            fast = point['service_class'] == 'fast_reuse'
            vector = [1., groups if fast else 0., groups if not fast else 0.]
            for i in range(3):
                values[i] += vector[i]/observed
                for j in range(3):
                    matrix[i][j] += vector[i]*vector[j]/observed**2
        fixed, fast, slow = solve3(matrix, values)
        require(0 <= fixed and 0 < fast < slow, '非负固定项/快慢服务约束未通过')
        parameters[str(chains)] = dict(fixed_cycle=fixed, fast_group_cycle=fast, slow_group_cycle=slow)
    return parameters


def time_prediction(case, parameters, fast):
    parameter = parameters[str(case['chains'])]
    groups = case['rounds']*case['work']['groups_per_sweep']
    return parameter['fixed_cycle']+groups*parameter['fast_group_cycle' if fast else 'slow_group_cycle']


def response_pairs():
    pairs = []
    for hint in (0, 100):
        for chains in (1, 4):
            pairs.append((f'cache_h{hint}_j{chains}', f'w128_h{hint}_cg_j{chains}', f'w128_h{hint}_ca_j{chains}'))
    for policy in ('ca', 'cg'):
        for chains in (1, 4):
            pairs.append((f'hint_{policy}_j{chains}', f'w128_h100_{policy}_j{chains}', f'w128_h0_{policy}_j{chains}'))
    for hint in (0, 100):
        for policy in ('ca', 'cg'):
            pairs.append((f'chains_h{hint}_{policy}', f'w128_h{hint}_{policy}_j1', f'w128_h{hint}_{policy}_j4'))
    return pairs


def evaluate(medians, frozen):
    require(set(medians) == {case['id'] for case in frozen['heldout_cases']}, '须评分原8个条件，不删条件')
    outcomes = {}
    for name, candidate in frozen['candidates'].items():
        cases = {identifier: dict(predicted_cycle=prediction, observed_cycle=medians[identifier],
                    relative_error=prediction/medians[identifier]-1,
                    passed=abs(prediction/medians[identifier]-1) <= TIME_LIMIT)
                 for identifier, prediction in candidate['predicted_cycle'].items()}
        responses = {}
        for identifier, baseline, variant in response_pairs():
            predicted = candidate['predicted_cycle'][variant]/candidate['predicted_cycle'][baseline]-1
            observed = medians[variant]/medians[baseline]-1
            responses[identifier] = dict(baseline=baseline, variant=variant,
                predicted=predicted, observed=observed, absolute_response_error=abs(observed-predicted),
                passed=abs(observed-predicted) <= RESPONSE_LIMIT)
        outcomes[name] = dict(cases=cases, responses=responses,
            passed=all(row['passed'] for row in cases.values()) and all(row['passed'] for row in responses.values()))
    survivors = [name for name, value in outcomes.items() if value['passed']]
    return dict(candidates=outcomes, survivors=survivors,
        conclusion='限定条件响应支持' if len(survivors) == 1 else '信息不足' if survivors else '两个具体形式被反证',
        physical_L1_capacity_identified=False, instruction_latency_identified=False, full_GEMM_C2=False, C3_tested=False)


def development(root, independent_score):
    runner = runner_at(root)
    plan = runner.load_plan(root)
    runner.verify_build(root)
    summary, score = read(root/'dev/summary.json'), read(independent_score)
    require(summary['complete'] and summary['independent_processes'] == 36, '开发36进程未完成')
    require(score['summary_sha256'] == sha(root/'dev/summary.json') and score['all_performance_eligible']
            and score['target_static_review_exact_match'], '独立开发评分身份/资格不符')
    require(summary['target_review_sha256'] == sha(root/'build/target-review.json')
            and read(root/'build/target-review.json')['passed'], '开发目标静态资格不符')
    cases = {case['id']: case for case in plan['cases'] if case['split'] == 'dev'}
    require(len(cases) == 12, '开发条件不符')
    groups = {name: [] for name in cases}
    evidence = {str(path): sha(path) for path in (root/'plan.json', root/'dev/summary.json',
        root/'build/identity.json', root/'build/target-review.json', independent_score)}
    for row in summary['results']:
        case = cases[row['case_id']]
        folder = root/row['path']
        require(sha(folder/'validated.json') == row['validated_sha256'], '开发validated身份不符')
        actual = runner.validate(read(folder/'stdout'), case, row['seed'])
        require(all(actual[key] == row[key] for key in actual), '开发原始观测与摘要不符')
        require(actual['device_uuid'] == summary['device_uuid'], '开发设备不符')
        evidence[str(folder/'stdout')] = sha(folder/'stdout')
        evidence[str(folder/'validated.json')] = sha(folder/'validated.json')
        groups[case['id']].append(row)
    points = []
    for name, case in cases.items():
        rows = groups[name]
        require(len(rows) == 3 and {row['seed'] for row in rows} == set(SEEDS), '开发seed不齐')
        cycles = [row['elapsed_cycle'] for row in rows]
        fast = case['policy'] == 'ca' and (case['working_set_kib'] == 16 or case['requested_carveout_percent'] == 0)
        points.append(dict(case=case, service_class='fast_reuse' if fast else 'streaming',
            samples_cycle=cycles, median_cycle=st.median(cycles), range_cycle=[min(cycles), max(cycles)]))
    return plan, summary, points, evidence


def cpu_checks(frozen):
    known = []
    for chains in (1, 4):
        for width, fast in ((16, True), (64, True), (16, False), (64, False), (256, False)):
            case = dict(chains=chains, rounds=256, work=dict(groups_per_sweep=width//chains))
            # Synthetic independent coordinates: T=7+G*(11 or 29).
            known.append(dict(case=case, median_cycle=7+256*width/chains*(11 if fast else 29),
                              service_class='fast_reuse' if fast else 'streaming'))
    fitted = fit_service(known)
    require(all(abs(value-target) < 1e-8 for p in fitted.values()
        for value, target in zip(p.values(), (7, 11, 29))), '解析参数恢复失败')
    scaled = fit_service([{**point, 'median_cycle':point['median_cycle']*3} for point in known])
    require(all(abs(scaled[j][key]/value-3) < 1e-8 for j, p in fitted.items() for key, value in p.items()),
            '相对最小二乘缩放核对失败')
    for name, candidate in frozen['candidates'].items():
        result = evaluate(candidate['predicted_cycle'], frozen)
        require(result['survivors'] == [name], '两候选未能按冻结判据区分')
    middle = dict(frozen['candidates']['reuse_at_128']['predicted_cycle'])
    other = frozen['candidates']['overflow_at_128']['predicted_cycle']
    for name in middle:
        middle[name] = (middle[name]+other[name])/2
    require(not evaluate(middle, frozen)['survivors'], '中间响应不应被强选候选')
    return dict(passed=True, checks=5, actual_GPU_samples=0,
        items=['relative_LS_analytic_recovery', 'relative_LS_scale', 'reuse_separates_overflow',
               'overflow_separates_reuse', 'intermediate_rejects_both'])


def prepare(development_root, independent_score, output):
    original, summary, points, evidence = development(development_root, independent_score)
    parameters = fit_service(points)
    for point in points:
        prediction = time_prediction(point['case'], parameters, point['service_class'] == 'fast_reuse')
        point.update(predicted_cycle=prediction, relative_error=prediction/point['median_cycle']-1)
    require(max(abs(point['relative_error']) for point in points) <= TIME_LIMIT, '共用服务结构开发失败，不冻结留出')
    heldout = [case for case in original['cases'] if case['split'] == 'heldout']
    require(len(heldout) == 8, '不是原8留出条件')
    frozen = dict(schema='gh200-q13-conditional-service-freeze/1', frozen=True,
        primary_metric='clock64 formal rounds256, same CTA', secondary='globaltimer and whole-kernel event: descriptive, not fitted',
        parameters=parameters, heldout_cases=heldout, development_summary_sha256=sha(development_root/'dev/summary.json'),
        development_score_sha256=sha(independent_score), device_uuid=summary['device_uuid'], candidates={})
    for name, reuse in (('reuse_at_128', True), ('overflow_at_128', False)):
        predictions = {case['id']:time_prediction(case, parameters,
            reuse and case['policy'] == 'ca' and case['requested_carveout_percent'] == 0) for case in heldout}
        frozen['candidates'][name] = dict(predicted_cycle=predictions,
            observed_cycle_acceptance={key:[value/(1+TIME_LIMIT), value/(1-TIME_LIMIT)] for key,value in predictions.items()},
            response_predictions={key:predictions[variant]/predictions[baseline]-1 for key,baseline,variant in response_pairs()},
            effective_reuse_boundary_hint0='at least 128 KiB' if reuse else 'at least 64 and below 128 KiB',
            physical_partition_capacity_claim=False)
    frozen['criteria'] = dict(all_eight_median_absolute_relative_errors_le=TIME_LIMIT,
        all_twelve_response_absolute_errors_le=RESPONSE_LIMIT,
        threshold_origin='This task freezes numeric heldout criteria before sampling; the earlier 5% descriptive development screen was not a numeric C2 test.',
        relative_error_definition='predicted/observed-1; response=variant_median/baseline_median-1',
        selection='Evaluate both frozen forms; one passes and other fails => conditional response support. No minimum-error selection or refit.',
        qualification='all 24 original heldout processes, exact source/build/GPU identity and original checksum/resource gates; no deleted conditions',
        uncertainties='Three-process ranges descriptive, no IID/confidence or actual-carveout claim; actual_carveout remains null.')
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    source.mkdir()
    shutil.copy2(Path(__file__), source/Path(__file__).name)
    execution = output/'execution'
    execution.mkdir()
    for relative in ('plan.json', 'device-binding.json', 'dev/summary.json',
        *[f'source/{name}' for name in original['source_sha256']], 'build/identity.json', 'build/probe',
        'build/target-review.json', 'build/disassemble/stdout', 'build/version/stdout',
        'build/compile/stdout', 'build/compile/stderr'):
        target = execution/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(development_root/relative, target)
    save(output/'development-inputs.json', dict(files=evidence, source=str(development_root), independent_score=str(independent_score)))
    save(output/'fit.json', dict(method='relative least squares on all 12 process medians', parameters=parameters,
        development_points=points, max_absolute_relative_error=max(abs(p['relative_error']) for p in points),
        calibration_only=True, heldout_read=False, physical_instruction_prices=False))
    save(output/'predictions.json', frozen)
    save(output/'plan.json', dict(schema='gh200-q13-minimal-plan/1', date='2026-10-17 Europe/Paris',
        status='CPU freeze prepared; no heldout/GPU execution', capability='C2: conditional reuse service response at W128, not physical capacity',
        work='1 CTA,128 threads,16B pointer nodes; J1/J4 independent software chains,256 complete W sweeps,one warm sweep per launch',
        model='T_cycle=a_J+G(W,J)*tau_(class,J); G=256*W/(16*128*J). Fixed item includes fixed-round control/barriers; group service includes issue/return/address/checksum.',
        classes='ca W16/h100 and W64/h0 fast; other 12-development points slow. h100 W128 slow; h0 ca W128 fast or slow frozen alternatives.',
        fixed='Original probe/runner, all three short/formal checks, seeds3/22/41, original device/build and original eight heldout conditions.',
        original_node='romeo-a057; UUID is authoritative',
        limits=dict(candidates=2, conditions=8, processes=24, GPU_allocation_seconds_max=3600, runner_sampling_seconds_max=600, implementation_repairs_max=2),
        stop='Qualification failure stops; completed24 => score both fixed forms. No extra W/J/hint/seed, fit or repair of scientific form after heldout.',
        missing=['h0/J1 fast slope inferred through common fixed-term assumption, no second fast J1 development width',
                 'No actual carveout/physical cache residency/traffic evidence; no new J distinguishes latency versus throughput decomposition'],
        full_GEMM_transfer='cfg_b ordinary global/local demands are zero; no TMA price or fullGEMM C2/C3 replacement'))
    save(output/'cpu-validation.json', cpu_checks(frozen))
    readme = f'''# Q13：原8留出的最小条件服务验证准备

2026-10-17。只用原12开发条件36进程定参；本包未运行GPU、未读取留出。原探针和工作量不变。

`T=a_J+G*tau_class,J`，`G=256W/(16*128*J)`。每种J共用固定项和快/慢两条斜率，最小化全部开发中位数的相对平方误差。最大开发误差{100*max(abs(p['relative_error']) for p in points):.6f}%。这些是整个依赖/发出/消费循环的条件服务，不是单load价格。

冻结两种响应：`reuse_at_128`在hint0的128KiB `.ca`保留快复用；`overflow_at_128`进入慢分支。所有`.cg`及hint100均慢。开发仅约束hint100的有效边界在16至64KiB之间、hint0至少覆盖64KiB；两候选不指定未测的物理分区容量。J1/J4分别有原22/32寄存器及不同循环组织，均按源实际独立链数保存。

原8条件各3进程，共24；全部8个clock64中位绝对相对误差≤5%，全部12个预列响应误差≤5个百分点才通过一个具体形式。这是本轮采样前冻结的数值判据；旧开发筛查的5%描述门槛没有因此改称数值C2。两个形式均按固定参数评分，一项通过且另一项失败只支持此程序/工作集/请求hint下的条件响应。两者失败停止形式，均通过或观测不足记信息不足；不按最小误差重选/重拟合。globaltimer与event单列，三进程范围不是置信界。

## 执行与评分

主线程在原GPU `{summary['device_uuid']}`的串行免费ROMEO分配中执行，原节点romeo-a057（以UUID绑定为准）。Slurm分配≤1h，原runner采样≤600秒，两次实现修复上限。先绑定本包freeze，再采原8条件；无新增先导。

```sh
python -B PACKAGE/execution/source/l1_lsu_probe.py run --output PACKAGE/execution --split heldout --allow-gpu --frozen-predictions PACKAGE/predictions.json
python -B PACKAGE/source/q13_l1_minimal.py score --package PACKAGE --output NEW_SCORE_DIRECTORY
```

`PACKAGE`替换为本包绝对路径或传输后的路径。execution只复制原plan/source/build、设备绑定及开发摘要，不会向旧结果写入。binary为已审查的原CUDA12.9 ARM目标，源码/机器码与资源必须保持；更换影响科学条件的实现不能沿用此冻结。

CPU冻结复建：`python -B SOURCE/q13_l1_minimal.py prepare --development OLD_EXECUTION --development-score OLD_SCORE_JSON --output NEW_PACKAGE`。本包`development-inputs.json`保存开发原始文件SHA，`freeze.json`绑定预测、参数、源码和执行静态输入。后续评分不再读取开发观测或改参数。

## 未辨识范围

J1快分支只有一个开发W，固定项沿.ca/.cg同体共用是可检验假设；没有新的J不能把group服务分成硬件latency与throughput。actual_carveout仍null；本题不辨识物理容量、流量或通用load价格。cfg_b普通global/local需求为0，不把此价格接到其TMA。没有新全GEMM C2/C3验收。
'''
    (output/'README.md').write_text(readme)
    save(output/'freeze.json', dict(schema='gh200-q13-package-freeze/1',
        files={str(path.relative_to(output)):sha(path) for path in sorted(output.rglob('*')) if path.is_file()},
        GPU_executions=0, heldout_observations_read=False))
    print(json.dumps(dict(package=str(output), parameters=parameters,
        max_development_error_pct=100*max(abs(p['relative_error']) for p in points), predictions_sha256=sha(output/'predictions.json')),
        ensure_ascii=False))


def score(package, output):
    for name, expected in read(package/'freeze.json')['files'].items():
        require(sha(package/name) == expected, '冻结输入变化：'+name)
    frozen, root = read(package/'predictions.json'), package/'execution'
    runner = runner_at(root)
    runner.load_plan(root)
    runner.verify_build(root)
    summary = read(root/'heldout/summary.json')
    require(summary['complete'] and summary['split'] == 'heldout' and summary['independent_processes'] == 24,
            '原24留出尚未完成')
    require(summary['frozen_predictions']['sha256'] == sha(package/'predictions.json')
        and summary['frozen_predictions']['development_summary_sha256'] == frozen['development_summary_sha256']
        and summary['plan_sha256'] == sha(root/'plan.json')
        and summary['build_identity_sha256'] == sha(root/'build/identity.json')
        and summary['device_uuid'] == frozen['device_uuid'], '留出冻结/构建/设备身份不符')
    cases = {case['id']:case for case in frozen['heldout_cases']}
    groups = {name:[] for name in cases}
    rows = []
    review = read(root/'build/target-review.json')
    for item in summary['results']:
        case = cases[item['case_id']]
        folder = root/item['path']
        require(sha(folder/'validated.json') == item['validated_sha256'], '留出validated身份变化')
        raw = read(folder/'stdout')
        actual = runner.validate(raw, case, item['seed'])
        resources = review['functions'][f"j{case['chains']}_{case['policy']}"]['resources']
        require(all(raw['resources'][key] == resources[key] for key in
            ('registers_per_thread', 'static_smem_bytes', 'dynamic_smem_bytes', 'local_size_bytes', 'threads', 'grid_ctas')),
            '留出资源与原静态身份不符')
        require(all(actual[key] == item[key] for key in actual) and actual['device_uuid'] == frozen['device_uuid'],
            '留出原始观测与摘要不符')
        rows.append(dict(case_id=case['id'], seed=item['seed'], raw_sha256=sha(folder/'stdout'), **actual))
        groups[case['id']].append(item)
    require(len(rows) == 24 and all(len(values) == 3 and {r['seed'] for r in values} == set(SEEDS)
            for values in groups.values()), '条件/seed重复或缺失')
    medians = {name:st.median(row['elapsed_cycle'] for row in values) for name, values in groups.items()}
    result = evaluate(medians, frozen)
    result.update(schema='gh200-q13-heldout-score/1', predictions_sha256=sha(package/'predictions.json'),
        all_24_qualified=True, points={name:{key:dict(median=st.median(row[key] for row in values),
            minimum=min(row[key] for row in values), maximum=max(row[key] for row in values))
            for key in ('elapsed_cycle','elapsed_ns','event_ms')} for name,values in groups.items()},
        raw=rows, refit_performed=False, new_GPU_executions=0)
    output.mkdir(parents=True, exist_ok=False)
    save(output/'score.json', result)
    print(result['conclusion'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest='step', required=True)
    prepare_parser = steps.add_parser('prepare')
    prepare_parser.add_argument('--development', type=Path, required=True)
    prepare_parser.add_argument('--development-score', type=Path, required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    score_parser = steps.add_parser('score')
    score_parser.add_argument('--package', type=Path, required=True)
    score_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.step == 'prepare':
        prepare(args.development.resolve(), args.development_score.resolve(), args.output.resolve())
    else:
        score(args.package.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()

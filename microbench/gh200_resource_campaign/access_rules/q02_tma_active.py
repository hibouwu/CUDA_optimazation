#!/usr/bin/env python3
"""固定 Q4 暖机，判别正式 Q1/Q4 下的两种行距响应连接。"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import statistics
import time

import q02_tma_single_cta as single
import q02_tma_stride as stride
from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json

HERE = Path(__file__).resolve().parent
SCHEMA = 'gh200-q02-tma-active-execution/1'


def prepare(root, baseline, reference):
    prior = read_json(reference/'score.json')
    require(next(row['median_delta_cycles'] for row in prior['lengths'] if row['groups'] == 32) == 295,
            'Q4/G32 的校准增量不符')
    single.prepare(root, baseline)
    initial = read_json(root/'plan.json')
    shutil.move(root/'build', root/'baseline/build')
    (root/'source-before').mkdir()
    shutil.copy2(root/'source/tma_gemm_sources.cu', root/'source-before/tma_gemm_sources.cu')
    shutil.copy2(HERE/'probes/tma_gemm_sources.cu', root/'source/tma_gemm_sources.cu')
    shutil.copy2(baseline/'source/cutlass-3.9.2-include.tar.gz', root/'source/cutlass-3.9.2-include.tar.gz')
    for name in ('q02_tma_active.py', 'test_q02_tma_active.py'):
        shutil.copy2(HERE/name, root/'source'/name)
    (root/'reference').mkdir()
    for name in ('plan.json', 'score.json', 'hypotheses.json', 'commands.json', 'correctness_commands.json', 'worklist.txt'):
        shutil.copy2(reference/name, root/'reference'/name)
    shutil.copytree(reference/'runs', root/'reference/runs')
    shutil.copytree(reference/'source', root/'reference/source')
    (root/'reference/build').mkdir()
    shutil.copy2(reference/'build/target-review.json', root/'reference/build/target-review.json')
    for filename in ('correctness_commands.json', 'commands.json'):
        tasks = []
        for active in (1, 4):
            for original in read_json(root/filename):
                task = copy.deepcopy(original)
                task['id'] = f"measure-q{active}-"+task['id']
                task['active_groups'] = active
                task['groups'] = 32
                task['argv'] += ['--measure-groups', '32', '--measure-active', str(active)]
                if filename == 'correctness_commands.json':
                    task['expected_checked'] = 32*16384
                tasks.append(task)
        (root/filename).unlink()
        save_json(root/filename, tasks)
    (root/'hypotheses.json').unlink()
    save_json(root/'hypotheses.json', [
        dict(id='fixed-boundary', rule='一次窗口成本在正式 Q1/Q4 下均保持 295 cycle',
             delta_cycles={'1':295.0, '4':295.0}),
        dict(id='per-group-ready', rule='每组 full-ready 晚 295 cycle；Q1 的32组完全暴露，Q4稳态隐藏后只留一次边界',
             delta_cycles={'1':9440.0, '4':295.0})])
    (root/'README.md').write_text('''# Q02：固定暖机的正式 Q1/Q4 判别

M/N2304、K4096、A lda4096、allocA4096、allocB2360、ring64及物理4槽不变。每臂先用Q4/G64交替坐标(0,0)／(7,6)暖机10次，再正式测(7,6)的G32；正式Q1／Q4各配B行距2304／2344。seed3正序、22／41反序，先6个verify进程再6个plain进程，共24臂；verify每臂524288值。所有臂须在参考卡SM124，失败保存并停止，不重采。

Q4两条件已测，本轮作为同批控制与回归；新Q1两条件检验采前冻结连接。295 cycle来自Q4完整窗口差，尚未独立辨识为单请求价格。两具体形式由hypotheses.json维护：一次成本预测Q1/Q4均295；逐组full-ready形式还假设Q1完全暴露、Q4稳态隐藏，预测9440／295。每Q三seed增量中位的幅度误差≤5%，全部配对增量为正；四条件的cycle与ns窗口max−min≤各自中位5%。

--active 4控制暖机；host参数--measure-active仅在暖机后改正式实参，每个配对臂重置为Q4暖机，raw登记warmup_active_groups。GPU kernel保持原文。目标CUDA12.9须编译before/after并核kernel全部SASS机器编码、资源与原Q02构建一致，才执行资格测量；本次CPU准备未编译。预算10分钟、最多两次定向修复，不扫描第三Q值、长度或内部端点。

```sh
python source/q02_tma_active.py execute --root "$PWD"
python source/q02_tma_active.py score --root "$PWD" --output /tmp/q02-active-score.json
```

局部新Q1证据和旧Q4控制分列；只检验这两个明确形式，完整GEMM的C2/C3保持未验收。原长度结果在reference，原132CTA数据及构建在baseline。
''')
    (root/'plan.json').unlink()
    frozen = {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
    plan = dict(initial, schema=SCHEMA, qualification='development_localization', independent_C2=False,
        development_conditions=2, heldout_conditions=2, heldout_enabled=True,
        controls=dict(active_groups=4, conditions=2, prior=reference.name),
        new_intervention=dict(active_groups=1, conditions=2),
        correctness_processes=6, formal_processes=6, correctness_arm_checks=12, formal_arm_observations=12,
        active_groups=[1,4], groups=32, warmup_active_groups=4, warmup_groups=64, required_smid=124,
        reference=reference.name, reference_plan_sha256=digest(reference/'plan.json'),
        reference_score_sha256=digest(reference/'score.json'), numerical_response_prediction='hypotheses.json',
        prediction_relative_tolerance=0.05, window_relative_spread=0.05, all_pair_deltas_positive=True,
        full_gemm_C2=False, full_gemm_C3=False, frozen_at_ns=time.time_ns(), files_sha256=frozen)
    save_json(root/'plan.json', plan)
    save_json(root/'delivery-manifest.json', dict(schema='gh200-q02-tma-active-delivery/1',
        files={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}))
    print('PREPARED: Q4 controls + new Q1 intervention; 6 verify + 6 plain processes; no compilation/GPU')


def load_plan(root):
    plan = read_json(root/'plan.json')
    require(plan['schema'] == SCHEMA and plan['heldout_enabled'] is True, '执行包范围不符')
    for name, checksum in plan['files_sha256'].items():
        require(digest(root/name) == checksum, '冻结身份变更：'+name)
    require(digest(Path(__file__)) == plan['files_sha256']['source/q02_tma_active.py'], 'runner 身份不符')
    return plan


def target_review(root):
    report = stride.build_review(root)
    original = read_json(root/'baseline/target-review.json')['rows'][1]
    require(report['passed'] and all(row['instruction_words_sha256'] == original['instruction_words_sha256'] and
        row['resources'] == original['resources'] for row in report['rows']), 'kernel SASS/资源与原Q02不符')
    return report


def validate_pair(root, folder, task, uuid):
    rows = single.validate_pair(root, folder, task, uuid)
    require(all(row['ctas'][0]['smid'] == 124 for row in rows), '单 CTA 未落在冻结 SM124，停止且不重采')
    return rows


def summarize(root):
    plan = load_plan(root)
    require(target_review(root) == read_json(root/'build/target-review.json'), '目标构建证据不符')
    require(read_json(root/'allocation/identity.json')['uuid'] == plan['gpu_uuid'], '分配卡身份不符')
    numeric, formal = [], []
    for name, split, rows in (('correctness_commands.json','correctness',numeric), ('commands.json','formal',formal)):
        for task in read_json(root/name):
            rows.extend(validate_pair(root, root/'runs'/split/task['id'], task, plan['gpu_uuid']))
    by_numeric = {(r['active_groups'],r['seed'],r['ldb']):r for r in numeric}
    by_formal = {(r['active_groups'],r['seed'],r['ldb']):r for r in formal}
    require(len(by_numeric) == len(by_formal) == 12 and by_numeric.keys() == by_formal.keys(), '数值资格与正式臂不对应')
    pairs, conditions, responses = [], [], []
    for active in plan['active_groups']:
        deltas = []
        role = 'new_intervention' if active == 1 else 'control'
        for seed in plan['seeds']:
            a,b = [by_formal[(active,seed,ldb)] for ldb in (2304,2344)]
            for row in (a,b):
                row['numeric_source'] = by_numeric[(active,seed,row['ldb'])]['source']
            metrics = {name:dict(difference=b['metrics'][name]-value, response=b['metrics'][name]/value-1)
                       for name,value in a['metrics'].items() if name not in ('cycles_spread','ns_spread')}
            deltas.append(metrics['cycles_median']['difference'])
            pairs.append(dict(active_groups=active,role=role,seed=seed,base_ldb=2304,changed_ldb=2344,metrics=metrics))
        responses.append(dict(active_groups=active,role=role,delta_cycles=deltas,median_delta_cycles=statistics.median(deltas)))
        for ldb in (2304,2344):
            windows = {}
            for metric in ('cycles_median','ns_median'):
                values = [by_formal[(active,seed,ldb)]['metrics'][metric] for seed in plan['seeds']]
                median,spread = statistics.median(values), max(values)-min(values)
                windows[metric] = dict(values=values,median=median,spread=spread,relative_spread=spread/median,
                                       passed=spread<=plan['window_relative_spread']*median)
            conditions.append(dict(active_groups=active,role=role,ldb=ldb,windows=windows,
                                   passed=all(value['passed'] for value in windows.values())))
    stable = all(condition['passed'] for condition in conditions)
    positive = all(delta>0 for response in responses for delta in response['delta_cycles'])
    candidates = []
    for hypothesis in read_json(root/'hypotheses.json'):
        predictions = []
        for response in responses:
            expected = hypothesis['delta_cycles'][str(response['active_groups'])]
            observed = response['median_delta_cycles']
            error = abs(observed-expected)/expected
            predictions.append(dict(active_groups=response['active_groups'],role=response['role'],
                predicted_delta_cycles=expected,observed_median_delta_cycles=observed,relative_error=error,
                passed=error<=plan['prediction_relative_tolerance']))
        passed = stable and positive and all(prediction['passed'] for prediction in predictions)
        candidates.append(dict(id=hypothesis['id'],predictions=predictions,passed=passed,
            conclusion='信息不足' if not stable else '获得支持' if passed else '被反证'))
    supported = [candidate['id'] for candidate in candidates if candidate['passed']]
    return dict(schema='gh200-q02-tma-active-score/1',qualification=plan['qualification'],
        independent_C2=stable and positive and len(supported)==1,full_gemm_C2=False,full_gemm_C3=False,
        numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),numeric=numeric,formal=formal,
        pairs=pairs,responses=responses,conditions=conditions,stable_windows=stable,
        all_pair_deltas_positive=positive,candidates=candidates,supported_candidates=supported,
        development_conditions=2,heldout_conditions=2,plan_sha256=digest(root/'plan.json'),probe_sha256=digest(root/'build/probe'))


def execute(root):
    plan = load_plan(root)
    start = time.monotonic()
    environment = inspect_allocated_gpu(root/'allocation')
    require(environment['uuid'] == plan['gpu_uuid'], '分配卡与冻结研究对象不符')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        stride.build(root,start)
        require(target_review(root) == read_json(root/'build/target-review.json'), '原生构建核对不符')
        for name,split in (('correctness_commands.json','correctness'), ('commands.json','formal')):
            for task in read_json(root/name):
                folder = root/'runs'/split/task['id']
                command_record(folder,stride.probe_command(root,task),stride.remaining(start,60))
                rows = stride.validate_process(root,folder,task,environment['uuid'])
                save_json(folder/'validated.json',rows)
                smids = [row['ctas'][0]['smid'] for row in rows]
                save_json(folder/'pair-gate.json',dict(same_sm=smids[0]==smids[1],smids=smids,
                                                     required_smid=124,passed=all(smid==124 for smid in smids)))
                require(all(smid==124 for smid in smids), '单 CTA 未落在冻结 SM124，停止且不重采')
                print(json.dumps(dict(split=split,process=task['id'],arms=len(rows))),flush=True)
        save_json(root/'score.json',summarize(root))
    print('CLOSED: Q4 controls + new Q1 intervention; fixed connections scored',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','execute','score'):
        item = sub.add_parser(name)
        item.add_argument('--root',type=Path,required=True)
        if name == 'prepare':
            item.add_argument('--baseline',type=Path,required=True)
            item.add_argument('--reference',type=Path,required=True)
        if name == 'score':
            item.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'prepare':
        prepare(root,args.baseline.resolve(),args.reference.resolve())
    elif args.command == 'execute':
        execute(root)
    else:
        save_json(args.output,summarize(root))
        print('VERIFIED: fixed-warmup Q1/Q4 paired responses')


if __name__ == '__main__':
    main()

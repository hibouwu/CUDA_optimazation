#!/usr/bin/env python3
"""复用 Q02 二进制，冻结判别单 CTA 行距增量的长度规则。"""
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
SCHEMA = 'gh200-q02-tma-length-execution/1'


def prepare(root, baseline, reference):
    reference_plan = single.load_plan(reference)
    reference_score = read_json(reference/'score.json')
    require(statistics.median(row['metrics']['cycles_median']['difference']
                             for row in reference_score['pairs']) == 295, '64 组增量依据不符')
    single.prepare(root, baseline)
    initial = read_json(root/'plan.json')
    for name in ('q02_tma_length.py', 'test_q02_tma_length.py'):
        shutil.copy2(HERE/name, root/'source'/name)
    (root/'reference').mkdir()
    for name in ('plan.json', 'score.json', 'commands.json', 'correctness_commands.json', 'worklist.txt'):
        shutil.copy2(reference/name, root/'reference'/name)
    shutil.copytree(reference/'runs', root/'reference/runs')
    for filename in ('correctness_commands.json', 'commands.json'):
        tasks = []
        for groups in (16, 32):
            for original in read_json(root/filename):
                task = copy.deepcopy(original)
                task['id'] = f"g{groups}-"+task['id']
                task['groups'] = groups
                task['argv'] += ['--measure-groups', str(groups)]
                if filename == 'correctness_commands.json':
                    task['expected_checked'] = groups*16384
                tasks.append(task)
        (root/filename).unlink()
        save_json(root/filename, tasks)
    (root/'hypotheses.json').unlink()
    save_json(root/'hypotheses.json', [
        dict(id='fixed-boundary', rule='B80 增量由固定边界贡献，随测量组数保持不变',
             delta_cycles={'16':295.0, '32':295.0}),
        dict(id='per-group', rule='B80 增量按测量组数累加，64 组增量为 295 cycle',
             delta_cycles={'16':73.75, '32':147.5})])
    (root/'README.md').write_text('''# Q02：单 CTA 长度冻结判别

复用 Q02-stride-002 的 CUDA12.9 二进制与30份构建证据，不编译。M/N2304、K4096，A lda4096、allocA4096、allocB2360；Q4、物理4槽、ring64，工作坐标(0,0)／(7,6)，正式测第二轮。每臂固定10次交替暖机、每次64组；仅以 --measure-groups 16／32 改变正式测量长度。两臂实际B行距2304／2344，共用固定分配。

4个未采条件为两长度×两行距。每长度seed3正序、22／41反序，先6个verify进程（12臂）再6个plain进程（12臂）。16／32组的每verify臂分别检查262144／524288个加载值。所有verify与plain臂须在原SM124；失败保存原始输出后停止，不重采。预算10分钟，最多2次定向修复。

64组已测配对增量294／295／295 cycle，中位295；原始依据在reference/。采前冻结：固定边界模型对16／32组预测295／295 cycle；逐组模型预测73.75／147.5 cycle。每长度3个配对增量的中位须在预测±5%，全部配对增量须为正；每个条件的cycle与ns窗口max−min须≤中位5%。hypotheses.json维护冻结预测，plan.json绑定输入与源码，delivery-manifest.json列准备包身份。

```sh
python source/q02_tma_length.py execute --root "$PWD"
python source/q02_tma_length.py score --root "$PWD" --output /tmp/q02-length-score.json
python source/test_q02_tma_length.py --root "$PWD" --reference ../20261018-Q02-tma-single-cta-001
```

本轮仅判别单CTA匹配驻留条件下的长度规则，未定位具体硬件单元，未进行完整GEMM验收。原132CTA回放证据在baseline/replay，64组单CTA证据在reference/。
''')
    (root/'plan.json').unlink()
    (root/'build/target-review.json').unlink()
    frozen = {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
    plan = dict(initial, schema=SCHEMA, qualification='frozen_length_discrimination', independent_C2=False,
        development_conditions=0, heldout_conditions=4, heldout_enabled=True,
        correctness_processes=6, formal_processes=6, correctness_arm_checks=12, formal_arm_observations=12,
        measure_groups=[16,32], warmup_groups=64, required_smid=124,
        reference=reference.name, reference_plan_sha256=digest(reference/'plan.json'),
        reference_probe_sha256=reference_plan['original_probe_sha256'],
        numerical_response_prediction='hypotheses.json', prediction_relative_tolerance=0.05,
        window_relative_spread=0.05, all_pair_deltas_positive=True,
        frozen_at_ns=time.time_ns(), files_sha256=frozen)
    save_json(root/'plan.json', plan)
    save_json(root/'build/target-review.json', stride.build_review(root))
    save_json(root/'delivery-manifest.json', dict(schema='gh200-q02-tma-length-delivery/1',
        files={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}))
    print('PREPARED: frozen 4 length conditions; 6 verify + 6 plain processes; no GPU')


def load_plan(root):
    plan = read_json(root/'plan.json')
    require(plan['schema'] == SCHEMA and plan['heldout_enabled'] is True, '执行包范围不符')
    for name, checksum in plan['files_sha256'].items():
        require(digest(root/name) == checksum, '冻结身份变更：'+name)
    require(digest(Path(__file__)) == plan['files_sha256']['source/q02_tma_length.py'], 'runner 身份不符')
    review = stride.build_review(root)
    require(review == read_json(root/'build/target-review.json') and review['passed'], '原构建证据不符')
    return plan


def validate_pair(root, folder, task, uuid):
    rows = single.validate_pair(root, folder, task, uuid)
    require(all(row['ctas'][0]['smid'] == 124 for row in rows), '单 CTA 未落在冻结 SM124，停止且不重采')
    for row in rows:
        row['groups'] = task['groups']
    return rows


def summarize(root):
    plan = load_plan(root)
    require(read_json(root/'allocation/identity.json')['uuid'] == plan['gpu_uuid'], '分配卡身份不符')
    numeric, formal = [], []
    for name, split, rows in (('correctness_commands.json', 'correctness', numeric), ('commands.json', 'formal', formal)):
        for task in read_json(root/name):
            rows.extend(validate_pair(root, root/'runs'/split/task['id'], task, plan['gpu_uuid']))
    by_numeric = {(row['groups'],row['seed'],row['ldb']):row for row in numeric}
    by_formal = {(row['groups'],row['seed'],row['ldb']):row for row in formal}
    require(len(by_numeric) == len(by_formal) == 12 and by_numeric.keys() == by_formal.keys(), '数值资格与正式臂不对应')
    pairs, conditions, lengths = [], [], []
    for groups in plan['measure_groups']:
        deltas = []
        for seed in plan['seeds']:
            a,b = [by_formal[(groups,seed,ldb)] for ldb in (2304,2344)]
            for row in (a,b):
                row['numeric_source'] = by_numeric[(groups,seed,row['ldb'])]['source']
            metrics = {name:dict(difference=b['metrics'][name]-value, response=b['metrics'][name]/value-1)
                       for name,value in a['metrics'].items() if name not in ('cycles_spread','ns_spread')}
            deltas.append(metrics['cycles_median']['difference'])
            pairs.append(dict(groups=groups,seed=seed,base_ldb=2304,changed_ldb=2344,metrics=metrics))
        lengths.append(dict(groups=groups,delta_cycles=deltas,median_delta_cycles=statistics.median(deltas)))
        for ldb in (2304,2344):
            windows = {}
            for metric in ('cycles_median','ns_median'):
                values = [by_formal[(groups,seed,ldb)]['metrics'][metric] for seed in plan['seeds']]
                median = statistics.median(values)
                spread = max(values)-min(values)
                windows[metric] = dict(values=values,median=median,spread=spread,
                    relative_spread=spread/median,passed=spread<=plan['window_relative_spread']*median)
            conditions.append(dict(groups=groups,ldb=ldb,windows=windows,
                                   passed=all(value['passed'] for value in windows.values())))
    stable = all(condition['passed'] for condition in conditions)
    positive = all(delta>0 for length in lengths for delta in length['delta_cycles'])
    candidates = []
    for hypothesis in read_json(root/'hypotheses.json'):
        predictions = []
        for length in lengths:
            expected = hypothesis['delta_cycles'][str(length['groups'])]
            observed = length['median_delta_cycles']
            error = abs(observed-expected)/expected
            predictions.append(dict(groups=length['groups'],predicted_delta_cycles=expected,
                observed_median_delta_cycles=observed,relative_error=error,
                passed=error<=plan['prediction_relative_tolerance']))
        passed = stable and positive and all(prediction['passed'] for prediction in predictions)
        candidates.append(dict(id=hypothesis['id'],predictions=predictions,passed=passed,
            conclusion='信息不足' if not stable else '获得支持' if passed else '被反证'))
    supported = [candidate['id'] for candidate in candidates if candidate['passed']]
    return dict(schema='gh200-q02-tma-length-score/1',qualification=plan['qualification'],
        independent_C2=stable and positive and len(supported)==1,
        numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),numeric=numeric,formal=formal,
        pairs=pairs,lengths=lengths,conditions=conditions,stable_windows=stable,
        all_pair_deltas_positive=positive,candidates=candidates,supported_candidates=supported,
        heldout_conditions=4,plan_sha256=digest(root/'plan.json'),probe_sha256=digest(root/'build/probe'))


def execute(root):
    plan = load_plan(root)
    start = time.monotonic()
    environment = inspect_allocated_gpu(root/'allocation')
    require(environment['uuid'] == plan['gpu_uuid'], '分配卡与冻结研究对象不符')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        for name,split in (('correctness_commands.json','correctness'),('commands.json','formal')):
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
    print('CLOSED: 12 numeric arms + 12 plain arms; frozen length discrimination',flush=True)


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
        print('VERIFIED: 12 numeric arms + 12 plain arms; frozen predictions scored')


if __name__ == '__main__':
    main()

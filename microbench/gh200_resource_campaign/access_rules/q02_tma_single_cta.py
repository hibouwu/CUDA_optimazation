#!/usr/bin/env python3
"""复用 Q02 原构建的单 CTA 行距定位。"""
import argparse
import json
from pathlib import Path
import shutil
import time

import q02_tma_stride as stride
from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json

HERE = Path(__file__).resolve().parent
SCHEMA = 'gh200-q02-tma-single-cta-execution/1'


def prepare(root, baseline):
    original = read_json(baseline/'plan.json')
    for name, checksum in {**original['inputs_sha256'], **original['source_sha256']}.items():
        require(digest(baseline/name) == checksum, '原 002 冻结身份变更：'+name)
    review = stride.build_review(baseline)
    require(review == read_json(baseline/'build/target-review.json') and review['passed'], '原构建审核不符')
    lines = (baseline/'worklist.txt').read_text().splitlines()
    require(lines[0] == '2304 2304 4096 132', '原工作列表形状不符')
    root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir()
    (root/'baseline').mkdir()
    shutil.copytree(baseline/'build', root/'build')
    for name in ('plan.json', 'delivery-manifest.json', 'input-binding.json', 'program.json', 'worklist.txt'):
        shutil.copy2(baseline/name, root/'baseline'/name)
    shutil.copy2(baseline/'build/target-review.json', root/'baseline/target-review.json')
    (root/'build/target-review.json').unlink()
    shutil.copy2(baseline/'source/q02_tma_stride.py', root/'baseline/q02_tma_stride.py')
    shutil.copytree(baseline/'source-before', root/'baseline/source-before')
    for name in ('tma_gemm_sources.cu', 'r00_common.hpp'):
        shutil.copy2(baseline/'source'/name, root/'source'/name)
    for name in ('q02_tma_single_cta.py', 'q02_tma_stride.py', 'l2_saturation.py', 'test_q02_tma_single_cta.py'):
        shutil.copy2(HERE/name, root/'source'/name)
    replay = root/'baseline/replay'
    replay.mkdir()
    shutil.copy2(baseline/'worklist.txt', replay/'worklist.txt')
    (replay/'build').mkdir()
    shutil.copy2(baseline/'build/target-review.json', replay/'build/target-review.json')
    for name, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
        shutil.copy2(baseline/name, replay/name)
        for task in read_json(baseline/name):
            shutil.copytree(baseline/'runs'/split/task['id'], replay/'runs'/split/task['id'])
    (root/'worklist.txt').write_text('2304 2304 4096 1\n'+lines[1]+'\n'+lines[133]+'\n')
    for name, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
        tasks = []
        for task in read_json(baseline/name):
            if task['active_groups'] != 4:
                continue
            task['paired_ldbs'] = [ldb for ldb in task['paired_ldbs'] if ldb in (2304, 2344)]
            task['pair_indices'] = [0, 1]
            task['argv'][task['argv'].index('--ctas')+1] = '1'
            task['argv'][task['argv'].index('--paired-ldbs')+1] = ','.join(map(str, task['paired_ldbs']))
            if split == 'correctness':
                task['expected_checked'] = 64*16384
            tasks.append(task)
        save_json(root/name, tasks)
    save_json(root/'hypotheses.json', [
        dict(id='single-path', rule='单 CTA 的源地址路径可产生行距响应',
             response='单 CTA 保留非零响应时，继续定位该源路径的地址相关服务', numerical_prediction=None),
        dict(id='shared-load', rule='响应依赖全卡工作集或共享负载',
             response='单 CTA 响应接近零时，保留多 CTA 组织相关解释', numerical_prediction=None)])
    (root/'README.md').write_text('''# Q02：单 CTA 源定位

复用旧 Q02-stride-002 的 CUDA12.9 二进制和构建审核，不重新编译。M/N2304、K4096、A lda4096、allocA4096、allocB2360；实际 B 行距2304／2344同进程配对。Q4、物理4槽、ring64、64组，10次暖机交替原首CTA的两轮坐标(0,0)／(7,6)，正式测第二轮。全A/B分配、数据种子、准备与完成边界沿用原程序。

先执行3个verify进程（6臂，每臂1048576个加载值），全部合格后执行3个plain进程（6臂），seed为3／22／41。每个进程的两臂须落在同一SM；不匹配时保存失败并停止，不自动重采。相对原配对次序仅删除B32/B112，不扩条件。

```sh
python source/q02_tma_single_cta.py execute --root "$PWD"
python source/q02_tma_single_cta.py score --root "$PWD" --output /tmp/q02-single-cta-score.json
python source/test_q02_tma_single_cta.py --root "$PWD"
```

预算10分钟。两项解释为single-path与shared-load，数值幅度未知，本轮资格为development_localization，未启用留出。原132 CTA实测保存在baseline/replay供CPU验证；源与原构建证据身份列于plan.json，包清单列于delivery-manifest.json。
''')
    frozen = {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()
              and str(p.relative_to(root)) != 'build/target-review.json'}
    plan = dict(schema=SCHEMA, qualification='development_localization', independent_C2=False,
        gpu_uuid=original['gpu_uuid'], seeds=original['seeds'], development_conditions=2,
        correctness_processes=3, formal_processes=3, correctness_arm_checks=6, formal_arm_observations=6,
        allocation_budget_seconds=600, maximum_repairs=2, automatic_retries=False,
        heldout_enabled=False, numerical_response_prediction=None, ctas=1, active_groups=4,
        ring=64, physical_slots=4, groups=64, source_rounds=[0, 1], same_sm_pair_required=True,
        baseline=baseline.name, original_plan_sha256=digest(baseline/'plan.json'),
        original_probe_sha256=digest(baseline/'build/probe'), frozen_at_ns=time.time_ns(), files_sha256=frozen)
    save_json(root/'plan.json', plan)
    save_json(root/'build/target-review.json', stride.build_review(root))
    print('PREPARED: same binary; 3 verify + 3 plain paired processes; no GPU')


def load_plan(root):
    plan = read_json(root/'plan.json')
    require(plan['schema'] == SCHEMA and plan['heldout_enabled'] is False, '执行包范围不符')
    for name, checksum in plan['files_sha256'].items():
        require(digest(root/name) == checksum, '冻结身份变更：'+name)
    require(digest(Path(__file__)) == plan['files_sha256']['source/q02_tma_single_cta.py'], 'runner 身份不符')
    review = stride.build_review(root)
    require(review == read_json(root/'build/target-review.json') and review['passed'], '原构建证据不符')
    return plan


def validate_pair(root, folder, task, uuid):
    rows = stride.validate_process(root, folder, task, uuid)
    same_sm = rows[0]['ctas'][0]['smid'] == rows[1]['ctas'][0]['smid']
    require(same_sm, '单 CTA 配对落在不同 SM，停止且不自动重采')
    return rows


def summarize(root):
    plan = load_plan(root)
    require(read_json(root/'allocation/identity.json')['uuid'] == plan['gpu_uuid'], '分配卡身份不符')
    numeric, formal = [], []
    for name, split, rows in (('correctness_commands.json', 'correctness', numeric), ('commands.json', 'formal', formal)):
        for task in read_json(root/name):
            rows.extend(validate_pair(root, root/'runs'/split/task['id'], task, plan['gpu_uuid']))
    by_numeric = {(row['seed'], row['ldb']):row for row in numeric}
    by_formal = {(row['seed'], row['ldb']):row for row in formal}
    require(len(by_numeric) == len(by_formal) == 6 and by_numeric.keys() == by_formal.keys(), '数值资格与正式臂不对应')
    pairs = []
    for seed in plan['seeds']:
        a, b = [by_formal[(seed, ldb)] for ldb in (2304, 2344)]
        for row in (a, b):
            row['numeric_source'] = by_numeric[(seed, row['ldb'])]['source']
        pairs.append(dict(seed=seed, base_ldb=2304, changed_ldb=2344, smid_mapping_equal=True,
            metrics={name:dict(difference=b['metrics'][name]-value, response=b['metrics'][name]/value-1)
                     for name, value in a['metrics'].items() if name not in ('cycles_spread', 'ns_spread')}))
    return dict(schema='gh200-q02-tma-single-cta-score/1', qualification=plan['qualification'], independent_C2=False,
        numerical_response_prediction=None, numeric_arm_checks=len(numeric), formal_arm_observations=len(formal),
        numeric=numeric, formal=formal, pairs=pairs, heldout_enabled=False,
        plan_sha256=digest(root/'plan.json'), probe_sha256=digest(root/'build/probe'))


def execute(root):
    plan = load_plan(root)
    start = time.monotonic()
    environment = inspect_allocated_gpu(root/'allocation')
    require(environment['uuid'] == plan['gpu_uuid'], '分配卡与冻结研究对象不符')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        for name, split in (('correctness_commands.json', 'correctness'), ('commands.json', 'formal')):
            for task in read_json(root/name):
                folder = root/'runs'/split/task['id']
                command_record(folder, stride.probe_command(root, task), stride.remaining(start, 60))
                rows = stride.validate_process(root, folder, task, environment['uuid'])
                save_json(folder/'validated.json', rows)
                same_sm = rows[0]['ctas'][0]['smid'] == rows[1]['ctas'][0]['smid']
                save_json(folder/'pair-gate.json', dict(same_sm=same_sm, smids=[row['ctas'][0]['smid'] for row in rows]))
                require(same_sm, '单 CTA 配对落在不同 SM，停止且不自动重采')
                print(json.dumps(dict(split=split, process=task['id'], arms=len(rows))), flush=True)
        save_json(root/'score.json', summarize(root))
    print('CLOSED: 6 numeric arms + 6 formal arms; development localization', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'execute', 'score'):
        item = sub.add_parser(name)
        item.add_argument('--root', type=Path, required=True)
        if name == 'prepare':
            item.add_argument('--baseline', type=Path, required=True)
        if name == 'score':
            item.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'prepare':
        prepare(root, args.baseline.resolve())
    elif args.command == 'execute':
        execute(root)
    else:
        save_json(args.output, summarize(root))
        print('VERIFIED: 6 numeric arms, 6 formal arms and same-SM paired responses')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""真实单B与双B的有限配对执行。"""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import sys
import time

import q02_tma_stride as stride
from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json

SCHEMA = 'gh200-q02-single-b-discrimination/1'
HERE = Path(__file__).resolve().parent
HALF_PER_GROUP = {0: 0, 2: 4096, 6: 8192}
TRANSACTION_BYTES = {0: 0, 2: 8192, 6: 16384}


def prepare(root, reference):
    plan = read_json(root/'plan.json')
    require(plan['schema'] == SCHEMA, '登记范围不符')
    require(digest(root/'plan.json') == read_json(root/'registration-binding.json')['plan_sha256'], '原登记身份变化')
    for directory in ('source', 'reference'):
        (root/directory).mkdir(exist_ok=True)
    for name in ('q02_tma_stride.py', 'l2_saturation.py', 'r00_common.hpp', 'cutlass-3.9.2-include.tar.gz'):
        shutil.copy2(reference/'source'/name, root/'source'/name)
    shutil.copy2(reference/'worklist.txt', root/'worklist.txt')
    for name in ('score-dev.json', 'score-heldout.json', 'plan.json', 'binding.json', 'approved-target.json'):
        shutil.copy2(reference/name, root/'reference'/name)
    shutil.copy2(reference/'reference/plan.json', root/'reference/operand-plan.json')
    origins = read_json(root/'prior-origins.json')
    for origin in origins['paired_raw']:
        name, checksum, seed = origin['path'], origin['sha256'], origin['seed']
        original = Path(name)
        if not original.is_absolute():
            original = next(parent/name for parent in root.parents if (parent/name).is_file())
        require(digest(original) == checksum, '原operand raw身份变化')
        shutil.copy2(original, root/f'reference/operand-seed{seed}.jsonl')
    old_tasks = {task['seed']: task for task in read_json(reference/'commands-dev.json') if task['Q'] == 1 and task['G'] == 32}
    for stage, conditions in (('dev', plan['development']), ('heldout', plan['heldout'])):
        for verify, filename in ((1, 'correctness_commands'), (0, 'commands')):
            tasks = []
            for seed in (3, 22, 41):
                arms = conditions if seed == 3 else list(reversed(conditions))
                masks = [arm['mask'] for arm in arms]
                pairs = [[4096, arm['ldb']] for arm in arms]
                task = copy.deepcopy(old_tasks[seed])
                task.update(id=f'q1-g32-{stage}-single-b-seed{seed}', seed=seed, Q=1, G=32,
                    active_groups=1, groups=32, paired_request_masks=masks, paired_strides=pairs,
                    pair_indices=list(range(len(arms))), paired_traces=[0], paired_wait_delays=[0], role=stage+'_single_b')
                task.pop('expected_checked_by_arm', None)
                if verify:
                    task['expected_checked_by_arm'] = [32*HALF_PER_GROUP[mask] for mask in masks]
                argv = task['argv']
                for flag, value in (('--verify', str(verify)), ('--measure-active', '1'), ('--measure-groups', '32'),
                        ('--request-mask', '7'), ('--paired-strides', ','.join(f'{a}:{b}' for a, b in pairs)),
                        ('--paired-request-masks', ','.join(map(str, masks)))):
                    argv[argv.index(flag)+1] = value
                tasks.append(task)
            save_json(root/f'{filename}-{stage}.json', tasks)
    print('PREPARED: dev 3 verify +3 formal, four arms; heldout 3+3, two arms gated; no build/GPU', flush=True)


def load_plan(root):
    binding = read_json(root/'binding.json')
    for name, checksum in binding['files'].items():
        require(digest(root/name) == checksum, '冻结身份变更：'+name)
    plan = read_json(root/'plan.json')
    require(plan['schema'] == SCHEMA, '执行范围不符')
    require(digest(Path(__file__)) == binding['files']['source/q02_single_b.py'], 'runner身份不符')
    return plan


def approved_target(root):
    approved = read_json(root/'approved-target.json')
    require(approved['approved'] is True, '目标未获核准')
    for key, name in (('probe_sha256', 'build/probe'), ('kernel_code_sha256', 'build/kernel-code.json'),
            ('plan_sha256', 'plan.json'), ('probe_source_sha256', 'source/tma_gemm_sources.cu'),
            ('target_review_sha256', 'build/target-review.json')):
        require(approved[key] == digest(root/name), '目标批准身份变化：'+name)
    require((root/'build/probe').stat().st_mode & 0o111, 'binary无执行位')


def validate_process(root, folder, task, uuid):
    receipt = stride.successful_receipt(folder)
    command = receipt['argv'][:]
    probe = Path(command[0])
    require(probe.name == 'probe' and probe.parent.name == 'build', 'binary路径不符')
    require(command[command.index('--worklist')+1] == str(probe.parent.parent/'worklist.txt'), '工作列表路径不符')
    command[0] = task['argv'][0]
    command[command.index('--worklist')+1] = 'worklist.txt'
    require(command == task['argv'], '执行参数与冻结命令不符')
    raw = [json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.strip()]
    require(len(raw) == len(task['paired_request_masks']), '配对臂数不符')
    arguments = dict(zip(task['argv'][1::2], task['argv'][2::2]))
    fields = {'ctas': '--ctas', 'active_groups': '--measure-active', 'groups': '--measure-groups', 'ring': '--ring',
        'verify': '--verify', 'trace': '--trace', 'seed': '--seed', 'warmup': '--warmup', 'source_phase_stride': '--source-phase-stride',
        'source_tile_shift': '--source-tile-shift', 'work_index': '--work-index', 'warmup_work_offset': '--warmup-work-offset',
        'release_delay_ns': '--release-delay-ns', 'pollute_kib': '--pollute-kib', 'launch_skew_ns': '--launch-skew-ns',
        'warmup_skew_ns': '--warmup-skew-ns', 'alloc_lda': '--alloc-lda', 'alloc_ldb': '--alloc-ldb'}
    require((root/'worklist.txt').read_text().split() == ['2304', '2304', '4096', '1', '0', '0', '7', '6'], '工作列表不符')
    resources = read_json(root/'build/target-review.json')['rows']
    resource = next(row['resources'] for row in resources if row['label'] == 'after')
    checked = []
    for index, row in enumerate(raw):
        mask = task['paired_request_masks'][index]
        pair = task['paired_strides'][index]
        require(row['schema'] == 'cfg-b-gemm-source-probe/1' and row['gpu_uuid'] == uuid, 'schema/卡身份不符')
        require(all(row[name] == int(arguments[flag]) for name, flag in fields.items()), '逐臂参数不符')
        require(row['warmup_trace'] == -1 and row['warmup_active_groups'] == 4 and row['warmup_groups'] == 64 and
            row['warmup_source_phase_stride'] == 0 and row['record_warmup'] == 0 and not row['warmup_ctas'] and not row['group_events'], '暖机/trace范围不符')
        require(row['paired_strides'] == task['paired_strides'] and row['pair_index'] == row['stride_index'] == index and
            row['trace_index'] == row['wait_delay_index'] == 0 and row['paired_traces'] == row['paired_wait_delays'] == [0] and
            [row['lda'], row['ldb']] == pair, '逐臂stride/trace或次序不符')
        require(row['paired_request_masks'] == task['paired_request_masks'] and row['request_mask'] == mask and
            row['warmup_request_mask'] == 7 and row['transaction_bytes'] == TRANSACTION_BYTES[mask] and
            row['warmup_transaction_bytes'] == 32768 and row['wait_delay_cycles'] == row['warmup_wait_delay_cycles'] == 0, 'mask/事务字节/D0不符')
        require(all(re.fullmatch('[0-9a-f]{256}', row[name]) for name in ('descriptor_a', 'descriptor_b')), 'descriptor字节缺失')
        require(row['descriptor_a'] == raw[0]['descriptor_a'], '相同A行距descriptor变化')
        for previous in raw[:index]:
            if previous['ldb'] == row['ldb']:
                require(previous['descriptor_b'] == row['descriptor_b'], '相同B行距descriptor变化')
        require(all(row[name] == raw[0][name] for name in stride.ADDRESSES), '同进程地址变化')
        require(row['source_a_address'] > 0 and row['source_b_address'] > 0 and
            row['source_a_address'] % 128 == row['source_b_address'] % 128 == 0, '源地址/对齐不符')
        require(row['m'] == row['n'] == 2304 and row['allocated_a_elements'] == 9455616 and row['allocated_b_elements'] == 9666560, '形状/固定分配不符')
        expected = row['groups']*HALF_PER_GROUP[mask] if row['verify'] else 0
        require(row['errors'] == row['pollution_errors'] == row['pollution_checked'] == 0 and row['checked'] == expected, '加载数值/检查量不符')
        require(row['observed_sms'] == row['max_active_ctas_per_sm'] == 1 and row['smem'] == 132096 and row['local_bytes'] == 0 and
            row['registers'] == int(re.search(r'\bREG:(\d+)\b', resource)[1]), '驻留/构建资源不符')
        require(len(row['ctas_data']) == 1, 'CTA数量不符')
        cta = row['ctas_data'][0]
        require(cta['cta'] == 0 and cta['tile'] == [7, 6] and cta['smid'] == 124, 'CTA/坐标/SM124不符')
        for unit in (0, 1):
            require(type(cta['begin'][unit]) is int and type(cta['end'][unit]) is int and 0 < cta['begin'][unit] < cta['end'][unit], '完成戳无效')
        require('prefill_end' not in cta, 'plain错误读取暖机预填戳')
        checked.append(dict(seed=row['seed'], Q=row['active_groups'], G=row['groups'], request_mask=mask,
            lda=row['lda'], ldb=row['ldb'], pair_index=index, checked_values=row['checked'], numerical_qualified=bool(row['verify']),
            stdout_sha256=receipt['stdout_sha256'], source=str(folder.relative_to(root)), device_uuid=uuid,
            metrics=dict(cycles=cta['end'][0]-cta['begin'][0], ns=cta['end'][1]-cta['begin'][1])))
    return checked


def execute(root, stage):
    sys.path.insert(0, str(root))
    import score_single_b
    load_plan(root)
    approved_target(root)
    require(not (root/'runs'/stage).exists(), '本阶段已有执行记录')
    if stage == 'heldout':
        score_single_b.validate_frozen_predictions(root)
    uuid = read_json(root/'reference/operand-plan.json')['gpu_uuid']
    start = time.monotonic()
    environment = inspect_allocated_gpu(root/'allocation'/stage)
    require(environment['uuid'] == uuid, '分配卡不是冻结参考卡')
    with exclusive_lock(Path('/tmp')/f'gh200-measurement-{uuid}.lock'):
        for filename, split in (('correctness_commands', 'verify'), ('commands', 'formal')):
            for task in read_json(root/f'{filename}-{stage}.json'):
                folder = root/'runs'/stage/split/task['id']
                command_record(folder, stride.probe_command(root, task), stride.remaining(start, 60))
                save_json(folder/'validated.json', validate_process(root, folder, task, uuid))
                print(json.dumps(dict(stage=stage, split=split, process=task['id'], arms=len(task['paired_request_masks']))), flush=True)
        save_json(root/f'score-{stage}.json', score_single_b.summarize(root, stage))
    print('CLOSED: '+stage+' finite single-B observation', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('prepare', 'execute'):
        item = sub.add_parser(mode)
        item.add_argument('--root', type=Path, required=True)
        if mode == 'prepare':
            item.add_argument('--reference', type=Path, required=True)
        else:
            item.add_argument('--stage', choices=('dev', 'heldout'), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'prepare':
        prepare(root, args.reference.resolve())
    else:
        execute(root, args.stage)


if __name__ == '__main__':
    main()

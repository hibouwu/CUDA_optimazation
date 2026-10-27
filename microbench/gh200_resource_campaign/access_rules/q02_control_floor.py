#!/usr/bin/env python3
"""同一动态 mask 程序的零搬运控制窗口。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

import q02_tma_stride as stride
from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json

HERE = Path(__file__).resolve().parent
SCHEMA = 'gh200-q02-control-floor/1'


def prepare(root, reference):
    plan = read_json(root/'plan.json')
    require(plan['schema'] == SCHEMA, '登记范围不符')
    for name, checksum in read_json(root/'registration-binding.json').items():
        require(digest(root/name) == checksum, '原登记身份变化')
    for directory in ('source', 'source-before', 'reference'):
        (root/directory).mkdir(exist_ok=True)
    shutil.copy2(HERE/'q02_control_floor.py', root/'source/q02_control_floor.py')
    for name in ('q02_tma_stride.py', 'l2_saturation.py', 'r00_common.hpp', 'cutlass-3.9.2-include.tar.gz'):
        shutil.copy2(reference/'source'/name, root/'source'/name)
    original = (reference/'source/tma_gemm_sources.cu').read_text()
    changed = original.replace('(mask!=1 && mask!=6 && mask!=7)', '(mask!=0 && mask!=1 && mask!=6 && mask!=7)').replace(
        '(request_mask!=1 && request_mask!=6 && request_mask!=7)', '(request_mask!=0 && request_mask!=1 && request_mask!=6 && request_mask!=7)')
    require(changed != original and changed.count('mask!=0') == 2, 'host 两处值域不符')
    (root/'source-before/tma_gemm_sources.cu').write_text(original)
    (root/'source/tma_gemm_sources.cu').write_text(changed)
    shutil.copy2(reference/'worklist.txt', root/'worklist.txt')
    for name in ('score.json', 'plan.json', 'binding.json', 'approved-target.json'):
        shutil.copy2(reference/name, root/'reference'/name)
    shutil.copy2(reference/'build/after/disassemble/stdout', root/'reference/full-sass.txt')
    shutil.copy2(reference/'build/after/resources/stdout', root/'reference/full-resources.txt')
    save_json(root/'reference/origins.json', dict(reference=str(reference), files={
        name: digest(reference/name) for name in ('source/tma_gemm_sources.cu', 'source/q02_tma_stride.py',
        'source/l2_saturation.py', 'build/probe', 'build/after/disassemble/stdout', 'build/after/resources/stdout')}))
    old_tasks = {task['seed']: task for task in read_json(reference/'commands.json')}
    for stage, groups in (('dev', (16, 32)), ('heldout', (64,))):
        for verify, filename in ((1, 'correctness_commands'), (0, 'commands')):
            tasks = []
            for q in (1, 4):
                for g in groups:
                    for seed in plan['seeds']:
                        masks = ([7, 0] if seed == 3 else [0, 7]) if g == 32 else [0]
                        task = copy.deepcopy(old_tasks[seed])
                        task.update(id=f'q{q}-g{g}-mask-control-seed{seed}', seed=seed, Q=q, G=g,
                            active_groups=q, groups=g, paired_request_masks=masks,
                            paired_strides=[[4096, 2304] for _ in masks], pair_indices=list(range(len(masks))),
                            paired_traces=[0], paired_wait_delays=[0], role=stage+'_control_window')
                        task.pop('expected_checked_by_arm', None)
                        if verify:
                            task['expected_checked_by_arm'] = [g*16384 if mask else 0 for mask in masks]
                        argv = task['argv']
                        for flag, value in (('--verify', str(verify)), ('--measure-active', str(q)),
                                ('--measure-groups', str(g)), ('--request-mask', '7'),
                                ('--paired-strides', ','.join('4096:2304' for _ in masks)),
                                ('--paired-request-masks', ','.join(map(str, masks)))):
                            argv[argv.index(flag)+1] = value
                        tasks.append(task)
            require(len(tasks) == (12 if stage == 'dev' else 6), '进程清单不符')
            save_json(root/f'{filename}-{stage}.json', tasks)
    print('PREPARED: dev 12 verify +12 formal; heldout 6+6 gated; no build/GPU', flush=True)


def load_plan(root, execution=False):
    binding = read_json(root/('binding.json' if execution else 'build-input-binding.json'))
    for name, checksum in binding['files'].items():
        require(digest(root/name) == checksum, '冻结身份变更：'+name)
    plan = read_json(root/'plan.json')
    require(plan['schema'] == SCHEMA, '执行范围不符')
    return plan


def kernel_records(sass):
    records = {}
    for section in re.split(r'Function\s*:\s*', sass)[1:]:
        symbol = section.splitlines()[0].strip()
        code = []
        for line in section.splitlines()[1:]:
            match = re.match(r'\s*/\*([0-9a-f]+)\*/\s*(.*?)\s*/\* 0x([0-9a-f]+) \*/', line)
            if match:
                code.append(dict(pc=int(match[1], 16), op=match[2].strip(), words=[int(match[3], 16)]))
            else:
                word = re.search(r'/\* 0x([0-9a-f]+) \*/', line)
                if word and code:
                    code[-1]['words'].append(int(word[1], 16))
        require(code and all(len(row['words']) == 2 for row in code), '完整两机器字缺失')
        require(symbol not in records, '设备入口重复')
        records[symbol] = code
    require(len(records) == 7, '7 个设备函数缺失')
    return records


def review_build(root):
    base = stride.build_review(root)
    records = {label: kernel_records((root/f'build/{label}/disassemble/stdout').read_text()) for label in ('before', 'after')}
    records['reference'] = kernel_records((root/'reference/full-sass.txt').read_text())
    usages = {label: {match[1]: match[2].strip() for match in re.finditer(r'Function (\S+):\s*\n\s*([^\n]+)',
        (root/f'build/{label}/resources/stdout').read_text())} for label in ('before', 'after')}
    usages['reference'] = {match[1]: match[2].strip() for match in re.finditer(r'Function (\S+):\s*\n\s*([^\n]+)',
        (root/'reference/full-resources.txt').read_text())}
    require(all(set(usages[label]) == set(records[label]) for label in records), '设备函数资源不齐')
    exact = records['before'] == records['after'] == records['reference'] and usages['before'] == usages['after'] == usages['reference']
    version = (root/'build/version/stdout').read_text()
    tensor = next(symbol for symbol in records['after'] if symbol.startswith('_Z12tensor_probe'))
    passed = base['passed'] and exact and 'V12.9.41' in version and 'compiler.35813241_0' in version
    return dict(status='pending_root_full_device_code_approval', approved=False, passed=passed,
        all_seven_code_and_resources_exact=exact, functions=[dict(symbol=symbol, instructions=len(code),
            resources=usages['after'][symbol], machine_words_sha256=hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest())
            for symbol, code in records['after'].items()], rows=base['rows'], compiler_version=version,
        probe_sha256=digest(root/'build/probe'), kernel_code_sha256=digest(root/'build/kernel-code.json'),
        plan_sha256=digest(root/'plan.json'), probe_source_sha256=digest(root/'source/tma_gemm_sources.cu'),
        reference_sass_sha256=digest(root/'reference/full-sass.txt'))


def build_only(root):
    load_plan(root)
    stride.build(root, time.monotonic())
    save_json(root/'build/kernel-code.json', kernel_records((root/'build/after/disassemble/stdout').read_text()))
    review = review_build(root)
    save_json(root/'build/all-kernel-review.json', review)
    require(review['passed'], '全部 7 个设备函数机器字/资源或编译器门槛失败')
    print('BUILT ONLY: seven functions exact; root approval pending; no GPU', flush=True)


def approved_target(root):
    actual = review_build(root)
    saved = read_json(root/'build/all-kernel-review.json')
    approved = read_json(root/'approved-target.json')
    require(actual == saved and actual['passed'], '完整构建证据不符')
    require(approved['approved'] is True and all(approved[key] == actual[key] for key in
        ('probe_sha256', 'kernel_code_sha256', 'plan_sha256', 'probe_source_sha256')) and
        approved['target_review_sha256'] == digest(root/'build/all-kernel-review.json'), '目标未获核准')
    require((root/'build/probe').stat().st_mode & 0o111, 'binary 无执行位')


def validate_process(root, folder, task, uuid):
    receipt = stride.successful_receipt(folder)
    command = receipt['argv'][:]
    probe = Path(command[0])
    require(probe.name == 'probe' and probe.parent.name == 'build', 'binary 路径不符')
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
    resource = read_json(root/'build/target-review.json')['rows'][1]['resources']
    checked = []
    for index, row in enumerate(raw):
        mask = task['paired_request_masks'][index]
        require(row['schema'] == 'cfg-b-gemm-source-probe/1' and row['gpu_uuid'] == uuid, 'schema/卡身份不符')
        require(all(row[name] == int(arguments[flag]) for name, flag in fields.items()), '逐臂参数不符')
        require(row['warmup_trace'] == -1 and row['warmup_active_groups'] == 4 and row['warmup_groups'] == 64 and
            row['warmup_source_phase_stride'] == 0 and row['record_warmup'] == 0 and not row['warmup_ctas'] and not row['group_events'], '暖机/trace 范围不符')
        require(row['paired_strides'] == task['paired_strides'] and row['pair_index'] == row['stride_index'] == index and
            row['trace_index'] == row['wait_delay_index'] == 0 and row['paired_traces'] == row['paired_wait_delays'] == [0] and
            [row['lda'], row['ldb']] == [4096, 2304], '逐臂 stride/trace 或次序不符')
        require(row['paired_request_masks'] == task['paired_request_masks'] and row['request_mask'] == mask and
            row['warmup_request_mask'] == 7 and row['transaction_bytes'] == (32768 if mask else 0) and
            row['warmup_transaction_bytes'] == 32768 and row['wait_delay_cycles'] == row['warmup_wait_delay_cycles'] == 0, 'mask/事务字节/D0 不符')
        require(all(re.fullmatch('[0-9a-f]{256}', row[name]) for name in ('descriptor_a', 'descriptor_b')) and
            all(row[name] == raw[0][name] for name in ('descriptor_a', 'descriptor_b')), '相同行距 descriptor 变化')
        require(all(row[name] == raw[0][name] for name in stride.ADDRESSES), '同进程地址变化')
        require(row['source_a_address'] > 0 and row['source_b_address'] > 0 and
            row['source_a_address'] % 128 == row['source_b_address'] % 128 == 0, '源地址/对齐不符')
        require(row['m'] == row['n'] == 2304 and row['allocated_a_elements'] == 9455616 and row['allocated_b_elements'] == 9666560, '形状/固定分配不符')
        expected = row['groups']*16384 if row['verify'] and mask else 0
        require(row['errors'] == row['pollution_errors'] == row['pollution_checked'] == 0 and row['checked'] == expected, '加载数值/检查量不符')
        require(row['observed_sms'] == row['max_active_ctas_per_sm'] == 1 and row['smem'] == 132096 and row['local_bytes'] == 0 and
            row['registers'] == int(re.search(r'\bREG:(\d+)\b', resource)[1]), '驻留/构建资源不符')
        require(len(row['ctas_data']) == 1, 'CTA 数量不符')
        cta = row['ctas_data'][0]
        require(cta['cta'] == 0 and cta['tile'] == [7, 6] and cta['smid'] == 124, 'CTA/坐标/SM124 不符')
        for unit in (0, 1):
            require(type(cta['begin'][unit]) is int and type(cta['end'][unit]) is int and 0 < cta['begin'][unit] < cta['end'][unit], '完成戳无效')
        require('prefill_end' not in cta, 'plain 错误读取暖机预填戳')
        checked.append(dict(seed=row['seed'], Q=row['active_groups'], G=row['groups'], request_mask=mask,
            lda=row['lda'], ldb=row['ldb'], pair_index=index, checked_values=row['checked'], numerical_qualified=bool(row['verify']),
            stdout_sha256=receipt['stdout_sha256'], source=str(folder.relative_to(root)), device_uuid=uuid,
            metrics=dict(cycles=cta['end'][0]-cta['begin'][0], ns=cta['end'][1]-cta['begin'][1])))
    return checked


def execute(root, stage):
    sys.path.insert(0, str(root))
    load_plan(root, True)
    approved_target(root)
    require(not (root/'runs'/stage).exists(), '本阶段已有执行记录')
    if stage == 'heldout':
        import score_control
        score_control.validate_frozen_predictions(root)
    uuid = read_json(root/'reference/plan.json')['gpu_uuid']
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
        import score_control
        save_json(root/f'score-{stage}.json', score_control.summarize(root, stage))
    print('CLOSED: '+stage+' finite control-window observation', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('prepare', 'build-only', 'execute'):
        item = sub.add_parser(mode)
        item.add_argument('--root', type=Path, required=True)
        if mode == 'prepare':
            item.add_argument('--reference', type=Path, required=True)
        if mode == 'execute':
            item.add_argument('--stage', choices=('dev', 'heldout'), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'prepare':
        prepare(root, args.reference.resolve())
    elif args.mode == 'build-only':
        build_only(root)
    else:
        execute(root, args.stage)


if __name__ == '__main__':
    main()

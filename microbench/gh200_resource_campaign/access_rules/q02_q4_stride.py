"""保存的 tensor probe：Q4 基础／A16 两臂独立外推。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from l2_saturation import (command_record, exclusive_lock, inspect_allocated_gpu,
                           read_json, require, save_json)


SCHEMA = 'gh200-q02-q4-stride-heldout/1'
ADDRESSES = ('source_a_address', 'source_b_address', 'records_address',
             'ctas_address', 'worklist_address', 'gate_address', 'pollution_address')
FLAGS = dict(fits=0, model_calls=0, default_model_changed=False,
             full_gemm_C2=False, full_gemm_C3=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_command(root, task):
    argv = task['argv'][:]
    argv[0] = str(root/'build/probe')
    argv[argv.index('--worklist')+1] = str(root/'worklist.txt')
    return argv


def load_inputs(root):
    binding = read_json(root/'binding.json')
    for name, expected in binding['files'].items():
        require(sha(root/name) == expected, '冻结输入变化：'+name)
    require(sha(__file__) == binding['files']['source/q02_q4_stride.py'], '执行脚本不是冻结版本')
    plan, commands = read_json(root/'plan.json'), read_json(root/'commands.json')
    freeze, predictions = read_json(root/'freeze.json'), read_json(root/'predictions.json')
    require(plan['schema'] == SCHEMA, '登记范围不符')
    require(sha(root/'binding.json') == freeze['binding_sha256'], '冻结 binding 身份不符')
    for name in ('plan', 'commands', 'predictions'):
        require(sha(root/(name+'.json')) == freeze[name+'_sha256'], '冻结记录不符：'+name)
    require(sha(root/'build/probe') == freeze['native_sha256'] == plan['native_sha256'], '保存 binary 变化')
    require((root/'build/probe').stat().st_mode & 0o111, '保存 binary 无执行位')
    require((root/'worklist.txt').read_text().split() ==
            ['2304', '2304', '4096', '1', '0', '0', '7', '6'], '工作列表不符')
    expected = [(role, seed, verify) for role, verify in (('verify', 1), ('formal', 0))
                for seed in plan['seeds']]
    require([(t['role'], t['seed'], t['verify']) for t in commands] == expected,
            '六进程清单或数值／正式顺序不符')
    for task in commands:
        args = dict(zip(task['argv'][1::2], task['argv'][2::2]))
        strides = [[4096, 2304], [4104, 2304]]
        if task['seed'] != 3:
            strides.reverse()
        require(task['paired_strides'] == strides and task['paired_request_masks'] == [7, 7],
                '配对行距／mask 清单不符')
        require(args['--paired-strides'] == ','.join(f'{a}:{b}' for a, b in strides) and
                args['--paired-request-masks'] == '7,7', '配对命令不符')
        fixed = {'--ctas': 1, '--active': 4, '--groups': 64, '--ring': 64,
                 '--verify': task['verify'], '--trace': 0, '--seed': task['seed'],
                 '--warmup': 10, '--source-phase-stride': 0, '--source-tile-shift': 0,
                 '--work-index': 1, '--warmup-work-offset': 0, '--release-delay-ns': 0,
                 '--pollute-kib': 0, '--launch-skew-ns': 0, '--warmup-skew-ns': 0,
                 '--lda': 4096, '--ldb': 2304, '--alloc-lda': 4104, '--alloc-ldb': 2360,
                 '--measure-groups': 32, '--measure-active': 4, '--wait-delay-cycles': 0,
                 '--request-mask': 7}
        require(all(int(args[key]) == value for key, value in fixed.items()) and
                args['--paired-traces'] == args['--paired-wait-delays'] == '0', '固定命令变化')
    return plan, commands, freeze, predictions


def validate_rows(raw, task, plan, source):
    require(len(raw) == 2, '每进程应有两臂')
    checked = []
    args = dict(zip(task['argv'][1::2], task['argv'][2::2]))
    fields = {'ctas': '--ctas', 'active_groups': '--measure-active', 'groups': '--measure-groups',
              'ring': '--ring', 'verify': '--verify', 'trace': '--trace', 'seed': '--seed',
              'warmup': '--warmup', 'source_phase_stride': '--source-phase-stride',
              'source_tile_shift': '--source-tile-shift', 'work_index': '--work-index',
              'warmup_work_offset': '--warmup-work-offset', 'release_delay_ns': '--release-delay-ns',
              'pollute_kib': '--pollute-kib', 'launch_skew_ns': '--launch-skew-ns',
              'warmup_skew_ns': '--warmup-skew-ns', 'alloc_lda': '--alloc-lda', 'alloc_ldb': '--alloc-ldb'}
    for index, row in enumerate(raw):
        require(row['schema'] == 'cfg-b-gemm-source-probe/1' and row['gpu_uuid'] == plan['gpu_uuid'],
                'schema 或卡身份不符')
        require(all(row[key] == int(args[flag]) for key, flag in fields.items()), '逐臂参数不符')
        require(row['paired_strides'] == task['paired_strides'] and
                [row['lda'], row['ldb']] == task['paired_strides'][index] and
                row['pair_index'] == row['stride_index'] == index and
                row['trace_index'] == row['wait_delay_index'] == 0, '配对行距／次序不符')
        mask = task['paired_request_masks'][index]
        require(row['paired_request_masks'] == task['paired_request_masks'] and row['request_mask'] == mask and
                row['paired_traces'] == row['paired_wait_delays'] == [0], 'mask／trace 清单不符')
        transaction_bytes = 16384*((1 if mask & 1 else 0)+(1 if mask & 6 else 0))
        require(row['transaction_bytes'] == transaction_bytes and row['warmup_transaction_bytes'] == 32768 and
                row['warmup_request_mask'] == 7 and row['wait_delay_cycles'] ==
                row['warmup_wait_delay_cycles'] == 0, '事务字节或等待干预不符')
        require(row['warmup_trace'] == -1 and row['warmup_active_groups'] == 4 and
                row['warmup_groups'] == 64 and row['warmup_source_phase_stride'] == 0 and
                row['record_warmup'] == 0 and not row['warmup_ctas'] and not row['group_events'],
                '暖机／plain 计时协议不符')
        require(row['m'] == row['n'] == 2304 and
                row['allocated_a_elements'] == 9455616 and row['allocated_b_elements'] == 9666560,
                '问题形状／固定分配不符')
        require(all(row[name] == raw[0][name] for name in ADDRESSES), '同进程地址变化')
        require(row['source_a_address'] > 0 and row['source_b_address'] > 0 and
                row['source_a_address'] % 128 == row['source_b_address'] % 128 == 0, '源地址或对齐不符')
        require(all(re.fullmatch('[0-9a-f]{256}', row[name]) for name in ('descriptor_a', 'descriptor_b')),
                '描述符记录不完整')
        expected = plan['processes']['checked_per_verify_arm'] if task['verify'] and mask else 0
        require(row['errors'] == row['pollution_errors'] == row['pollution_checked'] == 0 and
                row['checked'] == expected, '数值或检查量不符')
        require(row['observed_sms'] == row['max_active_ctas_per_sm'] == 1 and
                row['registers'] == 52 and row['smem'] == 132096 and row['local_bytes'] == 0,
                '资源／驻留不符')
        require(len(row['ctas_data']) == 1, 'CTA 数量不符')
        cta = row['ctas_data'][0]
        require(cta['cta'] == 0 and cta['tile'] == [7, 6] and cta['smid'] == 124, '工作坐标或 SM124 不符')
        for unit in (0, 1):
            require(type(cta['begin'][unit]) is int and type(cta['end'][unit]) is int and
                    0 < cta['begin'][unit] < cta['end'][unit], '完成边界无效')
        require('prefill_end' not in cta, 'plain 出现额外打点')
        checked.append(dict(role=task['role'], seed=task['seed'], lda=row['lda'], ldb=row['ldb'],
            pair_index=index, checked_values=row['checked'], source=source,
            metrics=dict(cycles=cta['end'][0]-cta['begin'][0], ns=cta['end'][1]-cta['begin'][1])))
    for index, name in enumerate(('descriptor_a', 'descriptor_b')):
        same_stride = task['paired_strides'][0][index] == task['paired_strides'][1][index]
        require((raw[0][name] == raw[1][name]) == same_stride, '描述符与配对行距不符')
    return checked


def validate_process(root, task, plan, freeze):
    folder = root/'runs'/'heldout'/task['role']/task['id']
    receipt = read_json(folder/'receipt.json')
    require(receipt['returncode'] == 0 and receipt['timed_out'] is False and receipt['failure'] is None,
            '命令未成功完成')
    argv = receipt['argv'][:]
    probe = Path(argv[0])
    worklist_index = argv.index('--worklist')+1
    require(probe.name == 'probe' and probe.parent.name == 'build' and
            Path(argv[worklist_index]) == probe.parent.parent/'worklist.txt', '执行路径不符')
    record = read_json(folder/'command.json')
    require(record['argv'] == receipt['argv'] and record['cwd'] ==
            str(probe.parent.parent/'runs'/'heldout'/task['role']/task['id']) and
            0 < record['timeout_seconds'] <= 60, '命令凭据不符')
    argv[0] = task['argv'][0]
    argv[worklist_index] = task['argv'][worklist_index]
    require(argv == task['argv'], '执行命令与冻结清单不符')
    require(receipt['start_ns'] >= freeze['frozen_at_ns'] and receipt['stop_ns'] >= receipt['start_ns'],
            '进程早于预测冻结或时间记录不符')
    require(receipt['stdout_sha256'] == sha(folder/'stdout') and receipt['stderr_sha256'] == sha(folder/'stderr'),
            '原始输出身份变化')
    raw = [json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.strip()]
    return validate_rows(raw, task, plan, str(folder.relative_to(root)))


def spread(values, threshold):
    median = statistics.median(values)
    width = max(values)-min(values)
    relative = width/abs(median) if median else (0. if width == 0 else None)
    return dict(values=values, median=median, relative_spread=relative, threshold=threshold,
                passed=relative is not None and relative <= threshold)


def relative_check(observed, predicted, threshold):
    error = abs(observed-predicted)/predicted if predicted > 0 else None
    return dict(observed=observed, predicted=predicted, relative_error_prediction_denominator=error,
                threshold=threshold, passed=error is not None and error <= threshold)


def summarize(plan, predictions, rows):
    numeric = [row for row in rows if row['role'] == 'verify']
    formal = [row for row in rows if row['role'] == 'formal']
    expected = {(seed, lda) for seed in plan['seeds'] for lda in (4096, 4104)}
    require(len(numeric) == len(formal) == 6 and
            {(r['seed'], r['lda']) for r in numeric} == expected and
            {(r['seed'], r['lda']) for r in formal} == expected, '数值／正式六臂不完整')
    gates = plan['gates']
    windows = {}
    for lda in (4096, 4104):
        windows[str(lda)] = {unit: spread([next(r['metrics'][unit] for r in formal
            if r['seed'] == seed and r['lda'] == lda) for seed in plan['seeds']],
            gates['window_relative_spread']) for unit in ('cycles', 'ns')}
    differences = {}
    for unit in ('cycles', 'ns'):
        differences[unit] = [next(r['metrics'][unit] for r in formal if r['seed'] == seed and r['lda'] == 4104)-
                             next(r['metrics'][unit] for r in formal if r['seed'] == seed and r['lda'] == 4096)
                             for seed in plan['seeds']]
    paired = spread(differences['cycles'], gates['paired_cycle_relative_spread'])
    paired['ns_values'] = differences['ns']
    paired['ns_median'] = statistics.median(differences['ns'])
    bridge = {unit: relative_check(windows['4096'][unit]['median'], gates['base_reference_'+unit],
                                  gates['base_bridge_relative_error']) for unit in ('cycles', 'ns')}
    qualified = (all(v['passed'] for w in windows.values() for v in w.values()) and
                 paired['passed'] and all(v['passed'] for v in bridge.values()))
    checks = {lda: relative_check(windows[lda]['cycles']['median'], value,
                                 gates['prediction_denominator_relative_error'])
              for lda, value in predictions['windows_cycles'].items()}
    checks['paired'] = relative_check(paired['median'], predictions['paired_cycles'],
                                      gates['prediction_denominator_relative_error'])
    passed = qualified and all(v['passed'] for v in checks.values())
    decision = plan['decisions']['qualification_failure'] if not qualified else (
        plan['decisions']['passed'] if passed else plan['decisions']['prediction_failure'])
    return dict(schema=SCHEMA, decision=decision, qualification=plan['qualification'],
                observation_qualified=qualified, passed=passed, numeric=numeric, formal=formal,
                windows=windows, paired=paired, base_bridge=bridge, model_checks=checks,
                prediction_unit=plan['prediction_unit'], ns_role=plan['ns_role'], gates=gates, **FLAGS)


def score(root):
    plan, tasks, freeze, predictions = load_inputs(root)
    rows = []
    for task in tasks:
        rows.extend(validate_process(root, task, plan, freeze))
    return summarize(plan, predictions, rows)


def execute(root):
    plan, tasks, freeze, predictions = load_inputs(root)
    require(not (root/'runs'/'heldout').exists(), '本轮已有进程记录')
    start = time.monotonic()
    rows = []
    try:
        environment = inspect_allocated_gpu(root/'allocation'/'heldout')
        require(environment['uuid'] == plan['gpu_uuid'], '分配卡不是冻结参考卡')
        with exclusive_lock(Path('/tmp')/f'gh200-measurement-{plan["gpu_uuid"]}.lock'):
            for task in tasks:
                remaining = 480.-(time.monotonic()-start)
                require(remaining > 0, '完整轮480秒预算耗尽')
                folder = root/'runs'/'heldout'/task['role']/task['id']
                command_record(folder, probe_command(root, task), min(60., remaining))
                checked = validate_process(root, task, plan, freeze)
                rows.extend(checked)
                save_json(folder/'validated.json', checked)
                print(json.dumps(dict(role=task['role'], seed=task['seed'], arms=2)), flush=True)
        report = summarize(plan, predictions, rows)
    except Exception as error:
        save_json(root/'score-heldout.json', dict(schema=SCHEMA, decision=plan['decisions']['qualification_failure'],
            observation_qualified=False, passed=False, error=str(error), completed_rows=rows, **FLAGS))
        raise
    save_json(root/'score-heldout.json', report)
    print(json.dumps(dict(decision=report['decision'], passed=report['passed'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('execute', 'score'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'execute':
        execute(root)
    else:
        require(args.output is not None, 'score 需要 --output')
        save_json(args.output, score(root))


if __name__ == '__main__':
    main()

"""保存的 tensor probe：Q4 行距响应的单次 prefill 定位。"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

from l2_saturation import (command_record, exclusive_lock, inspect_allocated_gpu,
                           read_json, require, save_json)
from q02_q4_stride import (ADDRESSES, FLAGS, probe_command, relative_check, sha,
                           spread, validate_rows as validate_stride_rows)


SCHEMA = 'gh200-q02-q4-prefill-localization/1'
CONTROL_SCHEMA = 'gh200-q02-q4-control-prefill/1'


def load_inputs(root):
    binding = read_json(root/'binding.json')
    for name, expected in binding['files'].items():
        require(sha(root/name) == expected, '冻结输入变化：'+name)
    require(sha(__file__) == binding['files']['source/q02_q4_prefill.py'], '执行脚本不是冻结版本')
    plan, commands = read_json(root/'plan.json'), read_json(root/'commands.json')
    freeze, predictions = read_json(root/'freeze.json'), read_json(root/'predictions.json')
    require(plan['schema'] in (SCHEMA, CONTROL_SCHEMA), '登记范围不符')
    control = plan['schema'] == CONTROL_SCHEMA
    if control:
        require(plan['comparison_axis'] == 'request_mask' and plan['axis_values'] == [0, 7], '控制配对轴不符')
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
        masks = [7, 7]
        if control:
            strides = [[4096, 2304], [4096, 2304]]
            masks = [0, 7] if task['seed'] == 3 else [7, 0]
        traces = [-1, 0] if task['seed'] == 22 else [0, -1]
        require(task['paired_strides'] == strides and task['paired_request_masks'] == masks and
                task['paired_traces'] == traces and task['paired_wait_delays'] == [0] and
                task['pair_indices'] == [0, 1, 2, 3], '四臂清单不符')
        require(args['--paired-strides'] == ','.join(f'{a}:{b}' for a, b in strides) and
                args['--paired-request-masks'] == ','.join(map(str, masks)) and
                args['--paired-traces'] == ','.join(map(str, traces)) and
                args['--paired-wait-delays'] == '0', '配对命令不符')
        fixed = {'--ctas': 1, '--active': 4, '--groups': 64, '--ring': 64,
                 '--verify': task['verify'], '--trace': 0, '--seed': task['seed'],
                 '--warmup': 10, '--source-phase-stride': 0, '--source-tile-shift': 0,
                 '--work-index': 1, '--warmup-work-offset': 0, '--release-delay-ns': 0,
                 '--pollute-kib': 0, '--launch-skew-ns': 0, '--warmup-skew-ns': 0,
                 '--lda': 4096, '--ldb': 2304, '--alloc-lda': 4104, '--alloc-ldb': 2360,
                 '--measure-groups': 32, '--measure-active': 4, '--wait-delay-cycles': 0,
                 '--request-mask': 7}
        require(all(int(args[key]) == value for key, value in fixed.items()), '固定命令变化')
    return plan, commands, freeze, predictions


def validate_rows(raw, task, plan, source):
    require(len(raw) == 4, '每进程应有四臂')
    require(all(all(row[name] == raw[0][name] for name in ADDRESSES) for row in raw),
            '四臂同进程地址变化')
    checked = []
    for trace_index, trace in enumerate(task['paired_traces']):
        actual = raw[trace_index*2:trace_index*2+2]
        for stride_index, row in enumerate(actual):
            require(row['trace'] == trace and row['trace_index'] == trace_index and
                    row['stride_index'] == stride_index and row['pair_index'] == trace_index*2+stride_index and
                    row['paired_traces'] == task['paired_traces'] and
                    row['paired_request_masks'] == task['paired_request_masks'] and
                    row['request_mask'] == task['paired_request_masks'][stride_index] and
                    [row['lda'], row['ldb']] == task['paired_strides'][stride_index], '实际 trace／mask／四臂次序不符')
            cta = row['ctas_data'][0]
            if trace == -1:
                point = cta['prefill_end']
                require(len(point) == 2 and all(type(v) is int for v in point) and
                        all(cta['begin'][u] <= point[u] <= cta['end'][u] for u in (0, 1)),
                        'prefill 边界无效')
            else:
                require('prefill_end' not in cta, 'plain 出现额外打点')
        view, task_view = copy.deepcopy(actual), copy.deepcopy(task)
        task_view['paired_traces'] = [0]
        task_view['pair_indices'] = [0, 1]
        task_view['argv'][task_view['argv'].index('--paired-traces')+1] = '0'
        for index, row in enumerate(view):
            row['trace'] = row['trace_index'] = 0
            row['pair_index'] = index
            row['paired_traces'] = [0]
            row['ctas_data'][0].pop('prefill_end', None)
        validated = validate_stride_rows(view, task_view, plan, source)
        for index, row in enumerate(actual):
            item = validated[index]
            item.update(trace=trace, trace_index=trace_index, pair_index=trace_index*2+index)
            if plan.get('comparison_axis') == 'request_mask':
                item['request_mask'] = row['request_mask']
            if trace == -1:
                cta = row['ctas_data'][0]
                item['prefix'] = {unit: cta['prefill_end'][u]-cta['begin'][u]
                                  for u, unit in enumerate(('cycles', 'ns'))}
                item['suffix'] = {unit: cta['end'][u]-cta['prefill_end'][u]
                                  for u, unit in enumerate(('cycles', 'ns'))}
            checked.append(item)
    for index in (0, 1):
        require(all(raw[index][name] == raw[index+2][name]
                    for name in ('descriptor_a', 'descriptor_b')), '跨 trace 描述符变化')
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
            '进程早于冻结或时间记录不符')
    require(receipt['stdout_sha256'] == sha(folder/'stdout') and receipt['stderr_sha256'] == sha(folder/'stderr'),
            '原始输出身份变化')
    raw = [json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.strip()]
    return validate_rows(raw, task, plan, str(folder.relative_to(root)))


def summarize(plan, predictions, rows):
    numeric = [r for r in rows if r['role'] == 'verify']
    formal = [r for r in rows if r['role'] == 'formal']
    axis = plan.get('comparison_axis', 'lda')
    values = plan.get('axis_values', [4096, 4104])
    base_value, changed_value = values
    expected = {(seed, value, trace) for seed in plan['seeds'] for value in values for trace in (0, -1)}
    require(len(numeric) == len(formal) == 12 and
            {(r['seed'], r[axis], r['trace']) for r in numeric} == expected and
            {(r['seed'], r[axis], r['trace']) for r in formal} == expected, '数值／正式四臂不完整')
    by_condition = {(r['seed'], r[axis], r['trace']): r for r in formal}
    tolerance = plan['gates']['tolerance']
    windows, paired = {}, {}
    for trace in (0, -1):
        windows[str(trace)] = {str(value): {unit: spread([
            by_condition[seed, value, trace]['metrics'][unit] for seed in plan['seeds']], tolerance)
            for unit in ('cycles', 'ns')} for value in values}
        differences = {unit: [by_condition[seed, changed_value, trace]['metrics'][unit]-
                              by_condition[seed, base_value, trace]['metrics'][unit]
                              for seed in plan['seeds']] for unit in ('cycles', 'ns')}
        paired[str(trace)] = spread(differences['cycles'], tolerance)
        paired[str(trace)].update(ns_values=differences['ns'], ns_median=statistics.median(differences['ns']))
    prefix = {}
    for value in values:
        prefix_values = {unit: [by_condition[seed, value, -1]['prefix'][unit] for seed in plan['seeds']]
                  for unit in ('cycles', 'ns')}
        prefix[str(value)] = dict(cycles=spread(prefix_values['cycles'], tolerance),
                                  ns_values=prefix_values['ns'], ns_median=statistics.median(prefix_values['ns']))
    bridge = {value: {unit: relative_check(windows['0'][value][unit]['median'], reference[unit], tolerance)
                      for unit in ('cycles', 'ns')} for value, reference in plan['reference_windows'].items()}
    prefix_bridge = {value: relative_check(prefix[value]['cycles']['median'], reference, tolerance)
                     for value, reference in plan.get('reference_prefix_cycles', {}).items()}
    pair_bridge = {unit: relative_check(paired['0']['median'] if unit == 'cycles' else paired['0']['ns_median'],
                                       plan['reference_pair'][unit], tolerance) for unit in ('cycles', 'ns')}
    observer, response_checks, responses = [], [], []
    for seed in plan['seeds']:
        for value in values:
            plain, sparse = by_condition[seed, value, 0], by_condition[seed, value, -1]
            observer.append(dict(seed=seed, **{'lda' if axis == 'lda' else 'condition': value}, checks={unit: relative_check(
                sparse['metrics'][unit], plain['metrics'][unit], tolerance) for unit in ('cycles', 'ns')}))
        plain_delta = by_condition[seed, changed_value, 0]['metrics']['cycles']-by_condition[seed, base_value, 0]['metrics']['cycles']
        response = dict(seed=seed)
        for unit in ('cycles', 'ns'):
            changed, base = by_condition[seed, changed_value, -1], by_condition[seed, base_value, -1]
            for label, field in (('prefix', 'prefix'), ('suffix', 'suffix'), ('full', 'metrics')):
                response['delta_'+label+'_'+unit] = changed[field][unit]-base[field][unit]
            response['identity_'+unit] = (response['delta_prefix_'+unit]+response['delta_suffix_'+unit] ==
                                         response['delta_full_'+unit])
        responses.append(response)
        response_checks.append(dict(seed=seed, check=relative_check(response['delta_full_cycles'], plain_delta, tolerance)))
    qualified = (all(s['passed'] for traces in windows.values() for window in traces.values() for s in window.values()) and
                 all(p['cycles']['passed'] for p in prefix.values()) and
                 all(p['passed'] for p in paired.values()) and
                 all(c['passed'] for w in bridge.values() for c in w.values()) and
                 all(c['passed'] for c in prefix_bridge.values()) and
                 all(c['passed'] for c in pair_bridge.values()) and
                 all(c['passed'] for o in observer for c in o['checks'].values()) and
                 all(r['check']['passed'] for r in response_checks) and
                 all(r['identity_'+unit] for r in responses for unit in ('cycles', 'ns')))
    zero_band = plan['gates']['zero_band_fraction']*prefix[str(base_value)]['cycles']['median']
    delta_prefix = [r['delta_prefix_cycles'] for r in responses]
    position = ('within_zero_band' if all(abs(v) <= zero_band for v in delta_prefix) else
                'positive_prefill_response' if all(v > zero_band for v in delta_prefix) else 'signed_or_ambiguous')
    diagnostics = []
    for point in predictions['diagnostic_locations']:
        diagnostics.append(dict(name=point['name'], frozen_point=point,
            residuals=[dict(seed=r['seed'], prefix_cycles=r['delta_prefix_cycles']-point['prefix_cycles'],
                            suffix_cycles=r['delta_suffix_cycles']-point['suffix_cycles']) for r in responses]))
    report = dict(schema=plan['schema'], decision='development_localization_complete' if qualified else 'information_insufficient_stop',
                qualification=plan['qualification'], observation_qualified=qualified, passed=qualified,
                numeric=numeric, formal=formal, windows=windows, paired=paired, prefix=prefix,
                plain_bridge=bridge, plain_pair_bridge=pair_bridge, observer_checks=observer,
                sparse_response_checks=response_checks, responses=responses,
                position=dict(classification=position, zero_band_cycles=zero_band, delta_prefix_cycles=delta_prefix),
                diagnostic_residuals=diagnostics, gates=plan['gates'], **FLAGS)
    if axis == 'request_mask':
        report.update(comparison_axis=axis, axis_values=values, prefix_bridge=prefix_bridge)
    return report


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
                print(json.dumps(dict(role=task['role'], seed=task['seed'], arms=4)), flush=True)
        report = summarize(plan, predictions, rows)
    except Exception as error:
        save_json(root/'score-heldout.json', dict(schema=plan['schema'], decision='information_insufficient_stop',
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
    if args.mode == 'execute':
        execute(args.root.resolve())
    else:
        require(args.output is not None, 'score 需要 --output')
        save_json(args.output, score(args.root.resolve()))


if __name__ == '__main__':
    main()

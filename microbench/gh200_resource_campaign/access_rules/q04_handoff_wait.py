#!/usr/bin/env python3
"""驻留B/C短链的同binary full等待干预；复用原数值及计时检查。"""
import argparse
from pathlib import Path
import statistics
import time
import re

import cfg_b_handoff_chain as witness
import q04_index_control as reference
from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json


def build(root):
    for name, checksum in read_json(root/'build-inputs.json')['files'].items():
        require(digest(root/name) == checksum, '构建输入变化：'+name)
    reference.build(root)
    functions = []
    for section in re.split(r'Function\s*:\s*', (root/'build/disassemble/stdout').read_text())[1:]:
        symbol = section.splitlines()[0].strip()
        if not symbol.startswith('_Z13handoff_chainILb0ELb1ELb0'):
            continue
        instructions = [m[1].strip() for line in section.splitlines()[1:]
                        if (m := re.match(r'\s*/\*[0-9a-f]+\*/\s+([^;]+);', line))]
        functions.append(dict(symbol=symbol, instruction_count=len(instructions),
            compute=[x for x in instructions if x.startswith('HGMMA.')],
            compute_wait=[x for x in instructions if x.startswith('WARPGROUP.DEPBAR')],
            full_wait=[x for x in instructions if 'SYNCS.PHASECHK' in x or 'SYNCS.TRYWAIT' in x],
            polling_yield=[x for x in instructions if 'YIELD' in x]))
    require(len(functions) == 1 and len(functions[0]['compute']) == 8 and
            len(functions[0]['compute_wait']) == 2 and
            any('TRYWAIT' in x for x in functions[0]['full_wait']), '计算／等待路径计数不符')
    save_json(root/'build/local-wait-audit.json', dict(functions=functions,
        probe_sha256=digest(root/'build/probe'), sass_sha256=digest(root/'build/disassemble/stdout'),
        source_sha256=digest(root/'source/cfg_b_handoff_chain.cu'),
        main_thread_control_review='待核对同地址phase及共用计算／释放／输出'))


def load_plan(root):
    plan = read_json(root/'plan.json')
    for name, checksum in read_json(root/'execution-binding.json')['files'].items():
        require(digest(root/name) == checksum, '执行输入变化：'+name)
    require(read_json(root/'approved-target.json')['approved'], '目标构建未获核对')
    return plan


def validate(folder, case, seed, uuid):
    raw = read_json(folder/'stdout')
    receipt = read_json(folder/'receipt.json')
    require(receipt['returncode'] == 0 and not receipt['timed_out'], '原始执行未成功')
    checked = witness.validate(folder, raw, case, seed)
    require(checked['performance_eligible'], '暖机资格未通过')
    require(raw['index_mode'] == 'incremental' and raw['device']['uuid'] == uuid, '程序或卡身份不符')
    if case['binary'] == 'runtime':
        require(raw['full_wait_policy'] == case['wait_policy'] and
                raw['warmup_full_wait_policy'] == 0, '正式或暖机等待方式不符')
    return dict(case_id=case['id'], seed=seed, binary=case['binary'], wait_policy=case['wait_policy'],
        delay_cycles=case['delay_cycles'], cycles=raw['timing']['elapsed_cycles'],
        ns=raw['timing']['elapsed_ns'], smid=raw['timing']['smid_start'], resources=raw['resources'],
        output_witness=checked['output_witness'])


def summarize(root):
    plan = load_plan(root)
    rows = []
    for task in plan['order']:
        case = next(row for row in plan['cases'] if row['id'] == task['case_id'])
        rows.append(validate(root/'runs'/case['id']/f"seed{task['seed']}", case, task['seed'], plan['gpu_uuid']))
    conditions, responses = [], []
    for case in plan['cases']:
        group = [row for row in rows if row['case_id'] == case['id']]
        windows = {}
        for unit in ('cycles', 'ns'):
            values = [row[unit] for row in group]
            median = statistics.median(values)
            windows[unit] = dict(values=values, median=median,
                relative_spread=(max(values)-min(values))/median,
                passed=max(values)-min(values) <= .05*median)
        conditions.append(dict(case_id=case['id'], windows=windows))
    for binary, policy in (('original', 0), ('runtime', 0), ('runtime', 1)):
        pairs = []
        for seed in plan['seeds']:
            paired = [next(row for row in rows if row['binary'] == binary and
                row['wait_policy'] == policy and row['seed'] == seed and row['delay_cycles'] == delay)
                for delay in (0, 4096)]
            pairs.append(dict(seed=seed, **{unit:paired[1][unit]-paired[0][unit] for unit in ('cycles', 'ns')}))
        responses.append(dict(binary=binary, wait_policy=policy, pairs=pairs,
            medians={unit:statistics.median(pair[unit] for pair in pairs) for unit in ('cycles', 'ns')}))
    bridge = []
    medians = {row['case_id']:row['windows'] for row in conditions}
    for case in plan['cases']:
        if case['binary'] == 'original':
            expected = plan['old_windows'][str(case['delay_cycles'])]
            label = 'original_vs_saved'
        elif case['wait_policy'] == 0:
            expected = {unit:medians[f"original_d{case['delay_cycles']}"][unit]['median'] for unit in ('cycles', 'ns')}
            label = 'runtime_blocking_vs_original'
        else:
            continue
        for unit in ('cycles', 'ns'):
            observed = medians[case['id']][unit]['median']
            bridge.append(dict(kind=label, case_id=case['id'], unit=unit, expected=expected[unit],
                observed=observed, relative_error=abs(observed-expected[unit])/expected[unit],
                passed=abs(observed-expected[unit]) <= .05*expected[unit]))
    for actual, expected, label in ((responses[0]['medians'], plan['old_response'], 'original_response_vs_saved'),
                                     (responses[1]['medians'], responses[0]['medians'], 'runtime_blocking_response_vs_original')):
        for unit in ('cycles', 'ns'):
            bridge.append(dict(kind=label, unit=unit, observed=actual[unit], expected=expected[unit],
                relative_error=None if expected[unit] == 0 else abs(actual[unit]-expected[unit])/abs(expected[unit]),
                passed=abs(actual[unit]-expected[unit]) <= .05*abs(expected[unit])))
    qualified = all(window['passed'] for row in conditions for window in row['windows'].values()) and all(row['passed'] for row in bridge)
    observed = responses[2]['medians']['cycles']
    hypotheses = [dict(**candidate, observed_cycle=observed,
        matched=qualified and abs(observed-candidate['prediction_cycle']) <= .05*candidate['prediction_cycle'])
        for candidate in plan['hypotheses']]
    matched = [row['id'] for row in hypotheses if row['matched']]
    return dict(qualification='局部冻结开发判别；跨stage／长度验证未执行',
        qualified=qualified, decision=matched[0] if len(matched) == 1 else 'information_insufficient',
        hypotheses=hypotheses, conditions=conditions, responses=responses, bridge=bridge, rows=rows,
        output_elements_checked=sum(row['output_witness']['checked_elements'] for row in rows),
        independent_C2=False, independent_C3=False, full_gemm_C2=False, full_gemm_C3=False,
        plan_sha256=digest(root/'plan.json'), probe_sha256=digest(root/'build/probe'))


def execute(root):
    plan = load_plan(root)
    started = time.monotonic()
    allocation = inspect_allocated_gpu(root/'allocation')
    require(allocation['uuid'] == plan['gpu_uuid'], '分配GPU不符')
    completed = 0
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        try:
            for task in plan['order']:
                require(time.monotonic()-started < 510, '本轮时间预算耗尽')
                case = next(row for row in plan['cases'] if row['id'] == task['case_id'])
                binary = root/('build/probe' if case['binary'] == 'runtime' else 'reference/probe')
                command = witness.probe_command(binary, case, task['seed']) + ['--index-mode', 'incremental']
                if case['binary'] == 'runtime':
                    command += ['--full-wait-policy', str(case['wait_policy'])]
                folder = root/'runs'/case['id']/f"seed{task['seed']}"
                command_record(folder, command, 60)
                row = validate(folder, case, task['seed'], plan['gpu_uuid'])
                completed += 1
                print(f"{case['id']} seed{task['seed']}: {row['cycles']} cycle", flush=True)
            save_json(root/'score.json', summarize(root))
        except Exception as error:
            save_json(root/'failure.json', dict(completed_processes=completed, error=str(error),
                elapsed_seconds=time.monotonic()-started))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'execute', 'score'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'build':
        build(root)
    elif args.command == 'execute':
        execute(root)
    else:
        require(args.output is not None, 'score需要输出路径')
        save_json(args.output, summarize(root))


if __name__ == '__main__':
    main()

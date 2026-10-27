#!/usr/bin/env python3
"""原生B/D短链的固定控制、长度增长配对。"""
import argparse
import json
from pathlib import Path
import statistics
import time

from cfg_b_handoff_chain import timed_lifecycle, probe_command
from cfg_b_tma_chain import timed_lifecycle as input_lifecycle
from q02_native_tma_chain import validate_d
from q04_native_compute_wait import validate as validate_b
from l2_saturation import (command_record, digest, exclusive_lock,
                           inspect_allocated_gpu, read_json, require, save_json)


def predict(root):
    service = read_json(root/'reference/service.json')
    parameters = service['parameters']
    windows, bodies = {}, {}
    calls = 0
    for ktiles in (4, 13):
        b = timed_lifecycle(ktiles, 6, 0, **parameters)
        d = input_lifecycle(ktiles, 6,
            request_service=lambda t, r: dict(source_ready=t, source_id=3*r['tile']+
                ('A','B0','B1').index(r['operand']), source_key=(r['operand'],r['source_k_tile']),
                merged=False, source_busy_ns=0., service_start=t, service_end=t, issue_return_ns=t),
            compute_ns=parameters['compute_cycles'], commit_lead_ns=parameters['commit_lead_cycles'],
            completion_lag_ns=parameters['completion_lag_cycles'],
            loop_advance_ns=parameters['consumer_advance_cycles'],
            release_visibility_ns=parameters['release_visibility_cycles'],
            source_tiles=32, source_profile='generation')
        calls += 2
        require(b['local_end_cycle'] == d['tail_wait0_and_release_ns'], '零源B/D末端不同')
        bodies[str(ktiles)] = b['local_end_cycle']
        windows['b_k'+str(ktiles)] = b['local_end_cycle']+service['common_window_cycle']
    growth = bodies['13']-bodies['4']
    require(growth == 4608, '登记增长与固定控制模型不符')
    return dict(unit='SM clock64 cycle', parameters=parameters, zero_source_bodies=bodies,
        b_window_cycle=windows, growth_cycle=growth, tolerance_cycle=.05*growth,
        difference_of_growths_cycle=0., common_D_window='not_calibrated',
        model_calls=calls, algebra_calibrations=0, independent_C2=False, independent_C3=False)


def summarize(root, rows):
    plan, freeze = read_json(root/'plan.json'), read_json(root/'freeze.json')
    complete = len(rows) == 12
    ids = ['b_k4','b_k13']+(['d_k4','d_k13'] if complete else [])
    windows, qualification = {}, []
    for key in ids:
        group = [r for r in rows if r['case_id'] == key]
        require(sorted(r['seed'] for r in group) == plan['seeds'], '进程不齐或重复')
        windows[key] = {u:statistics.median(r[u] for r in group) for u in ('cycles','ns')}
        for unit in ('cycles','ns'):
            values = [r[unit] for r in group]
            qualification.append(dict(kind='stability', case_id=key, unit=unit,
                observed_spread=max(values)-min(values), tolerance=.05*windows[key][unit],
                passed=max(values)-min(values) <= .05*windows[key][unit]))
    for unit in ('cycles','ns'):
        expected = plan['saved_b_k13'][unit]
        actual = windows['b_k13'][unit]
        qualification.append(dict(kind='B_bridge', unit=unit, observed=actual, expected=expected,
            passed=abs(actual-expected) <= .05*expected))
    for key in ('b_k4','b_k13'):
        expected = freeze['b_window_cycle'][key]
        actual = windows[key]['cycles']
        qualification.append(dict(kind='B_local_window', case_id=key, observed=actual,
            expected=expected, passed=abs(actual-expected) <= .05*expected))
    growths = {}
    for arm in ('b','d') if complete else ('b',):
        growths[arm] = {unit:statistics.median(
            next(r[unit] for r in rows if r['case_id']==arm+'_k13' and r['seed']==seed)-
            next(r[unit] for r in rows if r['case_id']==arm+'_k4' and r['seed']==seed)
            for seed in plan['seeds']) for unit in ('cycles','ns')}
    qualification.append(dict(kind='B_length_growth', observed=growths['b']['cycles'],
        expected=freeze['growth_cycle'], passed=abs(growths['b']['cycles']-freeze['growth_cycle'])
            <= freeze['tolerance_cycle']))
    qualified = all(c['passed'] for c in qualification)
    checks, excess = [], None
    if complete:
        observed = growths['d']['cycles']
        excess = growths['d']['cycles']-growths['b']['cycles']
        checks = [dict(endpoint='D_length_growth', observed_cycle=observed,
            predicted_cycle=freeze['growth_cycle'], tolerance_cycle=freeze['tolerance_cycle'],
            passed=abs(observed-freeze['growth_cycle']) <= freeze['tolerance_cycle']),
            dict(endpoint='D_minus_B_growth', observed_cycle=excess, predicted_cycle=0.,
            tolerance_cycle=freeze['tolerance_cycle'], passed=abs(excess) <= freeze['tolerance_cycle'])]
    retained = qualified and complete and all(c['passed'] for c in checks)
    return dict(qualified=qualified, complete=complete, qualification=qualification,
        windows=windows, growths=growths, growth_excess_cycle=excess, hidden_input_checks=checks,
        decision='information_insufficient' if not qualified else
            'input_hidden_development_supported' if retained else
            'input_hidden_form_not_passed' if complete else 'B_control_qualified',
        checked_output_elements=sum(r['output_elements_checked'] for r in rows),
        generation_checks=sum(r['generation_checks'] for r in rows), rows=rows,
        independent_C2=False, independent_C3=False, full_gemm_C2=False, full_gemm_C3=False)


def validate(folder, case, seed, uuid):
    if case['arm'] == 'D':
        return validate_d(folder, case, seed, uuid)
    row = validate_b(folder, case, seed, uuid)
    return dict(case_id=case['id'], seed=seed, cycles=row['cycles'], ns=row['ns'], smid=row['smid'],
        output_elements_checked=row['output_elements'], generation_checks=row['generation_checked'])


def replay(root, data_root):
    plan = read_json(root/'plan.json')
    rows = []
    for arm in ('B','D'):
        for trial, seed in enumerate(plan['seeds']):
            cases = [c for c in plan['cases'] if c['arm']==arm]
            for case in cases[trial % 2:]+cases[:trial % 2]:
                folder = data_root/'runs'/case['id']/f'seed{seed}'
                if folder.exists():
                    rows.append(validate(folder,case,seed,plan['gpu_uuid']))
    return summarize(root, rows)


def execute(root):
    for name, sha in read_json(root/'execution-binding.json')['files'].items():
        require(digest(root/name)==sha, '冻结输入变化：'+name)
    plan = read_json(root/'plan.json')
    require(read_json(root/'build/target-review.json')['passed'], 'D构建审核失败')
    require(read_json(root/'build/approved-target.json')['approved'], 'D目标核对缺失')
    allocation = inspect_allocated_gpu(root/'allocation')
    require(allocation['uuid']==plan['gpu_uuid'], '卡身份不符')
    rows = []
    started = time.monotonic()
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        try:
            for arm in ('B','D'):
                if arm == 'D':
                    initial = summarize(root, rows)
                    save_json(root/'initial-score.json',initial)
                    require(initial['qualified'], 'B同批控制失败，停止D')
                for trial, seed in enumerate(plan['seeds']):
                    cases = [c for c in plan['cases'] if c['arm']==arm]
                    for case in cases[trial % 2:]+cases[:trial % 2]:
                        require(time.monotonic()-started < 510, '本轮GPU预算耗尽')
                        binary = root/('reference/probe' if arm=='B' else 'build/probe')
                        argv = probe_command(binary,case,seed)+['--index-mode','incremental'] if arm=='B' else \
                            [binary,'--stages',case['stages'],'--ktiles',case['ktiles'],
                             '--trace','0','--seed',seed,'--warmup-limit','60']
                        folder = root/'runs'/case['id']/f'seed{seed}'
                        command_record(folder,list(map(str,argv)),60)
                        rows.append(validate(folder,case,seed,plan['gpu_uuid']))
                        print(f"{case['id']} seed{seed}: {rows[-1]['cycles']} cycle",flush=True)
            save_json(root/'score.json',summarize(root,rows))
        except Exception as error:
            save_json(root/'failure.json',dict(error=str(error),completed_processes=len(rows),rows=rows))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('predict','execute','score'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'execute':
        execute(root)
    else:
        result = predict(root) if args.command=='predict' else replay(root,args.data_root.resolve())
        save_json(args.output,result)


if __name__ == '__main__':
    main()

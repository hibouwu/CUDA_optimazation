#!/usr/bin/env python3
"""B/D首次等待局部预测类的固定延迟外推验证。"""
import argparse
from pathlib import Path
import statistics
import time

from q04_first_entry_control import validate
from l2_saturation import (command_record, digest, exclusive_lock,
                           inspect_allocated_gpu, read_json, require, save_json)


def score(root, rows):
    plan = read_json(root/'plan.json')
    complete = len(rows) == plan['budgets']['processes']
    cases = [c for c in plan['cases'] if complete or c['split'] == 'dev']
    windows, qualification = {}, []
    for case in cases:
        group = [r for r in rows if r['case_id'] == case['id']]
        require(sorted(r['seed'] for r in group) == plan['seeds'], '进程不齐或重复')
        windows[case['id']] = {u: statistics.median(r[u] for r in group)
                               for u in ('cycles', 'ns')}
        for unit in ('cycles', 'ns'):
            values = [r[unit] for r in group]
            qualification.append(dict(kind='stability', case_id=case['id'], unit=unit,
                passed=max(values)-min(values) <= .05*windows[case['id']][unit]))
    for case_id, saved in plan['saved_windows'].items():
        for unit, expected in saved.items():
            actual = windows[case_id][unit]
            qualification.append(dict(kind='saved_bridge', case_id=case_id, unit=unit,
                observed=actual, expected=expected, passed=abs(actual-expected) <= .05*expected))
    qualified = all(c['passed'] for c in qualification)
    responses, pairs, checks = {}, [], []
    if complete:
        for seed in plan['seeds']:
            row = {r['case_id']: r for r in rows if r['seed'] == seed}
            b = row['b1536']['cycles']-row['b0']['cycles']
            d = row['d1536']['cycles']-row['d0']['cycles']
            pairs.append(dict(seed=seed, B_response_cycle=b, D_response_cycle=d, gap_cycle=b-d))
        responses = dict(B_response_cycle=statistics.median(r['B_response_cycle'] for r in pairs),
                         D_response_cycle=statistics.median(r['D_response_cycle'] for r in pairs),
                         gap_cycle=statistics.median(r['gap_cycle'] for r in pairs))
        observed = dict(responses, B_window_cycle=windows['b1536']['cycles'],
                        D_window_cycle=windows['d1536']['cycles'])
        for endpoint, expected in plan['predictions'].items():
            actual = observed[endpoint]
            checks.append(dict(endpoint=endpoint, observed=actual, expected=expected,
                tolerance=.05*expected, passed=abs(actual-expected) <= .05*expected))
    passed = qualified and complete and all(c['passed'] for c in checks)
    return dict(complete=complete, qualified=qualified, qualification=qualification,
        windows=windows, pairs=pairs, responses=responses, checks=checks, rows=rows,
        decision='information_insufficient' if not qualified else 'development_qualified' if not complete
                 else 'local_prediction_class_supported' if passed else 'local_prediction_class_not_passed',
        checked_output_elements=sum(r['output_elements_checked'] for r in rows),
        generation_checks=sum(r['generation_checks'] for r in rows),
        internal_placement_C2=False, full_gemm_C2=False, full_gemm_C3=False)


def replay(root, data_root):
    plan = read_json(root/'plan.json')
    rows = []
    for split in ('dev', 'heldout'):
        cases = [c for c in plan['cases'] if c['split'] == split]
        for trial, seed in enumerate(plan['seeds']):
            shift = trial % len(cases)
            for case in cases[shift:]+cases[:shift]:
                folder = data_root/'runs'/case['id']/f'seed{seed}'
                if folder.exists():
                    rows.append(validate(folder, case, seed, plan))
    return score(root, rows)


def execute(root):
    for name, sha in read_json(root/'execution-binding.json')['files'].items():
        require(digest(root/name) == sha, '冻结输入变化：'+name)
    plan = read_json(root/'plan.json')
    allocation = inspect_allocated_gpu(root/'allocation')
    require(allocation['uuid'] == plan['gpu_uuid'], '卡身份不符')
    rows, started = [], time.monotonic()
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        try:
            for split in ('dev', 'heldout'):
                if split == 'heldout':
                    initial = score(root, rows)
                    save_json(root/'initial-score.json', initial)
                    if not initial['qualified']:
                        save_json(root/'score.json', initial)
                        return
                cases = [c for c in plan['cases'] if c['split'] == split]
                for trial, seed in enumerate(plan['seeds']):
                    shift = trial % len(cases)
                    for case in cases[shift:]+cases[:shift]:
                        require(time.monotonic()-started < 510, 'GPU时间预算耗尽')
                        binary = root/('reference/d_probe' if case['program'] == 'D' else 'build/probe')
                        argv = [binary, '--stages', plan['stages'], '--ktiles', plan['ktiles'],
                            '--trace', '0', '--seed', seed, '--warmup-limit', '60',
                            '--consumer-entry-delay-cycles', case['entry_delay_cycles']]
                        if case['program'] == 'B':
                            argv += ['--delay-cycles', '0', '--index-mode', 'incremental']
                        folder = root/'runs'/case['id']/f'seed{seed}'
                        command_record(folder, list(map(str, argv)), 60)
                        rows.append(validate(folder, case, seed, plan))
                        print(f"{case['id']} seed{seed}: {rows[-1]['cycles']} cycle", flush=True)
            save_json(root/'score.json', score(root, rows))
        except Exception as error:
            save_json(root/'failure.json', dict(error=str(error), completed_processes=len(rows), rows=rows))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    execute(args.root.resolve())

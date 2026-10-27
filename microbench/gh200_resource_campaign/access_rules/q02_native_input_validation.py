#!/usr/bin/env python3
"""原生D的冻结stage／长度局部外推；采样不调用模型。"""
import argparse
from pathlib import Path
import statistics
import time

from cfg_b_handoff_chain import probe_command
from q02_native_input_growth import validate
from l2_saturation import (command_record, digest, exclusive_lock,
                           inspect_allocated_gpu, read_json, require, save_json)


def score(root, rows):
    plan, freeze = read_json(root/'plan.json'), read_json(root/'freeze.json')
    complete = len(rows) == 15
    cases = [c for c in plan['cases'] if complete or c['split']=='dev']
    windows, checks = {}, []
    for case in cases:
        group = [r for r in rows if r['case_id']==case['id']]
        require(sorted(r['seed'] for r in group)==plan['seeds'], '进程不齐或重复')
        windows[case['id']] = {u:statistics.median(r[u] for r in group) for u in ('cycles','ns')}
        for unit in ('cycles','ns'):
            values = [r[unit] for r in group]
            checks.append(dict(kind='stability',case_id=case['id'],unit=unit,
                passed=max(values)-min(values)<=.05*windows[case['id']][unit]))
    for key, saved in freeze['saved_windows'].items():
        for unit in ('cycles','ns'):
            observed, expected = windows[key][unit], saved[unit]
            checks.append(dict(kind='same_batch_bridge',case_id=key,unit=unit,
                observed=observed,expected=expected,passed=abs(observed-expected)<=.05*expected))
    def response(after,before):
        return statistics.median(next(r['cycles'] for r in rows if r['case_id']==after and r['seed']==seed)-
            next(r['cycles'] for r in rows if r['case_id']==before and r['seed']==seed) for seed in plan['seeds'])
    development = []
    for key in ('d_short','d_base'):
        predicted, actual = freeze['window_cycle'][key], windows[key]['cycles']
        development.append(dict(endpoint=key,predicted_cycle=predicted,observed_cycle=actual,
            passed=abs(actual-predicted)<=.05*predicted))
    actual, predicted = response('d_base','d_short'), freeze['dev_growth_cycle']
    development.append(dict(endpoint='development_growth',predicted_cycle=predicted,
        observed_cycle=actual,passed=abs(actual-predicted)<=.05*predicted))
    heldout = []
    if complete:
        for key in ('d_stage4','d_long'):
            predicted, actual = freeze['window_cycle'][key], windows[key]['cycles']
            heldout.append(dict(endpoint=key,predicted_cycle=predicted,observed_cycle=actual,
                error_percent=100*abs(actual-predicted)/predicted,
                passed=abs(actual-predicted)<=.05*predicted))
        for key,after,before in (('stage_response','d_stage4','d_base'),('length_growth','d_long','d_base')):
            actual, predicted = response(after,before), freeze['responses_cycle'][key]
            tolerance = .05*(abs(predicted) if predicted else freeze['window_cycle']['d_base'])
            heldout.append(dict(endpoint=key,predicted_cycle=predicted,observed_cycle=actual,
                tolerance_cycle=tolerance,passed=abs(actual-predicted)<=tolerance))
    qualified, developed = all(c['passed'] for c in checks), all(c['passed'] for c in development)
    passed = qualified and developed and complete and all(c['passed'] for c in heldout)
    return dict(complete=complete,qualified=qualified,qualification=checks,windows=windows,
        development_passed=developed,development_checks=development,heldout_checks=heldout,
        decision='information_insufficient' if not qualified else
            'development_form_not_passed' if not developed else
            'local_prediction_class_supported' if passed else
            'local_prediction_class_not_passed' if complete else 'development_qualified',
        rows=rows,checked_output_elements=sum(r['output_elements_checked'] for r in rows),
        generation_checks=sum(r['generation_checks'] for r in rows),
        independent_local_cycle_validation=passed,independent_local_discrimination=False,
        full_gemm_C2=False,full_gemm_C3=False)


def replay(root, data_root):
    plan = read_json(root/'plan.json')
    rows = []
    for split in ('dev','heldout'):
        cases = [c for c in plan['cases'] if c['split']==split]
        for trial,seed in enumerate(plan['seeds']):
            shift = trial % len(cases)
            for case in cases[shift:]+cases[:shift]:
                folder = data_root/'runs'/case['id']/f'seed{seed}'
                if folder.exists():
                    rows.append(validate(folder,case,seed,plan['gpu_uuid']))
    return score(root,rows)


def execute(root):
    for name,sha in read_json(root/'execution-binding.json')['files'].items():
        require(digest(root/name)==sha,'冻结输入变化：'+name)
    require(read_json(root/'build/target-review.json')['passed'],'D目标构建未通过')
    plan = read_json(root/'plan.json')
    allocation = inspect_allocated_gpu(root/'allocation')
    require(allocation['uuid']==plan['gpu_uuid'],'卡身份不符')
    rows = []
    started = time.monotonic()
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        try:
            for split in ('dev','heldout'):
                if split=='heldout':
                    initial = score(root,rows)
                    save_json(root/'initial-score.json',initial)
                    require(initial['qualified'] and initial['development_passed'],'开发资格或形式失败，停止留出')
                cases = [c for c in plan['cases'] if c['split']==split]
                for trial,seed in enumerate(plan['seeds']):
                    shift = trial % len(cases)
                    for case in cases[shift:]+cases[:shift]:
                        require(time.monotonic()-started<510,'GPU时间预算耗尽')
                        binary = root/('reference/probe' if case['arm']=='B' else 'build/probe')
                        argv = probe_command(binary,case,seed)+['--index-mode','incremental'] if case['arm']=='B' else \
                            [binary,'--stages',case['stages'],'--ktiles',case['ktiles'],
                             '--trace','0','--seed',seed,'--warmup-limit','60']
                        folder = root/'runs'/case['id']/f'seed{seed}'
                        command_record(folder,list(map(str,argv)),60)
                        rows.append(validate(folder,case,seed,plan['gpu_uuid']))
                        print(f"{case['id']} seed{seed}: {rows[-1]['cycles']} cycle",flush=True)
            save_json(root/'score.json',score(root,rows))
        except Exception as error:
            save_json(root/'failure.json',dict(error=str(error),completed_processes=len(rows),rows=rows))
            raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    execute(args.root.resolve())

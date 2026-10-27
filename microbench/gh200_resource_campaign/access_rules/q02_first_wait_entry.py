#!/usr/bin/env python3
"""原生D首次full等待前的软件延迟；固定两种窗口放置。"""
import argparse
from pathlib import Path
import statistics
import time

from q02_native_tma_chain import validate_d
from l2_saturation import (command_record, digest, exclusive_lock,
                           inspect_allocated_gpu, read_json, require, save_json)


def validate(folder, case, seed, plan):
    values = validate_d(folder, dict(id=case['id'], stages=plan['stages'],
        ktiles=plan['ktiles']), seed, plan['gpu_uuid'])
    raw = read_json(folder/'stdout')
    require(raw['resources']['registers_per_thread']==167 and
        raw['resources']['static_smem_bytes']==1152, '构建资源不符')
    if case['binary']=='new':
        require(raw['conditions']['consumer_entry_delay_cycles']==case['entry_delay_cycles'],
            '首次等待延迟不符')
    return values


def score(root, rows):
    plan, freeze = read_json(root/'plan.json'), read_json(root/'freeze.json')
    complete = len(rows)==plan['budgets']['processes']
    cases = [c for c in plan['cases'] if complete or c['split']=='dev']
    windows, qualification = {}, []
    for case in cases:
        group = [r for r in rows if r['case_id']==case['id']]
        require(sorted(r['seed'] for r in group)==plan['seeds'], '进程不齐或重复')
        windows[case['id']] = {u:statistics.median(r[u] for r in group) for u in ('cycles','ns')}
        for unit in ('cycles','ns'):
            values = [r[unit] for r in group]
            qualification.append(dict(kind='stability',case_id=case['id'],unit=unit,
                passed=max(values)-min(values)<=.05*windows[case['id']][unit]))
    for unit, saved in read_json(root/'execution-policy.json')['saved_old0'].items():
        actual = windows['old0'][unit]
        qualification.append(dict(kind='saved_bridge',unit=unit,observed=actual,
            expected=saved,passed=abs(actual-saved)<=.05*saved))
        old, new = windows['old0'][unit], windows['new0'][unit]
        qualification.append(dict(kind='same_batch_bridge',unit=unit,observed=new,
            expected=old,passed=abs(new-old)<=.05*old))

    def response(key):
        return statistics.median(next(r['cycles'] for r in rows if r['case_id']==key and r['seed']==seed)-
            next(r['cycles'] for r in rows if r['case_id']=='new0' and r['seed']==seed)
            for seed in plan['seeds'])

    models = {}
    for name, model in freeze['models'].items():
        checks = []
        for case_id, delay, split in [('new0','0','dev'),('new4096','4096','dev'),
                                       ('new2048','2048','heldout')]:
            if split=='heldout' and not complete:
                continue
            actual, predicted = windows[case_id]['cycles'], model['window_cycle'][delay]
            checks.append(dict(kind='window',case_id=case_id,split=split,observed_cycle=actual,
                predicted_cycle=predicted,passed=abs(actual-predicted)<=.05*predicted))
            if delay!='0':
                actual, predicted = response(case_id), model['responses_cycle'][delay]
                checks.append(dict(kind='response',case_id=case_id,split=split,observed_cycle=actual,
                    predicted_cycle=predicted,passed=abs(actual-predicted)<=.05*predicted))
        models[name] = dict(checks=checks,development_passed=all(c['passed'] for c in checks if c['split']=='dev'),
            all_passed=all(c['passed'] for c in checks))
    qualified = all(c['passed'] for c in qualification)
    survivors = [name for name, model in models.items() if model['development_passed']]
    supported = [name for name, model in models.items() if complete and qualified and model['all_passed']]
    return dict(complete=complete,qualified=qualified,qualification=qualification,windows=windows,
        response4096_cycle=response('new4096'),response2048_cycle=response('new2048') if complete else None,
        models=models,development_survivors=survivors,supported_models=supported,
        decision='information_insufficient' if not qualified else
            'both_development_forms_not_passed' if not survivors else
            'local_placement_supported' if len(supported)==1 else
            'local_forms_not_passed' if complete and not supported else
            'local_forms_undistinguished' if complete else 'development_qualified',
        rows=rows,checked_output_elements=sum(r['output_elements_checked'] for r in rows),
        generation_checks=sum(r['generation_checks'] for r in rows),
        full_gemm_C2=False,full_gemm_C3=False)


def replay(root, data_root):
    plan = read_json(root/'plan.json')
    rows = []
    for split in ('dev','heldout'):
        cases = [c for c in plan['cases'] if c['split']==split]
        for trial, seed in enumerate(plan['seeds']):
            shift = trial % len(cases)
            for case in cases[shift:]+cases[:shift]:
                folder = data_root/'runs'/case['id']/f'seed{seed}'
                if folder.exists():
                    rows.append(validate(folder,case,seed,plan))
    return score(root,rows)


def execute(root):
    for name,sha in read_json(root/'execution-binding.json')['files'].items():
        require(digest(root/name)==sha,'冻结输入变化：'+name)
    require(read_json(root/'build/approved-target.json')['approved'],'目标构建尚未批准')
    plan = read_json(root/'plan.json')
    allocation = inspect_allocated_gpu(root/'allocation')
    require(allocation['uuid']==plan['gpu_uuid'],'卡身份不符')
    rows, started = [], time.monotonic()
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        try:
            for split in ('dev','heldout'):
                if split=='heldout':
                    initial = score(root,rows)
                    save_json(root/'initial-score.json',initial)
                    if not initial['qualified'] or not initial['development_survivors']:
                        save_json(root/'score.json',initial)
                        return
                cases = [c for c in plan['cases'] if c['split']==split]
                for trial,seed in enumerate(plan['seeds']):
                    shift = trial % len(cases)
                    for case in cases[shift:]+cases[:shift]:
                        require(time.monotonic()-started<510,'GPU时间预算耗尽')
                        binary = root/('reference/probe' if case['binary']=='old' else 'build/probe')
                        argv = [binary,'--stages',plan['stages'],'--ktiles',plan['ktiles'],
                            '--trace','0','--seed',seed,'--warmup-limit','60']
                        if case['binary']=='new':
                            argv += ['--consumer-entry-delay-cycles',case['entry_delay_cycles']]
                        folder = root/'runs'/case['id']/f'seed{seed}'
                        command_record(folder,list(map(str,argv)),60)
                        rows.append(validate(folder,case,seed,plan))
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

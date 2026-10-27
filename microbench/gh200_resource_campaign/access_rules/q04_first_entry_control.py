#!/usr/bin/env python3
"""驻留B与原生D的首次等待延迟配对。"""
import argparse
from pathlib import Path
import re
import shutil
import statistics
import tarfile
import tempfile
import time

from q02_native_tma_chain import validate_d
from q04_native_compute_wait import target_function, validate as validate_b
from l2_saturation import (command_record, digest, exclusive_lock,
                           inspect_allocated_gpu, read_json, require, save_json)


def build(root):
    for name, sha in read_json(root/'build-inputs.json')['files'].items():
        require(digest(root/name)==sha, '构建输入变化：'+name)
    compiler = Path(shutil.which('nvcc')).resolve()
    folder = root/'build'
    command_record(folder/'version',[compiler,'--version'],20)
    flags = ['-std=c++17','-O3','-DNDEBUG','-lineinfo',
        '-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v',
        '-DGH200_CONSUMER_ENTRY_DELAY=1']
    with tempfile.TemporaryDirectory(prefix='q04-entry-headers-') as temp:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:
            archive.extractall(temp,filter='data')
        command_record(folder/'compile',[compiler,*flags,'-I'+temp+'/include',
            root/'source/cfg_b_handoff_chain.cu','-o',folder/'probe'],240)
    command_record(folder/'disassemble',[compiler.parent/'cuobjdump','--dump-sass',folder/'probe'],60)
    checked = target_function(folder)
    instructions = checked['instructions']
    mma = [x for x in instructions if 'HGMMA.' in x]
    waits = [x for x in instructions if 'WARPGROUP.DEPBAR' in x]
    passed = checked['resources']==[0,0,0,167,1152] and len(mma)==8 and len(waits)==2 and \
        any(x.endswith('0x1') for x in waits) and any(x.endswith('0x0') for x in waits) and \
        not any('UTMALDG.' in x or re.search(r'\b(?:LDL|STL)(?:\.|\s)',x) for x in instructions)
    save_json(folder/'target-review.json',dict(passed=passed,flags=flags,
        resources=checked['resources'],mma=mma,waits=waits,probe_sha256=digest(folder/'probe')))
    require(passed,'B首次等待目标审核失败')


def validate(folder, case, seed, plan):
    common = dict(id=case['id'],stages=plan['stages'],ktiles=plan['ktiles'])
    if case['program']=='D':
        values = validate_d(folder,common,seed,plan['gpu_uuid'])
    else:
        common.update(delay_cycles=0,trace=False,wait_retain=1,wgmma_count=8*plan['ktiles'],
            work_flop=2*128*128*64*plan['ktiles'])
        checked = validate_b(folder,common,seed,plan['gpu_uuid'])
        values = dict(case_id=case['id'],seed=seed,cycles=checked['cycles'],ns=checked['ns'],
            smid=checked['smid'],output_elements_checked=checked['output_elements'],
            generation_checks=checked['generation_checked'])
    raw = read_json(folder/'stdout')
    if case['binary']!='old':
        require(raw['conditions']['consumer_entry_delay_cycles']==case['entry_delay_cycles'],
            '首次等待延迟不符')
    return values


def score(root, rows):
    plan = read_json(root/'plan.json')
    complete = len(rows)==plan['budgets']['processes']
    cases = [c for c in plan['cases'] if complete or c['split']=='dev']
    windows, qualification = {}, []
    for case in cases:
        group = [r for r in rows if r['case_id']==case['id']]
        require(sorted(r['seed'] for r in group)==plan['seeds'],'进程不齐或重复')
        windows[case['id']] = {u:statistics.median(r[u] for r in group) for u in ('cycles','ns')}
        for unit in ('cycles','ns'):
            values = [r[unit] for r in group]
            qualification.append(dict(kind='stability',case_id=case['id'],unit=unit,
                passed=max(values)-min(values)<=.05*windows[case['id']][unit]))
    for case_id, saved in plan['saved_windows'].items():
        if case_id not in windows:
            continue
        for unit, expected in saved.items():
            actual = windows[case_id][unit]
            qualification.append(dict(kind='saved_bridge',case_id=case_id,unit=unit,
                observed=actual,expected=expected,passed=abs(actual-expected)<=.05*expected))
    for unit in ('cycles','ns'):
        old,new = windows['b_old0'][unit],windows['b_new0'][unit]
        qualification.append(dict(kind='B_zero_bridge',unit=unit,observed=new,expected=old,
            passed=abs(new-old)<=.05*old))
    responses, gaps = {}, []
    if complete:
        for seed in plan['seeds']:
            row = {r['case_id']:r for r in rows if r['seed']==seed}
            b = row['b_new4096']['cycles']-row['b_new0']['cycles']
            d = row['d4096']['cycles']-row['d0']['cycles']
            gaps.append(dict(seed=seed,B_response_cycle=b,D_response_cycle=d,gap_cycle=b-d))
        responses = dict(B_cycle=statistics.median(r['B_response_cycle'] for r in gaps),
            D_cycle=statistics.median(r['D_response_cycle'] for r in gaps),
            gap_cycle=statistics.median(r['gap_cycle'] for r in gaps))
        saved = plan['saved_d_response']
        qualification.append(dict(kind='D_response_bridge',observed=responses['D_cycle'],
            expected=saved,passed=abs(responses['D_cycle']-saved)<=.05*saved))
    qualified = all(c['passed'] for c in qualification)
    checks = []
    if complete:
        for name, model in plan['predictions'].items():
            checks.append(dict(name=name,observed=responses['gap_cycle'],expected=model['gap_cycle'],
                tolerance=model['tolerance_cycle'],passed=abs(responses['gap_cycle']-model['gap_cycle'])<=model['tolerance_cycle']))
    retained = [c['name'] for c in checks if qualified and c['passed']]
    return dict(complete=complete,qualified=qualified,qualification=qualification,windows=windows,
        responses=responses,pairs=gaps,checks=checks,retained=retained,
        decision='information_insufficient' if not qualified else 'development_qualified' if not complete else
            'matched_response_supported' if len(retained)==1 else 'matched_forms_not_passed',
        rows=rows,checked_output_elements=sum(r['output_elements_checked'] for r in rows),
        generation_checks=sum(r['generation_checks'] for r in rows),full_gemm_C2=False,full_gemm_C3=False)


def replay(root, data_root):
    plan = read_json(root/'plan.json')
    rows = []
    for split in ('dev','heldout'):
        cases = [c for c in plan['cases'] if c['split']==split]
        for trial,seed in enumerate(plan['seeds']):
            shift = trial % len(cases)
            for case in cases[shift:]+cases[:shift]:
                folder = data_root/'runs'/case['id']/f'seed{seed}'
                if folder.exists():rows.append(validate(folder,case,seed,plan))
    return score(root,rows)


def execute(root):
    for name,sha in read_json(root/'execution-binding.json')['files'].items():
        require(digest(root/name)==sha,'冻结输入变化：'+name)
    require(read_json(root/'build/approved-target.json')['approved'],'目标构建未批准')
    plan = read_json(root/'plan.json')
    allocation = inspect_allocated_gpu(root/'allocation')
    require(allocation['uuid']==plan['gpu_uuid'],'卡身份不符')
    rows,started = [],time.monotonic()
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        try:
            for split in ('dev','heldout'):
                if split=='heldout':
                    initial = score(root,rows)
                    save_json(root/'initial-score.json',initial)
                    if not initial['qualified']:
                        save_json(root/'score.json',initial)
                        return
                cases = [c for c in plan['cases'] if c['split']==split]
                for trial,seed in enumerate(plan['seeds']):
                    shift = trial % len(cases)
                    for case in cases[shift:]+cases[:shift]:
                        require(time.monotonic()-started<510,'GPU时间预算耗尽')
                        binary = root/('reference/d_probe' if case['program']=='D' else
                            'reference/b_probe' if case['binary']=='old' else 'build/probe')
                        argv = [binary,'--stages',plan['stages'],'--ktiles',plan['ktiles'],
                            '--trace','0','--seed',seed,'--warmup-limit','60']
                        if case['program']=='B':argv += ['--delay-cycles','0','--index-mode','incremental']
                        if case['binary']!='old':argv += ['--consumer-entry-delay-cycles',case['entry_delay_cycles']]
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
    parser.add_argument('--build',action='store_true')
    args = parser.parse_args()
    (build if args.build else execute)(args.root.resolve())

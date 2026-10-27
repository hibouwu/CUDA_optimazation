#!/usr/bin/env python3
"""原生驻留短链正常wait0的固定预测、构建和有限验证。"""
import argparse
import json
from pathlib import Path
import re
import shutil
import statistics
import tarfile
import tempfile
import time

from cfg_b_compute_chain import compute_timeline
from cfg_b_handoff_chain import timed_lifecycle, probe_command, validate as validate_handoff
from l2_saturation import (command_record, digest, exclusive_lock, inspect_allocated_gpu,
                           read_json, require, save_json)


def predict(root):
    plan = read_json(root/'plan.json')
    old = read_json(root/'calibration/native-freeze.json')
    linked = read_json(root/'calibration/connection.json')
    prices = [('d0', old['models']['empty_feedback']),
              ('d95', linked['models']['empty_feedback'])]
    models = {}
    calls = 0
    for name, prior in prices:
        params = prior['parameters']
        unique = {}
        for case in plan['cases']:
            key = f"s{case['stages']}_k{case['ktiles']}_wait{case['wait_retain']}"
            if key in unique:
                continue
            local = timed_lifecycle(case['ktiles'], case['stages'], 0,
                                    wait_retain=case['wait_retain'], **params)
            independent = compute_timeline([r['full_return_cycle'] for r in local['timeline']],
                wait_retain=case['wait_retain'], compute_cycles=params['compute_cycles'],
                commit_lead_cycles=params['commit_lead_cycles'],
                completion_lag_cycles=params['completion_lag_cycles'])
            calls += 2
            require(independent['tail_wait0_return_cycle'] == local['tail_wait0_return_cycle'],
                    '独立计算与局部链尾部不同')
            for row, checked in zip(local['timeline'], independent['timeline']):
                require(row['compute_complete_cycle'] == checked['complete_cycle'], '计算完成接线不符')
                require(row['release_emit_cycle'] >= checked['complete_cycle'], '释放早于完成')
            unique[key] = dict(window_cycle=local['local_end_cycle']+prior['fixed_window_cycle'],
                               timeline=local)
        predicted = {c['id']: unique[f"s{c['stages']}_k{c['ktiles']}_wait{c['wait_retain']}"]
                     for c in plan['cases']}
        models[name] = dict(parameters=params, fixed_window_cycle=prior['fixed_window_cycle'],
            predictions=predicted, responses=dict(
                wait0_increment=predicted['new_wait0']['window_cycle']-predicted['new_wait1']['window_cycle'],
                length_growth=predicted['long_wait0']['window_cycle']-predicted['new_wait0']['window_cycle'],
                stage_baseline=predicted['stage4_wait0']['window_cycle']-predicted['new_wait0']['window_cycle']))
    windows = [m['predictions']['new_wait0']['window_cycle'] for m in models.values()]
    distinguishable = min(windows)*1.05 < max(windows)*.95
    return dict(unit='SM clock64 cycle', models=models, model_calls=calls,
        development_windows_disjoint=distinguishable, cpu_gate=distinguishable,
        old_windows=old['old_windows']['0'], independent_C2=False, independent_C3=False)


def target_function(folder):
    text = (folder/'disassemble/stdout').read_text()
    sections = [s for s in re.split(r'Function\s*:\s*', text)[1:]
                if s.splitlines()[0].strip().startswith('_Z13handoff_chainILb0ELb1ELb0')]
    require(len(sections) == 1, '缺少唯一plain incremental kernel')
    symbol = sections[0].splitlines()[0].strip()
    instructions = [m[1].strip() for line in sections[0].splitlines()[1:]
                    if (m := re.match(r'\s*/\*[0-9a-f]+\*/\s+([^;]+);', line))]
    log = (folder/'compile/stderr').read_text()
    resource = re.search(r'Function properties for '+re.escape(symbol)+
        r'\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads'
        r'\s*\nptxas info\s*:\s*Used (\d+) registers[^\n]*, (\d+) bytes smem', log)
    require(resource is not None, 'plain资源记录缺失')
    return dict(symbol=symbol, instructions=instructions,
                resources=list(map(int, resource.groups())))


def build(root):
    for name, sha in read_json(root/'build-inputs.json')['files'].items():
        require(digest(root/name) == sha, '构建输入变化：'+name)
    compiler = shutil.which('nvcc')
    require(compiler is not None, 'nvcc缺失')
    folder = root/'build'
    folder.mkdir()
    command_record(folder/'version', [compiler, '--version'], 20)
    require('release 12.9,' in (folder/'version/stdout').read_text(), 'CUDA版本不符')
    flags = ['-std=c++17','-O3','-DNDEBUG','-lineinfo',
             '-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v']
    targets = {}
    with tempfile.TemporaryDirectory(prefix='q04-compute-wait-headers-') as temporary:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:
            archive.extractall(temporary, filter='data')
        for retain in (1, 0):
            target = folder/f'wait{retain}'
            target.mkdir()
            command_record(target/'compile', [compiler,*flags,f'-DGH200_NORMAL_WAIT={retain}',
                '-I'+temporary+'/include',root/'source/cfg_b_handoff_chain.cu','-o',target/'probe'], 240)
            disassembler = Path(compiler).resolve().parent/'cuobjdump'
            command_record(target/'disassemble', [disassembler,'--dump-sass',target/'probe'], 60)
            targets[str(retain)] = target_function(target)
    first, second = targets['1'], targets['0']
    a, b = first['instructions'], second['instructions']
    differences = [dict(index=i,wait1=x,wait0=y) for i,(x,y) in enumerate(zip(a,b)) if x != y]
    waits = {key:[x for x in target['instructions'] if 'WARPGROUP.DEPBAR' in x]
             for key,target in targets.items()}
    passed = len(a) == len(b) and len(differences) == 1
    passed = passed and all('WARPGROUP.DEPBAR' in d['wait1'] and
        re.sub(r'0x[01]$', 'WAIT', d['wait1']) == re.sub(r'0x[01]$', 'WAIT', d['wait0'])
        for d in differences)
    passed = passed and all(t['resources'] == [0,0,0,167,1152] and
        sum('HGMMA.' in x for x in t['instructions']) == 8 and
        not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)',x) for x in t['instructions'])
        for t in targets.values())
    passed = passed and len(waits['1']) == len(waits['0']) == 2
    passed = passed and all(re.search(r',\s*0x0$', w) for w in waits['0'])
    save_json(folder/'target-review.json', dict(passed=passed, targets=targets,
        differences=differences, waits=waits, binary_sha256={str(r):digest(folder/f'wait{r}/probe') for r in (0,1)}))
    require(passed, '正常wait立即数之外的SASS或资源变化，停止采样')


def validate(folder, case, seed, uuid):
    raw = read_json(folder/'stdout')
    receipt = read_json(folder/'receipt.json')
    require(receipt['returncode'] == 0 and not receipt['timed_out'], '进程执行失败')
    checked = validate_handoff(folder, raw, case, seed)
    require(checked['performance_eligible'], '暖机资格失败')
    require(raw['device']['uuid'] == uuid and raw['index_mode'] == 'incremental', '卡或索引不符')
    require(raw['conditions']['wait_groups'] == case['wait_retain'] and
            raw['conditions']['tail_wait_groups'] == 0 and
            raw['conditions']['prologue_wait'] is False, '计算等待组织不符')
    require(raw['resources']['registers_per_thread'] == 167, '动态资源与审核不符')
    return dict(case_id=case['id'], seed=seed, cycles=raw['timing']['elapsed_cycles'],
        ns=raw['timing']['elapsed_ns'], smid=raw['timing']['smid_start'],
        output_elements=checked['output_witness']['checked_elements'],
        generation_checked=raw['correctness']['generation_checked'])


def score(root, rows):
    plan, freeze = read_json(root/'plan.json'), read_json(root/'freeze.json')
    ids = ['original','new_wait1','new_wait0']
    complete = len(rows) == 15
    if complete:
        ids += ['stage4_wait0','long_wait0']
    medians, qualification = {}, []
    for case_id in ids:
        group = [r for r in rows if r['case_id'] == case_id]
        require(sorted(r['seed'] for r in group) == plan['seeds'], '条件进程不齐或重复')
        medians[case_id] = {}
        for unit in ('cycles','ns'):
            values = [r[unit] for r in group]
            median = statistics.median(values)
            medians[case_id][unit] = median
            qualification.append(dict(kind='stability',id=case_id,unit=unit,
                passed=max(values)-min(values) <= .05*median))
    for unit in ('cycles','ns'):
        original = medians['original'][unit]
        expected = freeze['old_windows'][unit]
        qualification.append(dict(kind='original_bridge',unit=unit,observed=original,expected=expected,
            passed=abs(original-expected) <= .05*expected))
        matched = medians['new_wait1'][unit]
        qualification.append(dict(kind='new_wait1_bridge',unit=unit,observed=matched,expected=original,
            passed=abs(matched-original) <= .05*original))
    qualified = all(c['passed'] for c in qualification)
    pairs = [('wait0_increment','new_wait0','new_wait1')]
    if complete:
        pairs += [('length_growth','long_wait0','new_wait0'),('stage_baseline','stage4_wait0','new_wait0')]
    observed_responses = {name:statistics.median(
        next(r['cycles'] for r in rows if r['case_id']==after and r['seed']==seed)-
        next(r['cycles'] for r in rows if r['case_id']==before and r['seed']==seed)
        for seed in plan['seeds']) for name,after,before in pairs}
    models = {}
    for name, model in freeze['models'].items():
        windows = []
        for case_id in ids:
            if case_id in ('original','new_wait1'):
                continue
            predicted = model['predictions'][case_id]['window_cycle']
            measured = medians[case_id]['cycles']
            windows.append(dict(id=case_id,predicted=predicted,observed=measured,
                passed=abs(measured-predicted) <= .05*predicted))
        responses = []
        for key,observed in observed_responses.items():
            expected = model['responses'][key]
            tolerance = .05*(abs(expected) if expected else freeze['old_windows']['cycles'])
            responses.append(dict(name=key,observed=observed,predicted=expected,tolerance=tolerance,
                passed=abs(observed-expected) <= tolerance))
        models[name] = dict(windows=windows,responses=responses,
            passed=qualified and all(c['passed'] for c in windows+responses))
    retained = [name for name,model in models.items() if model['passed']]
    return dict(qualified=qualified,complete=complete,qualification=qualification,medians=medians,
        models=models,retained=retained,decision='information_insufficient' if not qualified else
        'retained_local_completion_class' if retained else 'completion_forms_not_passed',
        output_elements=sum(r['output_elements'] for r in rows),
        generation_checked=sum(r['generation_checked'] for r in rows),rows=rows,
        independent_local_discrimination=complete and bool(retained),full_gemm_C2=False,full_gemm_C3=False)


def execute(root):
    for name, sha in read_json(root/'execution-binding.json')['files'].items():
        require(digest(root/name) == sha, '冻结输入变化：'+name)
    require(read_json(root/'build/target-review.json')['passed'], '目标审核未通过')
    plan = read_json(root/'plan.json')
    allocation = inspect_allocated_gpu(root/'allocation')
    require(allocation['uuid'] == plan['gpu_uuid'], '分配卡不符')
    rows = []
    started = time.monotonic()
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        try:
            for split in ('initial','heldout'):
                cases = [c for c in plan['cases'] if (c['split']=='heldout') == (split=='heldout')]
                if split == 'heldout':
                    initial = score(root, rows)
                    save_json(root/'initial-score.json', initial)
                    require(initial['qualified'] and initial['retained'], '开发资格或完成形式未通过，停止留出')
                for trial,seed in enumerate(plan['seeds']):
                    for case in cases[trial:]+cases[:trial]:
                        require(time.monotonic()-started < 510, 'GPU时间预算耗尽')
                        binary = root/'reference/probe' if case['binary']=='original' else root/f"build/{case['binary']}/probe"
                        folder = root/'runs'/case['id']/f'seed{seed}'
                        command_record(folder, probe_command(binary,case,seed)+['--index-mode','incremental'], 60)
                        rows.append(validate(folder,case,seed,plan['gpu_uuid']))
                        print(f"{case['id']} seed{seed}: {rows[-1]['cycles']} cycle", flush=True)
            save_json(root/'score.json', score(root,rows))
        except Exception as error:
            save_json(root/'failure.json', dict(error=str(error),completed_processes=len(rows),rows=rows))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('predict','build','execute'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'predict':
        with args.output.open('x') as stream:
            json.dump(predict(root),stream,ensure_ascii=False,sort_keys=True,indent=2)
            stream.write('\n')
    elif args.command == 'build':
        build(root)
    else:
        execute(root)


if __name__ == '__main__':
    main()

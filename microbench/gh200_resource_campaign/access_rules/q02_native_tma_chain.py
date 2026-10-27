#!/usr/bin/env python3
"""固定源、原生索引的D短链构建与有限采样。"""
import argparse
import math
from pathlib import Path
import re
import shutil
import statistics
import tarfile
import tempfile
import time

import numpy as np

from cfg_b_tma_chain import reference, timed_lifecycle
from cfg_b_handoff_chain import probe_command
from q04_handoff_wait import validate as validate_b
from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json


def build(root, first_wait_entry=False):
    for name, sha in read_json(root/'build-inputs.json')['files'].items():
        require(digest(root/name) == sha, '构建输入变化：'+name)
    compiler = Path(shutil.which('nvcc')).resolve()
    folder = root/'build'
    flags = ['-std=c++17','-O3','-DNDEBUG','-lineinfo','-DGH200_NATIVE_TMA=1',
             '-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v']
    if first_wait_entry:
        flags.append('-DGH200_FIRST_WAIT_ENTRY=1')
    command_record(folder/'version',[compiler,'--version'],20)
    with tempfile.TemporaryDirectory(prefix='q02-native-headers-') as temp:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:
            archive.extractall(temp,filter='data')
        command_record(folder/'compile',[compiler,*flags,'-I'+temp+'/include',
            root/'source/cfg_b_tma_chain.cu','-o',folder/'probe'],180)
    command_record(folder/'disassemble',[compiler.parent/'cuobjdump','--dump-sass',folder/'probe'],60)
    log = (folder/'compile/stderr').read_text()
    functions = []
    for section in re.split(r'Function\s*:\s*',(folder/'disassemble/stdout').read_text())[1:]:
        symbol = section.splitlines()[0].strip()
        if 'tma_chain' not in symbol:
            continue
        instructions = [m[1].strip() for line in section.splitlines()[1:]
                        if (m := re.match(r'\s*/\*[0-9a-f]+\*/\s+([^;]+);',line))]
        properties = re.search(r'Function properties for '+re.escape(symbol)+
            r'\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
        functions.append(dict(symbol=symbol,instruction_count=len(instructions),
            mma=[x for x in instructions if x.startswith('HGMMA.')],
            tma=[x for x in instructions if 'UTMALDG.3D' in x],
            waits=[x for x in instructions if x.startswith('WARPGROUP.DEPBAR')],
            reciprocal_count=sum('MUFU.RCP' in x for x in instructions),
            no_local=not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)',x) for x in instructions),
            zero_spill=bool(properties and all(int(x)==0 for x in properties.groups()))))
    passed = len(functions)==2 and 'C7515' not in log and all(len(f['mma'])==8 and
        len(f['tma'])==3 and len(f['waits'])==2 and f['no_local'] and f['zero_spill'] and
        any(re.search(r',\s*0x1$',w) for w in f['waits']) and
        any(re.search(r',\s*0x0$',w) for w in f['waits']) for f in functions)
    save_json(folder/'target-review.json',dict(passed=passed,functions=functions,flags=flags,
        probe_sha256=digest(folder/'probe'),source_sha256=digest(root/'source/cfg_b_tma_chain.cu')))
    require(passed,'原生D目标审核失败')


def validate_d(folder,case,seed,uuid):
    raw, receipt = read_json(folder/'stdout'),read_json(folder/'receipt.json')
    require(receipt['returncode']==0 and not receipt['timed_out'],'执行未成功')
    require(raw['schema']=='gh200-cfg-b-tma-chain/1' and raw['status']=='measured' and
        raw['stages']==case['stages'] and raw['ktiles']==case['ktiles'] and raw['seed']==seed and
        raw['source_profile']=='generation' and raw['source_tiles']==32 and raw['source_k']==2048 and
        raw['index_mode']=='incremental' and raw['trace'] is False,'程序／源条件不符')
    require(raw['device']['uuid']==uuid and raw['device']['sms']==132 and
        (raw['device']['major'],raw['device']['minor'])==(9,0),'目标卡不符')
    resources = raw['resources']
    require(resources['threads']==256 and resources['grid_ctas']==1 and
        resources['dynamic_smem_bytes']==196608 and resources['local_size_bytes']==0 and
        resources['occupancy_ctas_per_sm']==1,'资源不符')
    conditions = raw['conditions']
    require(conditions['prefilled_tiles']==0 and conditions['full_arrivals']==1 and
        conditions['empty_arrivals']==128 and conditions['transaction_bytes']==32768 and
        conditions['tma_request_bytes']==[16384,8192,8192] and conditions['smem_swizzle_bytes']==128 and
        raw['wgmma_count']==8*case['ktiles'],'事务／计算条件不符')
    correctness = raw['correctness']
    require(correctness['checked_elements']==16384 and correctness['errors']==0 and
        correctness['max_abs_error']==0 and correctness['generation_errors']==0 and
        correctness['generation_checked']==128*case['ktiles'],'正式数值／代次失败')
    expected = reference(case['ktiles'],case['stages'],seed,'generation')
    for name in ('output.f32','witness_output.f32'):
        require((folder/name).stat().st_size==65536 and
            np.array_equal(np.fromfile(folder/name,dtype='<f4'),expected),'独立整数输出失败：'+name)
    require([x['ktiles'] for x in raw['generation_witnesses']]==[1,2,case['stages']+1,case['ktiles']] and
        all(x['correctness']['errors']==0 and x['correctness']['generation_errors']==0 and
            x['correctness']['generation_checked']==128*x['ktiles'] for x in raw['generation_witnesses']),
        '短链代次见证失败')
    warm = raw['warmup']['elapsed_ns']
    mean = statistics.mean(warm[-5:]);cv=statistics.stdev(warm[-5:])/mean
    require(raw['warmup']['converged'] and raw['performance_eligible'] and cv<=.02 and
        math.isclose(cv,raw['warmup']['cv'],rel_tol=1e-7,abs_tol=1e-12),'暖机资格失败')
    timing = raw['timing']
    require(timing['smid_start']==timing['smid_end'],'SM改变')
    require(all(timing['end_'+unit]-timing['begin_'+unit]==timing[key] and timing[key]>0
        for unit,key in (('cycle','elapsed_cycles'),('ns','elapsed_ns'))),'计时不能复算')
    return dict(case_id=case['id'],seed=seed,cycles=timing['elapsed_cycles'],ns=timing['elapsed_ns'],
        smid=timing['smid_start'],resources=resources,output_elements_checked=32768,
        generation_checks=correctness['generation_checked'])


def load(root,split):
    name = 'execution-binding.json' if split=='dev' else 'heldout-binding.json'
    for path,sha in read_json(root/name)['files'].items():
        require(digest(root/path)==sha,'执行输入变化：'+path)
    require(read_json(root/'build/approved-target.json')['approved'],'目标尚未核对')
    return read_json(root/'plan.json')


def controls(rows,expected):
    checks = []
    for key, windows in expected.items():
        group = [r for r in rows if r['case_id']==key]
        for unit in ('cycles','ns'):
            values=[r[unit] for r in group];observed=statistics.median(values)
            checks.append(dict(case_id=key,unit=unit,observed=observed,expected=windows[unit],
                passed=abs(observed-windows[unit])<=.05*windows[unit] and max(values)-min(values)<=.05*observed))
    return checks


def predict(case, name, parameter):
    """引擎的时间标量在此以一个SM cycle为单位，不作频率换算。"""
    frontier = 0.
    def source(t, request):
        nonlocal frontier
        if name == 'independent_completion_latency':
            begin = end = t
            ready = t+parameter
        else:
            begin = max(t, frontier)
            end = begin+parameter*request['bytes']/32768
            frontier = ready = end
        return dict(source_id=3*request['tile']+('A','B0','B1').index(request['operand']),
            source_key=(request['operand'],request['source_k_tile']), merged=False,
            source_ready=ready, source_busy_ns=end-begin, service_start=begin,
            service_end=end, issue_return_ns=t)
    raw = timed_lifecycle(case['ktiles'],case['stages'], request_service=source,
        compute_ns=512.,commit_lead_ns=256.,source_tiles=32,source_profile='generation')
    def cycle_names(value):
        if isinstance(value,dict):
            return {key.removesuffix('_ns')+'_cycle' if key.endswith('_ns') else key:
                    cycle_names(item) for key,item in value.items()}
        if isinstance(value,list):return [cycle_names(item) for item in value]
        return value
    timeline = cycle_names(raw)
    timeline.update(unit='SM clock64 cycle', qualification='显式局部输入服务假设')
    return dict(window_cycle=raw['tail_wait0_and_release_ns']+916.,timeline=timeline)


def summarize(root, split):
    plan = read_json(root/'plan.json')
    selected = (['b_control','d_s6_k4','d_s6_k13'] if split=='dev' else
                ['b_control','d_s6_k4','d_s6_k13','d_s4_k13','d_s6_k32'])
    rows=[]
    for key in selected:
        case=next(c for c in plan['cases'] if c['id']==key)
        for seed in plan['seeds']:
            folder=root/split/key/f'seed{seed}'
            if key=='b_control':
                old=validate_b(folder,case,seed,plan['gpu_uuid'])
                row=dict(case_id=key,seed=seed,cycles=old['cycles'],ns=old['ns'],
                         output_elements_checked=16384,resources=old['resources'],smid=old['smid'])
            else:row=validate_d(folder,case,seed,plan['gpu_uuid'])
            rows.append(row)
    windows={key:{u:statistics.median(r[u] for r in rows if r['case_id']==key)
                  for u in ('cycles','ns')} for key in selected}
    expected={'b_control':dict(cycles=7572,ns=4256)}
    if split=='heldout':expected.update(read_json(root/'freeze.json')['dev_windows'])
    bridge=controls(rows,expected)
    stability=[]
    for key in selected:
        group=[r for r in rows if r['case_id']==key]
        for unit in ('cycles','ns'):
            spread=(max(r[unit] for r in group)-min(r[unit] for r in group))/windows[key][unit]
            stability.append(dict(case_id=key,unit=unit,relative_spread=spread,passed=spread<=.05))
    return dict(rows=rows,windows=windows,bridge=bridge,stability=stability,
        qualified=all(x['passed'] for x in bridge+stability),
        independent_C2=False,independent_C3=False)


def freeze(root):
    require(not (root/'heldout').exists(),'留出采样已开始')
    dev=summarize(root,'dev')
    require(dev['qualified'],'开发资格未通过')
    short=dev['windows']['d_s6_k4']['cycles']
    body=short-916.
    parameters={'independent_completion_latency':body-4*512,
        'shared_source_fifo':body-4*512 if body<=5*512 else (body-512)/4}
    models={};calls=0
    for name,parameter in parameters.items():
        if parameter<0:
            models[name]=dict(parameter_cycle=parameter,dev_passed=False,
                              decision='negative_parameter_at_fixed_boundary')
            continue
        predictions={}
        for case in read_json(root/'plan.json')['cases']:
            if case['binary']=='native_D':
                predictions[case['id']]=predict(case,name,parameter);calls+=1
        p4=predictions['d_s6_k4']['window_cycle'];p13=predictions['d_s6_k13']['window_cycle']
        require(math.isclose(p4,short,abs_tol=1e-8),'K4代数标定不能复现')
        observed=dev['windows']['d_s6_k13']['cycles']
        checks=[dict(endpoint='K13_window',observed_cycle=observed,predicted_cycle=p13,
                     tolerance_cycle=.05*p13,passed=abs(observed-p13)<=.05*p13),
                dict(endpoint='K13_minus_K4',observed_cycle=observed-short,predicted_cycle=p13-p4,
                     tolerance_cycle=.05*(p13-p4),passed=abs(observed-short-(p13-p4))<=.05*(p13-p4))]
        models[name]=dict(parameter_cycle=parameter,predictions=predictions,dev_checks=checks,
                         dev_passed=all(x['passed'] for x in checks))
    retained=[name for name,m in models.items() if m['dev_passed']]
    result=dict(unit='SM clock64 cycle',models=models,retained=retained,
        dev_windows={key:dev['windows'][key] for key in ('d_s6_k4','d_s6_k13')},
        cpu_gate=bool(retained),model_calls=calls,algebraic_calibrations=2,
        source_sha256=digest(root/'source/q02_native_tma_chain.py'),
        dev_summary=dev,independent_C2=False,independent_C3=False)
    save_json(root/'freeze.json',result)
    return result


def score(root):
    frozen=read_json(root/'freeze.json');actual=summarize(root,'heldout');models={}
    for name in frozen['retained']:
        predicted={k:v['window_cycle'] for k,v in frozen['models'][name]['predictions'].items()}
        checks=[]
        for key in ('d_s4_k13','d_s6_k32'):
            expected=predicted[key];observed=actual['windows'][key]['cycles']
            checks.append(dict(endpoint=key,observed_cycle=observed,predicted_cycle=expected,
                tolerance_cycle=.05*expected,passed=abs(observed-expected)<=.05*expected))
        for label,after,before in (('stage_response','d_s4_k13','d_s6_k13'),
                                   ('length_growth','d_s6_k32','d_s6_k13')):
            expected=predicted[after]-predicted[before]
            observed=statistics.median(next(r['cycles'] for r in actual['rows'] if
                r['case_id']==after and r['seed']==seed)-next(r['cycles'] for r in actual['rows'] if
                r['case_id']==before and r['seed']==seed) for seed in read_json(root/'plan.json')['seeds'])
            tolerance=.05*(abs(expected) if expected else predicted[before])
            checks.append(dict(endpoint=label,observed_cycle=observed,predicted_cycle=expected,
                tolerance_cycle=tolerance,passed=abs(observed-expected)<=tolerance))
        models[name]=dict(checks=checks,passed=actual['qualified'] and all(x['passed'] for x in checks))
    retained=[name for name,m in models.items() if m['passed']]
    return dict(actual=actual,models=models,retained=retained,
        decision='information_insufficient' if not actual['qualified'] else
                 'retained_local_prediction_class' if retained else 'local_prediction_class_not_passed',
        independent_local_cycle_validation=bool(retained),independent_C2=False,independent_C3=False,
        full_gemm_C2=False,full_gemm_C3=False)


def execute(root,split):
    plan=load(root,split);allocation=inspect_allocated_gpu(root/(split+'-allocation'))
    require(allocation['uuid']==plan['gpu_uuid'],'分配卡不符')
    cases={c['id']:c for c in plan['cases']}
    selected=(['b_control','d_s6_k4','d_s6_k13'] if split=='dev' else
              ['b_control','d_s6_k4','d_s6_k13','d_s4_k13','d_s6_k32'])
    expected={'b_control':dict(cycles=7572,ns=4256)}
    if split=='heldout':expected.update(read_json(root/'freeze.json')['dev_windows'])
    rows=[];started=time.monotonic()
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        try:
            for key in selected:
                if key=='d_s6_k4' and split=='dev' or key=='d_s4_k13' and split=='heldout':
                    gate=controls(rows,expected)
                    save_json(root/(split+'-control-gate.json'),gate)
                    require(all(x['passed'] for x in gate),'同批控制桥接失败')
                case=cases[key]
                for seed in plan['seeds']:
                    require(time.monotonic()-started<240,'GPU轮次预算耗尽')
                    folder=root/split/key/f'seed{seed}'
                    if key=='b_control':
                        argv=probe_command(root/'reference/probe',case,seed)+['--index-mode','incremental']
                    else:
                        argv=[root/'build/probe','--stages',case['stages'],'--ktiles',case['ktiles'],
                              '--trace',0,'--seed',seed,'--warmup-limit',60]
                    command_record(folder,list(map(str,argv)),60)
                    if key=='b_control':
                        old=validate_b(folder,case,seed,plan['gpu_uuid'])
                        row=dict(case_id=key,seed=seed,cycles=old['cycles'],ns=old['ns'],
                                 output_elements_checked=16384,resources=old['resources'],smid=old['smid'])
                    else:row=validate_d(folder,case,seed,plan['gpu_uuid'])
                    rows.append(row);print(f"{key} seed{seed}: {row['cycles']} cycle",flush=True)
            stability=[];windows={}
            for key in selected:
                group=[r for r in rows if r['case_id']==key]
                windows[key]={u:statistics.median(r[u] for r in group) for u in ('cycles','ns')}
                stability.extend(dict(case_id=key,unit=u,passed=(max(r[u] for r in group)-min(r[u] for r in group))<=.05*windows[key][u]) for u in ('cycles','ns'))
            save_json(root/(split+'-score.json'),dict(rows=rows,windows=windows,stability=stability,
                qualified=all(x['passed'] for x in stability),independent_C2=False,independent_C3=False))
        except Exception as error:
            save_json(root/('failure-'+split+'.json'),dict(error=str(error),completed_processes=len(rows)))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','execute','summarize','freeze','score'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--split',choices=('dev','heldout'),default='dev')
    args=parser.parse_args()
    root=args.root.resolve()
    if args.command=='build':build(root)
    elif args.command=='execute':execute(root,args.split)
    elif args.command=='summarize':save_json(root/(args.split+'-replay.json'),summarize(root,args.split))
    elif args.command=='freeze':freeze(root)
    else:save_json(root/'heldout-model-score.json',score(root))


if __name__=='__main__':main()

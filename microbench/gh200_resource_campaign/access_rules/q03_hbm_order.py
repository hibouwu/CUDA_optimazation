#!/usr/bin/env python3
"""Q03固定large/1:1的CTA内串行与交错；prepare/compile/review不调用GPU。"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import random
import re
import shutil
import statistics as st
import time

from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE if (HERE/'probes/global_duplex.cu').exists() else HERE.parent
CASES = {'interleaved_group': ('independent_r1_w1_large', 'gd_independent_cg_r1_w1'),
         'cta_serial_sweeps': ('independent_serial_r1_w1_large', 'gd_independent_serial_cg_r1_w1')}
SEEDS = (3, 22, 41)
ARRAY_BYTES = 253034496
THREADS, BLOCKS, ITERATIONS = 256, 528, 16
GPU_UUID = 'GPU-43269fbc-449d-3e0f-908a-9c81229546d3'


def prepare(output, old_csv):
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    (source/'probes').mkdir(parents=True)
    (source/'common').mkdir()
    for name in ('probes/global_duplex.cu', 'common/probe_runtime.cuh', 'common/global_duplex_reference.hpp',
                 'auditors/global_duplex.py'):
        target = source/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(CAMPAIGN/name, target)
    shutil.copy2(Path(__file__), source/Path(__file__).name)
    shutil.copy2(HERE/'l2_saturation.py', source/'l2_saturation.py')
    shutil.copy2(old_csv, output/'old-endpoints.csv')
    endpoints = [row for row in csv.DictReader(old_csv.open()) if row['case_id'] in
                 ('read_cg_large', 'write_large', 'independent_r1_w1_large')]
    save_json(output/'plan.json', dict(schema='gh200-q03-cta-order-plan/1', date='2026-10-18 Europe/Paris',
        cases=CASES, seeds=SEEDS, processes=6, threads=THREADS, blocks=BLOCKS, iterations=ITERATIONS,
        array_bytes=ARRAY_BYTES, aggregate_array_bytes=2*ARRAY_BYTES,
        read_payload_bytes=ITERATIONS*ARRAY_BYTES, write_payload_bytes=ITERATIONS*ARRAY_BYTES,
        groups_per_thread_per_sweep=ARRAY_BYTES//(16*THREADS*BLOCKS), device_uuid=GPU_UUID,
        dynamic_per_cta=dict(interleaved_group=dict(barrier_calls=2,group_loop_tests=117*16,iteration_loop_tests=16),
                             cta_serial_sweeps=dict(barrier_calls=2+2*16,group_loop_tests=2*117*16,iteration_loop_tests=16)),
        changed='Within each iteration: each CTA issues all its read groups then all write groups with CTA barriers between phases, or read+write per group.',
        fixed='16B cg reads/wb independent writes, address ownership, inputs, checksums, final device fence and CTA drain.',
        boundary='globaltimer full-grid envelope of one kernel; clock64 local CTA windows; CUDAevent full kernel separately.',
        qualification='Original1/2-sweep full checks, every formal read checksum/destination word, warmup8..30/last5CV<=2%; no local spill, exact target build/source/GPU; stop if runtime occupancy limits differ.',
        limitations=['CTA-local order has no grid barrier; different CTAs may overlap read and write phases.',
                     'The organization also changes loop/control, two CTA barriers per iteration and compiler allocation; no isolated physical HBM claim.',
                     'Logical requested bytes do not identify physical HBM traffic; no TMA price transfer.'],
        budget=dict(conditions=2, independent_processes_per_condition=3, sampling_seconds_max=600,
                    allocation_seconds_max=3600, implementation_repairs_max=2, extra_conditions=0),
        source_sha256={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()},
        old_endpoint_source=str(old_csv.resolve()), old_endpoint_sha256=digest(old_csv), old_endpoints=endpoints,
        role='development localization, not independent numeric C2 or full-GEMM C3'))
    save_json(output/'predictions.json', dict(schema='gh200-q03-cta-order-prediction/1', frozen=True,
        primary='R=median(serial globaltimer envelope)/median(interleaved globaltimer envelope)-1',
        zero_organization_response=dict(prediction=0, absolute_response_limit=.05),
        stability='Each three-process range width/median <=5%; otherwise information insufficient, no extra sampling.',
        organization_dependent_service=dict(prediction=None, status='working hypothesis; no justified amplitude'),
        old_pure_endpoint_forms='max/sum already failed the old mixed full window; kept as old context, not revived numeric predictions.',
        classification='Stable |R|<=5%: zero-response form not refuted here; stable |R|>5%: concrete zero-response form refuted. No price fit.',
        no_new_fit=True, no_absolute_time_prediction=True, independent_C2=False))
    (output/'README.md').write_text('''# Q03：固定1:1的CTA内读写组织

2026-10-18。仅两个组织、每项seed3/22/41，共6个独立进程。固定528CTA×256线程、16轮、每数组253034496B，读写各4048551936逻辑B。交错沿原每group读后写；串行每轮先完成本CTA全部读发出，再完成全部写发出，两个CTA barrier隔开相邻阶段。没有grid barrier，不称全GPU读写阶段分离。

首轮只检验`serial/interleaved−1=0`的5%响应门槛；每点三进程range宽度/median须≤5%，否则信息不足且不扩量。组织相关服务没有有依据的幅度，本轮为开发定位，不拟合价格、不称数值C2。两路径还改变循环控制、内部CTA同步和可能的编译寄存器分配，响应不归因裸HBM。

每进程在原输出/地址/线程checksum协议下完成1/2轮短检查，再8–30次暖机与16轮正式窗口。全部写配置保留末尾device fence、shared drain和CTA barrier。主指标为globaltimer整grid包络，clock64只同CTA相减；event完整kernel单列。

正式窗口每CTA的barrier调用数分别为2与34；每线程group循环检查数为1872与3744，外层迭代检查均16次。每线程1872个读向量和1872个写向量相同。目标runtime占用上限若不同，停止并报告资源差异，不追加条件校正。

```sh
python -B PACKAGE/source/q03_hbm_order.py compile --output PACKAGE --nvcc nvcc
python -B PACKAGE/source/q03_hbm_order.py review --output PACKAGE
python -B PACKAGE/source/q03_hbm_order.py run --output PACKAGE --allow-gpu
python -B PACKAGE/source/q03_hbm_order.py score --output PACKAGE --score-output NEW_SCORE_DIRECTORY
```

prepare/compile/review不执行probe。run由主线程在免费ROMEO同UUID串行分配中执行，最多600秒采样/1h分配、两次实现修复。CUDA13本地静态诊断不能代替CUDA12.9目标构建审查。旧18条件入口、原合同与原结果不改。
''')
    save_json(output/'freeze.json', dict(files={str(p.relative_to(output)):digest(p)
        for p in sorted(output.rglob('*')) if p.is_file()}, GPU_executions=0))
    print(f'PREPARED {output}: two organizations, six processes; GPU=0')


def load(root):
    for name, expected in read_json(root/'freeze.json')['files'].items():
        require(digest(root/name) == expected, 'frozen input changed: '+name)
    plan = read_json(root/'plan.json')
    require(digest(Path(__file__)) == plan['source_sha256']['q03_hbm_order.py'], 'execute frozen source/q03_hbm_order.py')
    return plan


def compile_probe(root, nvcc, flags):
    load(root)
    compiler = shutil.which(nvcc)
    require(compiler is not None, 'nvcc missing')
    build = root/'build'
    build.mkdir(exist_ok=False)
    command_record(build/'version', [compiler, '--version'], 30)
    command_record(build/'compile', [compiler, '-std=c++17', '-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a',
        '-Xptxas=-v', *flags, root/'source/probes/global_duplex.cu', '-o', build/'probe'], 180)
    command_record(build/'disassemble', [Path(compiler).resolve().parent/'cuobjdump', '--dump-sass', build/'probe'], 60)
    save_json(build/'identity.json', dict(source_sha256=digest(root/'source/probes/global_duplex.cu'),
        plan_sha256=digest(root/'plan.json'), probe_sha256=digest(build/'probe'), sass_sha256=digest(build/'disassemble/stdout')))
    print('COMPILED; GPU=0')


def review(root):
    load(root)
    build = root/'build'
    identity = read_json(build/'identity.json')
    require(identity['source_sha256'] == digest(root/'source/probes/global_duplex.cu') and
        identity['probe_sha256'] == digest(build/'probe') and identity['sass_sha256'] == digest(build/'disassemble/stdout'),
        'build identity changed')
    # Reuse only four frozen, pure provenance helpers; no old family runner/contract is changed.
    audit_source = (root/'source/auditors/global_duplex.py').read_text()
    selected = [node for node in ast.parse(audit_source).body if isinstance(node, ast.FunctionDef)
                and node.name in ('_instructions', '_opcode', '_vector_registers', 'dataflow_evidence')]
    namespace = dict(re=re, require=require)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<frozen-global-duplex-dataflow>', 'exec'), namespace)
    functions = {part.splitlines()[0].strip():part for part in re.split(r'Function\s*:\s*', (build/'disassemble/stdout').read_text())[1:]}
    resources = {}
    name = None
    for line in (build/'compile/stderr').read_text().splitlines():
        match = re.search(r"Compiling entry function '([^']+)'", line)
        if match: name=match[1];resources[name]={}
        if name:
            for key, pattern in (('registers_per_thread',r'Used (\d+) registers'),('static_smem_bytes',r'(\d+) bytes smem'),
                ('stack_bytes',r'(\d+) bytes stack frame'),('spill_store_bytes',r'(\d+) bytes spill stores'),
                ('spill_load_bytes',r'(\d+) bytes spill loads')):
                match=re.search(pattern,line)
                if match:resources[name][key]=int(match[1])
    evidence = {}
    for organization, (_, symbol) in CASES.items():
        ops = namespace['_instructions'](functions[symbol])
        timers=[pc for pc,op in ops if 'SR_GLOBALTIMERLO' in op]
        require(len(timers)==2, 'two globaltimer boundaries')
        timed=[(pc,op) for pc,op in ops if timers[0]<pc<timers[1]]
        accesses={token:[(pc,op) for pc,op in timed if re.search(r'\b'+token+r'\.',op)] for token in ('LDG','STG')}
        require(all(len(values)==1 for values in accesses.values()), 'exactly one read/write vector instruction')
        require(namespace['_opcode'](accesses['LDG'][0][1])=='LDG.E.128.STRONG.GPU' and
            namespace['_opcode'](accesses['STG'][0][1])=='STG.E.128.STRONG.SM', '16B cg/wb target forms')
        loops=[]
        for pc,op in timed:
            match=re.search(r'\bBRA\s+(?:[^,]*,\s*)?(0x[0-9a-fA-F]+)',op)
            if match and int(match[1],16)<pc:
                start=int(match[1],16)
                if any(start<=m<=pc for values in accesses.values() for m,_ in values):loops.append([start,pc])
        require(len(loops)==(2 if organization=='interleaved_group' else 3), 'group/sweep loop organization changed')
        outer=max(loops,key=lambda p:p[1]-p[0])
        inner=[loop for loop in loops if loop!=outer]
        read_pc,write_pc=accesses['LDG'][0][0],accesses['STG'][0][0]
        if organization=='interleaved_group':
            require(inner[0][0]<=read_pc<write_pc<=inner[0][1], 'interleaved same group loop')
        else:
            inner.sort()
            require(inner[0][0]<=read_pc<=inner[0][1]<inner[1][0]<=write_pc<=inner[1][1], 'serial distinct read then write loops')
        drains=[(pc,op) for pc,op in timed if pc>outer[1] and re.search(r'\bSTS\b',op)]
        fences=[(pc,op) for pc,op in timed if 'MEMBAR' in op]
        barriers=[pc for pc,op in timed if 'BAR.SYNC' in op]
        if organization=='cta_serial_sweeps':
            require(any(inner[0][1]<pc<inner[1][0] for pc in barriers) and
                any(inner[1][1]<pc<outer[1] for pc in barriers), 'CTA phase barriers between read/write and iterations')
        require(len(drains)==1 and len(fences)==1 and fences[0][1]=='MEMBAR.SC.GPU', 'one checksum drain/device fence')
        require(any(pc<outer[0] for pc in barriers) and any(fences[0][0]<pc<timers[1] for pc in barriers), 'start/end CTA gates')
        require(not any(re.search(r'\b(?:LDL|STL)\.',op) for _,op in ops), 'local traffic')
        flow=namespace['dataflow_evidence'](timed,outer,drains[0][0],False)
        resource=resources[symbol]
        require(resource['static_smem_bytes']==1024 and all(resource[k]==0 for k in
            ('stack_bytes','spill_store_bytes','spill_load_bytes')), 'resource allocation/spill')
        evidence[organization]=dict(symbol=symbol,resources=resource,timers=timers,loops=loops,accesses=accesses,
            checksum_dataflow=flow,drain=drains,fence=fences,barriers=barriers,
            logical_vectors_per_thread=dict(read=117*16,write=117*16),
            dynamic_barrier_calls_per_cta=2 if organization=='interleaved_group' else 34,
            group_loop_tests_per_thread=117*16 if organization=='interleaved_group' else 2*117*16,
            iteration_loop_tests_per_thread=16,physical_inflight_proven=False)
    target=bool(re.search(r'release 12\.9\b',(build/'version/stdout').read_text()))
    save_json(build/'target-review.json', dict(passed=True,target_cuda12_9=target,functions=evidence,
        build_identity_sha256=digest(build/'identity.json'),role='static target qualification' if target else 'CUDA13 local diagnostic only'))
    print(f'STATIC PASSED; target CUDA12.9={target}; GPU=0')


def validate(raw, organization, seed, review_data):
    require(len(raw)==3, 'device/short-check/formal JSON lines')
    device, check, trial=raw
    case,symbol=CASES[organization]
    require(device['uuid']==GPU_UUID and device['sms']==132 and device['cc']=='9.0', 'fixed GH200 identity')
    for row in (check,trial):
        require(row['case_id']==case and row['seed']==seed and row['threads']==THREADS and row['blocks']==BLOCKS
            and row['errors']==0 and row['scope']=='all_gpu', 'case/shape/correctness')
    expected=THREADS*BLOCKS+ARRAY_BYTES//4
    require([r['iterations'] for r in check['target_launches']]==[1,2] and len(check['checks'])==2, 'short1/2 checks')
    require(all(row['checked_elements']==expected and row['expected_elements']==expected and row['errors']==0
        and row['completed'] and row['verified_CTA_ids']==list(range(BLOCKS)) for row in check['checks']), 'complete short numerical coverage')
    require(trial['iterations']==ITERATIONS and trial['array_bytes']==ARRAY_BYTES and trial['aggregate_array_bytes']==2*ARRAY_BYTES
        and trial['read_payload_bytes']==trial['write_payload_bytes']==ITERATIONS*ARRAY_BYTES
        and trial['work_count']==2*ITERATIONS*ARRAY_BYTES and trial['issue_organization']==organization, 'fixed logical demand/organization')
    require(trial['correctness']['checked_elements']==expected and
        trial['correctness']['method']=='all_nonempty_thread_read_checksums_and_all_destination_words', 'formal full numerical consumers')
    resource=review_data['functions'][organization]['resources']
    require(trial['kernel_symbol']==symbol and trial['registers_per_thread']==resource['registers_per_thread']
        and trial['static_smem_bytes']==1024 and trial['local_size_bytes']==0 and trial['dynamic_smem_bytes']==0
        and trial['occupancy_limit_ctas_per_sm']>=4, 'target runtime resources')
    stamps=trial['blocks_detail']
    require(len(stamps)==BLOCKS and {s['block_id'] for s in stamps}==set(range(BLOCKS))
        and len({s['smid'] for s in stamps})==132, 'complete CTA/SM coverage')
    require(all(s['stop_ns']>s['start_ns'] and s['stop_cycle']>s['start_cycle'] for s in stamps), 'positive local windows')
    begin=min(s['start_ns'] for s in stamps);end=max(s['stop_ns'] for s in stamps)
    require(begin==trial['start_ns'] and end==trial['stop_ns'] and 0<end-begin<=trial['event_ms']*1.05e6, 'globaltimer/event boundary')
    warm=trial['warmup_samples_ns']
    require(trial['warmup_converged'] and 8<=len(warm)<=30 and st.stdev(warm[-5:])/st.mean(warm[-5:])<=.02, 'original warmup gate')
    return dict(organization=organization,case_id=case,seed=seed,elapsed_ns=end-begin,event_ms=trial['event_ms'],
        checked_elements=expected*3,registers_per_thread=trial['registers_per_thread'],sm_count=132,
        occupancy_limit_ctas_per_sm=trial['occupancy_limit_ctas_per_sm'],
        ctaseen_per_smid={str(sm):sum(s['smid']==sm for s in stamps) for sm in sorted({s['smid'] for s in stamps})})


def run(root, allow_gpu):
    require(allow_gpu,'run requires --allow-gpu')
    plan=load(root)
    review_data=read_json(root/'build/target-review.json')
    require(review_data['passed'] and review_data['target_cuda12_9'] and
        review_data['build_identity_sha256']==digest(root/'build/identity.json'), 'CUDA12.9 target review required')
    identity=read_json(root/'build/identity.json')
    require(identity['probe_sha256']==digest(root/'build/probe') and identity['sass_sha256']==digest(root/'build/disassemble/stdout')
        and identity['plan_sha256']==digest(root/'plan.json'), 'exact target binary/SASS/plan')
    folder=root/'dev';folder.mkdir(exist_ok=False)
    save_json(folder/'binding.json',dict(predictions_sha256=digest(root/'predictions.json'),plan_sha256=digest(root/'plan.json'),
        review_sha256=digest(root/'build/target-review.json'),device_uuid=plan['device_uuid']))
    allocated=inspect_allocated_gpu(folder/'allocation')
    require(allocated['uuid']==GPU_UUID,'allocation differs from registered device')
    results=[]
    with exclusive_lock(Path('/tmp')/f'gh200-measurement-{GPU_UUID}.lock'):
        started=time.monotonic()
        for index,seed in enumerate(SEEDS):
            organizations=list(CASES);random.Random(20261018+index).shuffle(organizations)
            for organization in organizations:
                remaining=600-(time.monotonic()-started);require(remaining>0,'600-second sampling budget exhausted')
                sample=folder/organization/f'process_{index:02}'
                command_record(sample,[root/'build/probe','order-probe',CASES[organization][0],'16',str(seed)],min(180,remaining))
                row=validate([json.loads(line) for line in (sample/'stdout').read_text().splitlines()],organization,seed,review_data)
                save_json(sample/'validated.json',row)
                results.append(dict(path=str(sample.relative_to(root)),raw_sha256=digest(sample/'stdout'),
                    validated_sha256=digest(sample/'validated.json'),**row))
                print(f'VALID {organization} seed={seed}',flush=True)
            require(results[-1]['occupancy_limit_ctas_per_sm']==results[-2]['occupancy_limit_ctas_per_sm'],
                    'organization changed runtime occupancy limit; stop without adding conditions')
    save_json(folder/'summary.json',dict(complete=True,independent_processes=6,binding_sha256=digest(folder/'binding.json'),results=results,
        fits=0,independent_C2=False,physical_HBM_traffic_proven=False))


def score(root, output):
    load(root)
    summary=read_json(root/'dev/summary.json');review_data=read_json(root/'build/target-review.json')
    require(summary['complete'] and summary['independent_processes']==6,'six processes required')
    binding=read_json(root/'dev/binding.json')
    require(summary['binding_sha256']==digest(root/'dev/binding.json') and review_data['passed']
        and review_data['target_cuda12_9'] and binding['predictions_sha256']==digest(root/'predictions.json')
        and binding['plan_sha256']==digest(root/'plan.json') and binding['review_sha256']==digest(root/'build/target-review.json')
        and binding['device_uuid']==GPU_UUID, 'sampling prediction/build/GPU binding')
    groups={name:[] for name in CASES}
    for row in summary['results']:
        path=root/row['path']
        require(digest(path/'stdout')==row['raw_sha256'] and digest(path/'validated.json')==row['validated_sha256'], 'raw/validated identity')
        actual=validate([json.loads(line) for line in (path/'stdout').read_text().splitlines()],row['organization'],row['seed'],review_data)
        require(all(actual[key]==row[key] for key in actual),'summary differs from raw')
        groups[row['organization']].append(actual)
    require(all(len(rows)==3 and {r['seed'] for r in rows}==set(SEEDS) for rows in groups.values()), 'original three seeds per organization')
    medians={name:st.median(row['elapsed_ns'] for row in rows) for name,rows in groups.items()}
    ranges={name:[min(row['elapsed_ns'] for row in rows),max(row['elapsed_ns'] for row in rows)] for name,rows in groups.items()}
    stable=all((ranges[name][1]-ranges[name][0])/medians[name]<=.05 for name in CASES)
    response=medians['cta_serial_sweeps']/medians['interleaved_group']-1
    output.mkdir(parents=True,exist_ok=False)
    result=dict(qualified_processes=6,medians_ns=medians,ranges_ns=ranges,response=response,stable=stable,
        zero_organization_form='not_refuted' if stable and abs(response)<=.05 else 'refuted' if stable else 'information_insufficient',
        numerical_amplitude_fitted=False,independent_C2=False,physical_HBM_price=False,
        prediction_sha256=digest(root/'predictions.json'),raw_summary_sha256=digest(root/'dev/summary.json'))
    save_json(output/'score.json',result);print(json.dumps(result,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='step',required=True)
    for name in ('prepare','compile','review','run','score'):
        child=sub.add_parser(name);child.add_argument('--output',type=Path,required=True)
        if name=='prepare':child.add_argument('--old-csv',type=Path,required=True)
        if name=='compile':
            child.add_argument('--nvcc',default='nvcc');child.add_argument('--nvcc-flag',action='append',default=[])
        if name=='run':child.add_argument('--allow-gpu',action='store_true')
        if name=='score':child.add_argument('--score-output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    if args.step=='prepare':prepare(root,args.old_csv.resolve())
    elif args.step=='compile':compile_probe(root,args.nvcc,args.nvcc_flag)
    elif args.step=='review':review(root)
    elif args.step=='run':run(root,args.allow_gpu)
    else:score(root,args.score_output.resolve())


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Q07四组织准备、静态构建与独立全值评分；采样由已授权主线程执行。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import time

import numpy as np

HERE=Path(__file__).resolve().parent
MODES=('input','output','serial','concurrent')
SEEDS=(3,22,41)
SCHEMA='gh200-q07-input-output-plan/1'
PREPARED_SCHEMA='gh200-q07-prepared-joint-plan/1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def references(mode,seed):
    row=np.arange(128,dtype=np.int64)[:,None]
    inner=np.arange(64,dtype=np.int64)[None,:]
    payload=[]
    for slot in range(4):
        a=((7*row+13*(slot*64+inner)+3*seed)%17-8)/32
        b=((5*(slot*64+inner)+11*row+5*seed)%17-8)/32
        payload.extend((a.astype('<f2').reshape(-1),b.astype('<f2').reshape(-1)))
    payload=np.concatenate(payload)
    if mode=='output':
        payload.fill(0)
    output=np.full(128*128+32,0x7fc12345,dtype='<u4')
    if mode!='input':
        column=np.arange(128,dtype=np.int64)[None,:]
        values=(((7*row+11*column+3*seed)%17-8)/32).astype('<f4')
        output[16:-16]=values.reshape(-1).view('<u4')
    return payload,output


def check_witnesses(folder,mode,seed):
    expected_payload,expected_output=references(mode,seed)
    require((folder/'payload.f16').stat().st_size==expected_payload.nbytes,'input payload byte length differs')
    require((folder/'output-guarded.u32').stat().st_size==expected_output.nbytes,'guarded D byte length differs')
    payload=np.fromfile(folder/'payload.f16',dtype='<f2')
    output=np.fromfile(folder/'output-guarded.u32',dtype='<u4')
    require(payload.shape==expected_payload.shape and np.array_equal(payload,expected_payload),
            'four input payloads differ from independent coordinate reference')
    require(output.shape==expected_output.shape and np.array_equal(output,expected_output),
            'full D or guard differs from independent coordinate reference')
    return dict(payload_elements=payload.size,guarded_output_elements=output.size,
                payload_sha256=sha(folder/'payload.f16'),output_sha256=sha(folder/'output-guarded.u32'))


def prepare(root,headers,prepared=False):
    root.mkdir(parents=True,exist_ok=False)
    source=root/'source';source.mkdir()
    shutil.copy2(HERE/'q07_input_output_joint.py',source/'q07_input_output_joint.py')
    for name in ('q07_input_output_joint.cu','r00_common.hpp'):
        shutil.copy2(HERE/'probes'/name,source/name)
    shutil.copy2(headers,source/'cutlass-3.9.2-include.tar.gz')
    cases=[dict(id=mode,organization=mode,trace=False,input_groups=0 if mode=='output' else 4,
                output_subtiles=0 if mode=='input' else 8) for mode in MODES]
    plan=dict(schema=SCHEMA,cases=cases,seeds=SEEDS,processes=12,
        development_conditions=4,heldout_conditions=0,maximum_development_conditions=8,
        maximum_heldout_conditions=4,maximum_candidates=3,maximum_repairs=2,
        maximum_gpu_seconds=3600,qualification='development_joint_window',
        prediction=None,source_sha256={p.name:sha(p) for p in sorted(source.iterdir())})
    if prepared:
        plan.update(schema=PREPARED_SCHEMA,prepared=True,qualification='development_prepared_joint_window',
            maximum_development_conditions=4,maximum_heldout_conditions=0,maximum_candidates=2,
            relative_tolerance=.05,window_relative_spread=.05,independent_C2=False,independent_C3=False,
            prediction=dict(max='same-seed max(input,output)',serial='same-seed serial',
                units=['cycles','ns'],denominator='prediction',ambiguous='both or neither => information_insufficient'),
            preparation=dict(warpgroup=[128,255],floats_per_lane=128,elements=16384,bytes=65536,
                stage_elements=8192,stages=[4,5],operand_bytes=32768,
                mainloop_relative_ranges=[[65536,98304],[163840,196608]],
                mapping='linear=i*128+(threadIdx.x-128); first8192float to A, remainder to B',
                anchor='volatile shared stores then CTA barrier before begin; fragment retained to Epilogue'))
        shutil.copy2(HERE.parents[2]/'Docs/ModelEvaluation/gemm/PLAN.md',root/'campaign-PLAN.md')
    save(root/'plan.json',plan)
    commands=[]
    for trial,seed in enumerate(SEEDS):
        for index in range(4):
            mode=MODES[(index+trial)%4]
            commands.append(dict(case_id=mode,seed=seed,folder=f'runs/{mode}/seed{seed}',
                argv=['build/probe','--organization',mode,'--seed',str(seed),'--warmup-limit','30']+
                    (['--prepared','1'] if prepared else [])))
    save(root/'commands.json',dict(executed=False,commands=commands))
    save(root/'demand.json',dict(threads=384,roles={'input':[0,127],'output':[128,255],'idle':[256,383]},
        input=dict(physical_slots=6,used_slots=4,groups=4,requests_per_group=[16384,8192,8192],
                   payload_bytes=131072,reuse=False),
        output=dict(subtiles=8,subtile_shape=[64,32],dtype='FP32',bytes_per_subtile=8192,
                    source_slots=2,source_slot_bytes=8192,payload_bytes=65536,
                    collective='Epilogue.store + Epilogue.store_tail'),
        timing=dict(stamps=2,phase_joins=2,internal_trace=False,
                    input_end='four mbarrier transaction waits',
                    output_end='store_tail wait_group.read0 then common CTA joins',
                    global_D_check='after device/stream completion'),
        program='four input groups and one independently supplied output tile; no WGMMA'))
    (root/'README.md').write_text('# Q07 输入／输出联合窗口\n\n'
        '四组织使用同一kernel：输入单独、输出单独、串行、并发；每组织三个独立seed。'
        '四输入组使用独立槽，输出使用cfg_b原Epilogue工厂的八子块双槽。'
        '共同窗口只有首尾戳及两个阶段join；payload在末戳后导出，全D在stream完成后核对。\n\n'
        '当前为开发准备，无GPU采样或预测幅度。`commands.json`保存12个待执行命令；'
        '`source/q07_input_output_joint.py compile --output <包>`构建，'
        '`score --output <包>`独立重算全部见证并汇总联合窗口。\n')
    if prepared:
        (root/'README.md').write_text('''# Q07 准备完成后的输入／输出联合窗口

四组织仍为input、output、serial、concurrent，各三独立seed，共12进程。输出WG128–255将每lane全部128个float fragment写至真实未用mainloop A/B槽4/5的两段各32KiB，volatile写后CTA同步再起戳；fragment保留并直接交给原Epilogue。真实输入仅槽0–3；epi在mainloop之前独立存储，八子块／双槽、store_tail、两阶段join和首尾两戳不改。所有组织都做相同锚定，准备工作不计入联合窗口。

每种组织cycle/ns的三进程range≤中位5%，完整数值、卡身份、local0/occupancy1及实际SASS准备边界全部合格才判。以同seed max(input,output)或serial预测concurrent，两个单位各沿abs(obs−pred)/pred≤5%，三seed全过才支持该形式。仅max命中则保留独立连接工作假设；仅serial命中则保留条件联合服务；both/neither或资格失败均为信息不足，联合窗及原失败保留。开发定位，独立C2/C3未验收，不转移准备价格。

CPU构建先审核全部fragment生成及见证写在begin前、begin后无fragment重算、原输入/输出链、无local/spill和occupancy1，再批准采样。最多两次定向修复，免费ROMEO单次10min且GPU≤1小时，不扩条件、不自动重采。

```sh
python -B cpu-check.py
python -B source/q07_input_output_joint.py compile --output "$PWD"
python -B source/q07_input_output_joint.py score --output "$PWD"
```

准备阶段不编译/GPU；根代理负责运输、目标审核、批准和最终binding。
''')
        save(root/'preparation-repair.json',dict(repairs=0,compilation=False,gpu=False))
        save(root/'manifest.json',dict(schema='gh200-q07-prepared-input/1',files={str(p.relative_to(root)):sha(p)
            for p in sorted(root.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(status='prepared',output=str(root),processes=12,gpu_executions=0)))


def record_command(folder,command,timeout):
    folder.mkdir(parents=True,exist_ok=False)
    save(folder/'command.json',[str(x) for x in command])
    started=time.monotonic()
    result=subprocess.run([str(x) for x in command],capture_output=True,timeout=timeout)
    (folder/'stdout').write_bytes(result.stdout);(folder/'stderr').write_bytes(result.stderr)
    save(folder/'receipt.json',dict(returncode=result.returncode,elapsed_seconds=time.monotonic()-started))
    require(result.returncode==0,f'command failed: {folder}')


def compile_probe(root,nvcc):
    plan=json.loads((root/'plan.json').read_text())
    for name,digest in plan['source_sha256'].items():
        require(sha(root/'source'/name)==digest,f'frozen source changed: {name}')
    build=root/'build';build.mkdir(exist_ok=False)
    compiler=Path(shutil.which(nvcc)).resolve()
    flags=['-std=c++17','-O3','-DNDEBUG','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v']
    record_command(build/'version',[compiler,'--version'],30)
    with tempfile.TemporaryDirectory(prefix='q07-joint-headers-') as temporary:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:
            archive.extractall(temporary,filter='data')
        record_command(build/'compile',[compiler,*flags,'-I'+temporary+'/include',
            root/'source/q07_input_output_joint.cu','-o',build/'probe'],180)
    record_command(build/'disassemble',[compiler.parent/'cuobjdump','--dump-sass',build/'probe'],60)
    record_command(build/'inspect',[build/'probe','--inspect-program'],30)
    save(build/'identity.json',dict(plan_sha256=sha(root/'plan.json'),probe_sha256=sha(build/'probe'),
        sass_sha256=sha(build/'disassemble/stdout'),program=json.loads((build/'inspect/stdout').read_text()),
        static_instruction_review='pending independent target review'))


def prepared_judgment(plan,rows):
    windows=[];predictions=[]
    for mode in MODES:
        selected=[r for r in rows if r['organization']==mode]
        for unit in ('cycles','ns'):
            values=[r[unit] for r in selected];median=statistics.median(values)
            require(median>0,'positive joint windows required')
            spread=(max(values)-min(values))/median
            windows.append(dict(organization=mode,unit=unit,values=values,relative_spread=spread,
                passed=spread<=plan['window_relative_spread']))
    for seed in SEEDS:
        selected={r['organization']:r for r in rows if r['seed']==seed}
        for unit in ('cycles','ns'):
            observed=selected['concurrent'][unit]
            for candidate,predicted in (('max',max(selected['input'][unit],selected['output'][unit])),
                    ('serial',selected['serial'][unit])):
                require(predicted>0,'positive prediction required')
                error=abs(observed-predicted)/predicted
                predictions.append(dict(seed=seed,unit=unit,candidate=candidate,observed=observed,
                    predicted=predicted,relative_error=error,passed=error<=plan['relative_tolerance']))
    matches=[name for name in ('max','serial') if all(r['passed'] for r in predictions if r['candidate']==name)]
    qualified=all(r['passed'] for r in windows)
    classification=matches[0] if len(matches)==1 else 'both' if matches else 'neither'
    decision='information_insufficient' if not qualified or len(matches)!=1 else \
        'retain_independent_connection' if matches==['max'] else 'retain_conditional_joint_service'
    return dict(window_stability=windows,predictions=predictions,observation_qualified=qualified,
        matching_forms=matches,classification=classification,decision=decision,independent_C2=False,independent_C3=False)


def score(root,emit=True):
    plan=json.loads((root/'plan.json').read_text())
    require(plan['schema'] in (SCHEMA,PREPARED_SCHEMA),'plan schema mismatch')
    prepared=plan['schema']==PREPARED_SCHEMA
    rows=[]
    for mode in MODES:
        for seed in SEEDS:
            folder=root/'runs'/mode/f'seed{seed}'
            raw=json.loads((folder/'stdout').read_text())
            require(raw['schema']=='gh200-q07-input-output/1' and raw['status']=='measured','raw schema/status mismatch')
            require(raw['organization']==mode and raw['seed']==seed and raw['samples']==1 and raw['trace'] is False,'raw conditions mismatch')
            if prepared:require(raw.get('prepared') is True,'prepared anchor not enabled')
            require(raw['performance_eligible'] and raw['warmup']['converged'] and raw['warmup']['tail_cv']<=0.02,'performance ineligible')
            resource=raw['resources'];timing=raw['timing']
            require(resource['threads']==384 and resource['grid_ctas']==1 and resource['dynamic_smem_bytes']==215040
                and resource['local_size_bytes']==0 and resource['occupancy_ctas_per_sm']==1,'resources mismatch')
            require(raw['device']['sms']==132 and 'GH200' in raw['device']['name'],'wrong device')
            require(timing['smid_begin']==timing['smid_end'],'SM changed')
            for unit,field in (('cycle','elapsed_cycles'),('ns','elapsed_ns')):
                require(timing['end_'+unit]>timing['begin_'+unit] and timing[field]==timing['end_'+unit]-timing['begin_'+unit], 'timing mismatch')
            require(raw['correctness']==dict(checked_elements=81952,errors=0),'numeric report mismatch')
            witness=check_witnesses(folder,mode,seed)
            rows.append(dict(organization=mode,seed=seed,cycles=timing['elapsed_cycles'],ns=timing['elapsed_ns'],
                uuid=raw['device']['uuid'],resources=resource,witness=witness,raw_sha256=sha(folder/'stdout')))
    require(len({r['uuid'] for r in rows})==1,'device identity changed')
    require(all(r['resources']==rows[0]['resources'] for r in rows),'same-kernel resources changed')
    conditions=[]
    for mode in MODES:
        subset=[r for r in rows if r['organization']==mode]
        conditions.append(dict(organization=mode,**{field:dict(median=statistics.median(r[field] for r in subset),
            minimum=min(r[field] for r in subset),maximum=max(r[field] for r in subset)) for field in ('cycles','ns')}))
    paired=[]
    for seed in SEEDS:
        r={v['organization']:v for v in rows if v['seed']==seed}
        paired.append(dict(seed=seed,**{field:dict(serial_minus_concurrent=r['serial'][field]-r['concurrent'][field],
            concurrent_minus_larger_single=r['concurrent'][field]-max(r['input'][field],r['output'][field]),
            serial_minus_sum_singles=r['serial'][field]-r['input'][field]-r['output'][field]) for field in ('cycles','ns')}))
    report=dict(schema='gh200-q07-input-output-score/1',rows=rows,conditions=conditions,pairs=paired,
        all_performance_eligible=True,full_witness_elements=12*81952,qualification='development_joint_window',
        absolute_price_prediction=None,independent_C2=False,plan_sha256=sha(root/'plan.json'),build_sha256=sha(root/'build/identity.json'))
    if prepared:
        report.update(schema='gh200-q07-prepared-joint-score/1',qualification=plan['qualification'],prepared=True,
            **prepared_judgment(plan,rows))
    save(root/'score.json',report)
    if emit:print(json.dumps(dict(status='scored',processes=12,full_witness_elements=12*81952)))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,required=True);p.add_argument('--headers',type=Path,required=True)
    p.add_argument('--prepared',action='store_true')
    p=sub.add_parser('compile');p.add_argument('--output',type=Path,required=True);p.add_argument('--nvcc',default='nvcc')
    p=sub.add_parser('score');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    if args.command=='prepare':prepare(root,args.headers,args.prepared)
    elif args.command=='compile':compile_probe(root,args.nvcc)
    else:score(root)


if __name__=='__main__':
    main()

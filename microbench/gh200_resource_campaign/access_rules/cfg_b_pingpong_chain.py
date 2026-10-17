#!/usr/bin/env python3
"""cfg_b 三输出真实pingpong程序的有限CPU准备。"""
import argparse
import json
import math
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from cfg_b_program import build_program, expand_cta, epilogue_smem_addresses
from l2_saturation import SEEDS, command_record, digest, read_json, require, save_json
from mbarrier_service import BarrierPhase

HERE = Path(__file__).resolve().parent
SCHEMA = 'gh200-cfg-b-pingpong-plan/1'
SHAPE = dict(m=384,n=128,k=832,stages=6,output_tiles=3,ktiles_per_output=13)


def cases():
    return [dict(id='e_s6_k13_tiles3_'+('trace' if trace else 'plain'),trace=trace,
                 split='dev',**SHAPE) for trace in (False,True)]


def reference(seed):
    rows = np.arange(384,dtype=np.int64)[:,None]
    columns = np.arange(128,dtype=np.int64)[None,:]
    # 832 = 48*17+16；按完整17周期的整数点积及余段独立复算。
    inner = np.arange(17,dtype=np.int64)
    a = (7*rows+13*inner+3*seed) % 17-8
    b = (5*inner[:,None]+11*columns+5*seed) % 17-8
    numerator = 48*(a @ b)+(a[:,:16] @ b[:16,:])
    return (numerator/1024).astype(np.float32)


def program_reference():
    row=dict(id='e_s6_k13_tiles3',config='cfg_b',m=384,n=128,k=832,stages=6,
             sm_count=1,swizzle=1,raster_order='along_m',input_mode='dyadic',seed=3)
    setup=dict(grid=[1,1,1],stages=6,requested_sm_count=1,scheduler_sm_count=1)
    program=build_program(row,setup,dict(qualification='CPU声明程序；grid待运行查询核对'))
    graph=expand_cta(program,0)
    return dict(qualification='无硬件时间的既有程序图展开',program=program,graph=graph,
        counts=dict(input_groups=len(graph['groups']),input_requests=len(graph['requests']),
            wgmma_instructions=8*len(graph['groups']),output_atoms=3*8,
            input_logical_bytes=sum(item['bytes'] for item in graph['requests']),
            output_bytes=384*128*4))


def order_reference():
    """复用计数barrier核对OrderedSequenceBarrier<2,2>的六次交接。"""
    barriers=[[BarrierPhase(128) for _ in range(2)] for _ in range(2)]
    records=[]
    for output in range(3):
        group=output % 2
        phase=(1 if group==0 else 0) ^ ((output//2) % 2)
        for stage,kind in enumerate(('math','epilogue')):
            require(barriers[stage][group].wait_complete(phase),'order等待未满足')
            target=(group+1) % 2
            for _ in range(128):
                barriers[stage][target].arrive()
            records.append(dict(output=output,group=group,stage=stage,kind=kind,
                wait_phase=phase,signalling_group=target,arrivals=128,
                target_phase_after=barriers[stage][target].phase))
    return dict(qualification='CPU相位与到达数见证，无时间价格',records=records,
                barriers=[[barrier.snapshot() for barrier in row] for row in barriers])


def output_slots():
    return [dict(subtile=subtile,source_slot=subtile % 2,bytes=8192,
                 sts64_word_addresses=epilogue_smem_addresses(subtile)) for subtile in range(8)]


EVENTS = ('first_mma','mma_return','math_arrive_after','main_end','epilogue_permit',
          'store_return','store_tail_return','epilogue_arrive_after')


def validate(folder,row,case,seed):
    require(row['schema']=='gh200-cfg-b-pingpong-chain/1' and row['status']=='measured',
            'probe状态不符')
    require(row['trace']==case['trace'] and row['seed']==seed,'测量条件不符')
    require(all(row['conditions'].get(key)==value for key,value in SHAPE.items()) and
            row['conditions']['scheduler_sm_count']==1,'真实程序条件不符')
    device,resources=row['device'],row['resources']
    require('GH200' in device['name'] and device['sms']==132 and
            (device['major'],device['minor'])==(9,0),'目标设备不符')
    require(resources['threads']==384 and resources['grid']==[1,1,1] and
        resources['local_size_bytes']==0 and resources['occupancy_ctas_per_sm']==1,
        '资源或物理grid不符')
    require(row['timing']['cuda_event_ms']>0,'完整kernel计时不符')
    correctness=row['correctness']
    require(correctness['checked_elements']==49152 and correctness['errors']==0 and
        correctness['max_abs_error']==0 and correctness['guard_elements']==32 and
        correctness['guard_errors']==0,'完整输出或分配边界错误')
    require(row['guard_bits']==[0x7fc12345]*32,'首尾sentinel被改写')
    require(row['output_file']=='output.f32' and row['output_elements']==49152 and
            row['output_bytes']==196608,'完整输出见证不符')
    output=folder/'output.f32'
    require(output.stat().st_size==196608 and
        np.array_equal(np.fromfile(output,dtype='<f4').reshape(384,128),reference(seed)),
        '独立整数参考与完整输出不符')
    warmup=row['warmup']
    windows=warmup['elapsed_ms']
    require(warmup['limit']==60 and 8<=len(windows)<=60 and min(windows)>0,'暖机记录不符')
    mean=sum(windows[-5:])/5
    cv=math.sqrt(sum((value-mean)**2 for value in windows[-5:])/4)/mean
    require(warmup['converged']==(cv<=.02) and
        math.isclose(cv,warmup['cv'],rel_tol=1e-7,abs_tol=1e-12),'暖机CV不能复算')
    intervals=[]
    if case['trace']:
        trace=row['trace_detail']
        require(len(trace['outputs'])==3 and len(trace['finals'])==2,'trace输出／角色不完整')
        require(type(trace['entry']['smid']) is int and trace['entry']['smid']>=0,'trace SM身份不符')
        for output_id,item in enumerate(trace['outputs']):
            require(item['output']==output_id and item['consumer_wg']==output_id%2 and
                item['local_seq']==output_id//2 and item['coord']==[output_id,0],
                'trace工作列表不符')
            for unit in ('cycle','ns'):
                points=[item[event][unit] for event in EVENTS]
                require(trace['entry'][unit]<=points[0] and points==sorted(points),
                        '输出内端点顺序错误')
                if output_id:
                    previous=trace['outputs'][output_id-1]
                    require(item['first_mma'][unit]>=previous['mma_return'][unit],
                            '下一数学组早于必要数学交接')
                    require(item['epilogue_permit'][unit]>=previous['store_tail_return'][unit],
                            '下一输出早于必要输出源释放')
                if output_id==2:
                    require(item['first_mma'][unit]>=trace['outputs'][0]['store_tail_return'][unit],
                            '同一数学WG过早重入')
            intervals.append(dict(output=output_id,
                final_compute_upper={unit:item['main_end'][unit] for unit in ('cycle','ns')},
                output_source_read_upper={unit:item['store_tail_return'][unit] for unit in ('cycle','ns')}))
        for group,item in enumerate(trace['finals']):
            require(item['consumer_wg']==group and item['tiles']==(2 if group==0 else 1) and
                item['smid']==trace['entry']['smid'],'消费者退出身份不符')
            last=trace['outputs'][2 if group==0 else 1]
            for unit in ('cycle','ns'):
                require(item[unit]>=last['epilogue_arrive_after'][unit],'退出早于最后交接')
                require(trace['entry'][unit]<=trace['producer_first_work'][unit]<=
                    trace['outputs'][0]['first_mma'][unit],'producer起点顺序错误')
    else:
        require(row['trace_detail'] is None,'plain包含trace')
    return dict(device_uuid=device['uuid'],resources=resources,timing=row['timing'],
        output_sha256=digest(output),performance_eligible=warmup['converged'],
        intervals=intervals,qualification='完整kernel事件时间及软件端点；全局输出在stream完成后校验')


def make_overlay(source):
    kernel='cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp'
    collective='cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp'
    for name in (kernel,collective):
        target=source/'overlay'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'overlay-template'/name,target)
    path=source/'overlay'/kernel
    text=path.read_text()
    math_site="        // Cue for next Math WG's MMA to start\n        math_wg_order_barrier.arrive();"
    store_site='        );\n\n        // TMA store pipeline wait'
    epilogue_site="        // Cue for next Math WG's Epilogue to start\n        math_wg_order_barrier.arrive();"
    for site,replacement in (
        (math_site,'        e_stamp(E_MMA_RETURN, v06_tile_seq);\n'+math_site+
                   '\n        e_stamp(E_MATH_ARRIVE_AFTER, v06_tile_seq);'),
        (store_site,'        );\n        e_stamp(E_STORE_RETURN, v06_tile_seq);\n\n        // TMA store pipeline wait'),
        (epilogue_site,epilogue_site+'\n        e_stamp(E_EPILOGUE_ARRIVE_AFTER, v06_tile_seq);')):
        require(text.count(site)==1,'冻结overlay插入点已变化')
        text=text.replace(site,replacement,1)
    path.write_text(text)
    (source/'overlay/v06_trace.hpp').write_text('#pragma once\n#include "../cfg_b_pingpong_trace.hpp"\n')


def prepare(root,headers,overlay_root):
    root.mkdir(parents=True,exist_ok=False)
    source=root/'source'
    source.mkdir()
    for name in ('cfg_b_pingpong_chain.py','cfg_b_program.py','case_inputs.py','register_quota.py',
                 'supply_model.py','v08_model.py','v06_model.py','mbarrier_service.py','l2_saturation.py',
                 'test_cfg_b_pingpong_chain.py','test_cfg_b_pingpong_records.py'):
        shutil.copy2(HERE/name,source/name)
    for name in ('cfg_b_pingpong_chain.cu','cfg_b_pingpong_trace.hpp','r00_common.hpp'):
        shutil.copy2(HERE/'probes'/name,source/name)
    shutil.copy2(headers,source/'cutlass-3.9.2-include.tar.gz')
    for name in ('cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp',
                 'cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp'):
        target=source/'overlay-template'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(overlay_root/name,target)
    make_overlay(source)
    save_json(root/'plan.json',dict(schema=SCHEMA,cases=cases(),seeds=list(SEEDS),
        GPU_execution=False,dev_processes=6,heldout_processes=0,
        source_sha256={str(path.relative_to(source)):digest(path)
                       for path in sorted(source.rglob('*')) if path.is_file()}))
    save_json(root/'program-reference.json',program_reference())
    save_json(root/'order-reference.json',order_reference())
    save_json(root/'output-source-slots.json',output_slots())
    references=root/'integer-reference'
    references.mkdir()
    for seed in SEEDS:
        reference(seed).astype('<f4').tofile(references/f'seed-{seed}.f32')
    save_json(root/'commands.json',dict(executed=False,commands=[dict(case_id=row['id'],seed=seed,
        argv=[str(root/'build'/('trace' if row['trace'] else 'plain')),'--seed',str(seed),
              '--warmup-limit','60']) for row in cases() for seed in SEEDS]))
    print(f'PREPARED {root}: 2形状／6未执行开发命令；GPU0次')


def compile_probe(root,nvcc):
    plan=read_json(root/'plan.json')
    require(plan['schema']==SCHEMA,'准备包schema不符')
    for name,checksum in plan['source_sha256'].items():
        require(digest(root/'source'/name)==checksum,f'准备包源码变更：{name}')
    compiler=shutil.which(nvcc)
    require(compiler is not None,'nvcc未找到')
    build=root/'build'
    build.mkdir(exist_ok=False)
    command_record(build/'version',[compiler,'--version'],30)
    flags=['-std=c++17','-O3','-DNDEBUG','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v']
    with tempfile.TemporaryDirectory(prefix='cfg-b-pingpong-headers-') as temporary:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:
            archive.extractall(temporary,filter='data')
        for row in cases():
            name='trace' if row['trace'] else 'plain'
            variant_flags=['-DCFG_B_PINGPONG_TRACE','-I'+str(root/'source/overlay')] if row['trace'] else []
            command_record(build/(name+'-compile'),[compiler,*flags,*variant_flags,
                '-I'+temporary+'/include',root/'source/cfg_b_pingpong_chain.cu','-o',build/name],180)
    cuobjdump=Path(compiler).resolve().parent/'cuobjdump'
    for name in ('plain','trace'):
        command_record(build/(name+'-disassemble'),[cuobjdump,'--dump-sass',build/name],60)
        command_record(build/(name+'-inspect'),[build/name,'--inspect-program'],30)
    save_json(build/'identity.json',dict(plan_sha256=digest(root/'plan.json'),
        functions={name:dict(probe_sha256=digest(build/name),
            sass_sha256=digest(build/(name+'-disassemble/stdout')),
            program_query=read_json(build/(name+'-inspect/stdout'))) for name in ('plain','trace')},
        flags=flags,compiler=str(Path(compiler).resolve()),static_instruction_review='待核对'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--headers',type=Path,required=True)
    prep.add_argument('--overlay-root',type=Path,required=True)
    build=sub.add_parser('compile')
    build.add_argument('--output',type=Path,required=True)
    build.add_argument('--nvcc',default='nvcc')
    check=sub.add_parser('check')
    check.add_argument('--folder',type=Path,required=True)
    check.add_argument('--plan',type=Path,required=True)
    check.add_argument('--case-id',required=True)
    check.add_argument('--seed',type=int,choices=SEEDS,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.output.resolve(),args.headers.resolve(),args.overlay_root.resolve())
    elif args.command=='compile':
        compile_probe(args.output.resolve(),args.nvcc)
    else:
        case=next(row for row in read_json(args.plan)['cases'] if row['id']==args.case_id)
        print(json.dumps(validate(args.folder,read_json(args.folder/'stdout'),case,args.seed),
                         ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

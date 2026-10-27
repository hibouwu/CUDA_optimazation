#!/usr/bin/env python3
"""A/B请求顺序的有限开发判别：固定描述符、来源目的和访问字节。"""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import statistics
import tarfile
import tempfile
import time

import q02_prefill_observer as observer
import q02_tma_stride as stride
from l2_saturation import command_record,digest,exclusive_lock,inspect_allocated_gpu,read_json,require,save_json

HERE=Path(__file__).resolve().parent
SCHEMA='gh200-q02-request-order/1'
STRIDES=[[4096,2304],[4104,2304]]


def prepare(root,reference):
    old=read_json(reference/'score.json')
    require(old['observation_qualified'] is True,'752363参考资格未通过')
    controls={str(c['lda']):{metric:value['median'] for metric,value in c['windows'].items()}
        for c in old['conditions'] if c['descriptor_factor']==1 and c['trace']==0}
    prefixes={str(c['lda']):c['prefill']['cycles']['median'] for c in old['conditions']
        if c['descriptor_factor']==1 and c['trace']==-1}
    effect=next(r for r in old['responses'] if r['descriptor_factor']==1)['prefill_A16']['cycles']['median']
    require(controls=={'4096':{'cycles_median':21808,'ns_median':10944},
        '4104':{'cycles_median':22298,'ns_median':11200}} and
        prefixes=={'4096':1349,'4104':1839} and effect==490,'旧Q4控制数值不符')
    require(not (root/'plan.json').exists(),'本轮准备已经存在')
    for directory in ('source','source-before','reference'):(root/directory).mkdir(parents=True,exist_ok=True)
    shutil.copy2(HERE/'q02_request_order.py',root/'source/q02_request_order.py')
    for name in ('q02_prefill_observer.py','q02_tma_stride.py','l2_saturation.py','r00_common.hpp','cutlass-3.9.2-include.tar.gz'):
        shutil.copy2(reference/'source'/name,root/'source'/name)
    shutil.copy2(reference/'source/tma_gemm_sources.cu',root/'source-before/tma_gemm_sources.cu')
    shutil.copy2(reference/'worklist.txt',root/'worklist.txt')
    for name in ('score.json','binding.json','plan.json','approved-target.json'):
        shutil.copy2(reference/name,root/'reference'/name)
    for name in ('probe','kernel-code.json','target-review.json'):
        shutil.copy2(reference/'build'/name,root/'reference'/name)
    shutil.copytree(reference/'build/after/disassemble',root/'reference/disassemble')
    raw_origins=[]
    for filename,verify,split in (('commands.json',0,'formal'),('correctness_commands.json',1,'correctness')):
        tasks=[]
        for prior in read_json(reference/filename):
            task=copy.deepcopy(prior);seed=task['seed']
            raw_path=reference/'runs'/split/task['id']/'stdout'
            lines=raw_path.read_text().splitlines(keepends=True)
            selected=[i for i,line in enumerate(lines) if json.loads(line)['descriptor_factor']==1]
            target=root/f'reference/{split}-seed{seed}.ndjson'
            target.write_text(''.join(lines[i] for i in selected))
            raw_origins.append(dict(source=str(raw_path),source_sha256=digest(raw_path),selected_lines_zero_based=selected,
                subset=str(target.relative_to(root)),subset_sha256=digest(target)))
            pairs=[(order,shape) for order in (1,2) for shape in STRIDES]
            if seed!=3:pairs.reverse()
            task.pop('paired_source_steps',None)
            task.update(id=f'q4-request-order-seed{seed}',role='development_request_order',
                paired_strides=[shape for _,shape in pairs],paired_request_orders=[order for order,_ in pairs],
                paired_descriptor_factors=[1,1,1,1],
                paired_traces=[0,-1] if seed==3 else [-1,0])
            argv=task['argv']
            if '--paired-source-steps' in argv:
                index=argv.index('--paired-source-steps');del argv[index:index+2]
            for flag,value in (('--verify',str(verify)),('--paired-traces',','.join(map(str,task['paired_traces']))),
                    ('--paired-strides',','.join(f'{a}:{b}' for a,b in task['paired_strides'])),
                    ('--source-k-step','1'),('--descriptor-factor','1'),('--paired-descriptor-factors','1,1,1,1'),
                    ('--request-order','1'),
                    ('--paired-request-orders',','.join(map(str,task['paired_request_orders'])))):
                if flag in argv:argv[argv.index(flag)+1]=value
                else:argv.extend([flag,value])
            if verify:task['expected_checked']=524288
            tasks.append(task)
        require(len(tasks)==3,'每类必须三个八臂进程');save_json(root/filename,tasks)
    plan=dict(schema=SCHEMA,qualification='development_request_order',intervention='request_order',
        independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        gpu_uuid=read_json(reference/'plan.json')['gpu_uuid'],required_smid=124,seeds=[3,22,41],active_groups=[4],
        request_orders=[1,2],strides=STRIDES,traces=[0,-1],source_k_step=1,m=2304,n=2304,k=4096,
        ctas=1,groups=32,ring=64,physical_slots=4,alloc_lda=4104,alloc_ldb=2360,tile=[7,6],
        request_mask=7,request_bytes=[16384,8192,8192],swizzle='SW128',release_delay_ns=0,wait_delay_cycles=0,
        warmup_request_order=1,warmup_source_k_step=1,warmup_groups=64,warmup_active_groups=4,
        warmup_calls=10,warmup_trace=-1,development_conditions=8,correctness_processes=3,formal_processes=3,
        numeric_arm_checks=24,formal_arm_observations=24,numeric_checked_values=12582912,
        relative_tolerance=.05,window_relative_spread=.05,prefill_cycle_relative_spread=.05,
        perturbation_tolerance=.05,ratio_absolute_tolerance=.05,control_windows=controls,
        control_prefill_cycles=prefixes,control_prefill_effect_cycles=effect,prefill_ns_report_only=True,
        full_window_stability_unit='cycles_and_ns',reference_job=752363,
        reference_score_sha256=digest(reference/'score.json'),reference_binding_sha256=digest(reference/'binding.json'),
        before_source_sha256=digest(root/'source-before/tma_gemm_sources.cu'),original_probe_sha256=old['probe_sha256'],
        allocation_budget_seconds=600,maximum_gpu_seconds=3600,maximum_repairs=2,automatic_retries=False,
        descriptor_factor=1,warmup_descriptor_factor=1,
        order1=['A','B0','B1'],order2=['B0','B1','A'],
        candidates=[dict(name='order_invariant_response',ratio=1)],
        ratio_definition='median_seed(prefill_A16_order2-prefill_A0_order2)/median_seed(prefill_A16_order1-prefill_A0_order1), cycles',
        classification='abs(R-1)<=.05: response retained under this request order; otherwise stop order-invariant form; qualification failure: information insufficient',
        frozen_at_ns=time.time_ns())
    save_json(root/'reference/raw-origins.json',raw_origins)
    save_json(root/'plan.json',plan)
    (root/'README.md').write_text('''# Q02：A/B请求顺序

order1的A→B0→B1与order2的B0→B1→A，各配A lda4096/4104及trace0/−1，共八开发条件。Q4/G32、B ldb2304、SW128、四物理槽、完整描述符、source_step1与descriptor_factor1固定。每seed同进程八臂，order与stride同长度zip、trace外层；seed3正序，22/41整体反序。各臂统一order1、factor1、step1、Q4/G64、trace−1十次暖机。先3verify再3formal，共48臂、12582912值。

order1 plain完整窗须复现752363 factor1的21808/22298cycle、10944/11200ns；prefill须复现1349/1839cycle和同seed A16效应490cycle，各5%。完整双单位与prefill cycle三进程范围≤中位5%，同seed sparse/plain完整窗扰动≤5%。R为两个order各自同seed A16−A0 prefill增量中位之比；唯一采前null为R=1±.05。资格失败或零分母信息不足；合格但不命中停止order不变形式，均不自动重采。ns前缀只报告，独立C2/C3及完整GEMM验收均false。

stdout逐臂核order/zip、固定factor1/step1、同VA/SM124、卡/资源、完整加载值和唯一prefill点。同stride跨order及trace的A/B descriptor hex必须相同；A encode明文保留原builder字段。reference仅抽取752363 factor1原始行，raw-origins保存原stdout SHA和选中行号。

```sh
python -B source/q02_request_order.py build-only --root "$PWD"
python -B source/q02_request_order.py execute --root "$PWD"
python -B source/q02_request_order.py score --root "$PWD" --output /tmp/q02-request-order-score.json
```

prepare不绑定、不编译或采样。源码就绪后主线程冻结build输入，CUDA12.9.41 ARM构建保存完整机器字与资源；actual静态UTMA数原样报告，由主线程审核每动态路径三请求和顺序。execute核批准的plan/source/binary及GPU身份。分配10分钟、GPU≤1小时、最多两次定向修复，无自动重试、不接默认模型。
''')
    print('PREPARED: 8 request-order conditions; 3 verify +3 formal; no binding/build/GPU',flush=True)


def load_plan(root,execution=False):
    for name,checksum in read_json(root/('binding.json' if execution else 'build-input-binding.json'))['files'].items():
        require(digest(root/name)==checksum,'冻结身份变更：'+name)
    plan=read_json(root/'plan.json');require(plan['schema']==SCHEMA,'执行范围不符')
    return plan


def review_build(root):
    review=observer.review_build(root)
    review.update(status='pending_root_request_order_sass_review',request_orders=[1,2])
    review['rows'][0]['label']='request_order'
    review['basic_gate_passed']=all(row['zero_local_stack'] and not row['compile_warning_c7515']
        for row in review['rows'])
    review['static_utma_count_unreviewed']=True
    return review


def build_only(root):
    load_plan(root);start=time.monotonic();folder=root/'build';folder.mkdir(exist_ok=False)
    compiler=shutil.which('nvcc');require(compiler is not None,'nvcc缺失')
    command_record(folder/'version',[compiler,'--version'],stride.remaining(start,20))
    version=(folder/'version/stdout').read_text()
    require('V12.9.41' in version and 'compiler.35813241_0' in version,'需要冻结CUDA12.9.41')
    flags=['-std=c++17','-O3','-DNDEBUG','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v']
    disassembler=Path(compiler).resolve().parent/'cuobjdump'
    with tempfile.TemporaryDirectory(prefix='q02-order-cutlass-') as directory:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:archive.extractall(directory,filter='data')
        command_record(folder/'after/compile',[compiler,*flags,'-I'+directory+'/include','-I'+str(root/'source'),
            root/'source/tma_gemm_sources.cu','-o',folder/'probe'],stride.remaining(start,180))
        command_record(folder/'after/disassemble',[disassembler,'--dump-sass',folder/'probe'],stride.remaining(start,40))
        command_record(folder/'after/resources',[disassembler,'--dump-resource-usage',folder/'probe'],stride.remaining(start,30))
    save_json(folder/'kernel-code.json',observer.kernel_records((folder/'after/disassemble/stdout').read_text()))
    review=review_build(root);save_json(folder/'target-review.json',review)
    require(review['basic_gate_passed'],'无local和stack/C7515基础门槛未通过')
    print('BUILT ONLY: root request-order SASS approval pending; no GPU',flush=True)


def approved_target(root):
    review=review_build(root);approved=read_json(root/'approved-target.json')
    require(review==read_json(root/'build/target-review.json') and review['basic_gate_passed'],'构建证据不符')
    require(approved['approved'] is True and approved['probe_sha256']==review['probe_sha256'] and
        approved['kernel_code_sha256']==review['kernel_code_sha256'] and
        approved['target_review_sha256']==digest(root/'build/target-review.json') and
        approved['plan_sha256']==digest(root/'plan.json') and
        approved['probe_source_sha256']==digest(root/'source/tma_gemm_sources.cu'),'目标plan/source/binary未获核准')


def validate_process(root,folder,task,uuid):
    receipt=stride.successful_receipt(folder);command=receipt['argv'][:];probe=Path(command[0])
    require(probe.name=='probe' and probe.parent.name=='build','binary路径不符')
    require(command[command.index('--worklist')+1]==str(probe.parent.parent/'worklist.txt'),'工作列表路径不符')
    command[0]=task['argv'][0];command[command.index('--worklist')+1]='worklist.txt'
    require(command==task['argv'],'执行参数与冻结命令不符')
    raw=[json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.strip()]
    require(len(raw)==8,'八臂缺失');arguments=dict(zip(task['argv'][1::2],task['argv'][2::2]))
    require((root/'worklist.txt').read_text().split()==['2304','2304','4096','1','0','0','7','6'],'工作列表不符')
    fields={'ctas':'--ctas','active_groups':'--measure-active','groups':'--measure-groups','ring':'--ring',
        'verify':'--verify','seed':'--seed','warmup':'--warmup','source_phase_stride':'--source-phase-stride',
        'source_tile_shift':'--source-tile-shift','work_index':'--work-index','warmup_work_offset':'--warmup-work-offset',
        'release_delay_ns':'--release-delay-ns','pollute_kib':'--pollute-kib','launch_skew_ns':'--launch-skew-ns',
        'warmup_skew_ns':'--warmup-skew-ns','alloc_lda':'--alloc-lda','alloc_ldb':'--alloc-ldb'}
    resource=read_json(root/'build/target-review.json')['rows'][0]['resources'];checked=[];descriptors={}
    for index,row in enumerate(raw):
        require(row['schema']=='cfg-b-gemm-source-probe/1' and row['gpu_uuid']==uuid,'schema/卡身份不符')
        require(all(row[name]==int(arguments[flag]) for name,flag in fields.items()),'逐臂参数不符')
        require(row['warmup_trace']==-1 and row['warmup_active_groups']==4 and row['warmup_groups']==64 and
            row['warmup_source_phase_stride']==0 and row['record_warmup']==0 and
            not row['warmup_ctas'] and not row['group_events'],'暖机或trace范围不符')
        require(row['paired_strides']==task['paired_strides'] and row['pair_index']==index and
            row['trace_index']==index//4 and row['stride_index']==index%4 and
            row['paired_traces']==task['paired_traces'] and row['trace']==task['paired_traces'][index//4] and
            [row['lda'],row['ldb']]==task['paired_strides'][index%4],'逐臂stride/trace或次序不符')
        order=task['paired_request_orders'][index%4]
        require(row['paired_request_orders']==task['paired_request_orders'] and
            row['request_order']==order and row['warmup_request_order']==1 and
            row['descriptor_factor']==row['warmup_descriptor_factor']==1 and
            row['paired_descriptor_factors']==task['paired_descriptor_factors']==[1,1,1,1] and
            row['source_k_step']==row['warmup_source_k_step']==1,'请求顺序/固定descriptor factor和K步长不符')
        require(row['request_mask']==row['warmup_request_mask']==7 and
            row['transaction_bytes']==row['warmup_transaction_bytes']==32768 and
            row['wait_delay_cycles']==row['warmup_wait_delay_cycles']==0,'三请求/事务字节或D0不符')
        validate_encode(row)
        pair=(row['descriptor_a'],row['descriptor_b']);descriptor_key=(row['lda'],row['ldb'])
        require(all(re.fullmatch('[0-9a-f]{256}',value) for value in pair) and
            descriptors.setdefault(descriptor_key,pair)==pair,'同stride的order/trace descriptor变化')
        require(all(row[name]==raw[0][name] for name in stride.ADDRESSES),'同进程地址变化')
        require(row['source_a_address']>0 and row['source_b_address']>0 and
            row['source_a_address']%128==row['source_b_address']%128==0,'源地址/对齐不符')
        require(row['m']==row['n']==2304 and row['allocated_a_elements']==9455616 and
            row['allocated_b_elements']==9666560,'形状或固定分配不符')
        require(row['errors']==row['pollution_errors']==row['pollution_checked']==0 and
            row['checked']==(524288 if row['verify'] else 0),'完整加载数值不符')
        require(row['observed_sms']==row['max_active_ctas_per_sm']==1 and row['smem']==132096 and
            row['local_bytes']==0 and row['registers']==int(re.search(r'\bREG:(\d+)\b',resource)[1]),'驻留/构建资源不符')
        require(len(row['ctas_data'])==1,'CTA数量不符');cta=row['ctas_data'][0]
        require(cta['cta']==0 and cta['tile']==[7,6] and cta['smid']==124,'CTA/坐标/SM124不符')
        for unit in (0,1):require(type(cta['begin'][unit]) is int and type(cta['end'][unit]) is int and
            0<cta['begin'][unit]<cta['end'][unit],'完成戳无效')
        prefill=None
        if row['trace']==-1:
            point=cta['prefill_end'];require(len(point)==2 and all(type(point[u]) is int and
                cta['begin'][u]<point[u]<cta['end'][u] for u in (0,1)),'预填结束戳无效')
            prefill=dict(cycles=point[0]-cta['begin'][0],ns=point[1]-cta['begin'][1])
        else:require('prefill_end' not in cta,'plain读取了暖机预填戳')
        checked.append(dict(seed=row['seed'],active_groups=4,request_order=order,source_k_step=1,
            trace=row['trace'],lda=row['lda'],ldb=row['ldb'],pair_index=index,checked_values=row['checked'],
            numerical_qualified=bool(row['verify']),stdout_sha256=receipt['stdout_sha256'],source=str(folder.relative_to(root)),
            device_uuid=uuid,prefill=prefill,metrics=dict(cycles_median=cta['end'][0]-cta['begin'][0],
                ns_median=cta['end'][1]-cta['begin'][1],ns_envelope=cta['end'][1]-cta['begin'][1])))
    return checked


def validate_encode(row):
    """本轮两种order均保留原builder A编码。"""
    expected=dict(data_type='UINT16',rank=3,global_address=row['source_a_address'],
        global_dimensions=[4096,2304,1],global_strides_bytes=[row['lda']*2,0],
        box_dimensions=[64,128,1],element_strides=[1,1,1],interleave='NONE',swizzle='128B',
        l2_promotion='128B',oob_fill='NONE',load_cache_hint='EVICT_NORMAL',source='builder')
    require(row['descriptor_a_fields']==expected,'A encoder明文不符')
    require(re.fullmatch('[0-9a-f]{256}',row['descriptor_a_builder']) is not None and
        row['descriptor_a']==row['descriptor_a_builder'],'原builder descriptor变更')


def summarize_rows(plan,numeric,formal):
    key=lambda r:(r['seed'],r['request_order'],r['lda'],r['ldb'],r['trace'])
    by_numeric={key(r):r for r in numeric};by_formal={key(r):r for r in formal}
    expected={(seed,order,lda,ldb,trace) for seed in plan['seeds'] for order in plan['request_orders']
        for lda,ldb in plan['strides'] for trace in plan['traces']}
    require(len(numeric)==len(formal)==24 and by_numeric.keys()==by_formal.keys()==expected,'数值/正式24臂不对应')
    require(all(r['numerical_qualified'] and r['checked_values']==524288 for r in numeric) and
        sum(r['checked_values'] for r in numeric)==plan['numeric_checked_values'],'数值资格不符')
    conditions=[];controls=[];prefix_controls=[];perturbations=[];responses=[]
    for order in plan['request_orders']:
        for lda,ldb in plan['strides']:
            for trace in plan['traces']:
                rows=[by_formal[(seed,order,lda,ldb,trace)] for seed in plan['seeds']];windows={}
                for metric in ('cycles_median','ns_median'):
                    values=[r['metrics'][metric] for r in rows];median=statistics.median(values)
                    require(median>0,'完整窗须为正数');spread=(max(values)-min(values))/median
                    windows[metric]=dict(values=values,median=median,relative_spread=spread,passed=spread<=plan['window_relative_spread'])
                    if order==1 and trace==0:
                        old=plan['control_windows'][str(lda)][metric];error=abs(median-old)/old
                        controls.append(dict(lda=lda,metric=metric,expected=old,observed=median,relative_error=error,passed=error<=plan['relative_tolerance']))
                prefix=None
                if trace==-1:
                    prefix={}
                    for unit in ('cycles','ns'):
                        values=[r['prefill'][unit] for r in rows];median=statistics.median(values)
                        require(median>0,'prefill须为正数');spread=(max(values)-min(values))/median
                        prefix[unit]=dict(values=values,median=median,relative_spread=spread,
                            passed=spread<=plan['prefill_cycle_relative_spread'] if unit=='cycles' else None)
                        if order==1 and unit=='cycles':
                            old=plan['control_prefill_cycles'][str(lda)];error=abs(median-old)/old
                            prefix_controls.append(dict(lda=lda,expected=old,observed=median,relative_error=error,passed=error<=plan['relative_tolerance']))
                conditions.append(dict(active_groups=4,request_order=order,source_k_step=1,lda=lda,ldb=ldb,trace=trace,
                    windows=windows,prefill=prefix,passed=all(v['passed'] for v in windows.values()) and (trace==0 or prefix['cycles']['passed'])))
            for seed in plan['seeds']:
                plain=by_formal[(seed,order,lda,ldb,0)];sparse=by_formal[(seed,order,lda,ldb,-1)]
                for metric in ('cycles_median','ns_median'):
                    p=plain['metrics'][metric];s=sparse['metrics'][metric];error=abs(s-p)/p
                    perturbations.append(dict(seed=seed,request_order=order,lda=lda,metric=metric,plain=p,sparse=s,
                        relative_difference=error,passed=error<=plan['perturbation_tolerance']))
        prefix_response={};full_response={}
        for unit,metric in (('cycles','cycles_median'),('ns','ns_median')):
            values=[by_formal[(seed,order,4104,2304,-1)]['prefill'][unit]-
                by_formal[(seed,order,4096,2304,-1)]['prefill'][unit] for seed in plan['seeds']]
            prefix_response[unit]=dict(values=values,median=statistics.median(values));full_response[unit]={}
            for trace in plan['traces']:
                values=[by_formal[(seed,order,4104,2304,trace)]['metrics'][metric]-
                    by_formal[(seed,order,4096,2304,trace)]['metrics'][metric] for seed in plan['seeds']]
                full_response[unit][str(trace)]=dict(values=values,median=statistics.median(values))
        responses.append(dict(request_order=order,prefill_A16=prefix_response,full_window_A16=full_response))
    base=responses[0]['prefill_A16']['cycles']['median'];changed=responses[1]['prefill_A16']['cycles']['median']
    old=plan['control_prefill_effect_cycles'];error=abs(base-old)/old
    effect=dict(values=responses[0]['prefill_A16']['cycles']['values'],median=base,expected=old,relative_error=error,passed=error<=plan['relative_tolerance'])
    control_passed=all(r['passed'] for r in controls+prefix_controls+[effect]);stable=all(r['passed'] for r in conditions)
    perturbation_passed=all(r['passed'] for r in perturbations);qualified=control_passed and stable and perturbation_passed
    ratio=changed/base if base else None;matched=ratio is not None and abs(ratio-1)<=plan['ratio_absolute_tolerance']
    classification='response_retained_under_request_order' if matched else 'order_invariant_form_failed'
    if not qualified or ratio is None:classification='information_insufficient'
    decision='information_insufficient' if not qualified or ratio is None else \
        'development_compatible_order_invariant_response' if matched else 'stop_order_invariant_response'
    return dict(schema='gh200-q02-request-order-score/1',qualification=plan['qualification'],decision=decision,
        independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),numeric_checked_values=sum(r['checked_values'] for r in numeric),
        numeric=numeric,formal=formal,conditions=conditions,control=dict(windows=controls,prefill_cycles=prefix_controls,prefill_effect_cycles=effect,passed=control_passed),
        perturbations=perturbations,perturbation_passed=perturbation_passed,stable_windows_and_prefill=stable,
        observation_qualified=qualified,responses=responses,ratio_cycles=ratio,classification=classification,prefill_ns_report_only=True)


def summarize(root):
    plan=load_plan(root,True);approved_target(root)
    require(read_json(root/'allocation/identity.json')['uuid']==plan['gpu_uuid'],'分配卡身份不符')
    numeric=[];formal=[]
    for filename,split,rows in (('correctness_commands.json','correctness',numeric),('commands.json','formal',formal)):
        for task in read_json(root/filename):rows.extend(validate_process(root,root/'runs'/split/task['id'],task,plan['gpu_uuid']))
    score=summarize_rows(plan,numeric,formal);score.update(plan_sha256=digest(root/'plan.json'),probe_sha256=digest(root/'build/probe'))
    return score


def execute(root):
    plan=load_plan(root,True);approved_target(root);start=time.monotonic()
    environment=inspect_allocated_gpu(root/'allocation');require(environment['uuid']==plan['gpu_uuid'],'分配卡不是冻结参考卡')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        for filename,split in (('correctness_commands.json','correctness'),('commands.json','formal')):
            for task in read_json(root/filename):
                folder=root/'runs'/split/task['id'];command_record(folder,stride.probe_command(root,task),stride.remaining(start,60))
                save_json(folder/'validated.json',validate_process(root,folder,task,environment['uuid']))
                print(json.dumps(dict(split=split,process=task['id'],arms=8)),flush=True)
        save_json(root/'score.json',summarize(root))
    print('CLOSED: 24 numeric +24 formal arms; request-order development localization',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('prepare','build-only','execute','score'):
        item=sub.add_parser(mode);item.add_argument('--root',type=Path,required=True)
        if mode=='prepare':item.add_argument('--reference',type=Path,required=True)
        if mode=='score':item.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.mode=='prepare':prepare(root,args.reference.resolve())
    elif args.mode=='build-only':build_only(root)
    elif args.mode=='execute':execute(root)
    else:save_json(args.output,summarize(root))


if __name__=='__main__':main()

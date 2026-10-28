#!/usr/bin/env python3
"""首次预填结束的单点观测：固定分配配对、构建、执行与原始记录评分。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import tarfile
import tempfile
import time

import q02_tma_stride as stride
from l2_saturation import command_record,digest,exclusive_lock,inspect_allocated_gpu,read_json,require,save_json

HERE=Path(__file__).resolve().parent
SCHEMA='gh200-q02-prefill-observer/1'
STRIDES=[[4096,2304],[4104,2304]]


def prepare_prefill_gap(root,reference):
    prior=read_json(reference/'score.json')
    require(prior['observation_qualified'] is True,'752255原资格未通过')
    approved_target(reference)
    controls={str(c['lda']):{metric:window['median'] for metric,window in c['windows'].items()}
        for c in prior['conditions'] if c['source_k_step']==1 and c['trace']==0}
    prefixes={str(c['lda']):c['prefill']['cycles']['median'] for c in prior['conditions']
        if c['source_k_step']==1 and c['trace']==-1}
    response=next(row for row in prior['responses'] if row['source_k_step']==1)['prefill_A16']['cycles']['median']
    require(controls=={'4096':{'cycles_median':21819,'ns_median':10944},
        '4104':{'cycles_median':22317,'ns_median':11200}} and prefixes=={'4096':1389,'4104':1887} and response==498,
        '752255 step1完整窗/prefill/同seed响应控制不符')
    require(digest(reference/'build/probe')==prior['probe_sha256'],'752255 binary不符')
    root.mkdir(parents=True,exist_ok=False)
    for directory in ('source','source-before','reference'):(root/directory).mkdir()
    shutil.copy2(HERE/'q02_prefill_observer.py',root/'source/q02_prefill_observer.py')
    source=(HERE/'probes/tma_gemm_sources.cu').read_text()
    shutil.copy2(HERE/'probes/tma_gemm_sources.cu',root/'reference/canonical-source.cu')
    # 本轮固定三请求、D0、step1；保留参数签名及旧offset。
    source=source.replace('bool select_a = (request_mask & 1) != 0;','constexpr bool select_a = true;')
    source=source.replace('bool select_b = (request_mask & 6) != 0;','constexpr bool select_b = true;')
    source=source.replace('int transaction_bytes = 16384 * (int(select_a) + int(select_b));',
        'constexpr int transaction_bytes = 32768;')
    source=source.replace('      if (wait_delay_cycles) {\n        uint64_t wait_delay_begin=clock64();\n        while (clock64()-wait_delay_begin<uint64_t(wait_delay_cycles)) {}\n      }\n','')
    source=source.replace('(mask!=1 && mask!=6 && mask!=7)','mask!=7')
    source=source.replace('paired request masks must be 1, 6 or 7','prefill-gap paired request masks must be 7')
    source=source.replace('comma-separated 1/6/7 list','comma-separated 7 list')
    source=source.replace('end!=text.size() || cycles<0','end!=text.size() || cycles!=0')
    source=source.replace('paired wait delays must be nonnegative cycles','prefill-gap paired wait delays must be 0')
    source=source.replace('delay<0 || wait_delay_cycles<0 ||','delay<0 || wait_delay_cycles!=0 ||')
    source=source.replace('(request_mask!=1 && request_mask!=6 && request_mask!=7)','request_mask!=7')
    source=source.replace('(step!=1 && step!=2)','step!=1')
    source=source.replace('paired source steps must be 1 or 2','prefill-gap paired source steps must be 1')
    source=source.replace('comma-separated 1/2 list','comma-separated 1 list')
    source=source.replace('(source_k_step!=1 && source_k_step!=2)','source_k_step!=1')
    require('if (wait_delay_cycles)' not in source and 'constexpr int transaction_bytes = 32768;' in source and
        'source_k_step!=1 ||' in source,'私有三请求/D0/step1 specialization不符')
    (root/'source/tma_gemm_sources.cu').write_text(source)
    for name in ('q02_tma_stride.py','l2_saturation.py','r00_common.hpp','cutlass-3.9.2-include.tar.gz'):
        shutil.copy2(reference/'source'/name,root/'source'/name)
    shutil.copy2(reference/'source/tma_gemm_sources.cu',root/'source-before/tma_gemm_sources.cu')
    shutil.copy2(reference/'worklist.txt',root/'worklist.txt')
    shutil.copy2(HERE.parents[2]/'Docs/ModelEvaluation/gemm/PLAN.md',root/'reference/campaign-PLAN.md')
    for name in ('score.json','binding.json','plan.json','commands.json','correctness_commands.json','approved-target.json'):
        shutil.copy2(reference/name,root/'reference'/name)
    for name in ('probe','target-review.json','kernel-code.json'):
        shutil.copy2(reference/'build'/name,root/'reference'/name)
    shutil.copytree(reference/'build/after/disassemble',root/'reference/disassemble')
    (root/'reference/original-prefill').mkdir()
    for name in ('plan.json','score.json'):
        shutil.copy2(reference/'reference'/name,root/'reference/original-prefill'/name)
    for filename,verify,split in (('commands.json',0,'formal'),('correctness_commands.json',1,'correctness')):
        tasks=[]
        for old in read_json(reference/filename):
            task=copy.deepcopy(old);seed=task['seed']
            shutil.copy2(reference/'runs'/split/task['id']/'stdout',root/f'reference/{split}-seed{seed}.ndjson')
            pairs=[(gap,stride) for gap in (0,8192) for stride in STRIDES]
            if seed!=3:pairs.reverse()
            task.pop('paired_source_steps',None)
            task.update(id=f'q4-prefill-gap-seed{seed}',role='development_prefill_issue_feedback',pair_indices=list(range(8)),
                paired_strides=[stride for _,stride in pairs],paired_prefill_gaps=[gap for gap,_ in pairs],
                paired_traces=[0,-1] if seed==3 else [-1,0])
            argv=task['argv']
            if '--paired-source-steps' in argv:
                index=argv.index('--paired-source-steps');del argv[index:index+2]
            for flag,value in (('--verify',str(verify)),('--paired-traces',','.join(map(str,task['paired_traces']))),
                    ('--paired-strides',','.join(f'{a}:{b}' for a,b in task['paired_strides'])),
                    ('--source-k-step','1'),('--prefill-gap-cycles','0'),
                    ('--paired-prefill-gaps',','.join(map(str,task['paired_prefill_gaps']))),
                    ('--request-mask','7'),('--wait-delay-cycles','0')):
                if flag in argv:argv[argv.index(flag)+1]=value
                else:argv+=[flag,value]
            if verify:task['expected_checked']=524288
            tasks.append(task)
        require(len(tasks)==3,'每类必须三个八臂配对进程');save_json(root/filename,tasks)
    plan=dict(schema=SCHEMA,qualification='development_prefill_issue_feedback',intervention='prefill_gap_cycles',
        independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        gpu_uuid=read_json(reference/'plan.json')['gpu_uuid'],required_smid=124,seeds=[3,22,41],active_groups=[4],
        strides=STRIDES,prefill_gaps=[0,8192],source_k_step=1,traces=[0,-1],m=2304,n=2304,k=4096,ctas=1,groups=32,ring=64,
        physical_slots=4,alloc_lda=4104,alloc_ldb=2360,request_mask=7,request_bytes=[16384,8192,8192],swizzle='SW128',tile=[7,6],
        warmup_source_k_step=1,warmup_prefill_gap_cycles=0,warmup_groups=64,warmup_active_groups=4,warmup_calls=10,
        warmup_trace=-1,warmup_wait_delay_cycles=0,warmup_request_mask=7,release_delay_ns=0,wait_hint_ns=10000000,
        development_conditions=8,correctness_processes=3,formal_processes=3,numeric_arm_checks=24,
        formal_arm_observations=24,numeric_checked_values=12582912,relative_tolerance=.05,window_relative_spread=.05,
        prefill_cycle_relative_spread=.05,net_prefill_cycle_relative_spread=.05,perturbation_tolerance=.05,
        ratio_absolute_tolerance=.05,control_windows=controls,control_prefill_cycles=prefixes,control_prefill_effect_cycles=response,
        prefill_ns_report_only=True,full_window_stability_unit='cycles_and_ns',reference_job=752255,
        reference_score_sha256=digest(reference/'score.json'),reference_binding_sha256=digest(reference/'binding.json'),
        before_source_sha256=digest(root/'source-before/tma_gemm_sources.cu'),original_probe_sha256=prior['probe_sha256'],
        allocation_budget_seconds=600,maximum_gpu_seconds=3600,maximum_repairs=2,automatic_retries=False,
        prefill_location='after initial active groups issue; before first full wait',
        gap_location='only initial prefill g>0; start/last clock after previous issue_group return and before next issue',
        gap_sum_definition='sum(last-start) from required spin clock reads; register accumulation; store after end',
        net_prefill_definition='raw prefill cycles minus actual gap_sum_cycles, then paired seed A16-A0',
        candidates=[dict(name='spacing_hidden_response',ratio=0),dict(name='spacing_retained_response',ratio=1)],
        ratio_definition='median_seed(net_prefill_A16_gap8192-net_prefill_A0_gap8192)/median_seed(net_prefill_A16_gap0-net_prefill_A0_gap0), cycles',
        classification='abs(R)<=.05: response hidden by this gap; abs(R-1)<=.05: response retained with this gap; both miss: stop both; qualification failure: information insufficient',
        signed_or_zero_delta_allowed=True,no_bare_price=True,no_refill_extrapolation=True,frozen_at_ns=time.time_ns())
    save_json(root/'plan.json',plan)
    (root/'README.md').write_text('''# Q02：预填组间间隔的发出反馈判别

Q4/G32固定，gap0/8192×A lda4096/4104×trace0/−1共八开发条件，B ldb2304。gap只在首次prefill的g>0组发出前，从上一issue_group返回后的首个clock读取开始；refill、fullwait、四物理槽及代次、barrier、发布/释放、三请求/SW128和verify不变。source_step总1；私有kernel固定mask7/D0，旧尾参offset保留，再追加gap参数。每seed同进程八臂，gap/stride同长度zip，seed3正序22/41整体逆序；暖机总gap0、step1、mask7、Q4/G64、trace−1十次，正式恢复目标gap/trace。先3verify后3formal，共24数值臂/24正式臂、12582912值。

自旋只使用必需的start/last clock读取，累计实际gap_sum_cycles于寄存器，end后写回；gap0跳过所有新增clock读取，没有循环存出或新观察点。净prefill=raw prefill cycle−gap_sum_cycles，再按同seed A16−A0并取中位。配对超时合计30cycle可把真隐藏的raw R推至30/498=0.06024，因此同时保留raw响应和实际控制span，不减未知观察开销。ns前缀只报告，不用cycle span换算扣减。

gap0 plain完整窗须复现752255 step1的21819/22317cycle、10944/11200ns，prefill窗1389/1887cycle及配对增量498cycle，各5%。八条件完整cycle/ns、raw及net prefill cycle的三进程范围分别≤中位5%；同seed sparse/plain完整窗扰动≤5%。不另设gap_sum稳定门槛。verify只负责完整数值和身份。raw含24576cycle gap可能范围3.76%通过而net A16=[1389,1887,2385]范围52.8%失败，此时信息不足。

R=gap8192的净prefill A16增量中位/gap0对应中位；abs(R)≤.05说明该响应可被本间隔隐藏，abs(R−1)≤.05保留本间隔下仍持续的响应，两者均失败停止两形式；资格失败或零分母为信息不足，零/负增量保留。只判本间隔的可隐藏性，不指定完成/准入事件、物理容量或片段价格，不外推refill，独立C2/C3及完整GEMM验收均false。

```sh
python -B cpu-check.py
python -B source/q02_prefill_observer.py build-only --root "$PWD"
python -B source/q02_prefill_observer.py execute --root "$PWD"
python -B source/q02_prefill_observer.py score --root "$PWD" --output /tmp/q02-prefill-gap-score.json
```

prepare保存752255 score/binding/批准/binary/完整SASS/raw和旧source、canonical source及登记PLAN；不编译/GPU。根代理核新增gap路径及gap0绕过、prefill/refill三请求与槽序、actual gap计量、唯一prefill Point、无spill和SMEM/驻留后批准并冻结。沿用现有构建/执行/capture接口；CPU-only构建10分钟、分配10分钟且GPU≤1小时，最多两次定向修复，无自动重采或扩条件。
''')
    save_json(root/'preparation-repair.json',dict(repairs=0,reason='初始私有实现固定mask7/D0/step1；尚未构建或执行',
        compilation=False,gpu=False,automatic_retry=False))
    names=[str(p.relative_to(root)) for directory in ('source','source-before','reference')
        for p in (root/directory).rglob('*') if p.is_file()]
    names+=['plan.json','commands.json','correctness_commands.json','worklist.txt','README.md','preparation-repair.json']
    save_json(root/'build-input-binding.json',dict(schema='gh200-q02-prefill-build-input/1',
        files={name:digest(root/name) for name in sorted(names)}))
    print('PREPARED: 8 prefill-gap development conditions; 3 verify +3 formal processes; no compilation/GPU',flush=True)


def prepare_source_step(root,reference):
    prior=read_json(reference/'score.json')
    require(prior['observation_qualified'] is True,'752176原资格未通过')
    approved_target(reference)
    controls={str(c['lda']):{metric:window['median'] for metric,window in c['windows'].items()}
        for c in prior['conditions'] if c['active_groups']==4 and c['trace']==0}
    prefixes={str(c['lda']):c['prefill']['cycles']['median'] for c in prior['conditions']
        if c['active_groups']==4 and c['trace']==-1}
    response=next(row for row in prior['responses'] if row['active_groups']==4)['prefill_A16']['cycles']['median']
    require(controls=={'4096':{'cycles_median':21788,'ns_median':11008},
        '4104':{'cycles_median':22301,'ns_median':11264}} and prefixes=={'4096':1382,'4104':1895} and response==513,
        '752176 Q4完整窗/prefill/同seed响应控制不符')
    require(digest(reference/'build/probe')==prior['probe_sha256'],'752176 binary不符')
    root.mkdir(parents=True,exist_ok=False)
    for directory in ('source','source-before','reference'):(root/directory).mkdir()
    shutil.copy2(HERE/'q02_prefill_observer.py',root/'source/q02_prefill_observer.py')
    shutil.copy2(HERE/'probes/tma_gemm_sources.cu',root/'source/tma_gemm_sources.cu')
    for name in ('q02_tma_stride.py','l2_saturation.py','r00_common.hpp','cutlass-3.9.2-include.tar.gz'):
        shutil.copy2(reference/'source'/name,root/'source'/name)
    shutil.copy2(reference/'source/tma_gemm_sources.cu',root/'source-before/tma_gemm_sources.cu')
    previous=HERE.parents[2]/'results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261018-Q02-operand-split-001/source/tma_gemm_sources.cu'
    shutil.copy2(previous,root/'reference/before-source-step.cu')
    shutil.copy2(reference/'worklist.txt',root/'worklist.txt')
    shutil.copy2(HERE.parents[2]/'Docs/ModelEvaluation/gemm/PLAN.md',root/'reference/campaign-PLAN.md')
    for name in ('score.json','binding.json','plan.json','commands.json','correctness_commands.json','approved-target.json'):
        shutil.copy2(reference/name,root/'reference'/name)
    for name in ('probe','target-review.json','kernel-code.json'):
        shutil.copy2(reference/'build'/name,root/'reference'/name)
    shutil.copytree(reference/'build/after/disassemble',root/'reference/disassemble')
    for filename,verify,split in (('commands.json',0,'formal'),('correctness_commands.json',1,'correctness')):
        tasks=[]
        for old in read_json(reference/filename):
            if old['active_groups']!=4:continue
            task=copy.deepcopy(old);seed=task['seed']
            shutil.copy2(reference/'runs'/split/task['id']/'stdout',root/f'reference/{split}-seed{seed}.ndjson')
            pairs=[(step,stride) for step in (1,2) for stride in STRIDES]
            if seed!=3:pairs.reverse()
            task.update(id=f'q4-source-step-seed{seed}',role='development_source_state',pair_indices=list(range(8)),
                paired_strides=[stride for _,stride in pairs],paired_source_steps=[step for step,_ in pairs],
                paired_traces=[0,-1] if seed==3 else [-1,0])
            argv=task['argv']
            for flag,value in (('--verify',str(verify)),('--paired-traces',','.join(map(str,task['paired_traces']))),
                    ('--paired-strides',','.join(f'{a}:{b}' for a,b in task['paired_strides'])),
                    ('--source-k-step','1'),('--paired-source-steps',','.join(map(str,task['paired_source_steps']))),
                    ('--request-mask','7'),('--wait-delay-cycles','0')):
                if flag in argv:argv[argv.index(flag)+1]=value
                else:argv+=[flag,value]
            if verify:task['expected_checked']=524288
            tasks.append(task)
        require(len(tasks)==3,'每类必须三个八臂配对进程');save_json(root/filename,tasks)
    plan=dict(schema=SCHEMA,qualification='development_source_state',intervention='source_k_step',
        independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        gpu_uuid=read_json(reference/'plan.json')['gpu_uuid'],required_smid=124,seeds=[3,22,41],active_groups=[4],
        strides=STRIDES,source_steps=[1,2],traces=[0,-1],m=2304,n=2304,k=4096,ctas=1,groups=32,ring=64,
        physical_slots=4,alloc_lda=4104,alloc_ldb=2360,request_mask=7,request_bytes=[16384,8192,8192],swizzle='SW128',tile=[7,6],
        source_k_formula='(g*source_k_step+source_phase)%(k/64)',warmup_source_k_step=1,
        warmup_groups=64,warmup_active_groups=4,warmup_calls=10,warmup_trace=-1,warmup_wait_delay_cycles=0,
        warmup_request_mask=7,release_delay_ns=0,wait_hint_ns=10000000,development_conditions=8,
        correctness_processes=3,formal_processes=3,numeric_arm_checks=24,formal_arm_observations=24,numeric_checked_values=12582912,
        relative_tolerance=.05,window_relative_spread=.05,prefill_cycle_relative_spread=.05,perturbation_tolerance=.05,
        ratio_absolute_tolerance=.05,control_windows=controls,control_prefill_cycles=prefixes,control_prefill_effect_cycles=response,
        prefill_ns_report_only=True,full_window_stability_unit='cycles_and_ns',source_rounds=[0,1],reference_job=752176,
        reference_score_sha256=digest(reference/'score.json'),reference_binding_sha256=digest(reference/'binding.json'),
        before_source_sha256=digest(root/'source-before/tma_gemm_sources.cu'),
        before_step_source_sha256=digest(root/'reference/before-source-step.cu'),original_probe_sha256=prior['probe_sha256'],
        allocation_budget_seconds=600,maximum_gpu_seconds=3600,maximum_repairs=2,automatic_retries=False,
        prefill_location='after initial active groups issue; before first full wait',
        candidates=[dict(name='adjacent_source_boundary',ratio=0),dict(name='adjacency_independent_issue_pressure',ratio=1)],
        ratio_definition='median_seed(prefill_A16_step2-prefill_A0_step2)/median_seed(prefill_A16_step1-prefill_A0_step1), cycles',
        classification='abs(R)<=.05: retain adjacent address state; abs(R-1)<=.05: stop adjacency-only explanation; both miss: stop both; qualification failure: information insufficient',
        signed_or_zero_delta_allowed=True,no_bare_price=True,no_refill_extrapolation=True,frozen_at_ns=time.time_ns())
    save_json(root/'plan.json',plan)
    (root/'README.md').write_text('''# Q02：相邻Ktile源地址状态判别

Q4/G32固定，step1/2×A lda4096/4104×trace0/−1共八开发条件，B ldb2304。正式K坐标为(g×step+phase)%64，数值校验使用相同坐标；单请求覆盖相同，step2消除目标组间A共享边界，同时改变唯一源覆盖、K序列及可能的域映射。真实三请求、mask7、SW128、四物理槽及代次、barrier、发布/等待顺序、allocA4104/allocB2360、K4096/ring64、tile[7,6]和SM124固定。每seed同进程八臂，trace外层，step/stride按同长度列表zip；seed3正序22/41整体逆序。每臂重填同一分配，暖机总step1、mask7、Q4/G64、trace−1、D0十次，正式恢复目标step/trace。先3verify后3plain进程，共24数值臂/24正式臂、12582912值，仅保留原首次prefill单点，循环内无新打点。

step1 plain完整窗须复现752176的A0/A16=21788/22301cycle、11008/11264ns；prefill须复现1382/1895cycle，同seed A16增量中位513cycle，各5%。八条件完整cycle/ns及四个sparse prefill cycle的三进程范围≤中位5%；同seed同step/stride sparse/plain完整窗扰动≤5%；数值/身份/驻留全部通过才判别。verify只负责完整数值与身份，不用于时间判别，prefill ns只报告。

先同seed相减再取中位，R=step2 prefill A16增量中位/step1对应中位，cycle为主。abs(R)≤.05支持保留邻接地址状态；abs(R−1)≤.05停止邻接边界的单独解释，继续约束发出反馈；两者均失败则停止两形式。资格失败或零分母为信息不足，零/负增量保留。该结果不指定物理容量或片段价格，不外推refill，不替换默认组合，独立C2/C3及完整GEMM验收均false。

```sh
python -B cpu-check.py
python -B source/q02_prefill_observer.py build-only --root "$PWD"
python -B source/q02_prefill_observer.py execute --root "$PWD"
python -B source/q02_prefill_observer.py score --root "$PWD" --output /tmp/q02-source-step-score.json
```

prepare保存752176 score/binding/批准/binary/完整SASS/raw及旧源码、步长修改前源码和当前登记PLAN，不编译/GPU。沿用prefill构建/capture/批准流程；根代理核新增尾部参数与旧offset、加载及verify的K坐标、warmup step1恢复、所有prefill/refill的三请求/字节/顺序、唯一prefill Point、无spill与SMEM/驻留后批准并冻结执行身份。CPU-only构建10分钟、采样分配10分钟且GPU≤1小时，最多两次定向修复，无自动重采或扩条件。
''')
    names=[str(p.relative_to(root)) for directory in ('source','source-before','reference')
        for p in (root/directory).rglob('*') if p.is_file()]
    names+=['plan.json','commands.json','correctness_commands.json','worklist.txt','README.md']
    save_json(root/'build-input-binding.json',dict(schema='gh200-q02-prefill-build-input/1',
        files={name:digest(root/name) for name in sorted(names)}))
    print('PREPARED: 8 source-step development conditions; 3 verify +3 formal processes; no compilation/GPU',flush=True)


def prepare(root,reference):
    root.mkdir(parents=True,exist_ok=False)
    (root/'source').mkdir();(root/'source-before').mkdir();(root/'reference').mkdir()
    for name in ('q02_prefill_observer.py','q02_tma_stride.py','l2_saturation.py'):
        shutil.copy2(HERE/name,root/'source'/name)
    shutil.copy2(HERE/'probes/tma_gemm_sources.cu',root/'source/tma_gemm_sources.cu')
    for name in ('r00_common.hpp','cutlass-3.9.2-include.tar.gz'):
        shutil.copy2(reference/'source'/name,root/'source'/name)
    shutil.copy2(reference/'source/tma_gemm_sources.cu',root/'source-before/tma_gemm_sources.cu')
    shutil.copy2(reference/'worklist.txt',root/'worklist.txt')
    for name in ('score.json','binding.json'):
        shutil.copy2(reference/name,root/'reference'/name)
    shutil.copy2(reference/'build/target-review.json',root/'reference/target-review.json')
    for seed in (3,22,41):
        shutil.copy2(reference/f'runs/formal/q1-a-b-strides-seed{seed}/stdout',root/f'reference/formal-seed{seed}.ndjson')
    prior=read_json(reference/'score.json')
    controls={str(c['lda']):{metric:w['median'] for metric,w in c['windows'].items()}
        for c in prior['conditions'] if c['ldb']==2304}
    require(controls=={'4096':{'cycles_median':30845,'ns_median':15552},
        '4104':{'cycles_median':37645,'ns_median':19040}},'752146完整窗口控制不符')
    by_prior={(r['seed'],r['lda'],r['ldb']):r for r in prior['formal']}
    effects={metric:statistics.median(by_prior[(seed,4104,2304)]['metrics'][metric]-
        by_prior[(seed,4096,2304)]['metrics'][metric] for seed in (3,22,41))
        for metric in ('cycles_median','ns_median')}
    require(effects=={'cycles_median':6802,'ns_median':3488},'752146同seed配对响应不符')
    for filename,verify in (('commands.json',0),('correctness_commands.json',1)):
        tasks=[]
        for q in (1,4):
            for old in read_json(reference/'commands.json'):
                task=copy.deepcopy(old);seed=task['seed']
                task['id']=f'q{q}-prefill-paired-seed{seed}';task['active_groups']=q
                task['paired_strides']=STRIDES if seed==3 else list(reversed(STRIDES))
                task['paired_traces']=[0,-1] if seed==3 else [-1,0]
                task['role']='development_localization'
                argv=task['argv']
                argv[argv.index('--verify')+1]=str(verify)
                argv[argv.index('--measure-active')+1]=str(q)
                argv[argv.index('--paired-strides')+1]=','.join(f'{a}:{b}' for a,b in task['paired_strides'])
                argv+=['--paired-traces',','.join(map(str,task['paired_traces']))]
                if verify:task['expected_checked']=524288
                tasks.append(task)
        save_json(root/filename,tasks)
    plan=dict(schema=SCHEMA,qualification='development_localization',independent_C2=False,independent_C3=False,
        full_gemm_C2=False,full_gemm_C3=False,gpu_uuid=read_json(reference/'plan.json')['gpu_uuid'],
        required_smid=124,seeds=[3,22,41],active_groups=[1,4],strides=STRIDES,traces=[0,-1],
        m=2304,n=2304,k=4096,ctas=1,groups=32,ring=64,physical_slots=4,alloc_lda=4104,alloc_ldb=2360,
        warmup_groups=64,warmup_active_groups=4,warmup_calls=10,warmup_trace=-1,wait_hint_ns=10000000,
        development_conditions=8,correctness_processes=6,formal_processes=6,numeric_arm_checks=24,
        formal_arm_observations=24,numeric_checked_values=12582912,relative_tolerance=.05,
        window_relative_spread=.05,prefill_cycle_relative_spread=.05,perturbation_tolerance=.05,
        response_zero_band_fraction=.05,control_windows=controls,control_effects=effects,
        prefill_ns_report_only=True,full_window_stability_unit='cycles_and_ns',
        source_rounds=[0,1],reference_job=752146,reference_score_sha256=digest(reference/'score.json'),
        reference_binding_sha256=digest(reference/'binding.json'),before_source_sha256=digest(root/'source-before/tma_gemm_sources.cu'),
        original_probe_sha256=read_json(reference/'score.json')['probe_sha256'],
        allocation_budget_seconds=600,maximum_gpu_seconds=3600,maximum_repairs=2,automatic_retries=False,
        prefill_location='after initial active groups issue; before first full wait',
        classification='median inside zero band: no clear initial response; outside band and all same sign: signed initial response; mixed signs outside band: information insufficient',
        no_bare_price=True,no_refill_extrapolation=True,frozen_at_ns=time.time_ns())
    save_json(root/'plan.json',plan)
    (root/'README.md').write_text("""# Q02：首次预填的稀疏观测

Q1/Q4 × A lda4096/4104 × plain0/sparse−1，共8开发条件。每Q、seed同进程四臂，trace外层、stride内层；seed3两个列表正序，22/41都逆序。固定allocA4104/allocB2360、真实三请求/SW128、四物理槽、ring64、K4096、G32及SM124。每臂重新初始化同一分配，统一trace−1、Q4/G64暖机10次，正式再恢复目标Q和trace。先6verify后6plain进程，共48臂，完整检查12582912值。

sparse只在首次预填全部UTMA发出后采一次Point，保留到end后写首组issue_end；K循环没有新打点。raw的ctas_data[].prefill_end只出现在trace−1，group_events为空，plain不读暖机旧值。该点包括前缀与初始发出控制，只定位首次3/12请求。

Q1新plain的A0/A16 cycle/ns窗口须复现752146各5%；同seed配对增量6802cycle/3488ns也各5%。每shape同seed sparse/plain完整cycle/ns差≤5%；完整窗口cycle/ns及sparse prefill cycle的三进程极差≤中位5%。失败信息不足并停止，不减打点成本或重采。合格后按Q报告A16的同seed prefill增量，零带为A0 prefill cycle中位±5%；超带且三seed同向则报告正/负初始响应，带内报告无明显初始响应，带外混合方向信息不足。ns前缀只报告，不外推refill，不生成裸服务价格；独立C2/C3和完整GEMM验收均false。

```sh
python source/q02_prefill_observer.py build-only --root "$PWD"
python source/q02_prefill_observer.py execute --root "$PWD"
python source/q02_prefill_observer.py score --root "$PWD" --output /tmp/q02-prefill-score.json
```

build-only在ARM CUDA12.9.41只构建新binary，保存完整SASS及资源，不启动GPU。原before源码及hash保留；新binary不要求旧全字相同。根代理审核0/−1至首次预填末UTMA的路径一致、新戳一次且无提前写回、无spill及相同SMEM/驻留后写approved-target，再冻结最终执行绑定。执行只使用已批准binary。CPU构建与GPU采样各10分钟，GPU绝对上限1小时，最多两次定向实现修复。准备阶段未编译或GPU采样。
""")
    names=[str(p.relative_to(root)) for directory in ('source','source-before','reference') for p in (root/directory).iterdir() if p.is_file()]
    names+=['plan.json','commands.json','correctness_commands.json','worklist.txt','README.md']
    save_json(root/'build-input-binding.json',dict(schema='gh200-q02-prefill-build-input/1',
        files={name:digest(root/name) for name in sorted(names)}))
    print('PREPARED: 8 development conditions; 6 verify +6 formal processes; no compilation/GPU',flush=True)


def load_plan(root,execution=False):
    for name,checksum in read_json(root/('binding.json' if execution else 'build-input-binding.json'))['files'].items():
        require(digest(root/name)==checksum,'冻结身份变更：'+name)
    plan=read_json(root/'plan.json');require(plan['schema']==SCHEMA,'执行范围不符')
    return plan


def kernel_records(sass):
    records = {}
    for section in re.split(r'Function\s*:\s*',sass)[1:]:
        symbol = section.splitlines()[0].strip()
        if not symbol.startswith('_Z12tensor_probe'):continue
        code = []
        for line in section.splitlines()[1:]:
            match = re.match(r'\s*/\*([0-9a-f]+)\*/\s*(.*?)\s*/\* 0x([0-9a-f]+) \*/',line)
            if match:code.append(dict(pc=int(match[1],16),op=match[2].strip(),words=[int(match[3],16)]))
            else:
                word = re.search(r'/\* 0x([0-9a-f]+) \*/',line)
                if word and code:code[-1]['words'].append(int(word[1],16))
        require(code and all(len(row['words'])==2 for row in code), '完整两机器字缺失')
        require(symbol not in records, 'kernel入口重复')
        records[symbol] = code
    return records



def validate_process(root,folder,task,uuid):
    receipt=stride.successful_receipt(folder)
    command=receipt['argv'][:];probe=Path(command[0])
    require(probe.name=='probe' and probe.parent.name=='build','binary路径不符')
    require(command[command.index('--worklist')+1]==str(probe.parent.parent/'worklist.txt'),'工作列表路径不符')
    command[0]=task['argv'][0];command[command.index('--worklist')+1]='worklist.txt'
    require(command==task['argv'],'执行参数与冻结命令不符')
    raw=[json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.strip()]
    require(len(raw)==len(task['paired_strides'])*len(task['paired_traces']),'四臂缺失')
    arguments=dict(zip(task['argv'][1::2],task['argv'][2::2]))
    fields={'ctas':'--ctas','active_groups':'--measure-active','groups':'--measure-groups','ring':'--ring',
        'verify':'--verify','seed':'--seed','warmup':'--warmup',
        'source_phase_stride':'--source-phase-stride','source_tile_shift':'--source-tile-shift',
        'work_index':'--work-index','warmup_work_offset':'--warmup-work-offset',
        'release_delay_ns':'--release-delay-ns','pollute_kib':'--pollute-kib','launch_skew_ns':'--launch-skew-ns',
        'warmup_skew_ns':'--warmup-skew-ns','alloc_lda':'--alloc-lda','alloc_ldb':'--alloc-ldb'}
    require((root/'worklist.txt').read_text().split()==['2304','2304','4096','1','0','0','7','6'],'工作列表不符')
    step_mode='paired_source_steps' in task
    gap_mode='paired_prefill_gaps' in task
    checked=[];descriptors={}
    for index,row in enumerate(raw):
        require(row['schema']=='cfg-b-gemm-source-probe/1' and row['gpu_uuid']==uuid,'schema/卡身份不符')
        require(all(row[name]==int(arguments[flag]) for name,flag in fields.items()),'逐臂参数不符')
        require(row['warmup_trace']==-1 and row['warmup_active_groups']==4 and row['warmup_groups']==64 and
            row['warmup_source_phase_stride']==0 and row['record_warmup']==0 and
            not row['warmup_ctas'] and not row['group_events'],'暖机或trace范围不符')
        require(row['paired_strides']==task['paired_strides'] and row['pair_index']==index and
            row['trace_index']==index//len(task['paired_strides']) and
            row['stride_index']==index%len(task['paired_strides']) and
            row['paired_traces']==task['paired_traces'] and row['trace']==task['paired_traces'][row['trace_index']] and
            [row['lda'],row['ldb']]==task['paired_strides'][row['stride_index']],'逐臂stride或次序不符')
        if step_mode:
            require(row['paired_source_steps']==task['paired_source_steps'] and
                row['source_k_step']==task['paired_source_steps'][row['stride_index']] and
                row['warmup_source_k_step']==1,'源K步长/zip或暖机步长不符')
            require(row['request_mask']==row['warmup_request_mask']==7 and
                row['transaction_bytes']==row['warmup_transaction_bytes']==32768 and
                row['wait_delay_cycles']==row['warmup_wait_delay_cycles']==0,'三请求/事务字节或D0不符')
        if gap_mode:
            require(row['paired_prefill_gaps']==task['paired_prefill_gaps'] and
                row['prefill_gap_cycles']==task['paired_prefill_gaps'][row['stride_index']] and
                row['warmup_prefill_gap_cycles']==0,'预填gap/zip或暖机gap不符')
            require(row['source_k_step']==row['warmup_source_k_step']==1 and
                row['request_mask']==row['warmup_request_mask']==7 and
                row['transaction_bytes']==row['warmup_transaction_bytes']==32768 and
                row['wait_delay_cycles']==row['warmup_wait_delay_cycles']==0,'三请求/D0/step1不符')
            require(row['cta_record_bytes']==64 and row['group_record_bytes']==80,'追加gap计量后的记录布局不符')
        if step_mode or gap_mode:
            descriptor_key=(row['lda'],row['ldb']);pair=(row['descriptor_a'],row['descriptor_b'])
            require(all(re.fullmatch('[0-9a-f]{256}',value) for value in pair) and
                descriptors.setdefault(descriptor_key,pair)==pair,'同stride的step/trace descriptor变化')
        require(all(row[name]==raw[0][name] for name in stride.ADDRESSES),'同进程地址变化')
        require(row['source_a_address']>0 and row['source_b_address']>0 and
            row['source_a_address']%128==row['source_b_address']%128==0,'源地址/对齐不符')
        require(row['m']==row['n']==2304 and row['allocated_a_elements']==9455616 and
            row['allocated_b_elements']==9666560,'形状或固定分配不符')
        require(row['errors']==row['pollution_errors']==row['pollution_checked']==0,'加载/污染检查错误')
        require(row['checked']==(524288 if row['verify'] else 0),'完整加载检查量不符')
        require(row['observed_sms']==row['max_active_ctas_per_sm']==1 and row['smem']==132096 and
            row['local_bytes']==0,'驻留或资源不符')
        resource=read_json(root/'build/target-review.json')['rows'][0]['resources']
        require(row['registers']==int(re.search(r'\bREG:(\d+)\b',resource)[1]),'运行寄存器与构建不符')
        require(len(row['ctas_data'])==1,'CTA数量不符');cta=row['ctas_data'][0]
        require(cta['cta']==0 and cta['tile']==[7,6] and cta['smid']==124,'CTA/坐标/SM124不符')
        for unit in (0,1):
            require(type(cta['begin'][unit]) is int and type(cta['end'][unit]) is int and
                0<cta['begin'][unit]<cta['end'][unit],'完成戳无效')
        cycles=cta['end'][0]-cta['begin'][0];ns=cta['end'][1]-cta['begin'][1]
        if gap_mode:
            gap_sum=cta['gap_sum_cycles'];gap=row['prefill_gap_cycles']
            require(type(gap_sum) is int and 0<=gap_sum<cycles and
                (gap_sum==0 if gap==0 else gap==8192 and gap_sum>=3*gap),'实际gap总和无效')
        prefill=None
        if row['trace']==-1:
            point=cta['prefill_end']
            require(len(point)==2 and all(type(point[u]) is int and cta['begin'][u]<point[u]<cta['end'][u] for u in (0,1)),'预填结束戳无效')
            prefill=dict(cycles=point[0]-cta['begin'][0],ns=point[1]-cta['begin'][1])
            if gap_mode:require(gap_sum<prefill['cycles'],'实际gap总和超出prefill窗口')
        else:require('prefill_end' not in cta,'plain错误读取了暖机预填戳')
        checked.append(dict(seed=row['seed'],active_groups=row['active_groups'],trace=row['trace'],lda=row['lda'],ldb=row['ldb'],pair_index=index,
            checked_values=row['checked'],numerical_qualified=bool(row['verify']),
            stdout_sha256=receipt['stdout_sha256'],source=str(folder.relative_to(root)),device_uuid=uuid,prefill=prefill,
            metrics=dict(cycles_median=cycles,ns_median=ns,ns_envelope=ns)))
        if step_mode:checked[-1]['source_k_step']=row['source_k_step']
        if gap_mode:checked[-1].update(prefill_gap_cycles=row['prefill_gap_cycles'],gap_sum_cycles=gap_sum)
    return checked
def review_build(root):
    folder=root/'build'
    stride.successful_receipt(folder/'version')
    version=(folder/'version/stdout').read_text()
    require('V12.9.41' in version and 'compiler.35813241_0' in version,'需要冻结CUDA12.9.41')
    for step in ('compile','disassemble','resources'):stride.successful_receipt(folder/'after'/step)
    records=kernel_records((folder/'after/disassemble/stdout').read_text())
    require(len(records)==1 and records==read_json(folder/'kernel-code.json'),'tensor_probe唯一完整代码不符')
    symbol,code=next(iter(records.items()))
    resources=(folder/'after/resources/stdout').read_text()
    match=re.search(r'Function '+re.escape(symbol)+r':\s*\n\s*([^\n]+)',resources)
    require(match is not None,'tensor_probe资源缺失');resource=match[1].strip()
    log=(folder/'after/compile/stdout').read_text()+(folder/'after/compile/stderr').read_text()
    count=sum('UTMALDG' in row['op'] for row in code)
    zero=bool(re.search(r'\bSTACK:0\b',resource) and re.search(r'\bLOCAL:0\b',resource))
    warning='C7515' in log
    row=dict(label='prefill_observer',symbol=symbol,instruction_count=len(code),
        instruction_words_sha256=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest(),
        utma_count=count,resources=resource,zero_local_stack=zero,compile_warning_c7515=warning)
    report=dict(status='pending_root_prefix_sass_review',approved=False,basic_gate_passed=count==18 and zero and not warning,
        compiler_version=version,rows=[row],probe_sha256=digest(folder/'probe'),
        kernel_code_sha256=digest(folder/'kernel-code.json'),plan_sha256=digest(root/'plan.json'),
        before_source_sha256=digest(root/'source-before/tma_gemm_sources.cu'),
        original_probe_sha256=read_json(root/'plan.json')['original_probe_sha256'])
    if read_json(root/'plan.json').get('intervention')=='source_k_step':
        report.update(status='pending_root_source_step_sass_review',basic_gate_passed=count>0 and zero and not warning,
            static_utma_count_unreviewed=True,source_steps=[1,2],requests_per_group=3)
        row['label']='source_step'
    if read_json(root/'plan.json').get('intervention')=='prefill_gap_cycles':
        report.update(status='pending_root_prefill_gap_sass_review',basic_gate_passed=count>0 and zero and not warning,
            static_utma_count_unreviewed=True,prefill_gaps=[0,8192],requests_per_group=3)
        row['label']='prefill_gap'
    return report


def build_only(root):
    load_plan(root);start=time.monotonic();folder=root/'build';folder.mkdir(exist_ok=False)
    compiler=shutil.which('nvcc');require(compiler is not None,'nvcc缺失')
    command_record(folder/'version',[compiler,'--version'],stride.remaining(start,20))
    version=(folder/'version/stdout').read_text()
    require('V12.9.41' in version and 'compiler.35813241_0' in version,'需要冻结CUDA12.9.41')
    flags=['-std=c++17','-O3','-DNDEBUG','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v']
    disassembler=Path(compiler).resolve().parent/'cuobjdump'
    with tempfile.TemporaryDirectory(prefix='q02-prefill-cutlass-') as directory:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:
            archive.extractall(directory,filter='data')
        command_record(folder/'after/compile',[compiler,*flags,'-I'+directory+'/include','-I'+str(root/'source'),
            root/'source/tma_gemm_sources.cu','-o',folder/'probe'],stride.remaining(start,180))
        command_record(folder/'after/disassemble',[disassembler,'--dump-sass',folder/'probe'],stride.remaining(start,40))
        command_record(folder/'after/resources',[disassembler,'--dump-resource-usage',folder/'probe'],stride.remaining(start,30))
    save_json(folder/'kernel-code.json',kernel_records((folder/'after/disassemble/stdout').read_text()))
    report=review_build(root);save_json(folder/'target-review.json',report)
    require(report['basic_gate_passed'],'新kernel的18UTMA/无spill资源门槛未通过')
    print('BUILT ONLY: root prefix/SASS/resource approval pending; no GPU',flush=True)


def summarize_source_step(plan,numeric,formal):
    key=lambda row:(row['seed'],row['source_k_step'],row['lda'],row['ldb'],row['trace'])
    by_numeric={key(row):row for row in numeric};by_formal={key(row):row for row in formal}
    expected={(seed,step,lda,ldb,trace) for seed in plan['seeds'] for step in plan['source_steps']
        for lda,ldb in plan['strides'] for trace in plan['traces']}
    require(len(numeric)==len(formal)==24 and by_numeric.keys()==by_formal.keys()==expected,
        '数值/正式24臂完整step/stride/trace键不对应')
    require(all(row['numerical_qualified'] and row['checked_values']==524288 and row['active_groups']==4 for row in numeric) and
        all(row['active_groups']==4 for row in formal) and
        sum(row['checked_values'] for row in numeric)==plan['numeric_checked_values'],'数值/正式Q4资格不符')
    conditions=[];controls=[];prefix_controls=[];perturbations=[];responses=[]
    for step in plan['source_steps']:
        for lda,ldb in plan['strides']:
            for trace in plan['traces']:
                rows=[by_formal[(seed,step,lda,ldb,trace)] for seed in plan['seeds']];windows={}
                for metric in ('cycles_median','ns_median'):
                    values=[row['metrics'][metric] for row in rows];median=statistics.median(values)
                    require(median>0,'完整窗口须为正数');spread=(max(values)-min(values))/median
                    windows[metric]=dict(values=values,median=median,relative_spread=spread,
                        passed=spread<=plan['window_relative_spread'])
                    if step==1 and trace==0:
                        old=plan['control_windows'][str(lda)][metric];error=abs(median-old)/old
                        controls.append(dict(source_k_step=step,lda=lda,metric=metric,expected=old,observed=median,
                            relative_error=error,passed=error<=plan['relative_tolerance']))
                prefix=None
                if trace==-1:
                    prefix={}
                    for unit in ('cycles','ns'):
                        values=[row['prefill'][unit] for row in rows];median=statistics.median(values)
                        require(median>0,'prefill窗口须为正数');spread=(max(values)-min(values))/median
                        prefix[unit]=dict(values=values,median=median,relative_spread=spread,
                            passed=spread<=plan['prefill_cycle_relative_spread'] if unit=='cycles' else None)
                        if step==1 and unit=='cycles':
                            old=plan['control_prefill_cycles'][str(lda)];error=abs(median-old)/old
                            prefix_controls.append(dict(source_k_step=step,lda=lda,expected=old,observed=median,
                                relative_error=error,passed=error<=plan['relative_tolerance']))
                passed=all(window['passed'] for window in windows.values()) and (trace==0 or prefix['cycles']['passed'])
                conditions.append(dict(active_groups=4,source_k_step=step,lda=lda,ldb=ldb,trace=trace,
                    windows=windows,prefill=prefix,passed=passed))
            for seed in plan['seeds']:
                plain=by_formal[(seed,step,lda,ldb,0)];sparse=by_formal[(seed,step,lda,ldb,-1)]
                for metric in ('cycles_median','ns_median'):
                    p=plain['metrics'][metric];s=sparse['metrics'][metric];error=abs(s-p)/p
                    perturbations.append(dict(seed=seed,source_k_step=step,lda=lda,metric=metric,plain=p,sparse=s,
                        relative_difference=error,passed=error<=plan['perturbation_tolerance']))
        prefix_response={};full_response={}
        for unit,metric in (('cycles','cycles_median'),('ns','ns_median')):
            values=[by_formal[(seed,step,4104,2304,-1)]['prefill'][unit]-
                by_formal[(seed,step,4096,2304,-1)]['prefill'][unit] for seed in plan['seeds']]
            prefix_response[unit]=dict(values=values,median=statistics.median(values))
            full_response[unit]={}
            for trace in plan['traces']:
                values=[by_formal[(seed,step,4104,2304,trace)]['metrics'][metric]-
                    by_formal[(seed,step,4096,2304,trace)]['metrics'][metric] for seed in plan['seeds']]
                full_response[unit][str(trace)]=dict(values=values,median=statistics.median(values))
        responses.append(dict(source_k_step=step,prefill_A16=prefix_response,full_window_A16=full_response))
    base=responses[0]['prefill_A16']['cycles']['median'];changed=responses[1]['prefill_A16']['cycles']['median']
    old=plan['control_prefill_effect_cycles'];error=abs(base-old)/old
    effect=dict(values=responses[0]['prefill_A16']['cycles']['values'],median=base,expected=old,
        relative_error=error,passed=error<=plan['relative_tolerance'])
    control_passed=all(row['passed'] for row in controls+prefix_controls+[effect])
    stable=all(row['passed'] for row in conditions);perturbation_passed=all(row['passed'] for row in perturbations)
    qualified=control_passed and stable and perturbation_passed
    ratio=changed/base if base else None
    classification='information_insufficient'
    if ratio is not None:
        classification='adjacent_source_boundary' if abs(ratio)<=plan['ratio_absolute_tolerance'] else \
            'adjacency_independent_issue_pressure' if abs(ratio-1)<=plan['ratio_absolute_tolerance'] else 'both_forms_failed'
    decision='information_insufficient'
    if qualified:
        decision={'adjacent_source_boundary':'supports_adjacent_source_boundary',
            'adjacency_independent_issue_pressure':'stops_adjacency_only_explanation',
            'both_forms_failed':'stop_both_source_step_forms'}.get(classification,'information_insufficient')
    return dict(schema='gh200-q02-source-step-score/1',qualification=plan['qualification'],decision=decision,
        independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),
        numeric_checked_values=sum(row['checked_values'] for row in numeric),numeric=numeric,formal=formal,
        conditions=conditions,control=dict(windows=controls,prefill_cycles=prefix_controls,prefill_effect_cycles=effect,passed=control_passed),
        perturbations=perturbations,perturbation_passed=perturbation_passed,stable_windows_and_prefill=stable,
        observation_qualified=qualified,responses=responses,ratio_cycles=ratio,classification=classification,
        prefill_ns_report_only=True,signed_or_zero_delta_allowed=True,no_bare_price=True,no_refill_extrapolation=True)


def summarize_prefill_gap(plan,numeric,formal):
    key=lambda row:(row['seed'],row['prefill_gap_cycles'],row['lda'],row['ldb'],row['trace'])
    by_numeric={key(row):row for row in numeric};by_formal={key(row):row for row in formal}
    expected={(seed,gap,lda,ldb,trace) for seed in plan['seeds'] for gap in plan['prefill_gaps']
        for lda,ldb in plan['strides'] for trace in plan['traces']}
    require(len(numeric)==len(formal)==24 and by_numeric.keys()==by_formal.keys()==expected,
        '数值/正式24臂完整gap/stride/trace键不对应')
    require(all(row['numerical_qualified'] and row['checked_values']==524288 and row['active_groups']==4 for row in numeric) and
        all(row['active_groups']==4 for row in formal) and
        sum(row['checked_values'] for row in numeric)==plan['numeric_checked_values'],'数值/正式Q4资格不符')
    conditions=[];controls=[];prefix_controls=[];perturbations=[];responses=[]
    for gap in plan['prefill_gaps']:
        for lda,ldb in plan['strides']:
            for trace in plan['traces']:
                rows=[by_formal[(seed,gap,lda,ldb,trace)] for seed in plan['seeds']];windows={}
                for metric in ('cycles_median','ns_median'):
                    values=[row['metrics'][metric] for row in rows];median=statistics.median(values)
                    require(median>0,'完整窗口须为正数');spread=(max(values)-min(values))/median
                    windows[metric]=dict(values=values,median=median,relative_spread=spread,
                        passed=spread<=plan['window_relative_spread'])
                    if gap==0 and trace==0:
                        old=plan['control_windows'][str(lda)][metric];error=abs(median-old)/old
                        controls.append(dict(prefill_gap_cycles=gap,lda=lda,metric=metric,expected=old,observed=median,
                            relative_error=error,passed=error<=plan['relative_tolerance']))
                prefix=None;net_prefix=None
                if trace==-1:
                    prefix={}
                    for unit in ('cycles','ns'):
                        values=[row['prefill'][unit] for row in rows];median=statistics.median(values)
                        require(median>0,'prefill窗口须为正数');spread=(max(values)-min(values))/median
                        prefix[unit]=dict(values=values,median=median,relative_spread=spread,
                            passed=spread<=plan['prefill_cycle_relative_spread'] if unit=='cycles' else None)
                        if gap==0 and unit=='cycles':
                            old=plan['control_prefill_cycles'][str(lda)];error=abs(median-old)/old
                            prefix_controls.append(dict(prefill_gap_cycles=gap,lda=lda,expected=old,observed=median,
                                relative_error=error,passed=error<=plan['relative_tolerance']))
                    values=[row['prefill']['cycles']-row['gap_sum_cycles'] for row in rows]
                    median=statistics.median(values);require(median>0,'净prefill窗口须为正数')
                    spread=(max(values)-min(values))/median
                    net_prefix=dict(values=values,median=median,relative_spread=spread,
                        passed=spread<=plan['net_prefill_cycle_relative_spread'])
                passed=all(window['passed'] for window in windows.values()) and (trace==0 or prefix['cycles']['passed'] and net_prefix['passed'])
                conditions.append(dict(active_groups=4,prefill_gap_cycles=gap,lda=lda,ldb=ldb,trace=trace,
                    windows=windows,prefill=prefix,net_prefill_cycles=net_prefix,
                    gap_sum_cycles=dict(values=[row['gap_sum_cycles'] for row in rows],
                        median=statistics.median(row['gap_sum_cycles'] for row in rows)),passed=passed))
            for seed in plan['seeds']:
                plain=by_formal[(seed,gap,lda,ldb,0)];sparse=by_formal[(seed,gap,lda,ldb,-1)]
                for metric in ('cycles_median','ns_median'):
                    p=plain['metrics'][metric];s=sparse['metrics'][metric];error=abs(s-p)/p
                    perturbations.append(dict(seed=seed,prefill_gap_cycles=gap,lda=lda,metric=metric,plain=p,sparse=s,
                        relative_difference=error,passed=error<=plan['perturbation_tolerance']))
        prefix_response={};full_response={}
        for unit,metric in (('cycles','cycles_median'),('ns','ns_median')):
            values=[by_formal[(seed,gap,4104,2304,-1)]['prefill'][unit]-
                by_formal[(seed,gap,4096,2304,-1)]['prefill'][unit] for seed in plan['seeds']]
            prefix_response[unit]=dict(values=values,median=statistics.median(values))
            full_response[unit]={}
            for trace in plan['traces']:
                values=[by_formal[(seed,gap,4104,2304,trace)]['metrics'][metric]-
                    by_formal[(seed,gap,4096,2304,trace)]['metrics'][metric] for seed in plan['seeds']]
                full_response[unit][str(trace)]=dict(values=values,median=statistics.median(values))
        net_values=[(by_formal[(seed,gap,4104,2304,-1)]['prefill']['cycles']-
                by_formal[(seed,gap,4104,2304,-1)]['gap_sum_cycles'])-
            (by_formal[(seed,gap,4096,2304,-1)]['prefill']['cycles']-
                by_formal[(seed,gap,4096,2304,-1)]['gap_sum_cycles']) for seed in plan['seeds']]
        responses.append(dict(prefill_gap_cycles=gap,prefill_A16=prefix_response,
            net_prefill_A16=dict(cycles=dict(values=net_values,median=statistics.median(net_values))),
            full_window_A16=full_response))
    base=responses[0]['net_prefill_A16']['cycles']['median'];changed=responses[1]['net_prefill_A16']['cycles']['median']
    old=plan['control_prefill_effect_cycles'];error=abs(base-old)/old
    effect=dict(values=responses[0]['net_prefill_A16']['cycles']['values'],median=base,expected=old,
        relative_error=error,passed=error<=plan['relative_tolerance'])
    control_passed=all(row['passed'] for row in controls+prefix_controls+[effect])
    stable=all(row['passed'] for row in conditions);perturbation_passed=all(row['passed'] for row in perturbations)
    qualified=control_passed and stable and perturbation_passed
    ratio=changed/base if base else None
    raw_base=responses[0]['prefill_A16']['cycles']['median']
    raw_ratio=responses[1]['prefill_A16']['cycles']['median']/raw_base if raw_base else None
    classification='information_insufficient'
    if ratio is not None:
        classification='spacing_hidden_response' if abs(ratio)<=plan['ratio_absolute_tolerance'] else \
            'spacing_retained_response' if abs(ratio-1)<=plan['ratio_absolute_tolerance'] else 'both_forms_failed'
    decision='information_insufficient'
    if qualified:
        decision={'spacing_hidden_response':'supports_response_hidden_by_prefill_gap',
            'spacing_retained_response':'response_retained_with_prefill_gap',
            'both_forms_failed':'stop_both_prefill_gap_forms'}.get(classification,'information_insufficient')
    return dict(schema='gh200-q02-prefill-gap-score/1',qualification=plan['qualification'],decision=decision,
        independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),
        numeric_checked_values=sum(row['checked_values'] for row in numeric),numeric=numeric,formal=formal,
        conditions=conditions,control=dict(windows=controls,prefill_cycles=prefix_controls,prefill_effect_cycles=effect,passed=control_passed),
        perturbations=perturbations,perturbation_passed=perturbation_passed,stable_windows_and_prefill=stable,
        observation_qualified=qualified,responses=responses,ratio_cycles=ratio,ratio_raw_cycles=raw_ratio,classification=classification,
        net_prefill_definition=plan['net_prefill_definition'],prefill_ns_report_only=True,signed_or_zero_delta_allowed=True,no_bare_price=True,no_refill_extrapolation=True)


def summarize_rows(plan,numeric,formal):
    if plan.get('intervention')=='prefill_gap_cycles':return summarize_prefill_gap(plan,numeric,formal)
    if plan.get('intervention')=='source_k_step':return summarize_source_step(plan,numeric,formal)
    key=lambda r:(r['seed'],r['active_groups'],r['lda'],r['trace'])
    by_numeric={key(r):r for r in numeric};by_formal={key(r):r for r in formal}
    expected={(seed,q,lda,trace) for seed in plan['seeds'] for q in plan['active_groups']
        for lda,_ in plan['strides'] for trace in plan['traces']}
    require(len(numeric)==len(formal)==24 and by_numeric.keys()==by_formal.keys()==expected,'数值/正式24臂不对应')
    require(all(r['numerical_qualified'] and r['checked_values']==524288 for r in numeric) and
        sum(r['checked_values'] for r in numeric)==plan['numeric_checked_values'],'数值资格不符')
    conditions=[];controls=[];perturbations=[];effects=[];responses=[]
    for q in plan['active_groups']:
        for lda,_ in plan['strides']:
            for trace in plan['traces']:
                rows=[by_formal[(seed,q,lda,trace)] for seed in plan['seeds']]
                windows={}
                for metric in ('cycles_median','ns_median'):
                    values=[r['metrics'][metric] for r in rows];median=statistics.median(values)
                    spread=(max(values)-min(values))/median
                    windows[metric]=dict(values=values,median=median,relative_spread=spread,
                        passed=spread<=plan['window_relative_spread'])
                    if q==1 and trace==0:
                        predicted=plan['control_windows'][str(lda)][metric];error=abs(median-predicted)/predicted
                        controls.append(dict(lda=lda,metric=metric,expected=predicted,observed=median,
                            relative_error=error,passed=error<=plan['relative_tolerance']))
                prefill=None
                if trace==-1:
                    prefill={}
                    for unit in ('cycles','ns'):
                        values=[r['prefill'][unit] for r in rows];median=statistics.median(values)
                        spread=(max(values)-min(values))/median
                        prefill[unit]=dict(values=values,median=median,relative_spread=spread,
                            passed=spread<=plan['prefill_cycle_relative_spread'] if unit=='cycles' else None)
                passed=all(v['passed'] for v in windows.values()) and (trace==0 or prefill['cycles']['passed'])
                conditions.append(dict(active_groups=q,lda=lda,ldb=2304,trace=trace,windows=windows,prefill=prefill,passed=passed))
            for seed in plan['seeds']:
                plain=by_formal[(seed,q,lda,0)];sparse=by_formal[(seed,q,lda,-1)]
                for metric in ('cycles_median','ns_median'):
                    p=plain['metrics'][metric];s=sparse['metrics'][metric];error=abs(s-p)/p
                    perturbations.append(dict(seed=seed,active_groups=q,lda=lda,metric=metric,plain=p,sparse=s,
                        relative_difference=error,passed=error<=plan['perturbation_tolerance']))
        response={};full={}
        for unit in ('cycles','ns'):
            values=[by_formal[(seed,q,4104,-1)]['prefill'][unit]-by_formal[(seed,q,4096,-1)]['prefill'][unit]
                for seed in plan['seeds']]
            response[unit]=dict(values=values,median=statistics.median(values))
            metric='cycles_median' if unit=='cycles' else 'ns_median'
            full[unit]={str(trace):dict(values=[by_formal[(seed,q,4104,trace)]['metrics'][metric]-
                by_formal[(seed,q,4096,trace)]['metrics'][metric] for seed in plan['seeds']]) for trace in plan['traces']}
            for trace in plan['traces']:
                full[unit][str(trace)]['median']=statistics.median(full[unit][str(trace)]['values'])
        base=statistics.median(by_formal[(seed,q,4096,-1)]['prefill']['cycles'] for seed in plan['seeds'])
        band=base*plan['response_zero_band_fraction'];values=response['cycles']['values'];median=response['cycles']['median']
        classification='within_zero_band' if abs(median)<=band else 'initial_positive_response' if all(v>0 for v in values) else \
            'initial_negative_response' if all(v<0 for v in values) else 'information_insufficient'
        responses.append(dict(active_groups=q,prefill_A16=response,full_window_A16=full,
            A0_prefill_cycle_median=base,zero_band_cycles=band,classification=classification))
    for metric in ('cycles_median','ns_median'):
        values=[by_formal[(seed,1,4104,0)]['metrics'][metric]-by_formal[(seed,1,4096,0)]['metrics'][metric] for seed in plan['seeds']]
        median=statistics.median(values);predicted=plan['control_effects'][metric];error=abs(median-predicted)/predicted
        effects.append(dict(metric=metric,values=values,median=median,expected=predicted,relative_error=error,
            passed=error<=plan['relative_tolerance']))
    control_passed=all(row['passed'] for row in controls+effects)
    stable=all(row['passed'] for row in conditions);perturbation_passed=all(row['passed'] for row in perturbations)
    qualified=control_passed and stable and perturbation_passed
    ambiguous=any(row['classification']=='information_insufficient' for row in responses)
    return dict(schema='gh200-q02-prefill-observer-score/1',qualification=plan['qualification'],
        decision='development_localization_complete' if qualified and not ambiguous else 'information_insufficient',
        independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),
        numeric_checked_values=sum(r['checked_values'] for r in numeric),numeric=numeric,formal=formal,
        conditions=conditions,control=dict(windows=controls,effects=effects,passed=control_passed),
        perturbations=perturbations,perturbation_passed=perturbation_passed,stable_windows_and_prefill=stable,
        observation_qualified=qualified,responses=responses,prefill_ns_report_only=True)


def approved_target(root):
    approved=read_json(root/'approved-target.json');review=review_build(root)
    require(review==read_json(root/'build/target-review.json') and review['basic_gate_passed'],'构建证据不符')
    require(approved['approved'] is True and approved['probe_sha256']==review['probe_sha256'] and
        approved['kernel_code_sha256']==review['kernel_code_sha256'] and
        approved['target_review_sha256']==digest(root/'build/target-review.json'),'目标binary未获核准')
    if read_json(root/'plan.json').get('intervention') in ('source_k_step','prefill_gap_cycles'):
        require(approved['plan_sha256']==digest(root/'plan.json'),'源K步长批准未绑定当前plan')


def summarize(root):
    plan=load_plan(root,True);approved_target(root)
    require(read_json(root/'allocation/identity.json')['uuid']==plan['gpu_uuid'],'分配卡身份不符')
    numeric=[];formal=[]
    for filename,split,rows in (('correctness_commands.json','correctness',numeric),('commands.json','formal',formal)):
        for task in read_json(root/filename):rows.extend(validate_process(root,root/'runs'/split/task['id'],task,plan['gpu_uuid']))
    score=summarize_rows(plan,numeric,formal)
    score.update(plan_sha256=digest(root/'plan.json'),probe_sha256=digest(root/'build/probe'))
    return score


def execute(root):
    plan=load_plan(root,True);approved_target(root);start=time.monotonic()
    environment=inspect_allocated_gpu(root/'allocation');require(environment['uuid']==plan['gpu_uuid'],'分配卡不是冻结参考卡')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        for filename,split in (('correctness_commands.json','correctness'),('commands.json','formal')):
            for task in read_json(root/filename):
                folder=root/'runs'/split/task['id'];command_record(folder,stride.probe_command(root,task),stride.remaining(start,60))
                save_json(folder/'validated.json',validate_process(root,folder,task,environment['uuid']))
                print(json.dumps(dict(split=split,process=task['id'],arms=len(task['paired_strides'])*len(task['paired_traces']))),flush=True)
        save_json(root/'score.json',summarize(root))
    print('CLOSED: 24 numeric +24 formal arms; first-prefill development localization',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('prepare','build-only','execute','score'):
        p=sub.add_parser(mode);p.add_argument('--root',type=Path,required=True)
        if mode=='prepare':
            p.add_argument('--reference',type=Path,required=True)
            modes=p.add_mutually_exclusive_group()
            modes.add_argument('--source-step',action='store_true',help='固定Q4的step1/2源地址判别，参考752176')
            modes.add_argument('--prefill-gap',action='store_true',help='固定Q4/step1的预填gap0/8192判别，参考752255')
        if mode=='score':p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.mode=='prepare':
        if args.prefill_gap:prepare_prefill_gap(root,args.reference.resolve())
        elif args.source_step:prepare_source_step(root,args.reference.resolve())
        else:prepare(root,args.reference.resolve())
    elif args.mode=='build-only':build_only(root)
    elif args.mode=='execute':execute(root)
    else:save_json(args.output,summarize(root))


if __name__=='__main__':main()

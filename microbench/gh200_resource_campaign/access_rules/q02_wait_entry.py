#!/usr/bin/env python3
"""Q1 等待入口与操作数分解：固定分配、构建及双单位配对响应。"""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
import statistics
import time

import q02_prefill_observer as observer
from l2_saturation import command_record,digest,exclusive_lock,inspect_allocated_gpu,read_json,require,save_json

HERE=Path(__file__).resolve().parent
SCHEMA='gh200-q02-wait-entry/1'
STRIDES=[[4096,2304],[4104,2304]]
DELAYS=[0,8192]
SPLIT_ARMS=[[7,4096,2304],[7,4104,2304],[7,4096,2320],
    [1,4096,2304],[1,4104,2304],[6,4096,2304],[6,4096,2320],[6,4096,2344]]
METRICS=('cycles_median','ns_median')
QUALIFICATION=dict(independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False)
base_review_build=observer.review_build


def control_key(lda,ldb,b32=False):
    return f'{lda}:{ldb}' if b32 else str(lda)


def selected_checks(mask,groups=32):
    require(mask in (1,6,7),'request mask必须为1/6/7')
    return 8192*groups*(int(bool(mask&1))+int(bool(mask&6)))


def prepare_operand_split(root,reference,b_reference):
    a_score=read_json(reference/'score.json');b_score=read_json(b_reference/'score.json')
    references=(('a',reference,752191,a_score),('b',b_reference,752208,b_score))
    expected_windows=(('a',752191,4096,2304,30918,15616),('a',752191,4104,2304,37665,19040),
        ('b',752208,4096,2304,30930,15616),('b',752208,4096,2320,32840,16576))
    controls=[]
    for name,job,lda,ldb,cycles,ns in expected_windows:
        score=a_score if name=='a' else b_score
        condition=next(c for c in score['conditions'] if c['lda']==lda and c['ldb']==ldb and c['wait_delay_cycles']==0)
        values={metric:condition['windows'][metric]['median'] for metric in METRICS}
        require(values==dict(cycles_median=cycles,ns_median=ns),'操作数分解旧窗口控制不符')
        controls.append(dict(reference_job=job,request_mask=7,lda=lda,ldb=ldb,**values))
    effects={}
    for name,folder,job,score in references:
        approved_target(folder)
        require(score['probe_sha256']==digest(folder/'build/probe'),'旧参考binary不符')
        rows={(r['seed'],r['lda'],r['ldb']):r for r in score['formal'] if r['wait_delay_cycles']==0}
        changed=(4104,2304) if name=='a' else (4096,2320)
        values={metric:statistics.median(rows[(seed,*changed)]['metrics'][metric]-
            rows[(seed,4096,2304)]['metrics'][metric] for seed in (3,22,41)) for metric in METRICS}
        require(values==({'cycles_median':6737,'ns_median':3424} if name=='a' else
            {'cycles_median':1910,'ns_median':960}),'操作数分解旧配对响应不符')
        effects['A' if name=='a' else 'B32']=dict(reference_job=job,**values)
    require(digest(reference/'build/probe')==digest(b_reference/'build/probe') and
        read_json(reference/'plan.json')['gpu_uuid']==read_json(b_reference/'plan.json')['gpu_uuid'],
        '两个新协议参考binary/卡不同')
    root.mkdir(parents=True,exist_ok=False)
    for directory in ('source','source-before','reference'):(root/directory).mkdir()
    for name in ('q02_wait_entry.py',):shutil.copy2(HERE/name,root/'source'/name)
    shutil.copy2(HERE/'probes/tma_gemm_sources.cu',root/'source/tma_gemm_sources.cu')
    for name in ('q02_prefill_observer.py','q02_tma_stride.py','l2_saturation.py',
            'r00_common.hpp','cutlass-3.9.2-include.tar.gz'):
        shutil.copy2(reference/'source'/name,root/'source'/name)
    shutil.copy2(reference/'source/tma_gemm_sources.cu',root/'source-before/tma_gemm_sources.cu')
    shutil.copy2(reference/'worklist.txt',root/'worklist.txt')
    shutil.copy2(HERE.parents[2]/'Docs/ModelEvaluation/gemm/PLAN.md',root/'reference/campaign-PLAN.md')
    for name,folder,job,score in references:
        target=root/'reference'/name;target.mkdir()
        for filename in ('score.json','binding.json','plan.json','commands.json','correctness_commands.json','approved-target.json'):
            shutil.copy2(folder/filename,target/filename)
        shutil.copy2(folder/'source/tma_gemm_sources.cu',target/'tma_gemm_sources.cu')
        for filename in ('probe','target-review.json','kernel-code.json'):
            shutil.copy2(folder/'build'/filename,target/filename)
        shutil.copytree(folder/'build/after/disassemble',target/'disassemble')
        for split,filename in (('correctness','correctness_commands.json'),('formal','commands.json')):
            for task in read_json(folder/filename):
                shutil.copy2(folder/'runs'/split/task['id']/'stdout',target/f"{split}-seed{task['seed']}.ndjson")
    for filename,verify in (('commands.json',0),('correctness_commands.json',1)):
        tasks=[]
        for old in read_json(reference/'commands.json'):
            task=copy.deepcopy(old);seed=task['seed']
            arms=SPLIT_ARMS if seed==3 else list(reversed(SPLIT_ARMS))
            task.update(id=f'q1-operand-split-seed{seed}',role='development_decomposition',
                pair_indices=list(range(8)),paired_strides=[arm[1:] for arm in arms],
                paired_request_masks=[arm[0] for arm in arms],paired_traces=[0],paired_wait_delays=[0])
            task.pop('expected_checked',None)
            if verify:task['expected_checked_by_arm']=[selected_checks(arm[0]) for arm in arms]
            argv=task['argv']
            for flag,value in (('--verify',str(verify)),('--paired-traces','0'),('--paired-wait-delays','0'),
                    ('--wait-delay-cycles','0'),('--request-mask','7'),
                    ('--paired-strides',','.join(f'{a}:{b}' for _,a,b in arms)),
                    ('--paired-request-masks',','.join(str(mask) for mask,_,_ in arms))):
                if flag in argv:argv[argv.index(flag)+1]=value
                else:argv+=[flag,value]
            tasks.append(task)
        require(len(tasks)==3,'每类必须三个配对进程');save_json(root/filename,tasks)
    plan=dict(schema=SCHEMA,qualification='development_decomposition',contrast='operand_split',**QUALIFICATION,
        gpu_uuid=read_json(reference/'plan.json')['gpu_uuid'],required_smid=124,seeds=[3,22,41],
        arms=SPLIT_ARMS,strides=[arm[1:] for arm in SPLIT_ARMS],request_masks=[arm[0] for arm in SPLIT_ARMS],
        active_groups=[1],wait_delays=[0],traces=[0],m=2304,n=2304,k=4096,ctas=1,groups=32,ring=64,
        physical_slots=4,alloc_lda=4104,alloc_ldb=2360,request_bytes=[16384,8192,8192],swizzle='SW128',tile=[7,6],
        warmup_groups=64,warmup_active_groups=4,warmup_calls=10,warmup_trace=-1,warmup_wait_delay_cycles=0,
        warmup_request_mask=7,release_delay_ns=0,wait_hint_ns=10000000,development_conditions=8,
        correctness_processes=3,formal_processes=3,numeric_arm_checks=24,formal_arm_observations=24,
        numeric_checked_values=8650752,relative_tolerance=.05,window_relative_spread=.05,ratio_absolute_tolerance=.05,
        control_windows=controls,control_effects=effects,reference_jobs=[752191,752208],
        b_reference_original_decision=b_score['decision'],b_reference_original_observation_qualified=b_score['observation_qualified'],
        b_reference_use='752208仅为新协议软件开发控制；相对752146的原资格失败保留',
        before_source_sha256=digest(root/'source-before/tma_gemm_sources.cu'),original_probe_sha256=a_score['probe_sha256'],
        reference_identity={name:dict(job=job,score_sha256=digest(folder/'score.json'),binding_sha256=digest(folder/'binding.json'),
            probe_sha256=digest(folder/'build/probe')) for name,folder,job,_ in references},
        allocation_budget_seconds=600,maximum_gpu_seconds=3600,maximum_repairs=2,automatic_retries=False,
        candidates=[dict(name='response_depends_on_coexistence',ratio=0),dict(name='operand_response_preserved',ratio=1)],
        ratio_definition=dict(MA='median_seed(Aonly_A16-Aonly_base)/median_seed(all_A16-all_base)',
            MB='median_seed(Bpair_B32-Bpair_base)/median_seed(all_B32-all_base)'),
        classification='each operand: both units abs(M)<=.05 or abs(M-1)<=.05; disagreement/qualification failure: information insufficient; both miss: stop both positions',
        B80_report_only=True,signed_or_zero_delta_allowed=True,no_bare_price=True,no_refill_extrapolation=True,
        no_additive_window_assumption=True,frozen_at_ns=time.time_ns())
    save_json(root/'plan.json',plan)
    (root/'README.md').write_text('''# Q02：A／B请求的独立响应分解

八开发条件按(mask,lda,ldb)为(7,4096,2304)、(7,4104,2304)、(7,4096,2320)、(1,4096,2304)、(1,4104,2304)、(6,4096,2304)、(6,4096,2320)、(6,4096,2344)。mask1选择A，mask6选择B0/B1，mask7选择三请求；仅允许这三个值，默认7。保留真实描述符、SW128、四物理槽及代次，固定allocA4104/allocB2360、ring64/K4096、tile[7,6]、SM124。每seed同一分配八臂，seed3正序、22/41逆序；先3verify后3plain进程。每臂暖机总为mask7、Q4/G64、trace−1、D0十次，正式Q1/G32、trace0/D0。24数值臂只检查所选区域，共8650752值；24正式臂只保留begin/end。

mask7 base/A16窗口30918/37665cycle、15616/19040ns及同seed配对增量6737cycle/3424ns以752191为控制；mask7 base/B32窗口30930/32840cycle、15616/16576ns及配对增量1910cycle/960ns以752208为软件开发控制，均各5%。752208相对752146的原资格失败保留。完整数值/身份资格和八个正式plain窗口的三进程范围≤中位5%均须通过，verify计时不参与时间判别。

先按同seed相减再取中位，报告各mask的A16/B32/B80响应。MA=ΔAonly/ΔallA，MB=ΔBpair32/ΔallB32；每操作数的两单位均在0±.05支持响应依赖三请求共存，均在1±.05支持去掉另一操作数后响应保留。两单位冲突或资格失败为信息不足；两单位均未命中则停止该操作数的两位置形式。B80只报告新约束，不指定比例。零或负增量保留；零分母报告信息不足。此轮仅作新程序开发分解，不产生裸价格，不将独立窗口相加，独立C2/C3和完整GEMM验收均false。

```sh
python -B cpu-check.py
python -B source/q02_wait_entry.py build-only --root "$PWD"
python -B source/q02_wait_entry.py freeze-execution --root "$PWD"
python -B source/q02_wait_entry.py execute --root "$PWD"
python -B source/q02_wait_entry.py score --root "$PWD" --output /tmp/q02-operand-split-score.json
```

prepare只保存源码、登记PLAN、旧参考score/binding/binary/完整SASS及raw，不编译/GPU。沿用prefill构建与capture helpers；build-only仅给初步报告，静态UTMA数量不替代路径核验。根代理须核所有prefill/refill的1/2/3条请求、expect_tx、原参数偏移与发布/等待顺序、trace0无新戳、无spill及SMEM/驻留后批准新目标，再冻结执行身份。CPU-only构建10分钟、采样分配10分钟且GPU≤1小时，最多两次定向实现修复，无自动重采或扩展条件。
''')
    names=[str(p.relative_to(root)) for directory in ('source','source-before','reference')
        for p in (root/directory).rglob('*') if p.is_file()]
    names+=['plan.json','commands.json','correctness_commands.json','worklist.txt','README.md']
    save_json(root/'build-input-binding.json',dict(schema='gh200-q02-wait-entry-build-input/1',
        files={name:digest(root/name) for name in sorted(names)}))
    print('PREPARED: 8 operand-split development conditions; 3 verify +3 formal processes; no compilation/GPU',flush=True)


def prepare(root,reference,b32_reference=None):
    control_reference=b32_reference or reference
    strides=[[4096,2304],[4096,2320]] if b32_reference else STRIDES
    prior=read_json(control_reference/'score.json')
    controls={control_key(c['lda'],c['ldb'],bool(b32_reference)):
        {metric:w['median'] for metric,w in c['windows'].items()}
        for c in prior['conditions'] if [c['lda'],c['ldb']] in strides and
            c.get('active_groups',1)==1 and c.get('trace',0)==0}
    expected_controls={'4096:2304':{'cycles_median':30845,'ns_median':15552},
        '4096:2320':{'cycles_median':32869,'ns_median':16608}} if b32_reference else \
        {'4096':{'cycles_median':30920,'ns_median':15616},'4104':{'cycles_median':37715,'ns_median':19040}}
    require(controls==expected_controls,'旧完整窗口控制不符')
    old_rows={(r['seed'],r['lda'],r['ldb']):r for r in prior['formal']
        if [r['lda'],r['ldb']] in strides and r.get('active_groups',1)==1 and r.get('trace',0)==0}
    effects={metric:statistics.median(old_rows[(seed,*strides[1])]['metrics'][metric]-
        old_rows[(seed,*strides[0])]['metrics'][metric] for seed in (3,22,41)) for metric in METRICS}
    require(effects==({'cycles_median':2014,'ns_median':1056} if b32_reference else
        {'cycles_median':6806,'ns_median':3424}),'旧同seed配对响应不符')
    require(digest(control_reference/'build/probe')==prior['probe_sha256'],'旧参考binary不符')
    if b32_reference:
        approved_target(reference)
        require(read_json(reference/'plan.json')['gpu_uuid']==read_json(control_reference/'plan.json')['gpu_uuid'],
            '对照与复用binary参考卡不同')
    root.mkdir(parents=True,exist_ok=False)
    for directory in ('source','source-before','reference'):(root/directory).mkdir()
    shutil.copy2(HERE/'q02_wait_entry.py',root/'source/q02_wait_entry.py')
    shutil.copy2(reference/'source/tma_gemm_sources.cu' if b32_reference else HERE/'probes/tma_gemm_sources.cu',
        root/'source/tma_gemm_sources.cu')
    for name in ('q02_prefill_observer.py','q02_tma_stride.py','l2_saturation.py',
                 'r00_common.hpp','cutlass-3.9.2-include.tar.gz'):
        shutil.copy2(reference/'source'/name,root/'source'/name)
    shutil.copy2(control_reference/'source/tma_gemm_sources.cu',root/'source-before/tma_gemm_sources.cu')
    shutil.copy2(reference/'worklist.txt',root/'worklist.txt')
    for name in ('score.json','binding.json','plan.json','commands.json','correctness_commands.json','approved-target.json'):
        shutil.copy2(control_reference/name,root/'reference'/name)
    for old,new in (('build/probe','probe'),('build/target-review.json','target-review.json'),
                    ('build/kernel-code.json','kernel-code.json'),('build/after/disassemble/stdout','sass.txt')):
        if b32_reference and old=='build/kernel-code.json':continue
        shutil.copy2(control_reference/old,root/'reference'/new)
    if b32_reference:
        shutil.copytree(reference/'build',root/'build')
        shutil.copytree(control_reference/'build/after/disassemble',root/'reference/disassemble')
        (root/'binary-reference').mkdir()
        for name in ('plan.json','binding.json','approved-target.json','score.json'):
            shutil.copy2(reference/name,root/'binary-reference'/name)
    for seed in (3,22,41):
        for split in ('correctness','formal'):
            old_id=f'q1-a-b-strides-seed{seed}' if b32_reference else f'q1-prefill-paired-seed{seed}'
            shutil.copy2(control_reference/f'runs/{split}/{old_id}/stdout',
                root/f'reference/{split}-seed{seed}.ndjson')
    for filename,verify in (('commands.json',0),('correctness_commands.json',1)):
        tasks=[]
        for old in read_json(reference/'commands.json'):
            if old['active_groups']!=1:continue
            task=copy.deepcopy(old);seed=task['seed']
            task['id']=f'q1-b32-wait-paired-seed{seed}' if b32_reference else f'q1-wait-paired-seed{seed}'
            task['paired_traces']=[0]
            task['paired_strides']=strides if seed==3 else list(reversed(strides))
            task['paired_wait_delays']=DELAYS if seed==3 else list(reversed(DELAYS))
            argv=task['argv'];argv[argv.index('--verify')+1]=str(verify)
            argv[argv.index('--paired-traces')+1]='0'
            argv[argv.index('--paired-strides')+1]=','.join(f'{a}:{b}' for a,b in task['paired_strides'])
            for flag,value in (('--wait-delay-cycles','0'),
                    ('--paired-wait-delays',','.join(map(str,task['paired_wait_delays'])))):
                if flag in argv:argv[argv.index(flag)+1]=value
                else:argv+=[flag,value]
            if verify:task['expected_checked']=524288
            tasks.append(task)
        require(len(tasks)==3,'每类必须三个配对进程')
        save_json(root/filename,tasks)
    plan=dict(schema=SCHEMA,qualification='development_localization',**QUALIFICATION,
        gpu_uuid=read_json(reference/'plan.json')['gpu_uuid'],required_smid=124,seeds=[3,22,41],
        active_groups=[1],strides=strides,wait_delays=DELAYS,traces=[0],m=2304,n=2304,k=4096,
        ctas=1,groups=32,ring=64,physical_slots=4,alloc_lda=4104,alloc_ldb=2360,
        request_bytes=[16384,8192,8192],swizzle='SW128',tile=[7,6],
        warmup_groups=64,warmup_active_groups=4,warmup_calls=10,warmup_trace=-1,warmup_wait_delay_cycles=0,
        release_delay_ns=0,wait_hint_ns=10000000,development_conditions=4,correctness_processes=3,formal_processes=3,
        numeric_arm_checks=12,formal_arm_observations=12,numeric_checked_values=6291456,
        relative_tolerance=.05,window_relative_spread=.05,ratio_absolute_tolerance=.05,
        control_windows=controls,control_effects=effects,reference_job=752146 if b32_reference else 752176,
        reference_score_sha256=digest(control_reference/'score.json'),reference_binding_sha256=digest(control_reference/'binding.json'),
        before_source_sha256=digest(root/'source-before/tma_gemm_sources.cu'),original_probe_sha256=prior['probe_sha256'],
        allocation_budget_seconds=600,maximum_gpu_seconds=3600,maximum_repairs=2,automatic_retries=False,
        wait_delay_location='thread0 before every Barrier::wait; clock64 spin only for nonzero D',
        candidates=[dict(name='overlap_absorbed',ratio=0),dict(name='residual_unabsorbed',ratio=1)],
        ratio_definition='median_seed(T_A16_D8192-T_A0_D8192)/median_seed(T_A16_D0-T_A0_D0)',
        classification='both units within abs(R)<=.05: overlap; both within abs(R-1)<=.05: residual; both fail: stop both; disagreement: information insufficient',
        signed_or_zero_delta_allowed=True,no_bare_price=True,no_refill_extrapolation=True,frozen_at_ns=time.time_ns())
    if b32_reference:
        plan.update(contrast='B32',binary_reuse=True,binary_reference_job=752191,
            ratio_definition='median_seed(T_B32_D8192-T_B0_D8192)/median_seed(T_B32_D0-T_B0_D0)',
            reference_warmup_trace=0,warmup_protocol_identical=False,
            warmup_protocol_comparison='752146暖机trace0；新包trace-1/D0；仅由D0窗口及配对效应检验迁移',
            reused_binary_identity=dict(probe_sha256=digest(reference/'build/probe'),
                kernel_code_sha256=digest(reference/'build/kernel-code.json'),
                target_review_sha256=digest(reference/'build/target-review.json'),
                source_sha256=digest(reference/'source/tma_gemm_sources.cu'),
                reference_binding_sha256=digest(reference/'binding.json'),
                reference_approved_target_sha256=digest(reference/'approved-target.json')))
    save_json(root/'plan.json',plan)
    readme="""# Q02：Q1 等待入口干预

Q1/G32 × A lda4096/4104、B ldb2304 × wait_delay_cycles0/8192，共4开发条件。真实三请求A16KiB、B0/B1各8KiB、SW128、四物理槽、ring64、K4096、tile[7,6]及SM124；固定allocA4104/allocB2360。每seed3/22/41先1verify配对进程，再1plain配对进程，各进程4臂：延迟外层、stride内层，seed3都正序，22/41都逆序。每臂重填同一分配，统一trace−1、Q4/G64/D0暖机10次，正式恢复trace0、Q1/G32及本臂延迟。12数值臂检查6291456值；12正式臂只保留begin/end。

新参数追加在kernel参数末尾；D0跳过clock读取。D8192仅thread0在每组Barrier::wait前用clock64自旋8192cycle，不增加循环Point，不改release_delay_ns=0。保留旧binary、旧源码、旧score/binding、Q1原始记录和完整SASS。准备阶段不编译、不启动GPU。

数值及cycle/ns三进程窗口极差≤中位5%为资格条件。D0的A0/A16窗口须各复现752176的30920/37715cycle、15616/19040ns各5%；同seed配对增量中位6806cycle/3424ns也各5%。每单位先求同seed A16−A0，再求三seed中位，R=ΔD8192/ΔD0。abs(R)≤0.05支持overlap_absorbed；abs(R−1)≤0.05支持residual_unabsorbed。两单位一致才判定；两单位都失败则停止两形式；分类不一致或资格失败为信息不足。新Δ为0或负数仍照式评分，不据此拒绝资格。独立C2/C3和完整GEMM验收均false，不生成裸服务价格。

```sh
python -B source/q02_wait_entry.py build-only --root "$PWD"
python -B source/q02_wait_entry.py freeze-execution --root "$PWD"
python -B source/q02_wait_entry.py execute --root "$PWD"
python -B source/q02_wait_entry.py score --root "$PWD" --output /tmp/q02-wait-entry-score.json
```

build-only直接复用prefill runner构建/反汇编/资源流程，仅生成初步target-review。根代理须另审D0分支无新clock、D8192位于每组wait前、三UTMA请求及旧参数offset保持、无spill且SMEM/驻留一致，再写approved-target。freeze-execution仅在目标批准后保存最终binding；execute核对冻结身份后才运行。CPU构建与GPU采样各10分钟，GPU绝对上限1小时；最多两次定向实现修复，无自动重采或失败后扩展条件。
"""
    if b32_reference:
        readme="""# Q02：B32 响应的等待入口判别

Q1/G32、A lda4096固定 × B ldb2304/2320 × wait_delay_cycles0/8192，共4开发条件。复用wait-entry001已批准binary和完整build，不改CUDA源码、不编译。真实三请求A16KiB、B0/B1各8KiB、SW128、四物理槽、ring64、K4096、tile[7,6]及SM124；固定allocA4104/allocB2360。先3verify进程，再3plain进程，各seed3/22/41同进程四臂；延迟外层、stride内层，seed3两维正序，22/41都逆序。每臂重填同一分配，统一trace−1、Q4/G64/D0暖机10次，正式trace0、Q1/G32；12数值臂检查6291456值。

752146参考的暖机为trace0，新包为trace−1，协议并不相同。保存旧score/binding/binary/source、三seed原始记录，并用新D0的两窗口及同seed配对效应检验迁移。D0的B0/B32窗口须分别复现30845/32869cycle、15552/16608ns各5%；配对增量[2004,2014,2026]cycle、[992,1056,1056]ns的中位2014cycle/1056ns也各5%。数值/身份资格及四条件正式plain的cycle/ns三进程极差≤中位5%均须通过。

配对键为完整(lda,ldb)，每单位先求同seed B32−B0，再求三seed中位，R=ΔD8192/ΔD0。abs(R)≤0.05支持B32响应也可被该间隔隐藏；abs(R−1)≤0.05则下一版需区分A的可隐藏项与B的残留项。两单位一致才判定；两单位均未落入0/1带则记录部分响应并停止两形式，单位冲突或资格失败为信息不足。新Δ为0或负数仍照式评分，不据此拒绝资格。仍为开发定位，不产生裸服务价格；独立C2/C3和完整GEMM验收均false。

```sh
python -B source/q02_wait_entry.py freeze-execution --root "$PWD"
python -B source/q02_wait_entry.py execute --root "$PWD"
python -B source/q02_wait_entry.py score --root "$PWD" --output /tmp/q02-b32-wait-entry-score.json
```

旧build/target-review保持原文，binary-reference保存其原plan、binding及批准。根代理另写本轮approved-target：绑定当前plan、复用binary/kernel-code/target-review三SHA及旧批准/binding来源。最终binding冻结核心、旧完整build、新批准与CPU检查。分配10分钟，GPU绝对上限1小时，无编译、自动重采或失败后扩展条件。
"""
    (root/'README.md').write_text(readme)
    names=[str(p.relative_to(root)) for directory in ('source','source-before','reference')
        for p in (root/directory).rglob('*') if p.is_file()]
    names+=['plan.json','commands.json','correctness_commands.json','worklist.txt','README.md']
    if b32_reference:
        names += [str(p.relative_to(root)) for directory in ('build','binary-reference')
            for p in (root/directory).rglob('*') if p.is_file()]
    save_json(root/'build-input-binding.json',dict(schema='gh200-q02-wait-entry-build-input/1',
        files={name:digest(root/name) for name in sorted(names)}))
    print('PREPARED: 4 development conditions; 3 verify +3 formal processes; no compilation/GPU',flush=True)


def load_plan(root,execution=False):
    for name,checksum in read_json(root/('binding.json' if execution else 'build-input-binding.json'))['files'].items():
        require(digest(root/name)==checksum,'冻结身份变更：'+name)
    plan=read_json(root/'plan.json');require(plan['schema']==SCHEMA,'执行范围不符')
    return plan


def review_build(root):
    plan=read_json(root/'plan.json')
    if plan.get('binary_reuse'):
        load_plan(root)
        expected=plan['reused_binary_identity'];folder=root/'build'
        for step in ('version','after/compile','after/disassemble','after/resources'):
            observer.stride.successful_receipt(folder/step)
        report=read_json(folder/'target-review.json')
        require(report['basic_gate_passed'] and digest(folder/'probe')==expected['probe_sha256'] and
            digest(folder/'kernel-code.json')==expected['kernel_code_sha256'] and
            digest(folder/'target-review.json')==expected['target_review_sha256'] and
            digest(root/'source/tma_gemm_sources.cu')==expected['source_sha256'] and
            observer.kernel_records((folder/'after/disassemble/stdout').read_text())==read_json(folder/'kernel-code.json'),
            '复用binary/source/完整SASS身份不符')
        old=read_json(root/'binary-reference/approved-target.json')
        require(digest(root/'binary-reference/binding.json')==expected['reference_binding_sha256'] and
            digest(root/'binary-reference/approved-target.json')==expected['reference_approved_target_sha256'] and
            old['approved'] is True and all(old[key]==expected[key] for key in
                ('probe_sha256','kernel_code_sha256','target_review_sha256')) and
            report['plan_sha256']==digest(root/'binary-reference/plan.json'),'旧批准来源不符')
        return report
    report=base_review_build(root)
    if plan.get('contrast')=='operand_split':
        row=report['rows'][0]
        report.update(status='pending_root_operand_split_sass_review',
            basic_gate_passed=row['utma_count']>0 and row['zero_local_stack'] and not row['compile_warning_c7515'],
            static_utma_count_unreviewed=True,requests_per_group_by_mask={'1':1,'6':2,'7':3})
        row['label']='operand_split'
    else:
        report['status']='pending_root_wait_entry_sass_review'
        report['rows'][0]['label']='wait_entry'
    return report


@contextmanager
def build_helpers():
    old_schema,old_review=observer.SCHEMA,observer.review_build
    observer.SCHEMA,observer.review_build=SCHEMA,review_build
    try:yield
    finally:observer.SCHEMA,observer.review_build=old_schema,old_review


def build_only(root):
    require(not read_json(root/'plan.json').get('binary_reuse'),'本包复用已批准binary，不允许编译')
    with build_helpers():observer.build_only(root)


def approved_target(root):
    plan=read_json(root/'plan.json')
    if plan.get('binary_reuse'):
        report=review_build(root);approved=read_json(root/'approved-target.json');expected=plan['reused_binary_identity']
        require(approved['approved'] is True and approved['binary_reuse'] is True and
            approved['plan_sha256']==digest(root/'plan.json') and
            all(approved[key]==expected[key] for key in ('probe_sha256','kernel_code_sha256','target_review_sha256')) and
            approved['binary_reference_binding_sha256']==expected['reference_binding_sha256'] and
            approved['binary_reference_approved_target_sha256']==expected['reference_approved_target_sha256'],
            '本轮binary复用未获绑定当前plan的核准')
        return
    with build_helpers():observer.approved_target(root)
    if plan.get('contrast')=='operand_split':
        require(read_json(root/'approved-target.json')['plan_sha256']==digest(root/'plan.json'),
            '操作数分解批准未绑定当前plan')


def validate_process(root,folder,task,uuid):
    receipt=observer.stride.successful_receipt(folder)
    command=receipt['argv'][:];probe=Path(command[0])
    require(probe.name=='probe' and probe.parent.name=='build','binary路径不符')
    require(command[command.index('--worklist')+1]==str(probe.parent.parent/'worklist.txt'),'工作列表路径不符')
    command[0]=task['argv'][0];command[command.index('--worklist')+1]='worklist.txt'
    require(command==task['argv'],'执行参数与冻结命令不符')
    raw=[json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.strip()]
    split_mode='paired_request_masks' in task
    require(len(raw)==(8 if split_mode else 4),'配对臂缺失')
    arguments=dict(zip(task['argv'][1::2],task['argv'][2::2]))
    fields={'ctas':'--ctas','active_groups':'--measure-active','groups':'--measure-groups','ring':'--ring',
        'verify':'--verify','seed':'--seed','warmup':'--warmup','source_phase_stride':'--source-phase-stride',
        'source_tile_shift':'--source-tile-shift','work_index':'--work-index','warmup_work_offset':'--warmup-work-offset',
        'release_delay_ns':'--release-delay-ns','pollute_kib':'--pollute-kib','launch_skew_ns':'--launch-skew-ns',
        'warmup_skew_ns':'--warmup-skew-ns','alloc_lda':'--alloc-lda','alloc_ldb':'--alloc-ldb'}
    require((root/'worklist.txt').read_text().split()==['2304','2304','4096','1','0','0','7','6'],'工作列表不符')
    resource=read_json(root/'build/target-review.json')['rows'][0]['resources']
    checked=[];descriptors={}
    for index,row in enumerate(raw):
        require(row['schema']=='cfg-b-gemm-source-probe/1' and row['gpu_uuid']==uuid,'schema/卡身份不符')
        require(all(row[name]==int(arguments[flag]) for name,flag in fields.items()),'逐臂参数不符')
        require(row['warmup_trace']==-1 and row['warmup_active_groups']==4 and row['warmup_groups']==64 and
            row['warmup_wait_delay_cycles']==0 and row['warmup_source_phase_stride']==0 and
            row['record_warmup']==0 and not row['warmup_ctas'] and not row['group_events'],'暖机或trace范围不符')
        delay_index=index//len(task['paired_strides']);stride_index=index%len(task['paired_strides'])
        require(row['paired_strides']==task['paired_strides'] and row['pair_index']==index and
            row['trace_index']==0 and row['paired_traces']==[0] and row['trace']==0 and
            row['wait_delay_index']==delay_index and row['stride_index']==stride_index and
            row['paired_wait_delays']==task['paired_wait_delays'] and
            row['wait_delay_cycles']==task['paired_wait_delays'][delay_index] and
            [row['lda'],row['ldb']]==task['paired_strides'][stride_index],'逐臂延迟/stride或次序不符')
        require(all(row[name]==raw[0][name] for name in observer.stride.ADDRESSES),'同进程地址变化')
        require(row['source_a_address']>0 and row['source_b_address']>0 and
            row['source_a_address']%128==row['source_b_address']%128==0,'源地址/对齐不符')
        require(row['m']==row['n']==2304 and row['allocated_a_elements']==9455616 and
            row['allocated_b_elements']==9666560,'形状或固定分配不符')
        require(row['errors']==row['pollution_errors']==row['pollution_checked']==0,'加载/污染检查错误')
        expected_checked=524288
        if split_mode:
            mask=task['paired_request_masks'][stride_index]
            expected_checked=selected_checks(mask)
            require(row['request_mask']==mask and row['warmup_request_mask']==7 and
                row['paired_request_masks']==task['paired_request_masks'] and
                row['transaction_bytes']==selected_checks(mask,1)*2 and row['warmup_transaction_bytes']==32768,
                '所选请求/事务字节或暖机mask不符')
            descriptor_key=(row['lda'],row['ldb'])
            pair=(row['descriptor_a'],row['descriptor_b'])
            require(all(re.fullmatch('[0-9a-f]{256}',value) for value in pair),'descriptor长度/编码不符')
            require(descriptors.setdefault(descriptor_key,pair)==pair,'同stride不同mask的descriptor变化')
        require(row['checked']==(expected_checked if row['verify'] else 0),'完整加载检查量不符')
        require(row['observed_sms']==row['max_active_ctas_per_sm']==1 and row['smem']==132096 and
            row['local_bytes']==0 and row['registers']==int(re.search(r'\bREG:(\d+)\b',resource)[1]),'驻留或资源不符')
        require(len(row['ctas_data'])==1,'CTA数量不符');cta=row['ctas_data'][0]
        require(cta['cta']==0 and cta['tile']==[7,6] and cta['smid']==124 and 'prefill_end' not in cta,
            'CTA/坐标/SM124或plain戳不符')
        for unit in (0,1):
            require(type(cta['begin'][unit]) is int and type(cta['end'][unit]) is int and
                0<cta['begin'][unit]<cta['end'][unit],'完成戳无效')
        cycles=cta['end'][0]-cta['begin'][0];ns=cta['end'][1]-cta['begin'][1]
        checked.append(dict(seed=row['seed'],active_groups=1,trace=0,lda=row['lda'],ldb=row['ldb'],
            wait_delay_cycles=row['wait_delay_cycles'],pair_index=index,checked_values=row['checked'],
            numerical_qualified=bool(row['verify']),stdout_sha256=receipt['stdout_sha256'],
            source=str(folder.relative_to(root)),device_uuid=uuid,
            metrics=dict(cycles_median=cycles,ns_median=ns,ns_envelope=ns)))
        if split_mode:checked[-1]['request_mask']=row['request_mask']
    return checked


def summarize_operand_split(plan,numeric,formal):
    key=lambda r:(r['seed'],r['request_mask'],r['lda'],r['ldb'])
    by_numeric={key(r):r for r in numeric};by_formal={key(r):r for r in formal}
    expected={(seed,*arm) for seed in plan['seeds'] for arm in plan['arms']}
    require(len(numeric)==len(formal)==24 and by_numeric.keys()==by_formal.keys()==expected,
        '数值/正式24臂完整mask/stride键不对应')
    require(all(r['numerical_qualified'] and r['checked_values']==selected_checks(r['request_mask']) and
        r['active_groups']==1 and r['trace']==r['wait_delay_cycles']==0 for r in numeric) and
        all(r['active_groups']==1 and r['trace']==r['wait_delay_cycles']==0 for r in formal) and
        sum(r['checked_values'] for r in numeric)==plan['numeric_checked_values'],'所选区域数值/正式范围资格不符')
    conditions=[];controls=[];effects=[];responses=[];ratios=[]
    for mask,lda,ldb in plan['arms']:
        windows={}
        for metric in METRICS:
            values=[by_formal[(seed,mask,lda,ldb)]['metrics'][metric] for seed in plan['seeds']]
            median=statistics.median(values);require(median>0,'完整窗口须为正数')
            spread=(max(values)-min(values))/median
            windows[metric]=dict(values=values,median=median,relative_spread=spread,
                passed=spread<=plan['window_relative_spread'])
            for old in plan['control_windows']:
                if (mask,lda,ldb)==(old['request_mask'],old['lda'],old['ldb']):
                    error=abs(median-old[metric])/old[metric]
                    controls.append(dict(request_mask=mask,lda=lda,ldb=ldb,metric=metric,
                        reference_job=old['reference_job'],expected=old[metric],observed=median,
                        relative_error=error,passed=error<=plan['relative_tolerance']))
        conditions.append(dict(request_mask=mask,lda=lda,ldb=ldb,windows=windows,
            passed=all(window['passed'] for window in windows.values())))
    pairs=(('all_A16',7,4104,2304),('all_B32',7,4096,2320),('Aonly_A16',1,4104,2304),
        ('Bpair_B32',6,4096,2320),('Bpair_B80',6,4096,2344))
    by_response={}
    for name,mask,lda,ldb in pairs:
        units={}
        for metric in METRICS:
            values=[by_formal[(seed,mask,lda,ldb)]['metrics'][metric]-
                by_formal[(seed,mask,4096,2304)]['metrics'][metric] for seed in plan['seeds']]
            median=statistics.median(values);units[metric]=dict(values=values,median=median)
            if name in ('all_A16','all_B32'):
                old=plan['control_effects']['A' if name=='all_A16' else 'B32']
                error=abs(median-old[metric])/old[metric]
                effects.append(dict(response=name,metric=metric,reference_job=old['reference_job'],
                    values=values,median=median,expected=old[metric],relative_error=error,
                    passed=error<=plan['relative_tolerance']))
        response=dict(name=name,request_mask=mask,lda=lda,ldb=ldb,units=units)
        responses.append(response);by_response[name]=response
    positions={};consistent=True
    for operand,numerator,denominator in (('A','Aonly_A16','all_A16'),('B','Bpair_B32','all_B32')):
        classifications=[]
        for metric in METRICS:
            top=by_response[numerator]['units'][metric]['median']
            bottom=by_response[denominator]['units'][metric]['median']
            ratio=top/bottom if bottom else None
            classification='information_insufficient'
            if ratio is not None:
                classification='response_depends_on_coexistence' if abs(ratio)<=plan['ratio_absolute_tolerance'] else \
                    'operand_response_preserved' if abs(ratio-1)<=plan['ratio_absolute_tolerance'] else 'both_positions_failed'
            classifications.append(classification)
            ratios.append(dict(operand=operand,metric=metric,numerator=top,denominator=bottom,
                ratio=ratio,classification=classification))
        matched=len(set(classifications))==1
        consistent=consistent and matched
        positions[operand]=classifications[0] if matched else 'information_insufficient'
    control_passed=all(row['passed'] for row in controls+effects)
    stable=all(row['passed'] for row in conditions);qualified=control_passed and stable
    decision='information_insufficient'
    if qualified and consistent and 'information_insufficient' not in positions.values():
        decision='stop_unmatched_operand_position_forms' if 'both_positions_failed' in positions.values() else \
            'development_positions_classified'
    return dict(schema='gh200-q02-operand-split-score/1',qualification=plan['qualification'],decision=decision,
        **QUALIFICATION,numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),
        numeric_checked_values=sum(r['checked_values'] for r in numeric),numeric=numeric,formal=formal,
        conditions=conditions,control=dict(windows=controls,effects=effects,passed=control_passed),
        stable_windows=stable,observation_qualified=qualified,responses=responses,ratios=ratios,
        operand_positions=positions,unit_classifications_consistent=consistent,B80_report_only=True,
        signed_or_zero_delta_allowed=True,no_bare_price=True,no_refill_extrapolation=True,
        no_additive_window_assumption=True,b_reference_original_decision=plan['b_reference_original_decision'])


def summarize_rows(plan,numeric,formal):
    if plan.get('contrast')=='operand_split':return summarize_operand_split(plan,numeric,formal)
    key=lambda r:(r['seed'],r['lda'],r['ldb'],r['wait_delay_cycles'])
    by_numeric={key(r):r for r in numeric};by_formal={key(r):r for r in formal}
    expected={(seed,lda,ldb,delay) for seed in plan['seeds'] for lda,ldb in plan['strides'] for delay in plan['wait_delays']}
    require(len(numeric)==len(formal)==12 and by_numeric.keys()==by_formal.keys()==expected,'数值/正式12臂不对应')
    require(all(r['numerical_qualified'] and r['checked_values']==524288 for r in numeric) and
        sum(r['checked_values'] for r in numeric)==plan['numeric_checked_values'],'数值资格不符')
    conditions=[];controls=[];effects=[];ratios=[]
    for lda,ldb in plan['strides']:
        for delay in plan['wait_delays']:
            rows=[by_formal[(seed,lda,ldb,delay)] for seed in plan['seeds']];windows={}
            for metric in METRICS:
                values=[r['metrics'][metric] for r in rows];median=statistics.median(values)
                require(median>0,'完整窗口须为正数')
                spread=(max(values)-min(values))/median
                windows[metric]=dict(values=values,median=median,relative_spread=spread,
                    passed=spread<=plan['window_relative_spread'])
                if delay==0:
                    old=plan['control_windows'][control_key(lda,ldb,plan.get('contrast')=='B32')][metric]
                    error=abs(median-old)/old
                    control=dict(lda=lda,metric=metric,expected=old,observed=median,
                        relative_error=error,passed=error<=plan['relative_tolerance'])
                    if plan.get('contrast')=='B32':control['ldb']=ldb
                    controls.append(control)
            conditions.append(dict(lda=lda,ldb=ldb,wait_delay_cycles=delay,windows=windows,
                passed=all(w['passed'] for w in windows.values())))
    for metric in METRICS:
        responses={}
        for delay in plan['wait_delays']:
            values=[by_formal[(seed,*plan['strides'][1],delay)]['metrics'][metric]-
                by_formal[(seed,*plan['strides'][0],delay)]['metrics'][metric] for seed in plan['seeds']]
            responses[str(delay)]=dict(values=values,median=statistics.median(values))
        base=responses['0']['median'];old=plan['control_effects'][metric];error=abs(base-old)/old
        effects.append(dict(metric=metric,values=responses['0']['values'],median=base,expected=old,
            relative_error=error,passed=error<=plan['relative_tolerance']))
        ratio=responses['8192']['median']/base if base else None
        overlap=ratio is not None and abs(ratio)<=plan['ratio_absolute_tolerance']
        residual=ratio is not None and 1-plan['ratio_absolute_tolerance']<=ratio<=1+plan['ratio_absolute_tolerance']
        classification='overlap_absorbed' if overlap else 'residual_unabsorbed' if residual else \
            'both_forms_failed' if ratio is not None else 'information_insufficient'
        ratios.append(dict(metric=metric,responses=responses,ratio=ratio,overlap_absorbed=overlap,
            residual_unabsorbed=residual,classification=classification))
    control_passed=all(row['passed'] for row in controls+effects)
    stable=all(row['passed'] for row in conditions);qualified=control_passed and stable
    classifications=[row['classification'] for row in ratios]
    consistent=len(set(classifications))==1
    decision='information_insufficient'
    if qualified and consistent:
        decision={'overlap_absorbed':'supports_overlap_absorbed','residual_unabsorbed':'supports_residual_unabsorbed',
            'both_forms_failed':'stop_both_wait_entry_forms'}.get(classifications[0],'information_insufficient')
    return dict(schema='gh200-q02-wait-entry-score/1',qualification=plan['qualification'],decision=decision,
        **QUALIFICATION,numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),
        numeric_checked_values=sum(r['checked_values'] for r in numeric),numeric=numeric,formal=formal,
        conditions=conditions,control=dict(windows=controls,effects=effects,passed=control_passed),
        stable_windows=stable,observation_qualified=qualified,ratios=ratios,unit_classifications_consistent=consistent,
        signed_or_zero_delta_allowed=True,no_bare_price=True,no_refill_extrapolation=True)


def freeze_execution(root):
    load_plan(root);approved_target(root)
    require(not (root/'binding.json').exists(),'最终binding已存在')
    names=list(read_json(root/'build-input-binding.json')['files'])+['build-input-binding.json','approved-target.json']
    names += [str(p.relative_to(root)) for p in (root/'build').rglob('*') if p.is_file()]
    names += [name for name in ('cpu-check.py','cpu-check.json') if (root/name).is_file()]
    save_json(root/'binding.json',dict(schema='gh200-q02-wait-entry-execution-binding/1',frozen_at_ns=time.time_ns(),
        files={name:digest(root/name) for name in sorted(set(names))}))


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
                folder=root/'runs'/split/task['id']
                command_record(folder,observer.stride.probe_command(root,task),observer.stride.remaining(start,60))
                save_json(folder/'validated.json',validate_process(root,folder,task,environment['uuid']))
                print(json.dumps(dict(split=split,process=task['id'],arms=len(task.get('paired_request_masks',[])) or 4)),flush=True)
        save_json(root/'score.json',summarize(root))
    print(f"CLOSED: {plan['numeric_arm_checks']} numeric +{plan['formal_arm_observations']} formal arms; {plan['qualification']}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('prepare','build-only','freeze-execution','execute','score'):
        p=sub.add_parser(mode);p.add_argument('--root',type=Path,required=True)
        if mode=='prepare':
            p.add_argument('--reference',type=Path,required=True)
            p.add_argument('--b32-reference',type=Path,
                help='B32旧对照；指定时--reference为复用binary参考，省略时保持原A模式')
            p.add_argument('--operand-split-reference',type=Path,
                help='操作数分解的752208软件控制；--reference为752191新协议A控制，重新构建')
        if mode=='score':p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.mode=='prepare':
        require(not(args.b32_reference and args.operand_split_reference),'两种prepare模式不可同时指定')
        if args.operand_split_reference:prepare_operand_split(root,args.reference.resolve(),args.operand_split_reference.resolve())
        else:prepare(root,args.reference.resolve(),args.b32_reference.resolve() if args.b32_reference else None)
    elif args.mode=='build-only':build_only(root)
    elif args.mode=='freeze-execution':freeze_execution(root)
    elif args.mode=='execute':execute(root)
    else:save_json(args.output,summarize(root))


if __name__=='__main__':main()

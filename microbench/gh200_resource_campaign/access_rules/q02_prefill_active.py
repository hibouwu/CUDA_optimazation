#!/usr/bin/env python3
"""复用752255原binary，判别正式Q2/Q4的首次预填响应；不编译。"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import statistics
import time

import q02_prefill_observer as observer
from l2_saturation import command_record,digest,exclusive_lock,inspect_allocated_gpu,read_json,require,save_json

HERE=Path(__file__).resolve().parent
SCHEMA='gh200-q02-prefill-active/1'
APPROVAL_SCHEMA='gh200-q02-prefill-active-reuse-approval/1'
STRIDES=[[4096,2304],[4104,2304]]


def verify_reused_binary(root,plan):
    old=read_json(root/'reference/approved-target.json')
    identity=plan['reused_binary_identity']
    expected={'build/probe':old['probe_sha256'],'build/kernel-code.json':old['kernel_code_sha256'],
        'build/target-review.json':old['target_review_sha256']}
    require(old['approved'] is True and identity['files']==expected and
        all(digest(root/name)==value for name,value in expected.items()),'复用binary/完整代码/review身份不符')
    require(identity['reference_plan_sha256']==old['plan_sha256']==digest(root/'reference/plan.json') and
        identity['reference_approved_target_sha256']==digest(root/'reference/approved-target.json') and
        identity['source_sha256']==digest(root/'source/tma_gemm_sources.cu'),'原批准/plan/source身份不符')
    return identity


def approved_target(root,plan):
    identity=verify_reused_binary(root,plan)
    approved=read_json(root/'approved-target.json')
    require(approved['schema']==APPROVAL_SCHEMA and approved['approved'] is True and
        approved['new_plan_sha256']==digest(root/'plan.json') and
        approved['reused_binary_identity']==identity,'本轮复用批准未绑定当前plan和原binary身份')


def load_plan(root,execution=False):
    for name,value in read_json(root/('binding.json' if execution else 'build-input-binding.json'))['files'].items():
        require(digest(root/name)==value,'冻结身份变更：'+name)
    plan=read_json(root/'plan.json');require(plan['schema']==SCHEMA,'执行范围不符')
    verify_reused_binary(root,plan)
    return plan


def prepare(root,reference):
    old=read_json(reference/'score.json');old_approval=read_json(reference/'approved-target.json')
    require(old['observation_qualified'] is True and old_approval['approved'] is True,'752255未通过原资格或批准')
    native={'build/probe':old_approval['probe_sha256'],'build/kernel-code.json':old_approval['kernel_code_sha256'],
        'build/target-review.json':old_approval['target_review_sha256']}
    require(all(digest(reference/name)==value for name,value in native.items()) and
        old_approval['plan_sha256']==digest(reference/'plan.json') and old['probe_sha256']==native['build/probe'],
        '752255原binary/完整代码/review/plan不符')
    controls={str(c['lda']):{metric:window['median'] for metric,window in c['windows'].items()}
        for c in old['conditions'] if c['source_k_step']==1 and c['trace']==0}
    prefixes={str(c['lda']):c['prefill']['cycles']['median'] for c in old['conditions']
        if c['source_k_step']==1 and c['trace']==-1}
    effect=next(r for r in old['responses'] if r['source_k_step']==1)['prefill_A16']['cycles']['median']
    require(controls=={'4096':{'cycles_median':21819,'ns_median':10944},
        '4104':{'cycles_median':22317,'ns_median':11200}} and prefixes=={'4096':1389,'4104':1887} and effect==498,
        '752255 step1控制数字不符')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('source','build','reference'):(root/name).mkdir()
    shutil.copy2(HERE/'q02_prefill_active.py',root/'source/q02_prefill_active.py')
    old_binding=read_json(reference/'binding.json')['files']
    for name in ('q02_prefill_observer.py','q02_tma_stride.py','l2_saturation.py','tma_gemm_sources.cu'):
        require(digest(reference/'source'/name)==old_binding['source/'+name],'752255原source绑定不符：'+name)
        shutil.copy2(reference/'source'/name,root/'source'/name)
    for name in native:shutil.copy2(reference/name,root/name)
    for name in ('plan.json','score.json','approved-target.json'):shutil.copy2(reference/name,root/'reference'/name)
    shutil.copy2(reference/'worklist.txt',root/'worklist.txt')
    shutil.copy2(HERE.parents[2]/'Docs/ModelEvaluation/gemm/PLAN.md',root/'reference/campaign-PLAN.md')
    raw=[];raw_sources={}
    for task in read_json(reference/'commands.json'):
        path=reference/'runs/formal'/task['id']/'stdout'
        raw_sources[str(path.relative_to(reference))]=digest(path)
        raw.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip()
            and json.loads(line)['source_k_step']==1)
    require(len(raw)==12,'752255 step1原始四臂/三seed缺失')
    (root/'reference/formal.ndjson').write_text(''.join(json.dumps(row)+'\n' for row in raw))
    for filename,verify in (('commands.json',0),('correctness_commands.json',1)):
        tasks=[]
        for previous in read_json(reference/filename):
            seed=previous['seed']
            for q in ([2,4] if seed==3 else [4,2]):
                task=copy.deepcopy(previous)
                task.update(id=f'q{q}-prefill-active-seed{seed}',active_groups=q,role='development_initial_active_response',
                    pair_indices=list(range(4)),paired_strides=STRIDES if seed==3 else list(reversed(STRIDES)),
                    paired_source_steps=[1,1],paired_traces=[0,-1] if seed==3 else [-1,0])
                args=task['argv']
                for flag,value in (('--verify',str(verify)),('--measure-active',str(q)),
                        ('--paired-strides',','.join(f'{a}:{b}' for a,b in task['paired_strides'])),
                        ('--paired-source-steps','1,1'),('--paired-traces',','.join(map(str,task['paired_traces'])))):
                    args[args.index(flag)+1]=value
                if verify:task['expected_checked']=524288
                tasks.append(task)
        require(len(tasks)==6,'每类必须六个四臂进程');save_json(root/filename,tasks)
    identity=dict(reference_run=reference.name,reference_plan_sha256=digest(reference/'plan.json'),
        reference_approved_target_sha256=digest(reference/'approved-target.json'),
        source_sha256=digest(root/'source/tma_gemm_sources.cu'),files=native)
    plan=dict(schema=SCHEMA,qualification='development_initial_active_response',intervention='formal_active_groups',
        independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        gpu_uuid=read_json(reference/'plan.json')['gpu_uuid'],required_smid=124,seeds=[3,22,41],active_groups=[2,4],
        strides=STRIDES,traces=[0,-1],source_k_step=1,m=2304,n=2304,k=4096,ctas=1,groups=32,ring=64,
        physical_slots=4,slot_formula='g%4',phase_formula='(g/4)%2',alloc_lda=4104,alloc_ldb=2360,
        request_mask=7,request_bytes=[16384,8192,8192],swizzle='SW128',tile=[7,6],
        warmup_active_groups=4,warmup_groups=64,warmup_trace=-1,warmup_source_k_step=1,warmup_request_mask=7,
        warmup_wait_delay_cycles=0,warmup_calls=10,release_delay_ns=0,development_conditions=8,
        correctness_processes=6,formal_processes=6,numeric_arm_checks=24,formal_arm_observations=24,
        numeric_checked_values=12582912,relative_tolerance=.05,window_relative_spread=.05,
        prefill_cycle_relative_spread=.05,perturbation_tolerance=.05,ratio_absolute_tolerance=.05,
        control_active_groups=4,control_windows=controls,control_prefill_cycles=prefixes,control_prefill_effect_cycles=effect,
        ratio_definition='median_seed(prefill_A16_Q2-prefill_A0_Q2)/median_seed(prefill_A16_Q4-prefill_A0_Q4), cycles',
        candidates=[dict(name='response_after_second_group',ratio=0),dict(name='equal_postfirst_boundary_response',ratio=1/3),
            dict(name='first_extra_group_saturation',ratio=1)],
        actual_prefill_paths={'2':'original remainder loop twice','4':'original four-group unrolled path'},
        program_scope='formal active changes initial issuing organization and actual cadence; no physical capacity inference',
        prefill_ns_report_only=True,signed_or_zero_delta_allowed=True,no_bare_price=True,
        reused_binary_identity=identity,reference_score_sha256=digest(reference/'score.json'),reference_raw_sources=raw_sources,
        allocation_budget_seconds=600,maximum_gpu_seconds=3600,maximum_repairs=2,cpu_check_seconds_limit=120,
        automatic_retries=False,compilation=False,frozen_at_ns=time.time_ns())
    save_json(root/'plan.json',plan)
    (root/'README.md').write_text('''# Q02：初始在途组数的预填响应判别

直接复用752255已批准binary、原source及完整kernel-code/review，不编译。Q2/4×A lda4096/4104×trace0/−1共八开发条件，B ldb2304；每Q三verify、三formal进程，各进程四臂，共12进程/48臂、12582912数值。seed3按Q2/4、stride和trace正序，22/41整体逆序；不同Q不要求同一分配。每臂仍先做Q4/G64、step1、mask7、D0、trace−1十次交替暖机，正式仅恢复目标Q/G32。四物理槽g%4及代次(g/4)%2、fullwait/refill、三请求/SW128、固定alloc、K4096/ring64、tile[7,6]和SM124不变。

实际Q2走原remainder循环两次，Q4走原四组展开，发出节奏是本程序的一部分。原prefill单点仍在首次全部发出后、首次full wait前；plain不读暖机旧戳。原review保留原plan字段；新approved-target须由根代理明确绑定新plan与reused_binary_identity，旧批准只提供复用身份依据。

Q4 plain完整窗须复现752255 step1的21819/22317cycle、10944/11200ns，prefill窗1389/1887cycle及同seed增量498cycle，各5%。完整cycle/ns与稀疏prefill cycle的三进程范围≤中位5%，同seed同Q/stride sparse/plain完整窗扰动≤5%。verify只负责完整数值、身份及资源资格，prefill ns只报告。

R=Q2同seed prefill A16增量中位/同批Q4对应；三个有限形式预测0、1/3、1，各绝对0.05带。命中一个则保留本程序下的响应及实际发出组织，全部落空停止三形式；资格失败或零分母为信息不足，零/负差保留。不添加按Q补偿价，不指定物理容量，不均分旧τ/L/D，独立C2/C3及完整GEMM验收均false。

```sh
python -B cpu-check.py
python -B source/q02_prefill_active.py execute --root "$PWD"
python -B source/q02_prefill_active.py score --root "$PWD" --output /tmp/q02-prefill-active-score.json
```

准备与CPU检查不执行GPU。免费ROMEO单次10分钟分配且GPU≤1小时，最多两次定向实现修复，不扩Q、长度或地址，不自动重采。最终批准、binding与运输由根代理完成。
''')
    save_json(root/'preparation-repair.json',dict(repairs=0,compilation=False,gpu=False,automatic_retry=False))
    names=[str(p.relative_to(root)) for directory in ('source','build','reference') for p in (root/directory).rglob('*') if p.is_file()]
    names+=['plan.json','commands.json','correctness_commands.json','worklist.txt','README.md','preparation-repair.json']
    save_json(root/'build-input-binding.json',dict(schema='gh200-q02-prefill-active-input/1',
        files={name:digest(root/name) for name in sorted(names)}))
    print('PREPARED: reused752255 binary;8 conditions/12 processes/48 arms; no compilation/GPU',flush=True)


def summarize_rows(plan,numeric,formal):
    key=lambda r:(r['seed'],r['active_groups'],r['lda'],r['ldb'],r['trace'])
    by_numeric={key(r):r for r in numeric};by_formal={key(r):r for r in formal}
    expected={(seed,q,lda,ldb,trace) for seed in plan['seeds'] for q in plan['active_groups']
        for lda,ldb in plan['strides'] for trace in plan['traces']}
    require(len(numeric)==len(formal)==24 and by_numeric.keys()==by_formal.keys()==expected,'数值/正式24臂键不对应')
    require(all(r['numerical_qualified'] and r['checked_values']==524288 and r['source_k_step']==1 for r in numeric) and
        all(r['source_k_step']==1 for r in formal) and sum(r['checked_values'] for r in numeric)==12582912,'完整数值/step1资格不符')
    conditions=[];controls=[];prefix_controls=[];perturbations=[];responses=[]
    for q in plan['active_groups']:
        for lda,ldb in plan['strides']:
            for trace in plan['traces']:
                rows=[by_formal[seed,q,lda,ldb,trace] for seed in plan['seeds']];windows={}
                for metric in ('cycles_median','ns_median'):
                    values=[r['metrics'][metric] for r in rows];median=statistics.median(values)
                    require(median>0,'完整窗须为正数');spread=(max(values)-min(values))/median
                    windows[metric]=dict(values=values,median=median,relative_spread=spread,passed=spread<=plan['window_relative_spread'])
                    if q==4 and trace==0:
                        old=plan['control_windows'][str(lda)][metric];error=abs(median-old)/old
                        controls.append(dict(lda=lda,metric=metric,expected=old,observed=median,
                            relative_error=error,passed=error<=plan['relative_tolerance']))
                prefix=None
                if trace==-1:
                    prefix={}
                    for unit in ('cycles','ns'):
                        values=[r['prefill'][unit] for r in rows];median=statistics.median(values)
                        require(median>0,'prefill窗须为正数');spread=(max(values)-min(values))/median
                        prefix[unit]=dict(values=values,median=median,relative_spread=spread,
                            passed=spread<=plan['prefill_cycle_relative_spread'] if unit=='cycles' else None)
                        if q==4 and unit=='cycles':
                            old=plan['control_prefill_cycles'][str(lda)];error=abs(median-old)/old
                            prefix_controls.append(dict(lda=lda,expected=old,observed=median,
                                relative_error=error,passed=error<=plan['relative_tolerance']))
                conditions.append(dict(active_groups=q,lda=lda,ldb=ldb,trace=trace,windows=windows,prefill=prefix,
                    passed=all(w['passed'] for w in windows.values()) and (trace==0 or prefix['cycles']['passed'])))
            for seed in plan['seeds']:
                plain=by_formal[seed,q,lda,ldb,0];sparse=by_formal[seed,q,lda,ldb,-1]
                for metric in ('cycles_median','ns_median'):
                    p=plain['metrics'][metric];s=sparse['metrics'][metric];error=abs(s-p)/p
                    perturbations.append(dict(seed=seed,active_groups=q,lda=lda,metric=metric,
                        relative_difference=error,passed=error<=plan['perturbation_tolerance']))
        prefix_response={};full_response={}
        for unit,metric in (('cycles','cycles_median'),('ns','ns_median')):
            values=[by_formal[seed,q,4104,2304,-1]['prefill'][unit]-
                by_formal[seed,q,4096,2304,-1]['prefill'][unit] for seed in plan['seeds']]
            prefix_response[unit]=dict(values=values,median=statistics.median(values));full_response[unit]={}
            for trace in plan['traces']:
                values=[by_formal[seed,q,4104,2304,trace]['metrics'][metric]-
                    by_formal[seed,q,4096,2304,trace]['metrics'][metric] for seed in plan['seeds']]
                full_response[unit][str(trace)]=dict(values=values,median=statistics.median(values))
        responses.append(dict(active_groups=q,prefill_A16=prefix_response,full_window_A16=full_response))
    by_q={r['active_groups']:r for r in responses};base=by_q[4]['prefill_A16']['cycles']['median']
    expected_effect=plan['control_prefill_effect_cycles'];error=abs(base-expected_effect)/expected_effect
    effect=dict(values=by_q[4]['prefill_A16']['cycles']['values'],median=base,expected=expected_effect,
        relative_error=error,passed=error<=plan['relative_tolerance'])
    control_passed=all(r['passed'] for r in controls+prefix_controls+[effect])
    stable=all(r['passed'] for r in conditions);perturbation_passed=all(r['passed'] for r in perturbations)
    qualified=control_passed and stable and perturbation_passed
    ratio=by_q[2]['prefill_A16']['cycles']['median']/base if base else None
    matches=[c['name'] for c in plan['candidates'] if ratio is not None and abs(ratio-c['ratio'])<=plan['ratio_absolute_tolerance']]
    classification=matches[0] if len(matches)==1 else 'all_forms_failed' if ratio is not None else 'information_insufficient'
    decision='information_insufficient' if not qualified or ratio is None else \
        'retain_program_response_form' if matches else 'stop_all_three_forms'
    return dict(schema='gh200-q02-prefill-active-score/1',qualification=plan['qualification'],decision=decision,
        classification=classification,ratio_cycles=ratio,independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),numeric_checked_values=sum(r['checked_values'] for r in numeric),
        numeric=numeric,formal=formal,conditions=conditions,responses=responses,
        control=dict(windows=controls,prefill_cycles=prefix_controls,prefill_effect_cycles=effect,passed=control_passed),
        perturbations=perturbations,perturbation_passed=perturbation_passed,stable_windows_and_prefill=stable,
        observation_qualified=qualified,prefill_ns_report_only=True,signed_or_zero_delta_allowed=True,no_bare_price=True,
        actual_prefill_paths=plan['actual_prefill_paths'],no_physical_capacity_inference=True)


def summarize(root):
    plan=load_plan(root,True);approved_target(root,plan)
    require(read_json(root/'allocation/identity.json')['uuid']==plan['gpu_uuid'],'分配卡身份不符')
    numeric=[];formal=[]
    for filename,split,rows in (('correctness_commands.json','correctness',numeric),('commands.json','formal',formal)):
        for task in read_json(root/filename):
            rows.extend(observer.validate_process(root,root/'runs'/split/task['id'],task,plan['gpu_uuid']))
    result=summarize_rows(plan,numeric,formal)
    result.update(plan_sha256=digest(root/'plan.json'),probe_sha256=digest(root/'build/probe'))
    return result


def execute(root):
    plan=load_plan(root,True);approved_target(root,plan);start=time.monotonic()
    environment=inspect_allocated_gpu(root/'allocation');require(environment['uuid']==plan['gpu_uuid'],'分配卡不是冻结参考卡')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        for filename,split in (('correctness_commands.json','correctness'),('commands.json','formal')):
            for task in read_json(root/filename):
                folder=root/'runs'/split/task['id']
                command_record(folder,observer.stride.probe_command(root,task),observer.stride.remaining(start,60))
                save_json(folder/'validated.json',observer.validate_process(root,folder,task,environment['uuid']))
                print(json.dumps(dict(split=split,process=task['id'],arms=4)),flush=True)
        save_json(root/'score.json',summarize(root))
    print('CLOSED:6 verify+6 formal processes;24 numeric+24 formal arms; no compilation',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('prepare','execute','score'):
        p=sub.add_parser(mode);p.add_argument('--root',type=Path,required=True)
        if mode=='prepare':p.add_argument('--reference',type=Path,required=True)
        if mode=='score':p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.mode=='prepare':prepare(root,args.reference.resolve())
    elif args.mode=='execute':execute(root)
    else:save_json(args.output,summarize(root))


if __name__=='__main__':main()

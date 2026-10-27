#!/usr/bin/env python3
"""固定分配内的 A 行距分支判别：CPU 构建、目标执行与原始记录评分。"""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import statistics
import time

import q02_tma_stride as stride
from l2_saturation import command_record, digest, exclusive_lock, inspect_allocated_gpu, read_json, require, save_json

HERE = Path(__file__).resolve().parent
SCHEMA = 'gh200-q02-a-stride/1'
PAIRS = [[4096,2304],[4096,2320],[4104,2304],[4104,2320]]


def prepare(root, baseline, join):
    root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir(); (root/'source-before').mkdir()
    (root/'reference/granule').mkdir(parents=True); (root/'reference/join').mkdir()
    for name in ('q02_a_stride.py','q02_tma_stride.py','l2_saturation.py'):
        shutil.copy2(HERE/name, root/'source'/name)
    shutil.copy2(HERE/'probes/tma_gemm_sources.cu', root/'source/tma_gemm_sources.cu')
    for name in ('r00_common.hpp','cutlass-3.9.2-include.tar.gz'):
        shutil.copy2(baseline/'source'/name, root/'source'/name)
    shutil.copy2(baseline/'source/tma_gemm_sources.cu',root/'source-before/tma_gemm_sources.cu')
    shutil.copy2(baseline/'worklist.txt',root/'worklist.txt')
    for name in ('score.json','binding.json'):
        shutil.copy2(baseline/name, root/'reference/granule'/name)
        shutil.copy2(join/name, root/'reference/join'/name)
    shutil.copy2(baseline/'build/target-review.json',root/'reference/granule/target-review.json')
    for seed in (3,22,41):
        shutil.copy2(baseline/f'runs/formal/q1-four-ldb-seed{seed}/stdout',
                     root/f'reference/granule/formal-seed{seed}.ndjson')
    original=read_json(baseline/'commands.json')
    for filename,verify in (('commands.json',0),('correctness_commands.json',1)):
        tasks=[]
        for old in original:
            task=copy.deepcopy(old); seed=task['seed']
            pairs=PAIRS if seed==3 else list(reversed(PAIRS))
            task['id']=f'q1-a-b-strides-seed{seed}'; task.pop('paired_ldbs')
            task['paired_strides']=pairs; task['role']='control_and_frozen_intervention'
            argv=task['argv']; i=argv.index('--paired-ldbs')
            argv[i:i+2]=['--paired-strides',','.join(f'{a}:{b}' for a,b in pairs)]
            argv[argv.index('--verify')+1]=str(verify)
            argv+=['--lda','4096','--alloc-lda','4104']
            if verify: task['expected_checked']=524288
            tasks.append(task)
        save_json(root/filename,tasks)
    prior=read_json(baseline/'score.json')
    controls={str(c['ldb']):{metric:w['median'] for metric,w in c['windows'].items()}
              for c in prior['conditions'] if c['ldb'] in (2304,2320)}
    require(controls=={'2304':{'cycles_median':30876,'ns_median':15584},
                      '2320':{'cycles_median':32878,'ns_median':16608}},'752126控制值不符')
    plan=dict(schema=SCHEMA,qualification='frozen_A_stride_branch_discrimination',
        independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        gpu_uuid=read_json(baseline/'plan.json')['gpu_uuid'],required_smid=124,seeds=[3,22,41],
        pairs=PAIRS,control_pairs=PAIRS[:2],heldout_pairs=PAIRS[2:],ctas=1,groups=32,active_groups=1,
        m=2304,n=2304,k=4096,alloc_lda=4104,alloc_ldb=2360,ring=64,physical_slots=4,
        warmup_groups=64,warmup_active_groups=4,warmup_calls=10,wait_hint_ns=10000000,
        source_rounds=[0,1],correctness_processes=3,formal_processes=3,
        numeric_arm_checks=12,formal_arm_observations=12,numeric_checked_values=6291456,
        allocation_budget_seconds=600,maximum_gpu_seconds=3600,maximum_repairs=2,automatic_retries=False,
        control_windows=controls,control_delta_cycles=2002,relative_tolerance=.05,window_relative_spread=.05,
        response_ratio_absolute_tolerance=.05,cycle_primary=True,ns_response_report_only=True,
        response='R=median(delta_B32 at lda4104)/median(delta_B32 at lda4096)',
        proof=dict(source='reference/join/score.json',acceptance='abs(observed-predicted)/predicted<=1/20',
            b_lower='11255/7296',C_upper='171880/57',A16_over_B32_lower='624255/133',
            derivation=['positive delta32 puts B32/B80/B112 on B branch',
                'C<=T32/(1-t)-b_lower*WB32',
                '1.375*A-b*WB32>=1.375*T0/(1+t)-T32/(1-t)-.375*C_upper>0']),
        no_parameter_fit=True,original_probe_sha256=read_json(baseline/'plan.json')['original_probe_sha256'],
        reference_score_sha256=digest(baseline/'score.json'),reference_binding_sha256=digest(baseline/'binding.json'),
        frozen_at_ns=time.time_ns())
    save_json(root/'plan.json',plan)
    save_json(root/'hypotheses.json',[
        dict(id='A-independent-branch',rule='不变分支及B服务不受A行距改变，B32增量保持',R=1),
        dict(id='A-coverage-scaled-branch',rule='整个不变分支随A64B覆盖16到22KiB同比放大，遮住B32分支',R=0)])
    (root/'README.md').write_text('''# Q02：A 真实行距的分支判别

固定 M/N2304、逻辑 K4096、A16+B8+B8/SW128、物理4槽及ring64。同一进程四臂为 A lda4096/4104 × B ldb2304/2320；固定allocA4104、allocB2360。seed3正序，22/41全逆序。每臂由同一个A初始化kernel按实际lda填逻辑数据，其余分配填65504；Q4/G64交替两工作坐标暖机10次，正式Q1/G32。先3verify再3plain，共24臂，数值核6291456值。

此前七项5%约束按abs(observed−predicted)/predicted计算。对全部相容非负A、b、C：b≥11255/7296；C≤171880/57；A原支活动，且1.375A−b·WB32≥624255/133≈4693.647>0。因此整个不变分支随A覆盖同比放大时，新B32增量为0；不受A行距影响时保留原增量。R为新旧A行距的B32配对增量中位之比，冻结预测0/1，绝对误差各≤0.05。cycle主判定，ns同列报告。参数与证明来源在reference，未将整窗参数接入GEMM。

A0两cycle/ns完整窗口及2002cycle配对增量各须复现752126至5%；四条件三进程窗口极差≤各自中位5%；全值、SM124、UUID、同进程固定分配通过。控制失败信息不足，不重采。两候选均失败则停止，不调价或新增候选。完整GEMM独立C2/C3保持未验收。

```sh
python source/q02_a_stride.py build-only --root "$PWD"
python source/q02_a_stride.py execute --root "$PWD"
python source/q02_a_stride.py score --root "$PWD" --output /tmp/q02-a-stride-score.json
```

build-only只使用CUDA12.9.41进行before/after构建，复用stride.build核tensor_probe全部机器字与资源，并与752126原kernel核对；不检查或启动GPU。主线程审核实际构建后写approved-target.json并冻结最终binding，execute只使用该已批准binary。预算10分钟，最多两次定向实现修复，无自动重试。CPU准备未执行编译或GPU采样。
''')
    names=[str(p.relative_to(root)) for directory in ('source','source-before') for p in (root/directory).iterdir() if p.is_file()]
    names+=['plan.json','commands.json','correctness_commands.json','hypotheses.json','worklist.txt','README.md']
    names += [str(p.relative_to(root)) for p in (root/'reference').rglob('*') if p.is_file()]
    save_json(root/'build-input-binding.json',dict(schema='gh200-q02-a-stride-build-input/1',
        files={name:digest(root/name) for name in sorted(names)}))
    print('PREPARED: four paired strides; CPU build input frozen; no compilation/GPU',flush=True)


def load_plan(root, execution=False):
    name='binding.json' if execution else 'build-input-binding.json'
    for path,checksum in read_json(root/name)['files'].items():
        require(digest(root/path)==checksum,'冻结身份变更：'+path)
    plan=read_json(root/'plan.json'); require(plan['schema']==SCHEMA,'执行范围不符')
    return plan


def build_only(root):
    load_plan(root); start=time.monotonic(); stride.build(root,start)
    report=read_json(root/'build/target-review.json')
    require('V12.9.41' in report['compiler_version'] and 'compiler.35813241_0' in report['compiler_version'],
            '需要冻结CUDA12.9.41')
    original=read_json(root/'reference/granule/target-review.json')['rows'][1]
    require(report['passed'] and all(row['instruction_count']==1264 and
        row['instruction_words_sha256']==original['instruction_words_sha256'] and
        row['resources']==original['resources'] for row in report['rows']),
        'tensor_probe完整机器字/资源与752126不符')
    print('BUILT ONLY: before/after and original full kernel identical; root approval pending',flush=True)


def validate_process(root,folder,task,uuid):
    receipt=stride.successful_receipt(folder)
    command=receipt['argv'][:];probe=Path(command[0])
    require(probe.name=='probe' and probe.parent.name=='build','binary路径不符')
    require(command[command.index('--worklist')+1]==str(probe.parent.parent/'worklist.txt'),'工作列表路径不符')
    command[0]=task['argv'][0];command[command.index('--worklist')+1]='worklist.txt'
    require(command==task['argv'],'执行参数与冻结命令不符')
    raw=[json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.strip()]
    require(len(raw)==len(task['paired_strides']),'四臂缺失')
    arguments=dict(zip(task['argv'][1::2],task['argv'][2::2]))
    fields={'ctas':'--ctas','active_groups':'--measure-active','groups':'--measure-groups','ring':'--ring',
        'verify':'--verify','trace':'--trace','seed':'--seed','warmup':'--warmup',
        'source_phase_stride':'--source-phase-stride','source_tile_shift':'--source-tile-shift',
        'work_index':'--work-index','warmup_work_offset':'--warmup-work-offset',
        'release_delay_ns':'--release-delay-ns','pollute_kib':'--pollute-kib','launch_skew_ns':'--launch-skew-ns',
        'warmup_skew_ns':'--warmup-skew-ns','alloc_lda':'--alloc-lda','alloc_ldb':'--alloc-ldb'}
    require((root/'worklist.txt').read_text().split()==['2304','2304','4096','1','0','0','7','6'],'工作列表不符')
    checked=[]
    for index,row in enumerate(raw):
        require(row['schema']=='cfg-b-gemm-source-probe/1' and row['gpu_uuid']==uuid,'schema/卡身份不符')
        require(all(row[name]==int(arguments[flag]) for name,flag in fields.items()),'逐臂参数不符')
        require(row['warmup_active_groups']==4 and row['warmup_groups']==64 and
            row['warmup_source_phase_stride']==0 and row['record_warmup']==0 and
            not row['warmup_ctas'] and not row['group_events'],'暖机或trace范围不符')
        require(row['paired_strides']==task['paired_strides'] and row['pair_index']==index and
            [row['lda'],row['ldb']]==task['paired_strides'][index],'逐臂stride或次序不符')
        require(all(row[name]==raw[0][name] for name in stride.ADDRESSES),'同进程地址变化')
        require(row['source_a_address']>0 and row['source_b_address']>0 and
            row['source_a_address']%128==row['source_b_address']%128==0,'源地址/对齐不符')
        require(row['m']==row['n']==2304 and row['allocated_a_elements']==9455616 and
            row['allocated_b_elements']==9666560,'形状或固定分配不符')
        require(row['errors']==row['pollution_errors']==row['pollution_checked']==0,'加载/污染检查错误')
        require(row['checked']==(524288 if row['verify'] else 0),'完整加载检查量不符')
        require(row['observed_sms']==row['max_active_ctas_per_sm']==1 and row['smem']==132096 and
            row['local_bytes']==0,'驻留或资源不符')
        resource=read_json(root/'build/target-review.json')['rows'][1]['resources']
        require(row['registers']==int(re.search(r'\bREG:(\d+)\b',resource)[1]),'运行寄存器与构建不符')
        require(len(row['ctas_data'])==1,'CTA数量不符');cta=row['ctas_data'][0]
        require(cta['cta']==0 and cta['tile']==[7,6] and cta['smid']==124,'CTA/坐标/SM124不符')
        for unit in (0,1):
            require(type(cta['begin'][unit]) is int and type(cta['end'][unit]) is int and
                0<cta['begin'][unit]<cta['end'][unit],'完成戳无效')
        cycles=cta['end'][0]-cta['begin'][0];ns=cta['end'][1]-cta['begin'][1]
        checked.append(dict(seed=row['seed'],lda=row['lda'],ldb=row['ldb'],pair_index=index,
            checked_values=row['checked'],numerical_qualified=bool(row['verify']),
            stdout_sha256=receipt['stdout_sha256'],source=str(folder.relative_to(root)),device_uuid=uuid,
            metrics=dict(cycles_median=cycles,ns_median=ns,ns_envelope=ns)))
    return checked


def summarize_rows(plan,numeric,formal,hypotheses):
    by_numeric={(r['seed'],r['lda'],r['ldb']):r for r in numeric}
    by_formal={(r['seed'],r['lda'],r['ldb']):r for r in formal}
    require(len(by_numeric)==len(by_formal)==12 and by_numeric.keys()==by_formal.keys(),'数值/正式12臂不对应')
    conditions=[];control_windows=[];pairs=[];responses=[]
    for lda,ldb in plan['pairs']:
        windows={}
        for metric in ('cycles_median','ns_median'):
            values=[by_formal[(seed,lda,ldb)]['metrics'][metric] for seed in plan['seeds']]
            median=statistics.median(values);spread=max(values)-min(values)
            windows[metric]=dict(values=values,median=median,relative_spread=spread/median,
                passed=spread<=plan['window_relative_spread']*median)
            if lda==4096:
                expected=plan['control_windows'][str(ldb)][metric];error=abs(median-expected)/expected
                control_windows.append(dict(lda=lda,ldb=ldb,metric=metric,expected=expected,observed=median,
                    relative_error=error,passed=error<=plan['relative_tolerance']))
        conditions.append(dict(lda=lda,ldb=ldb,windows=windows,passed=all(v['passed'] for v in windows.values())))
    for lda in (4096,4104):
        deltas={metric:[] for metric in ('cycles_median','ns_median')}
        for seed in plan['seeds']:
            a,b=[by_formal[(seed,lda,ldb)] for ldb in (2304,2320)]
            difference={metric:b['metrics'][metric]-a['metrics'][metric] for metric in deltas}
            for metric,value in difference.items():deltas[metric].append(value)
            pairs.append(dict(seed=seed,lda=lda,differences=difference))
        responses.append(dict(lda=lda,**{metric:dict(values=values,median=statistics.median(values)) for metric,values in deltas.items()}))
    old,new=responses;control_delta=old['cycles_median']['median']
    error=abs(control_delta-plan['control_delta_cycles'])/plan['control_delta_cycles']
    stable=all(c['passed'] for c in conditions)
    control_passed=control_delta>0 and all(c['passed'] for c in control_windows) and error<=plan['relative_tolerance']
    qualified=stable and control_passed
    ratio=new['cycles_median']['median']/control_delta if control_delta else None
    old_ns=old['ns_median']['median'];ratio_ns=new['ns_median']['median']/old_ns if old_ns else None
    candidates=[]
    for hypothesis in hypotheses:
        distance=abs(ratio-hypothesis['R']) if ratio is not None else None
        passed=qualified and distance<=plan['response_ratio_absolute_tolerance']
        candidates.append(dict(id=hypothesis['id'],predicted_R=hypothesis['R'],observed_R=ratio,
            absolute_error=distance,passed=passed,conclusion='信息不足' if not qualified else '获得支持' if passed else '被反证'))
    supported=[c['id'] for c in candidates if c['passed']]
    decision='information_insufficient' if not qualified else 'supported_'+supported[0] if len(supported)==1 else 'both_fixed_forms_rejected' if not supported else 'indistinguishable'
    return dict(schema='gh200-q02-a-stride-score/1',qualification=plan['qualification'],decision=decision,
        independent_C2=qualified and len(supported)==1,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),
        numeric_checked_values=sum(r['checked_values'] for r in numeric),numeric=numeric,formal=formal,
        conditions=conditions,pairs=pairs,responses=responses,R_cycles=ratio,R_ns=ratio_ns,stable_windows=stable,
        control=dict(windows=control_windows,expected_delta_cycles=plan['control_delta_cycles'],
            observed_delta_cycles=control_delta,delta_relative_error=error,passed=control_passed),candidates=candidates)


def approved_target(root):
    approved=read_json(root/'approved-target.json');review=stride.build_review(root)
    require(review==read_json(root/'build/target-review.json') and review['passed'],'构建证据不符')
    require(approved['approved'] is True and approved['probe_sha256']==digest(root/'build/probe') and
        approved['instruction_words_sha256']==review['rows'][1]['instruction_words_sha256'] and
        approved['resources']==review['rows'][1]['resources'],'目标binary未获完整核准')


def summarize(root):
    plan=load_plan(root,True);approved_target(root)
    require(read_json(root/'allocation/identity.json')['uuid']==plan['gpu_uuid'],'分配卡身份不符')
    numeric=[];formal=[]
    for filename,split,rows in (('correctness_commands.json','correctness',numeric),('commands.json','formal',formal)):
        for task in read_json(root/filename):rows.extend(validate_process(root,root/'runs'/split/task['id'],task,plan['gpu_uuid']))
    score=summarize_rows(plan,numeric,formal,read_json(root/'hypotheses.json'))
    score.update(plan_sha256=digest(root/'plan.json'),probe_sha256=digest(root/'build/probe'))
    return score


def execute(root):
    plan=load_plan(root,True);approved_target(root);start=time.monotonic()
    environment=inspect_allocated_gpu(root/'allocation')
    require(environment['uuid']==plan['gpu_uuid'],'分配卡不是冻结参考卡')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        for filename,split in (('correctness_commands.json','correctness'),('commands.json','formal')):
            for task in read_json(root/filename):
                folder=root/'runs'/split/task['id']
                command_record(folder,stride.probe_command(root,task),stride.remaining(start,60))
                save_json(folder/'validated.json',validate_process(root,folder,task,environment['uuid']))
                print(json.dumps(dict(split=split,process=task['id'],arms=4)),flush=True)
        save_json(root/'score.json',summarize(root))
    print('CLOSED: 12 numeric +12 plain arms; frozen A-stride branch response',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','build-only','execute','score'):
        item=sub.add_parser(name);item.add_argument('--root',type=Path,required=True)
        if name=='prepare':
            item.add_argument('--baseline',type=Path,required=True);item.add_argument('--join',type=Path,required=True)
        if name=='score':item.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='prepare':prepare(root,args.baseline.resolve(),args.join.resolve())
    elif args.command=='build-only':build_only(root)
    elif args.command=='execute':execute(root)
    else:save_json(args.output,summarize(root));print('VERIFIED: four paired strides and frozen R predictions')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""同源集合、同槽/phase的Ktile排列：准备、CPU构建与有限开发评分。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import time

import q02_tma_stride as stride
from l2_saturation import command_record,digest,exclusive_lock,inspect_allocated_gpu,read_json,require,save_json

HERE=Path(__file__).resolve().parent
SCHEMA='gh200-q02-source-permutation/1'
SEEDS=(3,22,41)
METRICS=('cycles_median','ns_median')
CONTROL={4096:(21819,10944),4104:(22317,11200)}


def prepare(root,reference):
    plan=read_json(root/'plan.json');require(plan['schema']==SCHEMA,'登记范围不符')
    registration=read_json(root/'registration-binding.json')
    for name,checksum in registration.items():require(digest(root/name)==checksum,'原登记身份变化')
    require(read_json(reference/'score.json')['observation_qualified'] is True,'752255参考资格未通过')
    for directory in ('source','source-before','reference'):(root/directory).mkdir(exist_ok=True)
    shutil.copy2(HERE/'q02_source_permutation.py',root/'source/q02_source_permutation.py')
    for name in ('q02_tma_stride.py','l2_saturation.py','r00_common.hpp','cutlass-3.9.2-include.tar.gz'):
        shutil.copy2(reference/'source'/name,root/'source'/name)
    original=(reference/'source/tma_gemm_sources.cu').read_text()
    changed=original.replace('(step!=1 && step!=2)','(step!=1 && step!=2 && step!=57)').replace(
        '(source_k_step!=1 && source_k_step!=2)','(source_k_step!=1 && source_k_step!=2 && source_k_step!=57)')
    require(changed!=original and changed.count('&& step!=57')==1 and changed.count('&& source_k_step!=57')==1,'host两处值域不符')
    (root/'source-before/tma_gemm_sources.cu').write_text(original)
    (root/'source/tma_gemm_sources.cu').write_text(changed)
    shutil.copy2(reference/'worklist.txt',root/'worklist.txt')
    for name in ('score.json','binding.json','plan.json','approved-target.json'):
        shutil.copy2(reference/name,root/'reference'/name)
    for name in ('kernel-code.json','target-review.json'):
        shutil.copy2(reference/'build'/name,root/'reference'/name)
    shutil.copy2(reference/'build/after/disassemble/stdout',root/'reference/full-sass.txt')
    shutil.copy2(reference/'build/after/resources/stdout',root/'reference/full-resources.txt')
    origins=[]
    for filename,verify,split in (('commands.json',0,'formal'),('correctness_commands.json',1,'correctness')):
        tasks=[]
        for old in read_json(reference/filename):
            seed=old['seed'];source=reference/'runs'/split/old['id']/'stdout'
            selected=[line for line in source.read_text().splitlines() if (lambda r:r['source_k_step']==1 and r['trace']==0)(json.loads(line))]
            require(len(selected)==2,'参考两个step1 plain臂缺失')
            (root/f'reference/{split}-seed{seed}.ndjson').write_text('\n'.join(selected)+'\n')
            origins.append(dict(target=f'reference/{split}-seed{seed}.ndjson',source=str(source),source_sha256=digest(source)))
            for q,g in ((1,64),(4,32)):
                pairs=[(step,lda) for step in ((1,57) if q==1 else (1,)) for lda in (4096,4104)]
                if seed!=3:pairs.reverse()
                task=copy.deepcopy(old)
                task.update(id=f'q{q}-g{g}-permutation-seed{seed}',active_groups=q,groups=g,
                    role='development_permutation' if q==1 else 'bridge',pair_indices=list(range(len(pairs))),
                    paired_strides=[[lda,2304] for step,lda in pairs],paired_source_steps=[step for step,lda in pairs],paired_traces=[0])
                argv=task['argv']
                for flag,value in (('--verify',str(verify)),('--measure-active',str(q)),('--measure-groups',str(g)),
                        ('--paired-traces','0'),('--paired-source-steps',','.join(map(str,task['paired_source_steps']))),
                        ('--paired-strides',','.join(f'{lda}:2304' for step,lda in pairs))):
                    argv[argv.index(flag)+1]=value
                if verify:task['expected_checked']=g*16384
                else:task.pop('expected_checked',None)
                tasks.append(task)
        require(len(tasks)==6,'每类应为6进程');save_json(root/filename,tasks)
    save_json(root/'reference/raw-origins.json',origins)
    (root/'README.md').write_text('''# Q02：同源集合的Ktile排列

Q1/G64的step1/57×A lda4096/4104四条件，加Q4/G32/step1两条件桥接，共6开发条件。step57与64互质且57≡1(mod8)，两排列都覆盖64个Ktile，保留(K,slot,phase)、首K0和63次refill；最后源K由63变7。逐请求行区间与关联的CPU证明见cpu-geometry。A16+B8+B8、B ldb2304、SW128、四槽、barrier、固定allocA4104/allocB2360、tile[7,6]及SM124均复用。

每seed每任务同进程配对，seed3正序22/41逆序。每臂重新初始化同一分配，Q4/G64/step1/trace−1暖机10次，正式trace0。先6verify后6formal进程，共18数值臂和18正式臂，检查15728640值。Q1每臂1048576值，桥接每臂524288值。

先核全部7设备函数before/after/752253的128bit机器字与资源精确相同；tensor_probe另核REG48、18静态UTMA、无stack/local。执行使用根代理核准并绑定的binary，不再编译。六条件cycle/ns三进程范围≤中位5%；桥接两窗21819/22317cycle、10944/11200ns及同seed配对增量498cycle/256ns各复现5%。资格全过后，分别用cycle/ns算R=median_seed(ΔA16_step57)/median_seed(ΔA16_step1)，唯一冻结预测为1±.05；两单位冲突、零分母或资格失败为信息不足。双单位偏离停止该排列不变可加形式，命中只保留本次排列相容性。末K和跨组先后仍是不同解释，本轮不加服务价格或替换默认组合，独立C2/C3与完整GEMM验收均false。

```sh
python -B cpu-check.py --root "$PWD" --output /tmp/q02-permutation-cpu-check.json
python -B source/q02_source_permutation.py build-only --root "$PWD"
python -B source/q02_source_permutation.py execute --root "$PWD"
python -B source/q02_source_permutation.py score --root "$PWD" --output /tmp/q02-permutation-score.json
```

CPU-only ROMEO构建使用CUDA12.9.41、0GPU。正式分配10分钟、GPU绝对上限1小时、最多两次定向实现修复，不自动重采。plan.json为主线程原登记，prepare不改写。build-input-binding冻结准备输入，最终binding和approved-target由主线程核准。
''')
    print('PREPARED: 6 conditions, 6 verify +6 formal processes, no build/GPU',flush=True)


def load_plan(root,execution=False):
    binding=read_json(root/('binding.json' if execution else 'build-input-binding.json'))
    for name,checksum in binding['files'].items():require(digest(root/name)==checksum,'冻结身份变更：'+name)
    plan=read_json(root/'plan.json');require(plan['schema']==SCHEMA,'执行范围不符')
    return plan


def kernel_records(sass):
    records={}
    for section in re.split(r'Function\s*:\s*',sass)[1:]:
        symbol=section.splitlines()[0].strip();code=[]
        for line in section.splitlines()[1:]:
            match=re.match(r'\s*/\*([0-9a-f]+)\*/\s*(.*?)\s*/\* 0x([0-9a-f]+) \*/',line)
            if match:code.append(dict(pc=int(match[1],16),op=match[2].strip(),words=[int(match[3],16)]))
            else:
                word=re.search(r'/\* 0x([0-9a-f]+) \*/',line)
                if word and code:code[-1]['words'].append(int(word[1],16))
        require(code and all(len(row['words'])==2 for row in code),'完整两机器字缺失')
        require(symbol not in records,'设备入口重复');records[symbol]=code
    require(len(records)==7,'7个设备函数缺失')
    return records


def resources(text):
    return {match[1]:match[2].strip() for match in re.finditer(r'Function (\S+):\s*\n\s*([^\n]+)',text)}


def review_build(root):
    base=stride.build_review(root)
    records={label:kernel_records((root/f'build/{label}/disassemble/stdout').read_text()) for label in ('before','after')}
    records['reference']=kernel_records((root/'reference/full-sass.txt').read_text())
    usages={label:resources((root/f'build/{label}/resources/stdout').read_text()) for label in ('before','after')}
    usages['reference']=resources((root/'reference/full-resources.txt').read_text())
    require(all(set(usages[label])==set(records[label]) for label in records),'设备函数资源不齐')
    version=(root/'build/version/stdout').read_text()
    exact=records['before']==records['after']==records['reference'] and usages['before']==usages['after']==usages['reference']
    tensor=next(symbol for symbol in records['after'] if symbol.startswith('_Z12tensor_probe'))
    passed=base['passed'] and exact and 'V12.9.41' in version and 'compiler.35813241_0' in version and bool(re.search(r'\bREG:48\b',usages['after'][tensor]))
    return dict(status='pending_root_full_device_code_approval',approved=False,passed=passed,
        all_seven_code_and_resources_exact=exact,functions=[dict(symbol=symbol,instructions=len(code),resources=usages['after'][symbol],
            machine_words_sha256=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()) for symbol,code in records['after'].items()],
        rows=base['rows'],compiler_version=version,probe_sha256=digest(root/'build/probe'),
        kernel_code_sha256=digest(root/'build/kernel-code.json'),plan_sha256=digest(root/'plan.json'),
        probe_source_sha256=digest(root/'source/tma_gemm_sources.cu'),reference_sass_sha256=digest(root/'reference/full-sass.txt'))


def build_only(root):
    load_plan(root);stride.build(root,time.monotonic())
    save_json(root/'build/kernel-code.json',kernel_records((root/'build/after/disassemble/stdout').read_text()))
    review=review_build(root);save_json(root/'build/all-kernel-review.json',review)
    require(review['passed'],'全部7设备函数机器字/资源或编译器门槛失败')
    print('BUILT ONLY: 7 functions exact; root approval pending; no GPU',flush=True)


def approved_target(root):
    actual=review_build(root);saved=read_json(root/'build/all-kernel-review.json');approved=read_json(root/'approved-target.json')
    require(actual==saved and actual['passed'],'完整构建证据不符')
    require(approved['approved'] is True and all(approved[key]==actual[key] for key in
        ('probe_sha256','kernel_code_sha256','plan_sha256','probe_source_sha256')) and
        approved['target_review_sha256']==digest(root/'build/all-kernel-review.json'),'目标未获核准')


def validate_process(root,folder,task,uuid):
    receipt=stride.successful_receipt(folder);command=receipt['argv'][:];probe=Path(command[0])
    require(probe.name=='probe' and probe.parent.name=='build','binary路径不符')
    require(command[command.index('--worklist')+1]==str(probe.parent.parent/'worklist.txt'),'工作列表路径不符')
    command[0]=task['argv'][0];command[command.index('--worklist')+1]='worklist.txt'
    require(command==task['argv'],'执行参数与冻结命令不符')
    raw=[json.loads(line) for line in (folder/'stdout').read_text().splitlines() if line.strip()]
    require(len(raw)==len(task['paired_strides']),'配对臂数不符')
    arguments=dict(zip(task['argv'][1::2],task['argv'][2::2]))
    fields={'ctas':'--ctas','active_groups':'--measure-active','groups':'--measure-groups','ring':'--ring',
        'verify':'--verify','trace':'--trace','seed':'--seed','warmup':'--warmup','source_phase_stride':'--source-phase-stride',
        'source_tile_shift':'--source-tile-shift','work_index':'--work-index','warmup_work_offset':'--warmup-work-offset',
        'release_delay_ns':'--release-delay-ns','pollute_kib':'--pollute-kib','launch_skew_ns':'--launch-skew-ns',
        'warmup_skew_ns':'--warmup-skew-ns','alloc_lda':'--alloc-lda','alloc_ldb':'--alloc-ldb'}
    require((root/'worklist.txt').read_text().split()==['2304','2304','4096','1','0','0','7','6'],'工作列表不符')
    checked=[];descriptors={};resource=read_json(root/'build/target-review.json')['rows'][1]['resources']
    for index,row in enumerate(raw):
        require(row['schema']=='cfg-b-gemm-source-probe/1' and row['gpu_uuid']==uuid,'schema/卡身份不符')
        require(all(row[name]==int(arguments[flag]) for name,flag in fields.items()),'逐臂参数不符')
        require(row['warmup_trace']==-1 and row['warmup_active_groups']==4 and row['warmup_groups']==64 and
            row['warmup_source_phase_stride']==0 and row['record_warmup']==0 and not row['warmup_ctas'] and not row['group_events'],'暖机或trace范围不符')
        require(row['paired_strides']==task['paired_strides'] and row['pair_index']==row['stride_index']==index and
            row['trace_index']==0 and row['paired_traces']==[0] and [row['lda'],row['ldb']]==task['paired_strides'][index],
            '逐臂stride/trace或次序不符')
        require(row['paired_source_steps']==task['paired_source_steps'] and row['source_k_step']==task['paired_source_steps'][index] and
            row['warmup_source_k_step']==1,'源K步长/zip或暖机步长不符')
        require(row['request_mask']==row['warmup_request_mask']==7 and row['transaction_bytes']==row['warmup_transaction_bytes']==32768 and
            row['wait_delay_cycles']==row['warmup_wait_delay_cycles']==0,'三请求/事务字节或D0不符')
        pair=(row['descriptor_a'],row['descriptor_b']);key=(row['lda'],row['ldb'])
        require(all(re.fullmatch('[0-9a-f]{256}',value) for value in pair) and descriptors.setdefault(key,pair)==pair,
            '同stride跨step descriptor变化')
        require(all(row[name]==raw[0][name] for name in stride.ADDRESSES),'同进程地址变化')
        require(row['source_a_address']>0 and row['source_b_address']>0 and row['source_a_address']%128==row['source_b_address']%128==0,'源地址/对齐不符')
        require(row['m']==row['n']==2304 and row['allocated_a_elements']==9455616 and row['allocated_b_elements']==9666560,'形状或固定分配不符')
        require(row['errors']==row['pollution_errors']==row['pollution_checked']==0 and row['checked']==(row['groups']*16384 if row['verify'] else 0),'完整加载数值不符')
        require(row['observed_sms']==row['max_active_ctas_per_sm']==1 and row['smem']==132096 and row['local_bytes']==0 and
            row['registers']==int(re.search(r'\bREG:(\d+)\b',resource)[1]),'驻留/构建资源不符')
        require(len(row['ctas_data'])==1,'CTA数量不符');cta=row['ctas_data'][0]
        require(cta['cta']==0 and cta['tile']==[7,6] and cta['smid']==124,'CTA/坐标/SM124不符')
        for unit in (0,1):require(type(cta['begin'][unit]) is int and type(cta['end'][unit]) is int and 0<cta['begin'][unit]<cta['end'][unit],'完成戳无效')
        require('prefill_end' not in cta,'plain错误读取暖机预填戳')
        checked.append(dict(seed=row['seed'],Q=row['active_groups'],G=row['groups'],source_k_step=row['source_k_step'],
            lda=row['lda'],ldb=row['ldb'],pair_index=index,checked_values=row['checked'],numerical_qualified=bool(row['verify']),
            stdout_sha256=receipt['stdout_sha256'],source=str(folder.relative_to(root)),device_uuid=uuid,
            metrics=dict(cycles_median=cta['end'][0]-cta['begin'][0],ns_median=cta['end'][1]-cta['begin'][1])))
    return checked


def summarize_rows(plan,numeric,formal):
    key=lambda r:(r['seed'],r['Q'],r['G'],r['source_k_step'],r['lda'],r['ldb'])
    expected={(s,c['Q'],c['G'],c['step'],c['lda'],c['ldb']) for s in plan['seeds'] for c in plan['conditions']}
    by_numeric={key(r):r for r in numeric};by_formal={key(r):r for r in formal}
    require(len(numeric)==len(formal)==18 and set(by_numeric)==set(by_formal)==expected,'18逐臂条件不齐')
    require(all(r['numerical_qualified'] and r['checked_values']==r['G']*16384 for r in numeric) and
        all(not r['numerical_qualified'] for r in formal) and sum(r['checked_values'] for r in numeric)==plan['numeric_checked_values'],'数值资格不符')
    conditions=[]
    for c in plan['conditions']:
        windows={}
        for metric in METRICS:
            values=[by_formal[(s,c['Q'],c['G'],c['step'],c['lda'],c['ldb'])]['metrics'][metric] for s in plan['seeds']]
            median=statistics.median(values);spread=(max(values)-min(values))/median
            windows[metric]=dict(values=values,median=median,relative_spread=spread,passed=spread<=.05)
        conditions.append(dict(**c,windows=windows,passed=all(w['passed'] for w in windows.values())))
    controls=[]
    for lda in (4096,4104):
        window=next(c for c in conditions if c['Q']==4 and c['lda']==lda)['windows']
        for metric,old in zip(METRICS,CONTROL[lda]):
            observed=window[metric]['median'];error=abs(observed-old)/old
            controls.append(dict(lda=lda,metric=metric,observed=observed,expected=old,relative_error=error,passed=error<=.05))
    responses=[]
    for q,g,step in ((4,32,1),(1,64,1),(1,64,57)):
        deltas={}
        for metric in METRICS:
            values=[by_formal[(s,q,g,step,4104,2304)]['metrics'][metric]-by_formal[(s,q,g,step,4096,2304)]['metrics'][metric] for s in plan['seeds']]
            deltas[metric]=dict(values=values,median=statistics.median(values))
        responses.append(dict(Q=q,G=g,source_k_step=step,A16=deltas))
    effects=[]
    for metric,old in zip(METRICS,(498,256)):
        observed=responses[0]['A16'][metric]['median'];error=abs(observed-old)/old
        effects.append(dict(metric=metric,observed=observed,expected=old,relative_error=error,passed=error<=.05))
    control_passed=all(c['passed'] for c in controls+effects);stable=all(c['passed'] for c in conditions)
    ratios={metric:(responses[2]['A16'][metric]['median']/responses[1]['A16'][metric]['median']
        if responses[1]['A16'][metric]['median'] else None) for metric in METRICS}
    matches={metric:(ratio is not None and abs(ratio-1)<=.05) for metric,ratio in ratios.items()}
    qualified=control_passed and stable;unit_agree=len(set(matches.values()))==1
    sufficient=qualified and all(ratio is not None for ratio in ratios.values()) and unit_agree
    decision='information_insufficient' if not sufficient else 'development_compatible_permutation_invariant_response' if all(matches.values()) else 'stop_permutation_invariant_response'
    return dict(schema='gh200-q02-source-permutation-score/1',qualification='development_permutation',decision=decision,
        classification=decision,independent_C2=False,independent_C3=False,full_gemm_C2=False,full_gemm_C3=False,
        numeric_arm_checks=len(numeric),formal_arm_observations=len(formal),numeric_checked_values=sum(r['checked_values'] for r in numeric),
        numeric=numeric,formal=formal,conditions=conditions,control=dict(windows=controls,effects=effects,passed=control_passed),
        stable_windows=stable,observation_qualified=qualified,responses=responses,ratios=ratios,matches=matches,units_agree=unit_agree)


def summarize(root):
    plan=load_plan(root,True);approved_target(root)
    require(read_json(root/'allocation/identity.json')['uuid']==plan['gpu_uuid'],'分配卡身份不符')
    numeric=[];formal=[]
    for filename,split,rows in (('correctness_commands.json','correctness',numeric),('commands.json','formal',formal)):
        for task in read_json(root/filename):rows.extend(validate_process(root,root/'runs'/split/task['id'],task,plan['gpu_uuid']))
    result=summarize_rows(plan,numeric,formal);result.update(plan_sha256=digest(root/'plan.json'),probe_sha256=digest(root/'build/probe'))
    return result


def execute(root):
    plan=load_plan(root,True);approved_target(root);start=time.monotonic()
    environment=inspect_allocated_gpu(root/'allocation');require(environment['uuid']==plan['gpu_uuid'],'分配卡不是冻结参考卡')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        for filename,split in (('correctness_commands.json','correctness'),('commands.json','formal')):
            for task in read_json(root/filename):
                folder=root/'runs'/split/task['id'];command_record(folder,stride.probe_command(root,task),stride.remaining(start,60))
                save_json(folder/'validated.json',validate_process(root,folder,task,environment['uuid']))
                print(json.dumps(dict(split=split,process=task['id'],arms=len(task['paired_strides']))),flush=True)
        save_json(root/'score.json',summarize(root))
    print('CLOSED: 18 numeric +18 formal arms; finite development permutation check',flush=True)


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

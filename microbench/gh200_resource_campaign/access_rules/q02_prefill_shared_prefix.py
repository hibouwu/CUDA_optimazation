#!/usr/bin/env python3
"""Q2/Q4共享首次预填代码的有限开发轮；复用冻结评分与原始校验。"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

import q02_prefill_active as active
from l2_saturation import command_record,digest,exclusive_lock,inspect_allocated_gpu,read_json,require,save_json

observer=active.observer
HERE=Path(__file__).resolve().parent
SCHEMA='gh200-q02-prefill-shared-prefix/1'
APPROVAL_SCHEMA='gh200-q02-prefill-shared-prefix-approval/1'
OLD_LOOP='    for (int g=0; g<active; ++g) issue_group(g);'
NEW_PREFIX='''    issue_group(0);
    if (active>1) issue_group(1);
    if (active>2) issue_group(2);
    if (active>3) issue_group(3);'''


def load_plan(root,execution=False):
    for name,value in read_json(root/('binding.json' if execution else 'build-input-binding.json'))['files'].items():
        require(digest(root/name)==value,'冻结身份变更：'+name)
    plan=read_json(root/'plan.json');require(plan['schema']==SCHEMA,'执行范围不符')
    return plan


def prepare(root,reference,source_reference):
    old=read_json(reference/'score.json');old_plan=read_json(reference/'plan.json')
    require(old['observation_qualified'] is True and old['ratio_cycles']==0,'752300原资格或零响应不符')
    require(old_plan['control_windows']=={'4096':{'cycles_median':21819,'ns_median':10944},
        '4104':{'cycles_median':22317,'ns_median':11200}} and
        old_plan['control_prefill_cycles']=={'4096':1389,'4104':1887} and
        old_plan['control_prefill_effect_cycles']==498,'原Q4控制不符')
    root.mkdir(parents=True,exist_ok=False)
    for directory in ('source','source-before','reference'):(root/directory).mkdir()
    names=[]
    def snapshot(origin,name):
        destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(origin,destination);names.append(name)
    snapshot(HERE/'q02_prefill_shared_prefix.py','source/q02_prefill_shared_prefix.py')
    binding=read_json(reference/'binding.json')['files']
    for name in ('q02_prefill_active.py','q02_prefill_observer.py','q02_tma_stride.py','l2_saturation.py'):
        require(digest(reference/'source'/name)==binding['source/'+name],'冻结helper不符：'+name)
        snapshot(reference/'source'/name,'source/'+name)
    source_binding=read_json(source_reference/'binding.json')['files']
    for name in ('r00_common.hpp','cutlass-3.9.2-include.tar.gz','tma_gemm_sources.cu'):
        require(digest(source_reference/'source'/name)==source_binding['source/'+name],'752255 source身份不符：'+name)
        snapshot(source_reference/'source'/name,('source-before/' if name.endswith('.cu') else 'source/')+name)
    before=(root/'source-before/tma_gemm_sources.cu').read_text()
    require(before.count(OLD_LOOP)==1,'原首次prefill loop不唯一')
    (root/'source/tma_gemm_sources.cu').write_text(before.replace(OLD_LOOP,NEW_PREFIX,1));names.append('source/tma_gemm_sources.cu')
    for name in ('plan.json','score.json','binding.json','approved-target.json'):
        snapshot(reference/name,'reference/'+name)
    for name in ('probe','kernel-code.json','target-review.json'):
        snapshot(reference/'build'/name,'reference/build/'+name)
    snapshot(reference/'worklist.txt','worklist.txt')
    snapshot(HERE.parents[2]/'Docs/ModelEvaluation/gemm/PLAN.md','reference/campaign-PLAN.md')
    raw=[];raw_sources={}
    for task in read_json(reference/'commands.json'):
        path=reference/'runs/formal'/task['id']/'stdout';raw_sources[str(path.relative_to(reference))]=digest(path)
        raw.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    require(len(raw)==24,'旧24正式臂缺失')
    (root/'reference/formal.ndjson').write_text(''.join(json.dumps(row)+'\n' for row in raw));names.append('reference/formal.ndjson')
    for filename in ('commands.json','correctness_commands.json'):
        tasks=read_json(reference/filename)
        for task in tasks:
            task['id']=task['id'].replace('prefill-active','prefill-shared-prefix')
            task['role']='development_shared_initial_prefix'
        require(len(tasks)==6,'每类须六进程');save_json(root/filename,tasks);names.append(filename)
    plan=copy.deepcopy(old_plan)
    for name in ('reused_binary_identity','reference_raw_sources'):plan.pop(name,None)
    plan.update(schema=SCHEMA,qualification='development_shared_initial_prefix',intervention='shared_initial_prefix',
        compilation=True,candidates=[dict(name='q2_zero_initial_response',ratio=0)],
        actual_prefill_paths={'2':'explicit guarded groups0/1','4':'explicit guarded groups0/1/2/3'},
        program_scope='same first two prefill code groups required by root SASS review; no physical capacity inference',
        comparison=dict(reference_run=reference.name,source_reference_run=source_reference.name,
            reference_plan_sha256=digest(reference/'plan.json'),reference_score_sha256=digest(reference/'score.json'),
            reference_binding_sha256=digest(reference/'binding.json'),reference_raw_sources=raw_sources),
        original_probe_sha256=digest(reference/'build/probe'),expected_static_utma_count=15,
        reference_score_sha256=digest(reference/'score.json'),
        no_refill_extrapolation=True,automatic_retries=False,frozen_at_ns=time.time_ns())
    save_json(root/'plan.json',plan);names.append('plan.json')
    (root/'README.md').write_text('''# Q02 共同预填代码路径

752300的Q2首次A16响应0、Q4为498cycle，但Q2 remainder与Q4展开节奏混杂。私有CU仅将首次for(g<active)改成四组显式调用及active>1/2/3 guard；canonical probe、refill、fullwait、四槽/代次、三请求/SW128和唯一prefill Point不改。目标是实际SASS中Q2/Q4首两组共享同一段代码，不把初始组数解释成硬件容量。

Q2/Q4×A lda4096/4104×trace0/−1，共八开发条件、6verify＋6formal进程／48臂、12582912加载值；seed3正序，22/41逆序。固定step1、mask7/D0、allocA4104/B2360、G32/K4096/ring64、tile[7,6]、SM124；每臂仍先Q4/G64、trace−1十次交替暖机。Q4须复现plain21819/22317cycle、10944/11200ns、prefill1389/1887cycle及同seed增量498cycle，各5%。完整双单位与prefill cycle三进程范围≤中位5%，sparse/plain扰动≤5%，数值/身份/资源通过才判。

唯一预测R=median同seed Q2 prefill A16增量／同批Q4对应，沿0±0.05；cycle主判、prefill ns只报告。资格通过且未命中则stop_zero_response_form，资格失败信息不足，均不重采。零/负差保留；旧Q2仅作组织对照，不定价、均分τ/L/D或替换组合，独立C2/C3均false。

新binary须独立构建和批准，原reference binary只保存比较身份。预计15静态UTMA；root核Q2/Q4共享首两组实际代码、arrive→A→B0→B1、末戳先于wait、local/spill0及SMEM/驻留，并记录Q4控制变化后才批准GPU。最多两次定向修复，免费ROMEO CPU构建GPU0、采样10min且GPU≤1小时；模型回放0，无新条件轴或频率/暖机支线。

```sh
python -B cpu-check.py
python -B source/q02_prefill_shared_prefix.py compile-only --root "$PWD"
python -B source/q02_prefill_shared_prefix.py execute --root "$PWD"
python -B source/q02_prefill_shared_prefix.py score --root "$PWD" --output /tmp/q02-shared-prefix-score.json
```

准备阶段不构建/GPU；根代理在运输文件准备完毕后固定最终输入、运行CPU检查并负责新目标批准。
''');names.append('README.md')
    save_json(root/'preparation-repair.json',dict(repairs=0,compilation=False,gpu=False));names.append('preparation-repair.json')
    save_json(root/'build-input-binding.json',dict(schema='gh200-q02-shared-prefix-input/1',
        files={name:digest(root/name) for name in sorted(names)}))
    print('PREPARED: shared initial prefix;8 conditions/12 processes/48 arms; no compilation/GPU',flush=True)


def review_build(root):
    report=observer.review_build(root)
    row=report['rows'][0];row['label']='shared_initial_prefix'
    report.update(status='pending_root_shared_prefix_sass_review',
        basic_gate_passed=row['utma_count']>0 and row['zero_local_stack'] and not row['compile_warning_c7515'])
    return report


def compile_only(root):
    load_plan(root);start=time.monotonic();folder=root/'build';folder.mkdir(exist_ok=False)
    compiler=shutil.which('nvcc');require(compiler is not None,'nvcc缺失')
    command_record(folder/'version',[compiler,'--version'],observer.stride.remaining(start,20))
    version=(folder/'version/stdout').read_text()
    require('V12.9.41' in version and 'compiler.35813241_0' in version,'需要冻结CUDA12.9.41')
    flags=['-std=c++17','-O3','-DNDEBUG','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v']
    disassembler=Path(compiler).resolve().parent/'cuobjdump'
    with tempfile.TemporaryDirectory(prefix='q02-shared-prefix-headers-') as directory:
        with tarfile.open(root/'source/cutlass-3.9.2-include.tar.gz') as archive:archive.extractall(directory,filter='data')
        command_record(folder/'after/compile',[compiler,*flags,'-I'+directory+'/include','-I'+str(root/'source'),
            root/'source/tma_gemm_sources.cu','-o',folder/'probe'],observer.stride.remaining(start,180))
        command_record(folder/'after/disassemble',[disassembler,'--dump-sass',folder/'probe'],observer.stride.remaining(start,40))
        command_record(folder/'after/resources',[disassembler,'--dump-resource-usage',folder/'probe'],observer.stride.remaining(start,30))
    save_json(folder/'kernel-code.json',observer.kernel_records((folder/'after/disassemble/stdout').read_text()))
    report=review_build(root);save_json(folder/'target-review.json',report)
    require(report['basic_gate_passed'],'UTMA/无local-stack/编译警告基础检查失败')
    print('BUILT ONLY: independent root shared-prefix/SASS approval pending; no GPU',flush=True)


def approved_target(root):
    report=review_build(root);approved=read_json(root/'approved-target.json')
    require(report==read_json(root/'build/target-review.json') and report['basic_gate_passed'],'新构建证据不符')
    require(approved['schema']==APPROVAL_SCHEMA and approved['approved'] is True and
        approved['plan_sha256']==digest(root/'plan.json') and approved['probe_sha256']==digest(root/'build/probe') and
        approved['kernel_code_sha256']==digest(root/'build/kernel-code.json') and
        approved['target_review_sha256']==digest(root/'build/target-review.json'),'本轮新binary未独立核准')


def summarize_rows(plan,numeric,formal):
    result=active.summarize_rows(plan,numeric,formal)
    result['schema']='gh200-q02-prefill-shared-prefix-score/1'
    if result['observation_qualified'] and result['ratio_cycles'] is not None:
        retained=abs(result['ratio_cycles'])<=plan['ratio_absolute_tolerance']
        result['classification']='q2_zero_initial_response' if retained else 'zero_response_form_failed'
        result['decision']='retain_zero_response_form' if retained else 'stop_zero_response_form'
    return result


def summarize(root):
    plan=load_plan(root,True);approved_target(root)
    require(read_json(root/'allocation/identity.json')['uuid']==plan['gpu_uuid'],'分配卡身份不符')
    numeric=[];formal=[]
    for filename,split,rows in (('correctness_commands.json','correctness',numeric),('commands.json','formal',formal)):
        for task in read_json(root/filename):
            rows.extend(observer.validate_process(root,root/'runs'/split/task['id'],task,plan['gpu_uuid']))
    result=summarize_rows(plan,numeric,formal)
    result.update(plan_sha256=digest(root/'plan.json'),probe_sha256=digest(root/'build/probe'))
    return result


def execute(root):
    plan=load_plan(root,True);approved_target(root);start=time.monotonic()
    environment=inspect_allocated_gpu(root/'allocation');require(environment['uuid']==plan['gpu_uuid'],'分配卡不是冻结参考卡')
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{environment['uuid']}.lock"):
        for filename,split in (('correctness_commands.json','correctness'),('commands.json','formal')):
            for task in read_json(root/filename):
                folder=root/'runs'/split/task['id']
                command_record(folder,observer.stride.probe_command(root,task),observer.stride.remaining(start,60))
                save_json(folder/'validated.json',observer.validate_process(root,folder,task,environment['uuid']))
                print(json.dumps(dict(split=split,process=task['id'],arms=4)),flush=True)
        save_json(root/'score.json',summarize(root))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('prepare','compile-only','execute','score'):
        p=sub.add_parser(mode);p.add_argument('--root',type=Path,required=True)
        if mode=='prepare':
            p.add_argument('--reference',type=Path,required=True)
            p.add_argument('--source-reference',type=Path,required=True)
        if mode=='score':p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.mode=='prepare':prepare(root,args.reference.resolve(),args.source_reference.resolve())
    elif args.mode=='compile-only':compile_only(root)
    elif args.mode=='execute':execute(root)
    else:save_json(args.output,summarize(root))


if __name__=='__main__':main()

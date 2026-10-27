#!/usr/bin/env python3
"""V07/V08/V08F：调用各来源冻结 run_one；终态错误停止该来源。"""
import argparse
import ast
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
UUID = 'GPU-7473147b-6ca4-6e68-68c2-b1a086e1753e'
WARMUP = 'last five warmups CV exceeds 2%'


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_module(package):
    # 独立 CLI 每次只加载一个来源；CPU 回放切换来源时移除上一来源模块缓存。
    source = package/'source'
    for file in source.glob('*.py'):
        sys.modules.pop(file.stem,None)
    sys.path.insert(0,str(source))
    module = read(package/'manifest.json')['module']
    return importlib.import_module(module)


def prepare(args):
    repo,campaign,old,output = args.repo.resolve(),args.campaign.resolve(),args.old.resolve(),args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    audit = read(campaign/'legacy-v05-v08-retry-review.json')
    groups=[]
    for source in audit['sources']:
        if source['max_default_attempts']==1:
            continue
        root=repo/source['source'];is7='V07-' in root.name;can_resume='V08-job' in root.name
        rows={row['id']:row for row in read(root/'cases.json')}
        chunks=[];all_tasks=[];binaries={}
        for batch in source['r5_packages']:
            original=old/batch;manifest=read(original/'manifest.json');tasks=[]
            for task in manifest['tasks']:
                origin=task['origins'][0];record=read(repo/origin['record'])
                assert record['returncode']==0 and record['command'][1:]==task['argv'][1:]
                name=Path(origin['original_argv'][0]).name;binary=original/task['argv'][0]
                if name in binaries:
                    assert sha(binaries[name])==sha(binary)
                binaries[name]=binary
                tasks.append(dict(id=task['id'],case=origin['case'],variant=origin['variant'],trial=origin['trial'],
                    binary=name,argv=task['argv'][1:],binary_sha256=sha(binary),original_record=origin['record'],
                    original_record_sha256=sha(repo/origin['record']),original_raw=origin['original_stdout'],
                    original_raw_sha256=sha(repo/origin['original_stdout']),
                    original_wall_seconds=(record['host_stop_ns']-record['host_start_ns'])/1e9))
            chunks.append((batch,tasks));all_tasks.extend(tasks)
        assert len(all_tasks)==source['r5_success_processes']
        assert len({(task['case'],task['variant'],task['trial'])for task in all_tasks})==len(all_tasks)
        binding=dict(source=source['source'],cases_sha256=sha(root/'cases.json'),
            function_module='run_r18' if is7 else 'run_v08',
            source_sha256=sha(root/'source'/('run_r18.py'if is7 else'run_v08.py')))
        previous=None;entries=[]
        for batch,tasks in chunks+([('v08-resume-template',all_tasks)]if can_resume else[]):
            name='retry-'+batch;package=output/name;resume_template=batch=='v08-resume-template'
            (package/'source').mkdir(parents=True);(package/'build').mkdir()
            for file in (root/'source').glob('*.py'):
                shutil.copy2(file,package/'source'/file.name)
            if can_resume:
                shutil.copy2(root/'resume_v08.py',package/'resume_v08.py')
            for file in ('q00_a044_remeasure.py','l2_saturation.py'):
                shutil.copy2(HERE/file,package/file)
            shutil.copy2(Path(__file__),package/'run.py')
            for name_binary,binary in binaries.items():
                shutil.copy2(binary,package/'build'/name_binary);(package/'build'/name_binary).chmod(0o755)
            shutil.copy2(root/'cases.json',package/'cases.json');save(package/'all-slots.json',all_tasks)
            manifest=dict(schema='a044-legacy-original-run-one/1',id=name,source_binding=binding,module=binding['function_module'],
                tasks=tasks,logical_slots=len(tasks),whole_source_slots=len(all_tasks),required_predecessor=None if resume_template else previous,
                phase='resume'if resume_template else'default',resume_allowed=can_resume,
                maximum_attempt=5 if resume_template else 2,timeout_seconds=180 if is7 else 300,
                original_wall_seconds=sum(task['original_wall_seconds']for task in tasks),
                maximum_default_attempts=3*len(tasks) if not resume_template else None,
                resume_limit_per_invocation=120 if resume_template else None,
                target_node='romeo-a044',gpu_uuid=UUID,continuous_telemetry=True,
                terminal_error='停止该来源，保存 attempt 链与已完成槽；V08 仅暖机耗尽可在独立窗口用原 resume',
                original_retry_source=source['sources'],resume_source_sha256=sha(root/'resume_v08.py')if can_resume else None)
            save(package/'manifest.json',manifest)
            save(package/'package-identity.json',{str(file.relative_to(package)):sha(file)for file in package.rglob('*')if file.is_file()})
            archive=output/(name+'.tar.gz')
            with tarfile.open(archive,'w:gz')as stream:
                for file in sorted(package.iterdir()):stream.add(file,arcname=file.name)
            entry=dict(id=name,package=str(package),archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
                manifest_sha256=sha(package/'manifest.json'),phase=manifest['phase'],logical_slots=len(tasks),
                maximum_default_attempts=manifest['maximum_default_attempts'],original_wall_seconds=manifest['original_wall_seconds'],
                replaces_batch=batch if not resume_template else None,source=source['source'])
            entries.append(entry)
            if not resume_template:previous=dict(id=name,manifest_sha256=entry['manifest_sha256'])
        groups.append(dict(source=source['source'],logical_slots=len(all_tasks),packages=entries))
    save(output/'packages.json',dict(groups=groups,default_packages=20,logical_slots=6840,maximum_default_attempts=20520,
        optional_resume_template=1,scope='替代 r5 152–171；V05/V06 保持原输入；未执行 GPU'))
    print(json.dumps(dict(default_packages=20,logical_slots=6840,maximum_default_attempts=20520,optional_resume_template=1)))


def execute_slots(package,destination,prior=None,seed=None,limit=120,process_run=subprocess.run):
    manifest=read(package/'manifest.json');phase=manifest['phase'];binding=manifest['source_binding']
    if prior and prior['source_binding']!=binding:raise ValueError('来源状态不符')
    if phase=='default':
        if manifest['required_predecessor'] is not None and (not prior or prior['package']!=manifest['required_predecessor']):
            raise ValueError('需要该来源的前一块状态')
        if prior and prior['status']=='stopped':raise ValueError('该来源已停止，不能继续默认块')
    else:
        if not prior or not manifest['resume_allowed']:raise ValueError('恢复需要 V08 来源状态')
        if prior['phase']=='default' and not(prior['status']=='stopped'and prior['resume_eligible']and seed):
            raise ValueError('首次恢复需要默认暖机耗尽状态与原 attempt 链')
        if prior['phase']=='resume' and prior['status']=='stopped':raise ValueError('原 resume 已停止')
    completed=set(prior['completed_slots'])if prior else set();all_tasks=read(package/'all-slots.json')
    tasks=[task for task in (all_tasks if phase=='resume'else manifest['tasks'])if task['id']not in completed]
    if phase=='resume':tasks=tasks[:limit]
    destination.mkdir(parents=True,exist_ok=True);(destination/'build').symlink_to(package/'build',target_is_directory=True)
    shutil.copy2(package/'cases.json',destination/'cases.json')
    if seed:
        shutil.copytree(seed/'samples',destination/'samples',dirs_exist_ok=True)
    cases={row['id']:row for row in read(package/'cases.json')};setups={};save(destination/'static_setup.json',[])
    original=source_module(package);sampler=original.run_one
    if phase=='resume':
        tree=ast.parse((package/'resume_v08.py').read_text());function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='run_one')
        namespace=dict(json=json,gzip=gzip,time=time,Path=Path,run_v08=original,common=original.common,LIMIT=5)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(package/'resume_v08.py'),'exec'),namespace);sampler=namespace['run_one']
    actual_run=subprocess.run;attempts=[];outcomes=[];current={};terminal=None
    def tracked_run(argv,**kwargs):
        attempt=dict(slot=current['id'],case=current['case'],variant=current['variant'],trial=current['trial'],
            attempt=current['next_attempt'],argv=argv,start_ns=time.time_ns());current['next_attempt']+=1
        try:
            result=process_run(argv,**kwargs)
            attempt.update(returncode=result.returncode,stderr=result.stderr,stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest())
            for line in result.stdout.splitlines():
                event=json.loads(line)
                if event.get('event')=='setup':
                    setups[current['case']]=dict(case=current['case'],setup=event)
                    save(destination/'static_setup.json',list(setups.values()))
            return result
        except subprocess.TimeoutExpired as error:
            attempt.update(returncode=None,error=str(error));raise
        finally:
            attempt['stop_ns']=time.time_ns();attempts.append(attempt)
            with(destination/'attempts.jsonl').open('a')as stream:stream.write(json.dumps(attempt)+'\n')
    subprocess.run=tracked_run
    try:
        for task in tasks:
            expected_args=original.case_args(cases[task['case']])
            if expected_args!=task['argv']:raise ValueError('原函数参数与原调用不符：'+task['id'])
            folder=destination/'samples'/task['case'];prefix=f"{task['variant']}-{task['trial']:02}"
            previous=[read(path)for path in folder.glob(prefix+'*.json')if read(path)['variant']==task['variant']and read(path)['trial']==task['trial']]if folder.exists()else[]
            current=dict(task,next_attempt=max((record['attempt']for record in previous),default=-1)+1)
            before=len(attempts)
            try:
                result=sampler(destination,cases[task['case']],task['variant'],task['trial'])
                completed.add(task['id']);outcome=dict(slot=task['id'],status='original_run_one_passed',successful_attempt=result['attempt'])
            except(ValueError,subprocess.TimeoutExpired)as error:
                terminal=dict(slot=task['id'],error=str(error));outcome=dict(terminal,status='source_stopped')
            outcome['new_attempts']=len(attempts)-before;outcomes.append(outcome);save(destination/'slot-outcomes.json',outcomes)
            if terminal:break
    finally:
        subprocess.run=actual_run
    last=attempts[-1]if attempts else{}
    eligible=bool(terminal and phase=='default'and manifest['resume_allowed']and last.get('returncode')
        and WARMUP in last.get('stderr','')and last.get('attempt')==2)
    status='stopped'if terminal else('complete'if len(completed)==len(all_tasks)else'active')
    continuation=dict(source_binding=binding,package=dict(id=manifest['id'],manifest_sha256=sha(package/'manifest.json')),
        phase=phase,status=status,completed_slots=sorted(completed),terminal=terminal,resume_eligible=eligible)
    save(destination/'continuation.json',continuation)
    summary=dict(phase=phase,status=status,logical_slots_considered=len(tasks),actual_new_attempts=len(attempts),
        successful_process_attempts=sum(row.get('returncode')==0 for row in attempts),
        validated_logical_slots=sum(row['status']=='original_run_one_passed'for row in outcomes),
        source_completed_slots=len(completed),whole_source_slots=len(all_tasks),outcomes=outcomes,resume_eligible=eligible)
    save(destination/'summary.json',summary);return summary


def run(args):
    package=args.package.resolve();manifest=read(package/'manifest.json')
    print(json.dumps({key:manifest[key]for key in('id','phase','logical_slots','maximum_attempt','required_predecessor')},ensure_ascii=False))
    if not args.execute:return
    for name,checksum in read(package/'package-identity.json').items():
        if sha(package/name)!=checksum:raise ValueError('包内容变化：'+name)
    prior=read(args.continuation)if args.continuation else None
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    from q00_a044_remeasure import inspect_a044_allocation
    from l2_saturation import command_record,exclusive_lock
    allocation=inspect_a044_allocation(output/'allocation')
    if allocation['uuid']!=manifest['gpu_uuid']:raise ValueError('实际 GPU 不符')
    save(output/'request.json',dict(manifest_sha256=sha(package/'manifest.json'),allocation=allocation,prior_continuation=prior))
    query=['nvidia-smi','-i',allocation['uuid'],'--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu','--format=csv']
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        command_record(output/'environment-before',query,30)
        with(output/'telemetry.csv').open('wb')as stream:
            sampler=subprocess.Popen(query+['--loop-ms=1000'],stdout=stream,stderr=subprocess.STDOUT)
            try:summary=execute_slots(package,output,prior,args.seed,args.limit)
            finally:
                sampler.terminate();sampler.wait(timeout=10);command_record(output/'environment-after',query,30)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if summary['status']=='stopped':raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='step',required=True)
    prep=sub.add_parser('prepare')
    for name in('repo','campaign','old','output'):prep.add_argument('--'+name,type=Path,required=True)
    runp=sub.add_parser('run');runp.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    runp.add_argument('--output',type=Path);runp.add_argument('--continuation',type=Path);runp.add_argument('--seed',type=Path)
    runp.add_argument('--limit',type=int,default=120);runp.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.step=='prepare':prepare(args)
    else:
        if args.execute and not args.output:p.error('执行需要新 --output')
        run(args)


if __name__=='__main__':main()

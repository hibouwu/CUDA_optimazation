#!/usr/bin/env python3
"""V09 后续槽：直接使用冻结 run_one，按原规则保留每个 attempt。"""
import argparse
import copy
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
UUID = 'GPU-7473147b-6ca4-6e68-68c2-b1a086e1753e'
SOURCES = ('run_v08.py', 'v08_model.py', 'v06_run.py', 'v06_model.py', 'analyze_r18.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def modules(source):
    sys.path.insert(0, str(source))
    return importlib.import_module('run_v08')


def prepare(args):
    baseline, campaign, output = args.baseline.resolve(), args.campaign.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = modules(baseline/'source')
    cases = {row['id']: row for row in read(baseline/'cases.json')}
    view = args.received_view.resolve()
    validation = read(campaign/'v09-partial-720-validation.json')
    failed = {(row['case'], row['variant'], row['trial']): row
              for row in validation['summary']['failures']}
    tasks, origins = [], {}
    for index in range(4):
        package = campaign/f'v09-sequence-packages/v09-{index:02}'
        for task in read(package/'manifest.json')['tasks']:
            origin = task['origins'][0]
            key = (origin['case'], origin['variant'], origin['trial'])
            binary = Path(origin['original_argv'][0]).name
            assert original.case_args(cases[key[0]]) == task['argv'][1:]
            assert binary == original.common.binary_prefix(cases[key[0]])+'_'+key[1]
            item = dict(id=task['id'], case=key[0], variant=key[1], trial=key[2],
                first_attempt=1 if key in failed else 0, binary=binary,
                binary_sha256=sha(package/task['argv'][0]), argv=task['argv'][1:],
                original_sequence_index=task['protocol']['original_sequence_index'],
                source_manifest=str(package/'manifest.json'), source_manifest_sha256=sha(package/'manifest.json'))
            origins[key] = (package, task)
            tasks.append(item)
    groups = [('v09-recovery', [task for task in tasks if (task['case'],task['variant'],task['trial']) in failed])]
    groups += [(f'v09-pending-trial{trial:02}', [task for task in tasks if task['trial']==trial
                and cases[task['case']]['set']=='heldout']) for trial in range(4,10)]
    assert [len(group) for _,group in groups] == [20]+[120]*6
    selected = [task for _,group in groups for task in group]
    assert len({(task['case'],task['variant'],task['trial']) for task in selected}) == 740
    setups = {}
    for case in {task['case'] for task in selected}:
        record_path = next(path for path in sorted((view/'samples'/case).glob('*.json')) if read(path)['returncode']==0)
        record = read(record_path)
        with gzip.open(view/record['raw'], 'rt') as stream:
            setup = next(json.loads(line) for line in stream if json.loads(line).get('event')=='setup')
        assert setup['gpu_uuid'] == UUID
        setups[case] = dict(case=case, setup=setup, origin_record=str(record_path),
            origin_record_sha256=sha(record_path), raw_sha256=record['raw_sha256'])
    packages = []
    previous = None
    for name, group in groups:
        package = output/name
        (package/'source').mkdir(parents=True)
        (package/'build').mkdir()
        for source in SOURCES:
            shutil.copy2(baseline/'source'/source, package/'source'/source)
        for source in ('q00_a044_remeasure.py','l2_saturation.py'):
            shutil.copy2(HERE/source, package/source)
        shutil.copy2(Path(__file__), package/'run.py')
        for task in group:
            original_package, original_task = origins[task['case'],task['variant'],task['trial']]
            target = package/'build'/task['binary']
            if not target.exists():
                shutil.copy2(original_package/original_task['argv'][0], target)
                target.chmod(0o755)
            assert sha(target) == task['binary_sha256']
            if task['first_attempt']:
                source_record = view/'samples'/task['case']/f"{task['variant']}-{task['trial']:02}.json"
                record = read(source_record)
                assert record['returncode'] and record['attempt']==0 and 'last five warmups CV exceeds 2%' in record['stderr']
                for source_path, relative in ((source_record, source_record.relative_to(view)),(view/record['raw'],Path(record['raw']))):
                    target = package/'seed'/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, target)
        manifest = dict(schema='a044-v09-original-run-one/1', id=name, target_node='romeo-a044', gpu_uuid=UUID,
            purpose='恢复后补测窗口，接续既有 attempt0' if name=='v09-recovery' else '未运行槽，在槽位置按原函数即时重试',
            tasks=group, logical_slots=len(group), maximum_new_attempts=sum(3-task['first_attempt'] for task in group),
            required_predecessor=previous, continuous_telemetry=True,
            observation_protocol='本轮共同新增 1 秒 NVML 监测；原函数的数值、trace、暖机与 attempt 规则逐字节复用',
            source_run_one=dict(source='source/run_v08.py', sha256=sha(package/'source/run_v08.py'),lines=[119,145]),
            outer_failure_rule='run_v09.sample keep_going=True：记录失败，disabled(case,variant)，其他条件继续',
            original_outer_source=dict(path=str(baseline/'source/run_v09.py'),sha256=sha(baseline/'source/run_v09.py'),lines=[72,88]))
        save(package/'manifest.json', manifest)
        save(package/'cases.json', [cases[case] for case in sorted({task['case'] for task in group})])
        save(package/'static_setup.json', [setups[case] for case in sorted({task['case'] for task in group})])
        save(package/'package-identity.json', {str(path.relative_to(package)):sha(path)
            for path in package.rglob('*') if path.is_file()})
        archive = output/(name+'.tar.gz')
        with tarfile.open(archive, 'w:gz') as stream:
            for path in sorted(package.iterdir()):
                stream.add(path, arcname=path.name)
        previous = dict(id=name, manifest_sha256=sha(package/'manifest.json'))
        packages.append(dict(**previous, package=str(package), archive=str(archive), bytes=archive.stat().st_size,
            sha256=sha(archive), logical_slots=len(group), maximum_new_attempts=manifest['maximum_new_attempts'],
            run_command=['python3','run.py','run','--output','measure','--execute']+
                (['--continuation','prior-continuation.json'] if len(packages) else [])))
    save(output/'packages.json', dict(packages=packages, logical_slots=740, maximum_new_attempts=2200,
        existing_successful_slots=700, existing_failed_attempts=20,
        continuation='首包无需前态；后续包将前一包 measure/continuation.json 作为独立运行输入 prior-continuation.json。'))
    print(json.dumps(dict(packages=len(packages), logical_slots=740,maximum_new_attempts=2200),ensure_ascii=False))


def execute_slots(package, destination, prior=None, process_run=subprocess.run):
    """独立于 Slurm 的原函数适配；CPU 模拟可注入 process_run。"""
    manifest = read(package/'manifest.json')
    if manifest['required_predecessor'] is not None:
        if prior is None or prior['package'] != manifest['required_predecessor'] or not prior['complete']:
            raise ValueError('需要前一包完整 continuation.json')
    disabled = {tuple(pair) for pair in prior['disabled']} if prior else set()
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'build').symlink_to(package/'build', target_is_directory=True)
    for name in ('cases.json','static_setup.json'):
        shutil.copy2(package/name, destination/name)
    if (package/'seed').exists():
        shutil.copytree(package/'seed', destination, dirs_exist_ok=True)
    cases = {row['id']: row for row in read(package/'cases.json')}
    original = modules(package/'source')
    saved_subprocess = original.subprocess
    attempts, outcomes = [], []
    current = {}
    def tracked_run(argv, **kwargs):
        attempt = dict(slot=current['id'],case=current['case'],variant=current['variant'],trial=current['trial'],
            attempt=current['first_attempt']+current['executed'], argv=argv, start_ns=time.time_ns())
        current['executed'] += 1
        try:
            result = process_run(argv, **kwargs)
            attempt.update(returncode=result.returncode, stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                stderr=result.stderr, stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest())
            return result
        except subprocess.TimeoutExpired as error:
            attempt.update(returncode=None, error=str(error))
            raise
        finally:
            attempt['stop_ns'] = time.time_ns()
            attempts.append(attempt)
            with (destination/'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(attempt, ensure_ascii=False)+'\n')
    original.subprocess = types.SimpleNamespace(run=tracked_run)
    try:
        for task in manifest['tasks']:
            key = (task['case'],task['variant'])
            if key in disabled:
                outcomes.append(dict(slot=task['id'], case=key[0], variant=key[1], trial=task['trial'],
                    status='skipped_after_original_failure_rule', new_attempts=0))
                continue
            current = dict(task, executed=0)
            try:
                record = original.run_one(destination, cases[task['case']], task['variant'], task['trial'])
                outcome = dict(slot=task['id'],case=key[0],variant=key[1],trial=task['trial'],
                    status='original_run_one_passed', successful_attempt=record['attempt'])
            except (ValueError, subprocess.TimeoutExpired) as error:
                disabled.add(key)
                outcome = dict(slot=task['id'],case=key[0],variant=key[1],trial=task['trial'],
                    status='original_run_one_failed',error=str(error))
            outcome['new_attempts'] = current['executed']
            outcomes.append(outcome)
            save(destination/'slot-outcomes.json', outcomes)
    finally:
        original.subprocess = saved_subprocess
    save(destination/'slot-outcomes.json', outcomes)
    summary = dict(logical_slots=len(manifest['tasks']),actual_new_attempts=len(attempts),
        successful_process_attempts=sum(row.get('returncode')==0 for row in attempts),
        validated_logical_slots=sum(row['status']=='original_run_one_passed' for row in outcomes),
        failed_logical_slots=sum(row['status']=='original_run_one_failed' for row in outcomes),
        skipped_logical_slots=sum(row['status']=='skipped_after_original_failure_rule' for row in outcomes),
        seeded_previous_failed_attempts=sum(task['first_attempt'] for task in manifest['tasks']),outcomes=outcomes)
    save(destination/'summary.json', summary)
    save(destination/'continuation.json', dict(package=dict(id=manifest['id'],manifest_sha256=sha(package/'manifest.json')),
        complete=True, disabled=sorted(disabled), previous_package=prior['package'] if prior else None))
    return summary


def run(args):
    package = args.package.resolve()
    manifest = read(package/'manifest.json')
    print(json.dumps({key:manifest[key] for key in ('id','purpose','logical_slots','maximum_new_attempts','required_predecessor')},ensure_ascii=False))
    if not args.execute:
        return
    for name, checksum in read(package/'package-identity.json').items():
        if sha(package/name) != checksum:
            raise ValueError('输入字节变化：'+name)
    prior = read(args.continuation) if args.continuation else None
    destination = args.output.resolve()
    destination.mkdir(parents=True,exist_ok=False)
    from q00_a044_remeasure import inspect_a044_allocation
    from l2_saturation import command_record, exclusive_lock
    allocation = inspect_a044_allocation(destination/'allocation')
    if allocation['uuid'] != manifest['gpu_uuid']:
        raise ValueError('本轮 V09 目标 GPU 与实际分配不符')
    query = ['nvidia-smi','-i',allocation['uuid'],
        '--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu','--format=csv']
    save(destination/'request.json', dict(package=manifest['id'],manifest_sha256=sha(package/'manifest.json'),
        prior_continuation=prior,allocation=allocation))
    with exclusive_lock(Path('/tmp')/f"gh200-measurement-{allocation['uuid']}.lock"):
        command_record(destination/'environment-before',query,30)
        with (destination/'telemetry.csv').open('wb') as stream:
            sampler = subprocess.Popen(query+['--loop-ms=1000'],stdout=stream,stderr=subprocess.STDOUT)
            try:
                summary = execute_slots(package,destination,prior)
            finally:
                sampler.terminate()
                sampler.wait(timeout=10)
                command_record(destination/'environment-after',query,30)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def merge(args):
    """CPU：合并既有 q00 视图与原生 retry samples，齐备后调用冻结评分器。"""
    import q00_a044_v09 as adapter
    baseline, received, output = args.baseline.resolve(), args.received_view.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = {row['id']:row for row in read(baseline/'cases.json')}
    packages = {entry['manifest_sha256']:read(Path(entry['package'])/'manifest.json')
                for entry in read(args.followup_package/'packages.json')['packages']}
    records, duplicates = {}, 0
    sources = [(received, None)]
    for view in args.native_view:
        request = read(view/'request.json')
        manifest = packages[request['manifest_sha256']]
        if request['package'] != manifest['id'] or request['allocation']['uuid'] != manifest['gpu_uuid']:
            raise ValueError('原生采样的包或 GPU 身份不符')
        sources.append((view.resolve(),manifest))
    for view, manifest in sources:
        slots = {(task['case'],task['variant'],task['trial']):task for task in manifest['tasks']} if manifest else None
        for path in sorted((view/'samples').glob('*/*.json')):
            record = read(path)
            key = (record['case'],record['variant'],record['trial'],record['attempt'])
            raw = view/record['raw']
            if sha(raw) != record['raw_sha256']:
                raise ValueError('样本压缩字节不符：'+str(raw))
            if key in records:
                if record != records[key]:
                    raise ValueError('同一 attempt 对应不同记录：'+str(key))
                duplicates += 1
                continue
            if slots is not None:
                task = slots[key[:3]]
                if record['command'][1:] != task['argv'] or Path(record['command'][0]).name != task['binary']:
                    raise ValueError('新 attempt 的原参数变化：'+str(key))
                if not task['first_attempt'] <= key[3] <= 2:
                    raise ValueError('新 attempt 超出原重试范围：'+str(key))
            suffix = f'{key[1]}-{key[2]:02}'+(f'-retry{key[3]}' if key[3] else '')
            folder = output/'samples'/key[0]
            folder.mkdir(parents=True,exist_ok=True)
            target = folder/(suffix+'.txt.gz')
            shutil.copy2(raw,target)
            record = dict(record,raw=str(target.relative_to(output)))
            save(folder/(suffix+'.json'),record)
            records[key] = record
    save(output/'cases.json',list(cases.values()))
    expected = {(case,variant,trial) for case in cases for variant in adapter.VARIANTS for trial in range(10)}
    successes = {}
    for key, record in records.items():
        if not record['returncode']:
            if key[:3] in successes:
                raise ValueError('同一逻辑槽有多次成功：'+str(key[:3]))
            successes[key[:3]] = record
    missing = [dict(case=key[0],variant=key[1],trial=key[2]) for key in sorted(expected-successes.keys())]
    coverage = dict(expected_cases=len(cases),expected_processes=len(expected),expected_logical_slots=len(expected),
        returned_attempts=len(records),successful_process_candidates=len(successes),
        failed_attempts=sum(bool(record['returncode']) for record in records.values()),
        duplicate_seed_attempts_not_counted=duplicates,missing_successful_slots=missing,
        all_four_variants_returned=not missing,full_scoring_done=False)
    if missing:
        save(output/'coverage.json',coverage)
    else:
        save(output/'prediction_binding.json',dict(first_sample_unix_ns=min(row['host_start_ns'] for row in records.values()),
            prediction_sha256=sha(baseline/'frozen/predictions.json'),scope='旧冻结数值对本轮恢复后 samples 的回归迁移'))
        coverage['full_scoring_done'] = True
        adapter.evaluate(baseline,output,cases,coverage,[])
    print(json.dumps(coverage,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='step',required=True)
    prep = sub.add_parser('prepare')
    for name in ('baseline','campaign','received-view','output'):
        prep.add_argument('--'+name,type=Path,required=True)
    runtime = sub.add_parser('run')
    runtime.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    runtime.add_argument('--output',type=Path)
    runtime.add_argument('--continuation',type=Path)
    runtime.add_argument('--execute',action='store_true')
    merger = sub.add_parser('merge')
    for name in ('baseline','received-view','followup-package','output'):
        merger.add_argument('--'+name,type=Path,required=True)
    merger.add_argument('--native-view',type=Path,action='append',default=[])
    args = parser.parse_args()
    if args.step=='prepare':
        prepare(args)
    elif args.step=='run':
        if args.execute and not args.output:
            parser.error('执行需要新的 --output')
        run(args)
    else:
        merge(args)


if __name__=='__main__':
    main()

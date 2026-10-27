#!/usr/bin/env python3
"""把 a044 压缩采样适配为 V09 原评分视图，比较旧卡冻结预测；只使用 CPU。"""
import argparse
import collections
import gzip
import hashlib
import importlib
import io
import json
from pathlib import Path
import statistics
import sys
import tarfile

from q00_a044_remeasure import digest, read, save

VARIANTS = ('plain', 'wide', 'stamped', 'dual')


def planned(baseline, packages):
    all_cases = {row['id']: row for row in read(baseline/'cases.json')}
    tasks, slots = {}, set()
    manifests = []
    for package in packages:
        manifest = read(package/'manifest.json')
        manifests.append(dict(package=str(package), sha256=digest(package/'manifest.json')))
        for task in manifest['tasks']:
            origins = [o for o in task['origins'] if Path(o['run']).name == baseline.name
                       and o.get('case') in all_cases]
            selected = {(o['case'], o['variant']) for o in origins}
            if len(selected) != 1:
                raise ValueError('V09 task 不能唯一对应 case/variant：'+task['id'])
            case, variant = next(iter(selected))
            trials = sorted({o['trial'] for o in origins})
            if (task['id'] in tasks or len(trials) != task['repeats']
                    or variant not in VARIANTS or any(trial not in range(10) for trial in trials)):
                raise ValueError('V09 重复任务或重复数变化：'+task['id'])
            if any(o['original_argv'][1:] != task['argv'][1:] for o in origins):
                raise ValueError('V09 参数变化：'+task['id'])
            added = {(case, variant, trial) for trial in trials}
            if slots & added:
                raise ValueError('V09 重复进程：'+str(sorted(slots & added)))
            slots.update(added)
            tasks[task['id']] = dict(task=task, case=case, variant=variant, trials=trials)
    present = {case for case, _, _ in slots}
    required = {case for case, row in all_cases.items() if row['set'] == 'heldout'}
    if any(all_cases[case]['set'] != 'heldout' for case in present):
        required = set(all_cases)
    cases = {case: row for case, row in all_cases.items() if case in present | required}
    expected = {(case, variant, trial) for case in cases for variant in VARIANTS for trial in range(10)}
    if slots != expected:
        raise ValueError('V09 四变体覆盖不完整：'+str(sorted(expected-slots)))
    return cases, tasks, manifests


def archived_processes(wrapper, expected_manifest_hashes):
    """顺序读取外层归档，每个进程解开内层归档，核验原始字节身份。"""
    archive = wrapper/'output.tar.gz'
    receipt = read(wrapper/'receipt.json')
    if digest(archive) != receipt['sha256']:
        raise ValueError('回传归档 SHA256 不符：'+str(wrapper))
    pending = collections.defaultdict(dict)
    request = None
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if parts[-1] == 'request.json' and 'runs' not in parts:
                request = json.loads(stream.extractfile(member).read())
                if request['manifest_sha256'] not in expected_manifest_hashes:
                    raise ValueError('采集与准备包不符：'+str(wrapper))
            if 'runs' not in parts:
                continue
            index = parts.index('runs')
            if len(parts) != index+4:
                continue
            task, repeat, name = parts[index+1:]
            if name not in ('receipt.json', 'command.json', 'output-archive.json', 'output.tar.gz'):
                continue
            item = pending[task, int(repeat)]
            data = stream.extractfile(member).read()
            if name == 'output.tar.gz':
                item['archive_sha256'] = hashlib.sha256(data).hexdigest()
                with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as inner:
                    item['stdout'] = inner.extractfile('stdout').read()
                    item['stderr'] = inner.extractfile('stderr').read()
            else:
                item[name] = json.loads(data)
            if all(key in item for key in ('receipt.json', 'command.json', 'output-archive.json', 'stdout')):
                packed = item['output-archive.json']
                if item['archive_sha256'] != packed['sha256']:
                    raise ValueError('进程归档 SHA256 不符：'+task)
                for key in ('stdout', 'stderr'):
                    checksum = hashlib.sha256(item[key]).hexdigest()
                    if checksum != packed['members'][key]['sha256'] or checksum != item['receipt.json'][key+'_sha256']:
                        raise ValueError('进程输出 SHA256 不符：'+task+'/'+key)
                yield task, int(repeat), item
                del pending[task, int(repeat)]
    if request is None or pending:
        raise ValueError('采集归档缺少 request 或完整进程：'+str(wrapper))


def original_modules(baseline):
    sys.path.insert(0, str(baseline/'source'))
    scorer = importlib.import_module('run_v09')
    analyzer = importlib.import_module('analyze_r18')
    return scorer, analyzer


def evaluate(baseline, output, cases, coverage, failures):
    frozen = read(baseline/'frozen/predictions.json')
    if digest(baseline/'frozen/predictions.json') != read(baseline/'frozen/manifest.json')['files']['frozen/predictions.json']:
        raise ValueError('原冻结预测身份变化')
    scorer, analyzer = original_modules(baseline)
    scorer.score(output, frozen['predictions'], frozen['policy'], failures)
    scored = read(output/'score.json')
    old = {row['case']: row for row in read(baseline/'score.json')['cases']}
    coverage['heldout_four_variants_valid'] = all(
        all(row['processes'][variant] == 10 for variant in VARIANTS) and not row['errors']
        for row in scored['cases'])
    identities, endpoints, extra_failures, references = set(), [], [], []
    for case, row in cases.items():
        if row['set'] != 'heldout':
            times = {variant: [] for variant in VARIANTS}
            errors = []
            for path in sorted((output/'samples'/case).glob('*.json')):
                record = read(path)
                try:
                    if record['returncode']:
                        raise ValueError('进程返回非零')
                    observed = analyzer.replay(output, record, row)
                    times[record['variant']].append(observed['elapsed_us'])
                except ValueError as error:
                    errors.append(dict(variant=record['variant'], trial=record['trial'], error=str(error)))
            references.append(dict(case=case, processes={variant: len(values) for variant, values in times.items()},
                median_us={variant: statistics.median(values) if values else None for variant, values in times.items()},
                errors=errors, valid=not errors and all(len(values)==10 for values in times.values())))
            continue
        dual = []
        for path in sorted((output/'samples'/case).glob('dual-*.json')):
            record = read(path)
            if record['returncode']:
                continue
            try:
                observed = analyzer.replay(output, record, row)
            except ValueError as error:
                extra_failures.append(dict(case=case, trial=record['trial'], error=str(error)))
                continue
            with gzip.open(output/record['raw'], 'rt') as stream:
                setup = next(json.loads(line) for line in stream if '"event":"setup"' in line)
            identities.add(setup['gpu_uuid'])
            ctas = observed['ctas']
            frequencies = [(c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns']) for c in ctas
                           if c['end_ns'] > c['entry_ns']]
            prediction = frozen['predictions'][case]
            work_matches = None
            if prediction['status'] == 'predicted':
                expected_ctas = prediction['cta_events']
                work_matches = len(ctas) == len(expected_ctas)
                if work_matches:
                    work_matches = all(len(c['tiles_ns']) == len(p['events']) for c, p in zip(ctas, expected_ctas))
            dual.append(dict(trial=record['trial'], envelope_us=(max(c['end_ns'] for c in ctas)-min(c['entry_ns'] for c in ctas))/1000,
                median_cta_effective_ghz=statistics.median(frequencies), ctas=len(ctas),
                predicted_cta_event_counts_match=work_matches))
        new = next(item for item in scored['cases'] if item['case'] == case)
        endpoints.append(dict(case=case, new_plain_us=new['observed_plain_us'], old_plain_us=old[case]['observed_plain_us'],
            new_to_old_plain_relative_change=(new['observed_plain_us']/old[case]['observed_plain_us']-1)
                if new['plain_valid'] and old[case]['plain_valid'] else None,
            frozen_prediction_status=frozen['predictions'][case]['status'],
            frozen_frequency_ghz=frozen['predictions'][case].get('frequency_ghz'), dual_processes=dual))
    save(output/'endpoints.json', endpoints)
    save(output/'reference-checks.json', references)
    coverage['reference_four_variants_valid'] = all(row['valid'] for row in references)
    coverage['all_four_variants_valid'] = coverage['heldout_four_variants_valid'] and coverage['reference_four_variants_valid']
    save(output/'coverage.json', coverage)
    save(output/'migration-summary.json', dict(scope='旧卡冻结数值的跨卡回归迁移',
        baseline_gpu=frozen['gpu'], measured_gpu_uuids=sorted(identities),
        frozen_prediction_sha256=digest(baseline/'frozen/predictions.json'),
        original_scorer_sha256=digest(baseline/'source/run_v09.py'),
        original_numeric_trace_replay_sha256=digest(baseline/'source/analyze_r18.py'),
        model_execution='直接读取原 frozen/predictions.json；未执行模型、拟合或替换 UUID',
        original_support='score.prediction_supported 表示旧卡冻结表是否有数值；新卡支持域未扩展',
        classification='本次为已测条件回归迁移，不是新的独立留出',
        coverage=coverage, old_plain_metrics=read(baseline/'score.json')['plain_metrics'],
        migration_plain_metrics=scored['plain_metrics'],
        original_threshold_comparison=scored['complete_time_passed'],
        frozen_numeric_predictions=sum(p['status']=='predicted' for p in frozen['predictions'].values()),
        frozen_unsupported_cases=[case for case,p in frozen['predictions'].items() if p['status']!='predicted'],
        additional_replay_failures=extra_failures))
    return scored


def score(args):
    baseline, output = args.baseline.resolve(), args.output.resolve()
    cases, tasks, manifests = planned(baseline, [p.resolve() for p in args.package])
    output.mkdir(parents=True, exist_ok=False)
    save(output/'cases.json', list(cases.values()))
    save(output/'input-manifests.json', manifests)
    failures, seen, starts, setups = [], set(), [], []
    for wrapper in args.run:
        for task_id, repeat, item in archived_processes(wrapper.resolve(), {row['sha256'] for row in manifests}):
            if task_id not in tasks or repeat not in range(tasks[task_id]['task']['repeats']) or (task_id, repeat) in seen:
                raise ValueError('非预定或重复的 V09 进程：'+str((task_id, repeat)))
            seen.add((task_id, repeat))
            mapping = tasks[task_id]
            task, case, variant = mapping['task'], mapping['case'], mapping['variant']
            trial = mapping['trials'][repeat]
            receipt = item['receipt.json']
            command = receipt['argv']
            if command[1:] != task['argv'][1:] or Path(command[0]).name != Path(task['argv'][0]).name:
                raise ValueError('新进程参数/二进制身份与准备包不符：'+task_id)
            folder = output/'samples'/case
            folder.mkdir(parents=True, exist_ok=True)
            raw = folder/f'{variant}-{trial:02}.txt.gz'
            raw.write_bytes(gzip.compress(item['stdout'], compresslevel=1, mtime=0))
            record = dict(case=case, variant=variant, trial=trial, attempt=0, command=command,
                returncode=receipt['returncode'], host_start_ns=receipt['start_ns'], host_stop_ns=receipt['stop_ns'],
                stderr=item['stderr'].decode(), raw=str(raw.relative_to(output)), raw_sha256=digest(raw),
                remeasurement_origin=dict(wrapper=str(wrapper.resolve()), task=task_id, repeat=repeat,
                    stdout_sha256=receipt['stdout_sha256']))
            starts.append(receipt['start_ns'])
            try:
                events = [json.loads(line) for line in item['stdout'].decode().splitlines() if line.strip()]
                for event in events:
                    if event.get('event') == 'call':
                        record.update(elapsed_us=event['elapsed_us'], warmup=event['warmup_us'])
                    if event.get('event') == 'setup':
                        setups.append(dict(case=case, variant=variant, trial=trial, setup=event))
            except (ValueError, KeyError) as error:
                failures.append(dict(case=case, variant=variant, trial=trial, error=str(error)))
                record['returncode'] = record['returncode'] or 1
            save(folder/f'{variant}-{trial:02}.json', record)
    missing = [dict(case=mapping['case'], variant=mapping['variant'], trial=trial)
        for task_id,mapping in tasks.items() for repeat,trial in enumerate(mapping['trials']) if (task_id,repeat) not in seen]
    failures.extend(dict(**row,error='本次采集未返回该进程') for row in missing)
    coverage = dict(expected_cases=len(cases), expected_variants=list(VARIANTS), repeats_per_variant=10,
        expected_heldout_cases=sum(row['set']=='heldout' for row in cases.values()),
        expected_reference_cases=sum(row['set']!='heldout' for row in cases.values()),
        expected_processes=sum(mapping['task']['repeats'] for mapping in tasks.values()), returned_processes=len(seen), missing=missing,
        all_four_variants_returned=not missing)
    save(output/'actual-setups.json', setups)
    save(output/'sampling-failures.json', failures)
    save(output/'prediction_binding.json', dict(first_sample_unix_ns=min(starts),
        prediction_sha256=digest(baseline/'frozen/predictions.json'),
        scope='旧冻结预测对新采样的评分绑定；首进程时间取本次 receipt'))
    scored = evaluate(baseline, output, cases, coverage, failures)
    print(json.dumps(dict(coverage=coverage,plain_metrics=scored['plain_metrics']),ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('plan','score'))
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--package',type=Path,action='append',required=True)
    parser.add_argument('--run',type=Path,action='append')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.step == 'plan':
        cases,tasks,manifests=planned(args.baseline.resolve(),[p.resolve() for p in args.package])
        print(json.dumps(dict(cases=len(cases),tasks=len(tasks),processes=sum(t['task']['repeats'] for t in tasks.values()),manifests=manifests),indent=2))
    elif not args.run or args.output is None:
        parser.error('score 需要 --run 和新的 --output')
    else:
        score(args)


if __name__ == '__main__':
    main()

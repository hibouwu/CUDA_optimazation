#!/usr/bin/env python3
"""用资源实验原冻结 oracle 复核 a044 重放输出；不运行 GPU。"""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack_stdout(folder):
    index = read(folder / 'output-archive.json')
    archive = folder / index['archive']
    assert sha(archive) == index['sha256']
    payload = {}
    with tarfile.open(archive) as packed:
        for member in packed:
            content = packed.extractfile(member).read()
            expected = index['members'][member.name]
            assert len(content) == expected['size']
            assert hashlib.sha256(content).hexdigest() == expected['sha256']
            payload[member.name] = content
    assert set(payload) == set(index['members'])
    receipt = read(folder / 'receipt.json')
    for name in ('stdout', 'stderr'):
        assert hashlib.sha256(payload[name]).hexdigest() == receipt[name + '_sha256']
    return payload['stdout'], receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--statistics-source', type=Path)
    args = parser.parse_args()
    manifest = read(args.package / 'manifest.json')
    identity = read(args.package / 'package-identity.json')
    for name, checksum in identity.items():
        assert sha(args.package / name) == checksum, name
    assert read(args.capture / 'request.json')['manifest_sha256'] == sha(args.package / 'manifest.json')
    tasks = [t for t in manifest['tasks'] if t['stages'] == ['RES-' + args.family]]
    oldrun = Path(tasks[0]['origins'][0]['run']).resolve()
    assert all(Path(t['origins'][0]['run']).resolve() == oldrun for t in tasks)
    if args.statistics_source:
        module_spec = importlib.util.spec_from_file_location('statistics', args.statistics_source)
        archived_statistics = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(archived_statistics)
        sys.modules['statistics'] = archived_statistics
    sys.path.insert(0, str(oldrun / 'snapshot/repo/microbench/gh200_resource_campaign'))
    from auditors.suite import load_run, parse_raw, trial_evidence
    from common.suite_io import stats
    spec, contract, protocol, olddevice, family = load_run(oldrun)
    cases = {c['id']: c for c in contract['cases']}
    newdevice = None
    samples = []
    intervals = []
    for task in tasks:
        origin = task['origins'][0]
        point = origin['point']
        case = cases[point['case_id']]
        folder = args.capture / 'runs' / task['id'] / '000'
        stdout, receipt = unpack_stdout(folder)
        assert receipt['returncode'] == 0 and not receipt['timed_out']
        assert receipt['argv'][1:] == task['argv'][1:]
        assert Path(receipt['argv'][0]).name == spec['binary_sha256']
        assert sha(args.package / task['argv'][0]) == spec['binary_sha256']
        assert receipt['stop_ns'] > receipt['start_ns']
        intervals.append((receipt['start_ns'], receipt['stop_ns'], receipt['pid']))
        with tempfile.TemporaryDirectory(prefix='a044-resource-oracle-') as temporary:
            rawpath = Path(temporary) / 'raw.jsonl'
            rawpath.write_bytes(stdout)
            device, row = parse_raw(rawpath)
        family.validate_device(device)
        if newdevice is None:
            newdevice = device
        assert device == newdevice
        assert device['uuid'] == read(args.capture / 'capture-summary.json')['allocation']['uuid']
        item = dict(task=task['id'], case_id=case['id'], batch=point['batch'],
            trial=point['trial'], seed=point['seed'],
            receipt=str(folder / 'receipt.json'), stdout_sha256=receipt['stdout_sha256'],
            host_seconds=(receipt['stop_ns']-receipt['start_ns'])/1e9)
        oldreceipt = Path(origin['receipt'])
        assert sha(oldreceipt) == origin['source_receipt_sha256']
        oldresult = trial_evidence(oldrun, spec, case, olddevice, protocol,
                                  point['batch'], point['trial'], oldreceipt.parent)
        item['old'] = oldresult
        try:
            item['new'] = family.validate_trial(row, case, device, point['seed'], protocol)
            if spec.get('family') == 'auxiliary':
                item['new'].update(family.validate_final_outputs(oldrun, oldreceipt.parent, row))
            if 'resolved_resource_identity' in case:
                assert all(row.get(k) == v for k, v in case['resolved_resource_identity'].items()), 'kernel resources differ from B-bound calibrated binary'
            item['oracle_pass'] = True
        except (ValueError, AssertionError, KeyError) as error:
            item['oracle_pass'] = False
            item['oracle_error'] = str(error)
        samples.append(item)
    intervals.sort()
    assert len({(pid, begin) for begin, end, pid in intervals}) == len(intervals)
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    summaries = []
    coverage = next(c for c in manifest['coverage'] if c['family'] == args.family)
    published = {t['origins'][0]['point']['case_id'] for t in tasks if t['protocol']['published_case']}
    assert len(published) == coverage['qualified_conditions']
    for caseid, case in cases.items():
        subset = [r for r in samples if r['case_id'] == caseid]
        merged, batches, merged_companion = [], [], []
        for batch in sorted({r['batch'] for r in subset}):
            rows = [r for r in subset if r['batch'] == batch]
            values = [r['new']['value'] for r in rows if r['oracle_pass'] and r['new']['warmup_converged']]
            merged.extend(values)
            summary = stats(values) if len(values) >= 2 else None
            batches.append(dict(batch=batch, scheduled=len(rows), valid_warm=len(values),
                stats=summary, stable=len(values) == protocol['trials_per_batch'] and
                    summary['cv'] <= protocol['formal_cv_limit'] and stats(merged)['cv'] <= protocol['formal_cv_limit']))
            if spec.get('family') == 'auxiliary':
                companion = [r['new']['logical_requests_per_complete_CTA_cycle'] for r in rows if r['oracle_pass'] and r['new']['warmup_converged']]
                merged_companion.extend(companion)
                companion_stats = stats(companion) if len(companion) >= 2 else None
                accumulated = stats(merged_companion) if len(merged_companion) >= 2 else None
                batches[-1].update(companion_stats=companion_stats, merged_companion_stats=accumulated)
                batches[-1]['stable'] = bool(batches[-1]['stable'] and companion_stats['cv'] <= protocol['formal_cv_limit'] and accumulated['cv'] <= protocol['formal_cv_limit'])
        oldvalues = [r['old']['value'] for r in subset if r['old']['warmup_converged']]
        oldstats = stats(oldvalues) if len(oldvalues) >= 2 else None
        newstats = stats(merged) if len(merged) >= 2 else None
        summaries.append(dict(case_id=caseid, unit=case['metric']['unit'],
            old_published=caseid in published, exportable=case['exportable'],
            processes=len(subset), oracle_pass=sum(r['oracle_pass'] for r in subset),
            warmup_failed=sum(r['oracle_pass'] and not r['new']['warmup_converged'] for r in subset),
            fixed_replay_statistically_eligible=bool(batches and batches[-1]['stable'] and all(r['oracle_pass'] for r in subset)),
            new=newstats, old=oldstats, median_change_percent=(newstats['median']/oldstats['median']-1)*100 if newstats and oldstats else None,
            batches=batches))
    result = dict(family=args.family, package=str(args.package), capture=str(args.capture),
        old_run=str(oldrun), frozen_oracle=str(Path(sys.modules[family.__name__].__file__)),
        binary_sha256=spec['binary_sha256'], old_device=olddevice, new_device=newdevice,
        statistics_source=(dict(path=str(args.statistics_source), sha256=sha(args.statistics_source))
                           if args.statistics_source else dict(interpreter=sys.version)),
        protocol=protocol,
        statistical_rule=('固定原调用序列；各批完整 10 个暖机通过样本，周期/迭代及逻辑请求/周期的批内和累计 CV 均 ≤5%；以最后已执行批判定。' if spec.get('family') == 'auxiliary' else '固定原调用序列；各批完整 10 个暖机通过样本，批内和累计 CV ≤5%；以最后已执行批判定。'),
        totals=dict(processes=len(samples), oracle_pass=sum(r['oracle_pass'] for r in samples),
            warmup_failed=sum(r['oracle_pass'] and not r['new']['warmup_converged'] for r in samples),
            eligible_cases=sum(r['fixed_replay_statistically_eligible'] for r in summaries),
            published_eligible_cases=sum(r['old_published'] and r['fixed_replay_statistically_eligible'] for r in summaries),
            process_seconds=sum(r['host_seconds'] for r in samples),
            wall_seconds=(intervals[-1][1]-intervals[0][0])/1e9),
        cases=summaries, samples=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['totals'], ensure_ascii=False, indent=2))
    for row in summaries:
        print(row['case_id'], 'oracle', str(row['oracle_pass'])+'/'+str(row['processes']),
            'warmfail', row['warmup_failed'], 'eligible', row['fixed_replay_statistically_eligible'],
            'new', row['new'], 'old', row['old'], 'change%', row['median_change_percent'])


if __name__ == '__main__':
    main()

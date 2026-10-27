#!/usr/bin/env python3
"""准备 10-14/15 频率实验补集；保留原进程顺序和二进制内的空闲干预。"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from q00_a044_remeasure import digest, executable, inspect_stdout, read, resolve_file, save
from q00_a044_legacy_stdout import Text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main(args):
    source = REPO / 'results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prior = collections.defaultdict(list)
    for path in args.prior:
        manifest = read(path)
        for task in manifest['tasks']:
            prior[tuple(task['argv'])].append(dict(manifest=str(path), task=task['id'],
                repeats=task['repeats'], protocol=task['protocol']))
    coverage, blocks, audits = [], [], []
    for directory in sorted(source.iterdir()):
        if not directory.is_dir():
            continue
        returned = directory / 'returned'
        records = []
        files = list(directory.rglob('*'))
        if (returned / 'invocations.json').exists():
            for index, row in enumerate(read(returned / 'invocations.json')):
                raw = returned / 'samples' / f"r{row['repeat']:02d}-{row['id']}.json"
                assert digest(raw) == row['sha256']
                records.append(dict(row, raw_path=raw, record=returned/'invocations.json',
                    record_index=index, start=row['started_ns'], stop=row['ended_ns']))
        else:
            for path in sorted((returned / 'samples').rglob('*.json')):
                row = read(path)
                if not isinstance(row.get('command'), list):
                    continue
                raw = returned / row['raw']
                assert digest(raw) == row['raw_sha256']
                records.append(dict(row, raw_path=raw, record=path,
                    start=row['host_start_ns'], stop=row['host_stop_ns']))
        if not records:
            receipt = directory/'receipt.json'
            if receipt.exists():
                reason = '旧轮次失败且没有样本，后续002已覆盖'
                status = read(receipt)['returncode']
                assert status != 0
            else:
                reason, status = 'CPU开发目录，没有GPU采样', None
            coverage.append(dict(run=str(directory.relative_to(REPO)), calls=0, reason=reason,
                                 original_returncode=status))
            continue
        records.sort(key=lambda row:row['start'])
        assert all(row['returncode'] == 0 for row in records)
        name = f'frequency-{len(blocks):02d}'
        folder = output/name
        folder.mkdir()
        (folder/'assets').mkdir()
        assets, tasks = {}, []
        sequence = []
        raw_bytes = wall_seconds = 0
        for index, row in enumerate(records):
            binary, checksum = resolve_file(row['command'][0], files, binary=True, context=row['record'])
            if row.get('binary_sha256'):
                assert checksum == row['binary_sha256']
            argv = ['assets/'+checksum, *map(str,row['command'][1:])]
            assert all('/' not in value for value in argv[1:]), argv
            assets[checksum] = dict(source=str(binary), kind='binary')
            profile = Path(row['command'][0]).name.rsplit('_', 1)[-1]
            protocol = dict(profile=profile, packed_full_output=True, original_kind='measurement',
                cache_and_warmup='original_binary_and_argv', input_sha256={},
                process_order='original_recorded_sequence', sequence_run=directory.name)
            parsed = inspect_stdout(Text(row['raw_path']), protocol)
            assert parsed['valid'], (row['raw_path'], parsed)
            text = Text(row['raw_path']).read_text()
            raw_bytes += len(text.encode())
            wall_seconds += (row['stop']-row['start'])/1e9
            sequence.append(argv)
            task_id = f'{name}-{index:03d}'
            origin = dict(run=str(returned.relative_to(REPO)), record=str(row['record'].relative_to(REPO)),
                original_argv=row['command'], original_returncode=0,
                original_start_ns=row['start'], original_stop_ns=row['stop'],
                original_stdout=str(row['raw_path'].relative_to(REPO)),
                original_stdout_sha256=digest(row['raw_path']),
                original_record_sha256=digest(row['record']))
            if 'record_index' in row:
                origin['record_index'] = row['record_index']
            tasks.append(dict(id=task_id, stages=['Q09-frequency-history'], argv=argv, repeats=1,
                timeout_seconds=300, protocol=protocol, origins=[origin]))
            audits.append(dict(task=task_id, original_stdout=origin['original_stdout'],
                parser_status=parsed['status'], prior_binary_argv_matches=prior.get(tuple(argv), [])))
        sequence_hash = hashlib.sha256(json.dumps(sequence).encode()).hexdigest()
        for task in tasks:
            task['protocol']['sequence_sha256'] = sequence_hash
        # 外部顺序也是频率协议的一部分；只有整段原顺序相同时才视为已覆盖。
        previous_sequences = {item['protocol'].get('sequence_sha256')
            for argv in sequence for item in prior.get(tuple(argv), [])}
        if sequence_hash in previous_sequences:
            raise ValueError('原顺序已在既有包中；应复用已有序列：'+directory.name)
        condition_repeats = collections.Counter(tuple(task['argv']) for task in tasks)
        sampling_budget = 1.5*wall_seconds
        output_budget = .25*len(tasks)+3*raw_bytes/(20*1024*1024)
        assert sampling_budget+output_budget <= 600, directory.name
        manifest = dict(schema='gh200-a044-remeasure/1', source=str(directory),
            target_node='romeo-a044', target_clock_mhz=1980, tasks=tasks, assets=assets,
            qualification='旧频率条件的a044回归，冻结参数与旧成绩保持不变',
            coverage=[dict(run=str(directory.relative_to(REPO)), original_calls=len(records),
                           ready_calls=len(tasks), status='ready')], omitted=[],
            totals=dict(unique_conditions=len(condition_repeats), scheduled_processes=len(tasks)),
            sequence=dict(sha256=sequence_hash, order='original host_start_ns/started_ns',
                rationale='保留原平衡/随机交错；固定预热与idle-us均由原binary和argv执行'),
            runtime=dict(original_wall_seconds=wall_seconds, sampling_budget_seconds=sampling_budget,
                output_budget_seconds=output_budget, total_budget_seconds=sampling_budget+output_budget),
            condition_repeats=[dict(argv=list(argv), repeats=n) for argv,n in condition_repeats.items()])
        save(folder/'manifest.json',manifest)
        for checksum, entry in assets.items():
            shutil.copy2(entry['source'],folder/'assets'/checksum)
        for script in ('q00_a044_remeasure.py','l2_saturation.py'):
            shutil.copy2(HERE/script,folder/script)
        (folder/'README.md').write_text(f'# {directory.name}\n\n'
            f'保留原 {len(tasks)} 个成功进程的实际顺序和每个条件的次数。源实验记录：`{directory.relative_to(REPO)}`。\n\n'
            '测量输入、活动SM、K长度、stage和前序状态对频率及时间的影响；原冻结模型用于新卡回归，另存评分。\n\n'
            f'原进程主机时间合计 {wall_seconds:.3f} 秒；本块采集与输出预算 {sampling_budget+output_budget:.3f} 秒，网络另计。\n')
        save(folder/'package-identity.json',{str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file()})
        archive = folder.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
            for path in sorted(folder.rglob('*')):
                if path.is_file():tar.add(path,arcname=str(path.relative_to(folder)))
        blocks.append(dict(batch=name,source=str(directory.relative_to(REPO)),processes=len(tasks),
            unique_conditions=len(condition_repeats),archive_bytes=archive.stat().st_size,archive_sha256=digest(archive),
            total_budget_seconds=sampling_budget+output_budget))
        coverage.append(dict(run=str(directory.relative_to(REPO)), calls=len(records), batch=name,
            original_order_preserved=True, original_returncode=0))
        print(name,len(tasks),flush=True)
    save(output/'packages.json',blocks)
    save(output/'coverage.json',coverage)
    save(output/'stdout-and-prior-audit.json',audits)
    totals=dict(blocks=len(blocks),processes=sum(b['processes'] for b in blocks),
        unique_conditions=sum(b['unique_conditions'] for b in blocks),
        prior_binary_argv_overlap=sum(bool(row['prior_binary_argv_matches']) for row in audits),
        gpu_jobs_submitted=0)
    save(output/'summary.json',totals)
    print(json.dumps(totals))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior',type=Path,action='append',required=True)
    main(parser.parse_args())

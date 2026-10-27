#!/usr/bin/env python3
"""收集 legacy 原轮次的全部成功调用，按来源保留原采样设计。"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import re

from q00_a044_legacy import REPO, records
from q00_a044_remeasure import digest, executable, inspect_stdout, read, resolve_file, save
from q00_a044_legacy_stdout import Text, original_stdout
from q00_a044_legacy_batches import duration
from q00_a044_ordered_legacy import package

HERE = Path(__file__).resolve().parent


def call_key(origin):
    return json.dumps([origin['run'],origin['record'],origin.get('case'),origin.get('variant'),origin.get('trial')],sort_keys=True)


def inventory(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=bool(args.refresh_run))
    def write(path, data):
        if args.refresh_run and path.exists():
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
        else:
            save(path, data)
    old = read(args.parent/'manifest.json')
    old_calls = {}
    for task in old['tasks']:
        for origin in task['origins']:
            old_calls[call_key(origin)] = task
    sources, omitted, assets = [], [], (read(output/'assets.json') if args.refresh_run else copy.deepcopy(old['assets']))
    existing_sources = {row['id']:row for row in read(output/'sources.json')} if args.refresh_run else {}
    for source_index, entry in enumerate(old['coverage']):
        name=f'source-{source_index:03d}'
        if args.refresh_run and Path(entry['run']).name not in args.refresh_run:
            sources.append(existing_sources[name])
            continue
        root = REPO/entry['run']
        files = [path for path in root.rglob('*') if path.is_file() and
                 not set(path.relative_to(root).parts[:-1]).intersection({'spack-cache','__pycache__'})]
        rows, seen = [], set()
        for path, command, record in records(root, files, omitted):
            command = list(map(str, command))
            start = next((record.get(key) for key in ('host_start_ns','started_unix_ns','start_ns','started_ns')
                          if record.get(key)), None)
            identity = json.dumps([command,start,str(path),record.get('case'),record.get('variant'),record.get('trial')],sort_keys=True)
            if identity in seen:
                continue
            seen.add(identity)
            origin = dict(run=entry['run'], record=str(path.relative_to(REPO)),
                original_argv=command, original_returncode=record.get('returncode',0),
                case=record.get('case',record.get('case_id')), variant=record.get('variant'),
                trial=record.get('trial'), original_start_ns=start)
            if Path(command[0]).name in ('nvcc','cuobjdump') or record.get('returncode',0) != 0 or record.get('timed_out'):
                omitted.append(dict(**origin,reason='原构建/反汇编或失败调用，保留记录'))
                continue
            try:
                previous = old_calls.get(call_key(origin))
                if previous:
                    argv = previous['argv']
                    protocol = copy.deepcopy(previous['protocol'])
                else:
                    binary, checksum = resolve_file(command[0], files, binary=True, context=path)
                    assets[checksum] = dict(source=str(binary),kind='binary')
                    argv, inputs = ['assets/'+checksum], {}
                    for index, value in enumerate(command[1:],1):
                        if '/' not in value or value.startswith('--'):
                            argv.append(value)
                        else:
                            source, checksum = resolve_file(value,files,context=path)
                            assets[checksum] = dict(source=str(source),kind='input')
                            argv.append('assets/'+checksum)
                            inputs[str(index)] = checksum
                    protocol = dict(packed_full_output=True, original_kind='program_query' if argv[1:]==['device'] else 'measurement',
                                    cache_and_warmup='original_binary_and_argv',input_sha256=inputs)
                protocol['profile'] = Path(command[0]).name.rsplit('_',1)[-1]
                protocol['packed_full_output'] = True
                if argv[1:] == ['device']:
                    protocol['original_kind'] = 'program_query'
                if root.name == '20261007-ndebug-clock-fixedcost' and '--mode' in argv and argv[argv.index('--mode')+1]=='loop':
                    protocol['external_nvsmi'] = dict(fields=read(root/'protocol-clock.json')['nvsmi_fields'],
                        interval_ms=50,before_seconds=.5,after_seconds=.3,source=str(root/'source/r07_run.py')+':156-198')
                raw = original_stdout(origin)
                parsed = inspect_stdout(Text(raw),protocol)
                seconds, timing_source, size = duration(origin)
                budget = 1.5*seconds if seconds is not None else 5
                budget += .25+3*size/(20*1024*1024)
                if protocol.get('external_nvsmi'):
                    budget += .8
                origin.update(original_stdout=str(raw), original_stdout_sha256=digest(raw),
                              record_sha256=digest(path))
                if record.get('command_reconstruction'):
                    origin['command_reconstruction']=record['command_reconstruction']
                rows.append(dict(argv=argv,protocol=protocol,origin=origin,old_task_id=previous['id'] if previous else None,
                    original_wall_seconds=seconds,original_output_bytes=size,budget_seconds=budget,
                    stdout_status=parsed['status'],stdout_valid=parsed['valid'],
                    set=record.get('set'), trial=record.get('trial'),start_ns=start))
            except (ValueError,OSError,KeyError) as error:
                omitted.append(dict(**origin,reason=str(error)))
        if rows and all(row['start_ns'] for row in rows):
            rows.sort(key=lambda row:row['start_ns'])
            ordering='recorded process timestamp'
        else:
            ordering='needs original runner order' if rows else 'no successful GPU process'
        write(output/(name+'.json'),dict(run=entry['run'],labels=entry['experiments'],ordering=ordering,rows=rows))
        sources.append(dict(id=name,run=entry['run'],calls=len(rows),ordering=ordering,
            invalid_stdout=sum(not row['stdout_valid'] for row in rows),
            original_wall_seconds=sum(row['original_wall_seconds'] or 0 for row in rows),
            unknown_wall_processes=sum(row['original_wall_seconds'] is None for row in rows),
            budget_seconds=sum(row['budget_seconds'] for row in rows)))
        print(name,root.name,len(rows),ordering,flush=True)
    write(output/'assets.json',assets)
    write(output/'sources.json',sources)
    if args.refresh_run:
        omitted = [row for row in read(output/'omitted.json') if Path(row['run']).name not in args.refresh_run]+omitted
    write(output/'omitted.json',omitted)
    write(output/'summary.json',dict(source_runs=len(sources),successful_calls=sum(row['calls'] for row in sources),
        invalid_stdout=sum(row['invalid_stdout'] for row in sources),
        original_wall_seconds=sum(row['original_wall_seconds'] for row in sources),
        unknown_wall_processes=sum(row['unknown_wall_processes'] for row in sources),
        budget_seconds=sum(row['budget_seconds'] for row in sources),gpu_jobs_submitted=0))
    print(json.dumps(read(output/'summary.json'),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refresh-run',action='append')
    inventory(parser.parse_args())

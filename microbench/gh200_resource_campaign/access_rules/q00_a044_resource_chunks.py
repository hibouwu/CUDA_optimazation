#!/usr/bin/env python3
"""保持原顺序，以原 receipt 进程时间切分新的资源重放包。"""
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def queries(tasks, family):
    first = tasks[0]
    root = Path(first['origins'][0]['run'])
    names = {'dsm_multicast': ['legacy-capabilities', 'measured-capabilities'],
             'fp32_combination': ['capability'],
             'bf16_wgmma_combination': ['legacy-capability', 'measured-capability']}.get(family, [])
    added = []
    for name in names:
        source = root / name / 'receipt.json'
        old = read(source)
        key = hashlib.sha256(str(source).encode()).hexdigest()[:20]
        assert old['returncode'] == 0
        origin = dict(run=str(root), receipt=str(source), original_argv=old['argv'],
            original_returncode=old['returncode'], original_start_ns=old['host_start_ns'],
            source_receipt_sha256=sha(source), point=dict(kind='resources', query=name))
        added.append(dict(id=family+'-'+key, stages=['RES-'+family],
            argv=[first['argv'][0], *old['argv'][1:]], repeats=1,
            timeout_seconds=old['timeout_seconds'],
            protocol=dict(profile='plain', original_kind='resources', packed_full_output=True,
                stdout_format='resource_jsonl', full_output='per_process_cwd',
                cache_and_warmup='original_binary_and_argv', published_case=False),
            origins=[origin], original_seconds=(old['host_stop_ns']-old['host_start_ns'])/1e9))
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, action='append', required=True)
    parser.add_argument('--family', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=float, default=600)
    args = parser.parse_args()
    original = read(args.package / 'manifest.json')
    times = {r['id']: r['seconds'] for path in args.runtime for r in read(path)['calls']}
    args.output.mkdir(parents=True, exist_ok=False)
    plan = []
    for family in args.family:
        tasks = [copy.deepcopy(t) for t in original['tasks'] if t['stages'] == ['RES-'+family]]
        extra = queries(tasks, family)
        for task in extra:
            times[task['id']] = task.pop('original_seconds')
        tasks = extra + tasks
        assert len({t['id'] for t in tasks}) == len(tasks)
        chunks, chunk, charge = [], [], 0
        for task in tasks:
            known = times[task['id']]
            seconds = known if known is not None else task['timeout_seconds']
            if chunk and charge + seconds > args.budget_seconds:
                chunks.append(chunk)
                chunk, charge = [], 0
            chunk.append(task)
            charge += seconds
        if chunk:
            chunks.append(chunk)
        assert [t['id'] for c in chunks for t in c] == [t['id'] for t in tasks]
        for index, chunk in enumerate(chunks):
            name = 'resource-' + family.replace('_', '-') + '-chunk-%02d' % index
            folder = args.output / name
            folder.mkdir()
            manifest = copy.deepcopy(original)
            checksums = {Path(t['argv'][0]).name for t in chunk}
            manifest.update(tasks=chunk, assets={k:v for k,v in original['assets'].items() if k in checksums},
                source_manifest_sha256=sha(args.package / 'manifest.json'),
                subset=dict(family=family, index=index, chunks=len(chunks),
                    original_seconds=sum(times[t['id']] or 0 for t in chunk),
                    unknown_seconds_budget=sum(t['timeout_seconds'] for t in chunk if times[t['id']] is None),
                    ordering='按原 manifest 顺序连续切分；原 packs 外能力查询置于该家族首部'))
            manifest['totals'] = dict(families=1, scheduled_processes=len(chunk),
                stages={family:len(chunk)}, resources_queries_added=sum(t in extra for t in chunk))
            manifest['coverage'] = [dict(family=family, status='partial_chunk',
                original_coverage=next(c for c in original['coverage'] if c['family'] == family))]
            save(folder / 'manifest.json', manifest)
            for script in ('q00_a044_remeasure.py','q00_a044_resources.py','l2_saturation.py'):
                shutil.copy2(HERE / script, folder / script)
            (folder / 'assets').mkdir()
            for checksum in checksums:
                os.link(args.package / 'assets' / checksum, folder / 'assets' / checksum)
            save(folder / 'package-identity.json', {str(p.relative_to(folder)):sha(p)
                 for p in folder.rglob('*') if p.is_file()})
            archive = folder.with_suffix('.tar.gz')
            with tarfile.open(archive, 'w:gz', compresslevel=1) as packed:
                packed.add(folder, arcname=folder.name)
            plan.append(dict(name=name, package=str(folder), archive=str(archive),
                archive_sha256=sha(archive), calls=len(chunk), **manifest['subset']))
            print(name,len(chunk),manifest['subset'],flush=True)
    save(args.output / 'chunks.json', plan)


if __name__ == '__main__':
    main()

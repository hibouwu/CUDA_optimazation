#!/usr/bin/env python3
"""将已恢复的原子轮次按试次或相邻组边界打包；每次原调用独立保留。"""
import argparse, collections, copy, json, re
from pathlib import Path
from q00_a044_remeasure import read, save
from q00_a044_ordered_legacy import package
HERE=Path(__file__).resolve().parent


def boundaries(row):
    origin=row['origin'];record=Path(origin['record']);parts=record.parts
    if row['trial'] is not None:
        phase=row['set'];trial=row['trial']
    else:
        name=record.parent.name
        match=re.fullmatch(r'(?:plain|trace[^/]*|dual|stamped|unstamped)-(\d+)',name)
        if match:phase='sample';trial=int(match[1])
        else:
            match=re.search(r'(check|pilot|trial|formal|smoke|main|representative|tracecheck)-(\d+)', '/'.join(parts[-4:]))
            if match:phase=match[1];trial=int(match[2])
            elif name.isdigit():phase='sample';trial=int(name)
            else:phase=name;trial=None
    case=origin.get('case')
    if not case and 'samples' in parts:
        ix=parts.index('samples');case=parts[ix+1]
        if case in ('formal','representatives','pilot','main'):case=parts[ix+2]
    if isinstance(case,dict):case=case.get('id') or json.dumps(case,sort_keys=True)
    case=case or str(record.parent)
    case=re.sub(r'^(plain|trace|dual|stamped|unstamped)_','',case).removesuffix('_control')
    return (phase,trial),(phase,trial,case)


def consecutive(rows, selector):
    groups=[];last=object()
    for row in rows:
        key=selector(row)
        if not groups or key!=last:groups.append([])
        groups[-1].append(row);last=key
    return groups


def split(rows):
    if sum(r['budget_seconds'] for r in rows)<=600:return [rows],'complete original subrun'
    groups=consecutive(rows,lambda r:boundaries(r)[0]);atomic=[];used_finer=False
    for group in groups:
        if sum(r['budget_seconds'] for r in group)<=600:atomic.append(group)
        else:
            used_finer=True
            for block in consecutive(group,lambda r:boundaries(r)[1]):
                cost=sum(r['budget_seconds'] for r in block)
                assert cost<=600,(cost,[r['origin']['record'] for r in block][:3])
                atomic.append(block)
    chunks=[];chunk=[];cost=0
    for group in atomic:
        c=sum(r['budget_seconds'] for r in group)
        if chunk and cost+c>600:chunks.append(chunk);chunk=[];cost=0
        chunk+=group;cost+=c
    if chunk:chunks.append(chunk)
    return chunks,('original adjacent case/control or variant group' if used_finer else 'complete original trial')


def main(args):
    args.output.mkdir(exist_ok=False)
    assets=read(args.input/'assets.json');pool=args.input/'asset-pool';(pool/'assets').mkdir(parents=True,exist_ok=True)
    for key,value in assets.items():
        p=pool/'assets'/key
        if not p.exists():p.symlink_to(Path(value['source']).resolve())
    parent=read(args.parent/'manifest.json');old_tasks={t['id']:t for t in parent['tasks']}
    groups=read(args.order/'groups.json');packages=[];covered=[]
    selected=set(args.source or [])
    run_identities={};aliases=[]
    for gi,group in enumerate(groups):
        sid=group['source_id']
        if selected and sid not in selected:continue
        if sid in (args.exclude_source or []):continue
        identity=tuple((tuple(r['argv']),r['origin']['original_stdout_sha256']) for r in group['rows'])
        if identity in run_identities:
            aliases.append(dict(duplicate=group['subrun'],original=run_identities[identity],calls=len(group['rows'])))
            continue
        run_identities[identity]=group['subrun']
        for si,row in enumerate(group['rows']):row['sequence_index']=si
        chunks,boundary=split(group['rows'])
        for chunk in chunks:
            tasks=[]
            for row in chunk:
                si=row['sequence_index'];protocol=copy.deepcopy(row['protocol']);protocol.update(original_sequence_index=si,original_sequence_run=group['subrun'])
                old=old_tasks.get(row['old_task_id']);timeout=old['timeout_seconds'] if old else 300
                task=dict(id=f'{sid}-g{gi:03}-{si:05}',stages=group['labels'],argv=row['argv'],repeats=1,timeout_seconds=timeout,protocol=protocol,origins=[row['origin']]);tasks.append(task)
                covered.append(dict(source_id=sid,subrun=group['subrun'],sequence_index=si,record=row['origin']['record'],case=row['origin'].get('case'),variant=row['origin'].get('variant'),trial=row['origin'].get('trial')))
            name=f'legacy-seq-{len(packages):03}'
            manifest=dict(schema=parent['schema'],source=group['run'],target_node=parent['target_node'],target_clock_mhz=parent['target_clock_mhz'],tasks=tasks,assets=assets,
                continuous_telemetry=not any(t['protocol'].get('external_nvsmi') for t in tasks),
                qualification='原子轮次完整设计回归；原二进制、参数、试次和外部操作保留',
                sequence=dict(order=group['order_evidence'],subrun=group['subrun'],first_index=chunk[0]['sequence_index'],last_index=chunk[-1]['sequence_index'],split_boundary=boundary),
                runtime=dict(original_wall_seconds=sum(r['original_wall_seconds'] or 0 for r in chunk),unknown_wall_processes=sum(r['original_wall_seconds'] is None for r in chunk),total_budget_seconds=sum(r['budget_seconds'] for r in chunk)))
            result=package(args.output/name,manifest,pool,HERE);result.update(source_id=sid,source=group['run'],labels=group['labels'],**manifest['runtime'],**manifest['sequence'],archive=str((args.output/(name+'.tar.gz')).resolve()))
            packages.append(result);print(name,sid,len(tasks),round(result['total_budget_seconds'],1),flush=True)
    save(args.output/'packages.json',packages);save(args.output/'coverage.json',covered);save(args.output/'whole-run-aliases.json',aliases)
    save(args.output/'summary.json',dict(packages=len(packages),processes=len(covered),original_wall_seconds=sum(p['original_wall_seconds'] for p in packages),budget_seconds=sum(p['total_budget_seconds'] for p in packages),archive_bytes=sum(p['archive_bytes'] for p in packages),gpu_jobs_submitted=0))
    print(json.dumps(read(args.output/'summary.json'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','order','parent','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',action='append');p.add_argument('--exclude-source',action='append');main(p.parse_args())

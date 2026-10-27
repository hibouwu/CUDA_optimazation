#!/usr/bin/env python3
"""从索引到调用审计准备缺失的机制序列；只做CPU准备。"""
import argparse,collections,functools,hashlib,json,shutil,tarfile
from pathlib import Path
from q00_a044_remeasure import read,save,inspect_stdout,digest as hashfile
from q00_a044_legacy_stdout import Text
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
digest=functools.lru_cache(maxsize=None)(hashfile)

def original(root,call):
 path=REPO/call['record'];row=read(path)
 if call['index'] is not None:row=row[call['index']]
 if path.name=='command.json':
  row=next(read(path.parent/name) for name in ('receipt.json','process.json','result.json') if (path.parent/name).exists())
 if row.get('raw'):
  candidates=[parent/row['raw'] for parent in [path.parent,*path.parents] if parent.is_relative_to(root)]
  raw=next(p for p in candidates if p.is_file());expected=row.get('raw_sha256')
 elif path.name=='invocations.json':
  expected=row.get('sha256',row.get('stdout_sha256'))
  candidates=list((path.parent/'samples').rglob('*'))
  raw=next(p for p in candidates if p.is_file() and p.suffix in ('.json','.jsonl','.gz','.txt') and digest(p)==expected)
 else:
  candidates=[path.parent/name for name in ('stdout','stdout.json','stdout.txt','stdout.txt.gz')]
  if path.name.endswith('.process.json'):candidates.insert(0,path.with_name(path.name.removesuffix('.process.json')+'.stdout'))
  raw=next(p for p in candidates if p.is_file());expected=row.get('stdout_sha256')
 if expected and digest(raw)!=expected:raise ValueError('原stdout SHA不符')
 start=call['original_start_ns'];stop=next((row[x] for x in ('host_stop_ns','ended_ns','stop_ns','finished_unix_ns','ended_unix_ns') if x in row),None)
 wall=(stop-start)/1e9 if stop and start else row.get('wall_seconds',row.get('duration_seconds',0))
 return row,raw,max(0,wall or 0)

def main(args):
 audit=read(args.audit);output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
 packages=[];omitted=[];checks=[];covered=[]
 for item in audit['roots']:
  if not any(e.startswith('Q') for e in item['entries']):continue
  if ('/Q16-' in item['run'] and not args.include_c1_history) or '/Q09-frequency/frequency-mechanism/' in item['run']:continue
  pending=[c for c in item.get('calls',[]) if c['coverage'] in ('missing','same_binary_argv_requires_protocol_check')]
  if not pending:continue
  root=REPO/item['run'];records=[]
  # Keep original successful sequence for this missing experiment, including its own repeated controls.
  for call in item['calls']:
   if call['coverage'] not in ('missing','same_binary_argv_requires_protocol_check'):continue
   if not call.get('normalized_argv'):continue
   try:
    row,raw,wall=original(root,call)
    argv=call['normalized_argv'];kind='program_query' if any(x in call['original_argv'] for x in ('--inspect-layout','--inspect-program','--resources','resources','device')) or ('--mode' in call['original_argv'] and call['original_argv'][call['original_argv'].index('--mode')+1]=='setup') else 'measurement'
    binary_name=Path(call['original_argv'][0]).name
    profile=next((x for x in ('plain','dual','wide','stamped') if binary_name.endswith('_'+x)),binary_name)
    protocol=dict(profile=profile,packed_full_output=True,original_kind=kind,
      cache_and_warmup='original_binary_and_argv',input_sha256={k:v['sha256'] for k,v in call['inputs'].items()},
      process_order='original_recorded_sequence',sequence_run=root.name)
    parsed=inspect_stdout(Text(raw),protocol)
    check=dict(run=item['run'],record=call['record'],index=call['index'],raw=str(raw.relative_to(REPO)),parsed=parsed)
    checks.append(check)
    if not parsed['valid']:raise ValueError('原stdout解析：'+parsed['status'])
    records.append(dict(call=call,raw=raw,wall=wall,protocol=protocol,row=row))
   except (OSError,KeyError,ValueError,StopIteration) as error:
    omitted.append(dict(run=item['run'],record=call['record'],index=call['index'],reason=type(error).__name__+':'+str(error)))
  if not records:continue
  records.sort(key=lambda x:x['call']['original_start_ns'] or 0)
  # Bound each wrapper; a single source sequence may be split only at process boundaries.
  chunks=[];chunk=[];cost=0
  for record in records:
   raw_size=len(Text(record['raw']).read_text().encode());record['raw_bytes']=raw_size
   estimate=1.5*record['wall']+.25+3*raw_size/(20*1024*1024)
   if chunk and cost+estimate>600:chunks.append(chunk);chunk=[];cost=0
   chunk.append(record);cost+=estimate
  if chunk:chunks.append(chunk)
  sequence_hash=hashlib.sha256(json.dumps([r['call']['normalized_argv'] for r in records]).encode()).hexdigest()
  for part,chunk in enumerate(chunks):
   name=f'index-{len(packages):03d}';folder=output/name;folder.mkdir();(folder/'assets').mkdir();tasks=[];assets={}
   for i,record in enumerate(chunk):
    call=record['call'];sha=call['binary_sha256'];assets[sha]=dict(source=call['binary'],kind='binary')
    for v in call['inputs'].values():assets[v['sha256']]=dict(source=v['source'],kind='input')
    origin=dict(run=item['run'],record=call['record'],record_index=call['index'],
     original_argv=call['original_argv'],original_start_ns=call['original_start_ns'],original_returncode=0,
     original_stdout=str(record['raw'].relative_to(REPO)),original_stdout_sha256=digest(record['raw']),
     original_record_sha256=digest(REPO/call['record']),prior_coverage=call['coverage'],prior_matches=call.get('matches',[]))
    protocol=dict(record['protocol'],sequence_sha256=sequence_hash)
    tasks.append(dict(id=f'{name}-{i:04d}',stages=[item['run'].split('access_rules/')[1].split('/')[0].split('-')[0]],
     argv=call['normalized_argv'],protocol=protocol,repeats=1,timeout_seconds=300,origins=[origin]))
   wall=sum(r['wall'] for r in chunk);rawbytes=sum(r['raw_bytes'] for r in chunk);budget=1.5*wall+.25*len(chunk)+3*rawbytes/(20*1024*1024)
   manifest=dict(schema='gh200-a044-remeasure/1',source=str(root),target_node='romeo-a044',target_clock_mhz=1980,
    tasks=tasks,assets=assets,qualification='原已执行机制条件的a044回归；原评分与新评分分开保存',
    coverage=[dict(run=item['run'],entries=item['entries'],ready_calls=len(tasks),status='ready')],omitted=[],
    totals=dict(unique_conditions=len({tuple(t['argv']) for t in tasks}),scheduled_processes=len(tasks)),
    sequence=dict(sha256=sequence_hash,part=part+1,parts=len(chunks),order='original recorded host start'),
    runtime=dict(original_wall_seconds=wall,raw_bytes=rawbytes,total_budget_seconds=budget))
   save(folder/'manifest.json',manifest)
   for sha,asset in assets.items():shutil.copy2(asset['source'],folder/'assets'/sha)
   for script in ('q00_a044_remeasure.py','l2_saturation.py'):shutil.copy2(HERE/script,folder/script)
   (folder/'README.md').write_text(f'# {root.name}\n\n原实验：`{item["run"]}`。本块保留 {len(tasks)} 个原成功进程及其先后顺序，输入文件按 SHA256 随包保存。\n\n原进程时间合计 {wall:.3f} 秒；采集和输出预算 {budget:.3f} 秒，网络另计。原始输出先通过同一 runner 的 CPU 解析，再准备本包。\n')
   save(folder/'package-identity.json',{str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file()})
   archive=folder.with_suffix('.tar.gz')
   with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for p in sorted(folder.rglob('*')):
     if p.is_file():tar.add(p,arcname=str(p.relative_to(folder)))
   packages.append(dict(batch=name,source=item['run'],processes=len(tasks),archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,total_budget_seconds=budget,part=part+1,parts=len(chunks)))
   print(name,root.name,len(tasks),round(budget,1),flush=True)
  covered.append(dict(run=item['run'],entries=item['entries'],pending=len(pending),prepared=len(records)))
 save(output/'packages.json',packages);save(output/'omitted.json',omitted);save(output/'stdout-checks.json',checks);save(output/'coverage.json',covered)
 summary=dict(packages=len(packages),processes=sum(p['processes'] for p in packages),source_runs=len(covered),omitted=len(omitted),gpu_jobs_submitted=0)
 save(output/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--include-c1-history',action='store_true');main(p.parse_args())

#!/usr/bin/env python3
"""恢复 execution/commands 数组中的原GPU调用，保持数组顺序和显式环境干预。"""
import argparse,collections,functools,hashlib,json,shutil,tarfile,datetime
from pathlib import Path
import q00_a044_remeasure as q
from q00_a044_legacy_stdout import Text
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];BASE=REPO/'results/gh200_resource_campaign/access_rules'
q.digest=functools.lru_cache(None)(q.digest)
def walk(value,location):
 if isinstance(value,dict):
  condition=value.get('condition',{})
  command=value.get('command',value.get('argv',condition.get('command') if isinstance(condition,dict) else None))
  if isinstance(command,list) and command and 'returncode' in value:
   yield location,{**(condition if isinstance(condition,dict) else {}),**value,'command':command}
  for k,v in value.items():yield from walk(v,location+[k])
 elif isinstance(value,list):
  for i,v in enumerate(value):yield from walk(v,location+[i])
def main(args):
 output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);packages=[];omitted=[];audit=[]
 for root in sorted(BASE.glob('Q*/*/20*')):
  if root.name.startswith('20261023-Q15-a044') or root.name.startswith('20261023-Q16-a044'):continue
  files=[p for p in root.rglob('*') if p.is_file()];binaries=[p for p in files if p.suffix not in ('.json','.jsonl','.gz','.cu','.py','.md','.log','.sass') and q.executable(p)]
  records=[];seen=set()
  for path in files:
   if path.name not in ('execution.json','records.json','calls.json','commands.json','state.json'):continue
   if set(path.relative_to(root).parts[:-1]).intersection({'source','reference','baseline','source-before','before','build','samples','raw','reanalysis'}):continue
   try:value=q.read(path)
   except json.JSONDecodeError:continue
   for loc,row in walk(value,[]):
    argv=list(map(str,row['command']));name=Path(argv[0]).name
    if name in ('ptxas','python','python3','nvcc','cuobjdump','srun','bash','sh','scontrol','nvidia-smi','g++','gcc'):continue
    origin=dict(run=str(root.relative_to(REPO)),record=str(path.relative_to(REPO)),record_location=loc,original_argv=argv,original_environment=row.get('environment',{}),original_returncode=row['returncode'],original_id=row.get('id',row.get('condition',{}).get('id')))
    if row['returncode']!=0:omitted.append(dict(**origin,reason='原失败过程保留'));continue
    try:
     if row.get('raw'):
      raw=next(p/row['raw'] for p in (path.parent,*path.parents) if p.is_relative_to(root) and (p/row['raw']).is_file());rawsha=row.get('raw_sha256',row.get('sha256'))
     else:
      rawrel=next(n for n in row['files'] if n.endswith('stdout.json'));raw=next(p/rawrel for p in (path.parent,*path.parents) if p.is_relative_to(root) and (p/rawrel).is_file());rawsha=row['files'][rawrel]
     if str(raw) in seen:continue
     seen.add(str(raw));assert q.digest(raw)==rawsha
     binary,sha=q.resolve_file(argv[0],binaries,binary=True,context=path)
     if row.get('binary_sha256'):assert sha==row['binary_sha256']
     assets={sha:dict(source=str(binary),kind='binary')};normalized=['assets/'+sha]
     for i,val in enumerate(argv[1:],1):
      if argv[i-1] in ('--check-output','--output','--out'):
       normalized.append('outputs/'+Path(val).name)
      elif '/' in val and not val.startswith('--'):
       src,h=q.resolve_file(val,files,context=path);assets[h]=dict(source=str(src),kind='input');normalized.append('assets/'+h)
      else:normalized.append(val)
     env=row.get('environment',{})
     if env:normalized=['/usr/bin/env',*[k+'='+v for k,v in sorted(env.items())],*normalized]
     kind='correctness' if any(v.startswith('check-') for v in argv) else 'measurement'
     protocol=dict(profile=row.get('variant',name),packed_full_output=True,original_kind=kind,cache_and_warmup='original_binary_and_argv',process_order='original_recorded_sequence',original_environment=env)
     parsed=q.inspect_stdout(Text(raw),protocol);assert parsed['valid'],parsed
     origin.update(original_stdout=str(raw.relative_to(REPO)),original_stdout_sha256=rawsha,original_record_sha256=q.digest(path))
     wall=0
     if row.get('started_utc') and row.get('completed_utc'):wall=(datetime.datetime.fromisoformat(row['completed_utc'])-datetime.datetime.fromisoformat(row['started_utc'])).total_seconds()
     records.append(dict(argv=normalized,protocol=protocol,origin=origin,assets=assets,wall=wall,raw_bytes=len(Text(raw).read_text().encode())))
    except (ValueError,KeyError,OSError,StopIteration,AssertionError) as err:omitted.append(dict(**origin,reason=repr(err)))
  if not records:continue
  sequence=hashlib.sha256(json.dumps([r['argv'] for r in records]).encode()).hexdigest()
  for offset in range(0,len(records),180):
   chunk=records[offset:offset+180];name=f'nested-{len(packages):03d}';folder=output/name;folder.mkdir();(folder/'assets').mkdir();assets={};tasks=[]
   for i,r in enumerate(chunk):
    assets.update(r['assets']);tasks.append(dict(id=f'{name}-{i:03d}',stages=[root.relative_to(BASE).parts[0].split('-')[0]],argv=r['argv'],repeats=1,timeout_seconds=180,protocol=dict(r['protocol'],sequence_sha256=sequence),origins=[r['origin']]))
   budget=sum(max(2,1.5*r['wall'])+.25+3*r['raw_bytes']/(20*1024*1024) for r in chunk)
   m=dict(schema='gh200-a044-remeasure/1',source=str(root),target_node='romeo-a044',target_clock_mhz=1980,tasks=tasks,assets=assets,coverage=[dict(run=str(root.relative_to(REPO)),ready_calls=len(tasks))],omitted=[],totals=dict(unique_conditions=len({tuple(t['argv']) for t in tasks}),scheduled_processes=len(tasks)),sequence=dict(sha256=sequence,source_order='original metadata array',offset=offset,total=len(records)),runtime=dict(total_budget_seconds=budget,unknown_duration_budget_seconds=2))
   q.save(folder/'manifest.json',m)
   for sha,a in assets.items():shutil.copy2(a['source'],folder/'assets'/sha)
   for script in ('q00_a044_remeasure.py','l2_saturation.py'):shutil.copy2(HERE/script,folder/script)
   q.save(folder/'package-identity.json',{str(p.relative_to(folder)):q.digest(p) for p in folder.rglob('*') if p.is_file()})
   with tarfile.open(folder.with_suffix('.tar.gz'),'w:gz',compresslevel=1) as tar:
    for p in sorted(folder.rglob('*')):
     if p.is_file():tar.add(p,arcname=str(p.relative_to(folder)))
   packages.append(dict(batch=name,source=str(root.relative_to(REPO)),processes=len(tasks),archive_sha256=q.digest(folder.with_suffix('.tar.gz')),archive_bytes=folder.with_suffix('.tar.gz').stat().st_size,total_budget_seconds=budget));print(name,root.name,len(tasks),flush=True)
  audit.extend(r['origin'] for r in records)
 q.save(output/'packages.json',packages);q.save(output/'omitted.json',omitted);q.save(output/'source-calls.json',audit)
 summary=dict(packages=len(packages),processes=sum(p['processes'] for p in packages),omitted=len(omitted),gpu_jobs_submitted=0);q.save(output/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

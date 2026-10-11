import gzip,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();m=json.loads(Path('manifest.json').read_text());rows={r['id']:r for r in json.loads(Path('cases.json').read_text())}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in json.loads(Path('freeze.json').read_text())['files'].items():
 if sha(Path(name))!=digest:raise RuntimeError('frozen execution input changed: '+name)
for name,digest in m['analysis_sha256'].items():
 if sha(Path('analysis_snapshot')/name)!=digest:raise RuntimeError('analysis snapshot changed')
sys.path.insert(0,str(root/'analysis_snapshot'));from run_r18 import case_args
for name,digest in m['binary_sha256'].items():
 if sha(Path('build')/name)!=digest:raise RuntimeError('binary identity differs')
if not os.environ.get('CUDA_VISIBLE_DEVICES') or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('missing assigned CUDA visibility')
original={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES')}
def events(data):return {v['event']:v for l in data.decode().splitlines() if l.startswith('{') for v in [json.loads(l)]}
Path('device-setup').mkdir()
setup_row=next(r for r in rows.values() if r['stages']==6 and r['release_warp_observer']==0)
assigned=[]
for index,token in enumerate(original['CUDA_VISIBLE_DEVICES'].split(',')):
 if not token.strip():raise RuntimeError('empty assigned device token')
 probe_env=dict(os.environ,CUDA_VISIBLE_DEVICES=token.strip())
 p=subprocess.run(['build/cfg_b_s6_plain','--mode','setup',*case_args(setup_row)],env=probe_env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 Path(f'device-setup/{index}.jsonl').write_bytes(p.stdout)
 Path(f'device-setup/{index}.stderr').write_bytes(p.stderr)
 if p.returncode:raise RuntimeError('allocated device setup failed: '+token)
 setup=events(p.stdout)['setup']
 assigned.append(dict(assigned_visibility_token=token.strip(),gpu_uuid=setup['gpu_uuid']))
Path('assigned-devices.json').write_text(json.dumps(dict(original=original,allocated_devices=assigned,selected=m['gpu_uuid']),indent=2)+'\n')
if m['gpu_uuid'] not in [d['gpu_uuid'] for d in assigned]:raise RuntimeError('requested GPU is not in this step assigned CUDA visibility')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=m['gpu_uuid']);Path('raw').mkdir();Path('full-d').mkdir()
checks=[]
for stage in (4,6):
 row=next(r for r in rows.values() if r['stages']==stage and r['release_warp_observer']==0)
 for variant in ('plain','stamped'):
  name=f'cfg_b_s{stage}_{variant}';cmd=['build/'+name,'--mode','check-release-warps','--check-output','full-d/'+name+'.bin',*case_args(row)]
  p=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE);Path('raw/'+name+'_check.jsonl').write_bytes(p.stdout);Path('raw/'+name+'_check.stderr').write_bytes(p.stderr)
  es=events(p.stdout)
  if p.returncode or es['setup']['gpu_uuid']!=m['gpu_uuid'] or es['release_warps_check']['different'] or es['release_warps_check']['nonfinite']:raise RuntimeError('full D off/on or GPU check failed')
  checks.append(dict(binary=name,elements=es['release_warps_check']['elements'],D_sha256=sha(Path('full-d')/(name+'.bin'))))
 if (Path('full-d')/f'cfg_b_s{stage}_plain.bin').read_bytes()!=(Path('full-d')/f'cfg_b_s{stage}_stamped.bin').read_bytes():raise RuntimeError('full D plain/stamped mismatch')
Path('full-D-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
records=[]
for sample in m['samples']:
 row=rows[sample['case']];cmd=['build/'+sample['binary'],'--mode','warm',*case_args(row)]
 p=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 raw=Path('raw')/(sample['id']+'.jsonl.gz');raw.write_bytes(gzip.compress(p.stdout,mtime=0));Path('raw/'+sample['id']+'.stderr').write_bytes(p.stderr)
 records.append(dict(sample,raw=str(raw),raw_sha256=sha(raw),returncode=p.returncode))
 Path('records.json').write_text(json.dumps(records,indent=2)+'\n')
Path('collection-complete.json').write_text(json.dumps(dict(samples=len(records),nonzero_processes=sum(r['returncode']!=0 for r in records),manifest_sha256=sha(Path('manifest.json'))),indent=2)+'\n')

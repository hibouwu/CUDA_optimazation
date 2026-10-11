import gzip,hashlib,json,os,subprocess
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads(Path('manifest.json').read_text())
for p,h in json.loads(Path('freeze.json').read_text())['files'].items():
 if sha(p)!=h:raise RuntimeError('frozen input changed: '+p)
if sha('build/cfg_a_s6_dual')!=m['binary_sha256']:raise RuntimeError('binary changed')
for p,h in m['analysis_sha256'].items():
 if sha('analysis_snapshot/'+p)!=h:raise RuntimeError('analysis changed')
original={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_STEP_GPUS')}
if not original['CUDA_VISIBLE_DEVICES'] or not original['SLURM_JOB_ID']:raise RuntimeError('missing allocated visibility')
def events(data):return {x['event']:x for line in data.decode().splitlines() if line.startswith('{') for x in [json.loads(line)]}
Path('device-setup').mkdir();assigned=[]
for i,token in enumerate(original['CUDA_VISIBLE_DEVICES'].split(',')):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=token)
 p=subprocess.run(m['args']+['--mode','setup'],env=env,capture_output=True)
 Path(f'device-setup/{i}.jsonl').write_bytes(p.stdout);Path(f'device-setup/{i}.stderr').write_bytes(p.stderr)
 if p.returncode:raise RuntimeError('assigned setup failed')
 assigned.append(dict(token=token,gpu_uuid=events(p.stdout)['setup']['gpu_uuid']))
Path('assigned-devices.json').write_text(json.dumps(dict(original=original,devices=assigned),indent=2)+'\n')
if m['gpu_uuid'] not in [x['gpu_uuid'] for x in assigned]:raise RuntimeError('target is not allocated')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=m['gpu_uuid']);Path('raw').mkdir();records=[]
for sample in m['samples']:
 p=subprocess.run(m['args']+['--mode','warm','--first-output-observer',str(sample['flag'])],env=env,capture_output=True)
 raw=Path('raw')/(sample['id']+'.jsonl.gz');raw.write_bytes(gzip.compress(p.stdout,mtime=0));Path('raw',sample['id']+'.stderr').write_bytes(p.stderr)
 records.append(dict(sample,raw=str(raw),sha256=sha(raw),returncode=p.returncode))
 Path('records.json').write_text(json.dumps(records,indent=2)+'\n')
Path('complete.json').write_text(json.dumps(dict(manifest_sha256=sha('manifest.json'),records=len(records)))+'\n')

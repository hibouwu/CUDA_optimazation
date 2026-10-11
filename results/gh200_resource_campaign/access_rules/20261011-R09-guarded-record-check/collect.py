import gzip,hashlib,json,os,subprocess
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads(Path('manifest.json').read_text())
for p,h in json.loads(Path('freeze.json').read_text())['files'].items():
 if sha(p)!=h:raise RuntimeError('frozen input changed')
for p,h in m['binaries'].items():
 if sha('build/'+p)!=h:raise RuntimeError('binary/module changed')
if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('missing assigned device visibility')
Path('assigned-env.json').write_text(json.dumps({k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_STEP_GPUS')},indent=2)+'\n')
Path('raw').mkdir();records=[]
for sample in m['samples']:
 p=subprocess.run(sample['command'],capture_output=True)
 raw=Path('raw')/(sample['arm']+'.jsonl.gz');raw.write_bytes(gzip.compress(p.stdout,mtime=0));Path('raw',sample['arm']+'.stderr').write_bytes(p.stderr)
 records.append(dict(sample,raw=str(raw),sha256=sha(raw),returncode=p.returncode))
 Path('records.json').write_text(json.dumps(records,indent=2)+'\n')

from pathlib import Path
import subprocess,json,gzip,hashlib,os
m=json.loads(Path('manifest.json').read_text());Path('raw').mkdir();records=[]
assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'allocated CUDA devices required'
Path('assigned-context.json').write_text(json.dumps({k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES')},indent=2))
for s in m['samples']:
 p=subprocess.run(s['command'],capture_output=True);raw=Path('raw')/(s['id']+'.jsonl.gz');raw.write_bytes(gzip.compress(p.stdout,mtime=0));Path('raw',s['id']+'.stderr').write_bytes(p.stderr)
 records.append(dict(s,returncode=p.returncode,raw=str(raw),sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
 Path('records.json').write_text(json.dumps(records,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)

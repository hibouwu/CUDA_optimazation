"""Serial fixed collection; never retry a failed target or change its order."""
from pathlib import Path
import gzip,hashlib,json,os,subprocess
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads(Path('manifest.json').read_text())
if manifest['status']!='frozen':raise ValueError('not frozen')
for name,digest in json.loads(Path('freeze.json').read_text())['files'].items():
    if sha(Path(name))!=digest:raise ValueError('frozen input mismatch')
if sha(Path('build/cfg_c_s4_dual'))!=manifest['binary_sha256']:raise ValueError('binary mismatch')
if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('allocated devices required')
Path('assigned-context.json').write_text(json.dumps({k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES')},indent=2)+'\n')
# setup validates the actual CUDA device, not a presumed nvidia-smi index mapping.
command=list(manifest['samples'][0]['command'])
command[command.index('--mode')+1]='setup'
check=subprocess.run(command,capture_output=True)
Path('setup.stdout').write_bytes(check.stdout);Path('setup.stderr').write_bytes(check.stderr)
if check.returncode:raise SystemExit(check.returncode)
packets=[json.loads(line) for line in check.stdout.splitlines()]
setup=next(p for p in packets if p['event']=='setup')
if setup['gpu_uuid']!=manifest['gpu_uuid']:raise ValueError('actual device differs from frozen reference')
Path('raw').mkdir();records=[]
for sample in manifest['samples']:
    run=subprocess.run(sample['command'],capture_output=True)
    raw=Path('raw')/(sample['id']+'.jsonl.gz')
    raw.write_bytes(gzip.compress(run.stdout,mtime=0))
    Path('raw',sample['id']+'.stderr').write_bytes(run.stderr)
    records.append(dict(sample,returncode=run.returncode,raw=str(raw),sha256=sha(raw)))
    Path('records.json').write_text(json.dumps(records,indent=2)+'\n')
    if run.returncode:raise SystemExit(run.returncode)
Path('complete.json').write_text(json.dumps(dict(manifest_sha256=sha(Path('manifest.json')),records=len(records)))+'\n')

"""Serial fixed collection; never retry a failed target or change its order."""
from pathlib import Path
import gzip,hashlib,json,os,subprocess,sys
sys.path.insert(0,str(Path("analysis_snapshot").resolve()))
from decode_r13_wait_feedback import decode
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads(Path('manifest.json').read_text())
if manifest['status']!='frozen':raise ValueError('not frozen')
for name,digest in json.loads(Path('freeze.json').read_text())['files'].items():
    if sha(Path(name))!=digest:raise ValueError('frozen input mismatch')
for name,digest in manifest['binary_sha256'].items():
    if sha(Path('build')/name)!=digest:raise ValueError('binary mismatch')
for name,mode in json.loads(Path('freeze.json').read_text())['executables'].items():
    if Path(name).stat().st_mode & 0o777 != mode:raise ValueError('binary executable metadata mismatch')
if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('allocated devices required')
Path('assigned-context.json').write_text(json.dumps({k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES')},indent=2)+'\n')
# setup validates the actual CUDA device, not a presumed nvidia-smi index mapping.
command=list(manifest['samples'][0]['command'])
command[command.index('--mode')+1]='setup'
# Correctness export belongs only to the later full-D command.
if '--check-output' in command:
    pos=command.index('--check-output');del command[pos:pos+2]
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
    result=decode([json.loads(line) for line in run.stdout.splitlines()],manifest['gpu_uuid'])
    if result['stages']!=sample['stage'] or result['observer']!=sample['flag']:
        raise ValueError('sample-to-protocol mismatch')
    Path('raw',sample['id']+'.decoded.json').write_text(json.dumps(result,indent=2)+'\n')
Path('complete.json').write_text(json.dumps(dict(manifest_sha256=sha(Path('manifest.json')),records=len(records)))+'\n')

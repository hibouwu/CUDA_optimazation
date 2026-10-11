from pathlib import Path
import json,hashlib,os,subprocess
root=Path.cwd();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads(Path('manifest.json').read_text())
for name,digest in json.loads(Path('freeze.json').read_text())['files'].items():assert sha(Path(name))==digest
assert sha(Path('build/cfg_c_mma_reference'))==m['binary_sha256']
assert os.environ.get('CUDA_VISIBLE_DEVICES')
Path('assigned-context.json').write_text(json.dumps({k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_STEP_GPUS','SLURM_JOB_GPUS','CUDA_VISIBLE_DEVICES')},indent=2)+'\n')
records=[]
for sample in m['samples']:
 directory=Path('samples')/sample['id'];directory.mkdir(parents=True)
 result=subprocess.run(sample['command'],cwd=directory,capture_output=True)
 (directory/'stdout.json').write_bytes(result.stdout);(directory/'stderr.txt').write_bytes(result.stderr)
 record=dict(sample,returncode=result.returncode,files={str(p):sha(p) for p in directory.iterdir() if p.is_file()})
 records.append(record);Path('records.json').write_text(json.dumps(records,indent=2)+'\n')
 if result.returncode:raise SystemExit(result.returncode)
Path('complete.json').write_text(json.dumps(dict(records=6,manifest_sha256=sha(root/'manifest.json')))+'\n')

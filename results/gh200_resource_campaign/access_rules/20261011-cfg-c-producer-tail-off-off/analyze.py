"""Validate frozen acquisition before applying the incremental-only evaluator."""
from pathlib import Path
import gzip
import hashlib
import json
import sys

root=Path(__file__).resolve().parent
data=root/'returned'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((root/'manifest.json').read_text())
if manifest['status']!='frozen':raise ValueError('acquisition is not frozen')
freeze=json.loads((root/'freeze.json').read_text())
for name,digest in freeze['files'].items():
    if sha(root/name)!=digest or sha(data/name)!=digest:raise ValueError('frozen input mismatch: '+name)
receipt=json.loads((root/'receipt.json').read_text())
if receipt['returncode']!=0 or sha(root/'output.tar.gz')!=receipt['sha256']:raise ValueError('incomplete returned archive')
if sha(data/'build/cfg_c_s4_dual')!=manifest['binary_sha256']:raise ValueError('binary mismatch')
sys.path.insert(0,str(root/'analysis_snapshot'))
from analyze_cfg_c_tail_control import process,evaluate
records=json.loads((data/'records.json').read_text())
if [{k:r[k] for k in ('id','pair','flag','command','arm')} for r in records]!=manifest['samples']:
    raise ValueError('complete ordered manifest samples required')
if json.loads((data/'complete.json').read_text())!=dict(manifest_sha256=sha(root/'manifest.json'),records=20):
    raise ValueError('completion marker mismatch')
from validate_allocation import validate
validate((data/'allocation.txt').read_text())
processes=[]
for row in records:
    path=data/row['raw']
    if row['returncode']!=0 or sha(path)!=row['sha256']:raise ValueError('raw identity mismatch')
    with gzip.open(path,'rt') as stream:events=[json.loads(line) for line in stream]
    processes.append(process(events,row,manifest['gpu_uuid']))
result=evaluate(processes,manifest)
result['manifest_sha256']=sha(root/'manifest.json')
result['processes_verified']=len(processes)
result['series_per_process']=len(processes[0]['series'])
(root/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='local'}))

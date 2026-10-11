#!/usr/bin/env python3
"""Bind archived identities, replay sampled numbers and validate new packets."""
from pathlib import Path
import json,gzip,hashlib,sys
r=Path(__file__).resolve().parent;d=r/'returned'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'manifest.json').read_text());build=Path(m['build_run'])
assert sha(build/'source_hashes.json')==m['source_manifest_sha256']
assert sha(build/'build-audit.json')==m['build_audit_sha256']
for path,h in json.loads((build/'source_hashes.json').read_text()).items():assert sha(build/path)==h,path
for path,h in json.loads((r/'freeze.json').read_text())['files'].items():assert sha(r/path)==h and sha(d/path)==h,path
for path,h in m['binaries'].items():assert sha(r/'build'/path)==h and sha(d/'build'/path)==h,path
for path,h in m['analysis_sha256'].items():assert sha(r/'analysis_snapshot'/path)==h,path
receipt=json.loads((r/'receipt.json').read_text());assert receipt['returncode']==0 and sha(r/'output.tar.gz')==receipt['sha256']
sys.path.insert(0,str(r/'analysis_snapshot'))
from decode_r09_clock_observer import decode
from analyze_r18 import reference
records=json.loads((d/'records.json').read_text());assert [{k:v[k] for k in ('arm','command')} for v in records]==m['samples']
results=[];errors=[];gpu=None
for row in records:
    try:
        assert row['returncode']==0
        raw=d/row['raw'];assert sha(raw)==row['sha256']
        with gzip.open(raw,'rt') as stream:events={v['event']:v for line in stream for v in [json.loads(line)]}
        setup,call,check=events['setup'],events['call'],events['check']
        if gpu is None:gpu=setup['gpu_uuid']
        assert setup['gpu_uuid']==gpu
        assert check['samples']==4096 and len(check['checked_indices'])==len(check['checked_values'])==4096
        assert not check['nonfinite'] and not check['padding_errors']
        assert all(value==reference(index//1536,index%1536,4096,17) for index,value in zip(check['checked_indices'],check['checked_values']))
        result=decode(setup,call,row['arm'],gpu)
        results.append(dict(arm=row['arm'],gpu_uuid=gpu,numerical_values=4096,
            resources={k:setup[k] for k in ('smem','smem_epilogue','max_active_ctas_per_sm','sm_count','scheduler_sm_count')},
            raw_sha256=row['sha256'],decoded=result))
    except Exception as error:errors.append(dict(arm=row['arm'],error=repr(error)))
report=dict(results=results,errors=errors,all_records_valid=not errors and len(results)==3 and all(v['decoded']['record_valid'] for v in results),
    full_D_verified=False,observation_qualified=False,
    limits='One call per arm; sampled numerical and raw packet validation only. Same module/prefix identity is not perturbation or hardware-enclosure evidence.')
(r/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(errors=errors,all_records_valid=report['all_records_valid'],arms=[dict(arm=v['arm'],records=len(v['decoded']['records']),valid=v['decoded']['record_valid'],resources=v['resources']) for v in results]),indent=2))

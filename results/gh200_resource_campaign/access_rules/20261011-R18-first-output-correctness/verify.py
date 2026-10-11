#!/usr/bin/env python3
"""Replay full numerical witness and four raw traces; no observation qualification."""
from pathlib import Path
import gzip, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
sys.path.insert(0,str(REPO/'microbench/gh200_resource_campaign/access_rules'))
from analyze_r18 import reference
from decode_r18_first_output import decode
expected=np.array([[reference(i,j,4096,17) for j in range(17)] for i in range(17)],dtype=np.float32)
expected=expected[np.arange(960)[:,None]%17,np.arange(2816)[None,:]%17]
report=dict(full_D=[],records=[],errors=[],observation_qualified=False)
manifest=json.loads((ROOT/'manifest.json').read_text())
assert hashlib.sha256((ROOT/'returned/build/cfg_a_s6_dual').read_bytes()).hexdigest()==manifest['binary_sha256']
for extent in (960,1024):
    path=ROOT/f'returned/full-d/map{extent}.bin'
    data=np.fromfile(path,dtype='<f4').reshape(960,2816)
    count=int(np.count_nonzero(data!=expected));assert count==0
    report['full_D'].append(dict(map=extent,elements=data.size,mismatches=count,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    events=[json.loads(line) for line in (ROOT/f'returned/raw/map{extent}_full_check.jsonl').read_text().splitlines()]
    check=next(row for row in events if row['event']=='first_output_check')
    assert check['elements']==data.size and check['different']==0 and check['nonfinite']==0
    for flag in (0,1):
        path=ROOT/f'returned/raw/map{extent}_flag{flag}.jsonl.gz'
        with gzip.open(path,'rt') as stream:events=[json.loads(line) for line in stream]
        setup=next(row for row in events if row['event']=='setup')
        call=next(row for row in events if row['event']=='call')
        decoded=decode(setup,call,f'map{extent}_flag{flag}')
        report['records'].append(dict(map=extent,flag=flag,gpu_uuid=setup['gpu_uuid'],elapsed_us=call['elapsed_us'],decoded=decoded,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert report['full_D'][0]['sha256']==report['full_D'][1]['sha256']
(ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('Full D: 5406720 exact reference values; traces: 4; on writers: 24 + 24; qualified: false')

#!/usr/bin/env python3
from pathlib import Path
import gzip,hashlib,json,sys
import numpy as np
r=Path(__file__).resolve().parent;d=r/'returned';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'manifest.json').read_text());build=Path(m['build_run'])
assert sha(build/'build-audit.json')==m['build_audit_sha256']
for name,h in json.loads((build/'freeze.json').read_text())['files'].items():assert sha(build/name)==sha(build/'returned'/name)==h
build_receipt=json.loads((build/'receipt.json').read_text());assert build_receipt['returncode']==0 and sha(build/'output.tar.gz')==build_receipt['sha256']
assert sha(build/'returned/build/cfg_c_s4_dual')==sha(d/'build/cfg_c_s4_dual')==m['binary_sha256']
for name,h in json.loads((build/'source_hashes.json').read_text()).items():assert sha(build/name)==h
for name,h in json.loads((r/'freeze.json').read_text())['files'].items():assert sha(r/name)==sha(d/name)==h
for name,h in m['analysis_sha256'].items():assert sha(r/'analysis_snapshot'/name)==sha(d/'analysis_snapshot'/name)==h
receipt=json.loads((r/'receipt.json').read_text());assert receipt['returncode']==0 and sha(r/'output.tar.gz')==receipt['sha256']
sys.path.insert(0,str(r/'analysis_snapshot'))
from analyze_r18 import reference
from decode_cfg_c_producer_tail import decode
records=json.loads((d/'records.json').read_text());assert [{k:v[k] for k in ('id','command')} for v in records]==m['samples']
context=json.loads((d/'assigned-context.json').read_text());assert context['CUDA_VISIBLE_DEVICES'] and len(context['CUDA_VISIBLE_DEVICES'].split(','))==1
expected=np.array([[reference(i,j,4096,17) for j in range(17)] for i in range(17)],dtype='<f4')
data=np.fromfile(d/'full-d.bin',dtype='<f4').reshape(2304,3072)
assert np.all(np.isfinite(data)) and np.array_equal(data,expected[np.arange(2304)[:,None]%17,np.arange(3072)[None,:]%17])
results=[];gpu=None
for row in records:
    raw=d/row['raw'];assert row['returncode']==0 and sha(raw)==row['sha256']
    with gzip.open(raw,'rt') as f:packets=[json.loads(line) for line in f]
    events={p['event']:p for p in packets};assert len(events)==len(packets)
    setup=events['setup'];gpu=gpu or setup['gpu_uuid'];assert setup['gpu_uuid']==gpu
    if row['id']=='full_D':
        c=events['producer_tail_check'];assert c['elements']==data.size and c['comparison_flag']==1 and c['different']==c['nonfinite']==0
        results.append(dict(id=row['id'],raw_sha256=sha(raw),elements=data.size,off_on_bit_equal=True,exact_reference=True));continue
    c=events['check'];assert c['samples']==4096 and len(set(c['checked_indices']))==len(c['checked_values'])==4096 and not c['nonfinite'] and not c['padding_errors']
    assert all(type(i) is int and 0<=i<2304*3072 and v==reference(i//3072,i%3072,4096,17) for i,v in zip(c['checked_indices'],c['checked_values']))
    decoded=decode(setup,events['call'],gpu);assert decoded['record_valid'],decoded['errors']
    results.append(dict(id=row['id'],raw_sha256=sha(raw),decoded=decoded,numerical_values=4096))
report=dict(job=740088,gpu_uuid=gpu,allocation=context,full_D_sha256=sha(d/'full-d.bin'),results=results,observation_qualified=False,
    limits='One correctness comparison and one warm record per arm. No perturbation distribution, hardware read enclosure, service price or kernel-exit claim.')
(r/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(gpu_uuid=gpu,full_D_elements=data.size,warm_records=2,CTAs_per_record=132,sampled_values=8192,observation_qualified=False)))

"""Recompute the frozen analysis and independently check its contrasts from raw slots."""
from pathlib import Path
import gzip
import hashlib
import json
import statistics
import tarfile
from analyze import summarize, SERIES

root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'manifest.json').read_text())
freeze=json.loads((root/'freeze.json').read_text())
verified=json.loads((root/'verification.json').read_text())
assert verified['manifest_sha256']=='101e871485f2748eb2787d80ffbf6c0f1f5bfa979642bb633515fd28b42c3245'
assert verified['freeze_sha256']=='1cc512e256ca560af14a1f1c61b5807200d008af7e3918a540ca89f0e6762bc5'
assert sha(root/'analyze.py')==freeze['files']['analyze.py']
assert sha(root/'output.tar.gz')==verified['archive_sha256']
records=verified['records']
assert len(records)==96 and len({r['sm'] for r in records})==1
for actual,wanted in zip(records,manifest['samples'],strict=True):
    assert (actual['id'],actual['block'],actual['position'])==(wanted['id'],wanted['block'],wanted['position'])
result=summarize(records)
raw_points={}
with tarfile.open(root/'output.tar.gz') as archive:
    assert result==json.load(archive.extractfile('./analysis.json'))
    for sample in manifest['samples']:
        payload=archive.extractfile('./raw/'+sample['id']+'.jsonl.gz').read()
        packets=[json.loads(line) for line in gzip.decompress(payload).decode().splitlines()]
        trace=next(p for p in packets if p['event']=='call')['trace']
        raw_points[sample['block'],sample['stage'],sample['flag']]=(trace[41],trace[42])
for block,contrasts in enumerate(result['blocks']):
    for stage in (4,6):
        a,b=raw_points[block,stage,1]
        c,_=raw_points[block,stage,2]
        _,d=raw_points[block,stage,3]
        assert [contrasts[f'stage{stage}_{name}'] for name in ('L16','L2','F')]==[b-d,a-c,(b+c)-(a+d)]
for name in SERIES:
    values=sorted(row[name] for row in result['blocks'])
    assert result['summary'][name]['median']==(values[7]+values[8])/2
    assert result['summary'][name]['conditional_median_interval']==[values[2],values[13]]
result.update(archive_sha256=verified['archive_sha256'],verified_records_sha256=sha(root/'verification.json'),
    independent_raw_slot_contrasts_checked=16*6,independently_checked_values=verified['independently_checked_values'],
    sign_counts={name:dict(positive=sum(row[name]>0 for row in result['blocks']),
                           zero=sum(row[name]==0 for row in result['blocks']),
                           negative=sum(row[name]<0 for row in result['blocks'])) for name in SERIES},
    empirical_cells={f's{stage}_m{mode}':{key:statistics.median(r[key] for r in records if r['stages']==stage and r['observer']==mode)
        for key in ('early_cycles','selected_wait_cycles')} for stage in (4,6) for mode in (1,2,3)},
    scorer_sha256=sha(Path(__file__)))
with (root/'analysis.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(summary=result['summary'],sign_counts=result['sign_counts'],observation_qualified=False),indent=2))

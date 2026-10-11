"""Recompute the fixed 64-call diagnostic from a caller-pinned acquisition archive."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile

sha=lambda data:hashlib.sha256(data).hexdigest()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive-sha256',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
assert sha((root/'output.tar.gz').read_bytes())==args.archive_sha256
manifest_bytes=(root/'manifest.json').read_bytes()
freeze_bytes=(root/'freeze.json').read_bytes()
assert sha(manifest_bytes)=='655a7dd644ce4c46899d8648e63ffe651bdad8674221b17cc18bef934c5700f4'
assert sha(freeze_bytes)=='08e9ee5b9f1cbb17d770f73ce7fcacd80442a0d07772b31e40ec01dc6cf8b4da'
manifest=json.loads(manifest_bytes);freeze=json.loads(freeze_bytes)
receipt=json.loads((root/'receipt.json').read_text())
assert receipt['returncode']==0 and receipt['sha256']==args.archive_sha256
with tarfile.open(root/'output.tar.gz') as archive, tempfile.TemporaryDirectory(prefix='r13-joint-') as directory:
    def read(name):return archive.extractfile('./'+name).read()
    assert read('manifest.json')==manifest_bytes and read('freeze.json')==freeze_bytes
    for name,digest in freeze['files'].items():assert sha(read(name))==digest
    assert int(read('exit_code.txt'))==0
    assert json.loads(read('complete.json'))==dict(manifest_sha256=sha(manifest_bytes),records=64)
    snapshot=Path(directory)/'frozen/analysis_snapshot';snapshot.mkdir(parents=True)
    for name in manifest['analysis_sha256']:
        (snapshot/name).write_bytes(read('analysis_snapshot/'+name))
    (snapshot/'analyze.py').write_bytes(read('analyze.py'))
    sys.path.insert(0,str(snapshot))
    from decode_r13_wait_sum import decode
    from analyze import block_contrasts,summarize
    records=json.loads(read('records.json'))
    assert len(records)==64 and len({r['raw'] for r in records})==64
    assert len({r['sha256'] for r in records})==64
    decoded=[]
    for record,sample in zip(records,manifest['samples'],strict=True):
        assert all(record[key]==value for key,value in sample.items()) and record['returncode']==0
        raw=read(record['raw']);assert sha(raw)==record['sha256']
        result=decode([json.loads(line) for line in gzip.decompress(raw).decode().splitlines()],manifest['gpu_uuid'])
        assert result['stages']==sample['stage'] and result['observer']==sample['flag']
        assert result==json.loads(read('raw/'+sample['id']+'.decoded.json'))
        decoded.append(result)
    assert len({r['sm'] for r in decoded})==1
    result=summarize([block_contrasts(decoded[i:i+4]) for i in range(0,64,4)])
    assert result==json.loads(read('analysis.json'))
    # Retain all original progress endpoints and event data as descriptive checks.
    result.update(records=[dict(id=s['id'],block=s['block'],position=s['position'],**r)
                  for s,r in zip(manifest['samples'],decoded,strict=True)],
        archive_sha256=args.archive_sha256,manifest_sha256=sha(manifest_bytes),freeze_sha256=sha(freeze_bytes),
        frozen_inputs_verified=len(freeze['files']),independently_checked_values=64*4096,
        verifier_sha256=sha(Path(__file__).read_bytes()),full_D_verified=False)
with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(records=len(result['records']),values=result['independently_checked_values'],
                     summary=result['summary'],observation_qualified=False),indent=2))

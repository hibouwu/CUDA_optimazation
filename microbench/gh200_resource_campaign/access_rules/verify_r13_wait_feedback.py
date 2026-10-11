"""Verify a caller-pinned feedback acquisition, with the original frozen decoder."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile


def verify(root,archive_sha,manifest_sha,freeze_sha):
    sha=lambda payload:hashlib.sha256(payload).hexdigest()
    assert sha((root/'output.tar.gz').read_bytes())==archive_sha
    manifest_bytes=(root/'manifest.json').read_bytes();freeze_bytes=(root/'freeze.json').read_bytes()
    assert sha(manifest_bytes)==manifest_sha and sha(freeze_bytes)==freeze_sha
    manifest=json.loads(manifest_bytes);freeze=json.loads(freeze_bytes)
    receipt=json.loads((root/'receipt.json').read_text())
    assert receipt['returncode']==0 and receipt['sha256']==archive_sha
    decoded=[]
    with tarfile.open(root/'output.tar.gz') as archive,tempfile.TemporaryDirectory(prefix='feedback-verify-') as directory:
        def read(name):return archive.extractfile('./'+name).read()
        assert read('manifest.json')==manifest_bytes and read('freeze.json')==freeze_bytes
        for name,digest in freeze['files'].items():assert sha(read(name))==digest
        for name,mode in freeze.get('executables',{}).items():
            assert archive.getmember('./'+name).mode & 0o777 == mode
        for name,digest in manifest['binary_sha256'].items():assert sha(read('build/'+name))==digest
        snapshot=Path(directory)/'frozen/analysis_snapshot';snapshot.mkdir(parents=True)
        for name,digest in manifest['analysis_sha256'].items():
            payload=read('analysis_snapshot/'+name);assert sha(payload)==digest
            (snapshot/name).write_bytes(payload)
        sys.path.insert(0,str(snapshot))
        from decode_r13_wait_feedback import decode
        assert int(read('exit_code.txt'))==0
        count=len(manifest['samples'])
        assert json.loads(read('complete.json'))==dict(manifest_sha256=manifest_sha,records=count)
        records=json.loads(read('records.json'))
        assert len(records)==count and len({r['id'] for r in records})==count
        assert len({r['raw'] for r in records})==count and len({r['sha256'] for r in records})==count
        for record,sample in zip(records,manifest['samples'],strict=True):
            assert all(record[key]==value for key,value in sample.items()) and record['returncode']==0
            raw=read(record['raw']);assert sha(raw)==record['sha256']
            result=decode([json.loads(line) for line in gzip.decompress(raw).decode().splitlines()],manifest['gpu_uuid'])
            assert result['stages']==sample['stage'] and result['observer']==sample['flag']
            assert result==json.loads(read('raw/'+sample['id']+'.decoded.json'))
            decoded.append(dict(id=sample['id'],raw_sha256=record['sha256'],
                                **{k:sample[k] for k in ('block','position') if k in sample},**result))
    return dict(archive_sha256=archive_sha,manifest_sha256=manifest_sha,freeze_sha256=freeze_sha,
        source_manifest_sha256=manifest['source_manifest_sha256'],binary_sha256=manifest['binary_sha256'],
        decoder_sha256=manifest['analysis_sha256']['decode_r13_wait_feedback.py'],
        records=decoded,total_original_targets=len(decoded),retaken_targets=0,
        independently_checked_values=len(decoded)*4096,frozen_inputs_verified=len(freeze['files']),
        mode_format_and_sampled_correctness_passed=True,full_D_verified=False,
        observation_qualified=False,prediction_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--archive-sha256',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.run,args.archive_sha256,args.manifest_sha256,args.freeze_sha256)
    result['verifier_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('total_original_targets','independently_checked_values',
         'frozen_inputs_verified','mode_format_and_sampled_correctness_passed','observation_qualified')},indent=2))

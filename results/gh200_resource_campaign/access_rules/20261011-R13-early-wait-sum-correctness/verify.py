"""Verify the four original correctness targets across two preserved collection attempts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    attempts=[(root,'313d844177db6d8757eb5315e122a3c1d6b0c4dfccdce247daf8c829c3053e84',1),
              (root/'attempts/remaining3','8223b627870031e6fef163c4e797d65bc348e4e107f79b946a70b690b30f3230',0)]
    final_manifest=json.loads((attempts[1][0]/'manifest.json').read_text())
    collected=[];identity=[]
    with tempfile.TemporaryDirectory(prefix='r13-correctness-decode-') as directory:
        snapshot=Path(directory)/'frozen/analysis_snapshot'
        snapshot.mkdir(parents=True)
        # Use the explicitly frozen corrected decoder; never the moving checkout.
        with tarfile.open(attempts[1][0]/'output.tar.gz') as archive:
            for name,digest in final_manifest['analysis_sha256'].items():
                payload=archive.extractfile('./analysis_snapshot/'+name).read()
                assert sha(payload)==digest
                (snapshot/name).write_bytes(payload)
        sys.path.insert(0,str(snapshot))
        from decode_r13_wait_sum import decode
        for run,expected_archive,expected_returncode in attempts:
            archive_path=run/'output.tar.gz'
            assert sha(archive_path.read_bytes())==expected_archive
            receipt=json.loads((run/'receipt.json').read_text())
            assert receipt['sha256']==expected_archive and receipt['returncode']==expected_returncode
            manifest=json.loads((run/'manifest.json').read_text())
            freeze=json.loads((run/'freeze.json').read_text())
            with tarfile.open(archive_path) as archive:
                for name,digest in freeze['files'].items():
                    payload=archive.extractfile('./'+name).read()
                    assert sha(payload)==digest
                assert archive.extractfile('./manifest.json').read()==(run/'manifest.json').read_bytes()
                assert archive.extractfile('./freeze.json').read()==(run/'freeze.json').read_bytes()
                assert int(archive.extractfile('./exit_code.txt').read())==expected_returncode
                records=json.load(archive.extractfile('./records.json'))
                wanted=manifest['samples'][:1] if expected_returncode else manifest['samples']
                assert len(records)==len(wanted)
                if expected_returncode:
                    assert './complete.json' not in archive.getnames()
                else:
                    complete=json.load(archive.extractfile('./complete.json'))
                    assert complete==dict(manifest_sha256=sha((run/'manifest.json').read_bytes()),records=3)
                for record,sample in zip(records,wanted,strict=True):
                    assert all(record[key]==value for key,value in sample.items())
                    assert record['returncode']==0
                    raw=archive.extractfile('./'+record['raw']).read()
                    assert sha(raw)==record['sha256']
                    result=decode([json.loads(line) for line in gzip.decompress(raw).decode().splitlines()],manifest['gpu_uuid'])
                    assert result['stages']==sample['stage'] and result['observer']==sample['flag']
                    if not expected_returncode:
                        assert result==json.load(archive.extractfile('./raw/'+record['id']+'.decoded.json'))
                    collected.append(dict(id=record['id'],raw_sha256=record['sha256'],
                        original_attempt_returncode=expected_returncode,**result))
                identity.append(dict(archive_sha256=expected_archive,manifest_sha256=sha((run/'manifest.json').read_bytes()),
                    original_returncode=expected_returncode,records=len(records),frozen_files_verified=len(freeze['files'])))
    assert [r['id'] for r in collected]==['stage4_flag0','stage4_flag1','stage6_flag0','stage6_flag1']
    assert len({r['raw_sha256'] for r in collected})==4
    assert final_manifest['prior_attempt']['raw_sha256']==collected[0]['raw_sha256']
    assert final_manifest['prior_attempt']['output_archive_sha256']==identity[0]['archive_sha256']
    assert final_manifest['prior_attempt']['manifest_sha256']==identity[0]['manifest_sha256']
    report=dict(attempts=identity,records=collected,
        corrected_decoder_sha256=final_manifest['analysis_sha256']['decode_r13_wait_sum.py'],
        total_original_targets=4,retaken_targets=0,independently_checked_values=4*4096,
        original_failed_attempt_preserved=True,corrected_format_and_sampled_correctness_passed=True,
        full_D_verified=False,observation_qualified=False,prediction_qualified=False,
        limits=['One independent process per stage/flag, two allocations; no causal pairing or performance estimate.',
                'Q describes the instrumented waiting path; outside includes observer arithmetic and its scheduling effects.',
                'No physical-ns conversion, bare service parameter or critical-CTA evidence.'],
        verifier_sha256=sha(Path(__file__).read_bytes()))
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('total_original_targets','retaken_targets','independently_checked_values',
          'original_failed_attempt_preserved','corrected_format_and_sampled_correctness_passed','observation_qualified')},indent=2))


if __name__=='__main__':main()

"""Replay two real-record misbindings and a post-extraction label mutation in copies."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new report required')
    run = args.run.resolve()
    source = Path(__file__).resolve().parent
    protocol = source / 'joint-replay-protocol.json'
    archive_sha = '4e372a3dbe50e33657fd57a0c265faba2b7ba857eb2ea34e78803b60435d0b04'
    results = []
    for mode in ('swap_paths_and_hashes', 'swap_bytes_and_hashes'):
        with tempfile.TemporaryDirectory(prefix='joint-binding-') as temporary:
            copied = Path(temporary) / 'run'
            copied.mkdir()
            for name in ('manifest.json', 'output.tar.gz', 'receipt.json'):
                shutil.copyfile(run / name, copied / name)
            for name in ('returned', 'analysis_snapshot'):
                shutil.copytree(run / name, copied / name)
            records_path = copied / 'returned/records.json'
            records = json.loads(records_path.read_text())
            by_id = {record['id']: record for record in records}
            first = by_id['off_on_pair00_arm1']
            second = by_id['off_on_pair01_arm1']
            if mode == 'swap_paths_and_hashes':
                for field in ('raw', 'sha256'):
                    first[field], second[field] = second[field], first[field]
            else:
                a = copied / 'returned' / first['raw']
                b = copied / 'returned' / second['raw']
                original_a, original_b = a.read_bytes(), b.read_bytes()
                a.write_bytes(original_b)
                b.write_bytes(original_a)
                first['sha256'] = hashlib.sha256(original_b).hexdigest()
                second['sha256'] = hashlib.sha256(original_a).hexdigest()
            records_path.write_text(json.dumps(records, indent=2) + '\n')
            output = copied / 'extracted'
            command = [sys.executable, str(source / 'extract_joint.py'), '--run', str(copied),
                       '--protocol', str(protocol), '--expected-protocol-sha256',
                       hashlib.sha256(protocol.read_bytes()).hexdigest(),
                       '--expected-output-sha256', archive_sha, '--output', str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            reason = 'local file differs from pinned acquisition member: records.json'
            assert result.returncode != 0 and reason in result.stderr and not output.exists()
            results.append(dict(case=mode, returncode=result.returncode,
                                rejection=reason, output_created=False))

    # A correct marginal diagnostic cannot hide altered joint block labels.
    with tempfile.TemporaryDirectory(prefix='joint-label-') as temporary:
        copied = Path(temporary) / 'joint'
        shutil.copytree(source / 'joint-replay-v4', copied)
        path = copied / 'joint-process-records.json.gz'
        records = json.loads(gzip.decompress(path.read_bytes()))
        records[0]['block'] = 999
        path.write_bytes(gzip.compress(json.dumps(records, separators=(',', ':')).encode(), mtime=0))
        receipt_path = copied / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['output_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        result = subprocess.run([sys.executable, str(source / 'verify_joint.py'),
                                 '--run', str(run), '--expected-output-sha256', archive_sha,
                                 '--joint', str(copied), '--old-replay',
                                 str(source / 'data/old-control-replay.json.gz'),
                                 '--output', str(copied / 'verification.json')],
                                capture_output=True, text=True)
        assert result.returncode != 0 and "record['block']==original['pair']" in result.stderr
        results.append(dict(case='joint_block_changed_with_unchanged_marginal_diagnostic',
                            returncode=result.returncode, rejection='archived block identity mismatch',
                            output_created=(copied / 'verification.json').exists()))
    with args.output.open('x') as stream:
        json.dump(dict(acquisition_archive_sha256=archive_sha,
                       source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                      for name in ('acquisition_identity.py', 'extract_joint.py',
                                                   'joint_records.py', 'verify_joint.py',
                                                   'audit_acquisition_binding.py')},
                       cases=results, qualification=False), stream, indent=2)
        stream.write('\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

"""Finish a scratch-controller job whose login connection dropped after the node had already exported.

The node script streams the run to RETURN on the login node, verifies it there, then prints the receipt line
on srun's stdout. If the local ssh dies after the export, the archive and its source manifest stay in RETURN
and the receipt line is lost. This redoes the controller's remaining steps from those two files: download,
check member identities against the node-side source manifest (the same check as RECEIVE_CODE), read the
return code and peak bytes the node wrote into the archive, write the receipts, append progress, and remove
the login copies.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time

import a044_scratch_controller as controller


def members(archive):
    actual = {}
    with gzip.open(archive, 'rb') as compressed, tarfile.open(fileobj=compressed, mode='r|') as tar:
        for m in tar:
            name = PurePosixPath(m.name).as_posix()
            assert name not in actual, 'duplicate tar member: ' + name
            row = {'mode': m.mode}
            if m.isfile():
                row.update(type='file', size=m.size, sha256=hashlib.sha256(tar.extractfile(m).read()).hexdigest())
            elif m.isdir():
                row.update(type='directory')
            elif m.issym():
                row.update(type='symlink', target=m.linkname)
            else:
                raise ValueError('unexpected tar member: ' + name)
            actual[name] = row
    return actual


def member_text(archive, pattern):
    with tarfile.open(archive, 'r:gz') as tar:
        names = [m.name for m in tar if PurePosixPath(m.name).match(pattern)]
        assert len(names) == 1, (pattern, names)
        return tar.extractfile(names[0]).read().decode(), names[0]


def recover(jobs_path, name, job_id):
    jobs = controller.read(jobs_path)
    job = next(row for row in jobs if row['name'] == name)
    progress_path = jobs_path.with_suffix('.progress.json')
    progress = controller.read(progress_path)
    assert name not in {row['name'] for row in progress}
    folder = Path(job['output'])
    found = controller.login(f'ls {controller.RETURN}/{name}-*.tar.gz', capture_output=True, text=True,
                             check=True).stdout.split()
    assert len(found) == 1, found
    temporary = found[0]
    started = int(temporary.rsplit('-', 1)[1].removesuffix('.tar.gz'))
    listing = controller.login(shlex.join(['sha256sum', temporary, temporary + '.sources.json']),
                               capture_output=True, text=True, check=True).stdout.split()
    remote_sha, manifest_sha = listing[0], listing[2]
    storage = controller.DATA / folder.name
    local_archive = storage / 'output.tar.gz'
    subprocess.run(['scp', '-o', 'BatchMode=yes', 'romeo:' + temporary, str(local_archive)], check=True)
    subprocess.run(['scp', '-o', 'BatchMode=yes', 'romeo:' + temporary + '.sources.json',
                    str(folder / 'source-manifest.json')], check=True)
    assert controller.sha(local_archive) == remote_sha, 'login to local transfer hash'
    assert controller.sha(folder / 'source-manifest.json') == manifest_sha
    expected = controller.read(folder / 'source-manifest.json')['entries']
    actual = members(local_archive)
    assert actual == expected, 'source and archive member identities differ'
    returncode = int(member_text(local_archive, f'exit-code-{job_id}.txt')[0])
    space = json.loads(member_text(local_archive, f'space-{job_id}.json')[0])
    (folder / 'output.tar.gz').symlink_to(local_archive)
    remote = dict(login_archive=temporary, bytes=local_archive.stat().st_size, sha256=remote_sha,
                  source_manifest_sha256=manifest_sha, verified_members=len(actual),
                  verified_files=sum(r['type'] == 'file' for r in actual.values()),
                  run_dir=None, archive=None, returncode=returncode, peak_run_bytes=space['peak_run_bytes'],
                  recovered=dict(job_id=job_id, reason='login ssh dropped after node export; receipt line lost',
                                 verified='locally against node source manifest'))
    controller.write(folder / 'remote-receipt.json', remote)
    receipt = dict(returncode=returncode, started_unix_ns=started, stopped_unix_ns=time.time_ns(),
                   input_sha256=controller.sha(Path(job['input'])), sha256=remote_sha,
                   bytes=local_archive.stat().st_size, peak_run_bytes=space['peak_run_bytes'],
                   budget_bytes=space['budget_bytes'], recovered=True)
    assert receipt['input_sha256'] == job['input_sha256']
    controller.write(folder / 'receipt.json', receipt)
    controller.login(shlex.join(['rm', '-f', '--', temporary, temporary + '.sources.json']), check=True)
    controller.write(folder / 'temporary-cleanup.json', dict(local_archive_sha256=remote_sha,
                     node_directory=None, login_archive=temporary, quota_after=controller.scratch_quota()))
    progress.append(dict(name=name, folder=str(folder), receipt=receipt))
    controller.write(progress_path, progress)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs', type=Path)
    parser.add_argument('name')
    parser.add_argument('job_id')
    args = parser.parse_args()
    recover(args.jobs, args.name, args.job_id)

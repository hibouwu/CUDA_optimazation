"""Collect one ready private node namespace through its existing Slurm allocation."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shlex
import subprocess
import sys

if sys.flags.optimize != 0:
    raise RuntimeError("S20 requires active assertions; Python optimization forbidden")
import tarfile
import time

SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-i', '/home/jianyeshi/.ssh/id_ed25519', 'hibouwu@romeo1.univ-reims.fr']


def ssh_json(code):
    result = subprocess.run(SSH + ['PYTHONOPTIMIZE=0 python3 -B -c ' + shlex.quote(code)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def collect(deployment, job, output):
    if sys.flags.optimize != 0:
        raise RuntimeError('collector requires active assertions; Python optimization forbidden')
    assert job.isascii() and job.isdecimal()
    assert deployment.startswith('/gpfs/projet/r260073/hibouwu/gh200_resource_campaign/')
    output = Path(output)
    output.mkdir(exist_ok=False)
    deadline = time.monotonic() + 3 * 3600
    ready = None
    while time.monotonic() < deadline:
        code = "from pathlib import Path;import json,subprocess;p=Path(%r)/'transfer-ready.json';r=subprocess.run(['squeue','-j',%r,'-h','-o','%%i|%%T'],capture_output=True,text=True,timeout=15);print(json.dumps({'ready':json.loads(p.read_text()) if p.exists() else None,'queue':r.stdout,'returncode':r.returncode}))" % (deployment, job)
        observation = ssh_json(code)
        (output / 'latest-observation.json').write_text(json.dumps(observation, indent=2) + '\n')
        if observation['ready']:
            ready = observation['ready']
            assert ready['job'] == job and observation['returncode'] == 0 and observation['queue'].startswith(job + '|RUNNING')
            break
        if not observation['queue']:
            raise RuntimeError('original job not live; inspect original sacct and node pointer, never resubmit')
        time.sleep(10)
    assert ready is not None, 'ready marker timeout; preserve original job/namespace'
    (output / 'ready.json').write_text(json.dumps(ready, indent=2) + '\n')
    node_code = '''from pathlib import Path
import hashlib,json,os,sys,tarfile
if sys.flags.optimize!=0:raise RuntimeError('node verifier requires active assertions')
root=Path(%r);ready=%r
assert root.parent==Path('/tmp') and root.name.startswith('codex-s20-node-'+ready['job']+'-')
assert root.is_dir() and not root.is_symlink() and root.stat().st_uid==os.geteuid()==ready['uid']
assert os.uname().nodename==ready['host'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==ready['boot_id']
manifest=root/'transfer-manifest.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==ready['transfer_manifest_sha256']
members=json.loads(manifest.read_text())
assert len(members)==ready['file_count']
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='transfer-manifest.json'}
assert actual==set(members) and not any(p.is_symlink() for p in root.rglob('*'))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 t.add(manifest,arcname='transfer-manifest.json',recursive=False)
 for n,x in members.items():
  p=root/n;assert p.is_file() and p.stat().st_size==x['bytes'] and p.stat().st_mode&0o777==x['mode']
  assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
  t.add(p,arcname=n,recursive=False)
''' % (ready['node_root'], ready)
    argv = ['srun', '--jobid=' + job, '--overlap', '--exact', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=1', '--gres=none', 'python3', '-B', '-c', node_code]
    archive = output / 'actual.tar'
    with archive.open('xb') as stream:
        result = subprocess.run(SSH + [shlex.join(argv)], stdout=stream, stderr=subprocess.PIPE, timeout=550)
    (output / 'transfer-call.json').write_text(json.dumps({'argv': argv, 'returncode': result.returncode,
                        'stderr': result.stderr.decode()}, indent=2) + '\n')
    assert result.returncode == 0, 'transfer failed; original node files retained; no acknowledgment'
    actual = output / 'actual'
    actual.mkdir()
    with tarfile.open(archive) as packet:
        body = packet.extractfile('transfer-manifest.json').read()
        assert hashlib.sha256(body).hexdigest() == ready['transfer_manifest_sha256']
        members = json.loads(body)
        assert len(members) == ready['file_count']
        seen = set()
        for member in packet:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            assert member.name not in seen
            data = packet.extractfile(member).read()
            if member.name == 'transfer-manifest.json':
                assert data == body
            else:
                expected = members[member.name]
                assert len(data) == member.size == expected['bytes'] and member.mode == expected['mode']
                assert hashlib.sha256(data).hexdigest() == expected['sha256']
            target = actual / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
            target.chmod(member.mode)
            seen.add(member.name)
        assert seen == set(members) | {'transfer-manifest.json'}
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    receipt = {k: ready[k] for k in ('job', 'node_root', 'host', 'boot_id', 'transfer_manifest_sha256')}
    receipt.update(file_count=len(members), all_members_verified=True, durable_offhost_custody_proven=True,
                   outer_tar_sha256=digest.hexdigest(), outer_tar_bytes=archive.stat().st_size,
                   sampling_complete=ready['sampling_complete'])
    # Flush the offhost files and directories before acknowledging custody.
    import os
    for path in actual.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    for path in sorted([actual, *[p for p in actual.rglob('*') if p.is_dir()]], key=lambda p: len(p.parts), reverse=True):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    with archive.open('rb') as stream:
        os.fsync(stream.fileno())
    (output / 'collection-index.json').write_text(json.dumps({**receipt, 'members': members}, indent=2) + '\n')
    with (output / 'collection-index.json').open('rb') as stream:
        os.fsync(stream.fileno())
    for directory in (output, output.parent):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    code = "from pathlib import Path;import json,os;p=Path(%r)/'offhost-receipt.json';partial=p.with_suffix('.json.partial');assert not p.exists() and not partial.exists();b=json.dumps(%r,indent=2)+'\\n';f=partial.open('x');f.write(b);f.flush();os.fsync(f.fileno());f.close();os.link(partial,p);partial.unlink();fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps({'ack_written':True}))" % (deployment, receipt)
    assert ssh_json(code)['ack_written'] is True
    print(json.dumps({'collected_files': len(members), 'bytes': archive.stat().st_size, 'sampling_complete': ready['sampling_complete']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    collect(args.deployment, args.job, args.output)

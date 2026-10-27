"""Run prepared a044 packages with the working directory on ROMEO scratch, under a checked space budget.

Derived from the 20261023 C1 controller and its stream-export candidate (both kept unchanged).
Changes: node run directory and login staging live under /scratch_p/hibouwu/a044-runs; TMP/TMPDIR/TEMP
point inside the run directory; the GPFS user quota is checked before submission; the node checks
scratch, /tmp and /var/spool/slurmd before extracting; a watchdog stops the measurement when the run
directory exceeds its budget; leftover run directories block new submissions until recovered.
2026-10-27: GPU jobs took ~1 min while single-stream downloads took 5-20 min, so the next job is submitted
once the login copy matches the node hash, and downloads run in the background. A receipt line lost to a
dropped login ssh is rebuilt by verifying the login copy against the node source manifest.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import threading
import time

GPU = 'GPU-7473147b-6ca4-6e68-68c2-b1a086e1753e'
SCRATCH = '/scratch_p/hibouwu/a044-runs'
RETURN = SCRATCH + '/return'
RUN_PATTERN = r'/(scratch_p|gpfs/scratch)/hibouwu/a044-runs/run-[A-Za-z0-9]+'
DATA = Path('/data/jianyeshi/gh200-a044-20261023/scratch-controller')
MiB = 1 << 20
LOGIN_KEY = ('romeo1,romeo1.univ-reims.fr ssh-ed25519 '
             'AAAAC3NzaC1lZDI1NTE5AAAAIL9ihEEtfFd+0mLzLnuZDCfxWuEtE3hHouEy4ALJMwsv')
NODE_ROOT_MIN = 2048 * MiB   # /tmp and /var/spool/slurmd share a ~16 GiB root on a044
MARGIN = 256 * MiB

SOURCE_CODE = r'''
import hashlib,json,os,stat,sys
from pathlib import Path
root=Path.cwd(); entries={}
for parent,dirs,files in os.walk(root,followlinks=False):
 if Path(parent)==root:
  dirs[:]=[n for n in dirs if n not in ('tmp','spack-cache')]
  files=[n for n in files if n!='output.tar.gz']
 for name in ['.']+dirs+files if Path(parent)==root else dirs+files:
  path=Path(parent)/name; key=path.relative_to(root).as_posix(); info=path.lstat()
  row={'mode':stat.S_IMODE(info.st_mode)}
  if path.is_symlink():row.update(type='symlink',target=os.readlink(path))
  elif path.is_dir():row.update(type='directory')
  else:
   h=hashlib.sha256()
   with path.open('rb') as f:
    for block in iter(lambda:f.read(1048576),b''):h.update(block)
   row.update(type='file',size=info.st_size,sha256=h.hexdigest())
  entries[key]=row
Path(sys.argv[1]).write_text(json.dumps({'entries':entries},sort_keys=True)+'\n')
'''

RECEIVE_CODE = r'''
import gzip,hashlib,json,sys,tarfile
from pathlib import Path,PurePosixPath
archive=Path(sys.argv[1]); manifest=Path(sys.argv[2]); digest=hashlib.sha256()
with archive.open('xb') as out:
 for block in iter(lambda:sys.stdin.buffer.read(1048576),b''):
  out.write(block);digest.update(block)
expected=json.loads(manifest.read_text())['entries']; actual={}
with gzip.open(archive,'rb') as compressed:
 with tarfile.open(fileobj=compressed,mode='r|') as tar:
  for m in tar:
   name=PurePosixPath(m.name).as_posix()
   if name in actual:raise ValueError('duplicate tar member: '+name)
   row={'mode':m.mode}
   if m.isfile():
    h=hashlib.sha256();f=tar.extractfile(m)
    for block in iter(lambda:f.read(1048576),b''):h.update(block)
    row.update(type='file',size=m.size,sha256=h.hexdigest())
   elif m.isdir():row.update(type='directory')
   elif m.issym():row.update(type='symlink',target=m.linkname)
   else:raise ValueError('unexpected tar member: '+name)
   actual[name]=row
 for block in iter(lambda:compressed.read(1048576),b''):pass
if actual!=expected:raise ValueError('source and archive member identities differ')
print(json.dumps(dict(login_archive=str(archive),bytes=archive.stat().st_size,
 sha256=digest.hexdigest(),source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
 verified_members=len(actual),verified_files=sum(r['type']=='file' for r in actual.values()))))
'''

VERIFY_CODE = r'''
import gzip,hashlib,json,re,sys,tarfile
from pathlib import Path,PurePosixPath
archive=Path(sys.argv[1]); manifest=Path(archive.as_posix()+'.sources.json'); digest=hashlib.sha256()
with archive.open('rb') as f:
 for block in iter(lambda:f.read(1048576),b''):digest.update(block)
expected=json.loads(manifest.read_text())['entries']; actual={}; small={}
with gzip.open(archive,'rb') as compressed:
 with tarfile.open(fileobj=compressed,mode='r|') as tar:
  for m in tar:
   name=PurePosixPath(m.name).as_posix()
   if name in actual:raise ValueError('duplicate tar member: '+name)
   row={'mode':m.mode}
   if m.isfile():
    data=tar.extractfile(m).read()
    row.update(type='file',size=m.size,sha256=hashlib.sha256(data).hexdigest())
    if re.fullmatch(r'(exit-code|space)-[0-9]+\.(txt|json)',name):small[name.split('-')[0]]=data.decode()
   elif m.isdir():row.update(type='directory')
   elif m.issym():row.update(type='symlink',target=m.linkname)
   else:raise ValueError('unexpected tar member: '+name)
   actual[name]=row
if actual!=expected:raise ValueError('source and archive member identities differ')
print(json.dumps(dict(login_archive=str(archive),bytes=archive.stat().st_size,
 sha256=digest.hexdigest(),source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
 verified_members=len(actual),verified_files=sum(r['type']=='file' for r in actual.values()),
 archive=None,returncode=int(small['exit']),peak_run_bytes=json.loads(small['space'])['peak_run_bytes'])))
'''

NODE_SCRIPT = r'''set -euo pipefail
scratch=__SCRATCH__
budget=__BUDGET__
avail() { df -B1 --output=avail "$1" | tail -1 | tr -d ' '; }
ifree() { df --output=iavail "$1" | tail -1 | tr -d ' '; }
node_tmp=$(avail /tmp); node_spool=$(avail /var/spool/slurmd); scratch_avail=$(avail "$scratch"); scratch_inodes=$(ifree "$scratch")
profiling=$(grep -h RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null | tr -d ' ' || true)
leftover=$(ls -d /tmp/c1-a044-development-* 2>/dev/null | tr '\n' ' ' || true)
echo "PREFLIGHT node=$(hostname) tmp_avail=$node_tmp spool_avail=$node_spool scratch_avail=$scratch_avail scratch_inodes=$scratch_inodes profiling=$profiling tmp_leftover=[$leftover]" >&2
if [ "$node_tmp" -lt __NODE_ROOT_MIN__ ] || [ "$node_spool" -lt __NODE_ROOT_MIN__ ]; then echo "PREFLIGHT_FAIL node root below minimum" >&2; exit 90; fi
if [ "$scratch_avail" -lt $((2 * budget)) ] || [ "$scratch_inodes" -lt 50000 ]; then echo "PREFLIGHT_FAIL scratch space" >&2; exit 91; fi
run_dir=$(mktemp -d "$scratch/run-XXXXXXXX")
echo "JOB=$SLURM_JOB_ID NODE=$(hostname) RUN=$run_dir" >&2
cd "$run_dir"
tar -xzf -
mkdir -p tmp
export TMPDIR="$run_dir/tmp" TMP="$run_dir/tmp" TEMP="$run_dir/tmp" SPACK_USER_CACHE_PATH="$run_dir/spack-cache"
nvidia-smi --query-gpu=uuid,clocks.sm,clocks.max.sm,clocks.applications.graphics --format=csv > gpu-before.csv 2>&1 || true
set +e
set -m
(
 set -euo pipefail
 romeo_load_armgpu_env
 spack load cuda@12.9.0
 spack load /qwvf7fxkl3c54zlcl4mh57dzutilo3zx
 export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
 export CUDA_VISIBLE_DEVICES=__GPU__
 export V08_GPU="$CUDA_VISIBLE_DEVICES"
 __MEASUREMENT_COMMAND__
) >"run-${SLURM_JOB_ID}.log" 2>&1 &
measurement=$!
set +m
peak=0; exceeded=0
while kill -0 "$measurement" 2>/dev/null; do
 used=$(du -sb "$run_dir" 2>/dev/null | cut -f1)
 [ "${used:-0}" -gt "$peak" ] && peak=$used
 if [ "${used:-0}" -gt "$budget" ]; then exceeded=1; kill -TERM -- "-$measurement" 2>/dev/null; sleep 5; kill -KILL -- "-$measurement" 2>/dev/null; break; fi
 sleep 2
done
wait "$measurement"; rc=$?
[ "$exceeded" = 1 ] && rc=92
used=$(du -sb "$run_dir" | cut -f1); [ "$used" -gt "$peak" ] && peak=$used
printf '%s\n' "$rc" > "exit-code-${SLURM_JOB_ID}.txt"
printf '{"budget_bytes": %s, "peak_run_bytes": %s, "exceeded": %s}\n' "$budget" "$peak" "$exceeded" > "space-${SLURM_JOB_ID}.json"
nvidia-smi --query-gpu=uuid,clocks.sm,clocks.max.sm,clocks.applications.graphics --format=csv > gpu-after.csv 2>&1 || true
set -e
export C1_RETURN_CODE="$rc"
printf '%s\n' __LOGIN_KEY__ > tmp/login_known_hosts
login_archive=__LOGIN_ARCHIVE__
ssh_args=(-T -o BatchMode=yes -o StrictHostKeyChecking=yes -o GlobalKnownHostsFile=/dev/null -o UserKnownHostsFile="$run_dir/tmp/login_known_hosts" hibouwu@romeo1)
receive_command=__RECEIVE__
__SOURCE__
ssh "${ssh_args[@]}" "umask 077; set -C; cat > '$login_archive.sources.json'" < tmp/export-source-manifest.json
tar --hard-dereference --exclude=./tmp --exclude=./spack-cache --exclude=./output.tar.gz -czf - . | ssh "${ssh_args[@]}" "$receive_command '$login_archive' '$login_archive.sources.json'" > tmp/export-verified.json
python3 -c 'import hashlib,json,sys;from pathlib import Path;r=json.loads(Path("tmp/export-verified.json").read_text());assert r["source_manifest_sha256"]==hashlib.sha256(Path("tmp/export-source-manifest.json").read_bytes()).hexdigest();r.update(run_dir=str(Path.cwd()),archive=None,returncode=int(sys.argv[1]),peak_run_bytes=int(sys.argv[2]));print(json.dumps(r))' "$rc" "$peak"
cd "$scratch"
rm -rf -- "$run_dir"
exit "$rc"
'''


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def login(command, **kwargs):
    return subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', 'romeo', command], **kwargs)


def scratch_quota():
    """GPFS user quota of the scratch fileset in bytes; writes stop at the soft quota once grace expired."""
    text = login('mmlsquota -u hibouwu --block-size 1K gpfs', capture_output=True, text=True, check=True).stdout
    for line in text.splitlines():
        blocks, _, inodes = line.partition('|')
        fields, file_fields = blocks.split(), inodes.split()
        if len(fields) >= 8 and fields[:3] == ['gpfs', 'scratch', 'USR']:
            used, soft, hard, doubt = (int(value) * 1024 for value in fields[3:7])
            files, file_soft = int(file_fields[0]), int(file_fields[1])
            ceiling = soft if soft else hard
            return dict(used=used, soft=soft, hard=hard, in_doubt=doubt, grace=' '.join(fields[7:]),
                        headroom=ceiling - used - doubt, files=files, file_headroom=file_soft - files)
    raise RuntimeError('scratch quota row missing:\n' + text)


def preflight(budget, pending=()):
    """Refuse to submit when quota cannot hold the run directory plus its streamed archive, or old runs remain.
    Login copies still being downloaded (pending) are not leftovers."""
    quota = scratch_quota()
    login(shlex.join(['mkdir', '-p', RETURN]), check=True)
    expected = {name for path in pending for name in (path, path + '.sources.json')}
    leftovers = [path for path in login(f'ls -d {SCRATCH}/run-* {RETURN}/* 2>/dev/null || true',
                                        capture_output=True, text=True, check=True).stdout.split()
                 if path not in expected]
    state = dict(checked_unix_ns=time.time_ns(), budget_bytes=budget, quota=quota, leftovers=leftovers)
    if leftovers:
        raise RuntimeError('Recover leftover scratch runs before submitting: ' + ' '.join(leftovers))
    if quota['headroom'] < 2 * budget + MARGIN or quota['file_headroom'] < 50000:
        raise RuntimeError('Scratch quota headroom too small: ' + json.dumps(quota))
    return state


def recover_receipt(folder, temporary):
    """The local ssh can drop after the node has exported; verify the login copy there and rebuild the receipt line."""
    run = re.search(r'RUN=(\S+)', (folder / 'slurm.log').read_text())
    exists = login(shlex.join(['test', '-f', temporary + '.sources.json']))
    if not run or exists.returncode:
        raise RuntimeError('No archive receipt; inspect Slurm log: ' + str(folder))
    line = login(shlex.join(['python3', '-c', VERIFY_CODE, temporary]), capture_output=True, text=True,
                 check=True).stdout
    remote = dict(json.loads(line), run_dir=run.group(1), recovered='receipt rebuilt on login node')
    write(folder / 'recovered-receipt.json', remote)
    return remote


def submit(block, archive, folder, remote_command, budget=512 * MiB, time_limit='00:30:00', pending=()):
    """Run one job and check the node-to-login transfer; the download is left to fetch()."""
    state = preflight(budget, pending)
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'preflight.json', state)
    started = time.time_ns()
    temporary = f'{RETURN}/{block}-{started}.tar.gz'
    (DATA / folder.name).mkdir(parents=True, exist_ok=False)
    source = shlex.join(['python3', '-c', SOURCE_CODE, 'tmp/export-source-manifest.json'])
    replacements = {'__SCRATCH__': shlex.quote(SCRATCH), '__BUDGET__': str(budget),
                    '__NODE_ROOT_MIN__': str(NODE_ROOT_MIN), '__GPU__': GPU,
                    '__LOGIN_KEY__': shlex.quote(LOGIN_KEY), '__LOGIN_ARCHIVE__': shlex.quote(temporary),
                    '__RECEIVE__': shlex.quote(shlex.join(['python3', '-c', RECEIVE_CODE])),
                    '__SOURCE__': source, '__MEASUREMENT_COMMAND__': remote_command}
    script = NODE_SCRIPT
    for key, value in replacements.items():
        script = script.replace(key, value)
    command = ['srun', '--reservation=compteurgpu_R302026', '--account=r260073',
        '--constraint=armgpu', '--partition=instant', '--nodes=1', '--ntasks=1',
        '--cpus-per-task=8', '--mem=32G', '--gpus-per-node=4', '--exclusive',
        '--nodelist=romeo-a044', '--time=' + time_limit, '--job-name=a044-' + block,
        'bash', '-lc', script]
    write(folder / 'command.json', command)
    with archive.open('rb') as stdin, (folder / 'remote-receipt.txt').open('w') as output, \
            (folder / 'slurm.log').open('w') as error:
        result = login(shlex.join(command), stdin=stdin, stdout=output, stderr=error)
    records = [json.loads(line) for line in (folder / 'remote-receipt.txt').read_text().splitlines()
               if line.startswith('{')]
    if not records:
        records = [recover_receipt(folder, temporary)]
    remote = dict(records[-1], srun_returncode=result.returncode)
    assert re.fullmatch(RUN_PATTERN, remote['run_dir']), remote['run_dir']
    assert remote['archive'] is None and remote['login_archive'] == temporary
    digest = login(shlex.join(['sha256sum', temporary]), capture_output=True, text=True,
                   check=True).stdout.split()[0]
    assert digest == remote['sha256'], 'node to login transfer hash'
    write(folder / 'remote-receipt.json', dict(remote, started_unix_ns=started, input_sha256=sha(archive),
                                              budget_bytes=budget))
    return temporary


def fetch(folder):
    """Download a login copy checked by submit(), verify it locally, then remove the login copy."""
    remote = read(folder / 'remote-receipt.json')
    temporary = remote['login_archive']
    local_archive = DATA / folder.name / 'output.tar.gz'
    for attempt in range(20):
        copied = subprocess.run(['scp', '-o', 'BatchMode=yes', 'romeo:' + temporary, str(local_archive)])
        if copied.returncode == 0 and sha(local_archive) == remote['sha256']:
            break
        time.sleep(30)
    else:
        raise RuntimeError('login to local transfer failed: ' + str(folder))
    if not (folder / 'output.tar.gz').is_symlink():
        (folder / 'output.tar.gz').symlink_to(local_archive)
    subprocess.run(['scp', '-o', 'BatchMode=yes', 'romeo:' + temporary + '.sources.json',
                    str(folder / 'source-manifest.json')], check=True)
    assert sha(folder / 'source-manifest.json') == remote['source_manifest_sha256']
    with tarfile.open(folder / 'output.tar.gz', 'r:gz') as returned:
        returned.getmembers()
    receipt = dict(returncode=remote['returncode'], started_unix_ns=remote['started_unix_ns'],
                   stopped_unix_ns=time.time_ns(), input_sha256=remote['input_sha256'], sha256=remote['sha256'],
                   bytes=local_archive.stat().st_size, peak_run_bytes=remote['peak_run_bytes'],
                   budget_bytes=remote['budget_bytes'])
    write(folder / 'receipt.json', receipt)
    login(shlex.join(['rm', '-f', '--', temporary, temporary + '.sources.json']), check=True)
    write(folder / 'temporary-cleanup.json', dict(local_archive_sha256=receipt['sha256'],
          node_directory=remote['run_dir'], login_archive=temporary, quota_after=scratch_quota()))
    return receipt


def with_continuation(archive, previous, target):
    """Package members unchanged plus the previous run's measure/continuation.json as prior-continuation.json."""
    with tarfile.open(previous / 'output.tar.gz', 'r:gz') as returned:
        continuation = returned.extractfile('./measure/continuation.json').read()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as source, tarfile.open(target, 'x:gz') as combined:
        for member in source:
            combined.addfile(member, source.extractfile(member) if member.isfile() else None)
        info = tarfile.TarInfo('prior-continuation.json')
        info.size, info.mode, info.mtime = len(continuation), 0o644, 0
        combined.addfile(info, io.BytesIO(continuation))
    return hashlib.sha256(continuation).hexdigest()


def downloading(jobs_paths):
    """Login copies that another controller checked (remote-receipt.json) and has not finished downloading."""
    return [read(Path(job['output']) / 'remote-receipt.json')['login_archive']
            for path in jobs_paths if Path(path).exists() for job in read(path)
            if (Path(job['output']) / 'remote-receipt.json').exists()
            and not (Path(job['output']) / 'receipt.json').exists()]


def run_jobs(jobs_path, downloads=6, alongside=(), keep_going=False):
    """One GPU job at a time; each archive is checked on the login node before the next job is submitted
    and downloaded in the background. A failed job stops submission after its archive is downloaded."""
    jobs = read(jobs_path)
    progress_path = jobs_path.with_suffix('.progress.json')
    progress = read(progress_path) if progress_path.exists() else []
    done = {row['name'] for row in progress}
    lock = threading.Lock()
    pending = {}   # job name -> (login copy, download future)

    def finish(job, folder, extra):
        receipt = fetch(folder)
        with lock:
            progress.append(dict(name=job['name'], folder=str(folder), receipt=receipt, **extra))
            write(progress_path, progress)
            print('complete', job['name'], json.dumps(receipt), flush=True)
        return receipt

    with ThreadPoolExecutor(downloads) as pool:
        for job in jobs:
            folder = Path(job['output'])
            if job['name'] in done:
                continue
            if (folder / 'receipt.json').exists():   # downloaded and verified, progress row not yet written
                progress.append(dict(name=job['name'], folder=str(folder), receipt=read(folder / 'receipt.json')))
                write(progress_path, progress)
                print('complete', job['name'], json.dumps(progress[-1]['receipt']), flush=True)
                continue
            if (folder / 'remote-receipt.json').exists():   # checked on login before an interrupted download
                temporary = read(folder / 'remote-receipt.json')['login_archive']
                pending[job['name']] = temporary, pool.submit(finish, job, folder, {})
                continue
            archive = Path(job['input'])
            if sha(archive) != job['input_sha256']:
                raise RuntimeError('Input archive changed: ' + str(archive))
            extra = {}
            if job.get('continuation_from'):
                if job['continuation_from'] in pending:
                    pending.pop(job['continuation_from'])[1].result()
                previous = Path(next(row['folder'] for row in progress if row['name'] == job['continuation_from']))
                combined = DATA / 'inputs' / (job['name'] + '.tar.gz')
                extra = dict(package_sha256=job['input_sha256'], continuation_from=job['continuation_from'],
                             continuation_sha256=with_continuation(archive, previous, combined))
                archive = combined
            for name in [name for name, (_, future) in pending.items() if future.done()]:
                pending.pop(name)[1].result()
            print('submitting', job['name'], flush=True)
            temporary = submit(job['name'], archive, folder, job['command'], job.get('budget_bytes', 512 * MiB),
                               job.get('time', '00:30:00'), [t for t, _ in pending.values()] + downloading(alongside))
            pending[job['name']] = temporary, pool.submit(finish, job, folder, extra)
            if read(folder / 'remote-receipt.json')['returncode']:
                pending.pop(job['name'])[1].result()
                if not keep_going:
                    raise RuntimeError('Slurm/measurement failed; archive downloaded: ' + str(folder))
                print('failed', job['name'], 'returncode', read(folder / 'remote-receipt.json')['returncode'],
                      'archive downloaded; continuing', flush=True)
        for _, future in pending.values():
            future.result()
    print('all jobs complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('quota')
    jobs = sub.add_parser('jobs')
    jobs.add_argument('jobs', type=Path)
    jobs.add_argument('--downloads', type=int, default=6, help='parallel scp downloads (16 ssh streams were reset)')
    jobs.add_argument('--alongside', type=Path, nargs='*', default=[],
                      help="jobs files of a controller still downloading; its checked login copies are not leftovers")
    jobs.add_argument('--keep-going', action='store_true',
                      help='after a non-zero measurement, keep its verified archive and continue with the next job')
    args = parser.parse_args()
    if args.action == 'quota':
        print(json.dumps(scratch_quota(), indent=2))
    else:
        run_jobs(args.jobs, args.downloads, args.alongside, args.keep_going)

"""Filesystem, independent gates and bounded process primitives for suite v2."""
from __future__ import annotations
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import time

DIMENSIONS = ('goal_alignment', 'correctness', 'evidence_reproducibility', 'readability', 'compatibility_regression')
MUTABLE_GATES = {'implementation_status.json', 'campaign_status.json', 'COMPLETE'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=no_duplicates,
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + token)))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as out:
        out.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def relative(root, name, *, exists=True):
    """Disallow symlinks at every path component, including an otherwise in-root link."""
    require(isinstance(name, str) and name and '\\' not in name, 'invalid archive relative path')
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and '..' not in rel.parts and '.' not in rel.parts, 'path escape: ' + name)
    root = Path(root).resolve()
    item = root
    for part in rel.parts:
        item = item / part
        require(not item.is_symlink(), 'symlink forbidden: ' + name)
    require(item.resolve().is_relative_to(root), 'path escape: ' + name)
    if exists:
        require(item.is_file(), 'missing artifact: ' + name)
    return item


def verify_files(root, mapping):
    require(isinstance(mapping, dict) and mapping, 'empty file manifest')
    for name, expected in mapping.items():
        require(isinstance(expected, str) and len(expected) == 64, 'invalid digest: ' + name)
        require(sha(relative(root, name)) == expected, 'hash mismatch: ' + name)


def gate_mapping(review):
    gate = review.get('gate_files')
    if isinstance(gate, list):
        hashes = review.get('gate_files_sha256', {})
        require(len(gate) == len(set(gate)) and set(gate) == set(hashes), 'gate list/hash disagreement')
        gate = hashes
    require(isinstance(gate, dict) and gate, 'missing gate_files')
    require(not any(PurePosixPath(p).name in MUTABLE_GATES for p in gate), 'mutable gate file')
    return gate


def validate_gate(path, root, stage, phase):
    review = read_json(path)
    require(review.get('schema_version') == 2 and review.get('stage') == stage and review.get('phase', 'review') == phase, 'review identity')
    require(review.get('status') == 'pass', f'{stage}-{phase} gate not pass')
    require(review.get('reviewer') and review.get('implementer') and review['reviewer'] != review['implementer'], 'review not independent')
    checks = review.get('checks', {})
    require(all(checks.get(key, {}).get('status') == 'pass' for key in DIMENSIONS), 'review dimensions not pass')
    require(not any(f.get('status') not in ('closed', 'resolved', 'fixed', 'verified') for f in review.get('findings', [])), 'open review finding')
    verify_files(root, gate_mapping(review))
    return review


def cv(values):
    import statistics
    require(len(values) >= 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in values), 'CV values')
    return statistics.stdev(values) / statistics.mean(values)


def stats(values):
    import statistics
    require(len(values) >= 2, 'statistics need two samples')
    return {'samples': len(values), 'median': statistics.median(values), 'mean': statistics.mean(values), 'min': min(values), 'max': max(values), 'cv': cv(values)}


@contextlib.contextmanager
def file_lock(path):
    """Never unlink flock files: unlinking admits two owners on separate inodes."""
    path = Path(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o666)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid in (0, os.geteuid()), 'untrusted lock file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def suite_lock_path(suite):
    suite = Path(suite).resolve()
    # The suite lives on shared storage; an inode here serializes jobs on different nodes.
    return suite.parent / ('.' + suite.name + '.suite-v2.lock')


def gpu_paths(uuid):
    import re
    require(re.fullmatch(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', uuid, re.I) is not None, 'invalid physical GPU UUID')
    base = Path('/tmp') / ('gh200-suite-v2-' + uuid.lower())
    return Path(str(base) + '.lock'), Path(str(base) + '.quarantine.json')


@contextlib.contextmanager
def gpu_lock(uuid):
    lock, quarantine = gpu_paths(uuid)
    with file_lock(lock):
        require(not quarantine.exists(), 'GPU quarantined after unconfirmed process cleanup')
        from common.gpu_registry import reconcile
        recovered = reconcile(uuid)
        yield recovered


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_group(process, grace=2.0):
    actions = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not group_alive(process.pid):
            return True, actions
        try:
            os.killpg(process.pid, sig)
            actions.append(sig.name)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            process.poll()
            if not group_alive(process.pid):
                return True, actions
            time.sleep(0.02)
    process.poll()
    return not group_alive(process.pid), actions


def proc_start_ticks(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def bounded(argv, cwd, stdout, stderr, timeout, *, uuid=None):
    """Persist UUID-wide launch intent before Popen and cleanup proof before return."""
    from common import gpu_registry
    start = time.time_ns()
    receipt = {'argv': list(map(str, argv)), 'host_start_ns': start, 'timeout_seconds': timeout,
               'timed_out': False, 'cleanup_confirmed': True, 'signals': [], 'pid': None, 'pgid': None}
    registration = gpu_registry.begin(uuid, argv, cwd, stdout, stderr) if uuid else None
    process = None
    try:
        with Path(stdout).open('x') as out, Path(stderr).open('x') as err:
            process = subprocess.Popen(argv, cwd=cwd, stdout=out, stderr=err, start_new_session=True)
            receipt.update(pid=process.pid, pgid=process.pid)
            central = gpu_registry.spawned(registration, process) if registration else None
            atomic_json(Path(str(stdout) + '.active.json'), {'pid': process.pid, 'pgid': process.pid,
                'host': os.uname().nodename, 'boot_id': gpu_registry.controller_identity()['boot_id'], 'uid': os.geteuid(),
                'start_ticks': proc_start_ticks(process.pid), 'gpu_uuid': uuid, 'slurm_job': os.environ.get('SLURM_JOB_ID'),
                'argv': list(map(str, argv)), 'registry_id': central['registry_id'] if central else None})
            try:
                receipt['returncode'] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                receipt['timed_out'] = True
    except (OSError, ValueError) as exc:
        receipt.update(returncode=None, launch_error=str(exc))
    finally:
        if process is not None:
            if group_alive(process.pid):
                clean, signals = stop_group(process)
                receipt.update(cleanup_confirmed=clean, signals=signals)
            receipt['returncode'] = process.returncode
    receipt['host_stop_ns'] = time.time_ns()
    receipt['stdout_sha256'] = sha(stdout) if Path(stdout).is_file() else None
    receipt['stderr_sha256'] = sha(stderr) if Path(stderr).is_file() else None
    if registration:
        receipt['gpu_process_registration'] = gpu_registry.finish(registration, receipt)
    if uuid and not receipt['cleanup_confirmed']:
        atomic_json(gpu_paths(uuid)[1], {'uuid': uuid, 'reason': 'unconfirmed_process_group_cleanup', 'receipt': receipt})
    return receipt


def process_ok(receipt):
    return receipt.get('returncode') == 0 and not receipt['timed_out'] and receipt['cleanup_confirmed'] and 'launch_error' not in receipt


def state(root, status, **fields):
    value = {'schema_version': 2, 'status': status, 'updated_utc': utc(), **fields}
    atomic_json(Path(root) / 'campaign_status.json', value)
    with (Path(root) / 'progress.jsonl').open('a') as out:
        out.write(canonical(value) + '\n')
    return value

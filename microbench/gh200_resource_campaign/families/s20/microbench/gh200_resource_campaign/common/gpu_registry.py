"""Fixed node/UUID process registry. Caller must hold the UUID flock."""
from __future__ import annotations
import os
import copy
from pathlib import Path
import time
import uuid as uuidlib
from common.suite_io import atomic_json, gpu_paths, group_alive, proc_start_ticks, read_json, require, sha, stop_group


def controller_identity():
    return {'host': os.uname().nodename, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'uid': os.geteuid(), 'pid': os.getpid(), 'start_ticks': proc_start_ticks(os.getpid())}


def registry_root(uuid):
    lock, _ = gpu_paths(uuid)
    return lock.with_suffix('.processes')


def trusted_directory(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(mode=0o700)
    require(not path.is_symlink() and path.is_dir() and path.stat().st_uid == os.geteuid() and path.stat().st_mode & 0o077 == 0, 'untrusted GPU process registry directory')
    return path


def quarantine(uuid, reason, evidence):
    atomic_json(gpu_paths(uuid)[1], {'uuid': uuid, 'reason': reason, 'evidence': evidence})
    raise ValueError(reason + '; GPU quarantined')


def registry_dirs(uuid):
    root = trusted_directory(registry_root(uuid))
    return trusted_directory(root / 'active'), trusted_directory(root / 'closed')


def write_record(path, record):
    atomic_json(path, record)
    # Central state is authoritative for process discovery; a moved/read-only old archive
    # must not erase the central cleanup evidence.
    try:
        atomic_json(Path(record['archive_record']), record)
    except OSError as exc:
        record['archive_copy_error'] = str(exc)
        atomic_json(path, record)


def begin(uuid, argv, cwd, stdout, stderr, *, purpose='probe'):
    reconciled = reconcile(uuid)
    active, _ = registry_dirs(uuid)
    token = f'{time.time_ns()}-{os.getpid()}-{uuidlib.uuid4().hex}'
    path = active / (token + '.json')
    record = {'schema_version': 2, 'registry_id': token, 'gpu_uuid': uuid.lower(), 'state': 'launch_intent',
              'controller': controller_identity(), 'purpose': purpose, 'argv': list(map(str, argv)),
              'cwd': str(Path(cwd).resolve()), 'stdout': str(Path(stdout).resolve()), 'stderr': str(Path(stderr).resolve()),
              'archive_record': str(Path(str(stdout) + '.registry.json').resolve()),
              'events': [{'state': 'launch_intent', 'time_ns': time.time_ns()}], 'prelaunch_reconciliations': reconciled}
    write_record(path, record)  # Must commit before Popen: unknown PID after a crash fails closed.
    return path


def spawned(path, process):
    record = read_json(path)
    record.update(state='spawned', pid=process.pid, pgid=process.pid, start_ticks=proc_start_ticks(process.pid))
    record['events'].append({'state': 'spawned', 'time_ns': time.time_ns()})
    write_record(path, record)
    return record


def finish(path, receipt):
    record = read_json(path)
    record['cleanup_receipt'] = copy.deepcopy(receipt)
    record['state'] = 'cleanup_confirmed' if receipt.get('cleanup_confirmed') is True else 'cleanup_unconfirmed'
    record['events'].append({'state': record['state'], 'time_ns': time.time_ns()})
    write_record(path, record)
    if record['state'] == 'cleanup_confirmed':
        _, closed = registry_dirs(record['gpu_uuid'])
        path.replace(closed / path.name)
    return record


def owned_group_cleanup(active):
    """Returns positive cleanup proof or an unresolved reason; never kills by PID alone."""
    current = controller_identity()
    owner = active.get('controller') or {'host': active.get('host'), 'boot_id': active.get('boot_id'), 'uid': active.get('uid', os.geteuid())}
    if owner.get('host') != current['host'] or owner.get('uid') != current['uid']:
        return False, {'reason': 'host_or_uid_identity_unproven'}
    if owner.get('boot_id') and owner['boot_id'] != current['boot_id']:
        return True, {'reason': 'recorded_prior_boot_cannot_have_live_process'}
    pgid, pid = active.get('pgid'), active.get('pid')
    if type(pgid) is not int or type(pid) is not int or pgid != pid or pid <= 0:
        return False, {'reason': 'launch_intent_without_proven_spawn_identity'}
    if not group_alive(pgid):
        return True, {'reason': 'registered_group_absent', 'pid': pid, 'pgid': pgid}
    if active.get('start_ticks') is None or proc_start_ticks(pid) != active['start_ticks']:
        return False, {'reason': 'PID_start_ticks_unproven_or_reused', 'pid': pid, 'pgid': pgid}
    try:
        if Path(f'/proc/{pid}').stat().st_uid != current['uid'] or os.getpgid(pid) != pgid:
            return False, {'reason': 'live_process_owner_or_group_mismatch'}
    except (OSError, ProcessLookupError):
        return False, {'reason': 'identity_changed_during_reconciliation'}
    class PriorProcess:
        def __init__(self): self.pid = pid
        def poll(self): return None
    clean, signals = stop_group(PriorProcess())
    return clean, {'reason': 'owned_group_cleanup', 'pid': pid, 'pgid': pgid, 'signals': signals}


def reconcile_legacy_ncu(uuid):
    """Fallback for older profiler caches which predate the fixed UUID registry."""
    recovered = []
    current = controller_identity()
    for cache in Path('/tmp').glob('gh200-suite-v2-ncu-*'):
        if cache.is_symlink() or not cache.is_dir() or cache.stat().st_uid != os.geteuid():
            continue
        status_path = cache / 'status.json'
        if not status_path.is_file() or status_path.is_symlink():
            continue
        status = read_json(status_path)
        if str(status.get('fingerprint', {}).get('gpu_uuid', '')).lower() != uuid.lower():
            continue
        receipt_path = cache / 'receipt.json'
        if receipt_path.exists() and read_json(receipt_path).get('cleanup_confirmed') is True:
            continue
        if status.get('state') == 'tool_missing':
            continue  # No profiler process was launched.
        if status.get('launch_controller') == current:
            continue  # This caller is currently between its NCU sentinel and bounded Popen.
        proof_path = cache / 'reconciliation.json'
        if proof_path.exists():
            proof = read_json(proof_path)
            if proof.get('status_sha256') == sha(status_path) and proof.get('cleanup_confirmed') is True:
                continue
        active_path = cache / 'stdout.active.json'
        active = read_json(active_path) if active_path.exists() else {}
        if active.get('gpu_uuid', '').lower() != uuid.lower():
            quarantine(uuid, 'unresolved_legacy_NCU_launch_identity', {'cache': str(cache), 'status': status})
        clean, proof = owned_group_cleanup(active)
        evidence = {'cache': str(cache), 'status_sha256': sha(status_path), 'cleanup_confirmed': clean, 'proof': proof,
                    'execution_outcome': 'unchanged_no_blind_retry', 'reconciled_ns': time.time_ns()}
        atomic_json(proof_path, evidence)
        recovered.append(evidence)
        if not clean:
            quarantine(uuid, 'unresolved_legacy_NCU_process_cleanup', evidence)
    return recovered


def reconcile(uuid):
    active_dir, _ = registry_dirs(uuid)
    require(not gpu_paths(uuid)[1].exists(), 'GPU quarantined; reconciliation requires external verified release')
    recovered = []
    current = controller_identity()
    for path in sorted(active_dir.glob('*.json')):
        require(not path.is_symlink() and path.stat().st_uid == os.geteuid(), 'untrusted active registration')
        active = read_json(path)
        require(active.get('gpu_uuid') == uuid.lower(), 'registry UUID mismatch')
        # A live telemetry process owned by this exact controller is intentionally concurrent.
        if active.get('purpose') == 'telemetry' and active.get('controller') == current and active.get('state') == 'spawned':
            require(active.get('start_ticks') is not None and proc_start_ticks(active['pid']) == active['start_ticks'], 'current telemetry identity lost')
            continue
        clean, proof = owned_group_cleanup(active)
        receipt = {'cleanup_confirmed': clean, 'recovery': proof, 'reconciled_ns': time.time_ns()}
        recovered.append(finish(path, receipt))
        if not clean:
            quarantine(uuid, 'unresolved_registered_GPU_process', {'registry_id': active['registry_id'], 'proof': proof})
    recovered.extend(reconcile_legacy_ncu(uuid))
    return recovered

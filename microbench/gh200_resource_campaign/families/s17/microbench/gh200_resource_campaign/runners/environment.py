"""ROMEO allocation evidence, time budget and one-shot counter environment checks."""
from __future__ import annotations
import csv
import datetime as dt
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from common.suite_io import atomic_json, bounded, digest, process_ok, read_json, require, sha, verify_files


class Checkpoint(Exception):
    pass


def capture(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    require(result.returncode == 0, 'environment query failed: ' + ' '.join(argv) + '\n' + result.stderr)
    return result.stdout


def check_allocation_time(job_info):
    """Accept finite batch allocations; per-probe timeouts remain independent."""
    partitions = re.findall(r'(?:^|\s)Partition=([^\s]+)', job_info)
    limits = re.findall(r'(?:^|\s)TimeLimit=([^\s]+)', job_info)
    require(len(partitions) == 1 and len(limits) == 1,
            'one Slurm Partition and TimeLimit required')
    maxima = {'instant': 3600, 'short': 3 * 3600}
    require(partitions[0] in maxima, 'only instant or short allocations supported')
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)', limits[0])
    require(match is not None, 'finite Slurm TimeLimit in [days-]HH:MM:SS required')
    days, hours, minutes, seconds = (int(value or 0) for value in match.groups())
    duration = days * 86400 + hours * 3600 + minutes * 60 + seconds
    require(0 < duration <= maxima[partitions[0]],
            'allocation exceeds campaign limit: instant <= 1 hour, short <= 3 hours')


def inspect_allocation():
    job = os.environ.get('SLURM_JOB_ID', '')
    require(re.fullmatch(r'[0-9]+', job) is not None, 'active Slurm allocation required')
    config = capture(['scontrol', 'show', 'config'])
    require(re.search(r'ClusterName\s*=\s*\S*romeo', config, re.I) is not None, 'ROMEO cluster required')
    job_info = capture(['scontrol', 'show', 'job', '-o', job])
    require(re.search(r'JobState=RUNNING\b', job_info) is not None, 'Slurm job not RUNNING')
    check_allocation_time(job_info)
    nodes = re.search(r'\bNodeList=([^ ]+)', job_info)
    require(nodes is not None, 'Slurm node list missing')
    allocated_hosts = capture(['scontrol', 'show', 'hostnames', nodes[1]]).split()
    require(os.uname().nodename.split('.')[0] in [h.split('.')[0] for h in allocated_hosts], 'runner is not on allocated compute node')
    require(re.search(r'UserId=[^ ]*\(' + str(os.geteuid()) + r'\)', job_info) is not None, 'Slurm allocation ownership mismatch')
    require(re.search(r'(?:gres/gpu=1(?:[,\s]|$)|gres/gpu:[^=, ]+=1(?:[,\s]|$))', job_info) is not None, 'exactly one Slurm allocated GPU required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(visible and ',' not in visible and visible != '-1', 'exactly one visible CUDA device required')
    gpu_text = capture(['nvidia-smi', '-i', visible, '--query-gpu=uuid,name,driver_version', '--format=csv,noheader,nounits'])
    rows = list(csv.reader(gpu_text.strip().splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 3, 'single physical GPU query')
    uuid, name, driver = (value.strip() for value in rows[0])
    require('GH200' in name, 'GH200 required')
    nvcc = shutil.which('nvcc')
    require(nvcc is not None, 'nvcc required')
    compiler = capture([nvcc, '--version'])
    require('release 12.9' in compiler, 'CUDA 12.9 compiler required')
    tools = {}
    for name_ in ('nvcc', 'cuobjdump', 'nvidia-smi', 'scontrol'):
        path = shutil.which(name_)
        require(path is not None, 'missing tool: ' + name_)
        tools[name_] = {'path': str(Path(path).resolve()), 'sha256': sha(path)}
    tools['python'] = {'path': str(Path(sys.executable).resolve()), 'sha256': sha(sys.executable), 'version': sys.version}
    return {'uuid': uuid, 'name': name, 'driver': driver, 'compiler': compiler, 'tools': tools,
            'job': job, 'job_info': job_info, 'slurm_config': config, 'cuda_visible_devices': visible,
            'host': os.uname().nodename, 'execution_uid': os.geteuid()}


def environment_identity(env):
    return {key: env[key] for key in ('uuid', 'name', 'driver', 'compiler', 'tools', 'execution_uid')}


def remaining_seconds(job):
    row = capture(['scontrol', 'show', 'job', '-o', job])
    match = re.search(r'\bEndTime=([^\s]+)', row)
    require(match is not None and match[1] not in ('Unknown', 'None', 'N/A'), 'Slurm EndTime unavailable')
    end = dt.datetime.fromisoformat(match[1]).timestamp()
    return end - time.time()


def budget(env, timeout):
    left = remaining_seconds(env['job'])
    if left < timeout + 20:
        raise Checkpoint(f'job budget {left:.1f}s < timeout {timeout}s + cleanup/checkpoint 20s')


def permission_fingerprint(env, ncu):
    controls = {}
    for path in ('/proc/driver/nvidia/params', '/proc/self/status'):
        if Path(path).exists():
            text = Path(path).read_text()
            controls[path] = [line for line in text.splitlines() if any(key in line for key in ('RmProfilingAdminOnly', 'CapEff:', 'Groups:', 'NoNewPrivs:'))]
    version = capture([ncu, '--version']) if ncu else 'tool_missing'
    return {'gpu_uuid': env['uuid'], 'driver_version': env['driver'], 'ncu_version': version,
            'execution_identity': {'uid': os.geteuid(), 'groups': sorted(os.getgroups())}, 'observable_permission_configuration': controls}


def counters(run, spec, contract, env):
    """Caller holds UUID lock. Started sentinel prevents retry after a killed profiler."""
    from common.gpu_registry import reconcile, controller_identity
    run = Path(run)
    recovered = reconcile(env['uuid'])
    if recovered:
        atomic_json(run / 'environment' / ('ncu_prior_recovery_' + str(time.time_ns()) + '.json'), recovered)
    ncu = shutil.which('ncu')
    fingerprint = permission_fingerprint(env, ncu)
    key = digest(fingerprint)
    cache = Path('/tmp') / ('gh200-suite-v2-ncu-' + key)
    if cache.exists():
        require(not cache.is_symlink() and cache.stat().st_uid == os.geteuid(), 'untrusted NCU cache')
        old = read_json(cache / 'status.json')
        require(old['fingerprint'] == fingerprint, 'NCU cache fingerprint mismatch')
        if old.get('evidence_sha256'):
            verify_files(cache, old['evidence_sha256'])
        for item in cache.iterdir():
            require(item.is_file() and not item.is_symlink(), 'invalid NCU cache artifact')
            shutil.copyfile(item, run / 'environment' / ('ncu_' + item.name))
        require(old['state'] in ('available', 'permission_denied', 'tool_missing'), 'cached NCU outcome unresolved/failed; no blind retry or further sampling')
        return old
    budget(env, 120)
    cache.mkdir(mode=0o700)
    status = {'schema_version': 2, 'purpose': 'permission_capability_only', 'family_profile': False, 'cache_residency_proven': False, 'physical_hbm_bytes_proven': False, 'fingerprint': fingerprint, 'state': 'started_outcome_unknown', 'no_blind_retry': True, 'launch_controller': controller_identity()}
    atomic_json(cache / 'status.json', status)
    if not ncu:
        status['state'] = 'tool_missing'
        status['unavailable_reason'] = 'executable_not_found'
        atomic_json(cache / 'tool_lookup.json', {'executable': 'ncu', 'resolved_path': None, 'PATH': os.environ.get('PATH', '')})
        status['evidence_sha256'] = {'tool_lookup.json': sha(cache / 'tool_lookup.json')}
    else:
        cfg = contract.get('ncu', {'case': contract['cases'][0]['id'], 'target_kernel': '.*', 'metrics': ['gpu__time_duration.sum']})
        case = next(c for c in contract['cases'] if c['id'] == cfg['case'])
        argv = [ncu, '--clock-control', 'none', '--cache-control', 'none', '--kernel-name', 'regex:' + cfg['target_kernel'], '--launch-count', '1', '--metrics', ','.join(cfg['metrics']), '--csv', str(run / 'binary/probe'), case['id'], str(case['iterations']), '3']
        receipt = bounded(argv, run, cache / 'stdout', cache / 'stderr', 120, uuid=env['uuid'])
        atomic_json(cache / 'receipt.json', receipt)
        text = (cache / 'stdout').read_text() + (cache / 'stderr').read_text()
        status['state'] = 'cleanup_unconfirmed' if not receipt['cleanup_confirmed'] else 'timeout' if receipt['timed_out'] else 'failed_unknown' if receipt.get('launch_error') else 'available' if process_ok(receipt) else 'permission_denied' if 'ERR_NVGPUCTRPERM' in text else 'failed_unknown'
        status['evidence_sha256'] = {p.name: sha(p) for p in cache.iterdir() if p.name != 'status.json'}
    atomic_json(cache / 'status.json', status)
    for item in cache.iterdir():
        shutil.copyfile(item, run / 'environment' / ('ncu_' + item.name))
    require(status['state'] in ('available', 'permission_denied', 'tool_missing'), 'NCU execution or cleanup unresolved/failed; stop GPU sampling')
    return status

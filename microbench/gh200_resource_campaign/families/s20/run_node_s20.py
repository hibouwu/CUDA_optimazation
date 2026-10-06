"""One S20 node-local run and bounded final offhost handoff; no C qualification."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

if sys.flags.optimize != 0:
    raise RuntimeError('S20 requires active assertions; Python optimization forbidden')
import time
import uuid

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'microbench/gh200_resource_campaign'))
from common.suite_io import atomic_json, file_lock, read_json, require, sha, validate_gate, verify_files
from runners.environment import inspect_allocation, budget
from point_pack import inventory
from run_s20 import run, storage_budget


def execute(deployment, binary):
    require(ROOT.parent == deployment, 'exact approved deployment')
    require(not any((deployment / n).exists() for n in
                    ('node-pointer.json', 'transfer-ready.json', 'offhost-receipt.json', 'closure.json', 'checkpoint.json')),
            'prior run identity must be recovered, never overwritten')
    gate = validate_gate(ROOT / 'source-review.json', ROOT, 'S20', 'full-value-family-run-source-B')
    require(gate['authorization']['formal_after_full_short_and_own_pilot'] is True, 'conditional matrix admission')
    verify_files(ROOT, read_json(ROOT / 'source-manifest.json'))
    env = inspect_allocation()
    queue = subprocess.run(['squeue', '-u', str(os.geteuid()), '-h', '-o', '%i|%b'],
                           capture_output=True, text=True, timeout=30)
    require(queue.returncode == 0, 'owner queue unavailable')
    require([line.split('|')[0] for line in queue.stdout.splitlines() if 'gpu' in line.split('|')[1].lower()] == [env['job']],
            'only this owner GPU job may measure')
    require(shutil.disk_usage('/tmp').free >= storage_budget()['required_free_bytes'],
            'complete S20 packet/raw/reserve budget; per-point checks still required')
    budget(env, 1200)
    node = Path('/tmp') / ('codex-s20-node-' + env['job'] + '-' + uuid.uuid4().hex[:12])
    require(not node.exists(), 'new private node namespace')
    pointer = {'job': env['job'], 'node_root': str(node), 'host': env['host'], 'uid': os.geteuid(),
               'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'uuid': env['uuid'],
               'durable_offhost_custody_proven': False}
    atomic_json(deployment / 'node-pointer.json', pointer)
    failure = None
    try:
        run(binary, node)
    except BaseException as error:
        failure = repr(error)
    if not node.exists():
        node.mkdir(mode=0o700)
        atomic_json(node / 'failure-before-output.json', {'error': failure, 'targets_started': False, 'qualified': False})
    require(node.is_dir() and not node.is_symlink() and node.stat().st_uid == os.geteuid()
            and node.stat().st_mode & 0o077 == 0, 'original owned data tree must exist for handoff')
    # Partial raw is already inside the root; successful points retain their
    # complete packs. Neither branch discards prior evidence.
    shutil.copytree(ROOT, node / 'repo')
    verify_files(node / 'repo', read_json(ROOT / 'source-manifest.json'))
    manifest = inventory(node)
    atomic_json(node / 'transfer-manifest.json', manifest)
    ready = {**pointer, 'sampling_complete': failure is None, 'error': failure,
             'transfer_manifest_sha256': sha(node / 'transfer-manifest.json'),
             'file_count': len(manifest), 'file_bytes': sum(x['bytes'] for x in manifest.values()), 'case_count': 105}
    atomic_json(deployment / 'transfer-ready.json', ready)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        path = deployment / 'offhost-receipt.json'
        if path.exists():
            require(not path.is_symlink(), 'receipt symlink')
            receipt = read_json(path)
            require(all(receipt.get(k) == ready[k] for k in
                        ('job', 'node_root', 'host', 'boot_id', 'transfer_manifest_sha256')), 'original custody identity')
            require(receipt.get('all_members_verified') is True and receipt.get('durable_offhost_custody_proven') is True
                    and receipt.get('file_count') == len(manifest), 'complete offhost custody required')
            verify_files(node, {n: x['sha256'] for n, x in manifest.items()})
            atomic_json(deployment / 'closure.json', {'sampling_complete': failure is None,
                        'offhost_receipt_sha256': sha(path), 'node_files_deleted': False, 'formal_C_required': True})
            require(failure is None, 'failed target evidence preserved; no automatic retry')
            print(json.dumps({'status': 'S20_collected_pending_independent_C'}))
            return
        time.sleep(2)
    atomic_json(deployment / 'checkpoint.json', {**ready, 'reason': 'offhost_receipt_timeout600',
                'node_files_deleted': False, 'do_not_resample_unknown_points': True})
    raise SystemExit(75)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--binary', required=True)
    args = parser.parse_args()
    deployment = Path(args.deployment).resolve()
    require(ROOT.parent == deployment, 'controller lock belongs to this deployment')
    with file_lock(deployment / '.s20-controller.lock'):
        execute(deployment, args.binary)

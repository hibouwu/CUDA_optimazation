"""Controlled execution; family arithmetic stays in independent auditors."""
from __future__ import annotations
import contextlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from auditors.suite import adapter, audit, case_evidence, load_run, parse_raw, protocol_check, recompute, report
from common.suite_io import atomic_json, bounded, digest, file_lock, gate_mapping, gpu_lock, gpu_paths, group_alive, proc_start_ticks, process_ok, read_json, relative, require, sha, state, stop_group, suite_lock_path, utc, validate_gate, verify_files
from common.calibration import calibrated_cases, pilot_case, seal_resolution, resolved_contract, import_resolved_binary
from common.suite_snapshot import CAMPAIGN, check_snapshot, freeze, inventory_depfile
from runners.environment import Checkpoint, budget, counters, environment_identity, inspect_allocation


def preserve_orphan(root, prefix, env, gpu):
    files = [Path(str(prefix) + suffix) for suffix in ('.stdout', '.stderr', '.stdout.active.json')]
    if not any(p.exists() for p in files):
        return
    active_path = files[2]
    active = read_json(active_path) if active_path.exists() else None
    clean = False
    if active and active.get('host') == os.uname().nodename:
        clean = not group_alive(active['pgid'])
        if not clean and active.get('start_ticks') is not None and proc_start_ticks(active['pid']) == active['start_ticks']:
            class PriorProcess:
                pid = active['pid']
                def poll(self): return None
            clean, _ = stop_group(PriorProcess())
    elif not gpu:
        # CPU command evidence from a previous allocation may be retained without GPU claims.
        clean = True
    if not clean:
        atomic_json(gpu_paths(env['uuid'])[1], {'reason': 'orphan_command_cleanup_unconfirmed', 'active': active, 'prefix': str(prefix)})
        raise ValueError('orphan command group unconfirmed; GPU quarantined')
    folder = prefix.parent / (prefix.name + '.interrupted_' + str(time.time_ns()))
    folder.mkdir()
    for path in files:
        if path.exists():
            path.rename(folder / path.name)
    if prefix.name == 'compile' and (root / 'binary/probe').exists():
        (root / 'binary/probe').rename(folder / 'uncommitted_probe')
    atomic_json(folder / 'receipt.json', {'status': 'interrupted', 'cleanup_confirmed': True, 'prior_active': active,
                'artifacts': {p.name: sha(p) for p in folder.iterdir() if p.is_file()}})


def verify_interrupted_gpu_groups(root, env):
    """Before any new device query, rule out orphaned work from the prior allocation."""
    for path in root.rglob('*.active.json'):
        active = read_json(path)
        if not active.get('gpu_uuid'):
            continue
        if path.name == 'raw.jsonl.active.json' or path.name == 'telemetry.active.json':
            receipt_path = path.parent / 'receipt.json'
        else:
            receipt_path = path.with_name(path.name.removesuffix('.stdout.active.json') + '.receipt.json')
        if receipt_path.exists() and read_json(receipt_path).get('cleanup_confirmed') is True:
            continue
        clean = active.get('host') == os.uname().nodename and not group_alive(active['pgid'])
        if not clean and active.get('host') == os.uname().nodename and active.get('start_ticks') is not None and proc_start_ticks(active['pid']) == active['start_ticks']:
            class PriorProcess:
                pid = active['pid']
                def poll(self): return None
            clean, _ = stop_group(PriorProcess())
        if not clean:
            atomic_json(gpu_paths(env['uuid'])[1], {'reason': 'orphan_gpu_group_unconfirmed', 'active': active})
            raise ValueError('prior GPU group unconfirmed; no new device commands permitted')
        if path.name == 'telemetry.active.json' and not receipt_path.exists():
            atomic_json(receipt_path, {'status': 'interrupted', 'cleanup_confirmed': True,
                'samples_sha256': sha(path.parent / 'samples.csv'), 'stderr_sha256': sha(path.parent / 'stderr')})


def run_command(root, name, argv, timeout, env, *, gpu=False, cwd=None):
    prefix = root / name
    prior = Path(str(prefix) + '.receipt.json')
    if prior.exists():
        receipt = read_json(prior)
        require(process_ok(receipt) and receipt['argv'] == list(map(str, argv)), 'cannot reuse unsuccessful/changed command: ' + name)
        require(receipt['stdout_sha256'] == sha(Path(str(prefix) + '.stdout')) and receipt['stderr_sha256'] == sha(Path(str(prefix) + '.stderr')), 'command evidence changed: ' + name)
        return receipt
    preserve_orphan(root, prefix, env, gpu)
    budget(env, timeout)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    receipt = bounded(argv, cwd or root, Path(str(prefix) + '.stdout'), Path(str(prefix) + '.stderr'), timeout, uuid=env['uuid'] if gpu else None)
    atomic_json(Path(str(prefix) + '.receipt.json'), receipt)
    require(process_ok(receipt), 'command failed; retained evidence: ' + name)
    return receipt


def compile_probe(root, spec, contract, env):
    source = root / spec['source_path']
    command = [env['tools']['nvcc']['path'], *contract['build']['flags'], '-MD', '-MF', str(root / 'build/probe.d'), str(source), '-o', str(root / 'binary/probe')]
    atomic_json(root / 'build/command.json', {'argv': command, 'source_relative': spec['source_path'], 'target_relative': 'binary/probe'})
    run_command(root, 'build/compile', command, 180, env, cwd=root / 'snapshot/repo')
    inventory_depfile(root, root / 'build/probe.d', toolchain_roots=[Path(env['tools']['nvcc']['path']).resolve().parent.parent])
    run_command(root, 'build/disassemble', [env['tools']['cuobjdump']['path'], '--dump-sass', str(root / 'binary/probe')], 120, env)
    shutil.copyfile(root / 'build/disassemble.stdout', root / 'build/sass.txt')
    sass_evidence = adapter(contract).audit_sass((root / 'build/sass.txt').read_text(), contract)
    atomic_json(root / 'build/sass_audit.json', sass_evidence)
    # Shared libraries are external environment dependencies, never claimed copied.
    run_command(root, 'build/ldd', ['ldd', str(root / 'binary/probe')], 120, env)
    external = {}
    for line in (root / 'build/ldd.stdout').read_text().splitlines():
        for token in line.split():
            if token.startswith('/') and Path(token).is_file():
                external[str(Path(token).resolve())] = sha(token)
    atomic_json(root / 'build/shared_libraries.json', external)
    spec.update(binary_sha256=sha(root / 'binary/probe'), sass_sha256=sha(root / 'build/sass.txt'), dependencies_sha256=sha(root / 'build/dependencies.json'), shared_libraries_sha256=sha(root / 'build/shared_libraries.json'))
    atomic_json(root / 'run_spec.json', spec)


@contextlib.contextmanager
def telemetry(root, env):
    from common import gpu_registry
    directory = root / 'environment' / ('telemetry_' + str(time.time_ns()))
    directory.mkdir()
    out = err = process = registration = None
    start_ticks = None
    argv = [env['tools']['nvidia-smi']['path'], '-i', env['uuid'], '--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu', '--format=csv', '-lms', '100']
    try:
        out = (directory / 'samples.csv').open('x')
        err = (directory / 'stderr').open('x')
        registration = gpu_registry.begin(env['uuid'], argv, root, directory/'samples.csv', directory/'stderr', purpose='telemetry')
        process = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
        start_ticks = proc_start_ticks(process.pid)
        gpu_registry.spawned(registration, process)
        atomic_json(directory / 'telemetry.active.json', {'pid': process.pid, 'pgid': process.pid, 'host': os.uname().nodename, 'start_ticks': start_ticks, 'gpu_uuid': env['uuid']})
        yield
    finally:
        clean, signals = stop_group(process) if process is not None else (True, [])
        if out is not None: out.close()
        if err is not None: err.close()
        if registration is not None:
            receipt = {'pid': process.pid if process else None, 'pgid': process.pid if process else None,
                       'start_ticks': start_ticks, 'cleanup_confirmed': clean, 'signals': signals,
                       'returncode': process.returncode if process else None,
                       'samples_sha256': sha(directory/'samples.csv') if (directory/'samples.csv').exists() else None,
                       'stderr_sha256': sha(directory/'stderr') if (directory/'stderr').exists() else None}
            if not clean:
                atomic_json(gpu_paths(env['uuid'])[1], {'reason': 'telemetry_cleanup_unconfirmed', 'pid': receipt['pid']})
            receipt['gpu_process_registration'] = gpu_registry.finish(registration, receipt)
            atomic_json(directory / 'receipt.json', receipt)
        require(clean, 'telemetry cleanup unconfirmed: GPU quarantined')


def trial(root, spec, contract, case, device, protocol, batch, trial_index, env, *, preflight=False, calibration=False):
    require(not gpu_paths(env['uuid'])[1].exists(), 'GPU quarantined; no further trials')
    parent = root / ('calibration' if calibration else 'preflight' if preflight else 'batches') / case['id'] / f'batch_{batch:02d}' / f'trial_{trial_index:02d}'
    parent.mkdir(parents=True, exist_ok=True)
    for attempt in sorted(parent.glob('attempt_*')):
        if (attempt / 'receipt.json').exists():
            receipt = read_json(attempt / 'receipt.json')
            if receipt.get('status') in ('valid', 'warmup_unconverged'):
                # A frozen read validates an already accepted attempt before skipping it.
                from auditors.suite import trial_evidence
                return trial_evidence(root, spec, case, device, protocol, batch, trial_index, attempt)
            require(receipt.get('status') == 'interrupted', 'prior process failure prohibits replacing a sample')
        else:
            active_path = attempt / 'raw.jsonl.active.json'
            active = read_json(active_path) if active_path.exists() else None
            clean = not any(attempt.iterdir()) or (active is not None and active.get('host') == os.uname().nodename and not group_alive(active['pgid']))
            if not clean:
                atomic_json(gpu_paths(env['uuid'])[1], {'reason': 'orphan_trial_cleanup_unconfirmed', 'active': active, 'case': case['id']})
                raise ValueError('orphan trial group unconfirmed; GPU quarantined')
            atomic_json(attempt / 'receipt.json', {'status': 'interrupted', 'reason': 'no committed receipt after allocation interruption', 'files': {p.name: sha(p) for p in attempt.iterdir() if p.is_file()}})
    budget(env, protocol['process_timeout_seconds'])
    attempts = list(parent.glob('attempt_*'))
    require(len(attempts) < 3, 'too many allocation interruptions for one trial')
    folder = parent / f'attempt_{len(attempts):02d}'
    folder.mkdir()
    seed = 3 + trial_index * 19 + batch * 1009
    logical = ['binary/probe', case['id'], str(case['iterations']), str(seed)]
    receipt = bounded([str(root / logical[0]), *logical[1:]], root, folder / 'raw.jsonl', folder / 'stderr', protocol['process_timeout_seconds'], uuid=env['uuid'])
    receipt.update(relative_command=logical, case_id=case['id'], batch=batch, trial=trial_index, seed=seed,
                   binary_sha256=spec['binary_sha256'], raw_sha256=sha(folder / 'raw.jsonl'), status='failed')
    try:
        require(process_ok(receipt), 'probe process failed/timeout')
        observed_device, row = parse_raw(folder / 'raw.jsonl')
        require(observed_device == device, 'device changed')
        result = adapter(contract).validate_trial(row, case, device, seed, protocol)
        if 'resolved_resource_identity' in case:
            require(all(row.get(k) == v for k,v in case['resolved_resource_identity'].items()), 'kernel resources changed from B-bound calibration')
        receipt['status'] = 'valid' if result['warmup_converged'] else 'warmup_unconverged'
        receipt['validation'] = result
    except Exception as exc:
        receipt['validation_error'] = str(exc)
        atomic_json(folder / 'receipt.json', receipt)
        raise
    atomic_json(folder / 'receipt.json', receipt)
    return result


def formal_rounds(root, spec, contract, device, protocol, env):
    # Seed and every full round's order are archived. Checkpoints resume missing pairs.
    cases = {c['id']: c for c in contract['cases']}
    for batch in range(3):
        rows = {key: case_evidence(root, spec, c, device, protocol) for key, c in cases.items()}
        active = []
        for key, evidence in rows.items():
            if evidence['status'] != 'pending':
                continue
            prior = evidence['batches']
            next_batch = len(prior) if prior and (prior[-1]['complete'] or prior[-1]['warmup_failed']) else max(0, len(prior) - 1)
            if next_batch == batch:
                active.append(key)
        if not active:
            continue
        order_path = root / 'batches' / f'order_{batch:02d}.json'
        if order_path.exists():
            order = read_json(order_path)
        else:
            rng = random.Random(protocol['shuffle_seed'] + batch)
            order = []
            for index in range(10):
                shuffled = sorted(active)
                rng.shuffle(shuffled)
                order.extend([[key, index] for key in shuffled])
            atomic_json(order_path, order)
        # Recover active membership from the immutable planned round order.
        for key, index in order:
            evidence = case_evidence(root, spec, cases[key], device, protocol)
            if evidence['status'] != 'pending':
                continue
            current = next((b for b in evidence['batches'] if b['batch'] == batch), None)
            if current and current['warmup_failed']:
                continue
            result = trial(root, spec, contract, cases[key], device, protocol, batch, index, env)
            print(f'{key} batch={batch} trial={index} warmup={result["warmup_converged"]}', flush=True)
    return recompute(root, require_terminal=True)


def measurement_manifest(root):
    excluded = {'measurement_manifest.json', 'campaign_status.json', 'COMPLETE'}
    paths = [p for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in excluded and not p.relative_to(root).as_posix().startswith('reviews/C/')]
    mapping = {p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)}
    atomic_json(root / 'measurement_manifest.json', mapping)


def bind_verified_device(root, spec):
    actual = sha(root / 'environment/device.json')
    if 'device_sha256' in spec:
        require(spec['device_sha256'] == actual, 'retained device evidence changed')
    else:
        spec['device_sha256'] = actual
        atomic_json(root / 'run_spec.json', spec)


def execute(root, suite, *, preflight=False):
    root, suite = Path(root).resolve(), Path(suite).resolve()
    with file_lock(suite_lock_path(suite)):
        spec = read_json(root / 'run_spec.json')
        require(spec['kind'] == ('preflight' if preflight else 'formal'), 'cannot change run kind')
        require(not (root / 'measurement_manifest.json').exists() and not (root / 'preflight_summary.json').exists(), 'measurement already sealed; audit/review/finalize instead of resume')
        contract = check_snapshot(root, spec)
        family = adapter(contract)
        family.validate_contract(contract)
        if not preflight and calibrated_cases(contract):
            require(all(key in spec for key in ('calibration_import', 'resolved_cases', 'binary_sha256')), 'incomplete B binary import; preserve this run and import into a new ID, never recompile')
        protocol = read_json(root / 'snapshot/repo/microbench/gh200_resource_campaign/contracts/protocol.json')
        protocol_check(protocol)
        try:
            env = inspect_allocation()
            initial = read_json(root / 'environment/initial.json')
            require(environment_identity(env) == environment_identity(initial), 'environment drift: create a new run')
            atomic_json(root / 'environment' / ('allocation_' + str(time.time_ns()) + '.json'), env)
            if 'binary_sha256' not in spec:
                compile_probe(root, spec, contract, env)
            else:
                require(sha(root / 'binary/probe') == spec['binary_sha256'], 'retained binary changed')
                for path, expected in read_json(root / 'build/shared_libraries.json').items():
                    require(Path(path).is_file() and sha(path) == expected, 'shared library environment drift')
            with gpu_lock(env['uuid']) as recovered:
                if recovered:
                    atomic_json(root / 'environment' / ('UUID_recovery_' + str(time.time_ns()) + '.json'), recovered)
                verify_interrupted_gpu_groups(root, env)
                if not (root / 'environment/device.json').exists():
                    run_command(root, 'environment/device_query', [str(root / 'binary/probe'), 'device'], 120, env, gpu=True)
                    device = read_json(root / 'environment/device_query.stdout')
                    family.validate_device(device)
                    require(device['uuid'].lower() == env['uuid'].lower(), 'CUDA/nvidia-smi UUID mismatch')
                    atomic_json(root / 'environment/device.json', device)
                    spec['device_sha256'] = sha(root / 'environment/device.json')
                    atomic_json(root / 'run_spec.json', spec)
                else:
                    device = read_json(root / 'environment/device.json')
                    # New process verifies runtime identity even when resuming on the same UUID.
                    name = 'environment/resume_device_' + str(time.time_ns())
                    run_command(root, name, [str(root / 'binary/probe'), 'device'], 120, env, gpu=True)
                    require(read_json(root / (name + '.stdout')) == device, 'runtime/device drift')
                    bind_verified_device(root, spec)
                _, loaded_contract, _, _, _ = load_run(root)
                from common.family_b3 import enabled, verify_build
                if enabled(contract):
                    loaded_contract = verify_build(root, contract, device, live_environment=env)
                if not preflight or enabled(contract):
                    contract = loaded_contract
                state(root, 'preflight_running' if preflight else 'sampling', slurm_job=env['job'])
                with telemetry(root, env):
                    if preflight:
                        evidence = {}
                        targets = calibrated_cases(contract)
                        if targets:
                            for case in targets:
                                trial(root, spec, contract, pilot_case(case), device, protocol, 0, 0, env, preflight=True, calibration=True)
                            seal_resolution(root, spec, contract, device, protocol)
                            contract = resolved_contract(root, spec, contract, device, protocol)
                        for case in contract['cases']:
                            evidence[case['id']] = trial(root, spec, contract, case, device, protocol, 0, 0, env, preflight=True)
                    else:
                        # Counter check does not contaminate formal trial timing; shared lock serializes it.
                        if not (root / 'environment/ncu_status.json').exists():
                            counters(root, spec, contract, env)
                        summary = formal_rounds(root, spec, contract, device, protocol, env)
            if preflight:
                atomic_json(root / 'preflight_summary.json', {'schema_version': 2, 'status': 'preflight_pending_review', 'hardware_qualification': False, 'binary_sha256': spec['binary_sha256'], 'cases': evidence})
                state(root, 'preflight_pending_review', hardware_qualification=False)
            else:
                atomic_json(root / 'summary.json', summary)
                (root / 'REPORT.md').write_text(report(summary))
                state(root, 'collected_pending_review', hardware_qualification=False)
                measurement_manifest(root)
                audit(root)
            return 0
        except Checkpoint as exc:
            state(root, 'checkpoint', reason=str(exc), kind=spec['kind'])
            print('CHECKPOINT: ' + str(exc), flush=True)
            return 75
        except Exception as exc:
            state(root, 'failed', reason=str(exc), kind=spec['kind'], implementation_failure=True)
            raise


def initialize(repo, suite, root, contract_path, *, preflight=False, resolved_cases=None):
    repo, suite, root = Path(repo).resolve(), Path(suite).resolve(), Path(root).resolve()
    contract = read_json(contract_path)
    adapter(contract).validate_contract(contract)
    require(not preflight or resolved_cases is None, 'preflight cannot import another resolution')
    require(preflight or not calibrated_cases(contract) or resolved_cases is not None, 'calibrated formal run requires --resolved-cases with B-bound binary')
    require(calibrated_cases(contract) or resolved_cases is None, 'fixed-only family does not accept --resolved-cases')
    require(root.parent == suite / contract['family'], 'output must be suite/family/run')
    from common.family_b3 import validate_bundle
    validate_bundle(repo, contract)
    from common.suite_snapshot import required_reviews
    for stage, phase, name in required_reviews(contract, preflight):
        validate_gate(relative(suite, name), repo, stage, phase)
    with file_lock(suite_lock_path(suite)):
        require(not root.exists(), 'run directory exists; use explicit resume')
        env = inspect_allocation()
        budget(env, 180)
        root.mkdir(parents=True)
        for name in ('binary', 'build', 'environment', 'reviews', 'batches', 'failures'):
            (root / name).mkdir()
        try:
            frozen = freeze(repo, suite, root, contract_path, preflight=preflight)
            atomic_json(root / 'environment/initial.json', env)
            spec = {'schema_version': 2, 'kind': 'preflight' if preflight else 'formal', 'fixture': False, 'created_utc': utc(), 'suite_id': suite.name, 'family': contract['family'], **frozen,
                    'environment_sha256': sha(root / 'environment/initial.json')}
            atomic_json(root / 'run_spec.json', spec)
            if resolved_cases is not None:
                import_resolved_binary(repo, root, spec, contract, resolved_cases, env)
            state(root, 'frozen_pending_build' if 'binary_sha256' not in spec else 'frozen_B_binary_reused')
        except Exception as exc:
            state(root, 'failed', reason=str(exc), implementation_failure=True)
            raise
    return root / ('snapshot/repo/' + CAMPAIGN + '/run_suite.py')


def finalize(root, suite, review_path):
    root = Path(root).resolve()
    with file_lock(suite_lock_path(suite)):
        require(not (root / 'COMPLETE').exists(), 'already finalized')
        summary = audit(root)
        review = validate_gate(review_path, root, summary['stage'], 'C')
        require({'summary.json', 'measurement_manifest.json'} <= set(gate_mapping(review)), 'C gate must bind summary and immutable evidence manifest')
        folder = root / 'reviews/C'
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(review_path, folder / 'review.json')
        finish = {'schema_version': 2, 'summary_sha256': sha(root / 'summary.json'), 'measurement_manifest_sha256': sha(root / 'measurement_manifest.json'), 'C_review_sha256': sha(folder / 'review.json')}
        # Finalize must not append the measurement journal after it was sealed.
        atomic_json(root / 'campaign_status.json', {'schema_version': 2, 'status': 'complete', 'finalized_utc': utc(), 'qualification': 'independently_reviewed_measurement', **finish})
        atomic_json(root / 'COMPLETE', finish)
        audit(root, complete=True)
        return finish


def audit_preflight(root):
    from auditors.suite import trial_evidence, validate_telemetry
    root = Path(root)
    spec, contract, protocol, device, _ = load_run(root)
    require(spec['kind'] == 'preflight', 'preflight audit requires preflight kind')
    if calibrated_cases(contract):
        require('resolved_cases' in spec, 'preflight calibration resolution missing')
    evidence = {}
    for case in contract['cases']:
        folder = root / 'preflight' / case['id'] / 'batch_00/trial_00'
        accepted = [p for p in folder.glob('attempt_*/receipt.json') if read_json(p).get('status') in ('valid', 'warmup_unconverged')]
        require(len(accepted) == 1, 'preflight requires one independent process per case')
        evidence[case['id']] = trial_evidence(root, spec, case, device, protocol, 0, 0, accepted[0].parent)
    recorded = read_json(relative(root, 'preflight_summary.json'))
    require(recorded.get('hardware_qualification') is False and recorded.get('binary_sha256') == spec['binary_sha256'] and set(recorded['cases']) == set(evidence), 'preflight summary binding')
    for key, result in evidence.items():
        for field in ('value', 'warmup_converged', 'warmup_cv', 'unit', 'observed_sms'):
            require(recorded['cases'][key][field] == result[field], 'preflight recomputation differs')
    validate_telemetry(root, device['uuid'])
    require(not (root / 'COMPLETE').exists(), 'preflight must not have COMPLETE')
    return {'schema_version': 2, 'status': 'preflight_evidence_validated_pending_independent_B_review', 'hardware_qualification': False, 'cases': evidence}

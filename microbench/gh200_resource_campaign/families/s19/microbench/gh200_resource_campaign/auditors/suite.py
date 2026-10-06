"""Read-only replay of v2 evidence. No GPU calls and no archive writes."""
from __future__ import annotations
import json
import importlib
import random
import csv
from pathlib import Path
from auditors import memory_baseline
from common.suite_io import digest, gate_mapping, read_json, relative, require, sha, stats, process_ok, validate_gate, verify_files
from common.suite_snapshot import check_snapshot, required_reviews


ADAPTERS = {'tma_bulk_admission_v1': 'auditors.tma_bulk_admission_v1', 'global_duplex_formal_v1': 'auditors.global_duplex_formal_v1', 'async_copy_formal_v1': 'auditors.async_copy_formal_v1', 'memory_baseline_v2': 'auditors.memory_baseline', 'shared_memory_v2': 'auditors.shared_memory', 'low_precision_v2': 'auditors.low_precision',
            'legacy_fma_v2': 'auditors.legacy_compute', 'legacy_mma_v2': 'auditors.legacy_compute', 'legacy_wgmma_v2': 'auditors.legacy_compute'}


def adapter(contract):
    identity = contract.get('adapter_id')
    require(identity in ADAPTERS, 'adapter not implemented: ' + str(identity))
    return importlib.import_module(ADAPTERS[identity])


def parse_raw(path):
    lines = Path(path).read_text().splitlines()
    require(len(lines) == 2, 'raw must contain exactly device + trial rows')
    def pairs(items):
        row = {}
        for key, value in items:
            require(key not in row, 'duplicate raw key')
            row[key] = value
        return row
    return [json.loads(line, object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(ValueError('nonfinite raw'))) for line in lines]


def protocol_check(protocol):
    expected = {'schema_version': 2, 'trials_per_batch': 10, 'max_sampling_batches': 3, 'shuffle_seed': 20261001,
                'warmup_min_windows': 8, 'warmup_max_windows': 30, 'warmup_tail_windows': 5, 'warmup_cv_limit': .02,
                'formal_cv_limit': .05, 'process_timeout_seconds': 120, 'compile_timeout_seconds': 180}
    require(all(protocol.get(k) == v for k, v in expected.items()), 'frozen protocol values changed')


def load_run(run):
    run = Path(run)
    spec = read_json(relative(run, 'run_spec.json'))
    require(spec.get('schema_version') == 2 and spec.get('kind') in ('formal', 'preflight'), 'run schema/kind')
    require(spec.get('fixture') is False, 'CPU fixture cannot acquire GH200 qualification')
    contract = check_snapshot(run, spec)
    expected_reviews = set(required_reviews(contract, preflight=spec['kind'] == 'preflight'))
    require({(r['stage'], r['phase'], r['path']) for r in spec['reviews']} == expected_reviews, 'run review dependency set changed')
    family = adapter(contract)
    family.validate_contract(contract)
    protocol = read_json(relative(run, 'snapshot/repo/microbench/gh200_resource_campaign/contracts/protocol.json'))
    protocol_check(protocol)
    require(sha(relative(run, 'binary/probe')) == spec['binary_sha256'], 'binary changed')
    require(sha(relative(run, 'build/sass.txt')) == spec['sass_sha256'], 'SASS changed')
    require(sha(relative(run, 'build/dependencies.json')) == spec['dependencies_sha256'], 'dependency inventory changed')
    require(sha(relative(run, 'build/shared_libraries.json')) == spec['shared_libraries_sha256'], 'library inventory changed')
    require(sha(relative(run, 'environment/initial.json')) == spec['environment_sha256'], 'environment changed')
    device = read_json(relative(run, 'environment/device.json'))
    require(sha(run / 'environment/device.json') == spec['device_sha256'], 'device changed')
    family.validate_device(device)
    env = read_json(run / 'environment/initial.json')
    require(device['uuid'].lower() == env['uuid'].lower(), 'device/UUID lock identity mismatch')
    sass = family.audit_sass((run / 'build/sass.txt').read_text(), contract)
    require(read_json(relative(run, 'build/sass_audit.json')) == sass, 'SASS audit recomputation changed')
    command = read_json(relative(run, 'build/command.json'))
    argv = command['argv']
    flags = contract['build']['flags']
    require(argv[0] == env['tools']['nvcc']['path'] and argv[1:1+len(flags)] == flags and argv[1+len(flags):3+len(flags)] == ['-MD', '-MF'], 'compile flags/target changed')
    require(command['source_relative'] == spec['source_path'] and command['target_relative'] == 'binary/probe', 'compile source/target binding')
    require(argv[-3].endswith('/' + spec['source_path']) and argv[-2] == '-o' and argv[-1].endswith('/binary/probe'), 'compile source/output command')
    for name in ('compile', 'disassemble', 'ldd'):
        receipt = read_json(relative(run, 'build/' + name + '.receipt.json'))
        require(process_ok(receipt), 'unsuccessful build tool')
        require(receipt['stdout_sha256'] == sha(relative(run, 'build/' + name + '.stdout')) and receipt['stderr_sha256'] == sha(relative(run, 'build/' + name + '.stderr')), 'build tool evidence changed')
    from common.calibration import calibrated_cases, resolved_contract, verify_import_binding
    if calibrated_cases(contract):
        if spec['kind'] == 'formal':
            verify_import_binding(run, spec, contract)
        contract = resolved_contract(run, spec, contract, device, protocol, allow_pending=spec['kind'] == 'preflight')
    from common.family_b3 import verify_build
    contract = verify_build(run, contract, device)
    return spec, contract, protocol, device, family


def trial_evidence(run, spec, case, device, protocol, batch, trial, folder):
    receipt = read_json(relative(folder, 'receipt.json'))
    require(receipt.get('status') in ('valid', 'warmup_unconverged'), 'unsuccessful receipt is not a sample')
    seed = 3 + trial * 19 + batch * 1009
    require(receipt['case_id'] == case['id'] and receipt['batch'] == batch and receipt['trial'] == trial and receipt['seed'] == seed, 'receipt identity')
    require(receipt['relative_command'] == ['binary/probe', case['id'], str(case['iterations']), str(seed)], 'receipt command')
    require(receipt['binary_sha256'] == spec['binary_sha256'], 'receipt binary')
    require(receipt['raw_sha256'] == sha(relative(folder, 'raw.jsonl')), 'raw hash')
    require(receipt['stderr_sha256'] == sha(relative(folder, 'stderr')), 'stderr hash')
    require(type(receipt['pid']) is int and receipt['pid'] > 0 and receipt['pgid'] == receipt['pid'] and receipt['host_stop_ns'] > receipt['host_start_ns'], 'process identity/timestamps')
    require(receipt.get('returncode') == 0 and receipt.get('timed_out') is False and receipt.get('cleanup_confirmed') is True, 'unsuccessful process')
    observed_device, row = parse_raw(folder / 'raw.jsonl')
    require(observed_device == device, 'device drift within trial')
    contract = read_json(relative(run, spec['contract_path']))
    result = adapter(contract).validate_trial(row, case, device, seed, protocol)
    if 'resolved_resource_identity' in case:
        require(all(row.get(k) == v for k,v in case['resolved_resource_identity'].items()), 'kernel resources differ from B-bound calibrated binary')
    require((receipt['status'] == 'valid') == result['warmup_converged'], 'receipt warmup state')
    return {**result, 'case_id': case['id'], 'batch': batch, 'trial': trial, 'receipt': str((folder / 'receipt.json').relative_to(run)), 'raw_sha256': receipt['raw_sha256'], 'pid': receipt['pid'], 'host_start_ns': receipt['host_start_ns'], 'host_stop_ns': receipt['host_stop_ns']}


def case_evidence(run, spec, case, device, protocol):
    root = Path(run) / 'batches' / case['id']
    batches, merged = [], []
    if root.exists():
        require({p.name for p in root.iterdir() if p.is_dir()} <= {f'batch_{i:02d}' for i in range(3)}, 'unexpected sampling batch')
    finished = False
    for index in range(protocol['max_sampling_batches']):
        folder = root / f'batch_{index:02d}'
        if not folder.exists():
            require(not any((root / f'batch_{later:02d}').exists() for later in range(index + 1, 3)), 'batch gap')
            break
        require(not finished, 'unnecessary extra batch after stability accepted')
        require({p.name for p in folder.iterdir() if p.is_dir()} <= {f'trial_{i:02d}' for i in range(10)}, 'unexpected extra trial')
        values, samples, warm_failed = [], [], False
        missing = False
        for trial in range(protocol['trials_per_batch']):
            trial_root = folder / f'trial_{trial:02d}'
            receipts = sorted(trial_root.glob('attempt_*/receipt.json'))
            accepted = []
            for path in receipts:
                row = read_json(path)
                if row.get('status') in ('valid', 'warmup_unconverged'):
                    accepted.append(path)
                else:
                    require(row.get('status') == 'interrupted', 'failed process/implementation cannot become an allowed gap')
            require(len(accepted) <= 1, 'duplicate accepted trial process')
            if not accepted:
                missing = True
                continue
            require(not missing and not warm_failed, 'trial after missing/failed warmup')
            evidence = trial_evidence(Path(run), spec, case, device, protocol, index, trial, accepted[0].parent)
            samples.append(evidence)
            if evidence['warmup_converged']:
                values.append(evidence['value'])
                merged.append(evidence['value'])
            else:
                warm_failed = True
        complete = len(values) == 10
        summary = stats(values) if len(values) >= 2 else None
        merged_stats = stats(merged) if len(merged) >= 2 else None
        stable = complete and summary['cv'] <= protocol['formal_cv_limit'] and merged_stats['cv'] <= protocol['formal_cv_limit']
        done = stable or index == 2 and (complete or warm_failed)
        batches.append({'batch': index, 'complete': complete, 'warmup_failed': warm_failed, 'stats': summary, 'samples': samples})
        if stable:
            finished = True
        elif not complete and not warm_failed:
            break
        elif done:
            finished = True
    terminal = 'stable' if finished and batches[-1]['complete'] and batches[-1]['stats']['cv'] <= .05 and stats(merged)['cv'] <= .05 else 'unstable_after_bounded_remeasurement' if finished else 'pending'
    return {'case_id': case['id'], 'unit': case['metric']['unit'], 'scope': case['scope'], 'exportable': case['exportable'] and terminal == 'stable',
            'status': terminal, 'batches': batches, 'merged': stats(merged) if len(merged) >= 2 else None}


def validate_counter_evidence(run, counter, device):
    state = counter.get('state')
    require(state in ('available', 'permission_denied', 'tool_missing'), 'counter state unresolved or execution failed: ' + str(state))
    require(counter.get('purpose') == 'permission_capability_only' and counter.get('family_profile') is False
            and counter.get('cache_residency_proven') is False and counter.get('physical_hbm_bytes_proven') is False, 'counter capability qualification')
    require(counter.get('fingerprint', {}).get('gpu_uuid', '').lower() == device.get('uuid', '').lower() and device.get('uuid'), 'counter UUID identity')
    mapping = counter.get('evidence_sha256')
    require(isinstance(mapping, dict) and mapping, 'counter evidence hashes missing')
    for name, expected in mapping.items():
        require('/' not in name and '\\' not in name and name not in ('.','..'), 'counter evidence path')
        require(sha(relative(Path(run)/'environment', 'ncu_' + name)) == expected, 'counter evidence changed: ' + name)
    if state == 'tool_missing':
        require(counter.get('unavailable_reason') == 'executable_not_found' and 'tool_lookup.json' in mapping, 'unverified unavailable counter tool')
        lookup = read_json(relative(Path(run)/'environment', 'ncu_tool_lookup.json'))
        require(lookup.get('executable') == 'ncu' and lookup.get('resolved_path') is None and isinstance(lookup.get('PATH'),str), 'tool lookup evidence')
        return
    require({'stdout','stderr','receipt.json'} <= set(mapping), 'counter command evidence incomplete')
    receipt = read_json(relative(Path(run)/'environment', 'ncu_receipt.json'))
    require(receipt.get('cleanup_confirmed') is True and receipt.get('timed_out') is False and type(receipt.get('returncode')) is int and not receipt.get('launch_error'), 'counter execution/cleanup unconfirmed')
    require(receipt.get('stdout_sha256') == mapping['stdout'] and receipt.get('stderr_sha256') == mapping['stderr'], 'counter process output binding')
    registration = receipt.get('gpu_process_registration', {})
    require(registration.get('state') == 'cleanup_confirmed' and registration.get('gpu_uuid') == device['uuid'].lower()
            and registration.get('cleanup_receipt', {}).get('cleanup_confirmed') is True, 'counter UUID registration cleanup evidence')
    text = relative(Path(run)/'environment','ncu_stdout').read_text() + relative(Path(run)/'environment','ncu_stderr').read_text()
    if state == 'available':
        require(receipt['returncode'] == 0, 'available counter check failed')
    else:
        require(receipt['returncode'] != 0 and 'ERR_NVGPUCTRPERM' in text, 'permission denial not evidenced')


def recompute(run, *, require_terminal=False):
    run = Path(run)
    spec, contract, protocol, device, family = load_run(run)
    require(spec['kind'] == 'formal', 'preflight cannot become formal evidence')
    cases = [case_evidence(run, spec, c, device, protocol) for c in contract['cases']]
    require(set(p.name for p in (run / 'batches').iterdir() if p.is_dir()) <= {c['id'] for c in contract['cases']}, 'unexpected case directory')
    # One receipt corresponds to one independently launched process. Same PID reuse later is possible.
    process_ids = [(s['pid'], s['host_start_ns']) for c in cases for b in c['batches'] for s in b['samples']]
    require(len(process_ids) == len(set(process_ids)), 'process receipt reused across samples')
    intervals = sorted((s['host_start_ns'], s['host_stop_ns']) for c in cases for b in c['batches'] for s in b['samples'])
    require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), 'overlapping controlled GPU trials')
    observed_order = [(s['case_id'], s['batch'], s['trial']) for c in cases for b in c['batches'] for s in b['samples']]
    chronological = sorted([s for c in cases for b in c['batches'] for s in b['samples']], key=lambda s:s['host_start_ns'])
    planned = []
    for batch in range(3):
        path = run / 'batches' / f'order_{batch:02d}.json'
        if not path.exists():
            require(not any(b['batch'] == batch for c in cases for b in c['batches']), 'sampling order manifest missing')
            continue
        active = []
        for c in cases:
            prior = c['batches'][:batch]
            if batch == 0:
                active.append(c['case_id'])
            elif len(prior) == batch:
                values = [s['value'] for b in prior for s in b['samples'] if s['warmup_converged']]
                last = prior[-1]
                if last['warmup_failed'] or last['complete'] and (last['stats']['cv'] > .05 or stats(values)['cv'] > .05):
                    active.append(c['case_id'])
        rng = random.Random(protocol['shuffle_seed'] + batch)
        expected_order = []
        for trial in range(10):
            shuffled = sorted(active); rng.shuffle(shuffled)
            expected_order.extend([[key, trial] for key in shuffled])
        require(read_json(path) == expected_order, 'sampling shuffle/order changed')
        planned.extend((key, batch, trial) for key, trial in expected_order if (key, batch, trial) in observed_order)
    require([(s['case_id'],s['batch'],s['trial']) for s in chronological] == planned, 'process order differs from frozen schedule')
    terminal = all(c['status'] != 'pending' for c in cases)
    if require_terminal:
        require(terminal, 'missing samples/unfinished batches')
    ncu_path = run / 'environment/ncu_status.json'
    counters = read_json(ncu_path) if ncu_path.exists() else {'state': 'not_checked'}
    if require_terminal:
        validate_counter_evidence(run, counters, device)
    return {'schema_version': 2, 'family': contract['family'], 'stage': contract['stage'], 'qualification': 'measurement_evidence_pending_C_review', 'terminal': terminal,
            'device': device, 'cases': cases, 'counters': counters, 'cache_residency_proven': False, 'physical_hbm_bytes_proven': False,
            'snapshot_manifest_sha256': spec['snapshot_manifest_sha256'], 'binary_sha256': spec['binary_sha256']}


def audit(run, *, complete=False):
    run = Path(run)
    summary = recompute(run, require_terminal=True)
    require(read_json(relative(run, 'summary.json')) == summary, 'summary differs from raw recomputation')
    validate_telemetry(run, summary['device']['uuid'])
    manifest = read_json(relative(run, 'measurement_manifest.json'))
    verify_files(run, manifest)
    expected_paths = {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p.relative_to(run).as_posix() not in ('measurement_manifest.json', 'campaign_status.json', 'COMPLETE') and not p.relative_to(run).as_posix().startswith('reviews/C/')}
    require(set(manifest) == expected_paths, 'measurement manifest coverage changed')
    if complete:
        finish = read_json(relative(run, 'COMPLETE'))
        require(finish == {'schema_version': 2, 'summary_sha256': sha(run / 'summary.json'), 'measurement_manifest_sha256': sha(run / 'measurement_manifest.json'), 'C_review_sha256': sha(relative(run, 'reviews/C/review.json'))}, 'COMPLETE binding')
        review = validate_gate(run / 'reviews/C/review.json', run, summary['stage'], 'C')
        require({'measurement_manifest.json', 'summary.json'} <= set(gate_mapping(review)), 'C gate missing measurement manifest/summary')
        require(read_json(run / 'campaign_status.json')['status'] == 'complete', 'completion status mismatch')
    return summary


def validate_telemetry(run, uuid):
    samples = 0
    for path in (Path(run) / 'environment').glob('telemetry_*/receipt.json'):
        receipt = read_json(path)
        require(receipt['cleanup_confirmed'] is True, 'telemetry cleanup unconfirmed')
        data = relative(path.parent, 'samples.csv')
        require(sha(data) == receipt['samples_sha256'] and sha(relative(path.parent, 'stderr')) == receipt['stderr_sha256'], 'telemetry evidence changed')
        rows = list(csv.reader(data.read_text().splitlines()))
        if len(rows) > 1:
            require(len(rows[0]) >= 2 and rows[0][1].strip() == 'uuid', 'telemetry header')
            for row in rows[1:]:
                require(len(row) == len(rows[0]) and row[1].strip().lower() == uuid.lower(), 'telemetry UUID/columns')
                samples += 1
    require(samples > 0, 'no GPU telemetry samples')


def report(summary):
    lines = ['# GH200 资源测量', '', '状态：' + summary['qualification'], '', '| 案例 | 状态 | 单位 | 样本数 | 中位数 | CV |', '|---|---|---|---:|---:|---:|']
    for case in summary['cases']:
        sample = case['merged']
        lines.append(f"| {case['case_id']} | {case['status']} | {case['unit']} | {sample['samples'] if sample else 0} | {sample['median'] if sample else 'NA'} | {sample['cv'] if sample else 'NA'} |")
    lines += ['', '统计保留所有同协议有效正式样本，不选择最好批次。请求字节不代表物理 HBM 流量，空窗口不用于扣除有效载荷时间。', '', 'NCU：' + summary['counters']['state'], '']
    return '\n'.join(lines)

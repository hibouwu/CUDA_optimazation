"""B-stage pilot resolution and hash-bound reuse; never estimates from old results."""
from __future__ import annotations
import copy
from pathlib import Path
import shutil
from common.suite_io import atomic_json, gate_mapping, read_json, relative, require, sha


def calibrated_cases(contract):
    return [c for c in contract['cases'] if c.get('iteration_policy', {}).get('kind') == 'calibrated']


def pilot_case(case):
    case = copy.deepcopy(case)
    case['iterations'] = case['iteration_policy']['pilot_iterations']
    return case


def accepted_receipt(root, case_id, directory):
    folder = Path(root) / directory / case_id / 'batch_00/trial_00'
    paths = [p for p in folder.glob('attempt_*/receipt.json') if read_json(p).get('status') in ('valid', 'warmup_unconverged')]
    require(len(paths) == 1, 'one accepted calibration process required: ' + case_id)
    return paths[0]


def resolution_entry(root, spec, contract, case, device, protocol):
    from auditors.suite import adapter, parse_raw, trial_evidence
    root = Path(root)
    receipt_path = accepted_receipt(root, case['id'], 'calibration')
    receipt = read_json(receipt_path)
    trial_evidence(root, spec, pilot_case(case), device, protocol, 0, 0, receipt_path.parent)
    raw_path = receipt_path.parent / 'raw.jsonl'
    _, row = parse_raw(raw_path)
    result = adapter(contract).resolve_iterations(row, case, device)
    policy = case['iteration_policy']
    require(type(result.get('resolved_iterations')) is int and policy['min_iterations'] <= result['resolved_iterations'] <= policy['max_iterations'], 'resolver iteration range')
    require(result['pilot_iterations'] == policy['pilot_iterations'] and result['pilot_event_ms'] == row['event_ms'] and result['rounding_model'] == policy['rounding_model'], 'resolver pilot identity')
    return {**result, 'case_id': case['id'], 'binary_sha256': spec['binary_sha256'], 'contract_sha256': sha(relative(root, spec['contract_path'])),
            'device_identity': device, 'pilot_raw': {'path': raw_path.relative_to(root).as_posix(), 'sha256': sha(raw_path)},
            'pilot_receipt': {'path': receipt_path.relative_to(root).as_posix(), 'sha256': sha(receipt_path)},
            'resource_identity': {name: row[name] for name in ('registers_per_thread', 'occupancy_limit_ctas_per_sm', 'static_smem_bytes', 'local_size_bytes')},
            'pilot_warmup_converged': row['warmup_converged']}


def seal_resolution(root, spec, contract, device, protocol):
    entries = [resolution_entry(root, spec, contract, case, device, protocol) for case in calibrated_cases(contract)]
    value = {'schema_version': 2, 'family': contract['family'], 'phase': 'B-calibration', 'cases': entries}
    path = Path(root) / 'resolved_cases.json'
    if path.exists():
        require(read_json(path) == value, 'refuse to change previously resolved cases')
    else:
        atomic_json(path, value)
    spec['resolved_cases'] = {'path': 'resolved_cases.json', 'sha256': sha(path)}
    atomic_json(Path(root) / 'run_spec.json', spec)
    return value


def resolved_contract(root, spec, contract, device, protocol, *, allow_pending=False):
    """Pure verification/reconstruction; never writes or substitutes pilot placeholders."""
    root = Path(root)
    targets = calibrated_cases(contract)
    if not targets:
        require('resolved_cases' not in spec, 'unexpected resolution for fixed-only family')
        return contract
    binding = spec.get('resolved_cases')
    if not binding:
        require(allow_pending and spec['kind'] == 'preflight', 'calibrated formal run requires B-bound resolved cases')
        return contract
    path = relative(root, binding['path'])
    require(sha(path) == binding['sha256'], 'resolved cases hash changed')
    value = read_json(path)
    require(value.get('schema_version') == 2 and value.get('family') == contract['family'] and value.get('phase') == 'B-calibration', 'resolution schema/family/phase')
    ids = [row.get('case_id') for row in value['cases']]
    require(len(ids) == len(set(ids)) and set(ids) == {c['id'] for c in targets}, 'resolved case coverage/duplicates')
    entries = {row['case_id']: row for row in value['cases']}
    for case in targets:
        row = entries[case['id']]
        for artifact in ('pilot_raw', 'pilot_receipt'):
            require(sha(relative(root, row[artifact]['path'])) == row[artifact]['sha256'], 'pilot evidence changed')
        require(row == resolution_entry(root, spec, contract, case, device, protocol), 'resolution differs from independently recomputed pilot')
    result = copy.deepcopy(contract)
    for case in result['cases']:
        if case['id'] in entries:
            entry = entries[case['id']]
            case['iterations'] = entry['resolved_iterations']
            case['resolved_resource_identity'] = entry['resource_identity']
    from auditors.suite import adapter
    adapter(result).validate_contract(result)
    return result


def require_B_bound_artifacts(repo, origin, review, files):
    repo, origin = Path(repo).resolve(), Path(origin).resolve()
    require(origin.is_relative_to(repo), 'calibration origin must have repository-relative review identity')
    gate = gate_mapping(review)
    mapping = {}
    for name in files:
        path = relative(origin, name)
        key = path.relative_to(repo).as_posix()
        require(key in gate and gate[key] == sha(path), 'family B must directly bind calibration artifact: ' + key)
        mapping[name] = gate[key]
    return mapping


def import_resolved_binary(repo, root, spec, contract, resolved_path, env):
    """Copy exact B-approved executable and evidence; this path never invokes nvcc."""
    from auditors.suite import load_run
    from runners.environment import environment_identity
    root = Path(root)
    resolved_path = Path(resolved_path).resolve()
    require(resolved_path.name == 'resolved_cases.json', 'expected explicit resolved_cases.json')
    origin = resolved_path.parent
    old_spec, effective, protocol, device, _ = load_run(origin)
    from runners.suite_runner import audit_preflight
    audit_preflight(origin)
    require(old_spec['kind'] == 'preflight', 'calibration origin must be B preflight')
    require(sha(resolved_path) == old_spec['resolved_cases']['sha256'], 'origin resolution binding')
    require(read_json(relative(origin, old_spec['contract_path'])) == contract, 'calibration family contract changed')
    require(environment_identity(read_json(origin / 'environment/initial.json')) == environment_identity(env), 'calibration device/toolchain environment changed')
    # New framework files may evolve, but every compiled local dependency must be identical.
    dependencies = read_json(relative(origin, 'build/dependencies.json'))
    old_snapshot = read_json(relative(origin, 'snapshot/manifest.json'))
    new_snapshot = read_json(relative(root, 'snapshot/manifest.json'))
    for name in dependencies['local']:
        require(name in old_snapshot and name in new_snapshot and old_snapshot[name] == new_snapshot[name], 'calibration compiled source dependency changed: ' + name)
    b_review = next(r for r in spec['reviews'] if r['stage'] == contract['stage'] and r['phase'] == 'B')
    review = read_json(relative(root / 'snapshot/suite', b_review['path']))
    required = ['resolved_cases.json', 'binary/probe', 'build/sass.txt', 'build/dependencies.json', 'run_spec.json', 'preflight_summary.json', 'snapshot/manifest.json', 'environment/initial.json', 'environment/device.json']
    required += [p.relative_to(origin).as_posix() for p in (origin / 'build').rglob('*') if p.is_file()]
    required += [p.relative_to(origin).as_posix() for p in (origin / 'calibration').rglob('*') if p.is_file()]
    # B smoke at resolved length must also be explicitly bound, not only the pilot.
    required += [p.relative_to(origin).as_posix() for p in (origin / 'preflight').rglob('*') if p.is_file()]
    binding = require_B_bound_artifacts(repo, origin, review, sorted(set(required)))
    for name in ('binary/probe', 'resolved_cases.json'):
        destination = root / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative(origin, name), destination)
    for folder in ('build', 'calibration'):
        shutil.copytree(origin / folder, root / folder, dirs_exist_ok=True)
    # Preserve B smoke evidence under a distinct namespace; it never counts as formal samples.
    shutil.copytree(origin / 'preflight', root / 'calibration_origin/preflight')
    for name in ('run_spec.json', 'preflight_summary.json', 'snapshot/manifest.json', 'environment/initial.json', 'environment/device.json'):
        destination = root / 'calibration_origin' / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin / name, destination)
    atomic_json(root / 'calibration_origin/B_artifact_binding.json', {'schema_version': 2, 'origin_repository_path': origin.relative_to(Path(repo).resolve()).as_posix(), 'files': binding, 'review_sha256': b_review['sha256'], 'compiled_dependency_hashes': {name:old_snapshot[name] for name in dependencies['local']}, 'binary_recompiled': False})
    for key in ('binary_sha256', 'sass_sha256', 'dependencies_sha256', 'shared_libraries_sha256', 'resolved_cases'):
        spec[key] = old_spec[key]
    spec['calibration_import'] = {'path': 'calibration_origin/B_artifact_binding.json', 'sha256': sha(root / 'calibration_origin/B_artifact_binding.json')}
    atomic_json(root / 'run_spec.json', spec)
    return device


def verify_import_binding(root, spec, contract):
    require('calibration_import' in spec, 'formal calibrated binary must be B-bound reused artifact')
    record = spec['calibration_import']; path = relative(root, record['path'])
    require(sha(path) == record['sha256'], 'calibration import binding changed')
    binding = read_json(path)
    require(binding['binary_recompiled'] is False, 'calibration binary was rebuilt')
    b_review = next(r for r in spec['reviews'] if r['stage'] == contract['stage'] and r['phase'] == 'B')
    require(binding['review_sha256'] == b_review['sha256'], 'calibration B review mismatch')
    gates = gate_mapping(read_json(relative(Path(root) / 'snapshot/suite', b_review['path'])))
    for name, expected in binding['files'].items():
        key = binding['origin_repository_path'] + '/' + name
        require(gates.get(key) == expected, 'calibration artifact not bound by frozen B review')
        destination = 'calibration_origin/' + name if name.startswith(('preflight/', 'snapshot/', 'environment/')) or name in ('run_spec.json', 'preflight_summary.json') else name
        require(sha(relative(root, destination)) == expected, 'imported B artifact changed: ' + name)
    required = {'binary/probe', 'resolved_cases.json', 'build/sass.txt', 'build/dependencies.json', 'run_spec.json', 'preflight_summary.json', 'snapshot/manifest.json', 'environment/initial.json', 'environment/device.json'}
    require(required <= set(binding['files']), 'missing imported calibration identity evidence')
    require(binding['files'].get('binary/probe') == spec['binary_sha256'], 'imported binary identity mismatch')
    origin_spec = read_json(relative(root, 'calibration_origin/run_spec.json'))
    old_manifest = relative(root, 'calibration_origin/snapshot/manifest.json')
    require(sha(old_manifest) == origin_spec['snapshot_manifest_sha256'], 'origin dependency manifest binding')
    old_sources = read_json(old_manifest)
    new_sources = read_json(relative(root, 'snapshot/manifest.json'))
    require(binding['compiled_dependency_hashes'], 'empty calibration source dependency closure')
    require(set(binding['compiled_dependency_hashes']) == set(read_json(relative(root, 'build/dependencies.json'))['local']), 'calibration source dependency closure omitted files')
    for folder in ('build', 'calibration'):
        require({p.relative_to(root).as_posix() for p in (Path(root)/folder).rglob('*') if p.is_file()} <= set(binding['files']), 'unbound imported build/pilot evidence')
    require({p.relative_to(Path(root)/'calibration_origin').as_posix() for p in (Path(root)/'calibration_origin/preflight').rglob('*') if p.is_file()} <= set(binding['files']), 'unbound B smoke evidence')
    for name, expected in binding['compiled_dependency_hashes'].items():
        require(old_sources.get(name) == expected and new_sources.get(name) == expected, 'reused binary source dependency mismatch')
    value = read_json(relative(root, spec['resolved_cases']['path']))
    for entry in value['cases']:
        for field in ('pilot_raw', 'pilot_receipt'):
            require(binding['files'].get(entry[field]['path']) == entry[field]['sha256'], 'pilot not directly bound by B import')

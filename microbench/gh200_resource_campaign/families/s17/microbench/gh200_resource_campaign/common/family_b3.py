"""Read-only admission for the finite S12-v1 and S13-v2 execution revisions.

All paths resolve inside the supplied repository (live only before freezing,
then the run's snapshot). Occupancy is reused from reviewed B3 evidence after
identical source, machine code, static resources and environment are proved.
No GPU call, process launch, or new occupancy-query claim is made here.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import re

from common.suite_io import gate_mapping, read_json, relative, require, sha, validate_gate, verify_files
from auditors.async_copy import audit_sass, instructions, kernel_symbol
from auditors.async_copy_validation import validate_profile, validate_validation

REVISION = 'family-b3-v1'
ADAPTER = 'async_copy_formal_v1'
S13_ADAPTER = 'global_duplex_formal_v1'
S13_REVISION = 'family-b3-v2'
S14_REVISION = 'family-b3-s14-v1'
S14_ADAPTER = 'tma_bulk_admission_v1'
RESOURCE_FIELDS = ('registers_per_thread', 'static_smem_bytes', 'dynamic_smem_bytes',
                   'local_size_bytes', 'occupancy_limit_ctas_per_sm')
ENVIRONMENT_FIELDS = ('uuid', 'name', 'driver', 'compiler', 'tools', 'execution_uid')


def enabled(contract):
    revision = contract.get('execution_revision')
    if revision is None:
        require(contract.get('adapter_id') not in (ADAPTER, S13_ADAPTER, S14_ADAPTER) and 'family_b3' not in contract,
                'family B3 revision missing')
        return False
    legacy = (revision == REVISION and contract.get('adapter_id') == ADAPTER
              and contract.get('stage') == 'S12')
    s13 = (revision == S13_REVISION and contract.get('adapter_id') == S13_ADAPTER
           and contract.get('stage') == 'S13' and contract.get('family') == 'global_duplex')
    s14 = (revision == S14_REVISION and contract.get('adapter_id') == S14_ADAPTER
           and contract.get('stage') == 'S14' and contract.get('family') == 'tma_bulk')
    require(legacy or s13 or s14, 'unknown execution revision')
    return True


def policy(contract):
    """Two code-owned choices; contract data never names executable modules."""
    require(enabled(contract), 'B3 policy requires explicit revision')
    if contract['execution_revision'] == S14_REVISION:
        return {'stage': 'S14', 'family': 'tma_bulk', 'case_count': 24, 'launches': 72,
                'b3_review_phase': 'formal-packed-bridge-B', 'complete_target_identity': True}
    if contract['execution_revision'] == REVISION:
        return {'stage':'S12', 'family':'async_copy', 'case_count':24,
                'launches':48, 'elements':65680640, 'profile':validate_profile,
                'validation':validate_validation, 'sass':audit_sass, 'symbol':kernel_symbol,
                'complete_target_identity':False}
    from auditors import global_duplex, global_duplex_validation
    def checked_profile(profile):
        global_duplex_validation.profile_check(profile)
        return profile
    return {'stage':'S13', 'family':'global_duplex', 'case_count':18,
            'launches':36, 'elements':950501376, 'profile':checked_profile,
            'validation':global_duplex_validation.validate_validation,
            'sass':global_duplex.audit_sass, 'symbol':global_duplex.kernel_symbol,
            'complete_target_identity':True}


def target_identity(text, symbols):
    """Keep actual instruction PCs, predicates, operands and both encoding words."""
    functions = re.split(r'Function\s*:\s*', text)[1:]
    result = {}
    for symbol in sorted(symbols):
        matches = [part for part in functions if part.splitlines()[0].strip() == symbol]
        require(len(matches) == 1, 'missing/duplicate B3 machine-code target: '+symbol)
        rows = []
        for line in matches[0].splitlines()[1:]:
            head = re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?);', line)
            if head:
                rows.append({'pc':int(head[1],16), 'instruction':' '.join(head[2].split()),
                             'encoding':re.findall(r'/\* (0x[0-9a-fA-F]{16}) \*/', line)})
            elif rows:
                rows[-1]['encoding'].extend(re.findall(r'/\* (0x[0-9a-fA-F]{16}) \*/', line))
        require(rows and all(len(row['encoding']) == 2 for row in rows),
                'complete 128-bit target encodings required: '+symbol)
        result[symbol] = rows
    return result


def bound_file(repo, reference):
    require(set(reference) == {'path', 'sha256'}, 'B3 reference fields')
    path = relative(repo, reference['path'])
    require(sha(path) == reference['sha256'], 'B3 bound artifact changed: ' + reference['path'])
    return path


def unique_cases(rows, field):
    require(isinstance(rows, list) and rows, 'B3 cases missing')
    result = {}
    for row in rows:
        key = row[field]
        require(key not in result, 'duplicate B3 case: ' + key)
        result[key] = row
    return result


def environment_identity(env):
    return {key: env[key] for key in ENVIRONMENT_FIELDS}


def raw_rows(path):
    lines = path.read_text().splitlines()
    require(len(lines) == 2, 'B3 raw must contain device and validation')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate B3 raw field')
            result[key] = value
        return result
    return [json.loads(line, object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite B3 raw')))
            for line in lines]


def compile_resources(text, symbols):
    resources = {}
    for symbol in symbols:
        marker = "Compiling entry function '" + symbol + "'"
        require(text.count(marker) == 1, 'missing/duplicate target compile resource: ' + symbol)
        body = text.split(marker, 1)[1].split('Compiling entry function', 1)[0]
        require('0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in body,
                'target stack/spill changed')
        used = re.findall(r'Used (\d+) registers, used (\d+) barriers(?:, (\d+) bytes smem)?', body)
        require(len(used) == 1, 'unrecognized target compile resources')
        registers, barriers, shared = used[0]
        resources[symbol] = {'registers_per_thread': int(registers),
                             'static_smem_bytes': int(shared or 0), 'barriers': int(barriers)}
    return resources


def validate_bundle(repo, contract):
    """Verify B3 qualification and the exact finite transitive archives; return freeze names."""
    if not enabled(contract):
        return None
    if contract['execution_revision'] == S14_REVISION:
        from common.s14_admission_v1 import validate_bundle as s14_bundle
        return s14_bundle(repo, contract)
    selected = policy(contract)
    stage, family = selected['stage'], selected['family']
    repo = Path(repo).resolve()
    refs = contract.get('family_b3', {})
    require(set(refs) == {'parent_contract', 'profiles', 'review', 'coverage'}, 'B3 binding set')
    paths = {key: bound_file(repo, ref) for key, ref in refs.items()}
    parent = read_json(paths['parent_contract'])
    # Only admission metadata may change. In particular, cases/source/build are exact.
    metadata = {'adapter_id', 'execution_revision', 'family_b3', 'review_dependencies', 'status'}
    require({k:v for k,v in contract.items() if k not in metadata}
            == {k:v for k,v in parent.items() if k not in metadata}, 'formal contract changed original family semantics')
    cases = unique_cases(parent['cases'], 'id')
    require(len(cases) == selected['case_count'], 'finite B3 case count')
    review = validate_gate(paths['review'], repo, stage, 'early-validation-B3')
    # Compound implementer strings must not hide a self-review identity.
    authors = set(re.findall(r'/root(?:/[a-zA-Z0-9_]+)?', review['implementer']))
    require(review['reviewer'] not in authors, 'B3 review is not independent')
    require(review.get('authorization', {}).get('family_B3_qualified') is True, 'B3 purpose not qualified')
    coverage = read_json(paths['coverage'])
    require(review.get('coverage_record') == refs['coverage'], 'B3 coverage not directly bound')
    gates = gate_mapping(review)
    require(gates.get(refs['coverage']['path']) == refs['coverage']['sha256'], 'B3 coverage absent from gates')
    require(coverage.get('status') == 'pass' and coverage.get('family_B3_qualified') is True
            and coverage.get('stage') == stage and coverage.get('family') == family, 'B3 coverage identity/status')
    require(coverage['contract_sha256'] == refs['parent_contract']['sha256']
            and coverage['profiles_sha256'] == refs['profiles']['sha256'], 'B3 contract/profile drift')
    require(sha(relative(repo, parent['source'])) == coverage['source_sha256'], 'B3 source drift')
    profiles = read_json(paths['profiles'])
    require(profiles['family'] == family and len(profiles['profiles']) == 1, 'B3 profile bundle')
    profile = selected['profile'](profiles['profiles'][0])
    reference = profile['reference_identity']
    require(sha(relative(repo, reference['path'])) == reference['sha256'] == coverage['reference_sha256'],
            'B3 numerical reference drift')
    covered = unique_cases(coverage['cases'], 'case_id')
    require(set(covered) == set(cases) and coverage['case_count'] == selected['case_count'], 'B3 incomplete/extra case coverage')
    require(coverage['profile_id'] == profile['id'] and coverage['seed'] == 3
            and coverage['iterations_per_case'] == [1, 2], 'B3 fixed profile/seed')
    suite = paths['review'].parent.parent
    names = set(gates) | {ref['path'] for ref in refs.values()}
    records = {}
    total_elements = total_launches = 0
    common = None
    targets = {}
    read_elements = destination_elements = 0
    for case_id, case in cases.items():
        item = covered[case_id]
        require(item['run'] == family + '/short-v1-' + case_id, 'B3 case archive path')
        root = suite / item['run']
        manifest_path = relative(root, 'validation_manifest.json')
        require(sha(manifest_path) == item['validation_manifest_sha256'], 'B3 archive manifest drift')
        manifest = read_json(manifest_path)
        verify_files(root, manifest)
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
                  and p.name not in ('validation_manifest.json', 'validation_state.json')}
        require(actual == set(manifest), 'B3 transitive archive inventory mismatch')
        for name in [*manifest, 'validation_manifest.json']:
            names.add((root / name).relative_to(repo).as_posix())
        spec = read_json(relative(root, 'validation_spec.json'))
        require(sha(root / 'validation_spec.json') == item['validation_spec_sha256'], 'B3 case spec drift')
        require(spec['case_id'] == case_id and spec['seed'] == item['seed'] == 3
                and spec['profile_id'] == item['profile_id'] == profile['id'], 'B3 case identity')
        require(read_json(relative(root, spec['contract_path'])) == parent
                and sha(relative(root, spec['profiles_path'])) == refs['profiles']['sha256'], 'B3 case contract/profile drift')
        require(len(list((root/'attempts').glob('attempt_*'))) == 1, 'B3 attempt multiplicity')
        folder = root/'attempts/attempt_00'
        require(sha(relative(folder, 'raw.jsonl')) == item['raw_sha256']
                and sha(relative(folder, 'receipt.json')) == item['receipt_sha256'], 'B3 raw/receipt drift')
        device, row = raw_rows(folder/'raw.jsonl')
        result = selected['validation'](device, row, case, profile, 3)
        require(row['resource_identity'] == item['resource_identity'], 'B3 resource record drift')
        require(row['blocks'] == item['blocks'], 'B3 grid record drift')
        require(result['checked_elements'] == item['checked_elements']
                and item['target_launches'] == 2, 'B3 output/launch count')
        receipt = read_json(folder/'receipt.json')
        require(receipt['returncode'] == 0 and receipt['cleanup_confirmed'] is True
                and receipt['timed_out'] is False and receipt['timeout_seconds'] == 30, 'failed B3 execution')
        require(receipt['stdout_sha256'] == item['raw_sha256']
                and receipt['stderr_sha256'] == sha(relative(folder, 'stderr')), 'B3 process stream hash')
        require(receipt['argv'] == [spec['execution_root']+'/binary/probe', 'validate-only', case_id, profile['id'], '3'],
                'B3 process command')
        registration = receipt['gpu_process_registration']
        require(registration['state'] == 'cleanup_confirmed' and registration['gpu_uuid'] == device['uuid'].lower()
                and registration['pid'] == receipt['pid'] == receipt['pgid'] == registration['pgid'], 'B3 process identity/cleanup')
        require(registration['cleanup_receipt'] == {k:v for k,v in receipt.items() if k != 'gpu_process_registration'},
                'B3 cleanup receipt drift')
        inventory = read_json(relative(root, 'build/dependencies.json'))
        # Prove original device and host sources, including every compiler-local include.
        for name in inventory['local']:
            require(name.startswith('repo/'), 'B3 local dependency root')
            live_name = name.removeprefix('repo/')
            require(sha(relative(repo, live_name)) == sha(relative(root, 'snapshot/'+name)), 'B3 compiled host/header drift')
            names.add(live_name)
        environment = read_json(relative(root, 'environment/initial.json'))
        libraries = read_json(relative(root, 'build/shared_libraries.json'))
        identity = {'device':device, 'environment':environment_identity(environment),
                    'dependencies':inventory, 'libraries':libraries}
        if common is None:
            common = identity
        require(identity == common, 'B3 environment/dependency disagreement across cases')
        require(device == coverage['device_identity'], 'B3 coverage device drift')
        require(sha(relative(root,'binary/probe')) == item['binary_sha256'] == spec['binary_sha256']
                and sha(relative(root,'build/sass.stdout')) == item['sass_sha256'] == spec['sass_sha256'], 'B3 compiled identity')
        sass_text = (root/'build/sass.stdout').read_text()
        selected['sass'](sass_text, parent)
        symbol = selected['symbol'](case)
        resources = compile_resources((root/'build/compile.stderr').read_text(), [symbol])
        static = resources[symbol]
        if selected['complete_target_identity']:
            current = target_identity(sass_text, [symbol])[symbol]
            require(symbol not in targets or targets[symbol] == current, 'B3 target disagreement across cases')
            targets[symbol] = current
            p = case['parameters']
            reads = 2*row['blocks']*256 if p['read_requests_per_group'] else 0
            writes = 2*row['resource_identity']['extensions']['array_bytes']//4 if p['write_requests_per_group'] else 0
            require(item['read_checksum_elements'] == reads and item['destination_word_elements'] == writes
                    and item['checked_elements'] == reads+writes
                    and item['pure_write_zero_checksums_counted'] is False, 'S13 B3 output decomposition')
            read_elements += reads
            destination_elements += writes
        require(static['registers_per_thread'] == row['resource_identity']['registers_per_thread']
                and static['static_smem_bytes'] == row['resource_identity']['static_smem_bytes']
                and static['barriers'] == 1, 'B3 static resources')
        records[case_id] = {'resource':item['resource_identity'], 'blocks':item['blocks'], 'root':root}
        total_elements += item['checked_elements']
        total_launches += 2
    require(total_elements == coverage['checked_elements'] == selected['elements']
            and total_launches == coverage['target_launches'] == selected['launches'], 'B3 aggregate mismatch')
    if selected['complete_target_identity']:
        require(read_elements == coverage['read_checksum_elements'] == 4325376
                and destination_elements == coverage['destination_word_elements'] == 946176000
                and coverage['pure_write_zero_checksums_counted'] is False
                and len(targets) == 9, 'S13 exact full family decomposition/targets')
    return {'target_identities':targets, 'names':names, 'cases':records, 'identity':common, 'parent':parent,
            'review_sha256':refs['review']['sha256'], 'coverage_sha256':refs['coverage']['sha256']}


def verify_build(run, contract, device, *, live_environment=None):
    """Called by offline load and before new pilot/warmup. Returns enriched cases."""
    if not enabled(contract):
        return contract
    if contract['execution_revision'] == S14_REVISION:
        from common.s14_admission_v1 import verify_build as s14_build
        return s14_build(run, contract, device, live_environment=live_environment)
    selected = policy(contract)
    run = Path(run)
    bundle = validate_bundle(run/'snapshot/repo', contract)
    expected = bundle['identity']
    require(device == expected['device'], 'formal device differs from B3')
    environment = read_json(relative(run, 'environment/initial.json'))
    require(environment_identity(environment) == expected['environment'], 'formal toolchain/environment differs from B3')
    if live_environment is not None:
        require(environment_identity(live_environment) == expected['environment'], 'live toolchain/environment differs from B3')
    inventory = read_json(relative(run, 'build/dependencies.json'))
    require(inventory == expected['dependencies'], 'compiled dependency inventory differs from B3')
    libraries = read_json(relative(run, 'build/shared_libraries.json'))
    require(libraries == expected['libraries'], 'runtime library inventory differs from B3')
    if live_environment is not None:
        for name, digest in {**inventory['external_toolchain'], **libraries}.items():
            require(Path(name).is_file() and sha(name) == digest, 'live toolchain/runtime file drift: '+name)
    sass_text = relative(run, 'build/sass.txt').read_text()
    selected['sass'](sass_text, contract)
    symbols = {selected['symbol'](c) for c in contract['cases']}
    if selected['complete_target_identity']:
        require(target_identity(sass_text, symbols) == bundle['target_identities'],
                'formal complete target machine code differs from B3')
    resources = compile_resources(relative(run, 'build/compile.stderr').read_text(), symbols)
    result = deepcopy(contract)
    for case in result['cases']:
        record = bundle['cases'][case['id']]
        static = resources[selected['symbol'](case)]
        require(static['registers_per_thread'] == record['resource']['registers_per_thread']
                and static['static_smem_bytes'] == record['resource']['static_smem_bytes']
                and static['barriers'] == 1, 'formal static resources differ from B3')
        case['b3_resource_identity'] = record['resource']
        case['b3_blocks'] = record['blocks']
    return result

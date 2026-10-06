"""Finite S14 loading of signed packed history; no expanded short runs needed.

This consumes independently signed physical bridges and derived facts. It
never calls the construction-time live-tree function or executes packed code.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

from common.packed_evidence import PackedEvidence, json_object
from common.suite_io import gate_mapping, read_json, relative, require, sha, validate_gate
from common.s14_formal_packed_bridge import REVIEW, REVIEW_SHA, COVERAGE, COVERAGE_SHA

REVISION = 'family-b3-s14-v1'
ADAPTER = 'tma_bulk_admission_v1'
BRIDGE_SHA = '187fbe7a60196ba984196678f595b7916182054c2d7f169aca72c26163954c8c'
ORIGINAL_BRIDGE_SHA = '84ae425dccbf3bbc53b9901277400a489c98b7e8e5b70d2baf258dc8befb8432'
ENV_FIELDS = ('driver', 'compiler', 'execution_uid', 'tools')


def applies(contract):
    return (contract.get('execution_revision') == REVISION and contract.get('adapter_id') == ADAPTER
            and contract.get('stage') == 'S14' and contract.get('family') == 'tma_bulk')


def bound(repo, ref):
    require(set(ref) == {'path', 'sha256'}, 'S14 admission reference fields')
    path = relative(repo, ref['path'])
    require(sha(path) == ref['sha256'], 'S14 admission reference changed: ' + ref['path'])
    return path


def validate_bundle(repo, contract):
    from common.family_b3 import compile_resources
    from auditors import tma_bulk, tma_bulk_formal_validation_v1 as short
    require(applies(contract), 'unknown S14 admission revision')
    repo = Path(repo).resolve()
    refs = contract.get('family_b3', {})
    require(set(refs) == {'parent_contract', 'profiles', 'review', 'coverage', 'original_transport'},
            'S14 exact two-bridge binding set')
    paths = {key: bound(repo, value) for key, value in refs.items()}
    require(refs['review']['sha256'] == BRIDGE_SHA
            and refs['original_transport']['sha256'] == ORIGINAL_BRIDGE_SHA,
            'S14 finite independent bridge identities')
    bridge = validate_gate(paths['review'], repo, 'S14', 'formal-packed-bridge-B')
    legacy = validate_gate(paths['original_transport'], repo, 'S14', 'packed-evidence-bridge-B')
    require(bridge.get('bridge_kind') == 's14_capture_false72_packed_history'
            and legacy.get('bridge_kind') == 's14_original84_packed_history'
            and bridge['authorization']['historical_transport_bridge'] is True
            and legacy['authorization']['historical_transport_bridge'] is True,
            'two distinct S14 qualified transport purposes')
    require(bridge['derived_facts'] == refs['coverage'], 'S14 directly signed derived facts')
    facts = read_json(paths['coverage'])
    require(facts['parent_review'] == {'path': REVIEW, 'sha256': REVIEW_SHA}
            and facts['parent_coverage'] == {'path': COVERAGE, 'sha256': COVERAGE_SHA},
            'capture=false independent numeric parent identity')
    gates = gate_mapping(bridge)
    require(gates.get(REVIEW) == REVIEW_SHA and gates.get(COVERAGE) == COVERAGE_SHA,
            'independent numeric parent metadata must be frozen')
    require(bridge['historical_parent']['case_count'] == 24
            and bridge['historical_parent']['target_launches'] == 72
            and bridge['historical_parent']['capture_enabled'] is False,
            'new branch complete finite scope')
    parent = read_json(paths['parent_contract'])
    metadata = {'adapter_id', 'execution_revision', 'family_b3', 'review_dependencies', 'status'}
    require({k:v for k,v in contract.items() if k not in metadata}
            == {k:v for k,v in parent.items() if k not in metadata},
            'S14 admission changed frozen family semantics')
    tma_bulk.validate_contract(parent)
    profiles = read_json(paths['profiles'])
    require(profiles['family'] == 'tma_bulk' and len(profiles['profiles']) == 1, 'single exact new profile')
    short.profile_check(profiles['profiles'][0])
    original_ref = {'path': legacy['independent_replay']['normalized_path'],
                    'sha256': legacy['independent_replay']['normalized_sha256']}
    original = read_json(bound(repo, original_ref))
    require(gate_mapping(legacy).get(original_ref['path']) == original_ref['sha256'],
            'original84 normalized facts directly signed')
    require(original['actual_device_environment']['device'] == facts['device']
            and all(original['actual_device_environment']['environment'][k] == facts['environment'][k]
                    for k in ENV_FIELDS), 'S14 original and new runtime conditions differ')
    require({name:record['target_identity_sha256'] for name,record in original['actual_target_facts'].items()}
            == facts['target_facts'], 'S14 original and new complete device targets differ')
    require(original['profile_processes'] == original['target_launches'] == 84,
            'original source-release evidence incomplete')
    for key in ('archive', 'index', 'closure'):
        require(bridge['physical_evidence'][key]['path'] == facts[key]['path']
                and bridge['physical_evidence'][key]['sha256'] == facts[key]['sha256'],
                'new physical proof differs from derived facts')
    closure = read_json(bound(repo, facts['closure']))
    bundle = PackedEvidence(bound(repo, facts['archive']), bound(repo, facts['index']),
                            expected_index_sha256=facts['index']['sha256'],
                            expected_archive_sha256=facts['archive']['sha256'],
                            expected_closure=closure['members'])
    cases = {case['id']:case for case in parent['cases']}
    records = {record['case_id']:record for record in facts['records']}
    require(len(records) == len(facts['records']) == 24 and set(records) == set(cases),
            'S14 full unique new branch cases')
    names = set(gates) | set(gate_mapping(legacy)) | {value['path'] for value in refs.values()}
    small = set()
    for record in records.values():
        prefix = 'repo/' + record['run_path'] + '/'
        for local in ('validation_spec.json', 'validation_manifest.json', 'attempts/attempt_00/raw.jsonl',
                      'build/dependencies.json', 'build/shared_libraries.json', 'build/compile.stderr'):
            small.add(prefix + local)
        spec_member = prefix + 'validation_spec.json'
        require(spec_member in bundle.members, 'canonical physical spec missing')
    # The signed B3 predicates are reused. This pass rechecks every raw byte,
    # caching only explicit bounded metadata for the following closed view.
    bundle.verify_all(small_objects=sorted(small))
    identity = None
    normalized = {}
    with bundle.verified_transaction(sorted(small)) as view:
        for key, case in cases.items():
            record = records[key]; prefix = 'repo/' + record['run_path'] + '/'
            require(record['run_path'].endswith('formal-short-v1-' + key + '--formal_final_1_2_33_v1')
                    and record['seed'] == 3 and record['target_launches'] == 3 and record['status'] == 'pass',
                    'S14 signed new coordinate')
            require(view.file_sha256(prefix + 'validation_manifest.json') == record['manifest_sha256'],
                    'S14 signed original manifest')
            spec = view.read_json(prefix + 'validation_spec.json')
            require(spec['case_id'] == key and spec['profile_id'] == 'formal_final_1_2_33_v1'
                    and spec['seed'] == 3, 'S14 original spec identity')
            raw = [json_object(line) for line in view.read_bytes(prefix + 'attempts/attempt_00/raw.jsonl').splitlines()]
            require(len(raw) == 2 and raw[0] == facts['device'], 'S14 exact original device/raw rows')
            short.validate_validation(raw[0], raw[1], case, profiles['profiles'][0], 3)
            require(raw[1]['resource_identity'] == record['resource_identity']
                    and raw[1]['blocks'] == record['blocks'], 'S14 signed resources and grid')
            inventory = view.read_json(prefix + 'build/dependencies.json')
            libraries = view.read_json(prefix + 'build/shared_libraries.json')
            current = {'dependencies': inventory, 'libraries': libraries}
            if identity is None:
                identity = current
            require(current == identity, 'S14 case compiler or library conditions differ')
            for name in inventory['local']:
                require(name.startswith('repo/'), 'S14 compiler-local dependency namespace')
                live = name.removeprefix('repo/')
                require(sha(relative(repo, live)) == view.file_sha256(prefix + 'snapshot/' + name),
                        'S14 compiled local source or header changed')
                names.add(live)
            static = compile_resources(view.read_bytes(prefix + 'build/compile.stderr').decode(),
                                       [record['resource_identity']['kernel_symbol']])
            normalized[key] = {'resource': record['resource_identity'], 'blocks': record['blocks'],
                               'static': static[record['resource_identity']['kernel_symbol']]}
    return {'names': names, 'cases': normalized, 'identity': identity,
            'device': facts['device'], 'environment': facts['environment'],
            'target_facts': facts['target_facts'], 'parent': parent,
            'packed_transaction': bundle.transaction_receipt}


def verify_build(run, contract, device, *, live_environment=None):
    from common.family_b3 import compile_resources, target_identity
    from auditors.tma_bulk_admission_v1 import audit_sass
    root = Path(run)
    spec = read_json(relative(root, 'run_spec.json'))
    frozen = read_json(relative(root, spec['contract_path']))
    require(applies(frozen), 'S14 actual frozen execution revision')
    bundle = validate_bundle(root / 'snapshot/repo', frozen)
    require(device == bundle['device'], 'S14 runtime device differs from B3')
    env = read_json(relative(root, 'environment/initial.json'))
    for current in [env] + ([] if live_environment is None else [live_environment]):
        require(current['uuid'].lower() == device['uuid'].lower()
                and all(current[k] == bundle['environment'][k] for k in ENV_FIELDS),
                'S14 runtime toolchain or execution identity differs from B3')
    inventory = read_json(relative(root, 'build/dependencies.json'))
    libraries = read_json(relative(root, 'build/shared_libraries.json'))
    require(inventory == bundle['identity']['dependencies'] and libraries == bundle['identity']['libraries'],
            'S14 actual compiled dependency or library inventory differs')
    if live_environment is not None:
        for name, digest in {**inventory['external_toolchain'], **libraries}.items():
            require(Path(name).is_file() and sha(name) == digest, 'S14 live compiler or runtime file changed')
    text = relative(root, 'build/sass.txt').read_text()
    audit_sass(text, contract)
    identities = target_identity(text, bundle['target_facts'])
    digests = {name:hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
               for name,rows in identities.items()}
    require(digests == bundle['target_facts'], 'S14 complete target machine code differs from B3')
    static = compile_resources(relative(root, 'build/compile.stderr').read_text(), identities)
    result = deepcopy(contract)
    frozen_cases = {case['id']: case for case in frozen['cases']}
    require(len(result['cases']) == 24 and {case['id'] for case in result['cases']} == set(frozen_cases),
            'S14 resolved case set differs from frozen contract')
    for case in result['cases']:
        record = bundle['cases'][case['id']]
        original_case = frozen_cases[case['id']]
        normalized_case = {key:value for key,value in case.items() if key not in
                           ('b3_resource_identity', 'b3_blocks', 'resolved_resource_identity')}
        normalized_case['iterations'] = original_case['iterations']
        require(normalized_case == original_case, 'S14 runtime changed case semantics')
        if 'resolved_resource_identity' in case:
            require(case['resolved_resource_identity'] == {key:record['resource'][key] for key in
                    ('registers_per_thread', 'occupancy_limit_ctas_per_sm', 'static_smem_bytes', 'local_size_bytes')},
                    'S14 calibrated resources differ from B3')
        if 'b3_resource_identity' in case:
            require(case['b3_resource_identity'] == record['resource'] and case['b3_blocks'] == record['blocks'],
                    'S14 cached admission resources changed')
        require(static[record['resource']['kernel_symbol']] == record['static'], 'S14 static resources differ from B3')
        case['b3_resource_identity'] = record['resource']
        case['b3_blocks'] = record['blocks']
    return result

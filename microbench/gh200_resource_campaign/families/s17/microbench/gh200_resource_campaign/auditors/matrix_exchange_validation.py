"""Pure fixed ABI1 and complete-value replay for S10; no I/O or GPU calls."""
from pathlib import PurePosixPath
import hashlib
import json
import re
from common.suite_io import require
from auditors import matrix_exchange as original
from auditors.matrix_exchange_validation_baseline import BASELINE, CASE_HASHES, PROFILE_HASHES

ADAPTER_ID = 'matrix_exchange_validation_v1'
ADAPTER_ABI_VERSION = 1
REFERENCE_MODEL = 'matrix_exchange_python_coordinate_reference_v1'
REFERENCE_SHA256 = 'a3829a670b078ffa38a9c2c8b82de562f1cfba081775dc46c528d2e4a3f330bc'
validate_device = original.validate_device


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def selection(case, profile, seed):
    require(isinstance(case, dict) and case.get('id') in CASE_HASHES
            and canonical_sha(case) == CASE_HASHES[case['id']], 'frozen S10 case')
    require(isinstance(profile, dict) and profile.get('id') in PROFILE_HASHES
            and canonical_sha(profile) == PROFILE_HASHES[profile['id']], 'frozen S10 profile')
    require(type(seed) is int and seed == 3, 'fixed seed3')
    shuffle = case['parameters']['mode'] == 'shuffle'
    expected = 'shuffle_short_1_8_33_v1' if shuffle else 'matrix_short_1_2_v1'
    require(profile['id'] == expected, 'profile/case applicability')
    return [1, 8, 33] if shuffle else [1, 2]


def output_count(case):
    p = case['parameters']
    return case['threads']*p['streams'] if p['mode'] == 'shuffle' else 32+256*p['matrices']+2048


def relative_path(value):
    require(isinstance(value, str) and value, 'nonempty relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in value
            and value != '.' and path.as_posix() == value, 'noncanonical/escaping path')
    return value


def validation_argv(binary_relative, case, profile, seed):
    selection(case, profile, seed)
    return [relative_path(binary_relative), 'validate-only', case['id'], profile['id'], '3']


def validate_resources(resource, case, device):
    fields = {'kernel_symbol', 'registers_per_thread', 'static_smem_bytes', 'dynamic_smem_bytes',
              'local_size_bytes', 'occupancy_limit_ctas_per_sm', 'extensions'}
    require(isinstance(resource, dict) and set(resource) == fields, 'resource fields')
    actual = BASELINE[case['id']]
    require(resource['kernel_symbol'] == actual['symbol'] and resource['extensions'] == {}, 'resource symbol/extensions')
    for field in ('registers_per_thread', 'static_smem_bytes', 'local_size_bytes'):
        require(type(resource[field]) is int and resource[field] == actual[field], 'compiled resource drift: '+field)
    dynamic = 0 if case['parameters']['mode'] == 'shuffle' else 8192
    require(type(resource['dynamic_smem_bytes']) is int and resource['dynamic_smem_bytes'] == dynamic,
            'original dynamic shared allocation')
    occupancy = resource['occupancy_limit_ctas_per_sm']
    require(type(occupancy) is int and 1 <= occupancy <= 32, 'occupancy domain')
    shared = dynamic + resource['static_smem_bytes']
    require(shared <= device['smem_per_cta_optin_bytes'] and shared*occupancy <= device['smem_per_sm_bytes'],
            'shared capacity')
    require(occupancy*case['threads'] <= 2048
            and occupancy*case['threads']*resource['registers_per_thread'] <= device['registers_per_sm'],
            'thread/register capacity')


def validate_validation(device, row, case, profile, seed):
    lengths = selection(case, profile, seed)
    validate_device(device)
    fields = {'schema_version', 'validation_schema_version', 'type', 'case_id', 'profile_id', 'seed',
              'scope', 'threads', 'blocks', 'errors', 'target_launches', 'checks', 'resource_identity',
              'performance_eligible', 'warmup_executed', 'pilot_executed'}
    require(isinstance(row, dict) and set(row) == fields, 'validation fields')
    for field, value in [('schema_version',2), ('validation_schema_version',1), ('seed',3),
                         ('threads',case['threads']), ('blocks',1), ('errors',0)]:
        require(type(row[field]) is int and row[field] == value, 'validation integer '+field)
    require(row['type'] == 'validation' and row['case_id'] == case['id']
            and row['profile_id'] == profile['id'] and row['scope'] == 'one_cta', 'validation identity')
    require(all(row[field] is False for field in ('performance_eligible','warmup_executed','pilot_executed')),
            'no performance/warmup/pilot')
    validate_resources(row['resource_identity'], case, device)
    require(isinstance(row['target_launches'], list) and isinstance(row['checks'], list)
            and len(row['target_launches']) == len(row['checks']) == len(lengths), 'complete launch/check set')
    count = output_count(case)
    for index, length in enumerate(lengths):
        launch = row['target_launches'][index]
        require(isinstance(launch, dict) and all(type(launch.get(k)) is int for k in
                ('launch_index','iterations','threads','blocks')), 'launch integer fields')
        require(launch == {'launch_index':index, 'iterations':length, 'input_profile':'coordinate_seed_nonuniform',
                          'threads':case['threads'], 'blocks':1}, 'actual short count, never multiplied by8')
        check = row['checks'][index]
        required = {'launch_index','reference_model','reference_sha256','comparison','tolerance_id',
                    'checked_elements','expected_elements','errors','completed','verified_CTA_ids','output_artifacts'}
        require(isinstance(check, dict) and set(check) == required, 'check fields')
        for field, value in [('launch_index',index), ('checked_elements',count), ('expected_elements',count), ('errors',0)]:
            require(type(check[field]) is int and check[field] == value, 'check integer '+field)
        require(check['reference_model'] == REFERENCE_MODEL and check['reference_sha256'] == REFERENCE_SHA256
                and check['comparison'] == 'exact' and check['tolerance_id'] is None, 'frozen exact reference')
        require(check['completed'] is True and check['verified_CTA_ids'] == [0]
                and type(check['verified_CTA_ids'][0]) is int, 'complete CTA')
        artifacts = check['output_artifacts']
        require(isinstance(artifacts, list) and len(artifacts) == 1, 'one full artifact per launch')
        artifact = artifacts[0]
        require(set(artifact) == {'path','sha256','dtype','shape','evidence_kind'}, 'artifact fields')
        require(artifact['path'] == f'mx_output_{index}.u32le' and artifact['dtype'] == 'uint32'
                and artifact['shape'] == [count] and type(artifact['shape'][0]) is int
                and artifact['evidence_kind'] == 'full_values', 'full unique artifact layout')
        require(isinstance(artifact['sha256'], str) and re.fullmatch('[0-9a-f]{64}', artifact['sha256']), 'artifact hash')
    return {'status':'pass', 'case_id':case['id'], 'profile_id':profile['id'], 'target_launches':len(lengths),
            'checked_elements':count*len(lengths), 'output_evidence_kind':'full_values', 'performance_eligible':False}


def audit_artifact_values(case, profile, seed, arrays):
    """Caller verifies file bytes/hash/metadata, then supplies all decoded uint32 arrays."""
    lengths = selection(case, profile, seed)
    require(isinstance(arrays, dict) and set(arrays) == {f'mx_output_{i}.u32le' for i in range(len(lengths))},
            'complete unique value artifacts')
    total = 0
    for index, length in enumerate(lengths):
        values = arrays[f'mx_output_{index}.u32le']
        require(isinstance(values, list) and len(values) == output_count(case)
                and all(type(x) is int and 0 <= x <= 0xffffffff for x in values), 'complete uint32 values')
        require(values == original.reference(case, length, seed), 'full artifact values differ from independent reference')
        total += len(values)
    return {'status':'values_match_frozen_reference', 'case_id':case['id'], 'checked_elements':total,
            'family_B3_qualified':False, 'performance_eligible':False}


def validate_prior_evidence(evidence, request, frozen_refs):
    require(isinstance(evidence, dict) and isinstance(request, dict) and isinstance(frozen_refs, dict), 'prior evidence objects')
    selection(request.get('case'), request.get('profile'), request.get('seed',3))
    declared = evidence.get('source_artifacts_sha256', {})
    require(isinstance(declared, dict) and (not evidence or declared), 'prior artifact bindings')
    for name, digest in declared.items():
        relative_path(name)
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
                and frozen_refs.get(name) == digest, 'corrupt prior evidence')
    return {'status':'insufficient', 'case_id':request['case']['id'],
            'missing_requirements':['no independently approved S10 reuse mapping; no prior values imported']}


def audit_sass(text, contract):
    result = original.audit_sass(text, contract)
    functions = {part.splitlines()[0].strip():part for part in text.split('Function : ')[1:]}
    for case, item in zip(contract['cases'], result):
        baseline = BASELINE[case['id']]
        require(baseline['symbol'] in functions, 'missing unchanged S10 target')
        operations = []
        for line in functions[baseline['symbol']].splitlines():
            match = re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?);', line)
            if match: operations.append((int(match[1],16), ' '.join(match[2].split())))
        normalized = '\n'.join(f'{pc:04x}:{op}' for pc,op in operations)
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        require(len(operations) == baseline['instructions'] and digest == baseline['instructions_sha256'],
                'S10 target changed from archived compile-b; independent impact review required')
        item['unchanged_target_instructions_sha256'] = digest
    return result

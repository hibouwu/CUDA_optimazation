"""Pure ABI1 validation adapter for the finite S08 short profile.

No subprocess, GPU, filesystem reads/writes, or implicit prior-evidence approval.
Full output comparison and persistence of full output values are separate claims.
"""
from __future__ import annotations

from pathlib import PurePosixPath
import re

from auditors.low_precision import KINDS, audit_sass, validate_device
from common.suite_io import require

ADAPTER_ID = 'low_precision_validation_v1'
ADAPTER_ABI_VERSION = 1
PROFILE_ID = 'short_uniform_nonuniform_1_2'
REFERENCE_MODEL = 'low_precision_logical_integer_v1'
REFERENCE_PATH = 'microbench/gh200_resource_campaign/common/low_precision_reference.hpp'
REFERENCE_SHA256 = '13345802f9ea6523b42437bdaa5277bca9df5c01add140dcf0b08bff1fe07093'
PLAN = [(1, 'uniform'), (2, 'uniform'), (1, 'nonuniform'), (2, 'nonuniform')]
REQUIRED_PROFILE_FIELDS = {
    'id', 'target_iterations', 'input_profiles', 'maximum_target_launches',
    'maximum_explicit_auxiliary_launches', 'completion_requirements',
    'reference_identity', 'declared_output_check', 'output_evidence_kind',
}


def relative_path(value):
    require(isinstance(value, str) and value, 'nonempty relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts
            and '\\' not in value and path.as_posix() == value and value != '.',
            'relative path escape/noncanonical path')
    return value


def validate_profile(profile):
    require(isinstance(profile, dict) and REQUIRED_PROFILE_FIELDS <= profile.keys(),
            'complete executable S08 profile required')
    require(profile['id'] == PROFILE_ID and profile['target_iterations'] == [1, 2]
            and profile['input_profiles'] == ['uniform', 'nonuniform'], 'finite S08 profile')
    require(profile['maximum_target_launches'] == 4
            and profile['maximum_explicit_auxiliary_launches'] == 0, 'bounded target launches')
    require(profile.get('launch_plan') == [
        {'launch_index': i, 'iterations': count, 'input_profile': inputs}
        for i, (count, inputs) in enumerate(PLAN)
    ], 'frozen launch order')
    require(profile['reference_identity'] == {
        'model': REFERENCE_MODEL, 'path': REFERENCE_PATH, 'sha256': REFERENCE_SHA256,
    }, 'frozen independent reference identity')
    check = profile['declared_output_check']
    require(isinstance(check, dict) and check.get('comparison') == 'exact' and check.get('tolerance_id') is None
            and check.get('sampling') is False
            and check.get('elements_per_launch') == 'blocks*groups*chains*M*N',
            'full exact output comparison')
    require(profile['output_evidence_kind'] == 'error_count_only'
            and profile.get('resource_extensions') == {}, 'declared evidence/resources')
    require(profile.get('performance_eligible') is False
            and profile.get('warmup_allowed') is False and profile.get('pilot_allowed') is False,
            'short-only profile')
    require(profile.get('diagnostic_failure_detail_limit') == 8, 'bounded failure diagnostics')
    return profile


def case_identity(case):
    require(isinstance(case, dict) and isinstance(case.get('parameters'), dict), 'case/parameters object')
    p = case['parameters']
    require(p.get('path') in ('mma', 'wgmma') and p.get('input_type') in KINDS
            and type(p.get('groups')) is int and p['groups'] in (1, 2), 'S08 case domain')
    width, shape = (32, [16, 8, 32]) if p['path'] == 'mma' else (128, [64, 64, 32])
    require(case.get('threads') == width * p['groups'] and p.get('collective_width_threads') == width
            and p.get('shape') == shape and p.get('chains') == 2 and p.get('batch') == 16, 'original participants/work shape')
    require(case.get('scope') in ('one_cta', 'all_gpu') and case.get('id') ==
            f'{p["path"]}_{p["input_type"]}_g{p["groups"]}_{case["scope"]}', 'case identity')
    kind = p['input_type']
    acc = 'f32' if kind in KINDS[:2] else 's32'
    ptx = (f'wgmma.mma_async.sync.aligned.m64n64k32.{acc}.{kind}.{kind}' if width == 128
           else f'mma.sync.aligned.m16n8k32.row.col.{acc}.{kind}.{kind}.{acc}')
    require(p.get('ptx') == ptx and p.get('accumulator_type') == acc
            and p.get('wait') == (0 if width == 128 else None)
            and p.get('operand_source_form') == ('SS' if width == 128 else 'registers')
            and p.get('saturating') is False
            and p.get('uniform_input_value') == (0.0625 if kind in KINDS[:2] else 1)
            and p.get('nonuniform_validation_iterations') == [1, 2], 'same target/input/completion protocol')
    require(case.get('iterations') == 8192 and case.get('launch') == (
        {'kind': 'one_cta'} if case['scope'] == 'one_cta'
        else {'kind': 'sms_capped_occupancy', 'maximum_ctas_per_sm': 4}), 'frozen long case and launch policy')
    return p


def validation_argv(binary_relative, case, profile, seed):
    validate_profile(profile)
    case_identity(case)
    require(type(seed) is int and 0 <= seed <= 0xffffffff, 'uint32 seed')
    return [relative_path(binary_relative), 'validate-only', case['id'], profile['id'], str(seed)]


def validate_validation(device, row, case, profile, seed):
    validate_profile(profile)
    p = case_identity(case)
    validate_device(device)
    require(type(seed) is int and 0 <= seed <= 0xffffffff, 'uint32 seed')
    require(isinstance(row, dict), 'validation object')
    require(row.get('schema_version') == 2 and row.get('validation_schema_version') == 1
            and row.get('type') == 'validation', 'validation row schema, not trial')
    require(row.get('case_id') == case['id'] and row.get('profile_id') == PROFILE_ID
            and type(row.get('seed')) is int and row['seed'] == seed, 'validation identity')
    require(type(row.get('errors')) is int and row['errors'] == 0, 'validation errors')
    for flag in ('performance_eligible', 'warmup_executed', 'pilot_executed'):
        require(row.get(flag) is False, 'forbidden performance work: ' + flag)
    resource = row.get('resource_identity', {})
    require(isinstance(resource, dict) and set(resource) == {'kernel_symbol', 'registers_per_thread', 'static_smem_bytes',
                             'dynamic_smem_bytes', 'local_size_bytes',
                             'occupancy_limit_ctas_per_sm', 'extensions'}, 'complete resource identity')
    require(resource['kernel_symbol'] == f'lp_{p["path"]}_{p["input_type"]}_g{p["groups"]}'
            and resource['extensions'] == {}, 'kernel or undeclared resource extension')
    for field in ('registers_per_thread', 'static_smem_bytes', 'dynamic_smem_bytes',
                  'local_size_bytes', 'occupancy_limit_ctas_per_sm'):
        require(type(resource[field]) is int and resource[field] >= 0, 'resource integer: ' + field)
    occupancy = resource['occupancy_limit_ctas_per_sm']
    for field in ('smem_per_cta_optin_bytes', 'smem_per_sm_bytes', 'registers_per_sm'):
        require(type(device.get(field)) is int and device[field] > 0, 'device capacity: ' + field)
    require(1 <= occupancy <= 32 and 0 < resource['registers_per_thread'] <= 255
            and resource['dynamic_smem_bytes'] == resource['local_size_bytes'] == 0,
            'legal nonspilling target resources')
    minimum_smem = case['threads'] * 8 + 16 + (4096 if p['path'] == 'wgmma' else 0)
    require(minimum_smem <= resource['static_smem_bytes'] <= device['smem_per_cta_optin_bytes'],
            'operand/drain shared allocation')
    require(occupancy * resource['static_smem_bytes'] <= device['smem_per_sm_bytes']
            and occupancy * case['threads'] * resource['registers_per_thread'] <= device['registers_per_sm'],
            'necessary residency capacity bounds')
    blocks = 1 if case['scope'] == 'one_cta' else device['sms'] * min(4, occupancy)
    require(type(row.get('blocks')) is int and row['blocks'] == blocks
            and row.get('threads') == case['threads'] and row.get('scope') == case['scope'],
            'preserve case launch geometry')
    launches, checks = row.get('target_launches'), row.get('checks')
    require(isinstance(launches, list) and isinstance(checks, list)
            and len(launches) == len(checks) == 4, 'complete four-launch profile')
    require(all(isinstance(c, dict) and type(c.get('launch_index')) is int for c in checks)
            and {c['launch_index'] for c in checks} == set(range(4)), 'unique check launch IDs')
    indexed = {c['launch_index']: c for c in checks}
    count = blocks * p['groups'] * p['chains'] * p['shape'][0] * p['shape'][1]
    for index, (iterations, inputs) in enumerate(PLAN):
        require(isinstance(launches[index], dict)
                and all(type(launches[index].get(field)) is int
                        for field in ('launch_index', 'iterations', 'threads', 'blocks')), 'launch integer types')
        require(launches[index] == {'launch_index': index, 'iterations': iterations,
                                   'input_profile': inputs, 'threads': case['threads'], 'blocks': blocks},
                'launch profile/iteration/participant mismatch')
        check = indexed[index]
        require(set(check) == {'launch_index', 'reference_model', 'reference_sha256', 'comparison',
                              'tolerance_id', 'checked_elements', 'expected_elements', 'errors',
                              'completed', 'verified_CTA_ids', 'output_artifacts'}, 'complete check fields')
        require(check['reference_model'] == REFERENCE_MODEL and check['reference_sha256'] == REFERENCE_SHA256
                and check['comparison'] == 'exact' and check['tolerance_id'] is None, 'exact independent reference')
        require(type(check['errors']) is int and check['errors'] == 0 and check['completed'] is True,
                'numerical/completion failure')
        require(type(check['checked_elements']) is int and type(check['expected_elements']) is int
                and check['checked_elements'] == check['expected_elements'] == count, 'all output elements')
        ids = check['verified_CTA_ids']
        require(isinstance(ids, list) and all(type(i) is int for i in ids)
                and len(ids) == blocks and set(ids) == set(range(blocks)), 'all launched CTAs, not all physical SMs')
        require(check['output_artifacts'] == [], 'error_count_only profile has no value files')
    return {'status': 'pass', 'case_id': case['id'], 'profile_id': PROFILE_ID,
            'target_launches': 4, 'checked_elements': count * 4,
            'output_evidence_kind': 'error_count_only', 'performance_eligible': False}


def validate_prior_evidence(evidence, request, frozen_refs):
    """No reuse mapping is approved by this minimal S08 diagnostic adapter.

    Validate supplied artifact identities before reporting lack of coverage;
    corruption cannot be hidden as an opportunity to rerun. A later separately
    reviewed mapper can add reusable results without changing old evidence.
    """
    require(isinstance(evidence, dict) and isinstance(request, dict)
            and isinstance(frozen_refs, dict), 'JSON evidence/request/ref dictionaries')
    case = request.get('case')
    require(isinstance(case, dict), 'requested case object')
    case_identity(case)
    declared = evidence.get('source_artifacts_sha256', {})
    require(isinstance(declared, dict), 'artifact identity map')
    require(not evidence or bool(declared), 'prior evidence must declare bound artifacts')
    for path, digest in declared.items():
        relative_path(path)
        require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest)
                and frozen_refs.get(path) == digest, 'prior artifact hash mismatch')
    return {'status': 'insufficient', 'case_id': case['id'],
            'missing_requirements': ['independently approved S08 semantic reuse mapping is not implemented by this diagnostic adapter']}

"""Unregistered finite S14 formal observation wrapper.

The numerical auditor stays frozen. This wrapper additionally requires the
actual capture=false short resources and grid supplied by qualified admission.
No registry or core dispatch is changed by this module.
"""
import json
import math
import hashlib
from auditors import tma_bulk, tma_bulk_formal_v1
from auditors.tma_bulk_fullcode_baseline_v1 import TARGET_SHA256
from common.family_b3 import RESOURCE_FIELDS, target_identity
from common.suite_io import require

ADAPTER_ID = 'tma_bulk_admission_v1'
EXECUTION_REVISION = 'family-b3-s14-v1'
validate_device = tma_bulk.validate_device


def validate_contract(contract):
    require(contract.get('adapter_id') == ADAPTER_ID
            and contract.get('execution_revision') == EXECUTION_REVISION,
            'S14 finite formal admission revision required')
    require(contract.get('source') == 'microbench/gh200_resource_campaign/probes/tma_bulk_formal_v1.cu',
            'S14 capture=false reviewed formal host required')
    cases = []
    for case in contract['cases']:
        n = case.get('iterations')
        require(type(n) is int and (n == 32 or 128 <= n <= 65536), 'S14 pilot/resolved length domain')
        cases.append({**case, 'iterations': 32})
    tma_bulk.validate_contract({**contract, 'adapter_id': tma_bulk.ADAPTER_ID, 'cases': cases})
    # A qualified bundle and full B are checked by the eventual core integration;
    # declaring the revision here alone cannot authorize any process launch.
    require(isinstance(contract.get('family_b3'), dict) and contract['family_b3'],
            'S14 bound family B3 evidence missing')
    return contract


def audit_sass(text, contract):
    validate_contract(contract)
    static_contract = {**contract, 'adapter_id': tma_bulk.ADAPTER_ID,
                       'cases': [{**case, 'iterations': 32} for case in contract['cases']]}
    targets = target_identity(text, TARGET_SHA256)
    actual = {name:hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
              for name,rows in targets.items()}
    require(actual == TARGET_SHA256, 'S14 complete 128-bit target identity drift')
    return json.loads(json.dumps(tma_bulk.audit_sass(text, static_contract)))


def validate_trial(row, case, device, seed, protocol):
    require(type(case.get('iterations')) is int and 1 <= case['iterations'] <= 65536
            and type(row.get('iterations')) is int and row['iterations'] == case['iterations'],
            'S14 observed length differs from the resolved or pilot case')
    require('b3_resource_identity' in case and 'b3_blocks' in case,
            'S14 qualified capture=false resources missing')
    bound = case['b3_resource_identity']
    require(all(type(row.get(key)) is int and row[key] == bound[key] for key in RESOURCE_FIELDS),
            'S14 formal resources differ from capture=false short evidence')
    require(row.get('kernel_symbol') == bound['kernel_symbol']
            and type(row.get('blocks')) is int and row['blocks'] == case['b3_blocks'],
            'S14 formal specialization or grid differs from B3')
    extension = bound['extensions']
    require(extension.get('validation_role') == 'formal_final_transport'
            and extension.get('capture_enabled') is False,
            'S14 original capture=true evidence cannot admit this formal path')
    require(row.get('capture_enabled') is False,
            'S14 formal host unexpectedly captures short per-iteration output')
    for key in ('payload_bytes', 'global_slots_per_cta', 'global_allocation_bytes'):
        require(type(row.get(key)) is int and row[key] == extension[key],
                'S14 formal allocation differs from B3: ' + key)
    # The frozen family semantic checker identifies its original pilot32 case.
    # The declared resolved length was checked above; its observation arithmetic
    # uses row.iterations and remains unchanged. Do not alter the frozen auditor.
    return tma_bulk_formal_v1.validate_trial(row, {**case, 'iterations': 32}, device, seed, protocol)


def resolve_iterations(pilot_raw, case, device):
    validate_device(device)
    policy = case['iteration_policy']
    expected = {'kind': 'calibrated', 'pilot_iterations': 32, 'target_ns': 20000000,
                'minimum_pilot_ns': 10000, 'min_iterations': 128, 'max_iterations': 65536,
                'rounding_model': 'legacy_event_ms_truncate_then_clamp_v1'}
    require(policy == expected, 'S14 frozen calibration formula')
    require(pilot_raw.get('case_id') == case['id'] and type(pilot_raw.get('iterations')) is int
            and pilot_raw['iterations'] == 32, 'S14 exact pilot identity/length')
    ms = pilot_raw.get('event_ms')
    require(type(ms) in (int, float) and math.isfinite(ms) and ms > 0, 'S14 positive finite pilot CUDA event')
    n = int(32 * (policy['target_ns'] / 1e6) / max(ms, policy['minimum_pilot_ns'] / 1e6))
    n = max(policy['min_iterations'], min(policy['max_iterations'], n))
    return {'resolved_iterations': n, 'rounding_model': policy['rounding_model'],
            'pilot_iterations': 32, 'pilot_event_ms': ms,
            'formula_parameters': {key: policy[key] for key in
                                   ('target_ns', 'minimum_pilot_ns', 'min_iterations', 'max_iterations')}}

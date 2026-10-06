"""Common observation checks; callers supply independently computed family demands."""
from __future__ import annotations
import math
import hashlib
from pathlib import Path
from common.suite_io import cv, require

TIMER_REVISION = 'zero-work-controls-v3'
TIMER_FLAG = '-DGH_TIMER_RESOLUTION_V3=1'
TIMER_SUPPLEMENT = 'Docs/ModelEvaluation/gemm/experiments/gh200_sm90/INTERFACE-TIMER-RESOLUTION.md'


def timer_policy(case, expected=None):
    """Derive the exception from a frozen named zero-work model, never a raw flag."""
    revision=case.get('timer_resolution_revision')
    if revision is None:
        require('timer_acceptance' not in case,'unversioned timer exception')
        return False,False
    require(revision==TIMER_REVISION,'unknown timer revision')
    zero=case['work_model'] in ('empty_window','compute_empty_window_v1')
    one=case['scope']=='one_cta'
    require(case['scope'] in ('one_cta','all_gpu'),'unknown timer scope')
    policy='zero_work_one_cta_local_cycles' if zero and one else 'zero_work_grid_ns' if zero else 'strict_positive_ns'
    require(case.get('timer_acceptance')==policy,'timer acceptance disagrees with work model/scope')
    if zero:
        p=case['parameters']
        require((case['work_model']=='empty_window' and p.get('mode')=='empty' and p.get('bytes')==0)
                or (case['work_model']=='compute_empty_window_v1' and p.get('phase')=='empty_control'
                    and case['iterations']==0),'false empty-window model')
        if expected is not None:
            require(all(expected[k]==0 for k in ('work','read','write','operations')),'empty exception with counted work')
        metric={'numerator':'cta_clock64_cycles' if one else 'elapsed_ns','denominator':'one','scale':1,
                'unit':'cycles/window' if one else 'ns/window'}
        require(case['metric']==metric and case.get('exportable') is False,'empty metric/export policy')
    return zero,zero and one


def validate_timer_contract(contract):
    revision=contract.get('contract_revision')
    if revision is None:
        require('timer_resolution' not in contract and TIMER_FLAG not in contract['build']['flags'],
                'unversioned timer build/supplement')
        for case in contract['cases']:
            require('timer_resolution_revision' not in case and 'timer_acceptance' not in case,
                    'new timer case under old contract version')
            timer_policy(case)
        return False
    require(type(revision) is int and revision==3,'unknown contract revision')
    require(contract['family'] in ('memory_baseline','legacy_fma','legacy_mma','legacy_wgmma'),
            'timer revision family not registered')
    previous='microbench/gh200_resource_campaign/contracts/'+contract['family']+'.json'
    root=Path(__file__).resolve().parents[3]
    require(contract.get('previous_contract')=={'path':previous,'sha256':hashlib.sha256((root/previous).read_bytes()).hexdigest()},
            'parent contract provenance changed')
    require(contract.get('timer_resolution')=={'id':TIMER_REVISION,'supplement_path':TIMER_SUPPLEMENT,
                'supplement_sha256':hashlib.sha256((root/TIMER_SUPPLEMENT).read_bytes()).hexdigest(),
                'new_run_required':True,'old_contract_and_run_unchanged':True},'timer supplement provenance')
    require(previous in contract['dependencies'] and TIMER_SUPPLEMENT in contract['dependencies'],
            'timer revision snapshot dependencies')
    require(contract['build']['flags'].count(TIMER_FLAG)==1,'timer revision compile switch')
    for stage in ('S02',contract['stage']):
        require({'stage':stage,'phase':'timer-resolution-A','path':f'reviews/{stage}-timer-resolution-A-review.json'}
                in contract['review_dependencies'],'timer revision A review dependency')
    for case in contract['cases']:
        require(case.get('timer_resolution_revision')==TIMER_REVISION,'case timer revision missing')
        timer_policy(case)
    return True


def validate_observation(row, case, device, seed, protocol, expected):
    zero,local_control=timer_policy(case,expected)
    revision=case.get('timer_resolution_revision')
    require(row.get('timer_resolution_revision')==revision,'raw/frozen timer revision mismatch')
    require(row.get('schema_version') == 2 and row.get('type') == 'trial', 'trial schema')
    for key in ('iterations', 'seed', 'threads', 'blocks', 'errors', 'work_count',
                'read_payload_bytes', 'write_payload_bytes', 'start_ns', 'stop_ns'):
        require(type(row.get(key)) is int and row[key] >= 0, 'nonnegative integer: ' + key)
    require(row['case_id'] == case['id'] and row['iterations'] == case['iterations']
            and row['seed'] == seed, 'case/iterations/seed')
    require(row['threads'] == case['threads'] and row['blocks'] == expected['blocks']
            and row['scope'] == case['scope'], 'launch/scope')
    require(row['errors'] == 0 and row['correctness'] == expected['correctness'],
            'numerical validation evidence')
    require(row['work_unit'] == case['work_unit'] and row['work_count'] == expected['work'],
            'independently derived workload')
    require(row['read_payload_bytes'] == expected['read'] and row['write_payload_bytes'] == expected['write'],
            'independently derived directional bytes')
    require(row.get('cache_residency_proven') is False and row.get('physical_hbm_bytes_proven') is False,
            'unproven physical traffic/cache claim')
    details = row.get('blocks_detail', [])
    require(len(details) == expected['blocks'], 'CTA cardinality')
    require(all(type(x.get('block_id')) is int for x in details)
            and {x['block_id'] for x in details} == set(range(expected['blocks'])), 'CTA identities')
    for x in details:
        for key in ('smid', 'start_ns', 'stop_ns', 'start_cycle', 'stop_cycle'):
            require(type(x.get(key)) is int and x[key] >= 0, 'CTA integer: ' + key)
        require(x['stop_ns'] >= x['start_ns'] if zero else x['stop_ns'] > x['start_ns'], 'CTA globaltimer')
        require(x['stop_cycle'] > x['start_cycle'], 'CTA local cycles')
    ids = {x['smid'] for x in details}
    require(0 < len(ids) <= device['sms'], 'SM cardinality')
    if case['coverage_policy'] == 'one_sm':
        require(len(details) == 1 and len(ids) == 1, 'one CTA coverage')
    elif case['coverage_policy'] == 'exact_sms':
        require(len(ids) == device['sms'], 'full SM coverage')
    else:
        require(case['coverage_policy'] == 'observed_subset', 'unknown coverage policy')
    lo = min(x['start_ns'] for x in details); hi = max(x['stop_ns'] for x in details)
    require(row['start_ns'] == lo and row['stop_ns'] == hi, 'grid envelope')
    require(hi>=lo if local_control else hi>lo,'positive grid envelope required outside local zero-work control')
    if revision:
        require(type(row.get('globaltimer_distinguishable')) is bool
                and row['globaltimer_distinguishable']==(hi>lo),'globaltimer distinguishability flag')
        require(type(row.get('zero_ns_ctas')) is int and row['zero_ns_ctas']==sum(x['start_ns']==x['stop_ns'] for x in details),
                'zero-ns CTA diagnostic')
    event = row['event_ms']
    require(type(event) in (float, int) and math.isfinite(event) and event > 0
            and hi - lo <= event * 1.05e6, 'CUDA event cross-check')
    warm = row['warmup_samples_ns']
    require(protocol['warmup_min_windows'] <= len(warm) <= protocol['warmup_max_windows']
            and all(type(x) is int and (x>=0 if local_control else x>0) for x in warm), 'warmup samples')
    basis='cta_clock64_cycles' if local_control else 'elapsed_ns'
    selected=warm
    if revision:
        require(row.get('warmup_basis')==basis,'warmup basis mismatch')
        cycles=row.get('warmup_samples_cycles')
        require(type(cycles) is list,'warmup cycle array missing')
        if case['scope']=='one_cta':
            require(len(cycles)==len(warm) and all(type(x) is int and x>0 for x in cycles),'local warmup cycle array')
        else:
            require(cycles==[],'cross-CTA warmup cycles forbidden')
        if local_control:selected=cycles
    tail = protocol['warmup_tail_windows']
    converged = cv(selected[-tail:]) <= protocol['warmup_cv_limit']
    require(type(row['warmup_converged']) is bool and row['warmup_converged'] == converged,
            'warmup convergence flag')
    require(converged or len(warm) == protocol['warmup_max_windows'], 'premature warmup stop')
    require(all(cv(selected[n-tail:n]) > protocol['warmup_cv_limit']
                for n in range(protocol['warmup_min_windows'], len(warm))), 'missed first convergence')
    quantities = {'work_count': expected['work'], 'read_payload_bytes': expected['read'],
                  'write_payload_bytes': expected['write'], 'payload_bytes': expected['read'] + expected['write'],
                  'elapsed_ns': hi - lo, 'operation_count': expected['operations'], 'one': 1}
    if case['scope'] == 'one_cta':
        quantities['cta_clock64_cycles'] = details[0]['stop_cycle'] - details[0]['start_cycle']
    metric = case['metric']
    require(metric['numerator'] in quantities and metric['denominator'] in quantities, 'unknown metric')
    denominator = quantities[metric['denominator']]
    require(denominator > 0, 'metric denominator')
    value = quantities[metric['numerator']] / denominator * metric['scale']
    require(math.isfinite(value) and value > 0, 'metric value')
    result={'value': value, 'warmup_converged': converged, 'warmup_cv': cv(selected[-tail:]),
            'unit': metric['unit'], 'observed_sms': len(ids)}
    if revision:result.update(warmup_basis=basis,globaltimer_distinguishable=hi>lo)
    return result

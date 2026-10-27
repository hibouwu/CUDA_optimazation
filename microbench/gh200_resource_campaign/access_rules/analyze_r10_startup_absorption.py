#!/usr/bin/env python3
"""Test the existing source candidates against one prescribed startup shift."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_r10_granule_overlap import source_granules
from analyze_r10_supply_events import GPU, source_costs
from supply_model import cfg_b_source_program, shared_stage_events


PROGRESS_FIELDS = ('ready', 'starts', 'service_frontier', 'commits', 'completed',
                   'wait1_return', 'released', 'mma_returns', 'main_ends')
SHIFT = 512.
TARGET = 55


def progress_delta(base, changed, translation=0.):
    return {field: max(abs(y-x-translation)
        for a,b in zip(base['ctas'],changed['ctas']) for x,y in zip(a[field],b[field]))
        for field in PROGRESS_FIELDS}


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    def sha(path):
        path = path.resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ValueError('input changed: '+str(path))
        inputs[str(path)] = digest
        return digest
    def read(path):
        sha(path)
        return json.loads(path.read_text())
    dev = root/'Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1'
    prior = read(root/'Q03-source-service/R10-layout-cache/20261010-R10-raster-pitch/reanalysis/B-20261011-source-event-transfer-v3/B-r10-event-transfer.json')
    feedback = read(root/'Q03-source-service/R10-layout-cache/20261010-R10-raster-pitch/reanalysis/B-20261011-address-feedback-v3/B-address-feedback.json')
    observed = read(dev/'R10-producer-delay.json')
    ready_observed = read(dev/'R13-ready-release.json')
    data_rechecked = set()
    for report in (prior, feedback, observed, ready_observed):
        for name, digest in report['input_sha256'].items():
            path = Path(name)
            if path.is_relative_to(root):
                if sha(path) != digest:
                    raise ValueError('reused data input changed: '+name)
                data_rechecked.add(name)
    params = prior['parameters']
    if feedback['parameters'] != params or prior['raw_processes_verified'] != 100:
        raise ValueError('prior development tuple/replay scope changed')
    results = []
    simulated_demands = 0
    for pitch in (0,32):
        context = observed['contexts'][f'r10_proddelay_d0_p{pitch}']
        row,setup = context['row'],context['setup']
        if (setup['gpu_uuid'] != GPU or setup['stages'] != 6
                or (row['m'],row['n'],row['k'],row['ldb']) != (2304,3072,1024,3072+pitch//2)):
            raise ValueError('same-card startup geometry changed')
        program = cfg_b_source_program(row,setup)
        case = f'r10_raster_along_m_p{pitch}'
        old_context = prior['contexts'][case]
        if program != cfg_b_source_program(old_context['row'],old_context['setup']):
            raise ValueError('startup and raster source programs differ')
        delayed_context = observed['contexts'][f'r10_proddelay_d512_p{pitch}']
        if program != cfg_b_source_program(delayed_context['row'],delayed_context['setup']):
            raise ValueError('prescribed delay changed logical work')
        unique = {key for groups in program['source_keys'] for group in groups for key in group}
        for family in ('coverage_box','address32','address128'):
            candidate = {}
            if family == 'coverage_box':
                candidate['source_service_costs_ns'] = source_costs(row,setup,params['service_ns'],
                    prior['coverage_price_ns_per_KiB'])
                old_windows = prior['predicted'][case]['windows']
            else:
                width = int(family.removeprefix('address'))
                candidate.update(source_address_units={key:tuple((key[0],i)
                    for i in source_granules(key,width)) for key in unique},
                    source_unit_service_ns=params['service_ns']*width/16384)
                old_windows = feedback['predicted'][case][str(width)]['windows']
            models = {}
            for treatment in ('ordinary','target512','all512'):
                offsets = [SHIFT if treatment == 'all512' or
                    (treatment == 'target512' and c == TARGET) else 0. for c in range(132)]
                model = shared_stage_events(16,6,132,source_keys=program['source_keys'],
                    output_counts=program['output_counts'],share_inflight_sources=True,
                    trace_requests=True,diagnostic_producer_offsets_ns=offsets,**candidate,**params)
                if (model['input_demand_work']*16 != program['totals']['source_payload_KiB']
                        or model['input_delivery_work']*16 != program['totals']['destination_valid_KiB']
                        +program['totals']['destination_fill_KiB']):
                    raise ValueError('logical source/destination work changed')
                models[treatment] = model
                simulated_demands += model['input_demand_count']
            base,changed,translated = (models[n] for n in ('ordinary','target512','all512'))
            for w in old_windows:
                a = base['ctas'][w['cta']];j = w['j']
                if (a['main_ends'][j] != w['MAIN_ns'] or
                        a['main_ends'][j]-a['starts'][j*16] != w['L_ns']):
                    raise ValueError('prior zero-shift prediction changed')
            translation = progress_delta(base,translated,SHIFT)
            if max(translation.values()) > 1e-9:
                raise ValueError('common root translation changed consumer progression')
            a,b = base['ctas'][TARGET],changed['ctas'][TARGET]
            owners = {s['source_id']:s for s in base['sources']}
            prefix = [d for d in base['input_demands'] if d['cta'] == TARGET and d['index'] < 6]
            if len(prefix) != 18 or any(not d['merged'] or owners[d['source_id']]['cta'] == TARGET for d in prefix):
                raise ValueError('target initial stages are not externally owned subscriptions')
            witnesses = [dict(group=d['index'],kind=d['kind'],source_id=d['source_id'],
                owner_cta=owners[d['source_id']]['cta'],arrival_ns=d['arrival_ns'],
                owner_join_end_ns=owners[d['source_id']]['join_end_ns'],
                shifted_arrival_ns=next(x['arrival_ns'] for x in changed['input_demands']
                    if (x['cta'],x['index'],x['kind']) == (TARGET,d['index'],d['kind']))) for d in prefix]
            if any(w['shifted_arrival_ns'] >= w['owner_join_end_ns'] for w in witnesses):
                raise ValueError('prescribed startup shift crossed initial subscription cutoff')
            source_slack = min(w['owner_join_end_ns']-w['arrival_ns'] for w in witnesses)
            refill_slack = a['released'][0]-a['issued_b'][5]-params['producer_gap_ns']
            changed_progress = progress_delta(base,changed)
            # The first output is the causal test. Later output FIFO ties are
            # explicitly retained, rather than requiring the whole model to match.
            first_delta = dict(full_ready_ns=b['ready'][0]-a['ready'][0],
                FIRST_ns=b['starts'][0]-a['starts'][0],MAIN_ns=b['main_ends'][0]-a['main_ends'][0],
                L_ns=(b['main_ends'][0]-b['starts'][0])-(a['main_ends'][0]-a['starts'][0]),
                producer_to_MAIN_ns=b['main_ends'][0]-a['main_ends'][0]-SHIFT,
                issue_end_to_full_ns=b['ready'][0]-a['ready'][0]-(b['issued_b'][0]-a['issued_b'][0]))
            if any(abs(first_delta[k]) > 1e-9 for k in ('full_ready_ns','FIRST_ns','MAIN_ns','L_ns')):
                raise ValueError('existing candidate does not absorb the prescribed first-output shift')
            output_deltas = [dict(cta=c,tile=j,end_delta_ns=y['end']-x['end'])
                for c,(x,y) in enumerate(zip(base['ctas'],changed['ctas']))
                for j,(x,y) in enumerate(zip(x['outputs'],y['outputs'])) if x['end'] != y['end']]
            results.append(dict(pitch_bytes=pitch,family=family,prescribed_shift_ns=SHIFT,
                previous_zero_shift_windows_reproduced=len(old_windows),
                initial_subscriptions=witnesses,minimum_initial_subscription_slack_ns=source_slack,
                producer_first_refill_slack_ns=refill_slack,
                first_output_delta=first_delta,maximum_all_actor_progress_delta_ns=changed_progress,
                common_translation_maximum_error_ns=translation,
                input_sources_byte_equal=base['sources']==changed['sources'],
                source_generations=[m['input_source_count'] for m in models.values()],
                source_work=[m['input_source_work'] for m in models.values()],
                address_unit_service=[m.get('address_unit_service') for m in models.values()],
                output_completion_changes=output_deltas,
                makespan_delta_ns=changed['makespan_ns']-base['makespan_ns']))
            print(json.dumps(dict(pitch=pitch,family=family,subscription_slack_ns=source_slack,
                first_output_delta=first_delta)),flush=True)
        source_granules.cache_clear()
    for name in ('analyze_r10_startup_absorption.py','analyze_r10_supply_events.py',
                 'analyze_r10_granule_overlap.py','supply_model.py'):
        sha(Path(__file__).with_name(name))
    observations = [r for r in observed['direct_delay_summaries'] if r['cta'] == TARGET and r['j'] == 0]
    ready_bounds = [dict(pitch_bytes=r['pitch_bytes'],bounds=r['bounds'])
        for r in ready_observed['paired_summaries'] if r['kind'] == 'delay' and r['observer'] == 1]
    report = dict(scope='Existing fixed same-card candidates under one ideal startup shift; post-reveal development only.',
        parameters=params,coverage_price_ns_per_KiB=prior['coverage_price_ns_per_KiB'],
        results=results,simulated_runs=18,simulated_logical_demands=simulated_demands,
        reused_R10_first_output_observations=observations,related_R13_issue_to_ready_bounds=ready_bounds,
        related_R13_ready_checks=ready_observed['delay_bound_checks'],
        reused_data_inputs_rehashed=len(data_rechecked),new_numeric_replay=False,input_sha256=inputs,
        interpretation=[
            'All 18 initial target demands subscribe to sources owned by CTA1/54; +512 remains before their strict cutoffs.',
            'The first slot is not reused until after the shifted producer prefill has finished.',
            'Thus these candidate runs leave target full/FIRST/MAIN unchanged and predict producer-to-MAIN change -512ns.',
            'The existing R10 delay pairs instead move FIRST and MAIN later, with producer-to-MAIN change positive.',
            'The related R13 first-generation visibility bounds also increase issue-to-ready, unlike this candidate decrease.',
            'Source sharing does not reduce destination or D work; equal-time event order can still change later D FIFO positions.'],
        limits=[
            'Root +512 is an ideal prescribed input, not the measured timer duration or a fitted observed producer offset.',
            'Parameters have exploratory history; no new fit, root scan, physical latency split, GPU measurement or qualification.',
            'R10 and R13 use distinct instrumentation; related R13 bounds are a separate development constraint, not same-binary validation.',
            'The 1e-9 tolerance is floating-point arithmetic tolerance in the synthetic model, not a hardware timing gate.',
            'This checks these three fixed candidates, not all entry states or all source/cache/arbitration families.',
            'Source cutoff/refill slack describes this candidate state, not measured hardware capacity or a verified continuous perturbation interval.',
            'Historical observer failures, V09 scores and unsupported cross-card numerical transfer remain unchanged.'])
    (output/'B-startup-absorption.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# 当前在途共享候选吸收了512ns起步变化','',
        '固定既有GPU-ef参数，不拟价格或入口。每个B0/B32条件只计算原入口、CTA55理想后移512ns和全部入口共同后移512ns。','',
        '| 候选 / B增量 | 最小初始订阅余量ns | 首full / FIRST / MAIN变化ns | producer→MAIN变化ns |',
        '|---|---:|---|---:|']
    for r in results:
        d=r['first_output_delta']
        lines.append(f'| {r["family"]}/{r["pitch_bytes"]} | {r["minimum_initial_subscription_slack_ns"]:.6f} | '
            f'{d["full_ready_ns"]:.6f}/{d["FIRST_ns"]:.6f}/{d["MAIN_ns"]:.6f} | {d["producer_to_MAIN_ns"]:.6f} |')
    lines += ['','18个初始需求均订阅CTA1/54已在途的源，512ns没有越过截止；首slot释放也晚于延后的预填发出。',
        'R10实测producer→MAIN变化B0为+816ns[736,960]、B32为+288ns[0,416]，当前候选方向不符。',
        '相关R13首代可见性区间也显示issue→ready增加；该观察协议与R10不同，不合并成同binary验收。',
        '全部源、目的与D工作保留。后续D FIFO同刻顺序变化单独保存，未声称整个模型状态完全相同。',
        'SHA复核后复用旧数值/观察报告；本次仅模型计算，未新增硬件价格、GPU或留出资格。','']
    (output/'B-startup-absorption.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),scenarios=18,data_inputs=len(data_rechecked))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.output.resolve())

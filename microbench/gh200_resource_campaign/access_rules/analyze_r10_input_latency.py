#!/usr/bin/env python3
"""A demand-path floor and a necessary first-full response bound, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_r10_granule_overlap import source_granules
from analyze_r10_supply_events import source_costs
from supply_model import cfg_b_source_program, shared_stage_events


TARGET = 55
SHIFT = 512.


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    def sha(path):
        path=path.resolve()
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ValueError('input changed: '+str(path))
        inputs[str(path)]=digest
        return digest
    def read(path):
        sha(path)
        return json.loads(path.read_text())
    dev=root/'20261009-V09-heldout-job739011/reanalysis/candidate-development-v1'
    old=read(root/'20261010-R10-producer-delay/reanalysis/B-20261011-startup-absorption-v1/B-startup-absorption.json')
    observed=read(dev/'R10-producer-delay.json')
    params=old['parameters']
    rehashed=set()
    for name,digest in old['input_sha256'].items():
        path=Path(name)
        if path.is_relative_to(root):
            if sha(path) != digest:raise ValueError('reused data changed: '+name)
            rehashed.add(name)
    gap=params['operand_gap_ns'];producer_gap=params['producer_gap_ns']
    stages=6;kt=16;latency=params['read_latency_ns'];destination=params['destination_service_ns']
    issue_offsets=[0.,gap,2*gap]
    destination_suffix=[2*destination,destination,destination*.5]
    own_path_suffix=max(a+d for a,d in zip(issue_offsets,destination_suffix))
    prefill_last_issue=(stages-1)*(2*gap+producer_gap)+2*gap
    # Before this domain's endpoint, every initial issue finishes before even
    # the minimum read latency. Consumer feedback cannot alter the prefix.
    pre_consumer_shift_limit=latency-prefill_last_issue
    results=[];demands=0
    for previous in old['results']:
        pitch,family=previous['pitch_bytes'],previous['family']
        context=observed['contexts'][f'r10_proddelay_d0_p{pitch}']
        row,setup=context['row'],context['setup']
        program=cfg_b_source_program(row,setup)
        if (setup['gpu_uuid']!='GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'
                or setup['stages']!=stages or row['k']//64!=kt or len(program['output_counts'])!=132):
            raise ValueError('fixed development domain changed')
        initial=previous['initial_subscriptions']
        order={'issue_a':0,'issue_b0':1,'issue_b1':2}
        first=sorted((w for w in initial if w['group']==0),key=lambda w:order[w['kind']])
        if len(first)!=3 or any(w['owner_cta']==TARGET for w in initial):
            raise ValueError('initial source owner contract changed')
        source_ready=[w['owner_join_end_ns'] for w in first]
        if source_ready!=sorted(source_ready) or any(w['arrival_ns']!=issue_offsets[i] for i,w in enumerate(first)):
            raise ValueError('first-copy FIFO order not preserved by the floor')
        if any(w['owner_join_end_ns']<source_ready[-1] or w['arrival_ns']<issue_offsets[-1]
                for w in initial if w['group']>0):
            raise ValueError('future initial copies can precede the first B1 admission')
        shift_limit=min(pre_consumer_shift_limit,previous['minimum_initial_subscription_slack_ns'])
        if SHIFT>=shift_limit:raise ValueError('prescribed shift outside verified prefix domain')
        source_suffix=max(r+d for r,d in zip(source_ready,destination_suffix))
        formula=lambda delta,tau:max(source_suffix,delta+tau+own_path_suffix)
        args={}
        if family=='coverage_box':
            args['source_service_costs_ns']=source_costs(row,setup,params['service_ns'],old['coverage_price_ns_per_KiB'])
        else:
            width=int(family.removeprefix('address'))
            unique={key for groups in program['source_keys'] for group in groups for key in group}
            args.update(source_address_units={key:tuple((key[0],i) for i in source_granules(key,width)) for key in unique},
                source_unit_service_ns=params['service_ns']*width/16384)
        models=[]
        for shift in (0.,SHIFT):
            model=shared_stage_events(kt,stages,132,source_keys=program['source_keys'],
                output_counts=program['output_counts'],share_inflight_sources=True,trace_requests=True,
                input_min_latency_ns=latency,
                diagnostic_producer_offsets_ns=[shift if c==TARGET else 0. for c in range(132)],**args,**params)
            demands+=model['input_demand_count']
            actor=model['ctas'][TARGET]
            actual_prefix=sorted((d for d in model['input_demands'] if d['cta']==TARGET and d['index']==0),
                key=lambda d:order[d['kind']])
            if ([d['source_ready_ns'] for d in actual_prefix]!=source_ready
                    or any(not d['merged'] for d in actual_prefix)
                    or abs(actor['ready'][0]-formula(shift,latency))>1e-9
                    or actor['starts'][0]!=actor['ready'][0]):
                raise ValueError('first full/FIRST differs from independently derived FIFO expression')
            if (model['input_demand_work']*16!=program['totals']['source_payload_KiB']
                    or model['input_delivery_work']*16!=program['totals']['destination_valid_KiB']
                    +program['totals']['destination_fill_KiB']):
                raise ValueError('logical source/destination work changed')
            models.append(dict(shift_ns=shift,full_ns=actor['ready'][0],FIRST_ns=actor['starts'][0],
                MAIN_ns=actor['main_ends'][0],L_ns=actor['main_ends'][0]-actor['starts'][0],
                source_generations=model['input_source_count'],input_source_work=model['input_source_work'],
                first_copy_admissions=actual_prefix))
        base,changed=models
        first_change=changed['FIRST_ns']-base['FIRST_ns']
        if not 0<=first_change<=SHIFT:raise ValueError('constant-floor necessary bound violated')
        results.append(dict(pitch_bytes=pitch,family=family,
            initial_sources_ready_ns=source_ready,source_suffix_ns=source_suffix,
            own_path_suffix_ns=own_path_suffix,shift_domain_open_upper_ns=shift_limit,
            formula='FIRST(delta,tau)=max(source_suffix,delta+tau+own_path_suffix)',
            prescribed_tau_ns=latency,models=models,
            change=dict(full_ns=changed['full_ns']-base['full_ns'],FIRST_ns=first_change,
                MAIN_ns=changed['MAIN_ns']-base['MAIN_ns'],L_ns=changed['L_ns']-base['L_ns'],
                producer_to_FIRST_ns=first_change-SHIFT,
                producer_to_MAIN_ns=changed['MAIN_ns']-base['MAIN_ns']-SHIFT)))
        source_granules.cache_clear()
    observed_trials=[dict(pitch_bytes=r['pitch_bytes'],trial=r['trial'],
        producer_shift_ns=r['P0_ns_delta'],S_change_ns=r['S_ns_delta'],
        producer_shift_cycles=r['P0_cycles_delta'],S_change_cycles=r['S_cycles_delta'])
        for r in observed['measured_gate_effect_processes']]
    if (len(observed_trials)!=20 or any(r['producer_shift_ns']<0 or r['producer_shift_ns']>=pre_consumer_shift_limit
            or r['S_change_ns']<=0 or r['S_change_cycles']<=0 for r in observed_trials)):
        raise ValueError('observed per-trial source-start constraint changed')
    for name in ('analyze_r10_input_latency.py','analyze_r10_supply_events.py',
                 'analyze_r10_granule_overlap.py','supply_model.py','test_r13_input_latency.py'):
        sha(Path(__file__).with_name(name))
    report=dict(scope='Constant per-demand path hypothesis and initial-prefix response bound; post-reveal development.',
        parameters=params,coverage_price_ns_per_KiB=old['coverage_price_ns_per_KiB'],
        prescribed_tau='Reuse existing read_latency_ns; no independently identified new price.',
        results=results,simulated_runs=12,simulated_logical_demands=demands,
        constant_floor_bound=dict(tau_domain='all nonnegative constants',delta_domain_open_upper_ns=pre_consumer_shift_limit,
            conclusion='0 <= FIRST(delta,tau)-FIRST(0,tau) <= delta, so producer-to-FIRST cannot increase.',
            proof='With fixed source suffix V and own-path suffix H, FIRST=max(V,delta+tau+H). '
                'The max of one fixed and one delta-shifted predecessor is nondecreasing and 1-Lipschitz in delta.',
            prefix_conditions=['Initial target requests all join other owners before their strict cutoff.',
                'Every initial producer issue completes before read_latency_ns; no consumer feedback can precede this prefix.',
                'A/B0/B1 admissions preserve order; all later initial copies arrive after B1, and refill follows consumer release.',
                'One per-CTA FIFO and fixed service costs, no new contention/server/entry-history edge.']),
        observed_trials=observed_trials,reused_data_inputs_rehashed=len(rehashed),new_numeric_replay=False,input_sha256=inputs,
        decision='A fixed own-path floor is distinct from latency relocation and removes complete absorption at tau=L, '
            'but this entire constant-floor prefix family still cannot produce the observed positive S response. '
            'Stop before fitting tau or expanding a GPU matrix.',
        limits=['The algebraic exclusion is conditional on the verified fixed initial source pool/FIFO organization, not all supply families.',
            'Observed producer shifts only check the necessary-bound domain; they are not substituted as model prediction inputs.',
            'Ideal root movement is not an exact implementation of the timer loop or independent-process initial state.',
            'Native and ns observed S changes remain separate; no target clock converts synthetic ns service prices.',
            'tau=L is one explicit developmental assignment, not a measured hardware latency or accepted rule.',
            'Existing V09 failure, observer qualification limits and unsupported cross-card pricing stay unchanged.'])
    (output/'B-input-latency.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 每需求最短路径：迟发不再完全吸收，固定常数仍不足','',
        '目的FIFO入口=max(共享源完成+目的后程,本需求发出+τ)；源按原时刻退出pending，不因晚订阅者延长。',
        'τ取既有read_latency，不拟新价；它与将同一L全部搬到目的后程不等价。','',
        '| 候选 / B增量 | full/FIRST变化ns | MAIN变化ns | L变化ns | producer→FIRST变化ns |',
        '|---|---:|---:|---:|---:|']
    for r in results:
        d=r['change']
        lines.append(f'| {r["family"]}/{r["pitch_bytes"]} | {d["FIRST_ns"]:.6f} | {d["MAIN_ns"]:.6f} | '
            f'{d["L_ns"]:.6f} | {d["producer_to_FIRST_ns"]:.6f} |')
    lines+=['',f'初始前缀的保守根变化域为0≤δ<{pre_consumer_shift_limit:.6f}ns。',
        '任意非负常数τ下，FIRST(δ,τ)=max(V,δ+τ+H)，所以FIRST增加不超过δ，producer→FIRST不能增加。',
        '739619的20个原配对producer时移均在该域内，S变化在ns与native cycles中却全部为正。',
        '这些观测只作条件必要界检查，不作为预测入口；实际干预、隐藏初态与观察资格限制保留。',
        '因此不拟τ、不扩GPU矩阵；下一机制须改变源/投递共享状态或其它前驱，不能只加固定每需求延迟。','']
    (output/'B-input-latency.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),scenarios=12,demands=demands,rehashed_data=len(rehashed))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.output.resolve())

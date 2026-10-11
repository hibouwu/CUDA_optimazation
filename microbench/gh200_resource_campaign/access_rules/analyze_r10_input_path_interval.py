#!/usr/bin/env python3
"""Exact initial envelopes and a conservative finite-stage no-idle certificate."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from analyze_r10_granule_overlap import source_granules
from analyze_r10_shared_path_prefix import maximum_difference,upper_lines,value
from supply_model import cfg_b_source_program


def rational(number):
    return F.from_float(number) if isinstance(number,float) else F(number)


def initial_sources(program,stages,units,unit_cost,latency,operand,producer=F(0)):
    """All initial issues precede release; strict source expiry is checked directly."""
    rows=[];frontier=F(0);boxes={};pending_units={}
    for group in range(stages):
        for atom,work in enumerate((F(1),F(1,2),F(1,2))):
            issue=group*(2*operand+producer)+atom*operand
            for cta,groups in enumerate(program['source_keys']):
                key=groups[group][atom]
                if key in boxes and issue<boxes[key]:ready=boxes[key]
                else:
                    new=[u for u in units[key] if u not in pending_units or issue>=pending_units[u]]
                    joined=[pending_units[u] for u in units[key] if u in pending_units and issue<pending_units[u]]
                    if new:
                        frontier=max(frontier,issue)+len(new)*unit_cost;batch=frontier+latency
                        for unit in new:pending_units[unit]=batch
                    else:batch=issue
                    ready=max([batch]+joined);boxes[key]=ready
                rows.append(dict(cta=cta,group=group,atom=atom,issue=issue,source_ready=ready,work=work))
    return rows,frontier


def initial_destinations(rows,count,latency,destination):
    """Exact max-of-affine envelopes; each CTA's source ready must be ordered."""
    path=[(F(0),F(0))];local=[[(F(0),F(0))] for _ in range(count)]
    last_source=[F(0)]*count;full={}
    for row in rows:
        cta=row['cta'];work=row['work']
        if row['source_ready']<last_source[cta]:raise ValueError('initial source completion order changed')
        last_source[cta]=row['source_ready']
        path=upper_lines([(a+work,b) for a,b in path]+[(work,row['issue'])])
        admission=upper_lines([(a,b+latency) for a,b in path]+[(F(0),row['source_ready'])])
        local[cta]=upper_lines([(a,b+destination*work) for a,b in local[cta]+admission])
        if row['atom']==2:full[cta,row['group']]=local[cta]
    return full,path


def finite_stage_bounds(*,stages,kt,compute,lead,operand,latency,destination,
                        first_min,first_max,source_work,path_work,initial_source_end,
                        initial_path_end,initial_last_ready,producer=F(0),initial_last_issue=F(0)):
    """Sufficient bounds, independent of source ownership or within-cohort FIFO order."""
    spread=first_max-first_min;cohort_span=spread+2*operand
    first_refill_earliest=first_min+2*compute-lead
    first_refill_latest=first_max+2*compute-lead+2*operand
    admission_upper=first_refill_latest+max(source_work,path_work)+latency
    return dict(
        cohort_span_below_subscription_ns=latency-cohort_span,
        cohort_order_gap_ns=compute-cohort_span,
        producer_next_slot_gap_ns=compute-2*operand-producer,
        initial_producer_free_slack_ns=first_refill_earliest-initial_last_issue-producer,
        source_capacity_headroom_ns=compute-source_work,
        request_capacity_headroom_ns=compute-path_work,
        destination_capacity_headroom_ns=compute-2*destination,
        initial_source_cleared_ns=first_refill_earliest-initial_source_end,
        initial_request_cleared_ns=first_refill_earliest-initial_path_end,
        initial_destination_induction_slack_ns=admission_upper-initial_last_ready,
        refill_TC_slack_ns=(stages-2)*compute+lead-spread-2*operand
            -max(source_work,path_work)-latency-2*destination,
        D_after_all_input_issues_ns=first_min+kt*compute
            -(first_max+(kt-stages+1)*compute-lead+2*operand))


def encode(number):return dict(exact=str(number),ns=float(number))


def analyze(root,output):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def sha(path):
        digest=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.resolve())]=digest
        return digest
    def read(path):sha(path);return json.loads(path.read_text())
    previous=read(root/'20261010-R10-raster-pitch/reanalysis/B-20261011-input-path-v1/B-input-path.json')
    prefix=read(root/'20261010-R10-producer-delay/reanalysis/B-20261011-shared-path-prefix-v1/B-shared-path-prefix.json')
    original=read(root/'20261010-R10-raster-pitch/reanalysis/B-20261011-source-event-transfer-v3/B-r10-event-transfer.json')
    endpoints=read(root/'20261010-R10-raster-pitch/reanalysis/main-input-path-endpoints-v1/report.json')
    trace_attempts=[]
    for version in (2,3):
        directory=root/f'20261010-R10-raster-pitch/reanalysis/B-20261011-input-path-interval-v{version}'
        trace=read(directory/'B-input-path-interval.json');archived_source=directory/'branch-cell-source.py'
        expected=next(v for n,v in trace['input_sha256'].items() if n.endswith('analyze_r10_input_path_interval.py'))
        if sha(archived_source)!=expected:raise ValueError('historical trace source snapshot changed')
        trace_attempts.append(dict(version=version,source_sha256=expected,
            method='Full heap ordering' if version==2 else 'Exploratory CTA/request partial ordering',
            cell={name:trace['cells'][0][name] for name in ('lower','upper','comparisons')},
            remaining_open_intervals=trace['remaining_open_intervals'],
            interval_proven=False,boundary_points_checked=False,
            use='Local diagnostic only; neither method supplies the following closed-interval proof.'))
    rehashed=set()
    for prior in (previous,prefix,original,endpoints):
        for name,digest in prior['input_sha256'].items():
            path=Path(name)
            if path.is_relative_to(root):
                if sha(path)!=digest:raise ValueError('reused data changed: '+name)
                rehashed.add(name)
    params=previous['parameters']
    if params!=prefix['parameters']:raise ValueError('prefix parameter tuple changed')
    fixed_zero=('reuse_delay_ns','completion_lag_ns','issue_floor_ns',
        'wait_overhead_ns','loop_advance_ns','return_overhead_ns','handoff_delay_ns',
        'destination_latency_ns')
    if any(params.get(name,0)!=0 for name in fixed_zero):raise ValueError('no-idle recurrence assumptions changed')
    lower=F(previous['selection']['lower_exact']);upper=F(previous['selection']['upper_exact'])
    context=original['contexts']['r10_source_single126_p32'];row,setup=context['row'],context['setup']
    program=cfg_b_source_program(row,setup);count=len(program['output_counts']);stages=setup['stages'];kt=row['k']//64
    if (program['output_counts']!=[1]*126 or stages!=6 or kt!=16
            or setup['gpu_uuid']!='GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'):
        raise ValueError('same-card single126 B32 geometry changed')
    latency=rational(params['read_latency_ns']);operand=rational(params['operand_gap_ns'])
    producer=rational(params['producer_gap_ns'])
    compute=rational(params['compute_ns']);lead=rational(params['commit_lead_ns'])
    destination=rational(params['destination_service_ns']);unit_cost=rational(params['service_ns']/128)
    if not 0<=lead<=compute:raise ValueError('commit recurrence domain changed')
    keys={key for groups in program['source_keys'] for group in groups for key in group}
    units={key:tuple((key[0],i) for i in source_granules(key,128)) for key in keys}
    aligned_context=original['contexts']['r10_raster_along_m_p0']
    aligned=cfg_b_source_program(aligned_context['row'],aligned_context['setup'])
    aligned_owners={};aligned_keys=set()
    for groups in aligned['source_keys']:
        for group in groups[:stages]:
            for atom,key in enumerate(group):
                atom_units=tuple((key[0],i) for i in source_granules(key,128))
                if len(atom_units)!=128*(1 if atom==0 else F(1,2)):
                    raise ValueError('aligned source batch differs from whole-box service')
                for unit in atom_units:
                    if unit in aligned_owners and aligned_owners[unit]!=key:
                        raise ValueError('aligned prefix has partial inter-box source joins')
                    aligned_owners[unit]=key
                aligned_keys.add(key)
    cohort_units=[set().union(*(units[key] for groups in program['source_keys'] for key in groups[g])) for g in range(kt)]
    if any(cohort_units[a]&cohort_units[b] for a in range(kt) for b in range(a)):
        raise ValueError('address units shared between K cohorts')
    rows,source_end=initial_sources(program,stages,units,unit_cost,latency,operand,producer)
    if rows[-1]['issue']>=latency:raise ValueError('refill could alter the initial issue prefix')
    full,path=initial_destinations(rows,count,latency,destination)
    initial=[]
    for cta in range(count):
        for group in range(1,stages):
            gap,at=maximum_difference(full[cta,group],full[cta,0],[(lower,upper)],F(0))
            initial.append(dict(cta=cta,group=group,minimum_TC_slack=encode(group*compute-gap),at_price=str(at)))
    first_min=min(value(full[c,0],lower) for c in range(count))
    first_max=max(value(full[c,0],upper) for c in range(count))
    maximum_source=max(len(s) for s in cohort_units)*unit_cost
    bounds=finite_stage_bounds(stages=stages,kt=kt,compute=compute,lead=lead,operand=operand,
        latency=latency,destination=destination,first_min=first_min,first_max=first_max,
        source_work=maximum_source,path_work=2*count*upper,initial_source_end=source_end,
        initial_path_end=value(path,upper),initial_last_ready=max(value(full[c,stages-1],upper) for c in range(count)),
        producer=producer,initial_last_issue=rows[-1]['issue'])
    comparisons=[]
    for endpoint,price in (('lower_exact',lower),('upper_exact',upper)):
        matched=next(r for r in endpoints['rows'] if r['endpoint']==endpoint and r['case']==row['id'])
        if F(matched['exact_price'])!=price:raise ValueError('independent endpoint price changed')
        error=max(abs(float(value(full[c,0],price))-a['first_ns']) for c,a in enumerate(matched['ctas']))
        margin_error=max(abs(float(value(full[c,0],price)+g*compute-value(full[c,g],price))
            -a['ready_margin_ns'][g-1]) for c,a in enumerate(matched['ctas']) for g in range(1,stages))
        if max(error,margin_error)>1e-7:raise ValueError('initial envelope differs from independent float graph')
        comparisons.append(dict(endpoint=endpoint,first_endpoint_error_ns=error,initial_margin_error_ns=margin_error))
    midpoint=F(previous['selection']['selected_exact'])
    midpoint_error=max(abs(float(value(full[c,0],midpoint))-w['FIRST_ns'])
        for c,w in enumerate(previous['predicted'][row['id']]['windows']))
    if midpoint_error>1e-7:raise ValueError('initial envelope differs from previous full graph')
    proven=all(F(r['minimum_TC_slack']['exact'])>=0 for r in initial) and all(v>0 for v in bounds.values())
    for name in ('analyze_r10_input_path_interval.py','analyze_r10_shared_path_prefix.py',
                 'analyze_r10_granule_overlap.py','supply_model.py','test_r13_input_path_interval.py'):
        sha(Path(__file__).with_name(name))
    observed=next(r for r in previous['comparisons'] if r['pair']=='single126')
    report=dict(scope='Closed-interval sufficient no-TC-idle certificate: single126 B32 first outputs, same fixed request-path/source graph.',
        case=row['id'],parameters=params,price_interval=dict(lower=str(lower),upper=str(upper)),
        initial_prefix_demands=len(rows),initial_prefix_work_16KiB=sum(float(r['work']) for r in rows),
        aligned_B0_prefix_address128_equals_box_service=True,aligned_prefix_distinct_boxes=len(aligned_keys),
        K_cohort_address_units=[len(s) for s in cohort_units],K_cohorts_disjoint=True,
        initial_envelopes=[dict(cta=c,group=g,lines=[dict(slope=str(a),intercept=str(b)) for a,b in lines])
            for (c,g),lines in full.items()],
        first_min=encode(first_min),first_max=encode(first_max),first_spread_bound=encode(first_max-first_min),
        source_cohort_service_upper=encode(maximum_source),request_cohort_service_upper=encode(2*count*upper),
        initial_margins=initial,bounds={name:encode(v) for name,v in bounds.items()},
        exact_arithmetic_interval_proven=proven,all_first_L_ns=encode(kt*compute) if proven else None,
        predicted_B32_ratio_upper=0 if proven else None,observed_B32_response=observed['observed'],
        independent_endpoint_checks=comparisons,previous_midpoint_first_error_ns=midpoint_error,
        earlier_local_trace_attempts=trace_attempts,
        reused_data_inputs_rehashed=len(rehashed),input_sha256=inputs,
        assumptions=['First outputs only, S6, Ktile16, commit lead in [0,compute], fixed producer/operand gaps and zero other recurrence overheads.',
            'Every source address unit belongs to one K cohort; cohort issue span is below read latency, so each is charged at most once.',
            'Global source/request FIFO work per cohort is below the compute period; destination work per group is also below it.',
            'The initial prefix is exact; later bounds use a first-stall contradiction, not an assumed successful no-stall execution.',
            'Before any possible first stall, slot reuse fixes issue_A(g)=FIRST+(g-S+2)*compute-lead; B1 follows by 2*operand.',
            'The cohort bound includes any within-cohort ownership/arbitration order. D cannot enqueue before all first inputs issue.',
            'Certificate inequalities and max envelopes cover the entire closed interval, including every branch crossing and both endpoints.'],
        limits=['Exact arithmetic on the existing binary input constants; IEEE endpoint/midpoint parity is separate evidence, not a rounding proof.',
            'B0 has L>=Ktile*compute in this graph, hence B32 L=Ktile*compute rules out a positive single126 response.',
            'Rejects only the existing common nominal-work FIFO within the earlier B0 necessary interval and fixed tuple/source state.',
            'No exclusion of other request-work definitions, queue partitions, arbitration rules or source-state families.',
            'No new GPU/numerical replay, independent frozen local holdout, observer qualification or cfg_a/c timing.'])
    (output/'B-input-path-interval.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# single126 B32：必要价格闭区间内首窗无空闲证书','',
        f'价格区间 [{float(lower):.12f}, {float(upper):.12f}] ns/16KiB；精确有理数计算。',
        f'全区间证明通过：{proven}。126个首输出L均为{kt*compute}ns。','',
        '| 保守余量 | ns |','|---|---:|']
    for name,v in bounds.items():lines.append(f'| {name} | {float(v):.9f} |')
    lines+=['',f'初始g1～g5最小余量：{min(r["minimum_TC_slack"]["ns"] for r in initial):.9f}ns。',
        '这是一项固定模型的条件排除，不是新硬件规则或独立留出通过。','']
    (output/'B-input-path-interval.md').write_text('\n'.join(lines))
    print(json.dumps(dict(proven=proven,first_min=float(first_min),first_max=float(first_max),
        initial_minimum_TC_slack_ns=min(r['minimum_TC_slack']['ns'] for r in initial),
        bounds={name:float(v) for name,v in bounds.items()}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.data_root.resolve(),args.output.resolve())

#!/usr/bin/env python3
"""Exact necessary prefix constraints for two shared demand-path FIFO placements."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import gzip
import hashlib
import json
from pathlib import Path

from supply_model import cfg_b_source_program,shared_stage_events


TARGET=55


def rational(value):
    return F.from_float(value) if isinstance(value,float) else F(value)


def upper_lines(lines):
    """Upper envelope of affine lines for all nonnegative service prices."""
    best={}
    for slope,intercept in lines:best[slope]=max(intercept,best.get(slope,intercept))
    hull=[]
    for slope,intercept in sorted(best.items()):
        while hull:
            crossing=(hull[-1][1]-intercept)/(slope-hull[-1][0])
            if hull[-1][2] is None or crossing>hull[-1][2]:break
            hull.pop()
        hull.append((slope,intercept,crossing if hull else None))
    while len(hull)>1 and hull[1][2]<=0:hull.pop(0)
    return [(slope,intercept) for slope,intercept,_ in hull]


def value(lines,price):return max(slope*price+intercept for slope,intercept in lines)


def breakpoints(lines):
    return [(a[1]-b[1])/(b[0]-a[0]) for a,b in zip(lines,lines[1:])]


def active(lines,price):return max(lines,key=lambda line:line[0]*price+line[1])


def queue_envelopes(rows,placement,tau,destination):
    frontier=[(F(0),F(0))];local=[(F(0),F(0))];full={}
    for row in rows:
        arrival=row['source_ready'] if placement=='return' else row['issue']
        if placement=='return' and row['target']:arrival=max(arrival,row['issue']+tau)
        frontier=upper_lines([(slope+row['work'],intercept) for slope,intercept in frontier]
            +[(row['work'],arrival)])
        if row['target']:
            incoming=frontier if placement=='return' else upper_lines(
                [(slope,intercept+tau) for slope,intercept in frontier]+[(F(0),row['source_ready'])])
            local=upper_lines([(slope,intercept+destination*row['work'])
                for slope,intercept in incoming+local])
            if row['kind']=='issue_b1':full[row['group']]=local
    return full


def direct(rows,placement,tau,destination,price):
    frontier=F(0);local=F(0);full={}
    for row in rows:
        arrival=row['source_ready'] if placement=='return' else row['issue']
        if placement=='return' and row['target']:arrival=max(arrival,row['issue']+tau)
        frontier=max(frontier,arrival)+price*row['work']
        if row['target']:
            incoming=frontier if placement=='return' else max(row['source_ready'],frontier+tau)
            local=max(local,incoming)+destination*row['work']
            if row['kind']=='issue_b1':full[row['group']]=local
    return full


def feasible_regions(functions,constraints):
    """Project affine inequalities on each exact envelope segment."""
    cuts=sorted({F(0)}|{x for lines in functions.values() for x in breakpoints(lines) if x>0})
    regions=[]
    for index,left in enumerate(cuts):
        right=cuts[index+1] if index+1<len(cuts) else None
        point=(left+right)/2 if right is not None else left+1
        chosen={name:active(lines,point) for name,lines in functions.items()}
        low,high=left,right
        for terms,limit in constraints:
            slope=sum(coefficient*chosen[name][0] for name,coefficient in terms.items())
            intercept=sum(coefficient*chosen[name][1] for name,coefficient in terms.items())
            if slope==0:
                if intercept>limit:high=low-1;break
            elif slope>0:high=min(high,(limit-intercept)/slope) if high is not None else (limit-intercept)/slope
            else:low=max(low,(limit-intercept)/slope)
        if high is None or low<=high:regions.append((low,high))
    merged=[]
    for low,high in regions:
        if merged and merged[-1][1] is not None and low<=merged[-1][1]:
            merged[-1]=(merged[-1][0],None if high is None else max(merged[-1][1],high))
        else:merged.append((low,high))
    return merged


def maximum_difference(first,second,regions,offset):
    best=None
    for low,high in regions:
        if high is None:raise ValueError('baseline window did not bound the queue price')
        points={low,high}|{x for lines in (first,second) for x in breakpoints(lines) if low<x<high}
        for price in points:
            result=value(first,price)-value(second,price)-offset
            if best is None or result>best[0]:best=(result,price)
    return best


def encode_regions(regions):
    return [dict(lower=str(low),upper=str(high) if high is not None else None,
        lower_ns_per_16KiB=float(low),upper_ns_per_16KiB=float(high) if high is not None else None)
        for low,high in regions]


def analyze(root,output):
    output.mkdir(parents=True,exist_ok=False);inputs={}
    def sha(path):
        path=path.resolve();digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in inputs and inputs[str(path)]!=digest:raise ValueError('input changed')
        inputs[str(path)]=digest
        return digest
    def read(path):sha(path);return json.loads(path.read_text())
    old=read(root/'Q03-source-service/R10-layout-cache/20261010-R10-producer-delay/reanalysis/B-20261011-startup-absorption-v1/B-startup-absorption.json')
    observed=read(root/'Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-producer-delay.json')
    rehashed=set()
    for name,digest in old['input_sha256'].items():
        path=Path(name)
        if path.is_relative_to(root):
            if sha(path)!=digest:raise ValueError('reused data changed: '+name)
            rehashed.add(name)
    params=old['parameters'];context=observed['contexts']['r10_proddelay_d0_p0']
    program=cfg_b_source_program(context['row'],context['setup'])
    if (context['setup']['gpu_uuid']!='GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'
            or context['setup']['stages']!=6 or program['output_counts'][TARGET]!=3):
        raise ValueError('fixed aligned cfg_b development domain changed')
    model=shared_stage_events(16,6,132,source_keys=program['source_keys'],output_counts=program['output_counts'],
        share_inflight_sources=True,trace_requests=True,**params)
    prefix=[d for d in model['input_demands'] if d['index']<6]
    if len(prefix)!=2376:raise ValueError('incomplete six-stage prefix')
    tau=rational(params['read_latency_ns']);destination=rational(params['destination_service_ns'])
    own=[d for d in prefix if d['cta']==TARGET]
    if any(rational(d['source_ready_ns'])<rational(d['arrival_ns'])+tau for d in prefix):
        raise ValueError('ordinary prefix depends on tau within [0,L]')
    target_subscriptions=next(r for r in old['results'] if r['family']=='coverage_box' and r['pitch_bytes']==0)
    domain=min(tau-max(rational(d['arrival_ns']) for d in own),
        rational(target_subscriptions['minimum_initial_subscription_slack_ns']))
    trials=[r for r in observed['measured_gate_effect_processes'] if r['pitch_bytes']==0]
    shifts=sorted({F(512)}|{F(r['P0_ns_delta']) for r in trials})
    if any(delta<0 or delta>=domain for delta in shifts):raise ValueError('shift outside initial source-pool domain')
    # Refills use distinct boxes and address units. Their earliest possible
    # completion is later than every initial return admission, even at tau=L.
    initial_keys={k for groups in program['source_keys'] for group in groups[:6] for k in group}
    refill_keys={k for groups in program['source_keys'] for k in groups[6]}
    if initial_keys&refill_keys:raise ValueError('refill source can join an initial source')
    latest_initial=max(max(rational(d['source_ready_ns']),rational(d['arrival_ns'])+
        (max(shifts) if d['cta']==TARGET else 0)+tau) for d in prefix)
    earliest_refill=min(rational(a['released'][0]) for a in model['ctas'])+tau
    if latest_initial>=earliest_refill:raise ValueError('unmodeled refill can overtake the return prefix')
    encoded=gzip.compress(json.dumps(prefix,separators=(',',':')).encode(),mtime=0)
    (output/'B-initial-demands.json.gz').write_bytes(encoded)
    local_order={d['demand_id']:i for i,d in enumerate(sorted(own,key=lambda d:d['arrival_ns']))}
    if [d['index'] for d in sorted(own,key=lambda d:d['arrival_ns'])]!=[g for g in range(6) for _ in range(3)]:
        raise ValueError('target initial producer order changed')
    baseline=[]
    for d in prefix:
        baseline.append(dict(source_ready=rational(d['source_ready_ns']),issue=rational(d['arrival_ns']),
            work=rational(d['work']),target=d['cta']==TARGET,group=d['index'],kind=d['kind'],serial=d['demand_id']))
    def ordered(delta,placement):
        if delta==0:
            return sorted(baseline,key=lambda r:(r['source_ready'] if placement=='return' else r['issue'],r['serial']))
        # Every delayed target issue follows the peer's complete initial issue
        # prefix. Peer packets therefore precede target packets on a tie.
        if delta<=max(r['issue'] for r in baseline if not r['target']):
            raise ValueError('delayed target issue can precede peer prefix construction')
        buckets=defaultdict(F)
        for r in baseline:
            if not r['target']:buckets[r['source_ready'] if placement=='return' else r['issue']]+=r['work']
        rows=[dict(source_ready=t,issue=t,work=w,target=False,group=-1,kind='',serial=-1) for t,w in buckets.items()]
        # Later target requests cannot precede its first B1 and hence cannot
        # affect the first full. No source/destination work is merged or deleted
        # from the ledger: this is only the causal prefix cut for this endpoint.
        rows += [dict(r,issue=r['issue']+delta) for r in baseline if r['target'] and r['group']==0]
        return sorted(rows,key=lambda r:((max(r['source_ready'],r['issue']+tau) if r['target'] else r['source_ready'])
            if placement=='return' else r['issue'],r['target'],r['serial']))
    upper_L=F(next(r for r in observed['case_summaries'] if r['case']=='r10_proddelay_d0_p0')['boundaries']['L_ns']['process_max'])
    S_values=[r['S_ns_delta'] for r in trials];compute=rational(params['compute_ns'])
    summaries=[];oracle_checks=0
    for placement in ('return','request'):
        base_rows=ordered(F(0),placement);base=queue_envelopes(base_rows,placement,tau,destination)
        functions={f'full{g}':lines for g,lines in base.items()}
        conditions=[({f'full{g}':F(1),'full0':F(-1)},upper_L-(16-g)*compute) for g in range(1,6)]
        price_region=feasible_regions(functions,conditions)
        if not price_region:raise ValueError('baseline prefix alone already infeasible')
        delta_results=[]
        for delta in shifts:
            rows=ordered(delta,placement);changed=queue_envelopes(rows,placement,tau,destination)[0]
            curves=dict(functions,changed=changed)
            # Independent rational FIFO evaluation at each affine boundary and
            # interior, including the unbounded final interval.
            cuts=sorted({F(0)}|{x for lines in curves.values() for x in breakpoints(lines) if x>0})
            points=set(cuts)|{(a+b)/2 for a,b in zip(cuts,cuts[1:])}|{cuts[-1]+1}
            for price in points:
                expected=direct(base_rows,placement,tau,destination,price)
                actual=direct(rows,placement,tau,destination,price)[0]
                if any(value(base[g],price)!=v for g,v in expected.items()) or value(changed,price)!=actual:
                    raise ValueError('symbolic envelope differs from independent rational FIFO')
                oracle_checks+=1
            maximum,at_price=maximum_difference(changed,base[0],price_region,delta)
            # This is a generous interval screen at the prescribed ideal shift,
            # not a fitted price or an exact replay of the timer intervention.
            joint=feasible_regions(curves,conditions+[
                ({'changed':F(-1),'full0':F(1)},-F(min(S_values))-delta),
                ({'changed':F(1),'full0':F(-1)},F(max(S_values))+delta)])
            delta_results.append(dict(delta_ns=float(delta),maximum_S_change_ns=float(maximum),
                maximum_S_change_exact=str(maximum),maximum_at_price=str(at_price),
                joint_price_intervals=encode_regions(joint)))
        trial_checks=[]
        for r in trials:
            bound=next(x for x in delta_results if x['delta_ns']==r['P0_ns_delta'])
            trial_checks.append(dict(trial=r['trial'],producer_shift_ns=r['P0_ns_delta'],observed_S_change_ns=r['S_ns_delta'],
                maximum_model_S_change_ns=bound['maximum_S_change_ns'],
                shortfall_ns=r['S_ns_delta']-bound['maximum_S_change_ns']))
        summaries.append(dict(placement=placement,tau_ns=float(tau),
            baseline_L_upper_ns=int(upper_L),baseline_price_intervals=encode_regions(price_region),
            delta_results=delta_results,conditional_trial_checks=trial_checks,
            baseline_envelopes={name:[dict(slope=str(m),intercept=str(b)) for m,b in lines] for name,lines in functions.items()}))
    for name in ('analyze_r10_shared_path_prefix.py','supply_model.py'):
        sha(Path(__file__).with_name(name))
    report=dict(scope='Aligned same-card cfg_b initial-prefix necessary screen; no full lifecycle fit or accepted service.',
        parameters=params,prefix_demands=len(prefix),prefix_logical_work_16KiB=sum(d['work'] for d in prefix),
        shift_domain_open_upper_ns=float(domain),latest_initial_admission_ns=float(latest_initial),
        earliest_possible_refill_return_ns=float(earliest_refill),oracle_checks=oracle_checks,
        shared_price_unit='ns per 16KiB of logical destination demand; abstract service, not physical bandwidth',
        results=summaries,reused_data_inputs_rehashed=len(rehashed),new_numeric_replay=False,input_sha256=inputs,
        assumptions=['Source owners/initial read state and existing compute/local delivery prices are fixed developmental inputs.',
            'Return placement: max(source_ready,issue+tau) -> shared FIFO -> existing per-CTA FIFO.',
            'Request placement: issue -> shared FIFO -> tau, joined with source_ready -> existing per-CTA FIFO.',
            'One global work-conserving FIFO with a common nonnegative price; deterministic request order, no label prices.',
            'Every peer destination demand is paid. Aggregated equal-time peer work only compresses an exact FIFO prefix.',
            'L >= full_ready(g)-FIRST+(Ktiles-g)*compute_ns for every initial group; later constraints can only tighten it.'],
        return_tau_extension='Ordinary initial admissions are unchanged for 0<=tau<=L. Delaying the ordered target subsequence '
            'can only delay its first full, so tau=L is an upper S-response envelope for this return-FIFO family.',
        limits=['Rational arithmetic treats saved binary floating-point model inputs exactly; it does not establish hardware precision.',
            'Per-trial observed shifts only locate conditional necessary bounds, never become free prediction inputs.',
            'The prescribed ideal 512ns comparison and observed timer implementation/hidden initial states remain distinct.',
            'Only necessary aligned prefix constraints are screened; B32/raster/scale/spatial and complete lifecycle are not validated.',
            'Request placement feasibility would only justify a next full-model calculation, not adoption or a GPU request.',
            'tau>L, other arbitration/server partitions and state-changing source pools remain outside the return extension.',
            'No new fit, GPU, observer qualification, V09 change or cross-card numerical transfer.'])
    (output/'B-shared-path-prefix.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 共享路径FIFO的位置：首供给与原首窗必须同时满足','',
        '源/目的/D账目不合并；新增共享路径每个目的需求都付费，源服务价格保持原开发参数。',
        '首六stage为2376需求/1584个16KiB逻辑工作单位。全价格域使用精确有理数上包络，不扫网格拟价。','',
        '| 位置 | 原首窗必要价格域 ns/16KiB | 512ns迟发下最大S增量 | 同时落入已见S范围的价格域 |',
        '|---|---|---:|---|']
    for r in summaries:
        d=next(x for x in r['delta_results'] if x['delta_ns']==512)
        lines.append(f'| {r["placement"]} | {json.dumps(r["baseline_price_intervals"])} | '
            f'{d["maximum_S_change_ns"]:.6f}ns | {json.dumps(d["joint_price_intervals"])} |')
    lines+=['','每个实际起步差的条件上界、初始源/未来refill分界及独立直接FIFO复核写入JSON。',
        '回程FIFO对0≤τ≤原read_latency取最大τ作响应上包络；请求路径这里只检查τ=原read_latency。',
        '可行前缀仍不是完整预测；必须继续通过B32、方向/规模/空间、局部进度与独立留出。','']
    (output/'B-shared-path-prefix.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),oracle_checks=oracle_checks,
        results=[dict(placement=r['placement'],baseline=r['baseline_price_intervals'],
            prescribed512=next(x for x in r['delta_results'] if x['delta_ns']==512)) for r in summaries])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args=parser.parse_args()
    analyze(args.data_root.resolve(),args.output.resolve())

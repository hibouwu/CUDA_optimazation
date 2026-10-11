"""Static TMA requests and calibrated supply prediction, without fitting or IO.

All coefficients and conditional LP constraints are explicit caller inputs.
The request spans are descriptor coverage, not physical memory traffic.
"""
from collections import defaultdict
from functools import lru_cache
import math

import numpy as np


# tm, tn, cluster_m, cluster_n; fixed from V08/CUTLASS configurations.
CONFIGS = {"cfg_a": (128, 128, 2, 1), "cfg_b": (128, 128, 1, 1),
           "cfg_c": (256, 128, 1, 2)}


def source_sharing_delta_cycles(a, b, active_ctas, extra128_times_B_peer, kt=1):
    """Explicit development candidate; not used by the production predictor.

    a and b are shared calibrated coefficients. extra128_times_B_peer is the
    logical B extra-128-byte coverage in KiB times other users of that panel.
    It is not physical traffic. Inputs retain their numeric type, including
    Fraction for the offline parameter-domain analysis.
    """
    aligned = a * active_ctas
    pitched = aligned + b * extra128_times_B_peer
    return kt * (max(512, pitched) - max(512, aligned))


def admission_stage_events(kt, stages, producer_interval, ready_latency,
                           commit_lead, completion_lag, *, service_cycles=512.,
                           issue_floor=0., wait_overhead=0., loop_advance=0.,
                           return_overhead=0., reuse_delay=0., handoff_delay=0.,
                           observed_second_entry=None, producer_boundary_delay=0.,
                           second_prefill=False):
    """Development recurrence for two pingpong outputs; not a fitted hardware rule.

    I is the end-of-copy-pair marker. E is cumulative TC service progress,
    C group commit, D retire eligibility.
    Commit can overlap service: C=max(B+g,E-q), D=E+lambda. Restricting g<=c
    and q>=0 ensures B<=C<=E<=D. q is effective service lead in cycles,
    not a hardware queue capacity. All times and overheads are caller inputs.
    Ordinary release is optimistic at wait1 return; reuse_delay separately
    represents the release-to-next-pair-issue path, not a physical latency.
    handoff_delay is a sensitivity input from a sender's mma return to the
    next consumer becoming available. It is not a measured barrier price.
    observed_second_entry=(H_gap,B_gap) is a diagnostic cut relative to the
    preceding modeled mma return. It replaces handoff_delay at output 1 and
    preserves the producer, service and slot states. Inconsistent ready times
    raise ValueError; measured entry times must never become prediction inputs.
    producer_boundary_delay is a one-time control gap between the two producer
    load calls, before the empty-slot constraint; it is not a per-stage price.
    second_prefill waits for the second output's first S ready groups before
    its prologue, without releasing slots or resetting TC progress. Its begin/end
    are modeled events, not observed readiness. Only this enabled intervention
    adds a second_prefill diagnostic to the otherwise unchanged return schema.
    """
    values=(producer_interval,ready_latency,commit_lead,completion_lag,
            service_cycles,issue_floor,wait_overhead,loop_advance,
            return_overhead,reuse_delay,handoff_delay,producer_boundary_delay)
    if kt<2 or stages<2 or min(values)<0 or service_cycles<=0:
        raise ValueError('admission diagnostic requires Ktile/stages>=2 and nonnegative times')
    if issue_floor>service_cycles:
        raise ValueError('issue floor must not exceed per-tile service')
    if second_prefill and (kt<stages or observed_second_entry is not None):
        raise ValueError('second prefill requires Ktile>=stages and no observed entry cut')
    if observed_second_entry is not None:
        entry_h,entry_b=observed_second_entry
        if not 0<=entry_h<=entry_b:
            raise ValueError('observed entry requires 0 <= H gap <= B gap')
    issued=[];ready=[];available=[];starts=[];frontier=[];commits=[];done=[];wait=[]
    released=[0.]*(2*kt);returns=[];ends=[];next_consumer=0.;prefill=None
    for i in range(2*kt):
        # At this boundary all S release predecessors belong to output 0.
        # Schedule the initial fill before consuming it; later slots still wait
        # for output 1's actual releases, as in the untreated recurrence.
        through=kt+stages if i==kt and second_prefill else i+1
        while len(issued)<through:
            j=len(issued)
            issue=issued[-1]+producer_interval if issued else 0.
            if j==kt:issue+=producer_boundary_delay
            if j>=stages:issue=max(issue,released[j-stages]+reuse_delay)
            issued.append(issue);ready.append(issue+ready_latency)
        if i==kt and second_prefill:
            prefill=dict(begin=next_consumer,end=max(next_consumer,max(ready[kt:kt+stages])),
                waited_stages=stages)
            next_consumer=prefill['end']
        if i==kt and observed_second_entry is not None:
            next_consumer=returns[0]+entry_h
            start=returns[0]+entry_b
            if ready[i]>start+1e-8:
                raise ValueError('observed entry precedes modeled readiness')
        else:start=max(ready[i],next_consumer)
        available.append(next_consumer);starts.append(start)
        service_end=max(start,frontier[-1] if frontier else start)+service_cycles
        frontier.append(service_end)
        commit=max(start+issue_floor,service_end-commit_lead);commits.append(commit)
        done.append(service_end+completion_lag)
        if i%kt==0:
            returned=commit  # Prologue does not wait1 or release a preceding WG's group.
        else:
            returned=max(commit,done[i-1])+wait_overhead
            released[i-1]=returned
        wait.append(returned)
        if i%kt==kt-1:
            mma_return=returned+return_overhead
            end=max(mma_return,done[i]);released[i]=end
            returns.append(mma_return);ends.append(end)
            next_consumer=mma_return+handoff_delay  # Do not reset TC or slots, or wait for tail.
        else:next_consumer=returned+loop_advance
    result=dict(windows=[ends[j]-starts[j*kt] for j in range(2)],
        issued=issued,ready=ready,consumer_available=available,
        first_mma_or_commit=starts,service_frontier=frontier,commits=commits,
        completed=done,wait1_return=wait,released=released,
        mma_returns=returns,main_ends=ends,slot=[i%stages for i in range(2*kt)])
    if prefill is not None:result['second_prefill']=prefill
    return result


def shared_stage_events(kt, stages, cta_count, *, service_ns, read_latency_ns,
                        compute_ns, output_latency_ns=0., output_weight=1.,
                        producer_gap_ns=0., operand_gap_ns=0., reuse_delay_ns=0.,
                        commit_lead_ns=0., completion_lag_ns=0., issue_floor_ns=0.,
                        wait_overhead_ns=0., loop_advance_ns=0., return_overhead_ns=0.,
                        handoff_delay_ns=0., epilogue_prepare_ns=0., epilogue_write_ns=0.,
                        epilogue_control_ns=0., epilogue_tail_ns=0., epilogue_defer=0,
                        trace_requests=False, diagnostic_producer_offsets_ns=None,
                        share_inflight_sources=False, source_keys=None,
                        destination_service_ns=0., second_prefill=False,
                        source_join_boundary='ready',local_epilogue_exclusion='none',
                        output_counts=None, source_service_costs_ns=None):
    """Development FIFO server coupled to cfg_b's pingpong software graph.

    All times are ns supplied before prediction. Each input group sends A then
    B0 then B1: A is 16 KiB, each B atom is 8 KiB. Eight 8 KiB D subtiles use DelayTmaStore and two source
    buffers. A has work 1, B0/B1 each 1/2; an output has work eta/2.
    The abstract server is not identified with a physical cache or memory port.
    Local epilogue preparation/write costs do not contend with TC in this first
    candidate; failure of the local-output intervention can falsify this choice.

    CTAs enter together. Equal-time events use insertion order, initialized by
    CTA index; this deterministic FIFO hypothesis is not measured arbitration.
    Input slots and output buffers constrain requests; there is no fitted queue
    capacity. defer=1 delays CTA0 E0 until its M1; defer=2 delays every E0.
    Default prediction uses no measured entry, E0 timestamp or target clock.
    diagnostic_producer_offsets_ns is an explicit conditional cut at each
    producer first-work loop; observed values are never formal prediction inputs.
    Optional source sharing joins only reads still in flight at logical issue.
    Completion removes the key: this is not a resident cache. Each logical atom
    is separately delivered through its CTA's FIFO after source completion.
    source_join_boundary closes subscription at ready (legacy), service_end,
    or service_start. These are alternative event hypotheses, not cache facts.
    Admission closes strictly at that timestamp, including equal-time issues.
    Optional local epilogue exclusion blocks producer issue and/or MMA start
    during this CTA's prepare/write intervals. Zero-duration request/compute
    starts do not reserve the path themselves: this is a one-way exclusion
    hypothesis, not a symmetric instruction scheduler or measured SM port.
    second_prefill gates only CTA0's second output on all S destination slots.
    output_counts optionally specifies positive per-CTA output counts. Default
    is the original two-output graph. Starting output j>=2 also requires E[j-2]
    to return: the same WG cannot overwrite its accumulator before epilogue.
    Local scheduler/next-output preparation costs remain uncalibrated.
    source_service_costs_ns optionally gives explicit per-input-atom service
    durations. Logical payload/destination work stays fixed; outputs retain
    the original service rule. Equal source identities require equal costs.
    """
    import heapq

    costs=(service_ns,read_latency_ns,output_latency_ns,output_weight,producer_gap_ns,
        operand_gap_ns,reuse_delay_ns,commit_lead_ns,completion_lag_ns,issue_floor_ns,
        wait_overhead_ns,loop_advance_ns,return_overhead_ns,handoff_delay_ns,
        epilogue_prepare_ns,epilogue_write_ns,epilogue_control_ns,epilogue_tail_ns,
        destination_service_ns)
    if min(kt,stages)<2 or cta_count<1 or compute_ns<=0 or min(costs)<0:
        raise ValueError('shared stage candidate requires positive geometry/compute and nonnegative costs')
    if issue_floor_ns>compute_ns or epilogue_defer not in (0,1,2):
        raise ValueError('invalid issue floor or output-order treatment')
    if second_prefill and kt<stages:
        raise ValueError('second prefill requires Ktile>=stages')
    if source_join_boundary not in ('ready','service_end','service_start'):
        raise ValueError('unknown source subscription boundary')
    if local_epilogue_exclusion not in ('none','producer','consumer','both'):
        raise ValueError('unknown local epilogue exclusion scope')
    counts=[2]*cta_count if output_counts is None else list(output_counts)
    if len(counts)!=cta_count or any(type(n) is not int or n<1 for n in counts):
        raise ValueError('positive per-CTA output counts required')
    if ((epilogue_defer==2 and min(counts)<2) or
            ((epilogue_defer==1 or second_prefill) and counts[0]<2)):
        raise ValueError('second-output treatment requires the affected outputs')
    if source_service_costs_ns is not None:
        if len(source_service_costs_ns)!=cta_count or any(len(groups)!=counts[c]*kt or
                any(len(costs)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in costs)
                    for costs in groups) for c,groups in enumerate(source_service_costs_ns)):
            raise ValueError('finite nonnegative source costs require shape [CTA][outputs*Ktile][A/B0/B1]')
    if share_inflight_sources:
        if source_keys is None or len(source_keys)!=cta_count or any(
                len(groups)!=counts[c]*kt or any(len(keys)!=3 for keys in groups)
                for c,groups in enumerate(source_keys)):
            raise ValueError('source keys must have shape [CTA][outputs*Ktile][A/B0/B1]')
        weights={};source_costs={}
        for c,groups in enumerate(source_keys):
            for i,keys in enumerate(groups):
                for atom,(key,weight) in enumerate(zip(keys,(1.,.5,.5))):
                    if key in weights and weights[key]!=weight:
                        raise ValueError('shared source key has inconsistent atom weight')
                    weights[key]=weight
                    if source_service_costs_ns is not None:
                        cost=source_service_costs_ns[c][i][atom]
                        if key in source_costs and source_costs[key]!=cost:
                            raise ValueError('shared source key has inconsistent service cost')
                        source_costs[key]=cost
    offsets=[0.]*cta_count if diagnostic_producer_offsets_ns is None else list(diagnostic_producer_offsets_ns)
    if len(offsets)!=cta_count or min(offsets)<0:
        raise ValueError('diagnostic producer offsets must match CTA count and be nonnegative')
    events=[];serial=0;frontier=0.;busy=0.;requests=[];request_count=0
    pending={};source_generation_keys={};sources=[];demands=[];deliveries=[]
    input_demand_count=0;input_source_count=0;input_merged_count=0;input_delivery_count=0
    input_demand_work=0.;input_source_work=0.;input_merged_work=0.;input_delivery_work=0.
    input_source_busy=0.
    destination_frontiers=[0.]*cta_count;destination_busy=[0.]*cta_count
    local_intervals=[[] for _ in range(cta_count)]
    local_blocked={'producer':0.,'consumer':0.}
    routed_inputs=share_inflight_sources or destination_service_ns>0
    prefill=None
    actors=[]
    def push(time,kind,cta,index):
        nonlocal serial
        heapq.heappush(events,(time,serial,kind,cta,index));serial+=1
    for cta in range(cta_count):
        total=counts[cta]*kt
        actor={name:[None]*total for name in ('issued_a','issued_b0','issued_b1','issued_b','ready','starts',
            'service_frontier','commits','completed','wait1_return','released')}
        actor.update(ready_count=[0]*total,producer_index=0,producer_free=offsets[cta],producer_scheduled=False,
            consumer_index=0,consumer_free=0.,consumer_scheduled=False,
            mma_returns=[None]*counts[cta],main_ends=[None]*counts[cta],
            outputs=[dict(begin=None,end=None,scheduled=False,
                prepare_begin=[None]*8,prepare_end=[None]*8,write_begin=[None]*8,write_end=[None]*8,
                submitted=[None]*8,source_done=[None]*8) for _ in range(counts[cta])])
        actors.append(actor)

    def producer(cta):
        a=actors[cta];i=a['producer_index']
        if i==counts[cta]*kt or a['producer_scheduled']:return
        t=a['producer_free']
        if i>=stages:
            release=a['released'][i-stages]
            if release is None:return
            t=max(t,release+reuse_delay_ns)
        a['producer_scheduled']=True;push(t,'issue_a',cta,i)

    def consumer(cta):
        nonlocal prefill
        a=actors[cta];i=a['consumer_index']
        if i==counts[cta]*kt or a['consumer_scheduled'] or a['ready_count'][i]!=3:return
        if i%kt==0 and i//kt>=2:
            previous=a['outputs'][i//kt-2]['end']
            if previous is None:return
            a['consumer_free']=max(a['consumer_free'],previous)
        if second_prefill and cta==0 and i==kt:
            if any(n!=3 for n in a['ready_count'][kt:kt+stages]):return
            begin=a['consumer_free'];end=max(begin,max(a['ready'][kt:kt+stages]))
            prefill=dict(begin_ns=begin,end_ns=end,waited_stages=stages)
            a['consumer_free']=end
        a['consumer_scheduled']=True
        push(max(a['consumer_free'],a['ready'][i]),'mma',cta,i)

    def epilogue(cta,out):
        a=actors[cta]
        if out==counts[cta]:return
        e=a['outputs'][out]
        if e['scheduled'] or a['main_ends'][out] is None:return
        t=a['main_ends'][out]
        if out==0 and (epilogue_defer==2 or (epilogue_defer==1 and cta==0)):
            if a['mma_returns'][1] is None:return
            t=max(t,a['mma_returns'][1])
        if out>0:
            if a['outputs'][out-1]['end'] is None:return
            t=max(t,a['outputs'][out-1]['end'])
        e['scheduled']=True;push(t,'epi_begin',cta,out)

    def request(t,cta,kind,index,weight,latency,source_cost=None):
        nonlocal frontier,busy,request_count
        start=max(t,frontier);duration=service_ns*weight if source_cost is None else source_cost;frontier=start+duration
        busy+=duration;request_count+=1;ready=frontier+latency
        if trace_requests:requests.append(dict(cta=cta,kind=kind,index=index,arrival_ns=t,
            start_ns=start,service_end_ns=frontier,ready_ns=ready,work=weight))
        return ready

    def reserve_epilogue(cta,start,duration,kind,index):
        if local_epilogue_exclusion!='none' and duration>0:
            local_intervals[cta].append(dict(start_ns=start,end_ns=start+duration,
                kind=kind,index=index))

    def input_request(t,cta,kind,index,weight):
        nonlocal input_demand_count,input_source_count,input_merged_count,input_delivery_count
        nonlocal input_demand_work,input_source_work,input_merged_work,input_delivery_work
        nonlocal input_source_busy
        atom={'issue_a':0,'issue_b0':1,'issue_b1':2}[kind]
        key=source_keys[cta][index][atom] if share_inflight_sources else None
        existing=pending.get(key) if share_inflight_sources else None
        merged=existing is not None and t<existing[2]
        if merged:
            source_id,done,join_end=existing
            input_merged_count+=1;input_merged_work+=weight
        else:
            cost=None if source_service_costs_ns is None else source_service_costs_ns[cta][index][atom]
            done=request(t,cta,kind,index,weight,read_latency_ns,cost)
            duration=service_ns*weight if cost is None else cost
            input_source_busy+=duration
            join_end={'ready':done,'service_end':frontier,
                      'service_start':frontier-duration}[source_join_boundary]
            source_id=input_source_count;input_source_count+=1;input_source_work+=weight
            if share_inflight_sources:
                pending[key]=(source_id,done,join_end);source_generation_keys[source_id]=key
                push(done,'source_done',cta,source_id)
            if trace_requests:
                source=dict(requests[-1],source_id=source_id,source_key=key,join_end_ns=join_end)
                sources.append(source)
        demand_id=input_demand_count;input_demand_count+=1;input_demand_work+=weight
        if routed_inputs or trace_requests:
            demand=dict(demand_id=demand_id,source_id=source_id,source_key=key,merged=merged,
                cta=cta,kind=kind,index=index,arrival_ns=t,source_ready_ns=done,work=weight)
        if routed_inputs:
            demands.append(demand);push(done,'deliver_input',cta,demand_id)
        else:
            # Preserve the original zero-destination event ordering exactly.
            push(done,'input_ready',cta,index)
            input_delivery_count+=1;input_delivery_work+=weight
            if trace_requests:
                demands.append(demand)
                deliveries.append(dict(demand,start_ns=done,service_end_ns=done,ready_ns=done))

    for cta in range(cta_count):producer(cta)
    while events:
        t,_,kind,cta,i=heapq.heappop(events);a=actors[cta]
        local_role='producer' if kind in ('issue_a','issue_b0','issue_b1') else 'consumer' if kind=='mma' else None
        if local_role and local_epilogue_exclusion in (local_role,'both'):
            end=max((r['end_ns'] for r in local_intervals[cta] if r['start_ns']<=t<r['end_ns']),default=t)
            if end>t:
                local_blocked[local_role]+=end-t;push(end,kind,cta,i);continue
        if kind in ('issue_a','issue_b0','issue_b1'):
            field={'issue_a':'issued_a','issue_b0':'issued_b0','issue_b1':'issued_b1'}[kind]
            a[field][i]=t
            input_request(t,cta,kind,i,1. if kind=='issue_a' else .5)
            if kind=='issue_a':push(t+operand_gap_ns,'issue_b0',cta,i)
            elif kind=='issue_b0':push(t+operand_gap_ns,'issue_b1',cta,i)
            else:
                a['issued_b'][i]=t
                a['producer_index']=i+1;a['producer_free']=t+producer_gap_ns
                a['producer_scheduled']=False;producer(cta)
        elif kind=='source_done':
            key=source_generation_keys.pop(i)
            if pending.get(key,(None,None))[0]==i:del pending[key]
        elif kind=='deliver_input':
            demand=demands[i];weight=demand['work']
            start=max(t,destination_frontiers[cta]);duration=destination_service_ns*weight
            done=start+duration;destination_frontiers[cta]=done;destination_busy[cta]+=duration
            input_delivery_count+=1;input_delivery_work+=weight
            if trace_requests:deliveries.append(dict(demand,start_ns=start,service_end_ns=done,ready_ns=done))
            push(done,'input_ready',cta,demand['index'])
        elif kind=='input_ready':
            a['ready_count'][i]+=1;a['ready'][i]=t
            consumer(cta)
        elif kind=='mma':
            a['starts'][i]=t
            E=max(t,a['service_frontier'][i-1] if i else t)+compute_ns
            C=max(t+issue_floor_ns,E-commit_lead_ns);D=E+completion_lag_ns
            W=C if i%kt==0 else max(C,a['completed'][i-1])+wait_overhead_ns
            for name,value in [('service_frontier',E),('commits',C),('completed',D),('wait1_return',W)]:
                a[name][i]=value
            if i%kt:push(W,'release',cta,i-1)
            if i%kt==kt-1:
                out=i//kt;M=W+return_overhead_ns
                push(M,'mma_return',cta,out);push(max(M,D),'main_end',cta,out)
            else:
                a['consumer_index']=i+1;a['consumer_free']=W+loop_advance_ns
                a['consumer_scheduled']=False;consumer(cta)
        elif kind=='release':
            a['released'][i]=t;producer(cta)
        elif kind=='mma_return':
            a['mma_returns'][i]=t
            a['consumer_index']=(i+1)*kt;a['consumer_free']=t+handoff_delay_ns
            a['consumer_scheduled']=False;consumer(cta)
            epilogue(cta,0)
        elif kind=='main_end':
            a['main_ends'][i]=t;a['released'][(i+1)*kt-1]=t
            producer(cta);epilogue(cta,i)
        elif kind=='epi_begin':
            e=a['outputs'][i];e['begin']=t;e['prepare_begin'][0]=t
            reserve_epilogue(cta,t,epilogue_prepare_ns,'prepare',i*8)
            push(t+epilogue_prepare_ns,'epi_prepared',cta,i*8)
        elif kind=='epi_prepared':
            out,j=divmod(i,8);e=a['outputs'][out];e['prepare_end'][j]=t
            if j==0:
                e['write_begin'][j]=t
                reserve_epilogue(cta,t,epilogue_write_ns,'write',i)
                push(t+epilogue_write_ns,'epi_written',cta,i)
            else:push(t+epilogue_control_ns,'store',cta,i-1)
        elif kind=='epi_written':
            out,j=divmod(i,8);e=a['outputs'][out];e['write_end'][j]=t
            if j==7:push(t+epilogue_control_ns,'store',cta,i)
            else:
                e['prepare_begin'][j+1]=t
                reserve_epilogue(cta,t,epilogue_prepare_ns,'prepare',i+1)
                push(t+epilogue_prepare_ns,'epi_prepared',cta,i+1)
        elif kind=='store':
            out,j=divmod(i,8);e=a['outputs'][out];e['submitted'][j]=t
            done=request(t,cta,'store',i,output_weight/2,output_latency_ns);e['source_done'][j]=done
            if j==7:
                push(max(done,e['source_done'][j-1])+epilogue_tail_ns,'epi_end',cta,out)
            else:
                # After committing j, wait until at most one source read remains.
                # Then write j+1 into the buffer formerly used by j-1.
                begin=max(t,e['source_done'][j-1]) if j else t
                e['write_begin'][j+1]=begin
                reserve_epilogue(cta,begin,epilogue_write_ns,'write',i+1)
                push(begin+epilogue_write_ns,'epi_written',cta,i+1)
        elif kind=='epi_end':
            a['outputs'][i]['end']=t
            epilogue(cta,i+1);consumer(cta)
    if any(a['outputs'][-1]['end'] is None for a in actors):
        raise RuntimeError('shared stage graph did not complete')
    makespan=max(a['outputs'][-1]['end'] for a in actors)
    result=dict(ctas=actors,makespan_ns=makespan,shared_busy_ns=busy,
        last_service_end_ns=frontier,request_count=request_count,requests=requests,
        input_demand_count=input_demand_count,input_source_count=input_source_count,
        input_merged_count=input_merged_count,input_delivery_count=input_delivery_count,
        input_demand_work=input_demand_work,input_source_work=input_source_work,
        input_merged_work=input_merged_work,input_delivery_work=input_delivery_work,
        source_busy_ns=service_ns*input_source_work if source_service_costs_ns is None else input_source_busy,
        destination_busy_ns=sum(destination_busy),destination_busy_by_cta_ns=destination_busy,
        local_epilogue_blocked_ns=local_blocked,
        local_epilogue_intervals=local_intervals if trace_requests else [],
        sources=sources,input_demands=demands if trace_requests else [],deliveries=deliveries,
        scope=('cfg_b two-output development graph; endpoint is final source-read tail, not global D visibility or CUDA-event time'
            if all(n==2 for n in counts) else
            'cfg_b variable-output development graph; endpoint is final source-read tail, not global D visibility or CUDA-event time'))
    if prefill is not None:result['second_prefill']=prefill
    return result


def stage_reuse_window(intervals, stages, service_cycles, reuse_cycles, wait_overhead=0.):
    """Finite internal-window envelope for the constant reuse subclass.

    For C_j=max(C_(j-1)+c,C_(j-S+1)+Z+w), after the boundary segment,
    each S-1 interval period costs max((S-1)c,Z+w). A regular window starts
    at local j>=S-1. Unknown phase is retained;
    this is not an asymptotic average or a complete mainloop prediction.
    The reduction requires p<=c, g<=c-q, q+lambda<=c and q>=w+h.
    """
    if intervals<1 or stages<2 or service_cycles<=0 or min(reuse_cycles,wait_overhead)<0:
        raise ValueError('positive window/service and nonnegative reuse cost required')
    period=stages-1
    extra=max(0.,reuse_cycles+wait_overhead-period*service_cycles)
    return (intervals*service_cycles+(intervals//period)*extra,
            intervals*service_cycles+((intervals+period-1)//period)*extra)


@lru_cache(None)
def address_box(rows, row_bytes, pitch_bytes, start_mod128):
    """Valid address spans, counted per row; extra coverage is not physical traffic."""
    extra = []
    for width in (32, 128):
        minimum = (row_bytes + width - 1) // width
        count = sum(((start_mod128 + i*pitch_bytes) % width + row_bytes + width-1)//width - minimum
                    for i in range(rows))
        extra.append(count * width / 1024)
    return rows * row_bytes / 1024, *extra


FILL_FIELDS = ['valid_source_KiB', 'B_extra32_KiB', 'B_extra128_KiB',
               'A_fill_KiB', 'B_fill_KiB']


TMA_SOURCE_FIELDS = ['A_valid_KiB', 'A_extra32_KiB', 'A_extra128_KiB',
                     'B_valid_KiB', 'B_extra32_KiB', 'B_extra128_KiB']


SUPPLY_FIELDS = ['valid_source_KiB', 'A_extra32_KiB', 'A_extra128_KiB',
                 'B_extra32_KiB', 'B_extra128_KiB', 'A_fill_KiB', 'B_fill_KiB']


def static_tma_requests(row, setup):
    """Static FP16 row-major R18 requests, per CTA/output and complete Ktile.

    Source vectors sum valid address coverage of issued descriptor boxes, not
    physical L2/HBM traffic; overlapping granules across boxes are counted twice.
    A slices M by cluster-N; B slices K by cluster-M (the selected SW128 layouts).
    Multicast delivers every slice to its mask, including logical-output OOB CTAs.
    Destination valid/fill counts describe the whole operand in each CTA's SMEM.
    Origins use Ktile 0; subsequent Ktiles add 64 to K with unchanged coverage.
    Effective waves retain the old stage/Kt weighting, not measured concurrency.
    No general timing coefficients are assigned; cfg_b's old interface is separate.
    """
    from v08_model import scheduled_work
    if row.get('a_row_rotation',0) or setup.get('a_row_rotation',0):
        raise ValueError('ordinary static source boxes do not support A rotation')
    if row['config'] not in CONFIGS: raise ValueError('static requests support cfg_a/b/c only')
    tm,tn,cm,cn=CONFIGS[row['config']]
    m,n,k=row['m'],row['n'],row['k']
    map_m=row.get('input_map_m',m); map_n=row.get('input_map_n',n)
    lda=row.get('lda',k); ldb=row.get('ldb',n)
    if min(m,n,k)<=0 or k%64: raise ValueError('positive shapes and complete Ktiles required')
    if map_m<m or map_n<n: raise ValueError('R18 input maps must cover the logical input')
    if n%8 or map_n%8 or lda%8 or ldb%8 or lda<k or ldb<map_n:
        raise ValueError('row-major FP16 TMA requires legal extents and 16-byte pitches')
    if map_m>row.get('storage_m',map_m) or map_n>row.get('storage_n',map_n):
        raise ValueError('input map exceeds source storage')
    gx,gy,gz=setup['grid']
    if gx%cm or gy%cn: raise ValueError('grid must contain complete clusters')
    if tuple(setup.get('tile',(tm,tn,64)))!=(tm,tn,64) or tuple(setup.get('cluster',(cm,cn))) not in ((cm,cn),(cm,cn,1)):
        raise ValueError('setup tile/cluster differs from the fixed R18 configuration')
    if setup['stages']<=0: raise ValueError('positive stage count required')
    work=scheduled_work(row['config'],m,n,[gx,gy,gz],row['swizzle'],row.get('raster_order','heuristic'))
    ctas=[]
    for coords in work:
        tiles=[]
        for mi,ni in coords:
            rm,rn=mi%cm,ni%cn
            a_m=mi*tm+rn*(tm//cn); b_k=rm*(64//cm)
            a_rows=max(0,min(tm//cn,map_m-a_m))
            b_cols=max(0,min(tn,map_n-ni*tn))
            a=address_box(a_rows,128,2*lda,2*a_m*lda%128) if a_rows else (0,0,0)
            # B's MN-SW128 layout has a 64-column atom, repeated twice in N.
            # Its descriptor's outer K dimension is truncated by multicast.
            b_boxes=[]
            for offset in (0,64):
                b_n=ni*tn+offset; cols=max(0,min(64,map_n-b_n))
                value=address_box(64//cm,2*cols,2*ldb,(row.get('b_offset_bytes',0)+2*(b_k*ldb+b_n))%128) if cols else (0,0,0)
                b_boxes.append(dict(origin=[b_n,b_k],shape=[64,64//cm],source_KiB=list(value)))
            b=np.sum([box['source_KiB'] for box in b_boxes],axis=0).tolist()
            av=max(0,min(tm,map_m-mi*tm))*64*2/1024
            bv=b_cols*64*2/1024
            tiles.append(dict(coord=[mi,ni],cluster=[mi//cm,ni//cn],cluster_rank=rm+rn*cm,
                A=dict(source_origin=[a_m,0],source_shape=[tm//cn,64],
                    source_KiB=list(a),source_box_KiB=tm//cn*64*2/1024,
                    source_boxes=[dict(origin=[a_m,0],shape=[tm//cn,64],source_KiB=list(a))],
                    multicast_mask=sum(1<<(rm+y*cm) for y in range(cn)),
                    destination_valid_KiB=av,destination_fill_KiB=tm*64*2/1024-av),
                B=dict(source_origin=[ni*tn,b_k],source_shape=[tn,64//cm],
                    source_KiB=list(b),source_box_KiB=tn*(64//cm)*2/1024,
                    source_boxes=b_boxes,
                    multicast_mask=sum(1<<(x+rn*cm) for x in range(cm)),
                    destination_valid_KiB=bv,destination_fill_KiB=tn*64*2/1024-bv)))
        ctas.append(tiles)
    waves=[sum((np.array(tiles[j]['A']['source_KiB']+tiles[j]['B']['source_KiB'])
                for tiles in ctas if len(tiles)>j),np.zeros(6))
           for j in range(max(map(len,ctas)))]
    credit=min(setup['stages'],k//64)/(k//64)
    effective=[(1-credit)*v+credit*(waves[j+1] if j+1<len(waves) else np.zeros(6))
               for j,v in enumerate(waves)]
    return dict(source_fields=TMA_SOURCE_FIELDS,source_coefficients=None,destination_coefficients=None,
                ctas=ctas,source_waves=[v.tolist() for v in waves],
                effective_source_waves=[v.tolist() for v in effective])


def input_slot_state(group, stages):
    """Persistent input-ring state; group counts Ktiles across output tiles.

    These are the phase arguments used to wait, not a sampled barrier phase.
    The first producer waits on phase 1, the first consumer on phase 0.
    A slot may be acquired again only after the prior generation is released.
    """
    if not isinstance(group, int) or not isinstance(stages, int) or group < 0 or stages < 1:
        raise ValueError('nonnegative integer group and positive integer stages required')
    generation, slot = divmod(group, stages)
    return dict(group=group, slot=slot, generation=generation,
                full_wait_phase=generation % 2, empty_wait_phase=(generation + 1) % 2,
                previous_group=group-stages if group >= stages else None)


def supply_work_ledger(row, setup):
    """Software source boxes, multicast deliveries and output work, without prices.

    Source atoms are templates at Ktile 0; every later Ktile adds 64 to K.
    Destination dependencies name the actual issuing CTA and atom, including
    remote multicast slices. Distinct source identities are counted over the
    invocation, without asserting that repeated reads merge or remain cached.
    Payload/valid/fill KiB and address coverage have separate meanings.
    Output subtiles describe the fixed epilogue lattice, not physical traffic
    or a transferred cooperative timing graph. No observation is read.
    """
    if row.get('source_copies', setup.get('source_copies', 1)) != 1 or setup.get('source_copies', 1) != 1:
        raise ValueError('ledger requires one source allocation per operand')
    if row.get('work_swap55_91', 0) or setup.get('work_swap55_91', 0):
        raise ValueError('ledger does not support work exchange')
    for name in ('config', 'm', 'n', 'k', 'swizzle', 'lda', 'ldb', 'input_map_m', 'input_map_n', 'b_offset_bytes', 'raster_order'):
        if name in row and name in setup and row[name] != setup[name]:
            raise ValueError('ledger row/setup mismatch: '+name)
    request = static_tma_requests(row, setup)
    tm, tn, cm, cn = CONFIGS[row['config']]
    kt = row['k'] // 64
    stages = setup['stages']
    peers = {}
    outputs = []
    for cta, tiles in enumerate(request['ctas']):
        for j, tile in enumerate(tiles):
            key = (tuple(tile['cluster']), tile['cluster_rank'])
            if key in peers:
                raise ValueError('logical output tile appears more than once')
            peers[key] = len(outputs)
            valid_m = max(0, min(tm, row['m']-tile['coord'][0]*tm))
            valid_n = max(0, min(tn, row['n']-tile['coord'][1]*tn))
            # r18.cu: cfg_b uses 64x32; cooperative uses 128x32 FP32.
            epi_m = 64 if row['config'] == 'cfg_b' else 128
            outputs.append(dict(cta=cta, j=j, coord=tile['coord'],
                cluster=tile['cluster'], rank=tile['cluster_rank'],
                first_input_state=input_slot_state(j*kt, stages),
                last_input_state=input_slot_state((j+1)*kt-1, stages),
                operands={op: dict(valid_KiB=tile[op]['destination_valid_KiB'],
                    fill_KiB=tile[op]['destination_fill_KiB'], source_atoms=[])
                    for op in ('A', 'B')},
                output=dict(nominal_KiB=tm*tn*4/1024,
                    valid_KiB=valid_m*valid_n*4/1024,
                    subtile_shape=[epi_m, 32], nominal_subtiles=tm*tn//(epi_m*32))))
    atoms = []
    distinct = {}
    for cta, tiles in enumerate(request['ctas']):
        for j, tile in enumerate(tiles):
            for op in ('A', 'B'):
                for box in tile[op]['source_boxes']:
                    stride = [2*row.get('lda', row['k']), 2] if op == 'A' else [2, 2*row.get('ldb', row['n'])]
                    extents = [row.get('input_map_m', row['m']), row['k']] if op == 'A' else [row.get('input_map_n', row['n']), row['k']]
                    offset = 0 if op == 'A' else row.get('b_offset_bytes', 0)
                    identity = (op, tuple(extents), tuple(stride), offset,
                                tuple(box['origin']), tuple(box['shape']))
                    payload = math.prod(box['shape'])*2/1024
                    valid = box['source_KiB'][0]
                    destinations = []
                    for rank in range(cm*cn):
                        if tile[op]['multicast_mask'] & (1 << rank):
                            target = peers[(tuple(tile['cluster']), rank)]
                            outputs[target]['operands'][op]['source_atoms'].append(len(atoms))
                            destinations.append(dict(cta=outputs[target]['cta'], j=outputs[target]['j'], rank=rank))
                    atoms.append(dict(id=len(atoms), cta=cta, j=j, operand=op,
                        origin=box['origin'], shape=box['shape'], extents=extents,
                        stride_bytes=stride, base_offset_bytes=offset,
                        payload_KiB=payload, valid_KiB=valid, fill_KiB=payload-valid,
                        extra32_KiB=box['source_KiB'][1], extra128_KiB=box['source_KiB'][2],
                        multicast_mask=tile[op]['multicast_mask'], destinations=destinations))
                    distinct[identity] = valid
    for out in outputs:
        for op, value in out['operands'].items():
            incoming = [atoms[i] for i in value['source_atoms']]
            if sum(a['valid_KiB'] for a in incoming) != value['valid_KiB'] or sum(a['fill_KiB'] for a in incoming) != value['fill_KiB']:
                raise ValueError('source slices do not conserve destination valid/fill payload')
    totals = dict(source_requests=len(atoms)*kt,
        source_payload_KiB=sum(a['payload_KiB'] for a in atoms)*kt,
        source_valid_KiB=sum(a['valid_KiB'] for a in atoms)*kt,
        source_box_fill_KiB=sum(a['fill_KiB'] for a in atoms)*kt,
        distinct_source_boxes=len(distinct)*kt,
        distinct_valid_source_KiB=sum(distinct.values())*kt,
        destination_atoms=sum(len(a['destinations']) for a in atoms)*kt,
        destination_valid_KiB=sum(v['valid_KiB'] for o in outputs for v in o['operands'].values())*kt,
        destination_fill_KiB=sum(v['fill_KiB'] for o in outputs for v in o['operands'].values())*kt,
        output_tiles=len(outputs),
        output_nominal_KiB=sum(o['output']['nominal_KiB'] for o in outputs),
        output_valid_KiB=sum(o['output']['valid_KiB'] for o in outputs),
        output_nominal_subtiles=sum(o['output']['nominal_subtiles'] for o in outputs))
    if totals['output_valid_KiB'] != row['m']*row['n']*4/1024:
        raise ValueError('output work does not cover the logical D exactly')
    return dict(schema='gh200-supply-work-ledger/1', config=row['config'], ktiles=kt, stages=stages,
        source_atoms=atoms, outputs=outputs, totals=totals,
        legacy_clock_source_field='totals.source_payload_KiB',
        K_origin_increment=[0, 64],
        scope='within one invocation; logical source/delivery/output work only; no timing or physical traffic')


def cfg_b_source_program(row, setup):
    """Full-tile unicast cfg_b source identities for variable pingpong outputs.

    Identity is descriptor plus whole source-box origin/shape, within one call.
    This does not predict issue times, cache residence, source coalescing or a
    price for pitch. Boundary/OOB output needs a separate output graph.
    """
    if row['config'] != 'cfg_b':
        raise ValueError('source program requires cfg_b pingpong')
    ledger = supply_work_ledger(row, setup)
    if any(o['output']['valid_KiB'] != o['output']['nominal_KiB'] for o in ledger['outputs']):
        raise ValueError('source program requires complete logical output tiles')
    kt = ledger['ktiles']
    count = math.prod(setup['grid'])
    coords = [[] for _ in range(count)]
    grouped = defaultdict(list)
    for atom in ledger['source_atoms']:
        if atom['multicast_mask'] != 1 or atom['fill_KiB']:
            raise ValueError('source program requires valid unicast source boxes')
        grouped[atom['cta'], atom['j']].append(atom)
    keys = [[] for _ in range(count)]
    for output in ledger['outputs']:
        cta, j = output['cta'], output['j']
        if j != len(coords[cta]):
            raise ValueError('output ordinals are not contiguous')
        coords[cta].append(output['coord'])
        atoms = grouped[cta, j]
        if [(a['operand'], a['shape'], a['payload_KiB']) for a in atoms] != [
                ('A', [128,64], 16), ('B', [64,64], 8), ('B', [64,64], 8)]:
            raise ValueError('source program requires A16KiB/B0+B1 each8KiB')
        for k in range(kt):
            group = []
            for atom in atoms:
                group.append((atom['operand'], 'fp16', 'SW128', tuple(atom['extents']),
                    tuple(atom['stride_bytes']), atom['base_offset_bytes'],
                    (atom['origin'][0], atom['origin'][1]+64*k), tuple(atom['shape'])))
            keys[cta].append(tuple(group))
    counts = [len(c) for c in coords]
    if not all(counts):
        raise ValueError('source program does not model idle launched CTAs')
    return dict(source_keys=keys, output_counts=counts, cta_output_coords=coords,
        totals=ledger['totals'], ledger_schema=ledger['schema'])


def boundary_fill_features(row,setup):
    """Candidate coupling features from static requests, never from observed work.

    Mask sums count destination amounts (including replication), not traffic.
    Full-cluster coupling is a hypothesis, not proof of a hardware barrier.
    """
    request=static_tma_requests(row,setup);_,_,cm,cn=CONFIGS[row['config']]
    clusters=defaultdict(list)
    for tiles in request['ctas']:
        for tile in tiles:clusters[tuple(tile['cluster'])].append(tile)
    result=[]
    for cta,tiles in enumerate(request['ctas']):
        for j,tile in enumerate(tiles):
            peers=clusters[tuple(tile['cluster'])]
            if len(peers)!=cm*cn:raise ValueError('static logical work cluster is incomplete')
            feature=dict(cta=cta,j=j,T=len(tiles),coord=tile['coord'],phase='first' if j==0 else 'later',
                cluster=tile['cluster'],rank=tile['cluster_rank'],operands={})
            for op in ('A','B'):
                mask=tile[op]['multicast_mask']
                group=[p[op]['destination_fill_KiB'] for p in peers if mask&(1<<p['cluster_rank'])]
                whole=[p[op]['destination_fill_KiB'] for p in peers]
                feature['operands'][op]=dict(own=tile[op]['destination_fill_KiB'],mask=mask,
                    mask_max=max(group),mask_sum=sum(group),cluster_max=max(whole),cluster_sum=sum(whole),cluster_any=float(any(v>0 for v in whole)))
            result.append(feature)
    return result



def supply_request_rows(row, setup):
    """Pure per-CTA/output prediction rows, without timing observations."""
    request=static_tma_requests(row,setup); result=[]
    tm,tn,_,_=CONFIGS[row['config']]
    for c,tiles in enumerate(request['ctas']):
        for j,tile in enumerate(tiles):
            source=request['effective_source_waves'][j]
            X=[source[0]+source[3],source[1],source[2],source[4],source[5],
               tile['A']['destination_fill_KiB'],tile['B']['destination_fill_KiB']]
            result.append(dict(case=row.get('id','prediction'),config=row['config'],
                cta=c,coord=tile['coord'],j=j,T=len(tiles),kt=row['k']//64,
                compute_cycles=2*tm*tn*64/4096,X=X))
            if 'gpu_uuid' in setup:result[-1]['gpu_uuid']=setup['gpu_uuid']
    return result


def supply_predict(rows, model, frequency_ghz=None):
    """Pure full-L prediction; NS requires caller's f, never reads L observations.

    Return (values, supported). Unsupported requests leave the calibrated feature
    or active-Jacobian span; their numeric values must not be scored or frozen.
    """
    if not rows:return np.array([]),np.array([],dtype=bool)
    x=np.array([r['X'] for r in rows]);kt=np.array([r['kt'] for r in rows])
    if any(r['config']!=model['config'] for r in rows): raise ValueError('configuration mismatch')
    if any(r.get('gpu_uuid',model['gpu_uuid'])!=model['gpu_uuid'] for r in rows): raise ValueError('card mismatch')
    floor=np.array([r['compute_cycles'] for r in rows])
    if model['unit']=='ns':
        f=np.broadcast_to(np.asarray(frequency_ghz,dtype=float),(len(rows),))
        if not np.all(np.isfinite(f)&(f>0)): raise ValueError('explicit positive predicted frequency required')
        floor=floor/f
    z=x@np.array(model['projection']);p=np.array(model['parameters']);service=z@p[1:]
    value=p[0]+kt*np.maximum(floor,service)
    raw=x/np.array(model['request_scale']);basis=np.array(model['request_basis']).reshape(-1,7)
    jac=np.column_stack([np.ones(len(rows)),kt[:,None]*z*(service>floor)[:,None]])/np.array(model['jacobian_scale'])
    jb=np.array(model['jacobian_basis']).reshape(-1,len(p))
    supported=(np.linalg.norm(raw-(raw@basis.T)@basis,axis=1)<=1e-7*np.maximum(1,np.linalg.norm(raw,axis=1)))
    supported &= np.linalg.norm(jac-(jac@jb.T)@jb,axis=1)<=1e-7*np.maximum(1,np.linalg.norm(jac,axis=1))
    if 'phase' in model:supported &= np.array([(r['j']==0)==(model['phase']=='first') for r in rows])
    return value,supported


def supply_predict_bounds(rows, model, frequency_ghz, identification):
    """Pure conditional L bounds, with explicit f and calibrated coverage fractions."""
    from scipy.optimize import linprog

    if not rows:return []
    supply_predict(rows,model,frequency_ghz)  # Check card/configuration and explicit f.
    p=np.array(model['parameters']);projection=np.array(model['projection'])
    basis=np.array(model['request_basis']).reshape(-1,7);cache={};result=[]
    frequencies=np.broadcast_to(np.asarray(frequency_ghz,dtype=float),(len(rows),)) if model['unit']=='ns' else np.ones(len(rows))
    constraints=identification['constraints']
    for r,f in zip(rows,frequencies):
        if 'phase' in model and (r['j']==0)!=(model['phase']=='first'):
            result.append(dict(status='unsupported phase',minimum=None,maximum=None));continue
        raw=np.array(r['X'])/np.array(model['request_scale'])
        if np.linalg.norm(raw-(raw@basis.T)@basis)>1e-7*max(1,np.linalg.norm(raw)):
            result.append(dict(status='unsupported feature span',minimum=None,maximum=None));continue
        z=np.array(r['X'])@projection;floor=r['compute_cycles']/f;kt=r['kt']
        key=(*z,kt,floor)
        if key not in cache:
            # Minimise the epigraph of max; its maximum is the larger of two
            # linear maxima. No nearly-infeasible branch-boundary LP is needed.
            args=dict(A_eq=[row+[0] for row in constraints['A_eq']],b_eq=constraints['b_eq'],
                A_ub=[row+[0] for row in constraints['A_ub']]+[np.r_[np.zeros(len(p)),-1].tolist(),np.r_[0,z,-1].tolist()],
                b_ub=constraints['b_ub']+[-floor,0])
            fit=linprog(np.r_[1,np.zeros(len(p)-1),kt],**args,bounds=(0,None),method='highs',
                        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            if fit.status!=0:raise ValueError('prediction lower bound failed: '+fit.message)
            low=float(fit.fun);highs=[]
            for service_branch in (False,True):
                objective=np.r_[1,kt*z if service_branch else np.zeros(len(p)-1)]
                fit=linprog(-objective,**constraints,bounds=(0,None),method='highs',
                            options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                if fit.status not in (0,3):raise ValueError('prediction upper bound failed: '+fit.message)
                highs.append(np.inf if fit.status==3 else float(-fit.fun+(0 if service_branch else kt*floor)))
            representative=p[0]+kt*max(floor,z@p[1:])
            low=min(low,float(representative));high=max(*highs,float(representative))
            cache[key]=dict(status='identified prediction' if np.isfinite(high) and high-low<=1e-7*max(1,abs(high)) else 'bounded nonunique prediction',
                minimum=float(low) if np.isfinite(low) else None,maximum=float(high) if np.isfinite(high) else None)
        result.append(cache[key])
    return result


def supply_pitch_support(row, setup, model, frequency_ghz):
    """Static span support by pitch residue; not measured timing qualification."""
    result=[]
    for operand in ('A','B'):
        for residue in range(0,128,16):
            case=dict(row)
            case['lda']=row['k'];case['ldb']=((row.get('input_map_n',row['n'])+63)//64)*64
            case['lda' if operand=='A' else 'ldb']+=residue//2
            requests=[r for r in supply_request_rows(case,setup)
                      if (r['j']==0)==(model.get('phase','later')=='first')]
            _,support=supply_predict(requests,model,frequency_ghz)
            result.append(dict(operand=operand,pitch_mod128=residue,windows=len(requests),
                supported=int(sum(support)),status='supported by feature span' if len(requests) and all(support) else 'unsupported or no matching-phase windows'))
    return result


def fill_cta_features(row, setup):
    """Return [CTA][output tile][FILL_FIELDS] using only row/setup.

    CTA/tile order follows scheduled_work. The first three fields are effective
    software-wave demands; the last two retain each tile's local zero-fill demand.
    """
    from v08_model import scheduled_work
    if row['config']!='cfg_b': raise ValueError('this fill component is calibrated for cfg_b only')
    work=scheduled_work('cfg_b',row['m'],row['n'],setup['grid'],row['swizzle'],row.get('raster_order','heuristic'))
    kt=(row['k']+63)//64
    if row['k']%64: raise ValueError('this fill comparison uses complete Ktiles')
    map_m=row.get('input_map_m',row['m']); map_n=row.get('input_map_n',row['n'])
    requests=[]
    for coords in work:
        values=[]
        for mi,ni in coords:
            am=max(0,min(128,map_m-mi*128)); bn=max(0,min(128,map_n-ni*128))
            a=address_box(am,128,2*row['lda'],mi*256*row['lda']%128) if am else (0,0,0)
            b=address_box(64,2*bn,2*row['ldb'],ni*256%128) if bn else (0,0,0)
            if a[1] or a[2]: raise ValueError('A pitch is aligned in this same-card batch')
            values.append(np.array([a[0]+b[0],b[1],b[2],16-a[0],16-b[0]]))
        requests.append(values)
    waves=[sum((v[j][:3] for v in requests if len(v)>j),np.zeros(3))
           for j in range(max(map(len,requests)))]
    credit=min(setup['stages'],kt)/kt
    effective=[(1-credit)*v+credit*(waves[j+1] if j+1<len(waves) else np.zeros(3))
               for j,v in enumerate(waves)]
    return [[np.r_[effective[j],v[3:]] for j,v in enumerate(values)]
            for values in requests]


def fill_window_cycles(request_features, kt, parameters, unit='cycle', frequency_ghz=None):
    """Development CTA L replacement; NS service requires an explicit predicted f."""
    service=float(np.dot(request_features,parameters[1:]))
    if unit=='cycle': return parameters[0]+kt*max(512,service)
    if frequency_ghz is None or frequency_ghz<=0: raise ValueError('NS form requires an explicit positive frequency assumption')
    return frequency_ghz*parameters[0]+kt*max(512,frequency_ghz*service)

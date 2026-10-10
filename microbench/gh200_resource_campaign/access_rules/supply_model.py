"""Static TMA requests and calibrated supply prediction, without fitting or IO.

All coefficients and conditional LP constraints are explicit caller inputs.
The request spans are descriptor coverage, not physical memory traffic.
"""
from collections import defaultdict
from functools import lru_cache

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
                           observed_second_entry=None):
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
    """
    values=(producer_interval,ready_latency,commit_lead,completion_lag,
            service_cycles,issue_floor,wait_overhead,loop_advance,
            return_overhead,reuse_delay,handoff_delay)
    if kt<2 or stages<2 or min(values)<0 or service_cycles<=0:
        raise ValueError('admission diagnostic requires Ktile/stages>=2 and nonnegative times')
    if issue_floor>service_cycles:
        raise ValueError('issue floor must not exceed per-tile service')
    if observed_second_entry is not None:
        entry_h,entry_b=observed_second_entry
        if not 0<=entry_h<=entry_b:
            raise ValueError('observed entry requires 0 <= H gap <= B gap')
    issued=[];ready=[];available=[];starts=[];frontier=[];commits=[];done=[];wait=[]
    released=[0.]*(2*kt);returns=[];ends=[];next_consumer=0.
    for i in range(2*kt):
        issue=issued[-1]+producer_interval if issued else 0.
        if i>=stages:issue=max(issue,released[i-stages]+reuse_delay)
        issued.append(issue);ready.append(issue+ready_latency)
        if i==kt and observed_second_entry is not None:
            next_consumer=returns[0]+entry_h
            start=returns[0]+entry_b
            if ready[-1]>start+1e-8:
                raise ValueError('observed entry precedes modeled readiness')
        else:start=max(ready[-1],next_consumer)
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
    return dict(windows=[ends[j]-starts[j*kt] for j in range(2)],
        issued=issued,ready=ready,consumer_available=available,
        first_mma_or_commit=starts,service_frontier=frontier,commits=commits,
        completed=done,wait1_return=wait,released=released,
        mma_returns=returns,main_ends=ends,slot=[i%stages for i in range(2*kt)])


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

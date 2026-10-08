#!/usr/bin/env python3
"""Independent numeric replay and actual CTA tile/event analysis for R18/R19."""
import argparse,gzip,json,math,statistics
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import v06_model as model
import v06_run as common


@lru_cache(None)
def reference(row,col,k,seed=17):
    total=sum(((row*7+t*13+seed*3)%17-8)*((t*5+col*11+seed*5)%17-8) for t in range(17))
    remainder=sum(((row*7+t*13+seed*3)%17-8)*((t*5+col*11+seed*5)%17-8) for t in range(k%17))
    return (total*(k//17)+remainder)/1024


def random_references(indices,m,n,k,seed):
    """Regenerate logical FP16 inputs independently of the device-side output checker."""
    import numpy as np

    def values(rows,cols,is_a):
        x=np.uint32(seed)^np.uint32(0xa511e9b3 if is_a else 0x63d83595)
        x=x^(rows*np.uint32(0x9e3779b9))^(cols*np.uint32(0x85ebca6b))
        x=(x^(x>>16))*np.uint32(0x7feb352d)
        x=(x^(x>>15))*np.uint32(0x846ca68b)
        x=x^(x>>16)
        return ((x>>8).astype(np.float32)*np.float32(2**-23)-np.float32(1)).astype(np.float16).astype(np.float64)

    positions=np.asarray(indices,dtype=np.int64)
    rows=(positions//n).astype(np.uint32);cols=(positions%n).astype(np.uint32)
    inner=np.arange(k,dtype=np.uint32)[None,:]
    expected=[];tolerances=[]
    for start in range(0,len(indices),32):
        a=values(rows[start:start+32,None],inner,True)
        b=values(inner,cols[start:start+32,None],False)
        products=a*b
        expected.extend(products.sum(axis=1).tolist())
        tolerances.extend((2**-20+2**-21*np.abs(products).sum(axis=1)).tolist())
    return expected,tolerances


def replay(root,record,row):
    raw=root/record['raw']
    if common.sha(raw)!=record['raw_sha256']:raise ValueError('raw process identity changed')
    with gzip.open(raw,'rt') as stream:events={ev['event']:ev for line in stream if line.strip() for ev in [json.loads(line)]}
    setup,call,check=(events[k] for k in ['setup','call','check'])
    for key in ['m','n','k','lda','ldb','ldd','zero_m','zero_n','storage_m','storage_n','swizzle']:
        if setup[key]!=row[key]:raise ValueError('setup coordinate changed: '+key)
    mode=row.get('input_mode','dyadic');seed=row.get('seed',17)
    if setup.get('input_mode','dyadic')!=mode or setup.get('seed',17)!=seed:
        raise ValueError('input mode or seed changed')
    if setup.get('requested_sm_count',0)!=row.get('sm_count',0):raise ValueError('requested SM count changed')
    if check['status']!='ok' or check['padding_errors'] or len(check['checked_values'])!=4096:
        raise ValueError('incorrect or incomplete GEMM check')
    if len(set(check['checked_indices']))!=4096:raise ValueError('duplicate sampled values')
    if mode=='random':
        if setup['check_atol']!=2**-20 or setup['check_sum_abs_rtol']!=2**-21:
            raise ValueError('random input tolerance changed')
        expected_values,tolerances=random_references(check['checked_indices'],row['m'],row['n'],row['k'],seed)
    elif mode not in ['dyadic','zero']:
        raise ValueError('unknown input mode')
    zero_count=0
    for q,(index,value) in enumerate(zip(check['checked_indices'],check['checked_values'])):
        i,j=divmod(index,row['n'])
        if not 0<=i<row['m']:raise ValueError('invalid check position')
        zero=mode=='zero' or (row['zero_m']>=0 and i>=row['zero_m']) or (row['zero_n']>=0 and j>=row['zero_n'])
        zero_count+=zero
        expected=0 if zero else expected_values[q] if mode=='random' else reference(i%17,j%17,row['k'],seed)
        tolerance=(2**-20 if zero else tolerances[q]) if mode=='random' else 0
        if value is None or not math.isfinite(value) or abs(value-expected)>tolerance:
            raise ValueError('wrong GEMM output at '+str((i,j)))
    if row['kind']=='explicit_zero' and zero_count<1:raise ValueError('zero panel unchecked')
    warm=call['warmup_us'];warm_cv=statistics.pstdev(warm[-5:])/statistics.mean(warm[-5:])
    if not 8<=len(warm)<=30 or warm_cv>.02:raise ValueError('warmup did not converge')
    result=dict(case=row['id'],variant=record['variant'],trial=record['trial'],elapsed_us=call['elapsed_us'],checked_values=4096,zero_values=zero_count,ctas=[])
    if record['variant']=='plain':return result
    words=call['trace'];ctas=math.prod(setup['grid']);width=16+2*64*6
    if len(words)!=ctas*width:raise ValueError('wrong trace size')
    schedule='pingpong' if row['config']=='cfg_b' else 'cooperative'
    seen=[]
    light=setup['trace_version']=='r18-light-events'
    inferred_work=model.scheduled_work(row['config'],row['m'],row['n'],setup['grid']) if light else None
    for c in range(ctas):
        head=words[c*width:(c+1)*width]
        if head[10] or not head[0] or not head[2]:raise ValueError('missing/overflow CTA trace')
        roles=[];coords=[]
        for role in range(2):
            end,ns,count=head[4+3*role:7+3*role]
            if count>64 or head[11+role]!=head[2]:raise ValueError('invalid count/SM-local clock boundary')
            tiles=[];work=[]
            for j in range(count):
                x=16+(role*64+j)*6;tile=head[x:x+4];mi,ni=head[x+4:x+6]
                if light:
                    if schedule!='cooperative' or count!=len(inferred_work[c]):raise ValueError('light work-count mismatch')
                    mi,ni=(q+1 for q in inferred_work[c][j])
                if not all(tile) or min(mi,ni)<=0 or not tile[0]<=tile[1]<=tile[2]<=tile[3]<=end:
                    raise ValueError('invalid tile events/coordinates')
                tiles.append(tile);work.append((mi-1,ni-1))
            roles.append(dict(final_c=end,final_ns=ns,tiles=tiles));coords.append(work)
        if schedule=='cooperative':
            if coords[0]!=coords[1]:raise ValueError('consumer work mismatch')
            work=coords[0]
        else:
            work=[coords[j%2][j//2] for j in range(len(coords[0])+len(coords[1]))]
        entry=dict(entry_c=head[0],entry_ns=head[1],smid=head[2]-1,prod_c=head[3],roles=roles)
        tiles,end_c,end_ns=model.cta_timeline(entry,schedule)
        result['ctas'].append(dict(cta=c,entry_c=head[0],entry_ns=head[1],sm=head[2]-1,prod_c=head[3],
                                  end_c=end_c,end_ns=end_ns,tiles=tiles,work=work))
        seen+=work
    if len(seen)!=len(set(seen)):raise ValueError('duplicated logical work tiles')
    tm=256 if row['config'].startswith('cfg_c') else 128;tn=128
    required={(i,j) for i in range(model.cdiv(row['m'],tm)) for j in range(model.cdiv(row['n'],tn))}
    if not required<=set(seen):raise ValueError('actual scheduler omitted real tile')
    return result


def tile_class(row,mi,ni):
    axis=0 if row['config'].startswith('cfg_a') else 1
    q=(mi,ni)[axis];extent=row['m'] if axis==0 else row['n'];tile=128
    if q*tile>=extent:return 'oob'
    zero=row['zero_m'] if axis==0 else row['zero_n']
    if zero>=0 and q*tile>=zero:return 'explicit_zero'
    if not row['config'].endswith('1') and (q^1)*tile>=extent:return 'partner'
    if zero>=0 and (q^1)*tile>=zero:return 'zero_partner'
    if q==extent//tile and extent%tile:return 'partial'
    return 'other'


def analyze(root,output=None):
    common.verify(root)
    for name,h in json.loads((root/'build/sass_hashes.json').read_text()).items():
        if common.sha(root/'build'/name)!=h:raise ValueError('SASS changed')
    dest=output or root/'analysis';dest.mkdir(parents=True,exist_ok=True)
    rows=json.loads((root/'cases.json').read_text());results=[];classes=[];tails=[];paired=[];all_records={};increments=[];rejected=[]
    for row in rows:
        recs=[]
        for path in sorted((root/'samples'/row['id']).glob('*.json')):
            record=json.loads(path.read_text())
            if record['returncode']:
                if 'last five warmups CV exceeds 2%' in record['stderr']:
                    rejected.append(dict(path=str(path.relative_to(root)),reason='warmup_not_converged_no_output_check',record=record));continue
                raise ValueError('failed process remains unqualified')
            recs.append(replay(root,record,row))
        plain=[r['elapsed_us'] for r in recs if r['variant']=='plain'];trace=[r['elapsed_us'] for r in recs if r['variant']=='stamped']
        if not plain or len(plain)!=len(trace):raise ValueError('missing paired plain/trace')
        all_records[row['id']]=recs
        disturbance=statistics.median(trace)/statistics.median(plain)-1
        trace_cv=statistics.pstdev(trace)/statistics.mean(trace)
        cv=statistics.pstdev(plain)/statistics.mean(plain)
        qualified=len(plain)>=10 and cv<=.05
        results.append(dict(**row,plain_us=statistics.median(plain),plain_cv=cv,processes=len(plain),
                            perturbation=disturbance,trace_cv=trace_cv,plain_qualified=qualified,trace_qualified=qualified and abs(disturbance)<=.05 and trace_cv<=.05))
        per_class=defaultdict(list)
        for rec in recs:
            if rec['variant']!='stamped':continue
            active=[c for c in rec['ctas'] if c['tiles']]
            start=min(c['entry_ns'] for c in rec['ctas']);end=max(c['end_ns'] for c in active)
            critical=max(active,key=lambda c:c['end_ns'])
            durations=[c['end_c']-c['entry_c'] for c in active]
            tails.append(dict(case=row['id'],trial=rec['trial'],critical_cta=critical['cta'],critical_sm=critical['sm'],
                critical_work=critical['work'],sm_distribution=dict(Counter(c['sm'] for c in rec['ctas'])),critical_cycles=critical['end_c']-critical['entry_c'],median_cta_cycles=statistics.median(durations),
                max_cta_cycles=max(durations),critical_entry_skew_ns=critical['entry_ns']-start,
                entry_spread_ns=max(c['entry_ns'] for c in rec['ctas'])-start,envelope_ns=end-start,
                tiles_count_histogram=dict(Counter(len(c['tiles']) for c in rec['ctas'])),
                critical_mainloop_cycles=[t[1]-t[0] for t in critical['tiles']],
                critical_epilogue_cycles=[t[3]-t[2] for t in critical['tiles']],
                critical_permit_cycles=[t[2]-t[1] for t in critical['tiles']],
                critical_handoff_cycles=[b[0]-a[1] for a,b in zip(critical['tiles'],critical['tiles'][1:])]))
            groups=defaultdict(list)
            for c in active:
                for j,((mi,ni),t) in enumerate(zip(c['work'],c['tiles'])):
                    cls=tile_class(row,mi,ni) if row['config']!='cfg_b' else 'all'
                    groups[(j,cls)].append(t[1]-t[0])
            for key,vals in groups.items():per_class[key].append(statistics.median(vals))
        for (j,cls),vals in sorted(per_class.items()):
            classes.append(dict(case=row['id'],round=j,cls=cls,mainloop_cycles=statistics.median(vals),
                                stddev=statistics.pstdev(vals),processes=len(vals),qualified=qualified and abs(disturbance)<=.05 and trace_cv<=.05))
    for row in results:
        if row['kind'] not in ['oob','explicit_zero'] or row['config'].endswith('1'):continue
        base=next(x for x in results if x['group']==row['group'] and x['kind']=='ordinary' and x['config']==row['config'])
        paired.append(dict(case=row['id'],baseline=base['id'],delta_plain_us=row['plain_us']-base['plain_us'],
                           relative_plain=row['plain_us']/base['plain_us']-1,
                           qualified=row['plain_qualified'] and base['plain_qualified']))
    for row in results:
        if row['kind'] not in ['oob','explicit_zero','partial_even','partial_odd'] or row['config'].endswith('1'):continue
        base=next(x for x in results if x['group']==row['group'] and x['kind']=='ordinary' and x['config']==row['config'])
        controls={r['trial']:r for r in all_records[base['id']] if r['variant']=='stamped'}
        deltas=defaultdict(list)
        for rec in all_records[row['id']]:
            if rec['variant']!='stamped':continue
            control=controls[rec['trial']]
            if [c['work'] for c in rec['ctas']]!=[c['work'] for c in control['ctas']]:
                raise ValueError('boundary comparison changed physical tile list/CTA assignment')
            current=defaultdict(list)
            # Boundary columns/rows reached in each round; distinguish affected
            # neighbors from other CTAs, instead of pooling every valid tile.
            affected=defaultdict(set)
            for c in rec['ctas']:
                for j,(mi,ni) in enumerate(c['work']):
                    if tile_class(row,mi,ni)!='other':affected[j].add(ni if row['config']=='cfg_a' else mi)
            for c,b in zip(rec['ctas'],control['ctas']):
                for j,((mi,ni),t,bt) in enumerate(zip(c['work'],c['tiles'],b['tiles'])):
                    cls=tile_class(row,mi,ni)
                    transverse=ni if row['config']=='cfg_a' else mi
                    if cls=='other' and transverse in affected[j]:cls='same_column' if row['config']=='cfg_a' else 'same_row'
                    current[j,cls].append((t[1]-t[0])-(bt[1]-bt[0]))
            for key,values in current.items():deltas[key].append(statistics.median(values))
        for (j,cls),values in sorted(deltas.items()):
            increments.append(dict(case=row['id'],baseline=base['id'],round=j,cls=cls,k=row['k'],
             delta_cycles=statistics.median(values),stddev=statistics.pstdev(values),process_deltas=values,
             qualified=row['trace_qualified'] and base['trace_qualified'],
             scope='same physical coordinates/CTA assignment; per-process paired difference'))
    result=dict(conditions=results,rejected_processes=rejected,checked_values=sum(r['processes']*2*4096 for r in results),
                plain_qualified=sum(r['plain_qualified'] for r in results),trace_qualified=sum(r['trace_qualified'] for r in results),
                paired_complete_time=paired,analyzer_sha256=common.sha(Path(__file__)),scope='full traces record actual CUTLASS coordinates; light R18 traces infer unchanged static mapping checked against full cohort; traces failing 5% perturbation do not yield cycle parameters')
    for name,data in [('rules.json',result),('classes.json',classes),('tails.json',tails),('paired-tile-increments.json',increments)]:common.write_json(dest/name,data)
    print('conditions',len(results),'plain qualified',result['plain_qualified'],'trace qualified',result['trace_qualified'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();analyze(a.input.resolve(),a.output)

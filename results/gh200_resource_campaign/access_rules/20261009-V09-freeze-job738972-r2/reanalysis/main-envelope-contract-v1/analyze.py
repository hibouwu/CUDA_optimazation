#!/usr/bin/env python3
"""Exact piecewise-affine interpretation of frozen V09 r2 envelopes in x=1/f."""
import argparse
from collections import Counter
from fractions import Fraction as Q
import gzip,hashlib,json
from pathlib import Path
import sys

LOWER=Q(50,99)
CUTS=set()


def exact(value): return Q(str(value))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())


def compress(pieces):
    result=[]
    for lo,hi,a,b in pieces:
        CUTS.add(lo)
        if hi is not None:CUTS.add(hi)
        if result and result[-1][1]==lo and result[-1][2:]==(a,b):
            old=result.pop();result.append((old[0],hi,a,b))
        else:result.append((lo,hi,a,b))
    return result


def constant(value):return [(LOWER,None,Q(0),exact(value))]
def times(curve,value):
    value=exact(value)
    return compress([(lo,hi,a*value,b*value) for lo,hi,a,b in curve])


def combine(left,right,operation):
    i=j=0;pieces=[]
    while i<len(left) and j<len(right):
        l,h,a,b=left[i];ll,hh,aa,bb=right[j]
        lo=max(l,ll)
        hi=hh if h is None else h if hh is None else min(h,hh)
        stops=[lo,hi]
        if operation!='add' and a!=aa:
            crossing=(bb-b)/(a-aa)
            if crossing>lo and (hi is None or crossing<hi):stops=[lo,crossing,hi]
        for start,end in zip(stops,stops[1:]):
            if operation=='add':slope,offset=a+aa,b+bb
            else:
                probe=start+1 if end is None else (start+end)/2
                first=(a-aa)*probe+b-bb>=0
                if operation=='min':first=not first
                slope,offset=(a,b) if first else (aa,bb)
            pieces.append((start,end,slope,offset))
        if h==hi:i+=1
        if hh==hi:j+=1
    return compress(pieces)


def plus(a,b):return combine(a,b,'add')
def minus(a,b):return plus(a,times(b,-1))
def maximum(a,b):return combine(a,b,'max')
def minimum(a,b):return combine(a,b,'min')
def add_constant(a,b):return plus(a,constant(b))


def epilogue(p,start,a,b):
    # E plus lost progress at rate r: overlap duration is limited by both
    # the competing mainloop and the remaining epilogue work divided by r.
    zero=constant(0)
    before=maximum(minus(a,start),zero)
    remaining=maximum(minus(constant(p['E']),before),zero)
    overlap=maximum(minus(b,maximum(start,a)),zero)
    duration=minimum(overlap,times(remaining,1/p['r']))
    return add_constant(plus(start,times(duration,1-p['r'])),p['E'])


def cta_curve(group):
    p={k:Q(v) for k,v in group['params'].items()}
    loops=[]
    variable=[(LOWER,None,Q(1),Q(0))]
    for intercept,kt,compute,service in group['loops']:
        loops.append(add_constant(times(maximum(times(variable,Q(compute)),constant(Q(service))),Q(kt)),Q(intercept)))
    count=len(loops)
    if not count:return constant(p['P0'])
    if group['schedule']=='cooperative':
        end=constant(p['P0']+p['S'])
        for j,loop in enumerate(loops):
            if j:end=add_constant(end,p['h'])
            end=add_constant(plus(end,loop),p['w']+Q(group['epilogues'][j]))
    else:
        main=[];done=[]
        for j,loop in enumerate(loops):
            first=constant(p['P0']+p['S']) if not j else add_constant(main[j-1],p['gm'])
            if j>=2:first=maximum(first,add_constant(done[j-2],p['h']))
            main.append(plus(first,loop))
            permit=add_constant(main[j],p['w'])
            if j:permit=maximum(permit,add_constant(done[j-1],p['we']))
            if j+1<count:
                other=add_constant(main[j],p['gm'])
                if j:other=maximum(other,add_constant(done[j-1],p['h']))
                done.append(epilogue(p,permit,other,plus(other,loops[j+1])))
            else:done.append(add_constant(permit,p['E']))
        end=done[0]
        for value in done[1:]:end=maximum(end,value)
    end=add_constant(end,p['Etail'])
    return add_constant(times(end,1+p['xk']),p['x0']+p['x1']*count)


def encode(curve):return [[str(lo),None if hi is None else str(hi),str(a),str(b)] for lo,hi,a,b in curve]
def value(curve,x):
    return next(a*x+b for lo,hi,a,b in curve if lo<=x and (hi is None or x<=hi))


def case_groups(model,row,setup,cal):
    cfg=row['config'];spec=model.events.base.CONFIGS[cfg];tm,tn,tk=spec['tile']
    work=model.events.scheduled_work(cfg,row['m'],row['n'],setup['grid'],row['swizzle'])
    requests=model.supply.supply_request_rows(row,setup)
    per_cta=[[] for _ in work]
    for request in requests:
        fitted=cal['first_supply'][cfg] if request['j']==0 else cal['supply'][cfg]
        assert request['gpu_uuid']==fitted['gpu_uuid'] and fitted['unit']=='ns'
        X=[exact(v) for v in request['X']]
        z=[sum(x*exact(weight[j]) for x,weight in zip(X,fitted['projection'])) for j in range(len(fitted['parameters'])-1)]
        service=sum(v*exact(coefficient) for v,coefficient in zip(z,fitted['parameters'][1:]))
        per_cta[request['cta']].append([str(exact(fitted['parameters'][0])),str(request['kt']),str(exact(request['compute_cycles'])),str(service)])
    fractions=[[Q(max(0,min(tm,row['m']-mi*tm))*max(0,min(tn,row['n']-ni*tn)),tm*tn) for mi,ni in cta] for cta in work]
    first_bytes=sum(f[0]*tm*tn*4 for f in fractions if f)
    single=max(map(len,work))==1
    result=[];identity={};indices=[]
    for loops,valid in zip(per_cta,fractions):
        params={k:exact(v) for k,v in cal['events_ns'][cfg].items() if k!='rho'}
        epilogues=[];rule=cal['output_rules'].get(cfg)
        if rule and loops:
            assert spec['schedule']=='cooperative'
            if single:
                first=rule['E0_single'];e0=max(exact(first['floor_ns']),exact(first['beta_ns_per_MiB'])*first_bytes/2**20)+exact(first['R_ns'])
                params['Etail']=exact(rule['Etail_single_mean_ns'])
            else:
                e0=exact(rule['E0_multi']['effective_ns'])
                params['Etail']=(1-valid[-1])*exact(rule['Etail_oob_median_ns'])+valid[-1]*exact(rule['v09_Etail_multi_median_ns'])
            middle=rule['E_middle_ns']
            if middle is None:middle=rule['middle_padding_proxy']['ns']
            epilogues=[str(e0)]
            for j,fraction in enumerate(valid[1:],1):
                key='Elast' if j==len(valid)-1 else 'E_middle'
                full=rule['Elast_ns'] if key=='Elast' else middle
                epilogues.append(str((1-fraction)*exact(rule[key+'_oob_ns'])+fraction*exact(full)))
        elif spec['schedule']=='cooperative':
            epilogues=[str(params['E0'] if j==0 else params['Elast'] if j==len(loops)-1 else params['E']) for j in range(len(loops))]
        group=dict(schedule=spec['schedule'],params={k:str(v) for k,v in params.items()},loops=loops,epilogues=epilogues)
        key=json.dumps(group,sort_keys=True)
        if key not in identity:identity[key]=len(result);result.append(group)
        indices.append(identity[key])
    return result,indices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();frozen=a.frozen.resolve();out=a.output.resolve();out.mkdir(exist_ok=True)
    if (out/'envelope.json.gz').exists():raise ValueError('new output required')
    provenance={};approved=read(frozen/'frozen/manifest.json')['files']
    assert sha(frozen/'frozen/manifest.json')=='09da85d124eb697a95c4695bd05281b07c37877890d0eca18afbdd2cf213f2ab'
    assert sha(frozen/'frozen/predictions.json')=='f4df39e00cf1d2ab61947c911e699bd5bea4e533609d93f726de43077515d81e'
    for name in ('source_hashes.json','frozen/calibration.json','cases.json','freeze-setup.json','frozen/predictions.json'):
        path=frozen/name;assert sha(path)==approved[name];provenance[str(path)]=sha(path)
    provenance[str(frozen/'frozen/manifest.json')]=sha(frozen/'frozen/manifest.json')
    sys.path.insert(0,str(frozen/'source'));import v09_model as model
    cal=read(frozen/'frozen/calibration.json');predictions=read(frozen/'frozen/predictions.json')['predictions']
    setups={s['case']:s['setup'] for s in read(frozen/'freeze-setup.json')}
    reports=[];checks=0;largest_difference=0.
    global LOWER
    LOWER=1/exact(cal['clock']['coefficients'][0])
    for row in read(frozen/'cases.json'):
        if row['set']!='heldout':continue
        groups,indices=case_groups(model,row,setups[row['id']],cal)
        CUTS.clear();curves=[cta_curve(group) for group in groups]
        envelope=curves[0]
        for curve in curves[1:]:envelope=maximum(envelope,curve)
        cuts=sorted(CUTS|{LOWER})
        violations=[encode([p])[0] for p in envelope if p[2]<0]
        for before,after in zip(envelope,envelope[1:]):assert before[1]==after[0] and before[2]*before[1]+before[3]==after[2]*after[0]+after[3]
        numeric=[]
        for frequency in (.05,.1,.2,.4,.6,.8,1.,1.2,1.4,1.6,1.8,1.98):
            fixture=dict(cal,clock=dict(cal['clock'],coefficients=[frequency]+[0.]*6))
            observed=model.predict_components(row,setups[row['id']],fixture)['window_us']
            exact_ns=value(envelope,1/exact(frequency));difference=abs(float(exact_ns)/1000/observed-1)
            largest_difference=max(largest_difference,difference);assert difference<2e-12
            numeric.append(dict(frequency_ghz=frequency,numeric_us=observed,rational_ns=str(exact_ns),relative_difference=difference));checks+=1
        report=dict(case=row['id'],config=row['config'],frozen_status=predictions[row['id']]['status'],
            lower_x=str(LOWER),cta_group_indices=indices,groups=groups,cta_curves=[encode(c) for c in curves],
            construction_partition=[str(x) for x in cuts],envelope=encode(envelope),negative_slope_pieces=violations,
            positive_at_left=value(envelope,LOWER)>0,positive_tail_slope=envelope[-1][2]>0,numeric_checks=numeric)
        reports.append(report)
        print(json.dumps(dict(case=row['id'],groups=len(groups),pieces=len(envelope),partition=len(cuts),negative_slopes=len(violations))),flush=True)
    source_manifest=read(frozen/'source_hashes.json')
    for module in list(sys.modules.values()):
        path=Path(getattr(module,'__file__','') or '/')
        if path.is_relative_to(frozen/'source') and path.suffix=='.py':
            assert sha(path)==source_manifest[str(path.relative_to(frozen))];provenance[str(path)]=sha(path)
    result=dict(schema='gh200-r09-envelope-contract/1',
        interpretation='Exact real rational interpretation of decimal stored parameters in x=1/f; IEEE roundoff checked separately at finite points.',
        verification=dict(cases=len(reports),numeric_checks=checks,maximum_numeric_relative_difference=largest_difference,
            contract_passed=all(not r['negative_slope_pieces'] and r['positive_at_left'] and r['positive_tail_slope'] for r in reports)),
        cases=reports,input_sha256=provenance,analysis_sha256=sha(Path(__file__)),
        scope='Mathematical callback contract only; no timing accuracy, parameter support expansion, history fit, hardware observation qualification, or new holdout.')
    assert len(reports)==30
    with gzip.open(out/'envelope.json.gz','wt') as stream:json.dump(result,stream,allow_nan=False)
    print(json.dumps(result['verification'],indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent affine branch verification; imports no analyzer or GEMM model."""
import copy,gzip,hashlib,json
from fractions import Fraction as Q
from pathlib import Path

ROOT=Path(__file__).resolve().parent
with gzip.open(ROOT/'envelope.json.gz','rt') as stream:REPORT=json.load(stream)
GUARDS=0


def plus(a,b):return (a[0]+b[0],a[1]+b[1])
def minus(a,b):return (a[0]-b[0],a[1]-b[1])
def times(a,b):return (a[0]*b,a[1]*b)
def const(a):return (Q(0),Q(a))
def evaluate(a,x):return a[0]*x+a[1]


def verify_branch(diff,lo,hi):
    global GUARDS
    assert evaluate(diff,lo)>=0,'branch changes before the left endpoint'
    if hi is None:assert diff[0]>=0,'branch changes in the unbounded tail'
    else:assert evaluate(diff,hi)>=0,'branch changes before the right endpoint'
    GUARDS+=1


def maximum(a,b,lo,hi,probe):
    choose=evaluate(a,probe)>=evaluate(b,probe)
    verify_branch(minus(a,b) if choose else minus(b,a),lo,hi)
    return a if choose else b


def below(a,b,lo,hi,probe,equal=False):
    choose=evaluate(a,probe)<=evaluate(b,probe) if equal else evaluate(a,probe)<evaluate(b,probe)
    verify_branch(minus(b,a) if choose else minus(a,b),lo,hi)
    return choose


def epilogue_end(p,start,overlap,lo,hi,probe):
    # Original sequential work-depletion branches, not the analyzer's min/max
    # lost-progress formula. Every selected branch holds on the whole interval.
    work=const(p['E']);time=start
    if overlap is None:return plus(time,work)
    a,b=overlap
    if below(time,a,lo,hi,probe):
        if below(plus(time,work),a,lo,hi,probe,True):return plus(time,work)
        work=minus(work,minus(a,time));time=a
    if below(time,b,lo,hi,probe):
        if below(times(work,1/p['r']),minus(b,time),lo,hi,probe,True):return plus(time,times(work,1/p['r']))
        work=minus(work,times(minus(b,time),p['r']));time=b
    return plus(time,work)


def cta(group,lo,hi,probe):
    p={k:Q(v) for k,v in group['params'].items()};loops=[]
    for intercept,kt,compute,service in group['loops']:
        compute=(Q(compute),Q(0));service=const(service)
        loops.append(plus(const(intercept),times(maximum(compute,service,lo,hi,probe),Q(kt))))
    if not loops:return const(p['P0'])
    if group['schedule']=='cooperative':
        first=const(p['P0']+p['S']);done=None
        for index,loop in enumerate(loops):
            if index:first=plus(done,const(p['h']))
            done=plus(plus(first,loop),const(p['w']+Q(group['epilogues'][index])))
        end=done
    else:
        main=[];done=[]
        for index,loop in enumerate(loops):
            first=const(p['P0']+p['S']) if not index else plus(main[-1],const(p['gm']))
            if index>=2:first=maximum(first,plus(done[index-2],const(p['h'])),lo,hi,probe)
            main.append(plus(first,loop));permit=plus(main[-1],const(p['w']))
            if index:permit=maximum(permit,plus(done[index-1],const(p['we'])),lo,hi,probe)
            overlap=None
            if index+1<len(loops):
                other=plus(main[-1],const(p['gm']))
                if index:other=maximum(other,plus(done[index-1],const(p['h'])),lo,hi,probe)
                overlap=(other,plus(other,loops[index+1]))
            done.append(epilogue_end(p,permit,overlap,lo,hi,probe))
        end=done[0]
        for value in done[1:]:end=maximum(end,value,lo,hi,probe)
    end=plus(end,const(p['Etail']))
    return plus(times(end,1+p['xk']),const(p['x0']+p['x1']*len(loops)))


def match(curve,actual,lo,hi,probe):
    pieces=[(Q(l),None if h is None else Q(h),(Q(a),Q(b))) for l,h,a,b in curve]
    matches=[p for p in pieces if p[0]<=probe and (p[1] is None or probe<p[1])]
    assert len(matches)==1
    l,h,expected=matches[0]
    assert l<=lo and (h is None or hi is not None and hi<=h)
    assert expected==actual,'affine expression differs from the original recurrence'


def verify_cases(report):
    intervals=0;groups=0;pieces=0;critical=0
    for row in report['cases']:
        lower=Q(row['lower_x']);cuts=[Q(v) for v in row['construction_partition']]
        assert cuts[0]==lower and cuts==sorted(set(cuts))
        assert set(row['cta_group_indices'])==set(range(len(row['groups'])))
        for lo,hi in zip(cuts,cuts[1:]+[None]):
            probe=lo+1 if hi is None else (lo+hi)/2
            values=[]
            for group,curve in zip(row['groups'],row['cta_curves']):
                actual=cta(group,lo,hi,probe);match(curve,actual,lo,hi,probe);values.append(actual);groups+=1
            end=values[0]
            for value in values[1:]:end=maximum(end,value,lo,hi,probe)
            match(row['envelope'],end,lo,hi,probe);intervals+=1
        curve=[(Q(l),None if h is None else Q(h),Q(a),Q(b)) for l,h,a,b in row['envelope']]
        assert curve[0][0]==lower and curve[-1][1] is None
        for lo,hi,a,b in curve:
            assert a>=0 and a*lo+b>0
            pieces+=1
        for before,after in zip(curve,curve[1:]):
            assert before[1]==after[0] and before[2]*before[1]+before[3]==after[2]*after[0]+after[3]
        assert curve[-1][2]>0
        critical+=len(row['cta_group_indices'])
    return dict(verified_intervals=intervals,verified_group_intervals=groups,envelope_pieces=pieces,cta_instances=critical)


hashes=0
for name,digest in REPORT['input_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    hashes+=1
assert hashlib.sha256((ROOT/'analyze.py').read_bytes()).hexdigest()==REPORT['analysis_sha256']
result=verify_cases(REPORT);original_guards=GUARDS
negative=[]
changed=copy.deepcopy(REPORT);piece=changed['cases'][0]['envelope'][0];piece[3]=str(Q(piece[3])+1)
try:verify_cases(changed)
except AssertionError:negative.append('wrong envelope offset rejected')
else:raise AssertionError('corrupt envelope accepted')
changed=copy.deepcopy(REPORT)
target=next(r for r in changed['cases'] if len(r['construction_partition'])>1)
target['construction_partition'].pop(1)
try:verify_cases(changed)
except AssertionError:negative.append('missing branch breakpoint rejected')
else:raise AssertionError('incomplete partition accepted')
changed=copy.deepcopy(REPORT)
changed['cases'][0]['groups'][0]['loops'][0][2]=str(Q(changed['cases'][0]['groups'][0]['loops'][0][2])+1)
try:verify_cases(changed)
except AssertionError:negative.append('wrong compute floor rejected')
else:raise AssertionError('corrupt compute floor accepted')
result.update(cases=len(REPORT['cases']),input_hashes=hashes,affine_branch_guards=original_guards,
              negative_checks=negative,passed=True,
              scope='Exact recurrence certificate for the exported rational static inputs; extraction/IEEE behavior has separate provenance and numeric checks.')
(ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

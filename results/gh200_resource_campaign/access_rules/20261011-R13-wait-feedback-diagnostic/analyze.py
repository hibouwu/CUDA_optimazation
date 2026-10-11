"""Six predeclared within-block feedback contrasts; instrumented-path diagnosis only."""
from fractions import Fraction
import math
import statistics

SERIES=tuple(f'stage{stage}_{name}' for stage in (4,6) for name in ('L16','L2','F'))


def contrasts(records):
    if len(records)!=6:raise ValueError('six calls per block required')
    arms={}
    for record in records:
        key=(record['stages'],record['observer'])
        if key not in {(s,m) for s in (4,6) for m in (1,2,3)} or key in arms:
            raise ValueError('invalid or duplicate stage/mode')
        arms[key]=record
        mode=key[1]
        q2,q16=record['group2_wait_cycles'],record['group16_wait_cycles']
        for q,selected in ((q2,mode in (1,2)),(q16,mode in (1,3))):
            if selected:
                if type(q) is not int or q<0:raise ValueError('native nonnegative selected wait required')
            elif q is not None:raise ValueError('unselected group must remain unobserved')
        if mode!=1 and (record['full_wait_cycles'] is not None or record['outside_wait_cycles'] is not None):
            raise ValueError('single-group mode cannot provide full early partition')
    if len({r['sm'] for r in records})!=1:raise ValueError('SM mapping changed')
    result={}
    for stage in (4,6):
        all_groups,only2,only16=(arms[stage,mode] for mode in (1,2,3))
        l16=all_groups['group16_wait_cycles']-only16['group16_wait_cycles']
        l2=all_groups['group2_wait_cycles']-only2['group2_wait_cycles']
        result.update({f'stage{stage}_L16':l16,f'stage{stage}_L2':l2,f'stage{stage}_F':l16-l2})
    return result


def summarize(records):
    if len(records)!=96:raise ValueError('all 96 frozen calls required; no partial score')
    if len({r['sm'] for r in records})!=1:raise ValueError('SM mapping changed across blocks')
    blocks=[contrasts(records[i:i+6]) for i in range(0,96,6)]
    n=len(blocks);alpha=Fraction(1,20*len(SERIES))
    ranks=[]
    for k in range(1,n//2+1):
        tail=Fraction(2*sum(math.comb(n,j) for j in range(k)),2**n)
        if tail<=alpha:ranks.append((k,tail))
    rank,tail=ranks[-1]
    summary={}
    for name in SERIES:
        values=sorted(block[name] for block in blocks)
        summary[name]=dict(median=statistics.median(values),sum=sum(values),count=n,
            conditional_median_interval=[values[rank-1],values[n-rank]],
            minimum=values[0],maximum=values[-1])
    return dict(blocks=blocks,summary=summary,units='native same-thread clock durations, differenced across calls',
        family_size=len(SERIES),ranks_one_based=[rank,n-rank+1],per_interval_tail_bound=str(tail),
        family_tail_bound=str(tail*len(SERIES)),
        coverage_assumption='IID block contrast vectors with a common distribution; not established by random ordering',
        interpretation='L16 and L2 compare complete observation bundles; F is their joint contrast, not pure collector cost.',
        group2_control_equivalence_proven=False,uninstrumented_wait_identified=False,
        observation_qualified=False,prediction_qualified=False)

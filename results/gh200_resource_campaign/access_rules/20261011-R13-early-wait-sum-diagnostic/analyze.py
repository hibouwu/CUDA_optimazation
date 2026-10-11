"""Joint block contrasts for a fixed diagnostic; never grant observation qualification."""
from fractions import Fraction
import math
import statistics

CONTRASTS = ('stage_D_on', 'stage_Q_on', 'stage_R_on', 'wait_minus_outside',
             'stage_D_off', 'stage_effect_change', 'observer_D_s4', 'observer_D_s6')


def block_contrasts(records):
    by_arm = {}
    for record in records:
        arm = (record['stages'], record['observer'])
        if arm in by_arm or arm not in ((4,0), (4,1), (6,0), (6,1)):
            raise ValueError('duplicate or unknown arm')
        by_arm[arm] = record
    if len(by_arm) != 4:
        raise ValueError('incomplete block')
    if len({r['sm'] for r in records}) != 1:
        raise ValueError('block SM mapping changed')
    for (stage, flag), record in by_arm.items():
        d, q, r = (record[k] for k in ('early_cycles', 'full_wait_cycles', 'outside_wait_cycles'))
        if type(d) is not int or d <= 0:
            raise ValueError('invalid native-cycle duration')
        if flag:
            if any(type(v) is not int or v < 0 for v in (q,r)) or d != q+r:
                raise ValueError('joint partition failed')
        elif q is not None or r is not None:
            raise ValueError('off components must remain unobserved')
    a, b, c, e = (by_arm[arm] for arm in ((4,0), (4,1), (6,0), (6,1)))
    d_on = b['early_cycles']-e['early_cycles']
    d_off = a['early_cycles']-c['early_cycles']
    q = b['full_wait_cycles']-e['full_wait_cycles']
    r = b['outside_wait_cycles']-e['outside_wait_cycles']
    result = dict(zip(CONTRASTS, (d_on,q,r,q-r,d_off,d_on-d_off,
                 b['early_cycles']-a['early_cycles'],e['early_cycles']-c['early_cycles'])))
    assert result['stage_D_on'] == result['stage_Q_on']+result['stage_R_on']
    assert result['stage_effect_change'] == result['observer_D_s4']-result['observer_D_s6']
    return result


def summarize(blocks):
    if len(blocks) != 16:
        raise ValueError('fixed diagnostic requires all 16 blocks; no partial scoring')
    # Exact binomial sign interval, Bonferroni for eight predeclared medians.
    # Its coverage is conditional on IID blocks; random order does not prove IID.
    n = len(blocks)
    alpha = Fraction(1,20*len(CONTRASTS))
    eligible = []
    for rank in range(1,n//2+1):
        tail = Fraction(2*sum(math.comb(n,k) for k in range(rank)),2**n)
        if tail <= alpha:
            eligible.append((rank,tail))
    rank, tail = eligible[-1]
    summary = {}
    for name in CONTRASTS:
        values = sorted(block[name] for block in blocks)
        if any(type(v) is not int for v in values):
            raise ValueError('native integer block contrasts required')
        summary[name] = dict(median=statistics.median(values),sum=sum(values),count=n,
            conditional_median_interval=[values[rank-1],values[n-rank]],
            minimum=values[0],maximum=values[-1])
    return dict(blocks=blocks,summary=summary,units='same-SM clock cycles per call, differenced across calls',
        ranks_one_based=[rank,n-rank+1],family_size=len(CONTRASTS),
        per_interval_tail_bound=str(tail),family_tail_bound=str(tail*len(CONTRASTS)),
        coverage_assumption='IID block contrast vectors with a common distribution; not established by this experiment',
        median_components_additive=False,observation_qualified=False,prediction_qualified=False,
        scope='Instrumented stage contrasts and off/on total-window changes only; no inference of uninstrumented Q.')

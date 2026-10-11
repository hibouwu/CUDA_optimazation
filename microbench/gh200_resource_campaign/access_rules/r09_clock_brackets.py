"""Pure interval checks for CLOCK-before / GLOBALTIMER / CLOCK-after stamps.

No parsing, calibration, fitting, IO, or target-duration prediction. The caller
keeps CTA/role/endpoint identity. Bounds assume the globaltimer read corresponds
to a cycle inside the recorded clock bracket; the test does not prove that
measurement assumption or identify physical DVFS.
"""
from fractions import Fraction
import math
import statistics as st


PERTURBATION_LIMITS = dict(local_median_absolute_ns=384,local_pair_ns=1024,
    local_pairs_required=9,cycle_median_absolute=512,event_median_absolute_ns=512)


def window_bounds(begin, end):
    """Each stamp is (clock_before, globaltimer_ns, clock_after), all integers."""
    for stamp in (begin,end):
        if len(stamp)!=3 or any(not isinstance(v,int) or isinstance(v,bool) for v in stamp):
            raise ValueError('integer clock/globaltimer stamps required')
        if stamp[2]<stamp[0]:
            raise ValueError('clock bracket reversed')
    # Subtract integer timestamps before conversion, including for globaltimer.
    cycles_lo = max(0,end[0]-begin[2])
    cycles_hi = end[2]-begin[0]
    ns = end[1]-begin[1]
    if ns<=0 or cycles_hi<=0:
        raise ValueError('nonpositive window')
    return dict(cycles_lo=cycles_lo,cycles_hi=cycles_hi,ns=ns)


def _tolerance(value):
    if not math.isfinite(value) or value<0:
        raise ValueError('nonnegative finite interval tolerance required')
    return Fraction(str(value))


def _intersection(bounds, tolerance):
    lower,upper = Fraction(0),None
    for b in bounds:
        lower = max(lower,Fraction(b['cycles_lo'])/(b['ns']+tolerance))
        if b['ns']>tolerance:
            limit = Fraction(b['cycles_hi'])/(b['ns']-tolerance)
            upper = limit if upper is None else min(upper,limit)
    feasible = upper is None or lower<=upper
    return dict(feasible=feasible,lower_ghz=float(lower),
                upper_ghz=float(upper) if upper is not None else None,
                lower_open=(lower==0),upper_unbounded=(upper is None))


def common_rate_interval(windows, *, interval_tolerance_ns=128):
    """Intersect positive-f sets for matched L windows.

    128 ns is the TOTAL WINDOW-DIFFERENCE decision allowance: true duration is
    in [N-128,N+128]. It is not a 128 ns allowance at each endpoint. If the
    cycle interval is [Clo,Chi], f is in [Clo/(N+128),Chi/(N-128)], with an
    unbounded upper end when N<=128. The zero lower bound is always open.

    minimum_interval_slack_ns is the smallest TOTAL interval allowance needed
    for any common positive f, including zero existing allowance. The extra
    slack removes the declared allowance once, never once per endpoint.
    """
    if not windows:
        raise ValueError('at least one matched window required')
    bounds = [window_bounds(begin,end) for begin,end in windows]
    tolerance = _tolerance(interval_tolerance_ns)
    # Feasible in inverse rate u iff max((Ni-delta)/Chi) <=
    # min((Nj+delta)/Cloj). Every active lower/upper pair gives a necessary
    # delta; their maximum is also sufficient. Clo=0 imposes no upper-u bound.
    minimum = Fraction(0)
    witness = None
    for i,a in enumerate(bounds):
        for j,b in enumerate(bounds):
            if b['cycles_lo']:
                slack = Fraction(a['ns']*b['cycles_lo']-b['ns']*a['cycles_hi'],
                                 b['cycles_lo']+a['cycles_hi'])
                if slack>minimum:
                    minimum,witness = slack,(i,j)
    result = _intersection(bounds,tolerance)
    if not _intersection(bounds,minimum)['feasible']:
        raise ValueError('exact minimum slack did not produce an intersection')
    extra = max(Fraction(0),minimum-tolerance)
    result.update(window_bounds=bounds,interval_tolerance_ns=float(tolerance),
                  minimum_interval_slack_ns=float(minimum),
                  minimum_interval_slack_fraction=[minimum.numerator,minimum.denominator],
                  extra_interval_slack_ns=float(extra),
                  extra_endpoint_slack_equivalent_ns=float(extra/2),
                  witness_indices=witness)
    return result


def observer_perturbation_gate(pairs):
    """Prelisted screen for bracket-dual minus matched original-dual.

    Inputs are ten sequential-process paired changes, each phase already a
    median over same-role CTA L windows. Median ABSOLUTE changes are checked,
    so opposite signed changes cannot cancel. Passing is only a prerequisite
    for comparing the original and bracket observer, never proof of no
    observation effect. Plain comparisons are reported separately.
    """
    if len(pairs)!=10:
        raise ValueError('all ten prelisted pairs required')
    keys = ('first_L_ns_delta','later_L_ns_delta','first_L_cycles_delta',
            'later_L_cycles_delta','event_ns_delta')
    if any(not math.isfinite(p[k]) for p in pairs for k in keys):
        raise ValueError('finite paired native differences required')
    metrics = {}
    passes = True
    for key in keys:
        values = [p[key] for p in pairs]
        median_absolute = st.median(abs(v) for v in values)
        if key.endswith('cycles_delta'):
            limit = PERTURBATION_LIMITS['cycle_median_absolute']
        elif key=='event_ns_delta':
            limit = PERTURBATION_LIMITS['event_median_absolute_ns']
        else:
            limit = PERTURBATION_LIMITS['local_median_absolute_ns']
        ok = median_absolute<=limit
        item = dict(median_signed=st.median(values),median_absolute=median_absolute,
                    maximum_absolute=max(abs(v) for v in values),limit=limit,passes=ok)
        if key in ('first_L_ns_delta','later_L_ns_delta'):
            within = sum(abs(v)<=PERTURBATION_LIMITS['local_pair_ns'] for v in values)
            item['pairs_within_1024ns'] = within
            ok &= within>=PERTURBATION_LIMITS['local_pairs_required']
            item['passes'] = ok
        metrics[key] = item
        passes &= ok
    return dict(passes=passes,metrics=metrics,
                interpretation='Prerequisite only; if false, original-phenomenon attribution is inconclusive. Bracket arithmetic is still reported.')

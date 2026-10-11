"""Pure interval checks for CLOCK-before / GLOBALTIMER / CLOCK-after stamps.

No parsing, calibration, fitting, IO, or target-duration prediction. The caller
keeps CTA/role/endpoint identity. Bounds assume the globaltimer read corresponds
to a cycle inside the recorded clock bracket; the test does not prove that
measurement assumption or identify physical DVFS.
"""
from fractions import Fraction
import math
import statistics as st


SCHEMA = 'gh200-r09-clock-brackets/2'


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
    result.update(schema=SCHEMA,window_bounds=bounds,interval_tolerance_ns=float(tolerance),
                  minimum_interval_slack_ns=float(minimum),
                  minimum_interval_slack_fraction=[minimum.numerator,minimum.denominator],
                  extra_interval_slack_ns=float(extra),
                  extra_endpoint_slack_equivalent_ns=float(extra/2),
                  witness_indices=witness)
    return result


def paired_phase_summary(ctas, *, expected_ctas=132):
    """Audit every matched CTA/original role before taking phase centers.

    Each row has cta, role, phase (first/later), L_ns_delta, L_cycles_delta.
    Optional SM/work identities are retained by the caller and in the gate's
    original cta_pairs. The fixed probe owns role0/first and role1/later.
    """
    keys = {(i,role,phase) for i in range(expected_ctas)
            for role,phase in ((0,'first'),(1,'later'))}
    supplied = [(x['cta'],x['role'],x['phase']) for x in ctas]
    if len(supplied)!=len(keys) or set(supplied)!=keys:
        raise ValueError('complete unique CTA/original-role pairs required')
    if any(not math.isfinite(x[k]) for x in ctas for k in ('L_ns_delta','L_cycles_delta')):
        raise ValueError('finite native CTA deltas required')
    phases = {}
    for phase in ('first','later'):
        group = sorted((x for x in ctas if x['phase']==phase),key=lambda x:x['cta'])
        phases[phase] = dict(cta_ids=[x['cta'] for x in group])
        for key in ('L_ns_delta','L_cycles_delta'):
            signed = [x[key] for x in group]
            center = st.median(signed)
            absolute = sorted(abs(v) for v in signed)
            centered = sorted(abs(v-center) for v in signed)
            phases[phase][key] = dict(center_signed=center,signed_values=signed,
                absolute_values=absolute,median_absolute=st.median(absolute),
                p95_absolute=absolute[math.ceil(.95*len(absolute))-1],maximum_absolute=max(absolute),
                centered_absolute_values=centered,median_centered_absolute=st.median(centered),
                p95_centered_absolute=centered[math.ceil(.95*len(centered))-1],
                maximum_centered_absolute=max(centered))
    return phases


def observer_perturbation_gate(pairs, *, expected_ctas=132):
    """Two-layer screen: process centers and unqualified spatial variation.

    Inputs are TEN raw sequential-process pairs. Each pair contains event_ns_delta
    and cta_pairs with all CTA/role native deltas. Phase center is explicitly
    median(same-CTA signed deltas), never difference of separate CTA medians.
    The old 384/1024 limits apply only to these PROCESS CENTERS. Their absolute
    values across processes prevent process-level cancellation, but cannot
    qualify spatial stability. Every CTA signed/absolute distribution remains.

    No calibrated spatial tolerance exists. Nonzero CENTERED spatial deltas therefore
    leave original-phenomenon inference inconclusive even if center limits pass.
    Exact per-phase spatial shape preservation in cycle AND ns can pass as a
    literal identity (a uniform phase shift is screened by the center limits),
    not a noise-tolerance rule. Plain comparisons remain separately reported.
    """
    if len(pairs)!=10:
        raise ValueError('all ten prelisted pairs required')
    if any(not math.isfinite(p['event_ns_delta']) for p in pairs):
        raise ValueError('finite paired event differences required')
    process_centers,spatial = [],[]
    for index,p in enumerate(pairs):
        phases = paired_phase_summary(p['cta_pairs'],expected_ctas=expected_ctas)
        center = dict(event_ns_delta=p['event_ns_delta'])
        for phase in ('first','later'):
            for kind in ('ns','cycles'):
                center[phase+'_L_'+kind+'_delta'] = phases[phase]['L_'+kind+'_delta']['center_signed']
        process_centers.append(center)
        spatial.append(dict(pair_index=index,phases=phases,cta_pairs=p['cta_pairs']))
    keys = ('first_L_ns_delta','later_L_ns_delta','first_L_cycles_delta',
            'later_L_cycles_delta','event_ns_delta')
    metrics = {}
    passes = True
    for key in keys:
        values = [p[key] for p in process_centers]
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
    exact = all(p['phases'][phase][key]['maximum_centered_absolute']==0
                for p in spatial for phase in ('first','later')
                for key in ('L_ns_delta','L_cycles_delta'))
    return dict(schema=SCHEMA,passes=passes and exact,center_screen_passes=passes,metrics=metrics,
                spatial_screen_status='exact per-phase spatial shape identity' if exact else 'unresolved: no calibrated spatial tolerance',
                exact_spatial_identity=exact,spatial_processes=spatial,
                interpretation='Center limits alone do not qualify local stability. Nonzero centered spatial change is retained as unqualified; bracket arithmetic can still be reported, original-phenomenon attribution is inconclusive.')

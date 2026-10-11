"""Conditional GT-before / CLOCK / GT-after interval arithmetic only.

The caller establishes actual sampling enclosure, GT coordinate-error bounds,
record/source identity and matching CTA/WG markers. These functions establish
none of those facts and never parse or reinterpret old CLOCK/GT/CLOCK frames.
"""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class GtClockStamp:
    gpu_uuid: str
    sm: int
    gt_before: int
    cycle: int
    gt_after: int


def _exact(value, name, *, positive=False):
    if type(value) not in (int, Fraction):
        raise ValueError('explicit exact integer/Fraction required: ' + name)
    value = Fraction(value)
    if value < 0 or (positive and value == 0):
        raise ValueError('invalid nonnegative/positive value: ' + name)
    return value


def _validate_stamp(stamp):
    if type(stamp) is not GtClockStamp:
        raise ValueError('explicit GT/CLOCK/GT stamp required')
    if type(stamp.gpu_uuid) is not str or not stamp.gpu_uuid.strip():
        raise ValueError('GPU identity required')
    if type(stamp.sm) is not int or not 0 <= stamp.sm < 2**32:
        raise ValueError('nonnegative integer SM identity required')
    for value in (stamp.gt_before, stamp.cycle, stamp.gt_after):
        if type(value) is not int or not 0 < value < 2**64:
            raise ValueError('nonzero native uint64 stamp required')
    if stamp.gt_after < stamp.gt_before:
        raise ValueError('GT bracket reversed')


def window_interval(begin, end, *, gt_error_ns):
    """Enclose elapsed time and average recorded-counter rate C/T conditionally.

    gt_error_ns is an EXPLICIT conditional absolute error per GT read, not
    measured accuracy or a default tolerance. Each clock sampling time lies
    in [GT_before-error, GT_after+error] by hypothesis. Different endpoint
    errors are allowed independently; this is a marginal window bound.

    For overlapping time brackets, elapsed time remains strictly positive
    while its infimum may be zero. The rate upper bound is then unbounded.
    """
    _validate_stamp(begin)
    _validate_stamp(end)
    if begin.gpu_uuid != end.gpu_uuid or begin.sm != end.sm:
        raise ValueError('one GPU/SM counter domain required')
    error = _exact(gt_error_ns, 'gt_error_ns')
    cycles = end.cycle - begin.cycle
    if cycles <= 0:
        raise ValueError('positive native cycle difference without wrap required')
    raw_lower = Fraction(end.gt_before - begin.gt_after) - 2 * error
    upper = Fraction(end.gt_after - begin.gt_before) + 2 * error
    if upper <= 0:
        raise ValueError('stated time brackets cannot enclose positive elapsed time')
    lower = max(Fraction(0), raw_lower)
    return dict(gpu_uuid=begin.gpu_uuid, sm=begin.sm, cycles=cycles,
        gt_error_ns=error, elapsed_lower_ns=lower, elapsed_upper_ns=upper,
        elapsed_lower_open=(lower == 0),
        rate_lower_cycle_per_ns=Fraction(cycles)/upper,
        rate_upper_cycle_per_ns=Fraction(cycles)/lower if lower else None,
        begin_gt_width_ns=begin.gt_after-begin.gt_before,
        end_gt_width_ns=end.gt_after-end.gt_before,
        interpretation='Conditional marginal bound; hardware enclosure and GT accuracy are caller premises.',
        observation_qualified=False, prediction_qualified=False, joint_feasibility_proved=False)


def error_intervals(begin, end, *, gt_error_ns, predicted_cycles, predicted_rate):
    """Coupled cycle, conditional conversion and free-time errors for one T.

    At every admissible elapsed T, (1+e_C)(1+e_f(T))=1+e_N(T).
    The same lower/upper T produces matching error endpoints. Independent
    medians, different windows or possible interval choices do not compose.
    """
    bounds = window_interval(begin, end, gt_error_ns=gt_error_ns)
    candidate_cycles = _exact(predicted_cycles, 'predicted_cycles', positive=True)
    candidate_rate = _exact(predicted_rate, 'predicted_rate', positive=True)
    cycle_error = candidate_cycles/bounds['cycles'] - 1
    conversion_lower = bounds['rate_lower_cycle_per_ns']/candidate_rate - 1
    conversion_upper = (bounds['rate_upper_cycle_per_ns']/candidate_rate - 1
                        if bounds['rate_upper_cycle_per_ns'] is not None else None)
    prediction_ns = candidate_cycles/candidate_rate
    free_lower = prediction_ns/bounds['elapsed_upper_ns'] - 1
    free_upper = (prediction_ns/bounds['elapsed_lower_ns'] - 1
                  if bounds['elapsed_lower_ns'] else None)
    assert (1+cycle_error)*(1+conversion_lower) == 1+free_lower
    if conversion_upper is not None:
        assert (1+cycle_error)*(1+conversion_upper) == 1+free_upper
    bounds.update(predicted_cycles=candidate_cycles, predicted_rate=candidate_rate,
        predicted_ns=prediction_ns, cycle_error=cycle_error,
        conversion_error_lower=conversion_lower, conversion_error_upper=conversion_upper,
        free_error_lower=free_lower, free_error_upper=free_upper)
    return bounds


def relative_error_status(lower, upper, *, tolerance):
    """Classify one conditional closed error band; upper=None means +infinity.

    'within' means all admissible errors meet tolerance. 'outside' means none
    do. 'undetermined' must not be converted into a successful qualification.
    This is not a joint feasibility test for multiple shared endpoint reads.
    """
    if type(lower) not in (int, Fraction) or (upper is not None and type(upper) not in (int, Fraction)):
        raise ValueError('exact relative error endpoints required')
    lower = Fraction(lower)
    upper = Fraction(upper) if upper is not None else None
    if upper is not None and upper < lower:
        raise ValueError('relative error interval reversed')
    allowed = _exact(tolerance, 'tolerance')
    if lower <= 0 and (upper is None or upper >= 0):
        minimum_absolute = Fraction(0)
    elif upper is not None:
        minimum_absolute = min(abs(lower), abs(upper))
    else:
        minimum_absolute = abs(lower)
    maximum_absolute = max(abs(lower), abs(upper)) if upper is not None else None
    if maximum_absolute is not None and maximum_absolute <= allowed:
        status = 'within'
    elif minimum_absolute > allowed:
        status = 'outside'
    else:
        status = 'undetermined'
    return dict(status=status, minimum_absolute=minimum_absolute,
        maximum_absolute=maximum_absolute, tolerance=allowed,
        interpretation='Conditional single-window interval classification only; no observation qualification.')

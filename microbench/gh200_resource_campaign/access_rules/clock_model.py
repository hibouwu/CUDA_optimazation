"""Calibrated work/source clock and positive envelope closure, without fitting or IO."""
import math


def duration_term(window, form, tau):
    if form in ('frozen', 'refit_ac', 'full_log'):
        return math.log(window)
    if form == 'no_duration':
        return 0.0
    # Mean of an exponential relaxation over a window, not its instantaneous endpoint.
    x = window / tau
    return 1 + math.expm1(-x) / x


MODES = ('dyadic', 'zero', 'random')


COEFFICIENTS = ('a', 'd_dyadic', 'd_zero', 'd_random', 'e_dyadic', 'e_zero', 'e_random')


def source_frequency(point, window, model):
    a, *costs = model['coefficients']
    mode = MODES.index(point['input_mode'])
    tensor = costs[mode] * point['tensor_mean_cycles'] / (1000 * window)
    source = costs[mode + 3] * point['source_mean_kib'] / window * duration_term(window, 'relaxation', model['tau_us'])
    return dict(frequency_ghz=a - tensor - source, tensor_penalty_ghz=tensor, source_penalty_ghz=source)


def solve_source_envelope(point, model, envelope_us, *, rtol=1e-10, max_iter=200):
    """Couple the static Q/S clock to a caller's NS-recursion envelope, in us.

    Requires continuous, positive, nonincreasing E(f) for 0<f<=a, with
    E(f)->infinity as f->0 (positive work with a compute floor suffices).
    Since g is concave and g(0)=0, g(W)/W decreases: F(W) is nondecreasing.
    Thus h(f)=F(E(f))-f strictly decreases, h(0+)=a>0, h(a)<=0.
    Bisect the finite frequency bracket (0,a]; never evaluate E at zero or
    a negative clock mapping. Callback monotonicity is the caller's contract,
    not a claim that every overlapping event recursion has been proved so.

    Reads only input_mode/tensor_mean_cycles/source_mean_kib from point.
    Closure is F(returned W)/returned f - 1; returned W equals E(returned f).
    """
    a, *costs = model['coefficients']
    values = [a, *costs, model['tau_us'], point['tensor_mean_cycles'], point['source_mean_kib']]
    if (not all(math.isfinite(v) for v in values) or a <= 0 or min(costs) < 0
            or model['tau_us'] <= 0 or min(point['tensor_mean_cycles'], point['source_mean_kib']) < 0
            or not math.isfinite(rtol) or rtol <= 0 or max_iter < 1):
        raise ValueError('invalid source-envelope coefficient/work/solver domain')

    def evaluate(frequency):
        window = float(envelope_us(frequency))
        if not math.isfinite(window) or window <= 0:
            raise ValueError('envelope callback must return finite positive us')
        mapped = source_frequency(point, window, model)['frequency_ghz']
        if not math.isfinite(mapped):
            raise ValueError('nonfinite source-clock mapping')
        return window, mapped / frequency - 1

    lo, hi = 0.0, a  # zero is a symbolic limiting endpoint, never a callback input
    window, closure = evaluate(hi)
    if abs(closure) <= rtol:
        return dict(window_us=window, frequency_ghz=hi, closure=closure)
    for _ in range(max_iter):
        frequency = (lo + hi) / 2
        if frequency <= 0 or frequency in (lo, hi):
            break
        window, closure = evaluate(frequency)
        if abs(closure) <= rtol:
            return dict(window_us=window, frequency_ghz=frequency, closure=closure)
        if closure > 0:
            lo = frequency
        else:
            hi = frequency
    raise RuntimeError('source-envelope root did not converge; check callback contract')


def solve_source_clock(point, model):
    """The unique positive root lies in [target/a, (target+e*S)/a]."""
    a, *costs = model['coefficients']
    if not (a > 0 and min(costs) >= 0 and model['tau_us'] > 0
            and point['converted_cycles'] > 0 and point['tensor_mean_cycles'] >= 0 and point['source_mean_kib'] >= 0):
        raise ValueError('source-clock coefficient/work domain violated')
    mode = MODES.index(point['input_mode'])
    target = (point['converted_cycles'] + costs[mode] * point['tensor_mean_cycles']) / 1000
    source = costs[mode + 3] * point['source_mean_kib']
    if not (a > 0 and target > 0 and source >= 0):
        raise ValueError('source-clock root outside declared coefficient/work domain')
    lo, hi = target / a, (target + source) / a
    for _ in range(100):
        mid = (lo + hi) / 2
        residual = a * mid - source * duration_term(mid, 'relaxation', model['tau_us']) - target
        if residual < 0:
            lo = mid
        else:
            hi = mid
    window = (lo + hi) / 2
    frequency = point['converted_cycles'] / (1000 * window)
    closure = source_frequency(point, window, model)['frequency_ghz'] / frequency - 1
    if abs(closure) > 1e-8 or not 0 < frequency <= a + 1e-9:
        raise ValueError('source-clock positive root failed closure/cap check')
    return dict(window_us=window, frequency_ghz=frequency, closure=closure,
                old_model_transfer_us=point['fixed_us'] + window)

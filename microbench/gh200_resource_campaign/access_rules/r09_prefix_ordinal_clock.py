"""Development-only local conversion profile conditioned on a completed prefix.

The two output ordinals have separate calibrated C/N profiles. This predicts
recorded common-compute windows, not physical frequency or complete event time.
"""
from fractions import Fraction


SETUP_FIELDS = (
    'config', 'gpu_uuid', 'binary_sha256', 'm', 'n', 'k', 'grid', 'tile',
    'cluster', 'stages', 'threads', 'smem', 'max_active_ctas_per_sm',
    'sm_count', 'scheduler_sm_count', 'input_mode', 'seed', 'lda', 'ldb',
    'ldd', 'alloc_lda', 'alloc_ldb', 'storage_m', 'storage_n', 'zero_m',
    'zero_n', 'swizzle', 'raster_order', 'resolved_raster_order',
    'trace_version', 'trace_words', 'trace_tile_words', 'warmup_policy',
    'warmup_swizzle', 'warmup_fixed_calls', 'history_mode', 'history_protocol',
    'history_source', 'source_copies', 'mirror_target_slot',
    'producer_tail_observer', 'producer_tail_prefix_flag',
    'release_warp_observer', 'epilogue_defer', 'a_row_rotation',
    'second_prefill', 'first_supply_events', 'producer_delay_ns',
    'work_swap55_91', 'b_offset_bytes', 'b_allocation_extra_bytes', 'evict',
    'prefix_same_a', 'prefix_same_b', 'prefix_calls', 'target_calls',
    'preceding_target_calls',
)


def predict_rates(setup, last_prefix_event_us, calibration, *, ordinal_profile=True):
    """Return two cycles/ns ratios using static setup and one past host event.

    Current target cycles, ns, event, CTA/SM assignment and residuals are not
    inputs. The calibrated protocol and card must match exactly.
    """
    if calibration.get('schema') != 'r09-prefix-ordinal-clock-calibration/1':
        raise ValueError('known calibration schema required')
    expected = calibration['setup']
    if set(expected) != set(SETUP_FIELDS):
        raise ValueError('complete frozen static setup required')
    for key in SETUP_FIELDS:
        value = setup[key]
        if type(value) is not type(expected[key]) or value != expected[key]:
            raise ValueError('unsupported prefix/target protocol: ' + key)
    if type(last_prefix_event_us) not in (int, float, str, Fraction):
        raise ValueError('finite positive completed prefix event required')
    try:
        prefix_ns = Fraction(str(last_prefix_event_us)) * 1000
    except (ValueError, ZeroDivisionError):
        raise ValueError('finite positive completed prefix event required') from None
    if prefix_ns <= 0:
        raise ValueError('finite positive completed prefix event required')
    values = calibration['kappa_cycles'] if ordinal_profile else [calibration['common_kappa_cycles']] * 2
    if len(values) != 2:
        raise ValueError('two output ordinals required')
    kappas = [Fraction(v) for v in values]
    if min(kappas) <= 0:
        raise ValueError('positive calibrated profile required')
    return [kappa / prefix_ns for kappa in kappas]

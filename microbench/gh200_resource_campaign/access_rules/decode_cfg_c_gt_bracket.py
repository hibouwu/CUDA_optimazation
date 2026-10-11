"""Strict fixed cfg_c GT-C-GT packets. Decoding grants no observation qualification."""
import math
from decode_cfg_c_producer_tail import decode as decode_original_trace, equal, VERSION as BASE_VERSION

VERSION = "cfg-c-gt-bracket-v1"
SHADOW_WORDS = 9


def decode(setup, call, gpu_uuid):
    fixed = dict(gt_bracket_protocol=VERSION, gt_bracket_prefix_flag=0,
        gt_shadow_words_per_cta=SHADOW_WORDS, gt_shadow_bytes=132*SHADOW_WORDS*8,
        producer_tail_observer=0, producer_tail_prefix_flag=0,
        warmup_swizzle=1, warmup_fixed_calls=16, history_protocol="none",
        history_source="original", source_copies=1, mirror_target_slot=0,
        release_warp_observer=0, release_warp_warmup=0,
        epilogue_defer=0, epilogue_warmup_defer=0, second_prefill=0,
        second_prefill_warmup=0, a_row_rotation=0, first_supply_events=0,
        producer_delay_ns=0, work_swap55_91=0, b_offset_bytes=0,
        b_allocation_extra_bytes=0, b_payload_elements=4096*3072)
    for packet in (setup, call):
        if any(not equal(packet.get(key), value) for key, value in fixed.items()):
            raise ValueError("fixed GT bracket protocol mismatch")
        if not equal(packet.get("b_buffer_ptr"), packet.get("target_b_ptr")):
            raise ValueError("B allocation identity mismatch")
    setup_fixed = dict(trace_version=VERSION, mode="warm", threads=384,
        scheduler_sm_count=132, requested_sm_count=0, max_active_ctas_per_sm=1,
        raster_order="heuristic", resolved_raster_order="along_m", evict=0,
        history_capacity_bytes=0, trace_tile_capacity=64)
    call_fixed = dict(label="warm", target_swizzle=1, target_calls=1, preceding_target_calls=0)
    for packet, fields in ((setup, setup_fixed), (call, call_fixed)):
        if any(not equal(packet.get(key), value) for key, value in fields.items()):
            raise ValueError("fixed GT bracket execution mismatch")
    flag = setup.get("gt_bracket_observer")
    if type(flag) is not int or flag not in (0, 1) or not equal(call.get("gt_bracket_observer"), flag):
        raise ValueError("explicit matching GT observer flag required")
    elapsed = call.get("elapsed_us")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("positive finite target event required")
    shadow = call.get("gt_shadow")
    if not isinstance(shadow, list) or len(shadow) != 132*SHADOW_WORDS or any(
            type(v) is not int or not 0 <= v < 2**64 for v in shadow):
        raise ValueError("complete native uint64 shadow required")
    # Reuse only the original 1296-word structural checks after validating the
    # real new protocol. This local view is never exported as old-version data.
    structural_setup = dict(setup, trace_version=BASE_VERSION)
    original = decode_original_trace(structural_setup, call, gpu_uuid)
    errors = list(original["errors"])
    records = []
    for record in original["records"]:
        cta = record["cta"]
        values = shadow[cta*SHADOW_WORDS:(cta+1)*SHADOW_WORDS]
        try:
            if values[8]:
                raise ValueError("GT shadow overflow")
            if not flag and any(values):
                raise ValueError("disabled GT observer wrote shadow")
            for role in record["roles"]:
                wg = role["wg"]
                if len(role["tiles"]) > 2:
                    raise ValueError("GT shadow output capacity exceeded")
                for ordinal, tile in enumerate(role["tiles"]):
                    before = values[wg*4+ordinal*2:wg*4+ordinal*2+2]
                    if flag:
                        after = tile["ns"][:2]
                        if not (0 < before[0] <= after[0] < before[1] <= after[1]):
                            raise ValueError("zero/reversed/overlapping GT brackets")
                        tile["gt_brackets"] = [dict(gt_before=before[e], cycles=tile["cycles"][e],
                            gt_after=after[e]) for e in (0, 1)]
                        tile["conditional_elapsed_gt"] = [before[1]-after[0], after[1]-before[0]]
                        tile["post_gt_elapsed"] = after[1]-after[0]
                        tile["post_gt_error_bound"] = max(after[e]-before[e] for e in (0, 1))
                    else:
                        tile["gt_brackets"] = None
                unused = values[wg*4+len(role["tiles"])*2:(wg+1)*4]
                if any(unused):
                    raise ValueError("unused GT shadow slots nonzero")
            records.append(record)
        except ValueError as error:
            errors.append(dict(cta=cta, error=str(error)))
    return dict(version=VERSION, flag=flag, records=records, errors=errors,
        record_valid=not errors and len(records)==132,
        hardware_sampling_enclosure=False, observation_qualified=False,
        limits="Full fixed geometry and identities checked. Intervals are conditional on sampling order and GT reference-domain assumptions. Numerical correctness, binary identity, GT accuracy, perturbation and null controls remain external.")

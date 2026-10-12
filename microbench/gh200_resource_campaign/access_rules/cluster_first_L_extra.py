"""Static first-L residual interface for the rejected Task A form.

The declared coefficient has units ns/Ktile/unit cluster maximum fill fraction.
This module has no timing inputs and does not assign prices to missing parameters.
"""
import math

import supply_model as supply


CALIBRATED_OPERAND = {"cfg_a": "A", "cfg_c": "B"}


def first_loop_weights(row, setup):
    """Return one static coefficient multiplier per scheduled CTA.

    cfg_b has no calibrated cluster residual and receives zero component weight.
    Returning geometry does not grant model support or imply a fitted price.
    """
    weights = [0.0] * math.prod(setup["grid"])
    operand = CALIBRATED_OPERAND.get(row["config"])
    if operand is None:
        return weights
    if row["k"] <= 0 or row["k"] % 64:
        raise ValueError("first-L form requires positive complete Ktiles")
    tile_dimension = setup["tile"][0 if operand == "A" else 1]
    nominal_kib = tile_dimension * 64 * 2 / 1024
    for feature in supply.boundary_fill_features(row, setup):
        if feature["j"] == 0:
            fraction = feature["operands"][operand]["cluster_max"] / nominal_kib
            if not 0 <= fraction <= 1:
                raise ValueError("cluster fill fraction must lie in [0,1]")
            weights[feature["cta"]] = (row["k"] // 64) * fraction
    return weights


def first_loop_extra_ns(row, setup, parameters):
    """Reject an unfitted parameter; never treat null as a zero hardware price."""
    cfg = row["config"]
    if cfg not in CALIBRATED_OPERAND:
        return [0.0] * math.prod(setup["grid"])
    if not 1024 <= row["k"] <= 4096:
        raise ValueError("K outside reference calibration span")
    if row["lda"] * 2 % 128 or row["ldb"] * 2 % 128:
        raise ValueError("unaligned input pitch")
    if setup["stages"] != {"cfg_a": 6, "cfg_c": 4}[cfg]:
        raise ValueError("nondefault stage")
    beta = parameters["beta_ns_per_ktile_per_fill_fraction"].get(cfg)
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError("cluster residual was not priced")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("cluster residual price must be finite and nonnegative")
    return [beta * weight for weight in first_loop_weights(row, setup)]

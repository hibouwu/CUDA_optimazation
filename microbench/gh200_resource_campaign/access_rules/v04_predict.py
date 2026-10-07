#!/usr/bin/env python3
"""V04: transfer the V02 time model to three other CUTLASS configs, frozen before measurement.

Base (V02, configs/v02-predictions.json): 128x256x64 cooperative, cluster 2x1, 4 stage.
  C = P + R*(c_k*Kt + c_0 + E),  W = C/f(W),  f(W) = a - b*ln(W/us),  T = h + s + W + p + tau
Targets (probes/v04_probe.cu, all FP16->FP32, ElementC=void, persistent scheduler, -DNDEBUG):
  cfg_a 128x128x64 cooperative, cluster 2x1, stages auto
  cfg_b 128x128x64 pingpong,    cluster 1x1, stages auto
  cfg_c 256x128x64 cooperative, cluster 1x2, 4 stages

Transfer rules (each term keeps the V02 value unless a rule below changes it):
  T1 compute per Ktile: c_k' = c_k * F'/F_base, F = 2*TM*TN*64 per CTA (R00-B: m64n{64,128,256}k16,
     1 or 2 warpgroups all reach 4096 FLOP/cycle; c_k = 1024.03 measured by R09 for the base).
  T2 supply per Ktile: bytes delivered into one SM's SMEM per Ktile, B = (TM+TN)*64*2 (multicast
     does not reduce what each SM receives). If B/c_k' > S = 55.45 B/cycle (R05-D, 1 CTA/SM,
     shared small L2 source, best observed supply) the mainloop is supply-bound: B/S per Ktile.
     Not scored variant `l2req`: use L2-request bytes after cluster multicast instead of B.
  T3 per-tile mainloop constant: c_0' = c_0 - 0.62*(256 - N_wgmma) (R01: WGMMA+wait0 ~36+0.62N).
  T4 per-tile non-mainloop cost E (R09: E ~ epilogue segment, 5470-5850 cycles for 128 KiB/tile):
     E' = E * out_bytes'/out_bytes_base, out_bytes = TM*TN*4 (FP32 D, TMA store, rate-limited).
  T5 prefill P = P*(f_setup + (1-f_setup)*stage_bytes'/49152): producer setup part (R09
     producer_setup/prefill, warm M=2048) fixed, first-stage arrival part scales with stage bytes.
  T6 rounds: grid = 132 persistent CTAs (CUTLASS default sm_count), 1 CTA/SM if
     max_active_ctas_per_sm reported by the build is 1 (else see occupancy note in output);
     U = tiles after rounding tiles_m/tiles_n up to the cluster shape (R08), R = ceil(U/132).
  T7 pingpong (cfg_b): the two consumer warpgroups alternate tiles; mainloops are serialized by
     the order barrier and so are epilogues, epilogue of tile i overlaps mainloop of tile i+1:
     C = P + R*max(ML, E') + min(ML, E'),  ML = c_k'*Kt + c_0'.
  T8 clock, host gap, skew, post-store, tail: V02 values unchanged (no V03 rule at freeze time).

Usage: v04_predict.py --resources RUN/build/resources.json --output configs/v04-predictions.json
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[3]
V02_PREDICTIONS = Path(__file__).resolve().parent / "configs/v02-predictions.json"
R09 = REPO / "results/gh200_resource_campaign/access_rules/20261007-r09-clock-stages"
R05_SUPPLY = 55.45  # B/cycle/SM, R05-D shared small source, stage 2/4: 55.45/55.48 (R05 doc)
R01_WGMMA_SLOPE = 0.62  # cycle per N of WGMMA+wait0 dependent time, R01 (~36+0.62N)
SMS = 132
BASE = dict(tm=128, tn=256, n_wgmma=256)

CONFIGS = {
    "cfg_a": dict(tm=128, tn=128, cluster=(2, 1), schedule="cooperative", n_wgmma=128),
    "cfg_b": dict(tm=128, tn=128, cluster=(1, 1), schedule="pingpong", n_wgmma=128),
    "cfg_c": dict(tm=256, tn=128, cluster=(1, 2), schedule="cooperative", n_wgmma=128),
}

# (config, label, M, N, K): none of these (config, size) pairs has been run before.
TEST_SIZES = [
    ("cfg_a", "pad_fills_wave", 1408, 1408, 4096),   # 11x11=121 real, padded 12x11=132 -> R=1
    ("cfg_a", "pad_crosses_wave", 1400, 1536, 4096),  # 11x12=132 real, padded 144 -> R=2
    ("cfg_a", "just_over_wave", 2304, 1920, 3072),
    ("cfg_a", "long_k", 1024, 2048, 16384),
    ("cfg_a", "short_k", 4096, 3584, 512),
    ("cfg_a", "multi_wave", 6144, 6144, 2048),
    ("cfg_a", "large", 7168, 7168, 4096),
    ("cfg_b", "partial_wave", 1280, 1536, 4096),
    ("cfg_b", "just_over_wave", 1792, 1280, 3072),
    ("cfg_b", "long_k", 1024, 2048, 16384),
    ("cfg_b", "short_k", 4096, 3584, 512),
    ("cfg_b", "very_short_k", 4096, 4096, 128),
    ("cfg_b", "edges_odd", 2600, 3000, 2000),
    ("cfg_b", "multi_wave", 6144, 6144, 2048),
    ("cfg_c", "partial_wave", 1792, 2304, 4096),
    ("cfg_c", "pad_crosses_wave", 2560, 1664, 4096),
    ("cfg_c", "just_over_wave", 2560, 1792, 3072),
    ("cfg_c", "long_k", 1024, 3840, 16384),
    ("cfg_c", "short_k", 4096, 3584, 512),
    ("cfg_c", "edges_odd", 2600, 3000, 2000),
    ("cfg_c", "multi_wave", 6144, 6144, 2048),
    ("cfg_c", "large", 7168, 7168, 4096),
]


def ceil_div(a, b):
    return -(-a // b)


def prefill_setup_fraction():
    """R09 warm M=2048 traced cases: producer_setup_med / prefill_med (both ns)."""
    with (R09 / "cases.csv").open() as stream:
        rows = {(r["case"], r["call"]): r for r in csv.DictReader(stream)}
    cases = [f"trace_m2048_k{k}" for k in (256, 512, 1024, 2048, 4096)]
    return statistics.median(float(rows[(c, "warm")]["producer_setup_med"])
                             / float(rows[(c, "warm")]["prefill_med"]) for c in cases)


def mainloop_per_ktile(cfg, base_rules, bytes_mode):
    flops = 2 * cfg["tm"] * cfg["tn"] * 64
    compute = base_rules["c_k"]["value"] * flops / (2 * BASE["tm"] * BASE["tn"] * 64)
    delivered = (cfg["tm"] + cfg["tn"]) * 64 * 2
    cm, cn = cfg["cluster"]
    # Multicast: A tile is shared by the cn CTAs of a cluster row, B by the cm CTAs of a column.
    requested = cfg["tm"] * 64 * 2 / cn + cfg["tn"] * 64 * 2 / cm
    supply_bytes = delivered if bytes_mode == "delivered" else requested
    supply = supply_bytes / R05_SUPPLY
    return dict(compute_cycles=compute, delivered_bytes=delivered, requested_bytes=requested,
                needed_bytes_per_cycle=supply_bytes / compute, supply_cycles=supply,
                supply_bound=supply > compute, cycles=max(compute, supply))


def predict(base_rules, resources, config, m, n, k, bytes_mode="delivered"):
    cfg = CONFIGS[config]
    r = lambda name: base_rules[name]["value"]
    tiles_m, tiles_n, ktiles = ceil_div(m, cfg["tm"]), ceil_div(n, cfg["tn"]), ceil_div(k, 64)
    cm, cn = cfg["cluster"]
    units = cm * ceil_div(tiles_m, cm) * cn * ceil_div(tiles_n, cn)
    occupancy = resources[config]["max_active_ctas_per_sm"]
    resident = SMS  # CUTLASS persistent grid = sm_count CTAs, independent of occupancy
    rounds = ceil_div(units, resident)
    ml = mainloop_per_ktile(cfg, base_rules, bytes_mode)
    c_0 = r("c_0") - R01_WGMMA_SLOPE * (BASE["n_wgmma"] - cfg["n_wgmma"])
    mainloop = ml["cycles"] * ktiles + c_0
    out_bytes, base_out = cfg["tm"] * cfg["tn"] * 4, BASE["tm"] * BASE["tn"] * 4
    tile_extra = r("tile_extra") * out_bytes / base_out
    f_setup = prefill_setup_fraction()
    stage_bytes = (cfg["tm"] + cfg["tn"]) * 64 * 2
    prefill = r("prefill") * (f_setup + (1 - f_setup) * stage_bytes / 49152)
    if cfg["schedule"] == "pingpong":
        cycles = prefill + rounds * max(mainloop, tile_extra) + min(mainloop, tile_extra)
    else:
        cycles = prefill + rounds * (mainloop + tile_extra)
    clock = base_rules["clock"]
    ghz_of = lambda w: clock["a"] - clock["b"] * math.log(w)
    window = cycles / 1.8e3
    for _ in range(100):
        window = cycles / (ghz_of(window) * 1e3)
    device = r("entry_skew") + window + r("post_store") + r("tail")
    event = r("host_gap") + device
    return dict(config=config, m=m, n=n, k=k, tiles_m=tiles_m, tiles_n=tiles_n,
                tiles=tiles_m * tiles_n, units=units, rounds=rounds,
                ctas_at_max_rounds=units - (rounds - 1) * resident, ktiles=ktiles,
                occupancy_ctas_per_sm=occupancy, mainloop_cycles_per_ktile=ml["cycles"],
                compute_cycles_per_ktile=ml["compute_cycles"],
                needed_bytes_per_cycle=ml["needed_bytes_per_cycle"],
                supply_bound=ml["supply_bound"], c_0=c_0, mainloop_cycles_per_tile=mainloop,
                tile_extra_cycles=tile_extra, prefill_cycles=prefill, cta_cycles=cycles,
                clock_ghz=ghz_of(window), cta_window_us=window, device_us=device,
                host_gap_us=r("host_gap"), event_us=event,
                tflops=2.0 * m * n * k / event / 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = json.loads(V02_PREDICTIONS.read_text())
    rules = base["rules"]
    resources = json.loads(args.resources.read_text())
    for config, facts in resources.items():
        if facts["max_active_ctas_per_sm"] != 1:
            print(f"NOTE {config}: {facts['max_active_ctas_per_sm']} CTAs/SM possible; grid is "
                  "still 132 CTAs, model assumes they land on distinct SMs")
    predictions, variants = [], []
    for config, label, m, n, k in TEST_SIZES:
        predictions.append(dict(label=label, **predict(rules, resources, config, m, n, k)))
        alt = predict(rules, resources, config, m, n, k, bytes_mode="requested")
        variants.append(dict(label=label, config=config, event_us=alt["event_us"],
                             cta_cycles=alt["cta_cycles"],
                             mainloop_cycles_per_ktile=alt["mainloop_cycles_per_ktile"]))
    print(f"{'config':6} {'label':17} {'M':>5} {'N':>5} {'K':>5} {'U':>5} {'R':>3} {'Kt':>4} "
          f"{'cyc/Kt':>7} {'B/cyc':>6} {'E':>6} {'P':>6} {'cycles':>9} {'GHz':>6} {'T us':>8}")
    for p in predictions:
        print(f"{p['config']:6} {p['label']:17} {p['m']:5} {p['n']:5} {p['k']:5} {p['units']:5} "
              f"{p['rounds']:3} {p['ktiles']:4} {p['mainloop_cycles_per_ktile']:7.1f} "
              f"{p['needed_bytes_per_cycle']:6.1f} {p['tile_extra_cycles']:6.0f} "
              f"{p['prefill_cycles']:6.0f} {p['cta_cycles']:9.0f} {p['clock_ghz']:6.3f} "
              f"{p['event_us']:8.2f}")
    if args.output:
        args.output.write_text(json.dumps(dict(
            model=__doc__.split("Usage:")[0].strip(),
            generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            base_rules_source="configs/v02-predictions.json (V02 frozen rules)",
            base_rules=rules, prefill_setup_fraction=prefill_setup_fraction(),
            supply_bytes_per_cycle=R05_SUPPLY, resources=resources,
            predictions=predictions, variant_l2_request_bytes_not_scored=variants),
            indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

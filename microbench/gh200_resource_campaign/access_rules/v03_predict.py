#!/usr/bin/env python3
"""V03: V02 cycle model + a new in-call clock rule fitted on same-card calibration configs only.

Cycle model (V02, constants unchanged, read through v02_predict.load_rules()):
  tiles_m = ceil(M/128), tiles_n = ceil(N/256), Kt = ceil(K/64)
  U = scheduler units after cluster (2x1) and swizzle padding  (CUTLASS 3.9 tile_scheduler_params.h)
  G = persistent CTAs = min(sm_count or 132, U);  R = ceil(U/G)
  C = P + R*(c_k*Kt + c_0 + E)                                     critical-CTA cycles
  W = C / f(W, features)                                           critical-CTA window (fixed point)
  T_event = h + s + W + p + tau
Only change to V02's structure: U includes swizzle padding (R08 rule) and G may be < 132 (only
calibration configs use a reduced sm_count; held-out sizes use the default grid).

Clock-rule features (all predictable from M, N, K, swizzle, sm_count and W):
  lnW   = ln(W / us)
  phi   = time-averaged fraction of the 132 SMs busy: sum_i (P + n_i*t) / (132 * C)
  dram  = estimated DRAM bytes of one warm call / W, TB/s. Waves of G consecutive scheduler
          indices are simulated with the CUTLASS persistent mapping (raster, swizzle, cluster);
          each wave reads every A row-panel (128 x K) and B column-panel (256 x K) it touches once;
          a panel hits L2 if it is still resident from an earlier wave (LRU over whole panels,
          effective capacity L2_CAP; a wave whose own footprint exceeds L2_CAP flushes reuse).
          L2_CAP is chosen with the form from L2_CAPS (device l2CacheSize = 60 MiB and fractions;
          GH200's L2 is split into two partitions). The cache state carries over from the previous identical call
          (warm protocol). D (FP32, 4*M*N bytes) is always written to DRAM.
  mu    = mainloop share of the per-tile cycles: c_k*Kt / (c_k*Kt + c_0 + E)
Candidate forms (linear least squares in GHz) x L2_CAPS are compared by leave-one-out error on
the calibration configs; the selection rule is fixed in select_form() before any held-out measurement.

Usage:
  v03_predict.py design                                   calibration grid with the V02 rule
  v03_predict.py freeze --calibration RUN/calibration.csv --output configs/v03-predictions.json
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v02_predict  # noqa: E402  (V02 constants and clock rule, unchanged)
from v03_run import CALIBRATION  # noqa: E402

TILE_M, TILE_N, TILE_K = 128, 256, 64
CLUSTER_M = 2
SMS = 132
L2_BYTES = 62914560  # cudaDeviceProp.l2CacheSize on GH200 (R00 samples)
L2_CAPS = [L2_BYTES // 4, L2_BYTES // 3, L2_BYTES // 2, 2 * L2_BYTES // 3, L2_BYTES]

# Held-out sizes: never measured in R00/R07/R08/R09/V01/V02 (checked by check_new()) nor in the
# V03 calibration grid. (label, M, N, K, swizzle)
HELDOUT = [
    ("h_long_partial", 7168, 7168, 5120, 1),      # 1568 tiles, 12 rounds, last round 116
    ("h_tall_k6144", 12288, 4096, 6144, 1),       # 1536 tiles, 12 rounds, last 84
    ("h_swz8_long", 10240, 6144, 6144, 8),        # 1920 tiles, 15 rounds, last 72, swizzle 8
    ("h_r3_partial", 3584, 3584, 2560, 1),        # 392 tiles, 3 rounds, last 128
    ("h_r2_last22", 2816, 1792, 6144, 1),         # 154 tiles, 2 rounds, last 22
    ("h_r2_longk", 1280, 5120, 12288, 1),         # 200 tiles, 2 rounds, last 68
    ("h_r8_last44", 5632, 5632, 1536, 1),         # 968 tiles, 8 rounds, last 44
    ("h_huge_sw1", 14336, 14336, 2048, 1),        # 6272 tiles, 48 rounds, last 68
    ("h_r2_k32768", 2560, 2560, 32768, 1),        # 200 tiles, 2 rounds, long K
    ("h_small36", 768, 1536, 4096, 1),            # 36 tiles, 1 round, 36 CTAs
    ("h_wide_alongm", 4096, 12288, 3072, 1),      # 1536 tiles (raster along M), 12 rounds
]


def ceil_div(a, b):
    return -(-a // b)


def round_up(a, b):
    return ceil_div(a, b) * b


def log_swizzle(blocks_m, blocks_n, max_swizzle):
    """CUTLASS get_log_swizzle_size."""
    low = min(blocks_m, blocks_n)
    if max_swizzle >= 8 and low >= 6:
        return 3
    if max_swizzle >= 4 and low >= 3:
        return 2
    if max_swizzle >= 2 and low >= 2:
        return 1
    return 0


def geometry(m, n, k, swizzle=1, sm_count=0):
    tiles_m, tiles_n, ktiles = ceil_div(m, TILE_M), ceil_div(n, TILE_N), ceil_div(k, TILE_K)
    blocks_m = round_up(tiles_m, CLUSTER_M)
    log = log_swizzle(blocks_m, tiles_n, swizzle)
    blocks_m = round_up(blocks_m, (1 << log) * CLUSTER_M)
    blocks_n = round_up(tiles_n, 1 << log)
    units = blocks_m * blocks_n
    along_m = blocks_n > blocks_m  # Heuristic raster
    grid = min(sm_count or SMS, units)
    rounds = ceil_div(units, grid)
    tiles_per_cta = [len(range(i, units, grid)) for i in range(grid)]
    return dict(tiles_m=tiles_m, tiles_n=tiles_n, ktiles=ktiles, log_swizzle=log,
                blocks_m=blocks_m, blocks_n=blocks_n, units=units,
                raster="M" if along_m else "N", grid_ctas=grid, rounds=rounds,
                tiles_per_cta=tiles_per_cta,
                ctas_at_max_rounds=sum(1 for t in tiles_per_cta if t == rounds))


def tile_coord(linear, g):
    """CUTLASS StaticPersistentTileScheduler + get_work_idx_m_and_n for cluster 2x1."""
    s = 1 << g["log_swizzle"]
    if g["raster"] == "N":  # linear = blockIdx.x + blockIdx.y*2; minor = m, major = n
        in_cluster, cluster_id = linear % CLUSTER_M, linear // CLUSTER_M
        offset, extra = cluster_id & (s - 1), cluster_id >> g["log_swizzle"]
        minor_div, major = divmod(extra, g["blocks_n"])
        return (minor_div * s + offset) * CLUSTER_M + in_cluster, major
    in_cluster, cluster_id = linear % CLUSTER_M, linear // CLUSTER_M  # major = m, minor = n
    offset, extra = cluster_id & (s - 1), cluster_id >> g["log_swizzle"]
    minor_div, major = divmod(extra, g["blocks_m"] // CLUSTER_M)
    return major * CLUSTER_M + in_cluster, minor_div * s + offset


def dram_bytes(m, n, k, g, capacity=L2_BYTES):
    """Estimated DRAM bytes of one warm call (see module docstring)."""
    def panel_bytes(panel):
        kind, index = panel
        if kind == "A":
            return max(0, min(TILE_M, m - TILE_M * index)) * k * 2
        return max(0, min(TILE_N, n - TILE_N * index)) * k * 2

    waves = []
    for start in range(0, g["units"], g["grid_ctas"]):
        panels = set()
        for linear in range(start, min(start + g["grid_ctas"], g["units"])):
            tm, tn = tile_coord(linear, g)
            if tm < g["tiles_m"] and tn < g["tiles_n"]:
                panels.add(("A", tm))
                panels.add(("B", tn))
        waves.append(panels)
    cache = OrderedDict()
    total = 0
    for call in range(2):  # second call = the warm, timed call
        total = 0
        for panels in waves:
            footprint = sum(panel_bytes(p) for p in panels)
            total += sum(panel_bytes(p) for p in panels if p not in cache)
            for p in panels:
                cache.pop(p, None)
                cache[p] = panel_bytes(p)
            if footprint > capacity:
                cache.clear()
                continue
            while sum(cache.values()) > capacity:
                cache.popitem(last=False)
    return total + 4 * m * n


def cycle_model(rules, m, n, k, swizzle=1, sm_count=0, l2_cap=L2_BYTES):
    v = lambda name: rules[name]["value"]
    g = geometry(m, n, k, swizzle, sm_count)
    tile_cycles = v("c_k") * g["ktiles"] + v("c_0") + v("tile_extra")
    cta_cycles = v("prefill") + g["rounds"] * tile_cycles
    busy = sum(v("prefill") + t * tile_cycles for t in g["tiles_per_cta"])
    g.update(tile_cycles=tile_cycles, cta_cycles=cta_cycles, phi=busy / (SMS * cta_cycles),
             mu=v("c_k") * g["ktiles"] / tile_cycles, l2_cap=l2_cap,
             dram_bytes=dram_bytes(m, n, k, g, l2_cap))
    del g["tiles_per_cta"]
    return g


def features(g, window_us):
    return dict(lnW=math.log(window_us), phi=g["phi"], mu=g["mu"],
                dram=g["dram_bytes"] / (window_us * 1e-6) / 1e12)


# Candidate clock forms: GHz = a + sum(coef * feature).
FORMS = {
    "lnW": ["lnW"],
    "lnW+phi": ["lnW", "phi"],
    "lnW+dram": ["lnW", "dram"],
    "lnW+phi+dram": ["lnW", "phi", "dram"],
    "lnW+phi+dram+mu": ["lnW", "phi", "dram", "mu"],
    "phi*lnW+dram": ["phi*lnW", "dram"],
    "phi*lnW+phi+dram": ["phi*lnW", "phi", "dram"],
    "phi*lnW+dram+mu": ["phi*lnW", "dram", "mu"],
    "phi*lnW+phi+dram+mu": ["phi*lnW", "phi", "dram", "mu"],
}


def column(feats, name):
    if "*" in name:
        a, b = name.split("*")
        return feats[a] * feats[b]
    return feats[name]


def solve(rows, ys):
    """Least squares via normal equations (small systems)."""
    p = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(p)] for i in range(p)]
    aty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(p)]
    for i in range(p):  # Gauss-Jordan
        pivot = max(range(i, p), key=lambda q: abs(ata[q][i]))
        ata[i], ata[pivot], aty[i], aty[pivot] = ata[pivot], ata[i], aty[pivot], aty[i]
        for q in range(p):
            if q != i:
                factor = ata[q][i] / ata[i][i]
                ata[q] = [x - factor * y for x, y in zip(ata[q], ata[i])]
                aty[q] -= factor * aty[i]
    return [aty[i] / ata[i][i] for i in range(p)]


def fit_form(points, names):
    rows = [[1.0] + [column(f, nm) for nm in names] for f, _ in points]
    ys = [y for _, y in points]
    coef = solve(rows, ys)
    fitted = [sum(c * x for c, x in zip(coef, r)) for r in rows]
    residuals = [f / y - 1 for f, y in zip(fitted, ys)]
    loo = []
    for i in range(len(points)):
        c = solve(rows[:i] + rows[i + 1:], ys[:i] + ys[i + 1:])
        loo.append(sum(a * x for a, x in zip(c, rows[i])) / ys[i] - 1)
    return dict(features=names, coef=coef, residuals=residuals, loo=loo,
                rms=math.sqrt(statistics.fmean(r * r for r in residuals)),
                max_abs=max(abs(r) for r in residuals),
                loo_rms=math.sqrt(statistics.fmean(r * r for r in loo)),
                loo_max_abs=max(abs(r) for r in loo))


def select_form(fits):
    """Fixed rule: smallest LOO RMS; a form with fewer terms wins if its LOO RMS is within
    0.002 (0.2 %) of the best."""
    best = min(f["loo_rms"] for f in fits.values())
    ok = [name for name, f in fits.items() if f["loo_rms"] <= best + 0.002]
    return min(ok, key=lambda name: (len(fits[name]["features"]), fits[name]["loo_rms"]))


def clock_ghz(rule, feats):
    return rule["coef"][0] + sum(c * column(feats, nm)
                                 for c, nm in zip(rule["coef"][1:], rule["features"]))


def predict(rules, rule, label, m, n, k, swizzle=1, sm_count=0):
    """rule=None -> V02 clock rule (a - b ln W)."""
    v = lambda name: rules[name]["value"]
    g = cycle_model(rules, m, n, k, swizzle, sm_count,
                    rule["l2_cap"] if rule else L2_BYTES)
    window = g["cta_cycles"] / 1.7e3
    for _ in range(200):  # fixed point W = C / f(W); damped
        feats = features(g, window)
        ghz = (v02_predict.clock_ghz(rules, window) if rule is None else clock_ghz(rule, feats))
        window = 0.5 * window + 0.5 * g["cta_cycles"] / (ghz * 1e3)
    feats = features(g, window)
    ghz = v02_predict.clock_ghz(rules, window) if rule is None else clock_ghz(rule, feats)
    event = v("host_gap") + v("entry_skew") + window + v("post_store") + v("tail")
    return dict(label=label, m=m, n=n, k=k, swizzle=swizzle, sm_count=sm_count, **g,
                **{f"feat_{key}": val for key, val in feats.items()}, clock_ghz=ghz,
                cta_window_us=window, host_gap_us=v("host_gap"), event_us=event)


def measured_sizes():
    """(m, n, k) of every earlier access_rules sample (R00..V02)."""
    root = v02_predict.RESULTS
    sizes = set()
    for path in root.glob("**/samples.jsonl"):
        if "v03" in str(path):
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            args = row.get("args") or []
            if isinstance(args, list):
                pairs = {args[i]: args[i + 1] for i in range(len(args) - 1)
                         if args[i] in ("--m", "--n", "--k")}
                if len(pairs) == 3:
                    sizes.add(tuple(int(pairs[x]) for x in ("--m", "--n", "--k")))
            for key in ("setup", None):
                d = row.get(key) if key else row
                if isinstance(d, dict) and all(x in d for x in "mnk"):
                    try:
                        sizes.add((int(d["m"]), int(d["n"]), int(d["k"])))
                    except (TypeError, ValueError):
                        pass
    for path in root.glob("**/cases.csv"):
        if "v03" in str(path):
            continue
        with path.open() as stream:
            for row in csv.DictReader(stream):
                try:
                    sizes.add((int(float(row["m"])), int(float(row["n"])), int(float(row["k"]))))
                except (KeyError, ValueError, TypeError):
                    pass
    return sizes


def check_new():
    old = measured_sizes()
    calib = {(m, n, k) for _, m, n, k, _, _ in CALIBRATION}
    clash = [h for h in HELDOUT if (h[1], h[2], h[3]) in old | calib]
    if clash:
        raise ValueError(f"held-out sizes already measured: {clash}")
    return len(old)


def load_calibration(path: Path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def fit_rule(rules, calibration):
    """Features from the model structure at the measured critical-CTA window; target = measured
    critical-CTA clock. Every (form, L2 capacity) pair is fitted; select_form() picks one."""
    fits = {}
    for cap in L2_CAPS:
        points, labels = [], []
        for row in calibration:
            g = cycle_model(rules, int(row["m"]), int(row["n"]), int(row["k"]),
                            int(row["swizzle"]), int(row["sm_count"]), cap)
            points.append((features(g, float(row["window_us_meas"])), float(row["ghz_meas"])))
            labels.append(row["label"])
        for name, names in FORMS.items():
            f = fit_form(points, names)
            f["l2_cap"] = cap
            f["per_config"] = [dict(label=lb, feats=pt[0], ghz_meas=pt[1], residual=r, loo=lo)
                               for lb, pt, r, lo in zip(labels, points, f["residuals"], f["loo"])]
            fits[f"{name}@L2={cap / 2**20:.0f}MiB"] = f
    return select_form(fits), fits


def design():
    rules = v02_predict.load_rules()
    print(f"{'label':22} {'U':>6} {'G':>4} {'R':>3} {'Kt':>4} {'phi':>5} {'mu':>5} "
          f"{'DRAM GB':>8} {'W us':>8} {'TB/s':>5} {'GHz':>5}")
    sets = [(c[0], *c[1:]) for c in CALIBRATION]
    sets += [(h[0], *h[1:], 0) for h in HELDOUT]
    for label, m, n, k, s, g in sets:
        p = predict(rules, None, label, m, n, k, s, g)
        print(f"{label:22} {p['units']:6} {p['grid_ctas']:4} {p['rounds']:3} {p['ktiles']:4} "
              f"{p['phi']:5.2f} {p['mu']:5.2f} {p['dram_bytes'] / 1e9:8.2f} "
              f"{p['cta_window_us']:8.1f} {p['feat_dram']:5.2f} {p['clock_ghz']:5.3f}")
    print("previously measured sizes:", check_new())


def freeze(calibration_csv: Path, output: Path):
    check_new()
    rules = v02_predict.load_rules()
    calibration = load_calibration(calibration_csv)
    chosen, fits = fit_rule(rules, calibration)
    rule = dict(name=chosen, **{k: fits[chosen][k] for k in ("features", "coef", "l2_cap")})
    heldout = [predict(rules, rule, *h) for h in HELDOUT]
    secondary = [predict(rules, rule, label, m, n, k)
                 for label, m, n, k in v02_predict.TEST_SIZES]
    for p in secondary:
        p["v02_rule_event_us"] = predict(rules, None, p["label"], p["m"], p["n"], p["k"])[
            "event_us"]
    for f in fits.values():
        f.pop("residuals"), f.pop("loo")
    document = dict(
        model=__doc__.split("Usage:")[0].strip(),
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        calibration_csv=str(calibration_csv),
        calibration_csv_sha256=hashlib.sha256(calibration_csv.read_bytes()).hexdigest(),
        v02_rules={k: v for k, v in rules.items()}, clock_rule=rule, candidate_fits=fits,
        heldout=heldout,
        v02_secondary_note="V02 sizes re-predicted with the new rule; already measured in V02 "
                           "on another card, not held-out (secondary comparison only)",
        v02_secondary=secondary)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"rule {chosen}: features {rule['features']} coef {[round(c, 5) for c in rule['coef']]}")
    for name, f in sorted(fits.items(), key=lambda x: x[1]["loo_rms"])[:12]:
        print(f"  {name:32} rms {f['rms']:.4f} max {f['max_abs']:.4f} "
              f"loo_rms {f['loo_rms']:.4f} loo_max {f['loo_max_abs']:.4f}")
    for p in heldout + secondary:
        print(f"{p['label']:20} U {p['units']:5} R {p['rounds']:3} W {p['cta_window_us']:8.1f} "
              f"GHz {p['clock_ghz']:.3f} T {p['event_us']:8.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("design", "freeze"))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.step == "design":
        design()
    else:
        freeze(args.calibration, args.output)


if __name__ == "__main__":
    main()

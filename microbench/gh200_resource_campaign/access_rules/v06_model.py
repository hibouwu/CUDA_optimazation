#!/usr/bin/env python3
"""V06: per-CTA event-recursion cycle model + V03 clock rule with a card correction.

Cycle model (all intervals in SM cycles of one CTA, clock64 on its own SM):
  entry --P0--> producer first work --S--> first MMA of tile 0
  per tile j: fm[j] (first MMA) --L(Kt)--> me[j] (mainloop end, after mma_tail)
              ep[j] (epilogue permit) --E--> ed[j] (epilogue done)
  cooperative (both consumer WGs work on every tile):
      ep[j] = me[j] + w;  ed[j] = ep[j] + (E0 if j == 0 else E);  fm[j+1] = ed[j] + h
  pingpong (WG j%2 owns tile j; MMA order chain and epilogue order chain):
      fm[j] = max(me[j-1] + gm, ed[j-2] + h)      (j>=1; the ed term only for j>=2)
      ep[j] = max(me[j] + w,    ed[j-1] + we)     (the ed term only for j>=1)
      ed[j]: work E (alone) progressing at rate r while the other WG's mainloop of tile j+1
             [fm[j+1], me[j+1]] runs, rate 1 otherwise
  end = B + X, B = max(ed) + Etail; X = max-over-CTAs excess vs. the median-interval CTA,
      one of {0, x0, xk*B, x0 + x1*T} chosen per config by calibration leave-one-out
  L(Kt) = l0 + l1*Kt for tiles j>=1 (V05 linear-in-Kt form); tile 0 uses L(Kt) + dL0.
  C_cta = end - entry; critical C = max over CTAs (tile lists from the CUTLASS static scheduler).

Time:  T = F_cfg + C/f, f from the V03 rule form
  f = (a0 + da) - b*phi*ln(W/us) - (c0 + dc)*D - d*mu   GHz,   W = C/f (fixed point)
  a0=2.2287, b=0.03534, c0=0.0859, d=0.3872, L2 effective capacity 30 MiB (V03 frozen file).
  Card correction: da (and dc only if it lowers the calibration LOO RMS by > 0.2 pp).
  phi = sum_i C_i / (132 C), mu = T_crit*L / C, D = estimated DRAM bytes / W (V03 method
  generalised to the tile shape: per wave every touched A row panel and B column panel).
All parameters are fitted on calibration sizes only (stamped + plain runs, same card).
"""
from __future__ import annotations

from collections import OrderedDict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / "results/gh200_resource_campaign/access_rules"
SMS = 132
L2_CAP = 31457280  # V03 selected 30 MiB
# V03 frozen clock rule, configs/v03-predictions.json (SHA256 f73194c3...ffad), clock_rule.coef.
V03_RULE = dict(a=2.2287293009491083, b=0.0353398014693725, c=0.08590159393950943,
                d=0.3871575920181052)
TRACE_WORDS, HEAD, TILE_CAP = 16 + 2 * 64 * 4, 16, 64

CONFIGS = OrderedDict(
    cfg_a=dict(index=0, tile=(128, 128, 64), cluster=(2, 1), schedule="cooperative"),
    cfg_b=dict(index=1, tile=(128, 128, 64), cluster=(1, 1), schedule="pingpong"),
    cfg_c=dict(index=2, tile=(256, 128, 64), cluster=(1, 2), schedule="cooperative"),
)
# Calibration sizes (full 132-CTA grid for every config). Not held-out sizes.
CALIB = [
    ("c1_r_k512", 1536, 2816, 512),
    ("c2_k4096", 1536, 2816, 4096),
    ("c3_shortk", 3072, 5632, 256),
    ("c4_k2048", 3072, 5632, 2048),
    ("c5_k1024", 6144, 5632, 1024),
    ("c6_longk", 1536, 2816, 16384),
    ("c7_large", 6144, 11264, 2048),
    ("c8_tail", 2816, 2816, 1024),
    ("c9_thrash", 8192, 8192, 2048),  # per-wave panels exceed the 30 MiB L2 estimate
]
# Held-out sizes; padded=True -> lda/ldb/ldd rounded up to 128 B.
HELDOUT = [
    ("partial_wave", 1152, 1664, 2560, False),
    ("short_k", 3328, 4608, 384, False),
    ("mid_k", 4608, 3840, 1536, False),
    ("large_output", 7680, 6656, 2560, False),
    ("long_k", 1408, 2304, 10240, False),
    ("tails_pitched", 2984, 3432, 1704, True),
]


def cdiv(a, b):
    return -(-a // b)


def case_rows(kind):
    rows = []
    shapes = CALIB if kind == "calib" else HELDOUT
    for config in CONFIGS:
        for shape in shapes:
            label, m, n, k = shape[:4]
            padded = len(shape) > 4 and shape[4]
            rows.append(dict(id=f"{config}_{label}", config=config, label=label, m=m, n=n, k=k,
                             lda=cdiv(k, 64) * 64 if padded else k,
                             ldb=cdiv(n, 64) * 64 if padded else n,
                             ldd=cdiv(n, 32) * 32 if padded else n))
    return rows


# ---------------------------------------------------------------- scheduler (CUTLASS 3.9.2)
def emulate_grid(config, m, n):
    """Persistent grid used only for local design; frozen predictions use the queried grid."""
    tm, tn, _ = CONFIGS[config]["tile"]
    cm, cn = CONFIGS[config]["cluster"]
    nm, nn = cdiv(cdiv(m, tm), cm) * cm, cdiv(cdiv(n, tn), cn) * cn
    total = min(nm * nn, SMS // (cm * cn) * cm * cn)
    if nn <= nm:  # heuristic raster AlongN (tile_scheduler_params.h get_grid_shape)
        return [cm, total // cm, 1]
    return [total // cn, cn, 1]


def scheduled_work(config, m, n, grid):
    """Copied from v05_predict.scheduled_work (verified there against traced coordinates)."""
    tm, tn, _ = CONFIGS[config]["tile"]
    cm, cn = CONFIGS[config]["cluster"]
    gx, gy, gz = grid
    nm = cdiv(cdiv(m, tm), cm) * cm
    nn = cdiv(cdiv(n, tn), cn) * cn
    along_n = nn <= nm
    minor, major, major_blocks = (cm, cn, nn // cn) if along_n else (cn, cm, nm // cm)
    result = []
    for by in range(gy):
        for bx in range(gx):
            first = bx + by * gx if along_n else bx * gy + by
            offset = bx % cm if along_n else by % cn
            work = []
            for linear in range(first, nm * nn, gx * gy):
                cluster, major_offset = divmod(linear // minor, major)
                cluster_minor, cluster_major = divmod(cluster, major_blocks)
                mi = cluster_minor * minor + offset
                ma = cluster_major * major + major_offset
                work.append((mi, ma) if along_n else (ma, mi))
            result.append(work)
    tiles = [t for w in result for t in w]
    if len(tiles) != nm * nn or len(set(tiles)) != nm * nn:
        raise ValueError("scheduler mapping does not cover the padded tile grid")
    return result


def dram_bytes(config, m, n, k, work):
    """V03 estimator generalised to the tile shape; wave w = w-th tile of every CTA."""
    tm, tn, _ = CONFIGS[config]["tile"]

    def size(p):
        kind, i = p
        extent = (min(tm, m - tm * i) if kind == "A" else min(tn, n - tn * i))
        return max(0, extent) * k * 2

    waves = []
    for w in range(max(map(len, work))):
        panels = set()
        for tiles in work:
            if w < len(tiles):
                mi, ni = tiles[w]
                if mi * tm < m and ni * tn < n:
                    panels.update({("A", mi), ("B", ni)})
        waves.append(panels)
    cache, total = OrderedDict(), 0
    for _ in range(2):  # second identical call = the warm timed call
        total = 0
        for panels in waves:
            footprint = sum(size(p) for p in panels)
            total += sum(size(p) for p in panels if p not in cache)
            for p in sorted(panels):  # fixed order: set order follows the per-process hash seed
                cache.pop(p, None)
                cache[p] = size(p)
            if footprint > L2_CAP:
                cache.clear()
                continue
            while sum(cache.values()) > L2_CAP:
                cache.popitem(last=False)
    return total + 4 * m * n


# ---------------------------------------------------------------- trace parsing
def parse_trace(words, ctas):
    """Per-CTA events from the v06_trace.hpp layout (cycles; ns only at entry/final)."""
    out = []
    for c in range(ctas):
        r = words[c * TRACE_WORDS:(c + 1) * TRACE_WORDS]
        roles = []
        for role in range(2):
            fin_c, fin_ns, count = r[4 + 3 * role:7 + 3 * role]
            tiles = []
            for t in range(min(count, TILE_CAP)):
                base = HEAD + (role * TILE_CAP + t) * 4
                tiles.append(tuple(r[base:base + 4]))
            roles.append(dict(final_c=fin_c, final_ns=fin_ns, tiles=tiles))
        out.append(dict(entry_c=r[0], entry_ns=r[1], smid=r[2] - 1, prod_c=r[3], roles=roles))
    return out


def cta_timeline(cta, schedule):
    """Merge consumer roles into one CTA tile sequence of (fm, me, ep, ed) and the end stamp."""
    r1, r2 = cta["roles"]
    if schedule == "cooperative":
        if len(r1["tiles"]) != len(r2["tiles"]):
            raise ValueError("cooperative consumers disagree on tile count")
        tiles = [(min(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                 for a, b in zip(r1["tiles"], r2["tiles"])]
        end_c, end_ns = r2["final_c"], r2["final_ns"]  # thread 256: TMA-issuing thread
    else:
        tiles = []
        for j in range(len(r1["tiles"]) + len(r2["tiles"])):
            tiles.append((r1 if j % 2 == 0 else r2)["tiles"][j // 2])
        last = max((r1, r2), key=lambda r: r["final_c"] if r["tiles"] else 0)
        end_c, end_ns = last["final_c"], last["final_ns"]
    for t in tiles:
        if min(t) == 0:
            raise ValueError("missing tile stamp")
    return tiles, end_c, end_ns


def load_process(path):
    """One stdout.txt.gz -> setup, elapsed_us, check status, per-CTA timelines (stamped only)."""
    setup = call = check = None
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if not line.strip():
                continue
            ev = json.loads(line)
            kind = ev.get("event")
            if kind == "setup":
                setup = ev
            elif kind == "call":
                call = ev
            elif kind == "check":
                check = {k: v for k, v in ev.items() if not isinstance(v, list)}
    ctas = setup["grid"][0] * setup["grid"][1] * setup["grid"][2]
    rec = dict(setup=setup, elapsed_us=call["elapsed_us"], warmup_us=call["warmup_us"],
               check=check, ctas=None)
    if call.get("trace"):
        sched = CONFIGS[setup["config"]]["schedule"]
        ctas_out = []
        for cta in parse_trace(call["trace"], ctas):
            tiles, end_c, end_ns = cta_timeline(cta, sched)
            ctas_out.append(dict(entry_c=cta["entry_c"], entry_ns=cta["entry_ns"],
                                 prod_c=cta["prod_c"], smid=cta["smid"], tiles=tiles,
                                 end_c=end_c, end_ns=end_ns))
        rec["ctas"] = ctas_out
    return rec


def process_summary(rec):
    """Critical CTA cycles, window and in-call clock of one stamped process."""
    ctas = [c for c in rec["ctas"] if c["tiles"]]
    spans = [c["end_c"] - c["entry_c"] for c in ctas]
    windows = [(c["end_ns"] - c["entry_ns"]) for c in ctas]
    tmax = max(len(c["tiles"]) for c in ctas)
    crit = [c for c in ctas if len(c["tiles"]) == tmax]
    ghz = statistics.median((c["end_c"] - c["entry_c"]) / (c["end_ns"] - c["entry_ns"])
                            for c in crit)
    first_entry = min(c["entry_ns"] for c in rec["ctas"])
    last_end = max(c["end_ns"] for c in ctas)
    return dict(c_max=max(spans), window_us=max(windows) / 1e3, ghz=ghz, tiles_max=tmax,
                span_us=(last_end - first_entry) / 1e3)


# ---------------------------------------------------------------- recursion
def lmain(p, kt, first=False):
    return p["l0"] + p["l1"] * kt + (p["dL0"] if first else 0.0)


def epilogue_end(p, start, overlap):
    """Pingpong epilogue: E_alone of work, progress rate r while the other consumer WG's
    mainloop interval `overlap` = (a, b) is active, rate 1 otherwise."""
    work, t = p["E"], start
    if overlap is None:
        return t + work
    a, b = overlap
    if t < a:
        if t + work <= a:
            return t + work
        work -= a - t
        t = a
    if t < b:
        if work / p["r"] <= b - t:
            return t + work / p["r"]
        work -= p["r"] * (b - t)
        t = b
    return t + work


def cta_cycles(p, schedule, ntiles, kt, detail=False):
    """Event recursion for one CTA with ntiles output tiles; returns C (entry->end)."""
    if ntiles == 0:
        return (p["P0"], None) if detail else p["P0"]
    fm, me, ep, ed = [], [], [], []

    def start_mma(j):
        if j == 0:
            return p["P0"] + p["S"]
        if schedule == "cooperative":
            return ed[j - 1] + p["h"]
        f = me[j - 1] + p["gm"]
        return max(f, ed[j - 2] + p["h"]) if j >= 2 else f

    fm.append(start_mma(0))
    me.append(fm[0] + lmain(p, kt, True))
    for j in range(ntiles):
        if schedule == "cooperative":
            ep.append(me[j] + p["w"])
            ed.append(ep[j] + (p["E0"] if j == 0 else p["E"]))
            if j + 1 < ntiles:
                fm.append(start_mma(j + 1))
                me.append(fm[j + 1] + lmain(p, kt))
            continue
        e = me[j] + p["w"]
        if j >= 1:
            e = max(e, ed[j - 1] + p["we"])
        ep.append(e)
        if j + 1 < ntiles:  # next tile's MMA needs me[j] and ed[j-1] only
            fm.append(start_mma(j + 1))
            me.append(fm[j + 1] + lmain(p, kt))
        ed.append(epilogue_end(p, ep[j], (fm[j + 1], me[j + 1]) if j + 1 < ntiles else None))
    base = max(ed) + p["Etail"]
    excess = p["x0"] + p["x1"] * ntiles + p["xk"] * base
    end = base + excess
    if not detail:
        return end
    supply = p["P0"] + p["S"]
    mainloop = sum(lmain(p, kt, j == 0) for j in range(ntiles))
    last_epi = ed[-1] - ep[-1]
    return end, dict(supply=supply, mainloop=mainloop, last_epilogue=last_epi,
                     tail=p["Etail"], max_cta_excess=excess,
                     handoff_exposed=base - supply - mainloop - last_epi - p["Etail"])


def clock_ghz(rule, phi, lnw, dram, mu):
    return (rule["a"] - rule["b"] * phi * lnw - rule["c"] * dram - rule["d"] * mu)


def predict_case(row, grid, params, rule, fixed_us):
    cfg = CONFIGS[row["config"]]
    work = scheduled_work(row["config"], row["m"], row["n"], grid)
    kt = cdiv(row["k"], 64)
    counts = [len(w) for w in work]
    cyc = [cta_cycles(params, cfg["schedule"], t, kt) for t in counts]
    crit_tiles = max(counts)
    c_max, stages = cta_cycles(params, cfg["schedule"], crit_tiles, kt, detail=True)
    c_max = max(cyc + [c_max])
    phi = sum(cyc) / (SMS * c_max)
    mu = stages["mainloop"] / c_max
    dbytes = dram_bytes(row["config"], row["m"], row["n"], row["k"], work)
    w = c_max / 1.6e3
    for _ in range(200):
        f = clock_ghz(rule, phi, math.log(w), dbytes / (w * 1e-6) / 1e12, mu)
        w = 0.5 * w + 0.5 * c_max / (f * 1e3)
    f = clock_ghz(rule, phi, math.log(w), dbytes / (w * 1e-6) / 1e12, mu)
    return dict(tiles=sum(counts), grid=grid, ctas=len(work), rounds=crit_tiles,
                ctas_at_max=sum(1 for t in counts if t == crit_tiles), ktiles=kt,
                critical_cycles=c_max, stage_cycles=stages, phi=phi, mu=mu,
                dram_gb=dbytes / 1e9, dram_tbs=dbytes / (w * 1e-6) / 1e12,
                window_us=w, clock_ghz=f, fixed_us=fixed_us, predicted_us=fixed_us + w)


# ---------------------------------------------------------------- novelty of held-out sizes
def _visit(value, found):
    if isinstance(value, dict):
        shape = tuple(value.get(k) for k in ("m", "n", "k"))
        try:
            if all(x is not None for x in shape):
                found.add(tuple(int(float(x)) for x in shape))
        except (TypeError, ValueError):
            pass
        args = value.get("args") or value.get("command")
        if isinstance(args, list):
            opts = dict(zip(args[::1], args[1::1]))
            if all(k in opts for k in ("--m", "--n", "--k")):
                try:
                    found.add(tuple(int(opts[k]) for k in ("--m", "--n", "--k")))
                except ValueError:
                    pass
        for key, child in value.items():
            if key not in ("trace", "checked_indices", "checked_values", "warmup_us"):
                _visit(child, found)
    elif isinstance(value, list):
        if len(value) == 3 and all(isinstance(x, int) for x in value):
            found.add(tuple(value))  # e.g. prior_shapes.json entries
        for item in value:
            if isinstance(item, (dict, list)):
                _visit(item, found)


def prior_shapes(roots=(RESULTS, HERE / "configs")):
    """(m, n, k) of every earlier shape found in logs, case lists and prediction files."""
    found = set()
    for root in roots:
        for path in Path(root).rglob("*"):
            if "V06" in str(path) or not path.is_file():
                continue
            name = path.name
            try:
                if name.endswith(".jsonl"):
                    for line in path.open():
                        if line.strip():
                            try:
                                _visit(json.loads(line), found)
                            except json.JSONDecodeError:
                                pass
                elif name.endswith(".json") and path.stat().st_size < 50e6:
                    try:
                        _visit(json.loads(path.read_text()), found)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        text = path.read_text(errors="ignore")
                        for m in re.finditer(r'"--m",\s*"(\d+)",\s*"--n",\s*"(\d+)",\s*"--k",\s*"(\d+)"', text):
                            found.add(tuple(map(int, m.groups())))
                elif name.endswith(".csv"):
                    with path.open() as stream:
                        for row in csv.DictReader(stream):
                            try:
                                found.add(tuple(int(float(row[x])) for x in ("m", "n", "k")))
                            except (KeyError, ValueError, TypeError):
                                pass
            except OSError:
                pass
    return found


def check_novelty():
    old = prior_shapes()
    calib = {(m, n, k) for _, m, n, k in CALIB}
    clash = [h for h in HELDOUT if (h[1], h[2], h[3]) in old | calib]
    return len(old), clash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    n, clash = check_novelty()
    print("prior shapes found:", n, "held-out clashes:", clash)
    for kind in ("calib", "heldout"):
        for row in case_rows(kind):
            grid = emulate_grid(row["config"], row["m"], row["n"])
            work = scheduled_work(row["config"], row["m"], row["n"], grid)
            counts = [len(w) for w in work]
            print(f"{kind:7} {row['id']:24} {row['m']:6}x{row['n']:6}x{row['k']:6} grid {grid} "
                  f"tiles {sum(counts):5} rounds {max(counts):3} at_max {counts.count(max(counts)):3}"
                  f" kt {cdiv(row['k'], 64):4} ld {row['lda']},{row['ldb']},{row['ldd']}")

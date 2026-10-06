#!/usr/bin/env python3
"""Write the V01 prediction for the asynchronous target TC and the two CUTLASS held-out points.

Every input is read from an existing R00-R06 / EXP result file (path recorded next to the value);
no measurement of the new kernel is used.  The output file is created exclusively and must not
be edited after the held-out runs start.

  python3 v01_predict.py --output configs/v01-predictions-async.json
"""

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / "results/gh200_resource_campaign"
R00 = RESULTS / "access_rules/20261006-r00-job734996/formal-v1/cases.csv"
R01 = RESULTS / "access_rules/20261006-b-job735060/formal-r01-v1/rules.json"
R03 = RESULTS / "access_rules/20261006-c-job735059/r03-formal-v1/rules.json"
R05 = RESULTS / "access_rules/20261006-c-job735059/r05-formal-v2/rules.json"
R05A = RESULTS / "access_rules/20261006-c-job735059/r05-a-formal-v5/rules.json"
EXP15 = (
    RESULTS / "20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/"
    "node-sampling-v1/published-all68-v1/results.csv"
)
EXP11_DOC = "Docs/ModelEvaluation/gemm/experiments/gh200_sm90/EXP-11-sync.md"
EXP17_DOC = "Docs/ModelEvaluation/gemm/experiments/gh200_sm90/EXP-17-cluster.md"

KTILE_FLOP = 2 * 128 * 256 * 64  # 4194304
STAGE_BYTES = (128 + 256) * 64 * 2  # 49152
HALF_BYTES = 64 * 256 * 4  # 65536
SM_COUNT = 132
WAVES = 4  # grid 4 x 132, 1 CTA/SM (192 KiB + barriers of dynamic SMEM)


def rel(path):
    return str(Path(path).resolve().relative_to(REPO))


def csv_row(path, key, value):
    with open(path) as f:
        for row in csv.DictReader(f):
            if row[key] == value:
                return row
    raise KeyError(value)


def rule_entry(path, case_id):
    for entry in json.loads(Path(path).read_text()):
        if entry["configuration"]["id"] == case_id:
            return entry
    raise KeyError(case_id)


def collect_inputs():
    """Each input: value, unit, source file and the measured condition it comes from."""
    inputs = {}

    def put(name, value, unit, source, condition):
        inputs[name] = dict(value=value, unit=unit, source=source, condition=condition)

    row = csv_row(R00, "case_id", "wg_fp16_SS_n256_g2_one_cta")
    put(
        "wgmma_one_cta",
        float(row["median_rate"]),
        "FLOP/cycle/CTA",
        rel(R00),
        "R00-B FP16 SS m64n256k16, 2 warpgroups, wait1, one CTA",
    )
    row = csv_row(R00, "case_id", "wg_fp16_SS_n256_g2_all_gpu")
    put(
        "wgmma_all_gpu",
        float(row["median_rate"]) * 1e9,
        "FLOP/s/GPU",
        rel(R00),
        "R00-B same shape, all-GPU globaltimer envelope",
    )
    row = csv_row(R00, "case_id", "cutlass_fp16_4096_repeat")
    put(
        "cutlass_4096_ms",
        float(row["median_time"]),
        "ms",
        rel(R00),
        "R00-A fixed CUTLASS 128x256x64, cluster 2x1x1, 4096^3, repeat input, CUDA event",
    )

    fits = json.loads(R01.read_text())["fits"]["wgmma_n256"]
    put(
        "wgmma_dependent_step",
        fits["cycles_per_step"],
        "cycle/step",
        rel(R01),
        "R01 dependent m64n256k16 chain: issue + execution + completion per step",
    )

    d = rule_entry(R05, "D_s4_shared")
    put(
        "supply_shared",
        d["cta_rate_median"],
        "B/cycle/SM",
        rel(R05),
        "R05-D 1 CTA/SM, 4 x 48 KiB stages in flight, all CTAs share an L2/4 source, no compute",
    )
    d = rule_entry(R05, "D_s4_independent")
    put(
        "supply_independent",
        d["cta_rate_median"],
        "B/cycle/SM",
        rel(R05),
        "R05-D same, independent sources totalling 4 x L2",
    )
    e = rule_entry(R05, "E_pitch256")
    put(
        "tma_store_all_gpu",
        e["work_counts_per_process"]["requested_bytes"] / e["mean_window"],
        "B/ns/GPU",
        rel(R05),
        "R05-E all-GPU TMA 2D store, 16 KiB boxes, 128 B rows at 256 B pitch",
    )
    a = rule_entry(R05A, "A_tin_f0")
    put(
        "tma_load_16k_ready",
        a["mean_wait_return_cycles"],
        "cycle",
        rel(R05A),
        "R05-A one 16 KiB TMA load (shared source): request preparation -> mbarrier wait return",
    )

    for name, case in (("sts_4warps", "store_b16_w4"), ("sts_8warps", "store_b16_w8")):
        r = rule_entry(R03, case)
        put(
            name,
            r["mean_requested_bytes_per_cycle"],
            "B/cycle/CTA",
            rel(R03),
            f"R03 {case}: 16 B shared stores, {case[-1]} warps",
        )

    row = csv_row(EXP15, "case_id", "smem_to_gmem_64kib_continuous_none_one_cta")
    put(
        "tma_store_64k_one_cta",
        float(row["median"]),
        "B/cycle/CTA",
        rel(EXP15),
        "EXP-15 one CTA, one 64 KiB TMA 2D store in flight, commit + wait_group 0 + CTA barrier",
    )

    put("cta_barrier", 29.3, "cycle", EXP11_DOC, "EXP-11 CTA barrier, 256 threads, aligned arrival")
    put("proxy_fence", 10.9, "cycle", EXP11_DOC, "EXP-11 fence.proxy.async, no pending accesses")
    put("cluster_sync", 190.0, "ns", EXP17_DOC, "EXP-17 single-cluster full sync phase")
    return inputs


def predict(inputs):
    v = {k: x["value"] for k, x in inputs.items()}
    # Derived quantities.
    ktile_cycles = KTILE_FLOP / v["wgmma_one_cta"]  # WGMMA service per Ktile, 2 warpgroups
    clock_hz = v["wgmma_all_gpu"] / (SM_COUNT * v["wgmma_one_cta"])  # SM clock under WGMMA load
    ktile_demand = STAGE_BYTES / ktile_cycles  # B/cycle needed to keep WGMMA busy
    fill = v["cta_barrier"] + v["tma_load_16k_ready"] + (STAGE_BYTES - 16384) / v["supply_shared"]
    drain = (v["wgmma_dependent_step"] - 128.0) + v["cta_barrier"]  # 128 = one MMA at peak
    sts_half = HALF_BYTES / v["sts_4warps"]
    publish = v["proxy_fence"] + v["cta_barrier"]
    store_half_one = HALF_BYTES / v["tma_store_64k_one_cta"]  # includes wait0 + barrier
    output_one = 2 * (sts_half + publish + store_half_one)
    derived = dict(
        ktile_cycles=ktile_cycles,
        clock_hz=clock_hz,
        ktile_ns=ktile_cycles / clock_hz * 1e9,
        ktile_demand_bytes_per_cycle=ktile_demand,
        supply_limited=ktile_demand > v["supply_shared"],
        fill_cycles=fill,
        drain_cycles=drain,
        output_one_cta_cycles=output_one,
    )
    cases = []
    ns = 1e9 / clock_hz
    for k in (7, 19, 47):
        mainloop = k * ktile_cycles + drain
        total = fill + mainloop + output_one
        cases.append(
            dict(
                case_id=f"target_tc_k{k}_one_cta",
                scope="one_cta",
                length=k,
                predicted_time=total,
                unit="clock64_cycle/CTA",
                segments=dict(fill=fill, mainloop=mainloop, output=output_one),
                predicted_mainloop_cycles_per_ktile=mainloop / k,
                rules=["R00-B", "R01", "R05-A", "R05-D", "R03", "EXP-11", "EXP-15"],
                missing=[
                    "8 B STS rate taken from R03 16 B stores (4 warps)",
                    "8 x 8 KiB SW128 boxes in one bulk group taken as one 64 KiB request (EXP-15)",
                    "single-CTA supply taken from R05-D all-SM value (lower bound)",
                    "SMEM shared by WGMMA SS operand reads (~80 B/cycle) and TMA writes "
                    "(48 B/cycle): no joint-service rule, assumed independent",
                ],
            )
        )
    # All-GPU: 4 waves of 132 CTAs, identical work, assumed to run in lockstep per wave.
    store_gpu_ns = SM_COUNT * HALF_BYTES / v["tma_store_all_gpu"]
    store_one_ns = store_half_one * ns
    output_gpu_ns = 2 * ((sts_half + publish) * ns + max(store_gpu_ns, store_one_ns))
    derived.update(output_half_store_all_gpu_ns=store_gpu_ns, output_all_gpu_cta_ns=output_gpu_ns)
    for k in (7, 19, 47):
        cta = fill * ns + (k * ktile_cycles + drain) * ns + output_gpu_ns
        cases.append(
            dict(
                case_id=f"target_tc_k{k}_all_gpu",
                scope="all_gpu",
                length=k,
                predicted_time=WAVES * cta,
                unit="globaltimer_ns/GPU",
                segments=dict(per_cta_ns=cta, waves=WAVES, relaunch_gap_ns=0.0),
                rules=[
                    "R00-B all-GPU clock",
                    "R05-D",
                    "R05-E",
                    "R05-A",
                    "R01",
                    "R03",
                    "EXP-11",
                    "EXP-15",
                ],
                missing=[
                    "CTA relaunch gap between waves and first-wave launch skew: "
                    "no rule, taken as 0",
                    "SMEM shared by WGMMA operand reads and TMA writes: assumed independent",
                    "clock during output/fill assumed equal to the WGMMA-loaded clock",
                    "all 132 CTAs of a wave store concurrently (lockstep assumption)",
                ],
            )
        )
    # Complete CUTLASS kernel, M=N=2048: 16 x 8 = 128 tiles <= 132 SMs -> one tile per CTA.
    tiles = (2048 // 128) * (2048 // 256)
    epi_store_ns = max(
        tiles * 2 * HALF_BYTES / v["tma_store_all_gpu"],
        2 * HALF_BYTES / v["tma_store_64k_one_cta"] * ns,
    )
    epi_first_r2s_ns = (128 * 32 * 4) / v["sts_8warps"] * ns  # first 128x32 FP32 subtile
    anchor_ktile_ns = v["cutlass_4096_ms"] * 1e6 / (4 * 64)  # 4 tile rounds x 64 Ktiles
    for k in (2048, 8192):
        ktiles = k // 64
        primary = (
            v["cluster_sync"]
            + fill * ns
            + (ktiles * ktile_cycles + drain) * ns
            + epi_store_ns
            + epi_first_r2s_ns
        )
        cases.append(
            dict(
                case_id=f"cutlass_2048_2048_{k}",
                scope="all_gpu",
                m=2048,
                n=2048,
                k=k,
                predicted_time=primary / 1e6,
                unit="cuda_event_ms/GPU",
                model="primary: WGMMA-bound mainloop (supply >= demand) + fill + epilogue",
                components_ns=dict(
                    cluster_sync=v["cluster_sync"],
                    fill=fill * ns,
                    mainloop=(ktiles * ktile_cycles + drain) * ns,
                    epilogue=epi_store_ns + epi_first_r2s_ns,
                    launch_and_event=0.0,
                ),
                tiles=tiles,
                rounds=1,
                alternative=dict(
                    model="R00-A anchored: 4096^3 time per (round x Ktile) applied to 1 round",
                    ktile_ns=anchor_ktile_ns,
                    predicted_time=ktiles * anchor_ktile_ns / 1e6,
                ),
                rules=[
                    "R00-B",
                    "R05-D",
                    "R05-E",
                    "EXP-15",
                    "R03",
                    "R01",
                    "R05-A",
                    "EXP-17",
                    "R00-A (alternative)",
                ],
                missing=[
                    "launch + CUDA event overhead: no rule, taken as 0",
                    "B MN-major (.tnspB) and cluster multicast assumed to keep R00-B/R05-D rates",
                    "setmaxnreg producer/consumer split and persistent scheduler not modelled",
                    "SMEM shared by WGMMA operand reads and TMA writes: assumed independent",
                    "previous measurements of these two sizes (job 735062) were known when "
                    "this file was written; the model does not use them",
                ],
            )
        )
    return derived, cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    inputs = collect_inputs()
    derived, cases = predict(inputs)
    probe = HERE / "probes/v01.cu"
    plan = dict(
        status="frozen prediction; written before any timed run of target_async",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        generator=dict(
            script=rel(__file__), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
        probe=dict(
            path=rel(probe),
            sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
            function="target_async",
        ),
        formulas=dict(
            ktile="T_k = 4194304 FLOP / wgmma_one_cta; supply check 49152 B / T_k <= supply_shared",
            one_cta="fill + (K*T_k + drain) + output",
            fill="cta_barrier + tma_load_16k_ready + 32768 B / supply_shared",
            drain="(wgmma_dependent_step - 128) + cta_barrier",
            output="2 * (65536/sts_4warps + proxy_fence + cta_barrier"
            " + 65536/tma_store_64k_one_cta)",
            all_gpu="4 waves * [ (fill + K*T_k + drain)/f + 2*((sts+publish)/f + "
            "max(132*65536/tma_store_all_gpu, 65536/tma_store_64k_one_cta/f)) ], "
            "f = wgmma_all_gpu / (132 * wgmma_one_cta)",
            cutlass="cluster_sync + fill/f + (K/64*T_k + drain)/f + epilogue + launch(0); "
            "epilogue = max(128 tiles*128 KiB/tma_store_all_gpu, 128 KiB/tma_store_64k/f) "
            "+ first 16 KiB subtile / sts_8warps / f",
        ),
        prior_knowledge=[
            "2026-10-06 serialized target kernel results at 7/19/47 (different kernel)",
            "CUTLASS 2048^3 31.248 us and 2048x2048x8192 97.952 us on GPU-099dda56 (job 735062)",
        ],
        inputs=inputs,
        derived=derived,
        cases=cases,
    )
    with args.output.open("x") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest(), plan["created_utc"])


if __name__ == "__main__":
    main()

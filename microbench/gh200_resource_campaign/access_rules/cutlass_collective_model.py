#!/usr/bin/env python3
"""Freeze a phase-composed prediction for the fixed, original CUTLASS GPU kernel.

L0 supplies a compute lower bound. Matched collective measurements supply its net joint
increment, including transport/control, so those costs are not added a second time.
Output-source reuse and the edge to CUDA-event completion have different boundaries.
"""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from analyze_v01 import table


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    groups = {}
    evidence = {}
    for f in sorted((root / "phase-pair/samples").glob("phases_*/*/result.json")):
        r = json.loads(f.read_text())
        k = r["k"] // 64
        if r["m"] != 2048 or r["n"] != 2048 or r["seed"] != 17:
            raise ValueError("calibration conditions")
        if sha(f.parent / "diag.json") != r["diag_sha256"]:
            raise ValueError("trace identity")
        with gzip.open(f.parent / "diag_output.f32.gz", "rb") as src:
            data = src.read()
        if hashlib.sha256(data).hexdigest() != r["full_output_sha256"]:
            raise ValueError("full output identity")
        output = np.frombuffer(data, dtype="<f4").reshape(2048, 2048)
        expected = table(r["k"])[np.ix_(np.arange(2048) % 17, np.arange(2048) % 17)]
        if not np.all(output == expected):
            raise ValueError("full output correctness")
        trace = np.array(json.loads((f.parent / "diag.json").read_text()), dtype=np.uint64).reshape(-1, 2, 4, 3)
        active = trace[trace[:, :, 0, 0] != 0]
        if len(active) != 256 or not np.all(active[:, :, 2] == active[:, :1, 2]):
            raise ValueError("consumer/SM identity")
        if not np.all(active[:, :-1, 0] <= active[:, 1:, 0]):
            raise ValueError("phase order")
        main_c = active[:, 1, 0] - active[:, 0, 0]
        main_n = active[:, 1, 1] - active[:, 0, 1]
        epi_n = active[:, 3, 1] - active[:, 1, 1]
        envelope = int(active[:, 3, 1].max() - active[:, 0, 1].min())
        main = statistics.median(main_n)
        epi = statistics.median(epi_n)
        edge = r["elapsed_ms"] * 1e6 - envelope
        if edge < 0:
            raise ValueError("event excludes consumer window")
        groups.setdefault(k, []).append(dict(main_cycles=statistics.median(main_c),
            main_ratio=statistics.median(main_c / main_n), output_source_ns=epi,
            wave_slack_ns=envelope-main-epi, full_completion_edge_ns=edge))
        evidence[str(f.relative_to(root))] = sha(f)
    if set(groups) != {32, 128} or any(len(rows) != 10 for rows in groups.values()):
        raise ValueError("expected ten phase processes at each known calibration length")
    medians = {k: {key: statistics.median(r[key] for r in rows) for key in rows[0]}
               for k, rows in groups.items()}
    return medians, evidence


def predict(parameters, k):
    if k % 256 or not 2048 <= k <= 8192:
        raise ValueError("validated candidate domain: M=N=2048, K multiple256 in2048..8192")
    tiles = k // 64
    ratio = parameters["main_ratio_at32"] + (tiles-32) / 96 * (parameters["main_ratio_at128"]-parameters["main_ratio_at32"])
    cycles = parameters["main_start_cycles"] + tiles * (parameters["core_floor_cycles_per_tile"] + parameters["joint_increment_cycles_per_tile"])
    pieces = dict(mainloop_ns=cycles/ratio, output_source_reuse_ns=parameters["output_source_ns"],
                  wave_slack_ns=parameters["wave_slack_ns"], edge_to_full_completion_ns=parameters["full_completion_edge_ns"])
    return dict(case_id=f"cutlass_2048_2048_{k}", m=2048, n=2048, k=k, predicted_ns=sum(pieces.values()),
                pieces_ns=pieces, mainloop_cycles=cycles, main_cycles_per_ns=ratio)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.diagnostics.resolve()
    med, evidence = collect(root)
    rows = json.loads((args.layout / "layout-summary.json").read_text())
    rates = [r["work_flop"] / r["elapsed"] for r in rows if r["b_major"] == "MN" and r["scope"] == "one_cta"]
    core = 4194304 / statistics.median(rates)
    slope = (med[128]["main_cycles"]-med[32]["main_cycles"]) / 96
    parameters = dict(core_floor_cycles_per_tile=core, joint_increment_cycles_per_tile=slope-core,
                      main_start_cycles=med[32]["main_cycles"]-32*slope,
                      main_ratio_at32=med[32]["main_ratio"], main_ratio_at128=med[128]["main_ratio"])
    for key in ("output_source_ns", "wave_slack_ns", "full_completion_edge_ns"):
        parameters[key] = statistics.mean(med[k][key] for k in (32,128))
    if parameters["joint_increment_cycles_per_tile"] < 0:
        raise ValueError("matched collective below the declared core lower bound")
    equivalence = json.loads((root / "full-output/gpu-code-equivalence.json").read_text())
    if not equivalence["entire_device_kernel_identical"]:
        raise ValueError("host capture changes GPU kernel")
    plan = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                model="matched collective and completion-edge composition; no total-GEMM affine fit",
                gpu_uuid=json.loads((root / "phase-pair/environment.json").read_text())["gpu_uuid"],
                binary_sha256=sha(root / "full-output/build/r00_cutlass"),
                device_code_equivalence=equivalence,
                generator_sha256=sha(Path(__file__)), calibration_medians=med,
                calibration_evidence=evidence, layout_evidence_sha256=sha(args.layout / "layout-summary.json"),
                parameters=parameters,
                conditions=dict(m=2048,n=2048,tile=[128,256,64],cluster=[2,1,1],stages=4,
                                input="FP16 row-major coordinate dyadic seed17", accumulation="FP32", output="FP32",
                                cache="repeat", grid_ctas=128, waves=1, build="original assertions enabled / C7510"),
                limits=["Joint increment is an aggregate collective cost, not an identified bare-port latency.",
                        "Cycle/ns ratios are matched window conversions, not clock telemetry.",
                        "The final edge includes initialization/dispatch and the gap from source reuse to full completion.",
                        "Ktile0 mod4; no cross-shape, device, tile, stage, cluster or build transfer."],
                cases=[predict(parameters,k) for k in (5120,6144)])
    with args.output.open("x") as f:
        json.dump(plan,f,indent=2);f.write("\n")
    print(sha(args.output))
    for c in plan["cases"]:
        print(c["case_id"],c["predicted_ns"],c["pieces_ns"])


if __name__ == "__main__":
    main()

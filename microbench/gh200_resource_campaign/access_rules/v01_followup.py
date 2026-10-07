#!/usr/bin/env python3
"""Freeze predictions for three new V01 points, then reuse the unchanged archived binaries.

--prepare BASE --layout LAYOUT --predictions FILE creates the prediction exactly once.
--run BASE --predictions FILE --prediction-sha256 SHA --output NEW runs only those three points.
CPU replay: analyze_v01.py --input NEW --output REVIEW.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

import run_v01
import v01_predict

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def interpolate(rows, first, second, fraction):
    return rows[first] + fraction * (rows[second] - rows[first])


def prepare(base, layout, path):
    old = json.loads((base / "predictions.json").read_text())
    records = json.loads((layout / "layout-summary.json").read_text())
    env = json.loads((base / "environment.json").read_text())
    if any(r["max_output_error"] != 0 for r in records):
        raise ValueError("layout correctness")
    if json.loads((layout / "layout-environment.json").read_text())["gpu_uuid"] != env["gpu_uuid"]:
        raise ValueError("calibration GPU mismatch")
    predictions = {}
    for major in ("K", "MN"):
        inputs = json.loads(json.dumps(old["inputs"]))
        for scope, name in (("one_cta", "wgmma_one_cta"), ("all_gpu", "wgmma_all_gpu")):
            chosen = [r for r in records if r["scope"] == scope and r["b_major"] == major]
            rate = statistics.median(r["work_flop"] / r["elapsed"] for r in chosen)
            inputs[name].update(value=rate * (1e9 if scope == "all_gpu" else 1),
                                source=str(layout / "layout-summary.json"),
                                source_sha256=sha(layout / "layout-summary.json"),
                                condition=f"same-GPU FP16 SS n256 g2, B major {major}, batch16/wait1")
        _, cases = v01_predict.predict(inputs)
        predictions[major] = {c["case_id"]: c["predicted_time"] for c in cases}
    measured = {}
    for folder in (base / "samples").iterdir():
        values = []
        for f in folder.glob("formal-*/result.json"):
            r = json.loads(f.read_text())
            values.append(r["elapsed_ms"] if r["configuration"]["kind"] == "cutlass" else r["elapsed"])
        if values:
            measured[folder.name] = statistics.median(values)
    cases = []
    for scope in ("one_cta", "all_gpu"):
        first, second = f"target_tc_k19_{scope}", f"target_tc_k47_{scope}"
        cases.append(dict(case_id=f"target_tc_k31_{scope}", kind=2, scope=scope, length=31,
                          predicted_time=interpolate(predictions["K"], first, second, 12 / 28),
                          unit="clock64_cycle/CTA" if scope == "one_cta" else "globaltimer_ns/GPU",
                          alternative=dict(predicted_time=interpolate(measured, first, second, 12 / 28),
                                           model="conditional affine fit to known lengths 19/47")))
    first, second = "cutlass_2048_2048_2048", "cutlass_2048_2048_8192"
    cases.append(dict(case_id="cutlass_2048_2048_4096", kind="cutlass", scope="all_gpu",
                      m=2048, n=2048, k=4096, unit="cuda_event_ms/GPU",
                      predicted_time=interpolate(predictions["MN"], first, second, 1 / 3),
                      alternative=dict(predicted_time=interpolate(measured, first, second, 1 / 3),
                                       model="conditional affine Ktile fit, same complete kernel")))
    plan = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                status="frozen before new lengths; primary resource model and separate empirical alternative",
                gpu_uuid=env["gpu_uuid"],
                binary_hashes=json.loads((base / "build/binary_hashes.json").read_text()),
                probe=dict(sha256=sha(base / "source/probes/v01.cu")),
                generator_sha256=sha(Path(__file__)),
                predictor_sha256=sha(HERE / "v01_predict.py"),
                calibration=dict(archive=str(base), predictions_sha256=sha(base / "predictions.json"),
                                 layout_summary_sha256=sha(layout / "layout-summary.json"),
                                 known_medians=measured),
                limits=["Other resource parameters still transfer from older experiments.",
                        "Throughput ratio is a window estimate, not a frequency telemetry measurement.",
                        "Affine alternative is conditional on fixed kernel/GPU/cache/grid/M/N.",
                        "An affine fit does not identify layout, cluster, or scheduling causes."],
                cases=cases)
    with path.open("x") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")
    print(sha(path))
    for c in cases:
        print(c["case_id"], c["predicted_time"], c["alternative"]["predicted_time"])


def run(base, path, expected_sha, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("single-GPU Slurm allocation required")
    prediction_bytes = path.read_bytes()
    prediction_sha = hashlib.sha256(prediction_bytes).hexdigest()
    if prediction_sha != expected_sha:
        raise ValueError("prediction does not match the SHA published by --prepare")
    plan = json.loads(prediction_bytes)
    uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if uuid != plan["gpu_uuid"]:
        raise ValueError("prediction GPU mismatch")
    if sha(base / "source/probes/v01.cu") != plan["probe"]["sha256"]:
        raise ValueError("probe changed after prediction")
    for name, digest in plan["binary_hashes"].items():
        if sha(base / "build" / name) != digest:
            raise ValueError("binary changed after prediction: " + name)
    with open("/tmp/gh200-access-" + uuid + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        (output / "source/probes").mkdir(parents=True)
        (output / "build").mkdir()
        for name in ("v01_followup.py", "run_v01.py", "analyze_v01.py", "v01_predict.py"):
            shutil.copy2(HERE / name, output / "source" / name)
        for source in (base / "source/probes").iterdir():
            if source.is_file():
                shutil.copy2(source, output / "source/probes" / source.name)
        for name in ("v01", "r00_cutlass", "v01.sass", "r00_cutlass.sass", "compile.log", "binary_hashes.json", "r00_identity.json"):
            shutil.copy2(base / "build" / name, output / "build" / name)
        write(output / "source_hashes.json", {str(f.relative_to(output / "source")): sha(f)
                                             for f in sorted((output / "source").rglob("*")) if f.is_file()})
        env = json.loads((base / "environment.json").read_text())
        env.update(hostname=os.uname().nodename, slurm_job_id=os.environ["SLURM_JOB_ID"], mode="formal")
        write(output / "environment.json", env)
        (output / "predictions.json").write_bytes(prediction_bytes)
        write(output / "protocol.json", dict(role="new_length_prediction_check", prediction_sha256=prediction_sha,
                                             prediction_created_utc=plan["created_utc"],
                                             probe_changed_after_prediction=False,
                                             binary_reused_from=str(base),
                                             cache="repeat, same source for all controlled CTAs",
                                             end_event="controlled: final bulk wait0; CUTLASS: CUDA event"))
        cases = [dict(id=c["case_id"], **{k: c[k] for k in ("kind", "scope", "length", "m", "n", "k") if k in c})
                 for c in plan["cases"]]
        write(output / "cases.json", cases)
        counts = {c["id"]: 3 if c["scope"] == "one_cta" else 10 for c in cases}
        run_v01.sample(output, cases, counts, "formal", prediction_sha, extend_one_cta=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path)
    mode.add_argument("--run", type=Path)
    p.add_argument("--layout", type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--prediction-sha256")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.prepare:
        if not args.layout:
            p.error("--prepare requires --layout")
        prepare(args.prepare.resolve(), args.layout.resolve(), args.predictions)
    else:
        if not args.output or not args.prediction_sha256:
            p.error("--run requires --output and the SHA published by --prepare")
        run(args.run.resolve(), args.predictions.resolve(), args.prediction_sha256, args.output.resolve())

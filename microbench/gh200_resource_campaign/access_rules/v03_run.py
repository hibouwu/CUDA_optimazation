#!/usr/bin/env python3
"""V03: in-call clock calibration and frozen-prediction measurement of the fixed CUTLASS kernel.

Adapted from v02_run.py. Two builds of probes/v03_probe.cu: v03_plain (original headers, the
measured value) and v03_trace (overlay copies of two CUTLASS headers with clock64/globaltimer
stamps, a per-CTA tile counter and one stamp per tile). Run inside one single-GPU Slurm allocation:
  v03_run.py build  --cutlass-root DIR --output RUN
  v03_run.py sample --output RUN --set {pilot,calib,power}
  v03_run.py sample --output RUN --set main --predictions configs/v03-predictions.json
Each process is one independent sample (warm single call + sampled FP64 check). Raw stdout is kept
gzipped per sample; a summary row without traces is appended to RUN/samples.jsonl. A failed
process is recorded (status "failed") and sampling continues.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
KERNEL_HEADER = "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"
MAINLOOP_HEADER = "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp"

# Calibration grid (label, M, N, K, swizzle, sm_count; 0 = all 132 SMs). Factors: call duration
# (~15 us - 3 ms), active SMs (sm_count 32/66/132, last-round occupancy), mainloop fraction (K
# short/long at similar duration) and off-chip traffic (swizzle 1/8, L2-resident or not).
# Shapes M = 1536*r, N = 2816 give 132*r tiles (tiles_m = 12r, tiles_n = 11).
CALIBRATION = [
    # duration ladder, full rounds, light traffic (B panels reused from L2)
    ("c_r1_k1024", 1536, 2816, 1024, 1, 0),
    ("c_r1_k8192", 1536, 2816, 8192, 1, 0),
    ("c_r1_k32768", 1536, 2816, 32768, 1, 0),
    ("c_r4_k4096", 6144, 2816, 4096, 1, 0),
    ("c_r16_k512", 24576, 2816, 512, 1, 0),
    ("c_r8_k8192", 12288, 2816, 8192, 1, 0),
    ("c_r32_k1024", 49152, 2816, 1024, 1, 0),
    ("c_r16_k8192", 24576, 2816, 8192, 1, 0),
    ("c_r96_k1024", 147456, 2816, 1024, 1, 0),
    ("c_r40_k4096", 61440, 2816, 4096, 1, 0),
    # off-chip traffic: square shapes, swizzle 1 vs 8, plus an L2-resident-input multi-round case
    ("c_sq8192_k4096_s1", 8192, 8192, 4096, 1, 0),
    ("c_sq8192_k4096_s8", 8192, 8192, 4096, 8, 0),
    ("c_sq12288_k2048_s1", 12288, 12288, 2048, 1, 0),
    ("c_sq12288_k2048_s8", 12288, 12288, 2048, 8, 0),
    ("c_sq16384_k2048_s1", 16384, 16384, 2048, 1, 0),
    ("c_sq16384_k2048_s8", 16384, 16384, 2048, 8, 0),
    ("c_sq4096_k2048_res", 4096, 4096, 2048, 1, 0),
    ("c_sq8192_k1024_res", 8192, 8192, 1024, 1, 0),
    # active SMs: persistent grid limited to 66 / 32 CTAs
    ("c_r1_k4096_g66", 1536, 2816, 4096, 1, 66),
    ("c_r8_k4096_g66", 12288, 2816, 4096, 1, 66),
    ("c_r8_k4096_g32", 12288, 2816, 4096, 1, 32),
    ("c_r8_k4096_g132", 12288, 2816, 4096, 1, 0),
    # partially filled last round (U = 132R + x)
    ("c_u140_k4096", 1792, 2560, 4096, 1, 0),
    ("c_u176_k8192", 2048, 2816, 8192, 1, 0),
    ("c_u220_k8192", 2560, 2816, 8192, 1, 0),
    ("c_u1100_k4096", 2560, 14080, 4096, 1, 0),
]
# Long calibration configs repeated with the NVML poller on (power during the call).
POWER = ["c_r16_k8192", "c_r40_k4096", "c_sq16384_k2048_s1", "c_sq16384_k2048_s8",
         "c_r8_k4096_g32", "c_r1_k32768"]
NVML_PERIOD_US = 500

TRACE_HEADER = """\
#pragma once
// V03 stamp record (kTraceWords u64 per CTA): 12 slots x (clock64, globaltimer), word 24 = smid,
// word 25 = tiles processed by the CTA (count of mma_tail completions by thread 128), words
// 32 + 2t, 33 + 2t = (clock64, globaltimer) after mma_tail of the CTA's t-th tile (t < 128).
#include <cstdint>
__constant__ uint64_t* v03_trace_ptr;
constexpr int V03_WORDS = 32 + 2 * 128;
enum V03Slot : int {
  V03_ENTRY = 0,          // thread 0, first statement of the kernel body
  V03_PRODUCER_LOAD = 1,  // elected producer lane, just before the first producer_acquire/TMA
  V03_FIRST_MMA = 2,      // +wg: thread 128/256 after the first full-barrier wait, before HGMMA
  V03_MAIN_END = 4,       // +wg: after mma_tail (final warpgroup wait0 and stage release)
  V03_EPI_START = 6,      // +wg: just before collective_epilogue.store
  V03_STORE_DONE = 8,     // thread 128 (TMA-store warp) after store_tail: wait_group.read 0
  V03_EXIT = 9,           // +role: thread 0 / 128 / 256 at the end of the kernel body
};
__device__ __forceinline__ uint64_t* v03_record() {
  unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  return v03_trace_ptr + size_t(block) * V03_WORDS;
}
__device__ __forceinline__ void v03_stamp(int slot) {
  uint64_t cycle, nanos;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycle));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanos));
  uint64_t* dst = v03_record() + 2 * slot;
  dst[0] = cycle;
  dst[1] = nanos;
}
__device__ __forceinline__ void v03_exit_stamp() {
  if (threadIdx.x % 128 == 0) {
    v03_stamp(V03_EXIT + int(threadIdx.x / 128));
  }
  if (threadIdx.x == 0) {
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    v03_record()[24] = sm;
  }
}
__device__ __forceinline__ void v03_count_tile() {  // single writer: thread 128
  uint64_t cycle, nanos;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycle));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanos));
  uint64_t* record = v03_record();
  uint64_t tile = record[25];
  if (tile < 128) {
    record[32 + 2 * tile] = cycle;
    record[33 + 2 * tile] = nanos;
  }
  record[25] = tile + 1;
}
"""

CONSUMER = "int(threadIdx.x / 128) - 1"
KERNEL_EDITS = [
    ("    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);\n",
     "    if (threadIdx.x == 0) { v03_stamp(V03_ENTRY); }\n", "before"),
    ("          // Update starting mainloop pipeline state for the next tile\n",
     f"          if (threadIdx.x % 128 == 0) {{ v03_stamp(V03_MAIN_END + {CONSUMER}); }}\n"
     "          if (threadIdx.x == 128) { v03_count_tile(); }\n",
     "before"),
    ("          // Epilogue and write to gD\n",
     f"          if (threadIdx.x % 128 == 0) {{ v03_stamp(V03_EPI_START + {CONSUMER}); }}\n",
     "before"),
    ("    } // Consumer Warp Groups End\n#endif\n",
     "      if (threadIdx.x == 128) { v03_stamp(V03_STORE_DONE); }\n"
     "    } // Consumer Warp Groups End\n    v03_exit_stamp();\n#endif\n", "replace"),
]
MAINLOOP_EDITS = [
    ("      // Mainloop\n      CUTLASS_PRAGMA_NO_UNROLL\n      for ( ; k_tile_count > 0; "
     "--k_tile_count) {\n        // LOCK smem_pipe_write for _writing_\n",
     "      v03_stamp(V03_PRODUCER_LOAD);\n", "before"),
    ("      warpgroup_arrive();\n      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;\n",
     f"      if (threadIdx.x % 128 == 0) {{ v03_stamp(V03_FIRST_MMA + {CONSUMER}); }}\n",
     "before"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def make_overlay(cutlass: Path, overlay: Path):
    """Copy two CUTLASS headers with exact, unique-site insertions; originals untouched."""
    for header, edits in ((KERNEL_HEADER, KERNEL_EDITS), (MAINLOOP_HEADER, MAINLOOP_EDITS)):
        text = (cutlass / "include" / header).read_text()
        for site, insert, how in edits:
            if text.count(site) != 1:
                raise ValueError(f"site not unique in {header}: {site[:60]!r}")
            text = text.replace(site, insert if how == "replace" else insert + site)
        text = '#include "v03_trace.hpp"\n' + text
        target = overlay / header
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (overlay / "v03_trace.hpp").write_text(TRACE_HEADER)


def build(output: Path, cutlass: Path, only=None):
    source, directory = output / "source", output / "build"
    if only:  # rebuild selected targets from the current probe into an existing run
        shutil.copy2(ROOT / "probes/v03_probe.cu", source / "probes/v03_probe.cu")
        shutil.copy2(ROOT / "v03_run.py", source / "v03_run.py")
    else:
        (source / "probes").mkdir(parents=True)
        directory.mkdir()
    for name in ("r00_common.hpp", "v03_probe.cu"):
        shutil.copy2(ROOT / "probes" / name, source / "probes" / name)
    for name in ("v03_run.py", "v03_analyze.py", "v03_predict.py"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, source / name)
    make_overlay(cutlass, source / "overlay")
    include = [f"-I{cutlass}/include", f"-I{cutlass}/tools/util/include"]
    base = ["nvcc", "-std=c++17", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
            "--ptxas-options=-v", "-DNDEBUG"]
    probe = str(source / "probes/v03_probe.cu")
    targets = {
        "v03_plain": base + include + [probe],
        "v03_trace": base + ["-DV03_TRACE", f"-I{source / 'overlay'}"] + include + [probe],
        # Same flags as v03_trace; built later from the probe with main-thread NVML polling
        # (the first poller thread recorded no rows). Kernel code is identical.
        "v03_power": base + ["-DV03_TRACE", f"-I{source / 'overlay'}"] + include + [probe],
    }
    targets = {name: targets[name] for name in only} if only else targets
    commands = {}
    for name, command in targets.items():
        command = command + ["-o", str(directory / name), "-ldl", "-lpthread"]
        commands[name] = command
        print("compile", name, flush=True)
        with (directory / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        with (directory / f"{name}.sass").open("w") as log:
            subprocess.run(["cuobjdump", "--dump-sass", str(directory / name)], stdout=log,
                           check=True, timeout=120)
    if only:
        for name in only:
            write_json(directory / f"commands-{name}.json", commands[name])
            write_json(directory / f"binary_hashes-{name}.json", {name: sha(directory / name)})
        return
    write_json(directory / "commands.json", commands)
    write_json(directory / "binary_hashes.json", {n: sha(directory / n) for n in targets})
    hashes = {str(p.relative_to(output)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    hashes["cutlass/" + KERNEL_HEADER] = sha(cutlass / "include" / KERNEL_HEADER)
    hashes["cutlass/" + MAINLOOP_HEADER] = sha(cutlass / "include" / MAINLOOP_HEADER)
    write_json(output / "source_hashes.json", hashes)
    snapshot_gpu(output, "before")
    write_json(output / "environment.json", dict(
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), hostname=os.uname().nodename,
        gpu=smi("--query-gpu=name,uuid,driver_version,clocks.max.sm,enforced.power.limit,"
                "power.limit,power.default_limit", "--format=csv,noheader").strip(),
        nvcc=subprocess.check_output(["nvcc", "--version"], text=True),
        cutlass_root=str(cutlass),
        cutlass_version_h_sha256=sha(cutlass / "include/cutlass/version.h"),
    ))


def smi(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True)


def snapshot_gpu(output: Path, tag):
    (output / f"nvidia-smi-{tag}.txt").write_text(smi("-q", "-d", "POWER,CLOCK,PERFORMANCE"))


def case(binary, m, n, k, swizzle=1, sm_count=0, nvml_us=0):
    args = ["--mode", "warm", "--m", str(m), "--n", str(n), "--k", str(k)]
    if swizzle != 1:
        args += ["--swizzle", str(swizzle)]
    if sm_count:
        args += ["--sm-count", str(sm_count)]
    if nvml_us:
        args += ["--nvml-us", str(nvml_us)]
    return dict(binary=binary, args=args)


def case_sets(predictions: Path | None = None):
    """Returns {set: (groups, rounds)}; a group is a list of cases run adjacently (shuffled)."""
    calib = {label: {f"trace_{label}": case("v03_trace", m, n, k, s, g)}
             for label, m, n, k, s, g in CALIBRATION}
    by_label = {c[0]: c for c in CALIBRATION}
    power = {label: {f"power_{label}": case("v03_trace", *by_label[label][1:], NVML_PERIOD_US)}
             for label in POWER}
    pilot = {label: {f"pilot_{label}": case("v03_trace", *by_label[label][1:], NVML_PERIOD_US)}
             for label in ("c_r1_k1024", "c_r8_k4096_g32", "c_sq16384_k2048_s8")}
    # NVML diagnostic: sustained load (2 s of back-to-back synchronized calls before the timed
    # call), polled every 5 ms; not the warm protocol.
    nvml = {}
    for label in POWER:
        spec = case("v03_power", *by_label[label][1:], 5000)
        spec["args"] += ["--sustain-ms", "2000", "--samples", "256"]
        nvml[label] = {f"nvml_{label}": spec}
    sets = dict(pilot=(pilot, 1), calib=(calib, 6), power=(power, 2), nvml=(nvml, 2))
    if predictions is not None:
        frozen = json.loads(predictions.read_text())
        main = {}
        for p in frozen["heldout"] + frozen["v02_secondary"]:
            main[p["label"]] = {f"{kind}_{p['label']}": case(f"v03_{kind}", p["m"], p["n"], p["k"],
                                                              p["swizzle"])
                                for kind in ("plain", "trace")}
        sets["main"] = (main, 10)
    return sets


def run_one(output: Path, set_name, case_id, spec, trial):
    folder = output / "samples" / case_id / f"{set_name}-{trial:02d}"
    if (folder / "result.json").exists():
        return
    if folder.exists():
        folder = folder.with_name(folder.name + f"-retry-{time.time_ns()}")
    folder.mkdir(parents=True)
    command = [str(output / "build" / spec["binary"])] + spec["args"]
    write_json(folder / "command.json", command)
    started = time.time_ns()
    with (folder / "stdout.txt").open("w") as out, (folder / "stderr.log").open("w") as err:
        process = subprocess.run(command, cwd=folder, stdout=out, stderr=err, timeout=600)
    ended = time.time_ns()
    text = (folder / "stdout.txt").read_text()
    lines = [json.loads(x) for x in text.splitlines() if x.strip()]
    record = dict(case_id=case_id, set=set_name, trial=trial, binary=spec["binary"],
                  args=spec["args"], returncode=process.returncode, started_unix_ns=started,
                  ended_unix_ns=ended, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  folder=str(folder.relative_to(output)))
    calls = [x for x in lines if x.get("event") == "call"]
    record["calls"] = [{k: v for k, v in c.items() if k != "trace"} for c in calls]
    record["check"] = next((x for x in lines if x.get("event") == "check"), None)
    record["setup"] = next((x for x in lines if x.get("event") == "setup"), None)
    ok = process.returncode == 0 and record["check"] and record["check"]["status"] == "ok"
    record["status"] = "measured" if ok else "failed"
    with (folder / "stdout.txt").open("rb") as src, gzip.open(folder / "stdout.txt.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    (folder / "stdout.txt").unlink()
    write_json(folder / "result.json", record)
    with (output / "samples.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    shown = [round(c["elapsed_us"], 2) for c in record["calls"]]
    print(set_name, case_id, trial, shown, record["status"], flush=True)


def sample(output: Path, name, predictions: Path | None):
    groups, rounds = case_sets(predictions)[name]
    write_json(output / f"protocol-{name}.json", dict(groups=groups, rounds=rounds, seed=20261013))
    rng = random.Random(20261013)
    for trial in range(rounds):
        order = list(groups)
        rng.shuffle(order)
        for group in order:
            members = list(groups[group].items())
            rng.shuffle(members)
            for case_id, spec in members:
                run_one(output, name, case_id, spec, trial)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("build", "sample"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--set", choices=("pilot", "calib", "power", "nvml", "main"))
    parser.add_argument("--only", nargs="*", help="build: rebuild only these targets")
    args = parser.parse_args()
    if args.step == "build":
        if not args.only:
            args.output.mkdir(parents=True, exist_ok=False)
        build(args.output.resolve(), args.cutlass_root.resolve(), args.only)
    else:
        predictions = args.predictions.resolve() if args.predictions else None
        if args.set == "main":
            shutil.copy2(predictions, args.output / "source" / "v03-predictions.json")
            env = json.loads((args.output / "environment.json").read_text())
            env["predictions_sha256"] = sha(predictions)
            env["predictions_copied_unix_ns"] = time.time_ns()
            write_json(args.output / "environment.json", env)
        sample(args.output.resolve(), args.set, predictions)
        snapshot_gpu(args.output.resolve(), f"after-{args.set}")


if __name__ == "__main__":
    main()

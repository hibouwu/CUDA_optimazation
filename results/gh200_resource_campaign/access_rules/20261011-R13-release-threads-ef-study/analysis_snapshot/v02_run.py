#!/usr/bin/env python3
"""V02: measure the frozen-prediction test sizes of the fixed CUTLASS kernel (NDEBUG).

Adapted from r09_run.py. Two builds of probes/v02_probe.cu: v02_plain (original headers, the
measured value) and v02_trace (overlay copies of two CUTLASS headers with clock64/globaltimer
stamps and a per-CTA tile counter). Run inside one single-GPU Slurm allocation (CUDA 12.9):
  v02_run.py build  --cutlass-root DIR --predictions configs/v02-predictions.json --output RUN
  v02_run.py sample --output RUN --set {pilot,main}
Each process is one independent sample (warm single call + sampled FP64 check). Raw stdout is
kept gzipped per sample; a summary row without traces is appended to RUN/samples.jsonl.
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
# Frozen test sizes (label, M, N, K): must equal configs/v02-predictions.json (checked in build).
SIZES = [
    ("partial_wave", 1536, 2560, 4096), ("pad_crosses_wave", 1600, 2496, 4096),
    ("just_over_wave", 2304, 2048, 3072), ("long_k", 1024, 4096, 16384),
    ("edges_odd_pad", 2600, 3496, 2000), ("multi_tail_a", 3000, 5000, 4096),
    ("multi_tail_b", 3072, 6144, 3072), ("short_k", 4096, 4096, 512),
    ("mid_k1024", 4096, 8192, 1024), ("large_cube", 6144, 6144, 6144),
    ("large_flat", 10240, 10240, 4096),
]

TRACE_HEADER = """\
#pragma once
// V02 stamp record: per CTA kTraceWords u64 = 12 slots x (clock64, globaltimer), word 24 = smid,
// word 25 = tiles processed by the CTA (consumer warpgroup 0 count of mma_tail completions).
#include <cstdint>
#ifndef V02_MASK
#define V02_MASK 0xFFF  // bit s enables slot s; ablation builds pass a subset
#endif
__constant__ uint64_t* v02_trace_ptr;
enum V02Slot : int {
  V02_ENTRY = 0,          // thread 0, first statement of the kernel body
  V02_PRODUCER_LOAD = 1,  // elected producer lane, just before the first producer_acquire/TMA
  V02_FIRST_MMA = 2,      // +wg: thread 128/256 after the first full-barrier wait, before HGMMA
  V02_MAIN_END = 4,       // +wg: after mma_tail (final warpgroup wait0 and stage release)
  V02_EPI_START = 6,      // +wg: just before collective_epilogue.store
  V02_STORE_DONE = 8,     // thread 128 (TMA-store warp) after store_tail: wait_group.read 0
  V02_EXIT = 9,           // +role: thread 0 / 128 / 256 at the end of the kernel body
};
__device__ __forceinline__ void v02_stamp(int slot) {
  if (!((V02_MASK >> slot) & 1)) {
    return;
  }
  uint64_t cycle, nanos;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycle));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanos));
  unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  uint64_t* dst = v02_trace_ptr + block * 32 + 2 * slot;
  dst[0] = cycle;
  dst[1] = nanos;
}
__device__ __forceinline__ void v02_exit_stamp() {
  if (threadIdx.x % 128 == 0) {
    v02_stamp(V02_EXIT + int(threadIdx.x / 128));
  }
  if (threadIdx.x == 0 && ((V02_MASK >> V02_EXIT) & 1)) {
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    v02_trace_ptr[block * 32 + 24] = sm;
  }
}
__device__ __forceinline__ void v02_count_tile() {
  unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  v02_trace_ptr[block * 32 + 25] += 1;  // single writer: thread 128
}
"""

CONSUMER = "int(threadIdx.x / 128) - 1"
KERNEL_EDITS = [
    ("    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);\n",
     "    if (threadIdx.x == 0) { v02_stamp(V02_ENTRY); }\n", "before"),
    ("          // Update starting mainloop pipeline state for the next tile\n",
     f"          if (threadIdx.x % 128 == 0) {{ v02_stamp(V02_MAIN_END + {CONSUMER}); }}\n"
     "          if (threadIdx.x == 128) { v02_count_tile(); }\n",
     "before"),
    ("          // Epilogue and write to gD\n",
     f"          if (threadIdx.x % 128 == 0) {{ v02_stamp(V02_EPI_START + {CONSUMER}); }}\n",
     "before"),
    ("    } // Consumer Warp Groups End\n#endif\n",
     "      if (threadIdx.x == 128) { v02_stamp(V02_STORE_DONE); }\n"
     "    } // Consumer Warp Groups End\n    v02_exit_stamp();\n#endif\n", "replace"),
]
MAINLOOP_EDITS = [
    ("      // Mainloop\n      CUTLASS_PRAGMA_NO_UNROLL\n      for ( ; k_tile_count > 0; "
     "--k_tile_count) {\n        // LOCK smem_pipe_write for _writing_\n",
     "      v02_stamp(V02_PRODUCER_LOAD);\n", "before"),
    ("      warpgroup_arrive();\n      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;\n",
     f"      if (threadIdx.x % 128 == 0) {{ v02_stamp(V02_FIRST_MMA + {CONSUMER}); }}\n",
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
        text = '#include "v02_trace.hpp"\n' + text
        target = overlay / header
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (overlay / "v02_trace.hpp").write_text(TRACE_HEADER)


def build(output: Path, cutlass: Path, predictions: Path):
    frozen = json.loads(predictions.read_text())["predictions"]
    if [(p["label"], p["m"], p["n"], p["k"]) for p in frozen] != [tuple(s) for s in SIZES]:
        raise ValueError("SIZES differ from the frozen predictions")
    source, directory = output / "source", output / "build"
    (source / "probes").mkdir(parents=True)
    directory.mkdir()
    for name in ("r00_common.hpp", "v02_probe.cu"):
        shutil.copy2(ROOT / "probes" / name, source / "probes" / name)
    for name in ("v02_run.py", "v02_analyze.py", "v02_predict.py"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, source / name)
    shutil.copy2(predictions, source / "v02-predictions.json")
    make_overlay(cutlass, source / "overlay")
    include = [f"-I{cutlass}/include", f"-I{cutlass}/tools/util/include"]
    base = ["nvcc", "-std=c++17", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
            "--ptxas-options=-v", "-DNDEBUG"]
    probe = str(source / "probes/v02_probe.cu")
    targets = {
        "v02_plain": base + include + [probe],
        "v02_trace": base + ["-DV02_TRACE", f"-I{source / 'overlay'}"] + include + [probe],
    }
    commands = {}
    for name, command in targets.items():
        command = command + ["-o", str(directory / name)]
        commands[name] = command
        print("compile", name, flush=True)
        with (directory / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        with (directory / f"{name}.sass").open("w") as log:
            subprocess.run(["cuobjdump", "--dump-sass", str(directory / name)], stdout=log,
                           check=True, timeout=120)
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
        predictions_sha256=sha(predictions),
    ))


def smi(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True)


def snapshot_gpu(output: Path, tag):
    (output / f"nvidia-smi-{tag}.txt").write_text(smi("-q", "-d", "POWER,CLOCK,PERFORMANCE"))


def case(binary, m, n, k):
    return dict(binary=binary, args=["--mode", "warm", "--m", str(m), "--n", str(n),
                                     "--k", str(k)])


def case_sets():
    """Returns {set: (groups, rounds)}; a group is a list of cases run adjacently (shuffled)."""
    main = {label: {f"{kind}_{label}": case(f"v02_{kind}", m, n, k) for kind in ("plain", "trace")}
            for label, m, n, k in SIZES}
    pilot = {label: main[label] for label in ("partial_wave", "large_flat")}
    return dict(pilot=(pilot, 1), main=(main, 10))


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
        process = subprocess.run(command, cwd=folder, stdout=out, stderr=err, timeout=300)
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
    raw = folder / "stdout.txt"
    with raw.open("rb") as src, gzip.open(folder / "stdout.txt.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    (folder / "stdout.txt").unlink()
    write_json(folder / "result.json", record)
    with (output / "samples.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    shown = [round(c["elapsed_us"], 3) for c in record["calls"]]
    print(set_name, case_id, trial, shown, record["status"], flush=True)
    if not ok:
        raise RuntimeError(f"{case_id} trial {trial} failed: {folder}")


def sample(output: Path, name):
    groups, rounds = case_sets()[name]
    write_json(output / f"protocol-{name}.json", dict(groups=groups, rounds=rounds, seed=20261012))
    rng = random.Random(20261012)
    for trial in range(rounds):
        order = list(groups)
        rng.shuffle(order)
        for group in order:
            members = list(groups[group].items())
            rng.shuffle(members)
            for case_id, spec in members:
                run_one(output, name, case_id, spec, trial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("build", "sample"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--set", choices=tuple(case_sets()))
    args = parser.parse_args()
    if args.step == "build":
        args.output.mkdir(parents=True, exist_ok=False)
        build(args.output.resolve(), args.cutlass_root.resolve(), args.predictions.resolve())
    else:
        sample(args.output.resolve(), args.set)
        snapshot_gpu(args.output.resolve(), f"after-{args.set}")


if __name__ == "__main__":
    main()

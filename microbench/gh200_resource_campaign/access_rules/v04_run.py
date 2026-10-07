#!/usr/bin/env python3
"""V04: build and measure three alternative CUTLASS sm90 configs (config-transfer test).

Adapted from v02_run.py. probes/v04_probe.cu is compiled six times: -DV04_CFG=0/1/2 (cfg_a/b/c,
see the probe header) x {plain: original headers, trace: include overlay with clock64/globaltimer
stamps and per-warpgroup tile counters}. Run inside one single-GPU Slurm allocation (CUDA 12.9):

  v04_run.py build  --cutlass-root DIR --output RUN      compile, SASS, ptxas logs
  v04_run.py setup  --output RUN                         static facts per config (no GEMM call)
  (freeze predictions with v04_predict.py --resources RUN/build/resources.json)
  v04_run.py sample --output RUN --predictions FROZEN.json --set {pilot,main}

Each process is one independent sample (warm single call + sampled FP64 check). Raw stdout is
kept gzipped per sample; a summary row without traces is appended to RUN/samples.jsonl.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CONFIGS = {"cfg_a": 0, "cfg_b": 1, "cfg_c": 2}
COOPERATIVE_HEADER = "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"
PINGPONG_HEADER = "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp"
MAINLOOP_HEADER = "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp"

# Trace record per CTA (32 x u64): slot s -> words (2s, 2s+1) = (clock64, globaltimer).
TRACE_HEADER = """\
#pragma once
// V04 stamp record, per CTA 32 u64: 13 slots x (clock64, globaltimer); word 26 = smid;
// words 27/28 = tiles finished by consumer warpgroup 0/1 (count of mma_tail completions).
// Cooperative kernels: both warpgroups share each tile (CTA tiles = word 27).
// Pingpong kernels: warpgroups alternate tiles (CTA tiles = word 27 + word 28).
#include <cstdint>
__constant__ uint64_t* v04_trace_ptr;
enum V04Slot : int {
  V04_ENTRY = 0,          // thread 0, first statement of the kernel body
  V04_PRODUCER_LOAD = 1,  // elected producer lane, just before the first producer_acquire/TMA
  V04_FIRST_MMA = 2,      // +wg: thread 128/256 after the first full-barrier wait, before HGMMA
  V04_MAIN_END = 4,       // +wg: after mma_tail (final warpgroup wait0 and stage release)
  V04_EPI_START = 6,      // +wg: just before collective_epilogue.store
  V04_STORE_DONE = 8,     // +wg: after the consumer loop (last store_tail returned)
  V04_EXIT = 10,          // +role: thread 0 / 128 / 256 at the end of the kernel body
};
__device__ __forceinline__ unsigned v04_block() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}
__device__ __forceinline__ void v04_stamp(int slot) {
  uint64_t cycle, nanos;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycle));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanos));
  uint64_t* dst = v04_trace_ptr + v04_block() * 32 + 2 * slot;
  dst[0] = cycle;
  dst[1] = nanos;
}
__device__ __forceinline__ void v04_exit_stamp() {
  if (threadIdx.x % 128 == 0 && threadIdx.x < 384) {
    v04_stamp(V04_EXIT + int(threadIdx.x / 128));
  }
  if (threadIdx.x == 0) {
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    v04_trace_ptr[v04_block() * 32 + 26] = sm;
  }
}
__device__ __forceinline__ void v04_count_tile(int consumer) {
  v04_trace_ptr[v04_block() * 32 + 27 + consumer] += 1;  // single writer per word
}
"""

CONSUMER = "int(threadIdx.x / 128) - 1"
# (line text to find, stripped; code to insert; "before" the line or "replace" it). Each line must
# occur exactly once; inserted code takes the indentation of the matched line.
KERNEL_EDITS = [
    ("SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);",
     ["if (threadIdx.x == 0) { v04_stamp(V04_ENTRY); }"], "before"),
    ("// Update starting mainloop pipeline state for the next tile",
     [f"if (threadIdx.x % 128 == 0) {{ v04_stamp(V04_MAIN_END + {CONSUMER}); }}",
      f"if (threadIdx.x % 128 == 0) {{ v04_count_tile({CONSUMER}); }}"], "before"),
    ("// Epilogue and write to gD",
     [f"if (threadIdx.x % 128 == 0) {{ v04_stamp(V04_EPI_START + {CONSUMER}); }}"], "before"),
    ("} // Consumer Warp Groups End",
     [f"  if (threadIdx.x % 128 == 0) {{ v04_stamp(V04_STORE_DONE + {CONSUMER}); }}",
      "} // Consumer Warp Groups End", "v04_exit_stamp();"], "replace"),
]
# Mainloop collective: same insertions as V02 (exact multi-line sites).
MAINLOOP_SITES = [
    ("      // Mainloop\n      CUTLASS_PRAGMA_NO_UNROLL\n      for ( ; k_tile_count > 0; "
     "--k_tile_count) {\n        // LOCK smem_pipe_write for _writing_\n",
     "      v04_stamp(V04_PRODUCER_LOAD);\n"),
    ("      warpgroup_arrive();\n      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;\n",
     f"      if (threadIdx.x % 128 == 0) {{ v04_stamp(V04_FIRST_MMA + {CONSUMER}); }}\n"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def edit_lines(text: str, edits, header: str) -> str:
    lines = text.split("\n")
    for site, insert, how in edits:
        hits = [i for i, line in enumerate(lines) if line.strip() == site]
        if len(hits) != 1:
            raise ValueError(f"site not unique in {header}: {site!r} ({len(hits)})")
        i = hits[0]
        indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        new = [indent + x for x in insert]
        lines[i:i + 1] = new if how == "replace" else new + [lines[i]]
    return "\n".join(lines)


def make_overlay(cutlass: Path, overlay: Path):
    """Copies of three CUTLASS headers with unique-site insertions; originals untouched."""
    for header in (COOPERATIVE_HEADER, PINGPONG_HEADER):
        text = edit_lines((cutlass / "include" / header).read_text(), KERNEL_EDITS, header)
        target = overlay / header
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('#include "v04_trace.hpp"\n' + text)
    text = (cutlass / "include" / MAINLOOP_HEADER).read_text()
    for site, insert in MAINLOOP_SITES:
        if text.count(site) != 1:
            raise ValueError(f"site not unique in {MAINLOOP_HEADER}: {site[:60]!r}")
        text = text.replace(site, insert + site)
    target = overlay / MAINLOOP_HEADER
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('#include "v04_trace.hpp"\n' + text)
    (overlay / "v04_trace.hpp").write_text(TRACE_HEADER)


def compile_one(name, command, directory: Path):
    with (directory / f"{name}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    with (directory / f"{name}.sass").open("w") as log:
        subprocess.run(["cuobjdump", "--dump-sass", str(directory / name)], stdout=log,
                       check=True, timeout=300)
    return name


def gemm_kernel_sass(sass_text: str):
    """Instruction lines of the CUTLASS GEMM kernel function(s) in a cuobjdump listing."""
    functions = re.split(r"\n\s*Function : ", sass_text)
    gemm = [f for f in functions if "GemmUniversal" in f.split("\n", 1)[0]]
    return gemm


def sass_facts(directory: Path, name: str):
    text = (directory / f"{name}.sass").read_text()
    log = (directory / f"{name}.log").read_text()
    kernels = gemm_kernel_sass(text)
    body = "\n".join(kernels)
    registers = re.findall(r"Function properties for (\S+)[\s\S]*?Used (\d+) registers", log)
    gemm_regs = [int(r) for f, r in registers if "GemmUniversal" in f]
    spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
    return dict(
        gemm_functions=len(kernels),
        hgmma=len(re.findall(r"\bHGMMA\.", body)),
        hgmma_shapes=sorted(set(re.findall(r"HGMMA\.(\d+x\d+x\d+)", body))),
        warpgroup_depbar=len(re.findall(r"WARPGROUP\.DEPBAR", body)),
        warpgroup_arrive=len(re.findall(r"WARPGROUP\.ARRIVE", body)),
        c7510_warnings=log.count("C7510"),
        gemm_registers=gemm_regs,
        spill_lines=[list(map(int, s)) for s in spills if any(int(x) for x in s)],
    )


def build(output: Path, cutlass: Path):
    source, directory = output / "source", output / "build"
    (source / "probes").mkdir(parents=True)
    directory.mkdir()
    for name in ("r00_common.hpp", "v04_probe.cu"):
        shutil.copy2(ROOT / "probes" / name, source / "probes" / name)
    for name in ("v04_run.py", "v04_predict.py", "v04_analyze.py"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, source / name)
    make_overlay(cutlass, source / "overlay")
    include = [f"-I{cutlass}/include", f"-I{cutlass}/tools/util/include"]
    base = ["nvcc", "-std=c++17", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
            "--ptxas-options=-v", "-DNDEBUG"]
    probe = str(source / "probes/v04_probe.cu")
    commands = {}
    for config, index in CONFIGS.items():
        commands[f"{config}_plain"] = base + [f"-DV04_CFG={index}"] + include + [probe]
        commands[f"{config}_trace"] = (base + [f"-DV04_CFG={index}", "-DV04_TRACE",
                                               f"-I{source / 'overlay'}"] + include + [probe])
    for name in commands:
        commands[name] = commands[name] + ["-o", str(directory / name)]
    write_json(directory / "commands.json", commands)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name in pool.map(lambda kv: compile_one(kv[0], kv[1], directory), commands.items()):
            print("compiled", name, flush=True)
    write_json(directory / "binary_hashes.json", {n: sha(directory / n) for n in commands})
    write_json(directory / "sass_check.json", {n: sass_facts(directory, n) for n in commands})
    hashes = {str(p.relative_to(output)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    for header in (COOPERATIVE_HEADER, PINGPONG_HEADER, MAINLOOP_HEADER):
        hashes["cutlass/" + header] = sha(cutlass / "include" / header)
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


def setup(output: Path):
    """Static facts per config (stages, SMEM, occupancy, ...); runs no GEMM."""
    facts = json.loads((output / "build/sass_check.json").read_text())
    resources = {}
    for config in CONFIGS:
        line = subprocess.check_output(
            [str(output / "build" / f"{config}_plain"), "--mode", "setup",
             "--m", "2048", "--n", "2048", "--k", "1024"], text=True).strip().splitlines()[-1]
        record = json.loads(line)
        record["sass_plain"] = facts[f"{config}_plain"]
        record["sass_trace"] = facts[f"{config}_trace"]
        resources[config] = record
        print(config, json.dumps({k: record[k] for k in ("stages", "smem", "threads",
                                                         "max_active_ctas_per_sm")}), flush=True)
    write_json(output / "build/resources.json", resources)


def smi(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True)


def snapshot_gpu(output: Path, tag):
    (output / f"nvidia-smi-{tag}.txt").write_text(smi("-q", "-d", "POWER,CLOCK,PERFORMANCE"))


def case_sets(predictions):
    """{set: (groups, rounds)}; a group = plain and trace of one (config, size), run adjacently."""
    main = {}
    for p in predictions:
        group = f"{p['config']}_{p['label']}"
        args = ["--mode", "warm", "--m", str(p["m"]), "--n", str(p["n"]), "--k", str(p["k"])]
        main[group] = {f"{kind}_{group}": dict(binary=f"{p['config']}_{kind}", args=args)
                       for kind in ("plain", "trace")}
    pilot_groups = [next(g for g in main if g.startswith(c)) for c in CONFIGS]
    return dict(pilot=({g: main[g] for g in pilot_groups}, 1), main=(main, 10))


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
    with (folder / "stdout.txt").open("rb") as src, gzip.open(folder / "stdout.txt.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    (folder / "stdout.txt").unlink()
    write_json(folder / "result.json", record)
    with (output / "samples.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    shown = [round(c["elapsed_us"], 3) for c in record["calls"]]
    print(set_name, case_id, trial, shown, record["status"], flush=True)
    if not ok:
        raise RuntimeError(f"{case_id} trial {trial} failed: {folder}")


def sample(output: Path, predictions_path: Path, name):
    if os.access(predictions_path, os.W_OK):
        raise PermissionError(f"{predictions_path} is writable: freeze (chmod 444) first")
    frozen_copy = output / "source/v04-predictions.json"
    if frozen_copy.exists():
        if sha(frozen_copy) != sha(predictions_path):
            raise ValueError("predictions differ from the copy used by earlier samples")
    else:
        shutil.copy2(predictions_path, frozen_copy)
        env = json.loads((output / "environment.json").read_text())
        env["predictions_sha256"] = sha(predictions_path)
        env["first_sample_unix_ns"] = time.time_ns()
        write_json(output / "environment.json", env)
    predictions = json.loads(predictions_path.read_text())["predictions"]
    groups, rounds = case_sets(predictions)[name]
    write_json(output / f"protocol-{name}.json", dict(groups=groups, rounds=rounds, seed=20261014))
    rng = random.Random(20261014)
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
    parser.add_argument("step", choices=("build", "setup", "sample"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--set", choices=("pilot", "main"))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.step == "build":
        output.mkdir(parents=True, exist_ok=False)
        build(output, args.cutlass_root.resolve())
    elif args.step == "setup":
        setup(output)
    else:
        sample(output, args.predictions.resolve(), args.set)
        snapshot_gpu(output, f"after-{args.set}")


if __name__ == "__main__":
    main()

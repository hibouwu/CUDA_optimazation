#!/usr/bin/env python3
"""R09: in-kernel clock and stage split of the fixed CUTLASS kernel (NDEBUG).

Run inside one single-GPU Slurm allocation (CUDA 12.9 loaded):
  r09_run.py build  --cutlass-root DIR --output RUN
  r09_run.py sample --output RUN --set {pilot,clock,stages}
Each process is one independent sample. Raw stdout (with per-CTA traces) is kept gzipped per
sample; a summary row without traces is appended to RUN/samples.jsonl.

Public cfg_a/b/c clock/input batch (15 diagnostic cases; original modes unchanged):
  r09_run.py shared-list
  r09_run.py shared-prepare --shared-run SHARED_RUN --output NEW_RUN
Add --batch wide-input for M=N=20480, K=1024, nine input cases, plain/ends only.
Add --batch clock-calibration for 18 short/middle input cases plus three cfg_b long bridges.
Then run NEW_RUN/run.sh inside the assigned GPU step, with V08_GPU and
CUDA_VISIBLE_DEVICES set to the assigned UUID. Reuses the shared source/build only.
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
STAGE_KS = (256, 512, 1024, 2048, 4096)
PLAIN_ONLY_KS = (64, 128)
CLOCK_POINTS = ((2048, 512), (2048, 2048), (2048, 8192), (8192, 8192))

TRACE_HEADER = """\
#pragma once
// R09 stamp record: per CTA kTraceWords u64 = 12 slots x (clock64, globaltimer), word 24 = smid.
#include <cstdint>
#ifndef R09_MASK
#define R09_MASK 0xFFF  // bit s enables slot s; ablation builds pass a subset
#endif
__constant__ uint64_t* r09_trace_ptr;
enum R09Slot : int {
  R09_ENTRY = 0,          // thread 0, first statement of the kernel body
  R09_PRODUCER_LOAD = 1,  // elected producer lane, just before the first producer_acquire/TMA
  R09_FIRST_MMA = 2,      // +wg: thread 128/256 after the first full-barrier wait, before HGMMA
  R09_MAIN_END = 4,       // +wg: after mma_tail (final warpgroup wait0 and stage release)
  R09_EPI_START = 6,      // +wg: just before collective_epilogue.store
  R09_STORE_DONE = 8,     // thread 128 (TMA-store warp) after store_tail: wait_group.read 0
  R09_EXIT = 9,           // +role: thread 0 / 128 / 256 at the end of the kernel body
};
__device__ __forceinline__ void r09_stamp(int slot) {
  if (!((R09_MASK >> slot) & 1)) {
    return;
  }
  uint64_t cycle, nanos;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycle));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanos));
  unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  uint64_t* dst = r09_trace_ptr + block * 32 + 2 * slot;
  dst[0] = cycle;
  dst[1] = nanos;
}
__device__ __forceinline__ void r09_exit_stamp() {
  if (threadIdx.x % 128 == 0) {
    r09_stamp(R09_EXIT + int(threadIdx.x / 128));
  }
  if (threadIdx.x == 0 && ((R09_MASK >> R09_EXIT) & 1)) {
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    r09_trace_ptr[block * 32 + 24] = sm;
  }
}
"""

CONSUMER = "int(threadIdx.x / 128) - 1"
KERNEL_EDITS = [
    ("    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);\n",
     "    if (threadIdx.x == 0) { r09_stamp(R09_ENTRY); }\n", "before"),
    ("          // Update starting mainloop pipeline state for the next tile\n",
     f"          if (threadIdx.x % 128 == 0) {{ r09_stamp(R09_MAIN_END + {CONSUMER}); }}\n",
     "before"),
    ("          // Epilogue and write to gD\n",
     f"          if (threadIdx.x % 128 == 0) {{ r09_stamp(R09_EPI_START + {CONSUMER}); }}\n",
     "before"),
    ("    } // Consumer Warp Groups End\n#endif\n",
     "      if (threadIdx.x == 128) { r09_stamp(R09_STORE_DONE); }\n"
     "    } // Consumer Warp Groups End\n    r09_exit_stamp();\n#endif\n", "replace"),
]
MAINLOOP_EDITS = [
    ("      // Mainloop\n      CUTLASS_PRAGMA_NO_UNROLL\n      for ( ; k_tile_count > 0; "
     "--k_tile_count) {\n        // LOCK smem_pipe_write for _writing_\n",
     "      r09_stamp(R09_PRODUCER_LOAD);\n", "before"),
    ("      warpgroup_arrive();\n      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;\n",
     f"      if (threadIdx.x % 128 == 0) {{ r09_stamp(R09_FIRST_MMA + {CONSUMER}); }}\n",
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
        text = '#include "r09_trace.hpp"\n' + text
        target = overlay / header
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (overlay / "r09_trace.hpp").write_text(TRACE_HEADER)


def build(output: Path, cutlass: Path):
    source, directory = output / "source", output / "build"
    (source / "probes").mkdir(parents=True)
    directory.mkdir()
    for name in ("r00_common.hpp", "r00_cutlass.cu", "r09_probe.cu"):
        shutil.copy2(ROOT / "probes" / name, source / "probes" / name)
    for name in ("r09_run.py", "r09_analyze.py"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, source / name)
    make_overlay(cutlass, source / "overlay")
    include = [f"-I{cutlass}/include", f"-I{cutlass}/tools/util/include"]
    base = ["nvcc", "-std=c++17", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
            "--ptxas-options=-v", "-DNDEBUG"]
    probe = str(source / "probes/r09_probe.cu")
    targets = {
        "r09_plain": base + include + [probe],
        "r09_trace": base + ["-DR09_TRACE", f"-I{source / 'overlay'}"] + include + [probe],
        # Ablation: entry + three exit stamps only / all stamps except entry and exits.
        "r09_trace_ends": base + ["-DR09_TRACE", "-DR09_MASK=0xE01", f"-I{source / 'overlay'}"]
        + include + [probe],
        "r09_trace_inner": base + ["-DR09_TRACE", "-DR09_MASK=0x1FE", f"-I{source / 'overlay'}"]
        + include + [probe],
        # R07 K-fit binary (same flags as R07 r00_cutlass_ndebug), for protocol cross-check.
        "r00_cutlass_ndebug": base + include + [str(source / "probes/r00_cutlass.cu")],
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
    smi = lambda *a: subprocess.check_output(["nvidia-smi", *a], text=True)
    (output / "nvidia-smi-power.txt").write_text(smi("-q", "-d", "POWER,CLOCK"))
    write_json(output / "environment.json", dict(
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), hostname=os.uname().nodename,
        gpu=smi("--query-gpu=name,uuid,driver_version,clocks.max.sm,enforced.power.limit",
                "--format=csv,noheader").strip(),
        nvcc=subprocess.check_output(["nvcc", "--version"], text=True),
        cutlass_root=str(cutlass),
        cutlass_version_h_sha256=sha(cutlass / "include/cutlass/version.h"),
    ))


def case(binary, mode, m, k, extra=()):
    return dict(binary=binary, args=["--mode", mode, "--m", str(m), "--n", str(m), "--k", str(k),
                                     *extra])


def r00_case(m, k):
    return dict(binary="r00_cutlass_ndebug",
                args=["--backend", "cutlass", "--dtype", "fp16", "--m", str(m), "--n", str(m),
                      "--k", str(k), "--cache", "repeat"])


def case_sets():
    """Returns {set: (groups, rounds)}; a group is a list of cases run adjacently (shuffled)."""
    stages = {}
    for m in (2048, 256):
        for k in STAGE_KS:
            stages[f"m{m}_k{k}"] = {f"trace_m{m}_k{k}": case("r09_trace", "warm", m, k),
                                    f"plain_m{m}_k{k}": case("r09_plain", "warm", m, k)}
        for k in PLAIN_ONLY_KS:
            stages[f"m{m}_k{k}"] = {f"plain_m{m}_k{k}": case("r09_plain", "warm", m, k)}
    for k in (512, 2048):
        stages[f"r00_m2048_k{k}"] = {f"r00_m2048_k{k}": r00_case(2048, k)}
    clock = {f"idle_m{m}_k{k}": {f"idle_m{m}_k{k}": case("r09_trace", "idle_warm", m, k)}
             for m, k in CLOCK_POINTS}
    pilot = {
        "timer": {"timer_res": dict(binary="r09_trace", args=["--mode", "timer_res"])},
        "m2048": stages["m2048_k2048"],
        "m256": stages["m256_k256"],
        "idle": {"idle_m2048_k512": case("r09_trace", "idle_warm", 2048, 512)},
    }
    ablation = {}
    for m, k in ((2048, 256), (256, 256), (2048, 2048)):
        ablation[f"m{m}_k{k}"] = {
            f"{name}_m{m}_k{k}": case(binary, "warm", m, k)
            for name, binary in (("plain", "r09_plain"), ("trace", "r09_trace"),
                                 ("ends", "r09_trace_ends"), ("inner", "r09_trace_inner"))}
    timer = {f"timer_{i}": {"timer_res": dict(binary="r09_trace", args=["--mode", "timer_res"])}
             for i in range(1)}
    return dict(pilot=(pilot, 1), clock=(clock, 10), stages=(stages, 10), timer=(timer, 3),
                ablation=(ablation, 10))


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
    if spec["binary"].startswith("r00_"):
        r = lines[0]
        record.update(elapsed_us=r["elapsed_ms"] * 1e3, status=r["status"],
                      max_storage_reference_error=r["max_storage_reference_error"],
                      warmup_converged=r["warmup_converged"])
    else:
        calls = [x for x in lines if x.get("event") == "call"]
        record["calls"] = [{k: v for k, v in c.items() if k != "trace"} for c in calls]
        record["check"] = next((x for x in lines if x.get("event") == "check"), None)
        record["setup"] = next((x for x in lines if x.get("event") == "setup"), None)
        record["timer"] = next((x for x in lines if x.get("event") == "timer_res"), None)
        record["status"] = "measured" if process.returncode == 0 else "failed"
    raw = folder / "stdout.txt"
    with raw.open("rb") as src, gzip.open(folder / "stdout.txt.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    (folder / "stdout.txt").unlink()
    write_json(folder / "result.json", record)
    with (output / "samples.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    shown = record.get("elapsed_us") or [round(c["elapsed_us"], 3) for c in record.get("calls", [])]
    print(set_name, case_id, trial, shown, record["status"], flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"{case_id} trial {trial} failed: {folder}")


def sample(output: Path, name):
    groups, rounds = case_sets()[name]
    write_json(output / f"protocol-{name}.json", dict(groups=groups, rounds=rounds, seed=20261009))
    rng = random.Random(20261009)
    for trial in range(rounds):
        order = list(groups)
        rng.shuffle(order)
        for group in order:
            members = list(groups[group].items())
            rng.shuffle(members)
            for case_id, spec in members:
                run_one(output, name, case_id, spec, trial)


def shared_cases(batch="clock-input"):
    import run_v08

    rows = []
    if batch in ("wide-input", "clock-calibration"):
        extents = (20480,) if batch == 'wide-input' else (2048, 8192)
        for cfg in run_v08.CONFIGS:
            for extent in extents:
                for mode in ("dyadic", "zero", "random"):
                    row = run_v08.row(f"{cfg}_r09_m{extent}_k1024_{mode}", cfg, "ctrl", extent, extent, 1024)
                    row.update(input_mode=mode, seed=17, sm_count=0)
                    if batch == 'clock-calibration':
                        row['purpose'] = 'calibration'
                    rows.append(row)
        if batch == 'clock-calibration':
            for mode in ('dyadic', 'zero', 'random'):
                row = run_v08.row(f'cfg_b_r09_m20480_k1024_{mode}', 'cfg_b', 'ctrl', 20480, 20480, 1024)
                row.update(input_mode=mode, seed=17, sm_count=0, purpose='bridge')
                rows.append(row)
        return rows
    for cfg in run_v08.CONFIGS:
        for k in (1024, 8192, 20480):
            modes = ("dyadic", "zero", "random") if k == 20480 else ("dyadic",)
            for mode in modes:
                row = run_v08.row(f"{cfg}_r09_k{k}_{mode}", cfg, "ctrl", 3584, 3584, k)
                row.update(input_mode=mode, seed=17, sm_count=0)
                rows.append(row)
    return rows


def prepare_shared(output: Path, shared: Path, batch="clock-input", bridge_run=None):
    """Reuse a completed public build without copying its samples or changing its kernel."""
    import v06_run as common
    import run_v08

    common.verify(shared)
    if batch == 'clock-calibration':
        if bridge_run is None:
            raise ValueError('clock-calibration requires --bridge-run')
        common.verify(bridge_run)
    binaries = json.loads((shared / "build/binary_hashes.json").read_text())
    variants = ['plain', 'ends'] if batch in ('wide-input', 'clock-calibration') else run_v08.VARIANTS
    for cfg in run_v08.CONFIGS:
        for variant in variants:
            if f"build/{cfg}_{variant}" not in binaries:
                raise ValueError(f"shared build missing {cfg}_{variant}")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(shared / "source", output / "source")
    shutil.copytree(shared / "build", output / "build")
    for name in ("r09_run.py", "r09_analyze.py", "r09_input_error.py"):
        shutil.copy2(ROOT / name, output / "source" / name)
    rows = shared_cases(batch)
    write_json(output / "cases.json", rows)
    parent = json.loads((shared / "run_config.json").read_text())
    reference_gpu = None
    if batch == 'clock-calibration':
        previous = json.loads((bridge_run / 'analysis/summary.json').read_text())
        reference_gpu = previous['environment']['gpu'].split(',')[0]
        previous_binaries = json.loads((bridge_run / 'build/binary_hashes.json').read_text())
        for variant in ('plain', 'ends'):
            key = f'build/cfg_b_{variant}'
            if binaries[key] != previous_binaries[key]:
                raise ValueError('bridge binary differs: ' + key)
        write_json(output / 'bridge_reference.json', dict(
            source_run=str(bridge_run), environment=previous['environment'],
            summary_sha256=sha(bridge_run / 'analysis/summary.json'),
            cases={r['id']: previous['cases'][r['id']] for r in rows if r.get('purpose') == 'bridge'},
            scope='Same-card, cross-job reference only; no automatic offset or refit.'))
    write_json(output / "run_config.json", dict(
        family="r09-shared-clock-input", batch=batch, variants=variants,
        reference_gpu_uuid=reference_gpu,
        cases_sha256=sha(output / "cases.json"),
        shared_run=str(shared), shared_source_commit=parent.get("source_commit"),
        shared_source_hashes_sha256=sha(shared / "source_hashes.json"),
        r09_entry_analysis_sha256={name: sha(output / "source" / name)
                                  for name in ("r09_run.py", "r09_analyze.py", "r09_input_error.py")},
    ))
    write_json(output / "build/origin.json", dict(
        copied_from=str(shared / "build"),
        binary_hashes_sha256=sha(shared / "build/binary_hashes.json"),
    ))
    write_json(output / "source_hashes.json", {
        str(p.relative_to(output)): sha(p) for p in sorted((output / "source").rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    })
    script = output / "run.sh"
    script.write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"
python3 source/run_v08.py setup --output "$PWD"
python3 source/r09_run.py shared-sample --output "$PWD"
python3 source/r09_analyze.py --shared --input "$PWD" --output "$PWD/analysis"
''')
    if batch == 'clock-calibration':
        with script.open('a') as stream:
            stream.write('python3 source/r09_run.py shared-bridge --output "$PWD"\n')
    script.chmod(0o755)
    print("prepared", len(rows), "R09 cases;", variants, "x 10 processes:", output)


def shared_process(output: Path, row, variant, trial):
    """Keep a numeric failure local to its case; other public runner errors still stop."""
    import run_v08

    try:
        record = run_v08.run_one(output, row, variant, trial)
    except ValueError:
        paths = (output / "samples" / row["id"]).glob(f"{variant}-{trial:02}*.json")
        records = [json.loads(p.read_text()) for p in paths]
        if not records:
            raise
        record = max(records, key=lambda r: r.get("attempt", 0))
        if sha(output / record["raw"]) != record["raw_sha256"]:
            raise ValueError("numeric failure raw identity changed")
        with gzip.open(output / record["raw"], "rt") as stream:
            events = {e["event"]: e for line in stream if line.strip() for e in [json.loads(line)]}
        if events.get("check", {}).get("status") != "numeric_error":
            raise
        return dict(case=row["id"], status="numeric_error", raw=record["raw"],
                    variant=variant, trial=trial, tolerance=events.get("tolerance"),
                    max_storage_reference_error=events["check"]["max_storage_reference_error"])
    return dict(case=row["id"], status="ok", raw=record["raw"], variant=variant, trial=trial)


def check_shared(output: Path):
    """Check random cases, or every wide-input case; none count as formal trial 0."""
    import run_v08
    import v06_run as common

    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Slurm allocation required")
    common.verify(output)
    config = json.loads((output / "run_config.json").read_text())
    wide = config.get('batch') in ('wide-input', 'clock-calibration')
    if sha(output / "cases.json") != config["cases_sha256"]:
        raise ValueError("matrix changed")
    env = run_v08.identity()
    if env != json.loads((output / "environment.json").read_text()):
        raise ValueError("device or allocation changed")
    if config.get('reference_gpu_uuid') and env['gpu'].split(',')[0] != config['reference_gpu_uuid']:
        raise ValueError('clock calibration requires the bridge reference GPU')
    outcomes = []
    with open('/tmp/gh200-measurement-' + env['gpu'].split(',')[0] + '.lock', 'a') as gpu_lock, \
            (output / '.run.lock').open('a') as run_lock:
        for lock in (gpu_lock, run_lock):
            run_v08.fcntl.flock(lock, run_v08.fcntl.LOCK_EX | run_v08.fcntl.LOCK_NB)
        for row in json.loads((output / "cases.json").read_text()):
            if wide or row["input_mode"] == "random":
                # Same case, separate ID: these checks never count as formal trial 0.
                check = dict(row, id="check_" + row["id"])
                result = shared_process(output, check, "plain", 0)
                result["case"] = row["id"]
                outcomes.append(result)
                print("R09 precheck", row["id"], result["status"], flush=True)
    write_json(output / ("numeric_checks.json" if wide else "random_checks.json"), outcomes)
    return outcomes


def sample_shared(output: Path):
    """Original V08 trial order and processes, excluding only cases with numeric failures."""
    import run_v08

    checks = check_shared(output)
    failed = {r["case"]: r for r in checks if r["status"] == "numeric_error"}
    rows = json.loads((output / "cases.json").read_text())
    config = json.loads((output / "run_config.json").read_text())
    # Wide cooperative CTAs exceed 64 per-tile slots: never schedule stamped here.
    selected = ['plain', 'ends'] if config.get('batch') in ('wide-input', 'clock-calibration') else run_v08.VARIANTS
    env = run_v08.identity()
    with open('/tmp/gh200-measurement-' + env['gpu'].split(',')[0] + '.lock', 'a') as gpu_lock, \
            (output / '.run.lock').open('a') as run_lock:
        for lock in (gpu_lock, run_lock):
            run_v08.fcntl.flock(lock, run_v08.fcntl.LOCK_EX | run_v08.fcntl.LOCK_NB)
        for trial in range(10):
            group = rows[:]
            random.Random(20261009 + trial).shuffle(group)
            for row in group:
                if row["id"] in failed:
                    continue
                variants = selected[:]
                random.Random(f"{trial}-{row['id']}").shuffle(variants)
                for variant in variants:
                    result = shared_process(output, row, variant, trial)
                    if result["status"] == "numeric_error":
                        failed[row["id"]] = result
                        break
            print("R09 trial", trial, "complete; numeric failures:", list(failed), flush=True)
    write_json(output / "sampling.json", dict(
        planned_cases=[r["id"] for r in rows], numeric_failed=list(failed.values()),
        sampled_cases=[r["id"] for r in rows if r["id"] not in failed],
        processes_per_sampled_case=10 * len(selected), variants=selected,
    ))
    (output / 'nvidia-smi-after-ctrl.txt').write_text(subprocess.check_output(
        ['nvidia-smi', '-q', '-i', os.environ['V08_GPU']], text=True))


def report_bridge(output: Path):
    """Report cross-job changes only; do not modify measurements or model parameters."""
    reference = json.loads((output / 'bridge_reference.json').read_text())
    current = json.loads((output / 'analysis/summary.json').read_text())
    if current['environment']['gpu'].split(',')[0] != reference['environment']['gpu'].split(',')[0]:
        raise ValueError('bridge comparison is not same GPU')
    comparisons = []
    keys = ('plain_us', 'ends_us', 'c_max_ends', 'ghz_ends', 'window_ends')
    for cid, old in reference['cases'].items():
        new = current['cases'][cid]
        if new.get('status') == 'numeric_error':
            comparisons.append(dict(case=cid, status='numeric_error'))
            continue
        comparisons.append(dict(case=cid, status='measured',
            relative_change={k: new[k] / old[k] - 1 for k in keys},
            old={k: old[k] for k in (*keys, 'plain_cv', 'ends_cv')},
            new={k: new[k] for k in (*keys, 'plain_cv', 'ends_cv')}))
    write_json(output / 'analysis/bridge_comparison.json', dict(
        reference_run=reference['source_run'], environment=current['environment'], comparisons=comparisons,
        reuse_decision='Requires explicit assessment; no automatic correction or pooling.',
        scope='Only cfg_b bridges; agreement does not independently establish cfg_a/c cross-job stability.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("build", "sample", "shared-list", "shared-prepare", "shared-check", "shared-sample", "shared-bridge"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cutlass-root", type=Path)
    parser.add_argument("--shared-run", type=Path)
    parser.add_argument("--bridge-run", type=Path)
    parser.add_argument("--batch", choices=("clock-input", "wide-input", "clock-calibration"), default="clock-input")
    parser.add_argument("--set", choices=tuple(case_sets()))
    args = parser.parse_args()
    if args.step == "shared-list":
        print(json.dumps(shared_cases(args.batch), indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.step == "shared-prepare":
        if args.shared_run is None:
            parser.error("shared-prepare requires --shared-run")
        prepare_shared(args.output.resolve(), args.shared_run.resolve(), args.batch,
                       args.bridge_run.resolve() if args.bridge_run else None)
        return
    if args.step == "shared-check":
        check_shared(args.output.resolve())
        return
    if args.step == "shared-sample":
        sample_shared(args.output.resolve())
        return
    if args.step == "shared-bridge":
        report_bridge(args.output.resolve())
        return
    if args.step == "build":
        args.output.mkdir(parents=True, exist_ok=False)
        build(args.output.resolve(), args.cutlass_root.resolve())
    else:
        sample(args.output.resolve(), args.set)


if __name__ == "__main__":
    main()

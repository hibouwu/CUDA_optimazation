#!/usr/bin/env python3
"""V06 runner: prepare (CPU) -> build -> setup -> sample calib -> [freeze] -> sample heldout.

All paths inside build commands are relative to the run directory, so a prepared run can be
copied to the node's /tmp and back without changing commands.
  prepare --output RUN --cutlass-root CUTLASS   copy sources + CUTLASS headers, make overlay
  build   --output RUN                          nvcc (6 binaries), SASS facts, binary hashes
  setup   --output RUN                          static grid/stages of every case (no GEMM run)
  sample  --output RUN --set calib|heldout|smoke --procs N [--predictions FILE]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import random
import re
import shutil
import stat
import statistics
import subprocess
import time

import v06_model as model

ROOT = Path(__file__).resolve().parent
COOP = "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"
PING = "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp"
MAINLOOP = "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp"
EXPECTED_HGMMA = {"cfg_a": 8, "cfg_b": 16, "cfg_c": 16}  # V04 SASS table


def sha(path):
    return model.sha(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def insert_once(text, site, added, after=False):
    if text.count(site) != 1:
        raise ValueError(f"overlay site must be unique ({text.count(site)}): {site[:80]!r}")
    return text.replace(site, site + added if after else added + site)


def make_overlay(cutlass: Path, overlay: Path):
    """Light stamps in the two kernel headers and the mainloop collective (first MMA).

    Same sites as run_r14.make_overlay; the epilogue headers are not modified."""
    version = (cutlass / "include/cutlass/version.h").read_text()
    for name, value in (("MAJOR", 3), ("MINOR", 9), ("PATCH", 2)):
        if f"#define CUTLASS_{name} {value}" not in version:
            raise ValueError("V06 requires CUTLASS 3.9.2")
    for header in (COOP, PING):
        text = (cutlass / "include" / header).read_text()
        text = insert_once(
            text, "    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);\n",
            "    int v06_tile_seq = -1;\n    v06_entry();\n", after=True)
        start = text.index("      // Mainloop Producer Warp")
        end = text.index("else if (producer_warp_role == ProducerWarpRole::MainloopAux)", start)
        producer = insert_once(text[start:end], "        while (work_tile_info.is_valid()) {\n",
                               "          v06_begin(work_tile_info, ++v06_tile_seq);\n", after=True)
        text = text[:start] + producer + text[end:]
        start = text.index("    else if (warp_group_role == WarpGroupRole::Consumer0")
        consumer = insert_once(text[start:], "      while (work_tile_info.is_valid()) {\n",
                               "        ++v06_tile_seq;\n", after=True)
        consumer, n = re.subn(r"(collective_mainloop\.mma\([\s\S]*?params\.mainloop)(\n\s*\);)",
                              r"\1, v06_tile_seq\2", consumer)
        if n != 1:
            raise ValueError("mainloop call site mismatch")
        lines = consumer.splitlines(True)
        hits = [i for i, line in enumerate(lines)
                if line.strip() == "// Update starting mainloop pipeline state for the next tile"]
        if len(hits) != 1:
            raise ValueError("mainloop end site mismatch")
        lines.insert(hits[0], "          v06_stamp(V06_MAIN_END, v06_tile_seq);\n")
        consumer = "".join(lines)
        consumer = insert_once(consumer, "// Epilogue and write to gD",
                               "v06_stamp(V06_EPI_PERMIT, v06_tile_seq);\n          ")
        if header == COOP:
            consumer = insert_once(
                consumer, "          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;",
                "          v06_stamp(V06_EPI_DONE, v06_tile_seq);\n")
        else:
            consumer = insert_once(
                consumer, "        // Update starting load/store pipeline states for the next tile",
                "        v06_stamp(V06_EPI_DONE, v06_tile_seq);\n")
        consumer = insert_once(consumer, "    } // Consumer Warp Groups End",
                               "      v06_final(v06_tile_seq + 1);\n")
        text = text[:start] + consumer
        target = overlay / header
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('#include "v06_trace.hpp"\n#define V06_KERNEL_OVERLAY_ACTIVE 1\n' + text)
    text = (cutlass / "include" / MAINLOOP).read_text()
    signature = "      Params const& mainloop_params) {\n"
    if text.count(signature) != 1:
        raise ValueError("mainloop signature site mismatch")
    text = text.replace(signature, "      Params const& mainloop_params, int v06_tile_seq = 0) {\n")
    text = insert_once(text,
                       "      warpgroup_arrive();\n      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;\n",
                       "      v06_stamp(V06_FIRST_MMA, v06_tile_seq);\n")
    target = overlay / MAINLOOP
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('#include "v06_trace.hpp"\n' + text)
    shutil.copy2(ROOT / "probes/v06_trace.hpp", overlay / "v06_trace.hpp")


def prepare(output: Path, cutlass: Path):
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    (source / "probes").mkdir(parents=True)
    (output / "build").mkdir()
    for name in ("v06_run.py", "v06_model.py", "v06_fit.py", "v06_analyze.py"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, source / name)
    for name in ("v06.cu", "v06_trace.hpp", "gaps_common.hpp", "r00_common.hpp"):
        shutil.copy2(ROOT / "probes" / name, source / "probes" / name)
    for part in ("include", "tools/util/include"):
        shutil.copytree(cutlass / part, source / "cutlass" / part)
    make_overlay(source / "cutlass", source / "overlay")
    rows = model.case_rows("calib") + model.case_rows("heldout")
    write_json(output / "cases.json", rows)
    n_prior, clash = model.check_novelty()  # local history only exists off-node
    write_json(output / "novelty.json", dict(prior_shapes_scanned=n_prior, heldout_clashes=clash,
                                             roots=[str(model.RESULTS), str(model.HERE / "configs")]))
    if clash:
        raise ValueError(f"held-out sizes already measured: {clash}")
    commands = {}
    for config, spec in model.CONFIGS.items():
        for variant in ("plain", "stamped"):
            cmd = ["nvcc", "-std=c++17", "-O3", "-DNDEBUG", "-gencode=arch=compute_90a,code=sm_90a",
                   "-lineinfo", "--ptxas-options=-v", f"-DV06_CFG={spec['index']}"]
            if variant == "stamped":
                cmd += ["-DV06_TRACE", "-Isource/overlay"]
            cmd += ["-Isource/cutlass/include", "-Isource/cutlass/tools/util/include",
                    "source/probes/v06.cu", "-o", f"build/{config}_{variant}"]
            commands[f"{config}_{variant}"] = cmd
    write_json(output / "build/commands.json", commands)
    write_json(output / "source_hashes.json",
               {str(p.relative_to(output)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()})


def binary_prefix(row):
    """An explicit stage count selects a separately compiled binary."""
    return row['config']+(f"_s{row['stages']}" if 'stages' in row else '')


def compile_one(output, name, cmd):
    log = output / "build" / f"{name}.log"
    with log.open("w") as stream:
        subprocess.run(cmd, cwd=output, stdout=stream, stderr=subprocess.STDOUT, check=True,
                       timeout=3600)
    with (output / "build" / f"{name}.sass").open("w") as stream:
        subprocess.run(["cuobjdump", "--dump-sass", str(output / "build" / name)], stdout=stream,
                       check=True, timeout=600)


def sass_facts(output, name):
    text = (output / "build" / f"{name}.sass").read_text()
    log = (output / "build" / f"{name}.log").read_text()
    body = "\n".join(f for f in re.split(r"\n\s*Function : ", text)
                     if "GemmUniversal" in f.split("\n", 1)[0])
    regs = re.findall(r"Function properties for (\S+)[\s\S]*?Used (\d+) registers", log)
    spills = [s for s in re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
              if any(int(x) for x in s)]
    return dict(hgmma=len(re.findall(r"\bHGMMA\.", body)),
                hgmma_shapes=sorted(set(re.findall(r"HGMMA\.(\d+x\d+x\d+)", body))),
                c7510=log.count("C7510"), spills=spills,
                gemm_registers=[int(r) for f, r in regs if "GemmUniversal" in f])


def build(output: Path):
    for path, digest in json.loads((output / "source_hashes.json").read_text()).items():
        if sha(output / path) != digest:
            raise ValueError("source changed after prepare: " + path)
    commands = json.loads((output / "build/commands.json").read_text())
    with ThreadPoolExecutor(max_workers=int(os.environ.get("V06_BUILD_JOBS", "3"))) as pool:
        for job in [pool.submit(compile_one, output, n, c) for n, c in commands.items()]:
            job.result()
    facts = {name: sass_facts(output, name) for name in commands}
    problems = []
    for name, f in facts.items():
        config = re.sub(r"_s[0-9]+$", "", name.rsplit("_", 1)[0])
        if f["hgmma"] != EXPECTED_HGMMA[config] or f["c7510"] or f["spills"]:
            problems.append(name)
    write_json(output / "build/resources.json", dict(facts=facts, problems=problems))
    write_json(output / "build/binary_hashes.json",
               {f"build/{n}": sha(output / "build" / n) for n in commands})
    print(json.dumps(facts, indent=1))
    if problems:
        raise SystemExit("SASS check failed: " + ", ".join(problems))


def verify(output):
    for manifest in ("source_hashes.json", "build/binary_hashes.json"):
        for path, digest in json.loads((output / manifest).read_text()).items():
            if sha(output / path) != digest:
                raise ValueError("frozen artifact changed: " + path)


def case_args(row):
    return [x for key in ("m", "n", "k", "lda", "ldb", "ldd") for x in (f"--{key}", str(row[key]))]


def identity():
    command = ["nvidia-smi"]
    uuid = os.environ.get("V08_GPU")
    if uuid:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
            raise ValueError("V08_GPU must match the allocated CUDA_VISIBLE_DEVICES UUID")
        command += ["-i", uuid]
    gpu = subprocess.check_output(command + ["--query-gpu=uuid,name,driver_version",
                                   "--format=csv,noheader"], text=True).strip()
    if "\n" in gpu or "GH200" not in gpu:
        raise ValueError("exactly one GH200 expected: " + gpu)
    return dict(gpu=gpu, job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename)


def setup(output):
    verify(output)
    rows = json.loads((output / "cases.json").read_text())
    out = []
    for row in rows:
        raw = subprocess.check_output([str(output / "build" / f"{row['config']}_plain"),
                                       "--mode", "setup", *case_args(row)], text=True)
        ev = next(json.loads(l) for l in raw.splitlines() if '"setup"' in l)
        expected = model.emulate_grid(row["config"], row["m"], row["n"])
        out.append(dict(case_id=row["id"], grid=ev["grid"], stages=ev["stages"], smem=ev["smem"],
                        max_active_ctas_per_sm=ev["max_active_ctas_per_sm"],
                        emulated_grid=expected, grid_matches=ev["grid"] == expected))
    write_json(output / "static_setup.json", out)
    write_json(output / "environment.json", identity())
    bad = [s["case_id"] for s in out if not s["grid_matches"]]
    print("setup cases", len(out), "grid mismatches", bad)


def run_process(output, set_name, row, variant, trial):
    folder = output / "samples" / set_name / row["id"]
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{variant}-{trial:02d}"
    cmd = [str(output / "build" / f"{row['config']}_{variant}"), *case_args(row)]
    start = time.time_ns()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    end = time.time_ns()
    with gzip.open(f"{stem}.txt.gz", "wb") as stream:
        stream.write(proc.stdout)
    rec = dict(set=set_name, case=row["id"], config=row["config"], variant=variant, trial=trial,
               returncode=proc.returncode, start_ns=start, end_ns=end,
               stderr=proc.stderr.decode(errors="replace")[-500:],
               raw=str(Path(f"{stem}.txt.gz").relative_to(output)))
    for line in proc.stdout.decode(errors="replace").splitlines():
        if '"event":"call"' in line:
            call = json.loads(line)
            warm = call["warmup_us"]
            rec.update(elapsed_us=call["elapsed_us"], warmup_calls=len(warm),
                       last5_cv=statistics.pstdev(warm[-5:]) / statistics.mean(warm[-5:]))
        elif '"event":"check"' in line:
            check = json.loads(line)
            rec.update(check=check["status"], max_err=check["max_storage_reference_error"],
                       check_samples=check["samples"], padding_errors=check["padding_errors"])
    with (output / "samples" / f"{set_name}.jsonl").open("a") as stream:
        stream.write(json.dumps(rec) + "\n")
    print(set_name, row["id"], variant, trial, rec.get("elapsed_us"), rec.get("check"),
          proc.returncode, flush=True)


def sample(output, set_name, procs, predictions=None):
    verify(output)
    env = identity()
    if env["gpu"] != json.loads((output / "environment.json").read_text())["gpu"]:
        raise ValueError("GPU changed since setup")
    kind = "heldout" if set_name == "heldout" else "calib"
    rows = [r for r in json.loads((output / "cases.json").read_text())
            if r["id"] in {x["id"] for x in model.case_rows(kind)}]
    if set_name == "smoke":
        rows = [r for r in rows if r["label"] in ("c1_r_k512", "c7_large")]
    if set_name == "heldout":
        if predictions is None:
            raise ValueError("held-out sampling requires a frozen prediction file")
        if predictions.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("prediction file must be read-only")
        frozen = json.loads(predictions.read_text())
        now = time.time_ns()
        if frozen.get("status") != "frozen" or not 0 < frozen["frozen_unix_ns"] < now:
            raise ValueError("prediction not frozen before sampling")
        if frozen["gpu"] != env["gpu"]:
            raise ValueError("prediction calibrated on another GPU")
        if {r["id"] for r in rows} != set(frozen["predictions"]):
            raise ValueError("prediction/case mismatch")
        if (output / "samples/heldout.jsonl").exists():
            raise ValueError("held-out samples already exist")
        write_json(output / "prediction_binding.json",
                   dict(sha256=sha(predictions), frozen_unix_ns=frozen["frozen_unix_ns"],
                        first_sample_unix_ns=now, gpu=env["gpu"], job=env["job"]))
    rng = random.Random(20261008 + len(set_name))
    for trial in range(procs):
        rng.shuffle(rows)
        for row in rows:
            variants = ["plain", "stamped"]
            rng.shuffle(variants)
            for variant in variants:
                run_process(output, set_name, row, variant, trial)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("prepare", "build", "setup", "sample"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cutlass-root", type=Path)
    p.add_argument("--set", choices=("smoke", "calib", "heldout"))
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--predictions", type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    if a.step == "prepare":
        prepare(out, a.cutlass_root.resolve())
    elif a.step == "build":
        build(out)
    elif a.step == "setup":
        setup(out)
    else:
        sample(out, a.set, a.procs, a.predictions.resolve() if a.predictions else None)


if __name__ == "__main__":
    main()

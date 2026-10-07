#!/usr/bin/env python3
"""R14 per-tile handoff: prepare/build without a GPU; sample only in allocated GH200."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import time
from v04_run import (sha, write_json, compile_one, sass_facts,
                     COOPERATIVE_HEADER, PINGPONG_HEADER, MAINLOOP_HEADER)

ROOT = Path(__file__).resolve().parent
CONFIGS = {"baseline": 2, "cfg_a": 0, "cfg_b": 1}
PAIRS = {"main": 0, "output": 1, "supply": 2, "critical": 3}
EPI_HEADER = "cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp"
EPI_DISPATCH_HEADER = "cutlass/epilogue/collective/collective_epilogue.hpp"
EPI_BIAS_HEADER = "cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized_bias_elementwise.hpp"
HEADERS = (COOPERATIVE_HEADER, PINGPONG_HEADER, MAINLOOP_HEADER,
           EPI_HEADER, EPI_DISPATCH_HEADER, EPI_BIAS_HEADER)


def insert_once(text, site, added, after=False):
    if text.count(site) != 1:
        raise ValueError(f"overlay site must be unique ({text.count(site)}): {site[:100]!r}")
    return text.replace(site, site + added if after else added + site)


def make_overlay(cutlass, overlay):
    version = (cutlass / "include/cutlass/version.h").read_text()
    for name, value in (("MAJOR", 3), ("MINOR", 9), ("PATCH", 2)):
        if f"#define CUTLASS_{name} {value}" not in version:
            raise ValueError("R14 requires CUTLASS 3.9.2")
    for header in HEADERS:
        original = (cutlass / "include" / header).read_text()
        text = original
        if header in (COOPERATIVE_HEADER, PINGPONG_HEADER):
            text = insert_once(text,
                "    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);\n",
                "    int r14_tile_seq = -1;\n", after=True)
            # Restrict work-loop edits to the mainloop producer and consumer regions.
            start = text.index("      // Mainloop Producer Warp")
            end = text.index("else if (producer_warp_role == ProducerWarpRole::MainloopAux)", start)
            producer = text[start:end]
            marker = "        while (work_tile_info.is_valid()) {\n"
            producer = insert_once(producer, marker,
                                   "          r14_begin(work_tile_info, ++r14_tile_seq);\n", after=True)
            producer = insert_once(producer,
                "          // Update starting pipeline state for the next tile\n",
                "          r14_stamp(R14_LOAD_RETURN, r14_tile_seq);\n")
            text = text[:start] + producer + text[end:]
            start = text.index("    else if (warp_group_role == WarpGroupRole::Consumer0")
            consumer = text[start:]
            consumer = insert_once(consumer, "      while (work_tile_info.is_valid()) {\n",
                                   "        r14_begin(work_tile_info, ++r14_tile_seq);\n", after=True)
            consumer, matches = re.subn(
                r'(collective_mainloop\.mma\([\s\S]*?params\.mainloop)(\n\s*\);)',
                r'\1, r14_tile_seq\2', consumer)
            if matches != 1: raise ValueError("mainloop trace argument site mismatch")
            marker = "// Update starting mainloop pipeline state for the next tile"
            lines = consumer.splitlines(True)
            matches = [i for i, line in enumerate(lines) if line.strip() == marker]
            if len(matches) != 1: raise ValueError("mainloop end site mismatch")
            i = matches[0]
            lines.insert(i, "          r14_stamp(R14_MAIN_END, r14_tile_seq);\n")
            consumer = ''.join(lines)
            marker = "// Epilogue and write to gD"
            consumer = insert_once(consumer, marker,
                                   "r14_stamp(R14_EPI_PERMIT, r14_tile_seq);\n          ")
            if header == COOPERATIVE_HEADER:
                consumer = insert_once(consumer,
                    "          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;",
                    "          r14_stamp(R14_STORE_RETURN, r14_tile_seq);\n")
            else:
                consumer = insert_once(consumer,
                    "        // TMA store pipeline wait is only visible",
                    "        r14_stamp(R14_STORE_RETURN, r14_tile_seq);\n")
                consumer = insert_once(consumer,
                    "        // Update starting load/store pipeline states for the next tile",
                    "        r14_stamp(R14_SOURCE_RELEASE, r14_tile_seq,\n"
                    "                  scheduler.is_last_tile(work_tile_info, 1));\n")
            text = text[:start] + consumer
        elif header == MAINLOOP_HEADER:
            signature = "      Params const& mainloop_params) {\n"
            if text.count(signature) != 1: raise ValueError("mainloop signature site mismatch")
            text = text.replace(signature,
                "      Params const& mainloop_params, int r14_tile_seq = 0) {\n")
            text = insert_once(text,
                "      warpgroup_arrive();\n      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;\n",
                "      r14_stamp(R14_FIRST_MMA, r14_tile_seq);\n")
        elif header == EPI_HEADER:
            text = insert_once(text,
                "        store_pipeline.producer_acquire(store_pipe_producer_state);\n",
                "        r14_source_acquire(store_pipe_producer_state.count(),\n"
                "            StorePipeline::UnacquiredStages,\n"
                "            get_store_pipe_increment(CtaTileMNK{}));\n", after=True)
            text = insert_once(text,
                "    store_pipeline.producer_tail(store_pipe_producer_state);\n",
                "    r14_source_tail();\n", after=True)
        else:
            # Both the dispatcher and bias wrapper include this epilogue by
            # basename. Route both through overlay to avoid bypass or duplicate
            # definitions from original and instrumented physical header files.
            text = re.sub(r'#include "([^"/]+)"',
                          r'#include "cutlass/epilogue/collective/\1"', text)
        target = overlay / header
        target.parent.mkdir(parents=True, exist_ok=True)
        marker = '#define R14_EPILOGUE_OVERLAY_ACTIVE 1\n' if header == EPI_HEADER else ''
        target.write_text('#include "r14_trace.hpp"\n' + marker + text)
    shutil.copy2(ROOT / "probes/r14_trace.hpp", overlay / "r14_trace.hpp")


def cases():
    rows = []
    for config in CONFIGS:
        tn = 256 if config == "baseline" else 128
        for k in (128, 512, 4096):
            for tiles in (1, 8):
                rows.append(dict(case_id=f"{config}_k{k}_t{tiles}", config=config,
                                 m=256, n=tn * 2 * tiles, k=k, tiles_per_cta=tiles,
                                 grid_ctas=4))
    return rows


def prepare(output, cutlass, nvcc, ccbin, unsupported=False, pair='main',
            configs=('cfg_a', 'cfg_b'), reuse=None):
    output.mkdir(parents=True, exist_ok=False)
    source, build = output / "source", output / "build"
    (source / "probes").mkdir(parents=True)
    build.mkdir()
    for name in ("r14.cu", "r14_trace.hpp", "r00_common.hpp", "gaps_common.hpp"):
        shutil.copy2(ROOT / "probes" / name, source / "probes" / name)
    for name in ("run_r14.py", "analyze_r14.py", "v04_run.py"):
        shutil.copy2(ROOT / name, source / name)
    frozen_cutlass = source / "cutlass"
    for part in ("include", "tools/util/include"):
        shutil.copytree(cutlass / part, frozen_cutlass / part)
    make_overlay(frozen_cutlass, source / "overlay")
    base = [nvcc, "-std=c++17", "-O3", "-gencode=arch=compute_90a,code=sm_90a",
            "-lineinfo", "--ptxas-options=-v", "-DNDEBUG"]
    if ccbin: base += ["-ccbin", ccbin]
    if unsupported: base += ["-allow-unsupported-compiler"]
    commands = {}
    for config, index in CONFIGS.items():
        if config not in configs: continue
        for kind in ("plain", "trace"):
            name = f"{config}_{kind}"
            command = base + [f"-DR14_CFG={index}", f"-DR14_PAIR={PAIRS[pair]}"]
            if kind == "trace": command += ["-DR14_TRACE", f"-I{source / 'overlay'}"]
            commands[name] = command + [f"-I{frozen_cutlass}/include",
                f"-I{frozen_cutlass}/tools/util/include", str(source / "probes/r14.cu"),
                "-o", str(build / name)]
    write_json(build / "commands.json", commands)
    write_json(output / "source_hashes.json", {
        str(p.relative_to(output)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()})
    with (output / "cases.csv").open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(cases()[0]))
        writer.writeheader(); writer.writerows(cases())
    write_json(output / "summary.json", dict(status="prepared", cases=18,
        gpu_measurements=0, note="analysis results are written to new analysis subdirectories"))
    write_json(output / "environment.json", dict(hostname=os.uname().nodename,
        cutlass_root=str(cutlass), cutlass_snapshot=str(frozen_cutlass),
        trace_pair=pair, compiled_configs=list(configs),
        nvcc=(json.loads((reuse/'environment.json').read_text())['nvcc'] if reuse else
              subprocess.check_output([nvcc, "--version"], text=True)),
        gpu_used=False, prepared_unix_ns=time.time_ns()))
    if reuse: reuse_build(output, reuse)


def reuse_build(output, donor):
    """Copy verified native binaries only when every compiled input is byte-identical."""
    current = json.loads((output/'source_hashes.json').read_text())
    previous = json.loads((donor/'source_hashes.json').read_text())
    prefixes = ('source/probes/', 'source/overlay/', 'source/cutlass/')
    compiled = {p: h for p, h in current.items() if p.startswith(prefixes)}
    old_compiled = {p: h for p, h in previous.items() if p.startswith(prefixes)}
    if compiled != old_compiled:
        raise ValueError('compiled inputs differ; binary reuse refused')
    for path, digest in compiled.items():
        if sha(donor/path) != digest or sha(output/path) != digest:
            raise ValueError(f'compiled input hash mismatch: {path}')
    env = json.loads((output/'environment.json').read_text())
    old_env = json.loads((donor/'environment.json').read_text())
    if env['trace_pair'] != old_env['trace_pair'] or 'release 12.9' not in old_env['nvcc']:
        raise ValueError('reuse requires the same observation group and native CUDA 12.9')
    names = list(json.loads((output/'build/commands.json').read_text()))
    hashes = json.loads((donor/'build/binary_hashes.json').read_text())
    commands = json.loads((donor/'build/commands.json').read_text())
    resources = json.loads((donor/'build/resources.json').read_text())
    for name in names:
        if sha(donor/'build'/name) != hashes[name]: raise ValueError('donor binary hash mismatch')
        for suffix in ('', '.log', '.sass'):
            shutil.copy2(donor/'build'/(name+suffix), output/'build'/(name+suffix))
        if sha(output/'build'/name) != hashes[name]: raise ValueError('copied binary hash mismatch')
    write_json(output/'build/binary_hashes.json', {name: hashes[name] for name in names})
    write_json(output/'build/resources.json', {name: resources[name] for name in names})
    write_json(output/'build/commands.json', {name: commands[name] for name in names})
    shutil.copy2(donor/'source_hashes.json', output/'build/donor_source_hashes.json')
    shutil.copy2(donor/'build/binary_hashes.json', output/'build/donor_binary_hashes.json')
    write_json(output/'build/reuse.json', dict(donor_run=str(donor),
        donor_source_manifest_sha256=sha(donor/'source_hashes.json'),
        compiled_inputs_checked=len(compiled), binaries=names,
        note='commands.json records original donor compilation; no compilation in this run'))


def build(output):
    hashes = json.loads((output / "source_hashes.json").read_text())
    if any(sha(output / p) != value for p, value in hashes.items()):
        raise ValueError("source snapshot changed; prepare a fresh directory")
    if (output/'build/reuse.json').exists():
        hashes = json.loads((output/'build/binary_hashes.json').read_text())
        if any(sha(output/'build'/p) != digest for p, digest in hashes.items()):
            raise ValueError('reused binary changed')
        print('verified reused native binaries; no compilation', flush=True)
        return
    commands = json.loads((output / "build/commands.json").read_text())
    for name, command in commands.items():
        compile_one(name, command, output / "build")
        print("compiled", name, flush=True)
    write_json(output / "build/binary_hashes.json", {
        name: sha(output / "build" / name) for name in commands})
    facts = {name: sass_facts(output / "build", name) for name in commands}
    write_json(output / "build/resources.json", facts)
    if any(not f['hgmma'] or f['spill_lines'] for f in facts.values()):
        raise RuntimeError("missing HGMMA or register spills; inspect resources.json")


def _write_atomic(path, text):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _index_result(output, folder, record):
    """Small derived index; atomically replace it so quota errors cannot cut a line."""
    path = output / 'samples.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    entry = {key: record[key] for key in ('case_id', 'set', 'trial', 'status', 'raw')}
    if 'failure_kind' in record: entry['failure_kind'] = record['failure_kind']
    entry['result'] = str((folder / 'result.json').relative_to(output))
    previous = [row for row in rows if row.get('result') == entry['result']]
    if previous:
        if previous != [entry]: raise ValueError('index identity mismatch; recover explicitly')
        return
    rows.append(entry)
    _write_atomic(path, ''.join(json.dumps(row) + '\n' for row in rows))


def _warmup_failure_evidence(record, folder):
    """Recognize only the probe's exact, independently replayable warmup failure."""
    if (record.get('returncode') != 1 or record.get('timed_out') or
            record.get('parse_error') or record.get('check') is not None):
        return None
    message = 'last five warmups CV exceeds 2% after 30 calls'
    try:
        if (folder/'stderr.log').read_text().strip() != message: return None
        process = json.loads((folder/'process.json').read_text())
        if process['returncode'] != 1 or process.get('timed_out'): return None
        with gzip.open(folder/'stdout.txt.gz', 'rt') as stream:
            events = [json.loads(line) for line in stream if line.strip()]
        if [event.get('event') for event in events] != ['setup', 'call']: return None
        setup, call = events
        summary = {key: value for key, value in call.items() if key != 'trace'}
        if setup != record.get('setup') or [summary] != record.get('calls'): return None
        if call.get('label') != 'warm' or call.get('warmup_calls') != 30: return None
        warm = call.get('warmup_us')
        if not isinstance(warm, list) or len(warm) != 30: return None
        if not all(type(value) in (int, float) and math.isfinite(value) and value > 0
                   for value in warm): return None
        tail_cv = statistics.pstdev(warm[-5:]) / statistics.mean(warm[-5:])
        if tail_cv <= 0.02: return None
        return dict(warmup_calls=30, last_five_cv=tail_cv,
                    stderr_sha256=sha(folder/'stderr.log'))
    except (ValueError, KeyError, OSError, EOFError, TypeError):
        return None


def run_one(output: Path, set_name, case_id, spec, trial):
    """V04-compatible call interface; complete arrays exist only in stdout.txt.gz.

    Exact warmup nonconvergence is a failed terminal sample; other failures stop.
    Existing incomplete/failed samples never trigger an implicit GPU retry.
    Consumers importing this function must freeze run_r14.py and its v04_run.py dependency.
    """
    output = Path(output).resolve()
    folder = output / 'samples' / case_id / f'{set_name}-{trial:02d}'
    result_path = folder / 'result.json'
    identity = dict(case_id=case_id, set=set_name, trial=trial,
                    binary=spec['binary'], args=spec['args'])
    if folder.exists():
        if not result_path.exists():
            raise RuntimeError(f'incomplete sample; recover without rerunning: {folder}')
        record = json.loads(result_path.read_text())
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError(f'existing sample identity mismatch: {folder}')
        measured = record.get('status') == 'measured' and record.get('returncode') == 0
        warmup_failed = (record.get('status') == 'failed' and
                         record.get('failure_kind') == 'warmup_not_converged')
        if record.get('record_format') != 'r14_compact_v1' or not (measured or warmup_failed):
            raise RuntimeError(f'legacy/failed record; recover explicitly: {folder}')
        raw = record['raw']
        path = folder / 'stdout.txt.gz'
        if (raw['path'] != path.name or path.stat().st_size != raw['bytes'] or
                sha(path) != raw['sha256']):
            raise ValueError(f'raw identity mismatch: {folder}')
        with gzip.open(path, 'rb') as stream:
            while stream.read(1024 * 1024): pass  # Check the complete gzip CRC/trailer.
        if warmup_failed and _warmup_failure_evidence(record, folder) is None:
            raise RuntimeError(f'warmup failure evidence does not replay: {folder}')
        _index_result(output, folder, record)
        return record

    folder.mkdir(parents=True)
    command = [str(output / 'build' / spec['binary']), *spec['args']]
    write_json(folder / 'command.json', command)
    started = time.time_ns()
    timed_out = False
    with (folder / 'stderr.log').open('w') as error_stream:
        try:
            process = subprocess.run(command, cwd=folder, stdout=subprocess.PIPE,
                                     stderr=error_stream, timeout=300)
            stdout, returncode = process.stdout, process.returncode
        except subprocess.TimeoutExpired as error:
            stdout, returncode, timed_out = error.stdout or b'', None, True
    process_info = dict(returncode=returncode, timed_out=timed_out,
                        started_unix_ns=started, ended_unix_ns=time.time_ns(),
                        slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    _write_atomic(folder / 'process.json', json.dumps(process_info, indent=2) + '\n')

    raw_path = folder / 'stdout.txt.gz'
    raw_temporary = folder / 'stdout.txt.gz.tmp'
    with gzip.open(raw_temporary, 'wb') as stream:
        stream.write(stdout)
    os.replace(raw_temporary, raw_path)
    raw = dict(path=raw_path.name, sha256=sha(raw_path), bytes=raw_path.stat().st_size,
               uncompressed_bytes=len(stdout))
    calls, setup, check, parse_error = [], None, None, None
    try:
        # Reading back verifies gzip integrity; only the small event summaries are retained.
        with gzip.open(raw_path, 'rt') as stream:
            for line in stream:
                if not line.strip(): continue
                event = json.loads(line)
                if event.get('event') == 'call':
                    calls.append({key: value for key, value in event.items() if key != 'trace'})
                elif event.get('event') == 'setup':
                    setup = event
                elif event.get('event') == 'check':
                    check = {key: value for key, value in event.items()
                             if not isinstance(value, (list, dict))}
    except (ValueError, OSError, EOFError) as error:
        parse_error = str(error)
    ok = returncode == 0 and not parse_error and setup and calls and check and check.get('status') == 'ok'
    record = dict(**identity, **process_info, record_format='r14_compact_v1', raw=raw,
                  folder=str(folder.relative_to(output)), calls=calls, setup=setup, check=check,
                  status='measured' if ok else 'failed')
    if parse_error: record['parse_error'] = parse_error
    warmup_evidence = _warmup_failure_evidence(record, folder)
    if not ok and warmup_evidence:
        record.update(failure_kind='warmup_not_converged', failure_evidence=warmup_evidence)
    _write_atomic(result_path, json.dumps(record, indent=2) + '\n')
    _index_result(output, folder, record)
    print(set_name, case_id, trial, [round(call['elapsed_us'], 3) for call in calls],
          record['status'], flush=True)
    if not ok and not warmup_evidence:
        raise RuntimeError(f'sample failed; raw retained: {folder}')
    return record


def sample(output, name, pilot_analysis=None, pilot_k=128):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("sample requires the coordinator's single-GH200 Slurm allocation")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid",
                                   "--format=csv,noheader"], text=True).strip()
    if len(gpu.splitlines()) != 1 or "GH200" not in gpu:
        raise RuntimeError(f"requires one visible GH200: {gpu}")
    env = json.loads((output / 'environment.json').read_text())
    if 'sample_gpu' in env and env['sample_gpu'] != gpu:
        raise RuntimeError("GPU identity changed")
    env.update(sample_gpu=gpu, slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(output / 'environment.json', env)
    if 'release 12.9' not in env['nvcc']:
        raise RuntimeError("formal R14 sampling requires CUDA 12.9 build")
    for manifest in ('source_hashes.json', 'build/binary_hashes.json'):
        hashes = json.loads((output / manifest).read_text())
        for path, expected in hashes.items():
            actual = output / path if manifest == 'source_hashes.json' else output / 'build' / path
            if sha(actual) != expected: raise ValueError(f"snapshot changed: {actual}")
    if name == 'main':
        if pilot_analysis is None:
            raise ValueError("main requires --pilot-analysis from this run")
        pilot = json.loads(pilot_analysis.read_text())
        admitted_sources = {sha(output/'source_hashes.json')}
        if (output/'build/reuse.json').exists():
            admitted_sources.add(json.loads((output/'build/reuse.json').read_text())
                                 ['donor_source_manifest_sha256'])
        if (pilot.get('status') != 'ok' or pilot.get('set') != 'pilot' or
                pilot.get('trace_pair') != env['trace_pair'] or
                pilot.get('source_manifest_sha256') not in admitted_sources):
            raise ValueError("pilot has not passed with this source snapshot")
    selected = cases()
    if name == "pilot":
        selected = [r for r in selected if r['config'] in ('cfg_a', 'cfg_b')
                    and r['k'] == pilot_k and r['tiles_per_cta'] == 8]
    if any(r['config'] not in env['compiled_configs'] for r in selected):
        raise ValueError('requested matrix requires a new build with --configs baseline cfg_a cfg_b')
    rng = random.Random(20261007)
    protocol = dict(cases=selected, processes=10, seed=20261007,
                    pairing="adjacent plain/trace, randomized order")
    if name == 'pilot':
        protocol.update(pilot_k=pilot_k, reason=(
            'default short-K representative' if pilot_k == 128 else
            f'within-matrix K{pilot_k} observation-domain check after shorter-K perturbation failure; '
            'the 5% perturbation threshold is unchanged'))
    else:
        protocol.update(pilot_summary_sha256=sha(pilot_analysis),
                        pilot_source_manifest_sha256=pilot['source_manifest_sha256'])
    protocol_path = output/f'protocol-{name}.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('protocol changed; use a new run directory')
    if not protocol_path.exists(): write_json(protocol_path, protocol)
    for trial in range(10):
        rng.shuffle(selected)
        for row in selected:
            kinds = ['plain', 'trace']; rng.shuffle(kinds)
            for kind in kinds:
                spec = dict(binary=f"{row['config']}_{kind}", args=[
                    '--m', str(row['m']), '--n', str(row['n']), '--k', str(row['k'])])
                run_one(output, name, f"{kind}_{row['case_id']}", spec, trial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('prepare', 'build', 'sample', 'cpu-check'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cutlass-root', type=Path)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--ccbin')
    parser.add_argument('--allow-unsupported-compiler', action='store_true')
    parser.add_argument('--set', choices=('pilot', 'main'), default='pilot')
    parser.add_argument('--pilot-analysis', type=Path)
    parser.add_argument('--pair', choices=tuple(PAIRS), default='main')
    parser.add_argument('--configs', nargs='+', choices=tuple(CONFIGS), default=['cfg_a', 'cfg_b'])
    parser.add_argument('--pilot-k', type=int, choices=(128, 512, 4096), default=128)
    parser.add_argument('--reuse-build', type=Path)
    args = parser.parse_args()
    if args.step == 'cpu-check':
        rows = cases()
        assert len(rows) == 18 and len({r['case_id'] for r in rows}) == 18
        assert all((r['m']//128) * (r['n']//(256 if r['config']=='baseline' else 128))
                   == 4*r['tiles_per_cta'] for r in rows)
        print(json.dumps(dict(status='ok', cases=len(rows), grid_ctas=4)))
    elif args.step == 'prepare':
        reuse = args.reuse_build.resolve() if args.reuse_build else None
        if args.cutlass_root is None and reuse is None:
            parser.error('prepare requires --cutlass-root or --reuse-build')
        cutlass = args.cutlass_root.resolve() if args.cutlass_root else reuse/'source/cutlass'
        prepare(args.output.resolve(), cutlass, args.nvcc, args.ccbin,
                args.allow_unsupported_compiler, args.pair, args.configs, reuse)
    elif args.step == 'build': build(args.output.resolve())
    else: sample(args.output.resolve(), args.set, args.pilot_analysis, args.pilot_k)


if __name__ == '__main__': main()

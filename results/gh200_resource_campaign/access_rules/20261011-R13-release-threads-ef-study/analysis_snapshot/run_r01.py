#!/usr/bin/env python3
"""R01 dependency chains: build, short checks, controls, formal sampling, analysis.

Run inside a single-GPU Slurm allocation, e.g.
  python3 run_r01.py --output DIR --cutlass-root /path/to/cutlass-v3.9.2
run_r06.py reuses main() with the R06 matrix.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SEED = 20261006
# Dependent steps per chain in one timed loop iteration (R06: FFMA per chain per iteration).
UNROLL = {'r01': 64, 'r06': 32}
SCALAR_OPS = ('ffma', 'add', 'shared', 'global_small', 'global_large')


def matrix():
    cases = []
    for op in SCALAR_OPS:
        for steps in (128, 512, 2048):
            cases.append(dict(id=f'{op}_w1_s1_{steps}', op=op, warps=1, streams=1,
                              steps=steps, control=0))
    for n in (64, 128, 256):
        for steps in (128, 512, 2048):
            cases.append(dict(id=f'wgmma_n{n}_{steps}', op='wgmma', n=n, steps=steps,
                              warps=4, streams=1, control=0))
    for steps in (128, 512, 2048):
        cases.append(dict(id=f'ldmatrix_{steps}', op='ldmatrix', steps=steps, warps=1,
                          streams=1, control=0))
    for op in SCALAR_OPS:
        for streams in (1, 4):
            cases.append(dict(id=f'{op}_w4_s{streams}_2048', op=op, warps=4, streams=streams,
                              steps=2048, control=0))
    return cases


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def cutlass_identity(cutlass):
    """Check CUTLASS 3.9.2 and hash its include tree (headers are not copied)."""
    if cutlass is None or not (cutlass / 'include/cute/tensor.hpp').is_file():
        raise RuntimeError('CUTLASS v3.9.2 headers required (--cutlass-root)')
    version = (cutlass / 'include/cutlass/version.h').read_text()
    found = tuple(int(re.search(r'#define\s+CUTLASS_' + k + r'\s+(\d+)', version)[1])
                  for k in ('MAJOR', 'MINOR', 'PATCH'))
    if found != (3, 9, 2):
        raise RuntimeError(f'CUTLASS version mismatch: {found}')
    files = sorted(p for p in (cutlass / 'include').rglob('*') if p.is_file())
    manifest = '\n'.join(f'{p.relative_to(cutlass)} {sha(p)}' for p in files)
    return dict(path=str(cutlass), version='3.9.2', files=len(files),
                manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())


def build(output, cutlass, family):
    source = output / 'source'
    (source / 'probes').mkdir(parents=True)
    names = ['run_r01.py', 'analyze_r01.py']
    if family == 'r06':
        names += ['run_r06.py', 'analyze_r06.py']
    for name in names:
        shutil.copy2(ROOT / name, source / name)
    for name in (family + '.cu', 'r01_support.hpp', 'r00_common.hpp'):
        shutil.copy2(ROOT / 'probes' / name, source / 'probes' / name)
    write(output / 'source_hashes.json',
          {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()})
    include = []
    if family == 'r01':
        write(output / 'cutlass.json', cutlass_identity(cutlass))
        include = ['-I' + str(cutlass / 'include')]
    target = output / 'build'
    target.mkdir()
    binary = target / family
    cmd = ['nvcc', '-std=c++17', '-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo',
           '--ptxas-options=-v', f'-D{family.upper()}_UNROLL={UNROLL[family]}',
           str(source / 'probes' / (family + '.cu')), '-o', str(binary)] + include
    write(target / 'command.json', cmd)
    with (target / 'compile.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    with (target / 'sass.txt').open('w') as f:
        subprocess.run(['cuobjdump', '--dump-sass', str(binary)], stdout=f, check=True)
    write(target / 'binary_hash.json', sha(binary))


def expected_files(row, family):
    """Witness files and byte sizes every process must leave behind."""
    if family == 'r06':
        return {'output.bin': row['blocks'] * 128 * 8 * 4, 'extra.bin': row['blocks'] * 128 * 4}
    if row['op'] == 'wgmma':
        return {'output.bin': 128 * row['n'] // 2 * 4}
    if row['op'] == 'ldmatrix':
        return {'output.bin': 32 * 5 * 4}
    lanes = row['warps'] * 32 * row['streams'] * 4
    return {'output.bin': lanes, 'starts.u32': lanes, 'edges.u32': None}


def run_case(output, case, trial, family, environment):
    directory = output / 'samples' / case['id'] / trial
    directory.mkdir(parents=True)
    cmd = [str(output / 'build' / family)]
    for key, value in case.items():
        if key != 'id':
            cmd += ['--' + key, str(value)]
    write(directory / 'command.json', cmd)
    with (directory / 'stdout.json').open('w') as out, (directory / 'stderr.log').open('w') as err:
        process = subprocess.run(cmd, cwd=directory, stdout=out, stderr=err, timeout=300)
    if process.returncode:
        raise RuntimeError(f'exit {process.returncode}: {directory}')
    row = json.loads((directory / 'stdout.json').read_text())
    if row['status'] != 'measured' or row['unroll'] != UNROLL[family]:
        raise RuntimeError(f'numeric error or unroll mismatch: {directory}')
    row.update(case_id=case['id'], configuration=case, gpu_uuid=environment['gpu_uuid'],
               trial_role='formal' if trial.startswith('trial') else trial)
    row['files'] = {}
    for name, size in expected_files(row, family).items():
        path = directory / name
        if not path.is_file() or (size is not None and path.stat().st_size != size):
            raise RuntimeError(f'missing or truncated witness {path}')
        row['files'][name] = dict(bytes=path.stat().st_size, sha256=sha(path))
        with path.open('rb') as src, gzip.open(str(path) + '.gz', 'wb', compresslevel=1) as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
    write(directory / 'result.json', row)
    return row


def cv(values):
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def main(family='r01', cases=None):
    parser = argparse.ArgumentParser(description=__doc__ if family == 'r01' else
                                     'R06: fixed-work FFMA resource and concurrency matrix.')
    parser.add_argument('--list', action='store_true', help='print the matrix and exit')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cutlass-root', type=Path)
    parser.add_argument('--smoke', action='store_true', help='build + short checks only')
    parser.add_argument('--compile-only', action='store_true')
    args = parser.parse_args()
    cases = matrix() if cases is None else cases
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    if not args.output:
        parser.error('--output required')
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('single-GPU Slurm allocation required')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True).strip()
    if '\n' in uuid:
        parser.error('exactly one visible GPU required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    query = 'name,driver_version,memory.total,clocks.sm,clocks.max.sm,power.limit'
    environment = dict(
        gpu_uuid=uuid, job=os.environ['SLURM_JOB_ID'], hostname=os.uname().nodename,
        cuda=subprocess.check_output(['nvcc', '--version'], text=True),
        gpu=subprocess.check_output(['nvidia-smi', f'--query-gpu={query}', '--format=csv'],
                                    text=True),
        mode='smoke' if args.smoke else 'formal', family=family, unroll=UNROLL[family])
    if 'release 12.9' not in environment['cuda']:
        raise RuntimeError('CUDA 12.9 required')
    write(output / 'environment.json', environment)
    write(output / 'cases.json', cases)
    build(output, args.cutlass_root.resolve() if args.cutlass_root else None, family)
    if args.compile_only:
        return
    analyzer = [sys.executable, str(output / 'source' / f'analyze_{family}.py'), '--input',
                str(output)]

    # Short checks: one and two loop iterations for every instruction/resource coordinate.
    unroll = UNROLL[family]
    coordinates = {}
    for c in cases:
        coordinates[tuple((k, v) for k, v in c.items() if k not in ('id', 'steps'))] = c
    for c in coordinates.values():
        for steps in (unroll, 2 * unroll):
            run_case(output, dict(c, id=f"check_{c['id']}_{steps}", steps=steps), 'check',
                     family, environment)
    if args.smoke:
        subprocess.run(analyzer, check=True)
        return

    # Controls (R01): empty window (1) and consumer of preloaded values without the chain (2).
    controls = []
    if family == 'r01':
        for c in cases:
            if c['steps'] == 128 and c['op'] in ('ffma', 'ldmatrix', 'wgmma'):
                controls += [dict(c, id=f"control_{c['id']}_{k}", control=k) for k in (1, 2)]
        controls += [dict(id=f'control_ffma_w4_{k}', op='ffma', warps=4, streams=1, steps=128,
                          control=k) for k in (1, 2)]
        for c in controls:
            run_case(output, c, 'control', family, environment)

    counts = {c['id']: 10 if c.get('scope') == 'all_gpu' else 3 for c in cases}
    rows = {c['id']: [] for c in cases}
    rng = random.Random(SEED)

    def sample_round(start, end, selected):
        # Paired coordinates (1/4 streams, or R06 one_cta/all_gpu) stay adjacent; groups shuffle.
        groups = {}
        for c in selected:
            key = (c.get('op'), c.get('warps'), c.get('steps'), c.get('n')) if family == 'r01' \
                else c['registers']
            groups.setdefault(key, []).append(c)
        for trial in range(start, end):
            order = list(groups.values())
            rng.shuffle(order)
            for group in order:
                for c in group:
                    if trial < counts[c['id']]:
                        print(f"{c['id']} process {trial + 1}", flush=True)
                        rows[c['id']].append(
                            run_case(output, c, f'trial-{trial:02d}', family, environment))

    sample_round(0, 10, cases)
    # CV > 1%, or a 1/4-stream difference within max(64 cycles, 3 sigma): extend to 10.
    more = {c['id'] for c in cases if len(rows[c['id']]) == 3 and
            cv([r['elapsed'] for r in rows[c['id']]]) > 0.01}
    if family == 'r01':
        for op in SCALAR_OPS:
            a, b = f'{op}_w4_s1_2048', f'{op}_w4_s4_2048'
            x = [r['elapsed'] for r in rows[a]]
            y = [r['elapsed'] for r in rows[b]]
            noise = 3 * (statistics.stdev(x) ** 2 + statistics.stdev(y) ** 2) ** 0.5
            if abs(statistics.mean(x) - statistics.mean(y)) <= max(64, noise) or more & {a, b}:
                more.update((a, b))
    for key in more:
        counts[key] = 10
    sample_round(3, 10, [c for c in cases if c['id'] in more])
    # CV > 5%: at most two further batches of 10; every window is kept.
    for _ in range(2):
        unstable = [c for c in cases if cv([r['elapsed'] for r in rows[c['id']]]) > 0.05]
        if not unstable:
            break
        for c in unstable:
            start = counts[c['id']]
            counts[c['id']] += 10
            sample_round(start, start + 10, [c])

    write(output / 'protocol.json', dict(
        seed=SEED, unroll=unroll, process_counts=counts, controls=controls,
        lengths='R01 128/512/2048 steps per chain, R01-B 2048; R06 2048 FFMA per chain x 8 chains',
        checks=f'steps {unroll} and {2 * unroll} (one and two loop iterations) per coordinate',
        warmup='8-30 windows per process, stop when last 5 have CV <= 2%; then 1 formal window',
        endpoint='final volatile shared consumer and CTA barrier; R06 extra values consumed '
                 'after the window'))
    subprocess.run(analyzer, check=True)


if __name__ == '__main__':
    main()

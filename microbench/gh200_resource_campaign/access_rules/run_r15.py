#!/usr/bin/env python3
"""R15: build once; representative plain/trace pair before the finite 18-point matrix."""
import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODES = ('reg_smem', 'tma_only', 'full_output', 'background', 'serial', 'concurrent')


def matrix():
    return [dict(id=f'r15_{kib}k_b{buffers}_{MODES[mode]}', kib=kib, buffers=buffers,
                 mode=mode) for kib in (64, 128) for buffers in (1, 2)
            for mode in (range(6) if buffers == 1 else (2, 4, 5))]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def cpu_check():
    spec = importlib.util.spec_from_file_location('r15_analysis', ROOT / 'analyze_r15.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cpu_check()


def build(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, directory = output / 'source', output / 'build'
    (source / 'probes').mkdir(parents=True)
    directory.mkdir()
    for name in ('run_r15.py', 'analyze_r15.py'):
        shutil.copy2(ROOT / name, source / name)
    for name in ('r15.cu', 'r15_fragment_store.cuh', 'gaps_common.hpp', 'r00_common.hpp'):
        shutil.copy2(ROOT / 'probes' / name, source / 'probes' / name)
    cutlass = args.cutlass_root.resolve()
    shutil.copytree(cutlass / 'include', source / 'cutlass/include')
    write(output / 'source_hashes.json', {
        str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()})
    command = [args.nvcc, '-std=c++17', '-O3', '-DNDEBUG',
               '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '--ptxas-options=-v',
               '-I' + str(source / 'cutlass/include'), str(source / 'probes/r15.cu'),
               '-lcuda', '-o', str(directory / 'r15')]
    if args.ccbin:
        command += ['-ccbin', args.ccbin]
    if args.local_diagnostic:
        command += ['-allow-unsupported-compiler']
    version = subprocess.check_output([args.nvcc, '--version'], text=True)
    if not args.local_diagnostic and 'release 12.9' not in version:
        raise ValueError('formal build requires CUDA12.9; use --local-diagnostic for diagnostics')
    write(directory / 'command.json', command)
    write(directory / 'toolchain.json', dict(nvcc_version=version,
          diagnostic=args.local_diagnostic, nvcc_sha256=sha(shutil.which(args.nvcc))))
    write(output / 'cases.json', matrix())
    write(output / 'cpu-check.json', cpu_check())
    with (directory / 'compile.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
    with (directory / 'r15.sass').open('w') as log:
        subprocess.run(['cuobjdump', '--dump-sass', str(directory / 'r15')],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    write(directory / 'binary_hashes.json', dict(r15=sha(directory / 'r15')))


def process(output, case, ordinal, repeats, traced):
    name = f'{ordinal}-{ "trace" if traced else "plain"}'
    folder = output / 'samples' / case['id'] / name
    folder.mkdir(parents=True, exist_ok=False)
    command = [str(output / 'build/r15')]
    for key in ('kib', 'buffers', 'mode'):
        command += ['--' + key, str(case[key])]
    command += ['--repeats', str(repeats), '--trace', str(int(traced))]
    write(folder / 'command.json', command)
    with (folder / 'stdout.json').open('w') as out, (folder / 'stderr.log').open('w') as err:
        result = subprocess.run(command, cwd=folder, stdout=out, stderr=err, timeout=180)
    write(folder / 'process.json', dict(returncode=result.returncode))
    if result.returncode:
        raise RuntimeError(f'probe failed: {folder}')
    record = json.loads((folder / 'stdout.json').read_text())
    if record['status'] != 'measured' or record['local_bytes_per_thread']:
        raise ValueError('numeric/resource failure')
    record.update(configuration=case, role=ordinal.split('-')[0])
    record['raw_hashes'] = {}
    for name in ('output.f32', 'background.f32', 'smem.f32', 'trace.u64'):
        path = folder / name
        record['raw_hashes'][name] = sha(path)
        with path.open('rb') as src, gzip.open(str(path) + '.gz', 'wb', compresslevel=1) as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
    write(folder / 'result.json', record)
    return record


def sample(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('single-GPU Slurm allocation required; main dialogue schedules GPU work')
    output = args.output.resolve()
    toolchain = json.loads((output / 'build/toolchain.json').read_text())
    if toolchain['diagnostic']:
        raise ValueError('diagnostic local builds are not admitted to GPU sampling')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    if '\n' in uuid:
        raise ValueError('one physical GPU required')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version',
                                  '--format=csv,noheader'], text=True).strip()
    if 'GH200' not in gpu:
        raise ValueError('GH200 required')
    identities = json.loads((output / 'source_hashes.json').read_text())
    for name, digest in identities.items():
        if sha(output / 'source' / name) != digest:
            raise ValueError('archived source changed')
    if sha(output / 'build/r15') != json.loads((output / 'build/binary_hashes.json').read_text())['r15']:
        raise ValueError('binary changed')
    with open('/tmp/gh200-access-' + uuid + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = dict(gpu_uuid=uuid, gpu=gpu, hostname=os.uname().nodename,
                           slurm_job_id=os.environ['SLURM_JOB_ID'])
        env_file = output / 'environment.json'
        if env_file.exists():
            if json.loads(env_file.read_text()) != environment:
                raise ValueError('environment identity changed')
        else:
            write(env_file, environment)
        cases = matrix()
        if args.set == 'tma-trace-recheck':
            cases=[c for c in cases if c['mode']==1 and c['buffers']==1]
            write(output/'protocol-trace-recheck.json',dict(repeats=32,role='tracecheck',
                selector='last_repeat_last_chunk',observed_events=['read_done','full_done'],
                absent_events=['prepare','issued','released','background_done'],threshold=.05,
                pairing='adjacent plain/trace; random variant order; independent current processes',
                baseline='current-run plain only; do not merge with v4 formal statistics'))
            rng=random.Random(20261007)
            results={c['id']:{False:[],True:[]} for c in cases}
            def paired(c,trial):
                variants=[False,True];rng.shuffle(variants)
                for traced in variants:
                    row=process(output,c,f'tracecheck-{trial:02d}',32,traced)
                    results[c['id']][traced].append(row['elapsed_cycles'])
            for trial in range(3):
                order=list(cases);rng.shuffle(order)
                for c in order:paired(c,trial)
            for c in cases:
                plain=results[c['id']][False];trace=results[c['id']][True]
                noise=max(statistics.stdev(plain)/statistics.mean(plain),
                          statistics.stdev(trace)/statistics.mean(trace))
                perturbation=statistics.mean(trace)/statistics.mean(plain)-1
                if noise>.01 or abs(abs(perturbation)-.05)<=3*noise:
                    for trial in range(3,10):paired(c,trial)
            subprocess.run([sys.executable,str(output/'source/analyze_r15.py'),
                '--input',str(output),'--output',str(output/'trace-recheck-review')],check=True)
            return
        if args.set == 'representative':
            cases = [c for c in cases if c['kib'] == 64 and c['buffers'] == 1
                     and c['mode'] in (2, 5)]
            repeats, count, role = 3, 3, 'representative'
        else:
            qualification = output / 'representative-review/qualification.json'
            if not qualification.exists() or not json.loads(qualification.read_text())['admit_matrix']:
                raise ValueError('representative pair must pass CPU/SASS review before formal matrix')
            repeats, count, role = args.repeats, 3, 'formal'
        write(output / ('protocol-' + role + '.json'), dict(repeats=repeats,
              independent_processes=count, per_chunk_background_mma=16, chunk_bytes=16384,
              unit='SM clock64/CTA', end='full bulk wait0 plus both roles published',
              seed=20261007, plain_trace_pair=True))
        rng = random.Random(20261007)
        collected = {c['id']: [] for c in cases}
        for trial in range(count):
            blocks=[]
            pending=list(cases)
            while pending:
                c=pending.pop(0)
                companion=next((q for q in pending if q['kib']==c['kib']
                    and q['buffers']==c['buffers'] and {q['mode'],c['mode']}=={4,5}),None)
                block=[c]
                if companion is not None:pending.remove(companion);block.append(companion)
                rng.shuffle(block);blocks.append(block)
            rng.shuffle(blocks)
            order=[c for block in blocks for c in block]
            for c in order:
                variants = [False, True]
                rng.shuffle(variants)
                for traced in variants:
                    row = process(output, c, f'{role}-{trial:02d}', repeats, traced)
                    if not traced:
                        collected[c['id']].append(row['elapsed_cycles'])
        if role=='representative':
            a,b=[collected[c['id']] for c in cases]
            noisy=(any(statistics.stdev(v)/statistics.mean(v)>.01 for v in (a,b))
                   or abs(statistics.mean(a)-statistics.mean(b))<=3*max(
                       statistics.stdev(a),statistics.stdev(b),1))
            if noisy:
                for trial in range(3,10):
                    for c in cases:
                        variants=[False,True];rng.shuffle(variants)
                        for traced in variants:
                            process(output,c,f'{role}-{trial:02d}',repeats,traced)
        if role == 'formal':
            extra = {c['id'] for c in cases if statistics.stdev(collected[c['id']]) /
                     statistics.mean(collected[c['id']]) > .01}
            # Mixed pair close to process noise: extend BOTH serial/concurrent partners.
            for kib in (64, 128):
                for buffers in (1, 2):
                    pair = [c for c in cases if c['kib'] == kib and c['buffers'] == buffers
                            and c['mode'] in (4, 5)]
                    a, b = [collected[c['id']] for c in pair]
                    if abs(statistics.mean(a) - statistics.mean(b)) <= 3 * max(
                            statistics.stdev(a), statistics.stdev(b), 1):
                        extra.update(c['id'] for c in pair)
            for c in cases:
                if c['id'] in extra:
                    for trial in range(3, 10):
                        for traced in (False, True):
                            process(output, c, f'{role}-{trial:02d}', repeats, traced)
        subprocess.run([sys.executable, str(output / 'source/analyze_r15.py'),
                        '--input', str(output), '--output',
                        str(output / ('representative-review' if role != 'formal' else 'formal-review'))],
                       check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('list', 'cpu-check', 'build', 'sample'))
    p.add_argument('--output', type=Path)
    p.add_argument('--cutlass-root', type=Path)
    p.add_argument('--nvcc', default='nvcc')
    p.add_argument('--ccbin')
    p.add_argument('--local-diagnostic', action='store_true')
    p.add_argument('--set', choices=('representative', 'formal', 'tma-trace-recheck'), default='representative')
    p.add_argument('--repeats', type=int, default=32)
    args = p.parse_args()
    if args.action == 'list':
        print(json.dumps(matrix(), indent=2))
    elif args.action == 'cpu-check':
        print(json.dumps(cpu_check(), indent=2))
    else:
        if args.output is None:
            p.error('--output required')
        if args.action == 'build':
            if args.cutlass_root is None:
                p.error('--cutlass-root required')
            build(args)
        else:
            if not 1 <= args.repeats <= 1024:
                p.error('--repeats range1..1024')
            sample(args)


if __name__ == '__main__':
    main()

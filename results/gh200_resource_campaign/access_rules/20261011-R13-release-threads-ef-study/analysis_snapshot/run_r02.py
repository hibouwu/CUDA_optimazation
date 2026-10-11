#!/usr/bin/env python3
"""R02: one invocation prepares/builds/checks/samples/analyzes in a Slurm allocation.

Use --prepare-only locally to freeze the small package before copying to node /tmp.
Run its source/run_r02.py --resume --output RUN inside the allocation.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def matrix():
    return json.loads((ROOT / 'configs/r02.json').read_text())['cases']


def command(binary, case, steps, control=None, intermediate=1, position=0, banked=0, witness=0):
    args = dict(family='source', steps=steps, intermediate=intermediate, position=position, banked=banked, witness=witness)
    if case['subset'] == 'source':
        args.update(source=case['source'], chains=case['chains'], warps=case['warps'])
    elif case['subset'] == 'pool':
        args.update(family=case['operation'], pool=case['pool'], warps=case['warps'])
    else:
        args.update(family='mixed_return', pair=case['pair'],
                    split=int(case['organization'] == 'split_warps'))
    args['control'] = int(case.get('observation') == 'preloaded_consumer') if control is None else control
    return [str(binary)] + [str(x) for item in args.items() for x in ('--'+item[0], item[1])]


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    (source / 'probes').mkdir(parents=True)
    (source / 'configs').mkdir()
    for name in ['run_r02.py', 'analyze_r02.py']:
        shutil.copy2(ROOT / name, source / name)
    for name in ['r02.cu', 'r01_support.hpp', 'r00_common.hpp']:
        shutil.copy2(ROOT / 'probes' / name, source / 'probes' / name)
    shutil.copy2(ROOT / 'configs/r02.json', source / 'configs/r02.json')
    write(output / 'source_hashes.json', {str(p.relative_to(output)): sha(p)
          for p in sorted(source.rglob('*')) if p.is_file()})
    write(output / 'cases.json', matrix())


def build(output, extra_flags=()):
    build_dir = output / 'build'
    build_dir.mkdir(exist_ok=True)
    version = subprocess.check_output(['nvcc', '--version'], text=True)
    if 'release 12.9,' not in version:
        raise RuntimeError('formal R02 requires CUDA 12.9; use the ROMEO environment')
    cmd = ['nvcc', '-std=c++17', '-O3', '-DNDEBUG', '-gencode=arch=compute_90a,code=sm_90a',
           '-lineinfo', '--ptxas-options=-v', 'source/probes/r02.cu', '-o', 'build/r02']
    cmd[1:1] = extra_flags
    write(build_dir / 'command.json', cmd)
    (build_dir / 'nvcc-version.txt').write_text(version)
    with (build_dir / 'compile.log').open('w') as log:
        subprocess.run(cmd, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    with (build_dir / 'sass.txt').open('w') as log:
        subprocess.run(['cuobjdump', '--dump-sass', 'build/r02'], cwd=output, stdout=log,
                       check=True, timeout=120)
    write(build_dir / 'binary.json', dict(sha256=sha(build_dir / 'r02'), sass_sha256=sha(build_dir / 'sass.txt')))


def cv(values):
    return statistics.pstdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def run_one(output, case, label, steps, control=None, intermediate=1, position=0, banked=0, witness=0):
    directory = output / 'samples' / case['id'] / label
    if (directory / 'result.json').exists():
        record = json.loads((directory / 'result.json').read_text())
        # Reuse only the exact request; never let a short check satisfy a formal sample.
        if (record['steps'], record['requested_control'], record['intermediate'], record['position'], record['banked'], record['witness']) != (
                steps, control, bool(intermediate), position, bool(banked), bool(witness)):
            raise RuntimeError('resume request changed: '+str(directory))
        return record
    directory.mkdir(parents=True, exist_ok=False)
    cmd = command(output / 'build/r02', case, steps, control, intermediate, position, banked, witness)
    write(directory / 'command.json', cmd)
    with (directory / 'stdout.json').open('w') as out, (directory / 'stderr.log').open('w') as err:
        process = subprocess.run(cmd, cwd=directory, stdout=out, stderr=err, timeout=120)
    try:
        record = json.loads((directory / 'stdout.json').read_text())
    except (ValueError, FileNotFoundError):
        raise RuntimeError('probe did not produce a record: '+str(directory))
    record.update(case_id=case['id'], case=case, label=label, requested_control=control,
                  returncode=process.returncode)
    if record['check_errors'] or process.returncode not in (0, 3):
        write(directory / 'result.json', record)
        raise RuntimeError('probe correctness/execution failure: '+str(directory))
    if record['local_bytes'] != 0:
        write(directory / 'result.json', record)
        raise RuntimeError('register probe spilled: '+str(directory))
    files = {}
    for name, size in [('output.u64', record['threads']*49*8), ('input.f32', 64)]:
        path = directory / name
        if path.stat().st_size != size:
            raise RuntimeError('truncated witness: '+str(path))
        files[name] = dict(bytes=size, sha256=sha(path))
        with path.open('rb') as src, gzip.open(str(path)+'.gz', 'wb') as dest:
            shutil.copyfileobj(src, dest)
        path.unlink()
    record['files'] = files
    write(directory / 'result.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--procs', type=int, default=3)
    parser.add_argument('--cutlass-root', type=Path, help='accepted for interface compatibility; R02 needs no CUTLASS')
    args = parser.parse_args()
    cases = matrix()
    if args.plan_only:
        print(json.dumps(cases, indent=2)); return
    if not args.output:
        parser.error('--output required')
    if args.procs not in (3, 10):
        parser.error('--procs must be 3 or 10')
    output = args.output.resolve()
    if not args.resume:
        prepare(output)
    for name, expected in json.loads((output/'source_hashes.json').read_text()).items():
        if sha(output/name) != expected:
            raise RuntimeError('frozen source changed: '+name)
    if args.prepare_only:
        print(output); return
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('an active single-GPU Slurm allocation is required')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if '\n' in uuid:
        raise RuntimeError('expected one visible GPU')
    gpu_lock = open('/tmp/gh200-measurement-'+uuid+'.lock', 'a')
    output_lock = (output/'.run.lock').open('a')
    for lock in [gpu_lock, output_lock]:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(job=os.environ['SLURM_JOB_ID'], host=os.uname().nodename, gpu_uuid=uuid)
    if (output/'environment.json').exists():
        old = json.loads((output/'environment.json').read_text())
        if old != environment:
            raise RuntimeError('allocation/device changed; create a new run')
    else:
        write(output/'environment.json', environment)
    for suffix in ['before']:
        (output/('nvidia-smi-'+suffix+'.txt')).write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    if not (output/'build/binary.json').exists():
        build(output)
    binary = json.loads((output/'build/binary.json').read_text())
    if sha(output/'build/r02') != binary['sha256']:
        raise RuntimeError('binary changed since compilation')
    subprocess.run([sys.executable, str(output/'source/analyze_r02.py'), '--input', str(output),
                    '--sass-only'], check=True)
    # Numerics and actual instructions are checked before expanding formal measurements.
    for case in cases:
        run_one(output, case, 'smoke', 64, witness=1)
    subprocess.run([sys.executable, str(output/'source/analyze_r02.py'), '--input', str(output),
                    '--check-only'], check=True)
    if args.smoke_only:
        print('all R02 short checks passed'); return
    order = cases[:]
    random.Random(20261008).shuffle(order)
    # Each trial preserves adjacent production/preload/final-only controls.
    for case in order:
        done = []
        target = args.procs
        trial = 0
        while trial < target:
            steps = 4096 if case['subset']=='mixed_return' else 8192
            jobs = [('formal', None, 1)]
            if case.get('observation') != 'preloaded_consumer':
                jobs += [('preload', 1, 1), ('final', 0, 0)]
            else:
                jobs += [('final', 1, 0)]
            random.Random(20261008+cases.index(case)*100+trial).shuffle(jobs)
            companions = []
            for label, control, intermediate in jobs:
                r = run_one(output, case, f'{label}-{trial:02}', steps,
                            control=control, intermediate=intermediate)
                companions.append(r)
                if label == 'formal':done.append(r)
            trial += 1
            if trial == target:
                values = [r['consumed_cycles'] for r in done]
                progress = [r['progress_cycles'] for r in done]
                unstable = max(cv(values),cv(progress)) > (0.01 if target==3 else 0.05)
                unstable |= any(not r['warmup_converged'] for r in done+companions)
                if unstable and target < 30:
                    target = 10 if target==3 else target+10
        print(case['id'], 'processes', target, flush=True)
    # Same work at two compiled positions; pre-window preparation differences are recorded.
    for cid in ['source_reuse_pair_c8_w4','source_rotate_eight_pairs_c8_w4']:
        case = next(r for r in cases if r['id']==cid)
        for trial in range(3):
            run_one(output, case, f'position-{trial:02}', 8192, position=1)
    # Distinguish the LDS sender/address service from a generic RF-return claim.
    for organization in ['same_warp','split_warps']:
        case = next(r for r in cases if r['id']==f'mixed_fadd_lds128_{organization}_consume')
        for trial in range(3):
            run_one(output,case,f'banked-{trial:02}',4096,banked=1)
            run_one(output,case,f'banked-final-{trial:02}',4096,intermediate=0,banked=1)
    (output/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    subprocess.run([sys.executable,str(output/'source/analyze_r02.py'),'--input',str(output)],check=True)
    print('R02 archive:',output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""V05: freeze source, build, then measure only against a read-only prediction file."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import stat
import subprocess
import time
from v04_run import sha, write_json, compile_one, sass_facts
from run_r14 import make_overlay, run_one
from v05_predict import cases, prior_shapes

ROOT = Path(__file__).resolve().parent
CONFIGS = {'cfg_a': 0, 'cfg_b': 1, 'cfg_c': 2}


def prepare(output, cutlass, history, reuse=None):
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    (source / 'probes').mkdir(parents=True)
    (output / 'build').mkdir()
    for name in ('run_v05.py', 'v05_predict.py', 'analyze_v05.py', 'v05_calibrate.py', 'analyze_r14.py', 'v04_run.py', 'run_r14.py'):
        shutil.copy2(ROOT / name, source / name)
    for name in ('v05.cu', 'v05_trace.hpp', 'r14_trace.hpp',
                 'gaps_common.hpp', 'r00_common.hpp'):
        shutil.copy2(ROOT / 'probes' / name, source / 'probes' / name)
    for part in ('include', 'tools/util/include'):
        shutil.copytree(cutlass / part, source / 'cutlass' / part)
    make_overlay(source / 'cutlass', source / 'overlay')
    for path in (source / 'overlay').rglob('*'):
        if path.is_file():
            path.write_text(path.read_text().replace('R14', 'V05').replace('r14', 'v05'))
    (source / 'overlay/r14_trace.hpp').unlink()
    shutil.copy2(source / 'probes/v05_trace.hpp', source / 'overlay/v05_trace.hpp')
    used = (set(map(tuple,json.loads(history.read_text()))) if history.is_file()
            else prior_shapes(history))
    if history.is_file():
        write_json(output/'history_input.json', dict(path=str(history),sha256=sha(history)))
    write_json(output / 'cases.json', cases(used))
    write_json(output / 'prior_shapes.json', sorted(used))
    commands = {}
    for config, index in CONFIGS.items():
        for variant in ('plain', 'trace_main', 'trace_output', 'trace_supply', 'trace_critical'):
            command = ['nvcc', '-std=c++17', '-O3', '-DNDEBUG',
                       '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo',
                       '--ptxas-options=-v', f'-DV05_CFG={index}']
            pair = {'plain':0, 'trace_main':0, 'trace_output':1, 'trace_supply':2, 'trace_critical':3}[variant]
            command += [f'-DV05_PAIR={pair}']
            if variant != 'plain':
                command += ['-DV05_TRACE', '-I' + str(source / 'overlay')]
            name = config + '_' + variant
            commands[name] = command + ['-I' + str(source / 'cutlass/include'),
                '-I' + str(source / 'cutlass/tools/util/include'),
                str(source / 'probes/v05.cu'), '-o', str(output / 'build' / name)]
    write_json(output / 'build/commands.json', commands)
    write_json(output / 'source_hashes.json', {
        str(p.relative_to(output)): sha(p) for p in source.rglob('*') if p.is_file()})
    write_json(output / 'summary.json', {'status': 'prepared_not_predicted', 'cases': 18})

    if reuse:
        previous=json.loads((reuse/'source_hashes.json').read_text())
        current=json.loads((output/'source_hashes.json').read_text())
        prefixes=('source/probes/','source/overlay/','source/cutlass/')
        old={p:h for p,h in previous.items() if p.startswith(prefixes)}
        new={p:h for p,h in current.items() if p.startswith(prefixes)}
        if old!=new: raise ValueError('compiled inputs changed; cannot reuse binaries')
        for path,digest in old.items():
            if sha(reuse/path)!=digest: raise ValueError('donor source changed')
        for path,digest in json.loads((reuse/'build/binary_hashes.json').read_text()).items():
            if sha(reuse/path)!=digest: raise ValueError('donor binary changed')
        for path in (reuse/'build').iterdir():
            if path.name!='commands.json': shutil.copy2(path,output/'build'/path.name)
        shutil.copy2(reuse/'build/commands.json',output/'build/donor-commands.json')
        write_json(output/'build/reuse.json', dict(donor=str(reuse),
            donor_source_manifest_sha256=sha(reuse/'source_hashes.json'),
            donor_binary_manifest_sha256=sha(reuse/'build/binary_hashes.json'),
            compiled_inputs=len(new),reason='Python-only execution/analysis revision'))


def verify(output):
    for name in ('source_hashes.json', 'build/binary_hashes.json'):
        for path, digest in json.loads((output / name).read_text()).items():
            if sha(output / path) != digest:
                raise ValueError('frozen artifact changed: ' + path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=('prepare', 'build', 'setup', 'sample'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cutlass-root', type=Path)
    p.add_argument('--history', type=Path)
    p.add_argument('--reuse-build', type=Path)
    p.add_argument('--predictions', type=Path)
    p.add_argument('--pair', choices=('main','output','supply','critical'), default='main')
    a = p.parse_args()
    output = a.output.resolve()
    if a.step == 'prepare':
        if not a.cutlass_root or not a.history:
            p.error('prepare requires --cutlass-root and --history')
        prepare(output, a.cutlass_root.resolve(), a.history.resolve(),
                a.reuse_build.resolve() if a.reuse_build else None)
        return
    if a.step == 'build':
        if (output/'build/reuse.json').exists():
            verify(output)
            print('verified reused native binaries; no compilation')
            return
        for path, digest in json.loads((output/'source_hashes.json').read_text()).items():
            if sha(output/path) != digest: raise ValueError('source changed')
        commands = json.loads((output/'build/commands.json').read_text())
        with ThreadPoolExecutor(max_workers=int(os.environ.get('GH200_BUILD_JOBS','3'))) as pool:
            jobs = [pool.submit(compile_one,name,command,output/'build')
                    for name,command in commands.items()]
            for job in jobs: job.result()
        write_json(output/'build/binary_hashes.json', {
            'build/'+name: sha(output/'build'/name) for name in commands})
        write_json(output/'build/resources.json', {
            name: sass_facts(output/'build', name) for name in commands})
        return
    verify(output)
    if not os.environ.get('SLURM_JOB_ID'): raise ValueError('Slurm allocation required')
    identity = dict(gpu=subprocess.check_output([
        'nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],
        text=True).strip(), cuda=subprocess.check_output(['nvcc','--version'],text=True),
        job=os.environ['SLURM_JOB_ID'], hostname=os.uname().nodename)
    if ('release 12.9' not in identity['cuda'] or '\n' in identity['gpu']
            or 'GH200' not in identity['gpu']):
        raise ValueError('one GH200 GPU and CUDA 12.9 required')
    rows = json.loads((output/'cases.json').read_text())
    def args(row):
        return [item for key in ('m','n','k','lda','ldb','ldd')
                for item in ('--'+key, str(row[key]))]
    if a.step == 'setup':
        # No timed GEMM: query the compiled kernel's actual scheduler and resources.
        setups = []
        for row in rows:
            raw = subprocess.check_output([str(output/'build'/(row['config']+'_plain')),
                                           '--mode','setup'] + args(row), text=True)
            setup = next(json.loads(line) for line in raw.splitlines()
                         if json.loads(line).get('event') == 'setup')
            setups.append(dict(case_id=row['id'], **setup))
        write_json(output/'static_setup.json', setups)
        write_json(output/'environment.json', identity)
        return
    if identity != json.loads((output/'environment.json').read_text()):
        raise ValueError('device/job/toolchain changed since static setup')
    if a.predictions is None: p.error('sample requires --predictions')
    if a.predictions.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError('prediction file must be read-only before any V05 sample')
    prediction = json.loads(a.predictions.read_text())
    if prediction.get('status') != 'frozen': raise ValueError('prediction is not frozen')
    for key, path in (('source_manifest_sha256','source_hashes.json'),
                      ('binary_manifest_sha256','build/binary_hashes.json')):
        if prediction.get(key) != sha(output/path):
            raise ValueError('prediction build identity mismatch: '+path)
    if prediction['cases'] != rows: raise ValueError('prediction/case mismatch')
    if prediction.get('environment_sha256') != sha(output/'environment.json'):
        raise ValueError('prediction device identity mismatch')
    if prediction['static_setup_sha256'] != sha(output/'static_setup.json'):
        raise ValueError('scheduler setup changed after prediction')
    required = ('supply', 'mainloop', 'output', 'handoff', 'epi_permission_wait', 'critical_cta')
    predictions = prediction.get('predictions', {})
    for row in rows:
        phases = predictions.get(row['id'], {})
        for phase in required:
            value = phases.get(phase, {})
            if type(value.get('cycles')) not in (float, int) or not math.isfinite(value['cycles']) or (phase != 'handoff' and value['cycles'] < 0):
                raise ValueError('missing finite prediction: ' + row['id'] + '/' + phase)
            if not value.get('source') or not value.get('events'):
                raise ValueError('prediction needs provenance and event boundary')
    digest = sha(a.predictions)
    binding_path = output/'prediction_binding.json'
    if binding_path.exists():
        binding = json.loads(binding_path.read_text())
        if binding['sha256'] != digest:
            raise ValueError('cannot rebind old samples to another prediction')
        if sha(output/'predictions.json') != digest:
            raise ValueError('archived prediction changed')
    else:
        if any((output/'samples').rglob('result.json')):
            raise ValueError('unbound samples already exist')
        now = time.time_ns()
        if not 0 < prediction['frozen_unix_ns'] < now:
            raise ValueError('invalid prediction freeze timestamp')
        write_json(binding_path, dict(sha256=digest,
            frozen_unix_ns=prediction['frozen_unix_ns'], first_sample_unix_ns=now))
        shutil.copy2(a.predictions, output/'predictions.json')
    lock_name = '/tmp/gh200-access-' + identity['gpu'].split(',')[0] + '.lock'
    with open(lock_name, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rng = random.Random(20261007)
        for trial in range(10):
            rng.shuffle(rows)
            for row in rows:
                variants = ['plain','trace']; rng.shuffle(variants)
                for variant in variants:
                    binary = row['config'] + '_' + (variant if variant=='plain' else 'trace_'+a.pair)
                    spec = dict(binary=binary, args=args(row))
                    run_one(output,a.pair,variant+'_'+row['id'],spec,trial)



if __name__ == '__main__': main()

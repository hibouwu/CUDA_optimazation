#!/usr/bin/env python3
"""R10: nine B row strides; build locally, sample only on the scheduled GH200."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CONFIGS = {'cfg_a': 0, 'cfg_b': 1, 'cfg_c': 2}
KERNELS = ('cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp',
           'cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp')
MAIN = 'cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp'
TRACE = '''#pragma once
#include <cstdint>
__constant__ uint64_t* r10_trace_ptr;
__device__ __forceinline__ unsigned r10_block() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}
// 0/1: group0 begin/end; 2/3: group1 begin/end; 4/5 tile counts; 6 smid.
__device__ __forceinline__ void r10_stamp(int slot) {
#if R10_CFG == 1
  if (r10_block() >= 4 || slot > 1) return;
#endif
  uint64_t cycle;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycle));
  r10_trace_ptr[r10_block() * 8 + slot] = cycle;
}
__device__ __forceinline__ void r10_tail(int group) {
#if R10_CFG == 1
  if (r10_block() >= 4 || group != 0) return;
#endif
  r10_stamp(group * 2 + 1);
#if R10_CFG == 1
  r10_trace_ptr[r10_block() * 8 + 7] = 4;
#endif
  r10_trace_ptr[r10_block() * 8 + 4 + group] += 1;
  unsigned sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  if (group == 0) r10_trace_ptr[r10_block() * 8 + 6] = sm;
}
'''


def matrix():
    return [dict(id=f'{config}_ldb{ldb}', config=config, m=2600, n=3000, k=2000,
                 lda=2000, ldb=ldb, ldd=3000)
            for config in CONFIGS for ldb in (3000, 3008, 3072)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def overlay(cutlass, destination):
    for header in KERNELS:
        text = (cutlass / 'include' / header).read_text()
        site = '// Update starting mainloop pipeline state for the next tile'
        lines = text.splitlines()
        hits = [i for i, line in enumerate(lines) if line.strip() == site]
        if len(hits) != 1:
            raise ValueError('mainloop tail site not unique: ' + header)
        i = hits[0]
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        lines.insert(i, indent + 'if (threadIdx.x % 128 == 0) '
                     '{ r10_tail(int(threadIdx.x / 128) - 1); }')
        target = destination / header
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('#include "r10_trace.hpp"\n' + '\n'.join(lines) + '\n')
    text = (cutlass / 'include' / MAIN).read_text()
    site = '      warpgroup_arrive();\n      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;\n'
    if text.count(site) != 1:
        raise ValueError('first MMA site not unique')
    inserted = ('      if (threadIdx.x % 128 == 0) '
                '{ r10_stamp(2 * (int(threadIdx.x / 128) - 1)); }\n')
    target = destination / MAIN
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('#include "r10_trace.hpp"\n' + text.replace(site, inserted + site))
    (destination / 'r10_trace.hpp').write_text(TRACE)


def build(output, cutlass, only, host_compiler=None):
    output.mkdir(parents=True, exist_ok=False)
    source, directory = output / 'source', output / 'build'
    (source / 'probes').mkdir(parents=True)
    directory.mkdir()
    for name in ('run_r10.py', 'analyze_r10.py', 'probes/r10.cu', 'probes/gaps_common.hpp',
                 'probes/r00_common.hpp'):
        shutil.copy2(ROOT / name, source / name)
    # Exact dependency snapshot, including builder and scheduler headers.
    for part in ('include', 'tools/util/include'):
        shutil.copytree(cutlass / part, source / 'cutlass' / part)
    version = (source / 'cutlass/include/cutlass/version.h').read_text()
    versions = [re.search(r'#define\s+CUTLASS_' + k + r'\s+(\d+)', version)
                for k in ('MAJOR', 'MINOR', 'PATCH')]
    if tuple(int(v[1]) for v in versions) != (3, 9, 2):
        raise ValueError('requires CUTLASS3.9.2')
    overlay(source / 'cutlass', source / 'overlay')
    write(output / 'source_hashes.json', {
        str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()})
    write(output / 'build_environment.json', dict(
        hostname=os.uname().nodename,
        nvcc=subprocess.check_output(['nvcc', '--version'], text=True),
        gpu_executed=False, only_config=only, host_compiler=host_compiler))
    commands = {}
    for config, index in CONFIGS.items():
        if only and config != only:
            continue
        for mode in ('plain', 'trace'):
            name = config + '_' + mode
            command = ['nvcc', '-std=c++17', '-O3', '-DNDEBUG', '-lineinfo',
                       '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v',
                       f'-DR10_CFG={index}']
            if host_compiler:
                command += ['-ccbin', host_compiler]
            if mode == 'trace':
                command += ['-DR10_TRACE', '-I' + str(source / 'overlay')]
            command += ['-I' + str(source / 'cutlass/include'),
                        '-I' + str(source / 'cutlass/tools/util/include'),
                        str(source / 'probes/r10.cu'), '-o', str(directory / name)]
            commands[name] = command
            write(directory / 'commands.json', commands)
            with (directory / (name + '.log')).open('w') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=1800, check=True)
            with (directory / (name + '.sass')).open('w') as stream:
                subprocess.run(['cuobjdump', '--dump-sass', str(directory / name)],
                               stdout=stream, timeout=120, check=True)
    write(output / 'source_hashes.json', {
        str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()})
    write(directory / 'binary_hashes.json', {n: sha(directory / n) for n in commands})
    write(output / 'build_environment.json', dict(
        hostname=os.uname().nodename,
        nvcc=subprocess.check_output(['nvcc', '--version'], text=True),
        gpu_executed=False, only_config=only,
        host_compiler=host_compiler))


def sample(output, representatives):
    nvcc_version = subprocess.check_output(['nvcc', '--version'], text=True)
    built = json.loads((output / 'build_environment.json').read_text())
    if 'release 12.9' not in nvcc_version or built['nvcc'] != nvcc_version:
        raise RuntimeError('sampling must use an identical CUDA12.9 build and environment')
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU sampling requires the main-thread scheduled Slurm allocation')
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                   '--format=csv,noheader'], text=True).strip()
    if '\n' in uuid:
        raise RuntimeError('one visible GPU required')
    for name, digest in json.loads((output / 'build/binary_hashes.json').read_text()).items():
        if sha(output / 'build' / name) != digest:
            raise ValueError('binary identity changed')
    for name, digest in json.loads((output / 'source_hashes.json').read_text()).items():
        if sha(output / 'source' / name) != digest:
            raise ValueError('source identity changed')
    cases = matrix()
    if built.get('only_config'):
        cases = [c for c in cases if c['config'] == built['only_config']]
    if representatives:
        cases = [c for c in cases if c['config'] == 'cfg_b' and c['ldb'] in (3000, 3072)]
    with open('/tmp/gh200-access-' + uuid + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        set_name = 'representatives' if representatives else 'formal'
        marker = output / ('environment-' + set_name + '.json')
        if marker.exists():
            raise ValueError('sampling set exists; use a new run for repeat measurement')
        write(marker, dict(gpu_uuid=uuid, slurm_job_id=os.environ['SLURM_JOB_ID'],
                           gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version',
                                                        '--format=csv,noheader'], text=True),
                           nvcc=subprocess.check_output(['nvcc', '--version'], text=True)))
        write(output / ('protocol-' + set_name + '.json'),
              dict(cases=cases, processes_per_mode=10, seed=20261007,
                   preparation_protocol='symmetric_scratch_memset_sync_v3',
                   scratch_bytes=1024 * 8 * 8,
                   cfg_b_trace_scope='linear_CTA_0_to_3_consumer0_last_tile_v4',
                   trace_boundary='last tile: first full-barrier acquired to mma_tail returned'))
        rng = random.Random(20261007)
        for trial in range(10):
            # Compare ldb3000/3072 first; all layout variants of a config remain adjacent.
            configs = sorted({c['config'] for c in cases})
            rng.shuffle(configs)
            for config in configs:
                ordered = [c for c in cases if c['config'] == config]
                rng.shuffle(ordered)
                for case in ordered:
                    modes = ['plain', 'trace']
                    rng.shuffle(modes)
                    for mode in modes:
                        folder = output / 'samples' / set_name / case['id'] / mode / f'{trial:02}'
                        folder.mkdir(parents=True, exist_ok=False)
                        command = [str(output / 'build' / f'{config}_{mode}')]
                        for key in ('m', 'n', 'k', 'lda', 'ldb', 'ldd'):
                            command += ['--' + key, str(case[key])]
                        write(folder / 'command.json', command)
                        with (folder / 'stdout.json').open('w') as out:
                            with (folder / 'stderr.log').open('w') as err:
                                process = subprocess.run(command, cwd=folder, stdout=out,
                                                         stderr=err, timeout=300)
                        write(folder / 'process.json', dict(returncode=process.returncode))
                        if process.returncode:
                            raise RuntimeError('probe failed: ' + str(folder))
                        events = [json.loads(line) for line in (folder / 'stdout.json').read_text().splitlines()]
                        record = next(event for event in events if event['event'] == 'measurement')
                        record['check'] = next(event for event in events if event['event'] == 'check')
                        record.update(configuration=case, trial=trial, set=set_name, mode=mode,
                                      binary_sha256=sha(output / 'build' / f'{config}_{mode}'))
                        write(folder / 'result.json', record)
        subprocess.run([sys.executable, str(output / 'source/analyze_r10.py'),
                        '--input', str(output), '--set', set_name], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('list', 'cpu-check', 'build', 'sample'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cutlass-root', type=Path)
    parser.add_argument('--config', choices=CONFIGS)
    parser.add_argument('--representatives', action='store_true')
    parser.add_argument('--host-compiler', help='e.g. /usr/bin/gcc-15 for local CUDA13 checks')
    args = parser.parse_args()
    if args.step == 'list':
        print(json.dumps(matrix(), indent=2))
    elif args.step == 'cpu-check':
        from analyze_r10 import cpu_check
        cpu_check()
    elif args.step == 'build':
        build(args.output.resolve(), args.cutlass_root.resolve(), args.config,
              args.host_compiler)
    else:
        sample(args.output.resolve(), args.representatives)


if __name__ == '__main__':
    main()

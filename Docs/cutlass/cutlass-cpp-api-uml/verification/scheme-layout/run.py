#!/usr/bin/env python3
"""Compile and execute only Host layout/coordinate checks against the snapshot."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    atlas = json.loads((ROOT / 'data/atlas.json').read_text())
    files = {f['path']: f['sha256'] for f in atlas['files']}
    source_inputs = ['include/cutlass/detail/layout.hpp', 'include/cutlass/layout/matrix.h',
                     'include/cute/layout.hpp', 'include/cute/stride.hpp']
    source_hashes = {}
    for path in source_inputs:
        if path not in files:
            raise ValueError('Expected fixed source missing: ' + path)
        actual = sha(ROOT / 'snapshot' / path)
        if actual != files[path]:
            raise ValueError('Fixed source changed: ' + path)
        source_hashes[path] = actual
    artifacts = HERE / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    compiler = shutil.which('g++')
    if not compiler:
        raise RuntimeError('A Host C++ compiler is required; do not claim a run without one')
    binary = artifacts / 'layout-probe'
    command = [compiler, '-x', 'c++', '-std=c++17', '-I', str(ROOT / 'snapshot/include'),
               '-I', '/usr/local/cuda-13.0/targets/x86_64-linux/include',
               '-I', '/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl',
               str(HERE / 'layout_probe.cu'), '-o', str(binary)]
    compilation = subprocess.run(command, text=True, capture_output=True)
    (artifacts / 'compile.log').write_text(compilation.stdout + compilation.stderr)
    report = {'commit': atlas['commit'], 'command': command,
              'compiler_version': subprocess.run([compiler, '--version'], text=True, capture_output=True, check=True).stdout,
              'source_sha256': sha(HERE / 'layout_probe.cu'), 'runner_sha256': sha(Path(__file__)),
              'fixed_header_sha256': source_hashes, 'compile_exit_code': compilation.returncode,
              'scope': 'host_layout_coordinate_mapping_only', 'gpu_kernel_launched': False,
              'gemm_runtime_verified': False, 'performance_verified': False}
    if compilation.returncode:
        report['status'] = 'compile_failed'
    else:
        execution = subprocess.run([str(binary)], text=True, capture_output=True)
        output = execution.stdout + execution.stderr
        (artifacts / 'run.log').write_text(output)
        report.update({'execution_exit_code': execution.returncode, 'output': output,
                       'binary_sha256': sha(binary)})
        cases = [dict(zip(('padding', 'lda', 'ldb', 'a_offset_1_0', 'b_offset_1_1'), map(int, m)))
                 for m in re.findall(r'padding=(\d+) lda=(\d+) ldb=(\d+) A\(1,0\)=(\d+) B\(1,1\)=(\d+)', output)]
        counts = re.search(r'HOST_LAYOUT_PASS checked=(\d+) ignored_padding_mismatches=(\d+) wrong_B_coordinate_mismatches=(\d+)', output)
        expected_cases = [
            {'padding': 0, 'lda': 128, 'ldb': 256, 'a_offset_1_0': 128, 'b_offset_1_1': 257},
            {'padding': 8, 'lda': 136, 'ldb': 264, 'a_offset_1_0': 136, 'b_offset_1_1': 265}]
        passed = execution.returncode == 0 and cases == expected_cases and counts and int(counts[1]) == 131072 and int(counts[2]) > 0 and int(counts[3]) > 0
        report.update({'cases': cases, 'status': 'host_layout_pass' if passed else 'host_layout_failed'})
        if counts:
            report.update(dict(zip(('coordinates_checked', 'ignored_padding_mismatches', 'wrong_B_coordinate_mismatches'), map(int, counts.groups()))))
    (HERE / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'host_layout_pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

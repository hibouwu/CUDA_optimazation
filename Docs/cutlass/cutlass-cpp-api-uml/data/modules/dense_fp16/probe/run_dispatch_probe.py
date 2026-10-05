#!/usr/bin/env python3
"""Archive Dense gemm overload/type evidence; only the Host printer runs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parents[3]
NVCC = Path('/usr/local/cuda-13.0/bin/nvcc')
HOST = Path('/usr/bin/g++-14')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    scope = json.loads((ATLAS/'data/scope.json').read_text())
    for item in scope['files']:
        assert sha(ATLAS/'snapshot'/item['path']) == item['sha256'], item['path']
    previous = json.loads((HERE/'run-334xibhm/metadata.json').read_text())
    assert previous['status'] == 'static_probe_pass'
    assert sha(HERE/'run-334xibhm/input.cu') == previous['probe_source_sha256']
    out = Path(tempfile.mkdtemp(prefix='dispatch-run-', dir=HERE))
    print('OUTPUT', out, flush=True)
    (out/'input.cu').write_bytes((HERE/'gemm_dispatch_probe.cu').read_bytes())
    common = [str(NVCC), '-std=c++17', '--expt-relaxed-constexpr', '-ccbin', str(HOST),
              '-Xcompiler', '-U_GNU_SOURCE', '-D_DEFAULT_SOURCE=1', '-D_POSIX_C_SOURCE=200809L',
              '-I', str(ATLAS/'snapshot/include'), '-I', str(HERE), '-arch=sm_110a']
    metadata = {'schema_version': 1, 'snapshot_commit': scope['commit'],
        'snapshot_scope_sha256': sha(ATLAS/'data/scope.json'),
        'all_snapshot_file_hashes_verified': len(scope['files']),
        'nvcc_version': subprocess.check_output([str(NVCC), '--version'], text=True),
        'host_version': subprocess.check_output([str(HOST), '--version'], text=True),
        'target': 'sm_110a', 'recipe_arch_tag': 'cutlass::arch::Sm100',
        'gpu_executed': False, 'executed_artifact': 'Host-only shape/type printer',
        'full_kernel_evidence': 'data/modules/dense_fp16/probe/run-334xibhm/metadata.json',
        'inputs': {str(p.relative_to(ATLAS)): sha(p) for p in [
            out/'input.cu', HERE/'run-334xibhm/input.cu', HERE/'run-334xibhm/host_compat.hpp', Path(__file__)]},
        'commands': []}
    def run(label, command):
        start = time.monotonic()
        result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
        (out/(label+'.stdout.txt')).write_text(result.stdout)
        (out/(label+'.stderr.txt')).write_text(result.stderr)
        item = {'label': label, 'command': [str(x) for x in command], 'returncode': result.returncode,
                'elapsed_seconds': time.monotonic()-start,
                'stdout': str((out/(label+'.stdout.txt')).relative_to(ATLAS)),
                'stderr': str((out/(label+'.stderr.txt')).relative_to(ATLAS))}
        print(label, result.returncode, flush=True)
        return item
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(run, 'nvcc_host_types', common+[out/'input.cu', '-o', out/'host_types'])
        b = pool.submit(run, 'nvcc_debug_ptx', common+['-DCOMPILE_DISPATCH_KERNEL', '--ptx', '-G', out/'input.cu', '-o', out/'probe.ptx'])
        metadata['commands'] += [a.result(), b.result()]
    if metadata['commands'][0]['returncode'] == 0:
        metadata['commands'].append(run('host_type_dump', [out/'host_types']))
    if metadata['commands'][1]['returncode'] == 0:
        metadata['commands'].append(run('ptxas_debug', [NVCC.parent/'ptxas', '-arch=sm_110a', '-g', out/'probe.ptx', '-o', out/'probe.cubin']))
        if metadata['commands'][-1]['returncode'] == 0:
            metadata['commands'].append(run('readelf_debug_info', ['/usr/bin/readelf', '--debug-dump=info', out/'probe.cubin']))
    ptx = (out/'probe.ptx').read_text() if (out/'probe.ptx').exists() else ''
    files = {int(a): b for a,b in re.findall(r'\.file\s+(\d+)\s+"([^"]+)"', ptx)}
    selected = {'include/cute/algorithm/gemm.hpp': {88, 298, 148, 197}, 'include/cute/atom/mma_atom.hpp': {104}}
    locations = []
    for line_no, line in enumerate(ptx.splitlines(), 1):
        match = re.search(r'\.loc\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if match:
            file_no, source_line, column = map(int, match.groups())
            filename = files[file_no]
            for path, relevant_lines in selected.items():
                if filename.endswith('/'+path) and source_line in relevant_lines:
                    locations.append({'ptx_line': line_no, 'path': path, 'source_line': source_line, 'column': column})
    found = {(x['path'], x['source_line']) for x in locations}
    expected = {(p,l) for p,lines in selected.items() for l in lines}
    metadata['static_observations'] = {'selected_source_locations': locations,
        'all_expected_source_locations_present': expected <= found,
        'ptx_tcgen05_f16_2sm_count': len(re.findall(r'tcgen05\.mma\.cta_group::2\.kind::f16\b', ptx))}
    metadata['status'] = 'static_dispatch_probe_pass' if all(x['returncode']==0 for x in metadata['commands']) and expected <= found else 'probe_incomplete_or_failed'
    metadata['artifacts'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    (out/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(out), 'status': metadata['status'], 'observations': metadata['static_observations']}, indent=2), flush=True)

if __name__ == '__main__':
    main()

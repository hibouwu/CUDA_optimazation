#!/usr/bin/env python3
"""NVCC static Dense recipe evidence; never call the compile-only launch anchor."""
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
BASELINE = ATLAS.parent / '02_cutlass_and_gemm/exemples/dense_baseline.cu'
OLD = BASELINE.parent / 'verification/results'
EXPECTED_COMMIT = '8f50b052e1099fb982392a622caab69b97b63128'
EXPECTED_BASELINE_SHA = '66f6dfc529bb9e66a1f5bd28acddc95adc4aa903169e851b65601dd3b457e217'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    scope = json.loads((ATLAS/'data/scope.json').read_text())
    assert scope['commit'] == EXPECTED_COMMIT
    assert sha(BASELINE) == EXPECTED_BASELINE_SHA
    old = json.loads((OLD/'manifest.json').read_text())
    assert old['cutlass_commit'] == EXPECTED_COMMIT
    assert old['source_hashes']['dense_baseline.cu'] == sha(BASELINE)
    for entry in scope['files']:
        assert sha(ATLAS/'snapshot'/entry['path']) == entry['sha256'], entry['path']
    output = Path(tempfile.mkdtemp(prefix='run-', dir=HERE))
    print('OUTPUT', output, flush=True)
    source = output/'input.cu'
    source.write_bytes((HERE/'dense_type_probe.cu').read_bytes())
    (output/'host_compat.hpp').write_bytes((HERE/'host_compat.hpp').read_bytes())
    common = [str(NVCC), '-std=c++17', '--expt-relaxed-constexpr', '-ccbin', str(HOST), '-Xcompiler', '-U_GNU_SOURCE',
              '-D_DEFAULT_SOURCE=1', '-D_POSIX_C_SOURCE=200809L', '-I', str(ATLAS/'snapshot/include'), '-arch=sm_110a']
    metadata = {'snapshot_commit': EXPECTED_COMMIT, 'snapshot_scope_sha256': sha(ATLAS/'data/scope.json'),
        'all_snapshot_file_hashes_verified': len(scope['files']),
        'baseline': {'path': str(BASELINE.relative_to(ATLAS.parent)), 'sha256': sha(BASELINE)},
        'prior_evidence': {'manifest_path': str((OLD/'manifest.json').relative_to(ATLAS.parent)),
            'manifest_sha256': sha(OLD/'manifest.json'), 'baseline_hash_matches': True,
            'commit_matches': True, 'prior_nvcc': old['nvcc'], 'prior_types_sha256': sha(OLD/'dense.types.txt')},
        'probe_source_sha256': sha(source), 'runner_sha256': sha(__file__), 'host_compatibility_shim_sha256':sha(HERE/'host_compat.hpp'),
        'nvcc_version': subprocess.check_output([str(NVCC), '--version'], text=True),
        'host_compiler_version': subprocess.check_output([str(HOST), '--version'], text=True),
        'target': 'sm_110a', 'recipe_arch_tag': 'cutlass::arch::Sm100',
        'runtime_gpu_launched': False, 'runtime_validation': 'not performed',
        'host_type_printer_only': True, 'commands': []}
    def run(label, command):
        started = time.monotonic()
        result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
        (output/(label+'.stdout.txt')).write_text(result.stdout)
        (output/(label+'.stderr.txt')).write_text(result.stderr)
        item = {'label':label, 'command':[str(x)for x in command], 'returncode':result.returncode,
                'elapsed_seconds':time.monotonic()-started,
                'stdout':str((output/(label+'.stdout.txt')).relative_to(ATLAS)),
                'stderr':str((output/(label+'.stderr.txt')).relative_to(ATLAS))}
        print(label, 'returncode', result.returncode, 'seconds', round(item['elapsed_seconds'],2), flush=True)
        return item
    with ThreadPoolExecutor(max_workers=2) as pool:
        ptx_future = pool.submit(run, 'nvcc_ptx', common+['--ptx', '-lineinfo', str(source), '-o', str(output/'probe.ptx')])
        host_future = pool.submit(run, 'nvcc_host_types', common+['-DPROBE_HOST_TYPES', str(source), '-o', str(output/'host_types')])
        metadata['commands'].extend([ptx_future.result(), host_future.result()])
    if metadata['commands'][1]['returncode'] == 0:
        metadata['commands'].append(run('host_type_dump', [output/'host_types']))
    if metadata['commands'][0]['returncode'] == 0:
        metadata['commands'].append(run('ptxas', [NVCC.parent/'ptxas','-arch=sm_110a','-v',output/'probe.ptx','-o',output/'probe.cubin']))
        if metadata['commands'][-1]['returncode'] == 0:
            metadata['commands'].append(run('nvdisasm',[NVCC.parent/'nvdisasm',output/'probe.cubin']))
    metadata['artifacts'] = {p.name: {'sha256':sha(p), 'bytes':p.stat().st_size}
                             for p in sorted(output.iterdir())if p.is_file()}
    ptx = (output/'probe.ptx').read_text()if(output/'probe.ptx').exists()else''
    sass = (output/'nvdisasm.stdout.txt').read_text()if(output/'nvdisasm.stdout.txt').exists()else''
    metadata['static_observations'] = {'device_entry_count':len(re.findall(r'^\s*(?:(?:\.visible|\.weak)\s+)?\.entry\s+',ptx,re.M)),
        'target_directives':re.findall(r'^\.target.*$',ptx,re.M),
        'mma_f16_cta_group_2_count':len(re.findall(r'tcgen05\.mma\.cta_group::2\.kind::f16\b',ptx)),
        'sass_mma_lines':[x.strip()for x in sass.splitlines()if re.search(r'\b(?:UTCHMMA|HMMA)\b',x)]}
    metadata['status'] = 'static_probe_pass' if all(c['returncode']==0 for c in metadata['commands']) and metadata['static_observations']['mma_f16_cta_group_2_count']>0 and metadata['static_observations']['device_entry_count']>0 else 'probe_incomplete_or_failed'
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':metadata['status'],'output':str(output),'observations':metadata['static_observations']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()

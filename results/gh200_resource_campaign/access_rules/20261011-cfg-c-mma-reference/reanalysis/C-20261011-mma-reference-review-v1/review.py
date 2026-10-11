#!/usr/bin/env python3
"""C-only offline source, formal-array and native-cycle review; never launch a GPU."""
from array import array
import collections
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys

sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent
run=out.parents[1]
base=run.parent
repo=run.parents[3]
build=base/'20261011-cfg-c-mma-reference-preparation'
source_review=build/'reanalysis/C-20261011-mma-reference-source-review-v1'
snapshot=source_review/'snapshot'
inputs={}
sha=lambda value:hashlib.sha256(value).hexdigest()

def file_sha(path):
    digest=sha(path.read_bytes());inputs[str(path)]=digest;return digest

def read(path):
    file_sha(path);return json.loads(path.read_text())

def archive_check(path,directory,required):
    import tarfile
    file_sha(path);names=set()
    with tarfile.open(path,'r:gz') as archive:
        for member in archive:
            if not member.isfile():continue
            name=member.name.removeprefix('./')
            assert not name.startswith('/') and '..' not in Path(name).parts and name not in names
            names.add(name)
            assert sha(archive.extractfile(member).read())==file_sha(directory/name),name
    assert set(required)<=names
    return len(names)

archive_counts={}
for root,commit in ((build,'05700b2'),(run,'0f357f40eb6cba814abd18f549719a769b6b7fcc')):
    freeze=read(root/'freeze.json')
    for name,digest in freeze['files'].items():
        assert file_sha(root/name)==file_sha(root/'returned'/name)==digest
    assert file_sha(root/'freeze.json')==file_sha(root/'returned/freeze.json')
    for name in ('freeze.json','input.tar.gz'):
        blob=subprocess.check_output(['git','show',commit+':'+str((root/name).relative_to(repo))],cwd=repo)
        assert sha(blob)==file_sha(root/name)
    receipt=read(root/'receipt.json')
    assert receipt['returncode']==0 and receipt['sha256']==file_sha(root/'output.tar.gz')
    assert (root/'returned/exit_code.txt').read_text().strip()=='0'
    archive_counts[root.name]=dict(input=archive_check(root/'input.tar.gz',root,[*freeze['files'],'freeze.json']),
        output=archive_check(root/'output.tar.gz',root/'returned',[*freeze['files'],'freeze.json']))
sources=read(build/'source_hashes.json')
assert len(sources)==841
for name,digest in sources.items():assert file_sha(build/name)==digest
assert read(run/'origin-source_hashes.json')==sources
assert read(run/'origin-build-audit.json')==read(build/'build-audit.json')

contract=read(build/'source/contract.json')
for name,digest in contract['files'].items():assert file_sha(build/'source'/name)==digest
assert file_sha(snapshot/'generator/probes/r00_wgmma.cu')==contract['source_r00_sha256']
assert file_sha(repo/'microbench/gh200_resource_campaign/access_rules/probes/r00_wgmma.cu')==contract['source_r00_sha256']
for name in ('cfg_c_mma_reference.cu','r00_common.hpp','contract.json'):
    assert file_sha(source_review/'generated'/name)==file_sha(build/'source'/name)
source=(build/'source/cfg_c_mma_reference.cu').read_text()
assert 'constexpr int Batch = 4;' in source and 'for (int i = 0; i < 16; ++i)' in source
assert 'for (int i = 0; i < 30; ++i)' not in source
assert 'G::ScaleOut::One' in source and '(o.iterations!=64 && o.iterations!=1024)' in source
assert source.index('  asm volatile("wgmma.wait_group.sync.aligned 0;')<source.index('  drain[threadIdx.x] = sum;')<source.index('    uint64_t end_cycle = clock64()')

sass_path=build/'returned/build/reference.sass'
file_sha(sass_path)
sections=sass_path.read_text().split('Function : ')
main=next(s for s in sections if 'Li4ELb1ELb0EE' in s.splitlines()[0])
short=next(s for s in sections if 'Li1ELb1ELb0EE' in s.splitlines()[0])
ops={int(m.group(1),16):line.strip() for line in main.splitlines() if (m:=re.search(r'/\*([0-9a-f]+)\*/',line))}
mma=[line for line in ops.values() if 'HGMMA.' in line]
assert len(mma)==8 and all('HGMMA.64x128x16.F32' in s and 'gdesc[UR8].tnspB' in s for s in mma)
assert sum(' R88,' in s for s in mma)==sum(' R24,' in s for s in mma)==4
assert 'WARPGROUP.ARRIVE' in ops[0x1650] and 'BRA 0x1650' in ops[0x1710]
assert 'WARPGROUP.DEPBAR.LE gsb0, 0x1' in ops[0x1700]
assert 'WARPGROUP.DEPBAR.LE gsb0, 0x0' in ops[0x1720]
assert 'SR_CLOCKLO' in ops[0x1510] and 'SR_CLOCKLO' in ops[0x1fd0]
assert 'BAR.SYNC' in ops[0x1540] and 'BAR.SYNC' in ops[0x1fb0]
checksum=[line for pc,line in ops.items() if 0x1730<=pc<=0x1f90 and 'FADD' in line]
assert len(checksum)==128 and 'STS ' in ops[0x1fa0]
assert any('ULEA ' in line for pc,line in ops.items() if 0x1570<=pc<=0x1640)
assert len([s for s in short.splitlines() if 'HGMMA.' in s])==2
compile_log=(build/'returned/build/compile.log').read_text();file_sha(build/'returned/build/compile.log')
assert compile_log.count('Used 154 registers, used 1 barriers, 1152 bytes smem')==2
assert compile_log.count('0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads')==3
build_fields=dict(re.findall(r'(\w+)=(\S+)',(build/'returned/allocation.txt').read_text()))
assert build_fields['JobId']=='740094' and 'gres/gpu' not in build_fields['AllocTRES']

# Independent inversion of the output register mapping, with exact dyadic references.
short_arrays=[]
mapping=set()
for local in range(128):
    for j in range(64):
        row=(local//32)*16+(local%32)//4+((j//2)%2)*8
        col=(local%4)*2+j%2+(j//4)*8
        mapping.add((row,col))
assert mapping=={(r,c) for r in range(64) for c in range(128)}
for count in (1,2):
    values=[]
    for thread in range(256):
        local=thread%128;group=thread//128
        for chain in range(2):
            for j in range(64):
                row=(local//32)*16+(local%32)//4+((j//2)%2)*8
                col=(local%4)*2+j%2+(j//4)*8
                exact=sum(Fraction((1+(row+2*k)%7)*(1+(col+3*k)%11),512) for k in range(16))
                exact=exact*count+Fraction(1+group+chain,64)
                value=float(exact)
                assert Fraction.from_float(value)==exact
                assert struct.unpack('<f',struct.pack('<f',value))[0]==value
                values.append(value)
    raw=struct.pack('<'+str(len(values))+'f',*values)
    (out/f'expected-short-{count}.f32').write_bytes(raw)
    short_arrays.append(dict(count=count,values=len(values),min=min(values),max=max(values),sha256=sha(raw),
        device_values_exported=False,independent_device_comparison=False))

manifest=read(run/'manifest.json');data=run/'returned'
assert manifest['status']=='frozen' and manifest['gpu_uuid']=='GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'
assert file_sha(data/'build/cfg_c_mma_reference')==file_sha(build/'returned/build/cfg_c_mma_reference')==manifest['binary_sha256']
sys.path.insert(0,str(run))
from validate_allocation import validate
allocation=(data/'allocation.txt').read_text();validate(allocation)
fields=dict(re.findall(r'(\w+)=(\S+)',allocation))
assert fields['JobId']=='740095' and fields['Restarts']=='0'
assert read(data/'assigned-context.json')['CUDA_VISIBLE_DEVICES']=='0,1,2,3'
records=read(data/'records.json')
assert [{k:r[k] for k in ('id','trial','iterations','command')} for r in records]==manifest['samples']
assert len(records)==6 and read(data/'complete.json')==dict(records=6,manifest_sha256=file_sha(run/'manifest.json'))
results=[]
for i,record in enumerate(records):
    n=64 if i%2==0 else 1024
    assert type(record['iterations']) is int and record['iterations']==n and record['trial']==i//2
    assert record['command']==['../../build/cfg_c_mma_reference','--dtype','fp16','--source','SS','--n','128',
        '--groups','2','--b-major','MN','--scope','one_cta','--mode','cfg_c_batch8_wait1','--iterations',str(n)]
    assert record['returncode']==0
    assert set(record['files'])=={f"samples/{record['id']}/{name}" for name in ('stdout.json','stderr.txt','output.f32')}
    for name,digest in record['files'].items():assert file_sha(data/name)==digest
    directory=data/'samples'/record['id'];packet=read(directory/'stdout.json')
    fixed=dict(kind='wgmma',status='measured',dtype='fp16',source='SS',scope='one_cta',n=128,k=16,chains=2,groups=2,
        threads=256,blocks=1,iterations=n,per_chain_batch=4,wait=1,b_major='MN',gpu_uuid=manifest['gpu_uuid'],
        warmup_calls=16,warmup_policy='fixed16_same_probe',unit='clock64_cycle/CTA',registers_per_thread=154,
        local_bytes_per_thread=0,static_smem_bytes=1152,dynamic_smem_bytes=32768,occupancy_limit_ctas_per_sm=1,
        max_output_error=0,input_profile='uniform_positive',coordinate_checks=[1,2],warmup_converged=True,
        output_file='output.f32',output_elements=32768)
    for key,value in fixed.items():
        assert type(packet[key]) is type(value) or key=='max_output_error'
        assert packet[key]==value,(key,packet[key])
    work=2*64*128*16*2*4*2*n
    assert packet['work_flop']==work and len(packet['stamps'])==1
    ns0,ns1,c0,c1,sm=packet['stamps'][0]
    assert all(type(v) is int for v in (ns0,ns1,c0,c1,sm)) and 0<ns0<ns1 and 0<c0<c1
    assert sm==124 and packet['elapsed']==c1-c0
    expected=Fraction(1,16)*4*n
    raw=(directory/'output.f32').read_bytes()
    assert len(raw)==131072
    values=struct.unpack('<32768f',raw)
    assert all(v==float(expected) for v in values)
    results.append(dict(id=record['id'],iterations=n,cycles=c1-c0,work_flop=work,
        flop_per_SM_cycle=work/(c1-c0),output_values_checked=len(values),sm=sm))
assert [r['cycles'] for r in results]==[66339,1049379]*3
verification=read(run/'verification.json')
assert verification['results']==results and verification['formal_output_values_checked']==196608
assert verification['short_coordinate_checks_from_host'] is True
assert verification['observation_qualified'] is verification['prediction_qualified'] is False
for summary in verification['summary']:
    rows=[r for r in results if r['iterations']==summary['iterations']]
    assert summary['cycles']==[r['cycles'] for r in rows]
    assert summary['median_cycles']==statistics.median(r['cycles'] for r in rows)
    assert summary['median_flop_per_SM_cycle']==statistics.median(r['flop_per_SM_cycle'] for r in rows)

# Recheck the supplied GEMM span in its own CTA/native-clock domain, never pool raw clocks across SMs.
gemm=base/'20261011-cfg-c-producer-tail-off-off'
spans=[]
for row in read(gemm/'returned/records.json'):
    raw=gemm/'returned'/row['raw'];assert file_sha(raw)==row['sha256']
    with gzip.open(raw,'rt') as stream:packets={p['event']:p for p in map(json.loads,stream)}
    trace=packets['call']['trace']
    for cta in range(132):
        head=trace[cta*1296:(cta+1)*1296]
        assert head[2]==head[11]==head[12] and head[6]==head[9]
        for j in range(head[6]):
            tiles=[head[16+640*wg+10*j:26+640*wg+10*j] for wg in range(2)]
            assert tiles[0][4:6]==tiles[1][4:6]
            spans.append(max(t[1] for t in tiles)-min(t[0] for t in tiles))
assert len(spans)==4320 and [min(spans),statistics.median(spans),max(spans)]==[65776,65943,67553]
gemm_source=build.parent/'20261011-cfg-c-producer-tail-check-build/source/overlay/cutlass/gemm'
collective=gemm_source/'collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp'
kernel=gemm_source/'kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp'
file_sha(collective);file_sha(kernel)
c=collective.read_text();k=kernel.read_text()
assert c.index('pipeline.consumer_wait(smem_pipe_read, barrier_token);')<c.index('v06_stamp(V06_FIRST_MMA')
assert 'tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;' in c
assert k.index('collective_mainloop.mma_tail(')<k.index('v06_stamp(V06_MAIN_END')

report=dict(build_job=740094,execution_job=740095,freeze_commit='0f357f40eb6cba814abd18f549719a769b6b7fcc',
    input_sha256=inputs,archive_files_verified=archive_counts,source_files_verified=841,
    generated_source_exact_match=True,original_R00_unchanged=True,
    per_wg_mma_per_group=8,chains_per_wg=2,warpgroups=2,registers=154,spill_bytes=0,
    sass_cycle_window=dict(start_pc='1510',stop_pc='1fd0',start_cta_sync_pc='1540',
        descriptor_setup_pcs=['1570','1640'],loop_arrive_pc='1650',wait1_pc='1700',final_wait0_pc='1720',
        checksum_fadds=128,drain_store_pc='1fa0',stop_cta_sync_pc='1fb0'),
    cpu_short_mapping_bijective=True,cpu_expected_short_arrays=short_arrays,independent_short_device_arrays_checked=False,
    formal_output_values_checked=196608,formal_values_exact=True,results=results,
    conditional_4096_reference=dict(is_measured_capacity=False,cycles_for_64_groups=65536,
        cycles_for_1024_groups=1048576,observed_excess_cycles=[803,803],intercept_fit=False,subtracted_from_GEMM=False),
    prior_gemm_span=dict(windows=4320,minimum=65776,median=65943,maximum=67553,
        unit='same-CTA native cycles',observation_qualified=False),
    missing_exported_evidence=['Nonuniform short device arrays and per-check maximum errors.',
        'Sixteen individual warmup durations; recorded converged booleans cannot be independently recomputed.'],
    gemm_differences=['All WGs/chains/q repeats use the same K16 A/B descriptors; nominal K64 is FLOP equivalence only.',
        'Reference has 256 threads and 32KiB dynamic SMEM; GEMM has producer WG, stage ring, TMA and cluster traffic.',
        'Reference always accumulates into initialized zeros; actual GEMM first group uses ScaleOut::Zero.',
        'Reference start includes stamp store/CTA sync/descriptor setup; stop includes 128 FADD checksum/drain/CTA sync.',
        'GEMM FIRST is after first consumer_wait, MAIN after mma_tail wait0 and pipeline releases; no reference checksum chain.',
        'Single CTA on SM124 with uniform repeated input and fixed16 reference prefix does not establish all-SM or GEMM service.'],
    observation_qualified=False,prediction_qualified=False,new_gpu_runs_by_C=0,public_files_modified=False,
    conclusion='Accept formal-array correctness and the reported single-SM widened-window instruction reference; no GEMM timing qualification or universal bound.')
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','gemm_differences')},indent=2))

#!/usr/bin/env python3
"""Read-only host/numerical audit; write this C-only report once."""
from array import array
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
run=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
inputs={}
def sha(path):
    value=hashlib.sha256(path.read_bytes()).hexdigest()
    inputs[str(path)]=value
    return value
def read(path):
    sha(path)
    return json.loads(path.read_text())

manifest=read(run/'manifest.json');build=Path(manifest['build_run'])
old=build.parent/'20261011-cfg-c-producer-tail-build-v2'
for root in (build,run):
    for name,digest in read(root/'freeze.json')['files'].items():
        assert sha(root/name)==sha(root/'returned'/name)==digest
    receipt=read(root/'receipt.json')
    assert receipt['returncode']==0 and sha(root/'output.tar.gz')==receipt['sha256']
assert (run/'returned/exit_code.txt').read_text().strip()=='0'
source_hashes=read(build/'source_hashes.json')
for name,digest in source_hashes.items():assert sha(build/name)==digest
for name,digest in manifest['analysis_sha256'].items():
    assert sha(run/'analysis_snapshot'/name)==sha(run/'returned/analysis_snapshot'/name)==digest
assert sha(build/'build-audit.json')==manifest['build_audit_sha256']
assert sha(build/'returned/build/cfg_c_s4_dual')==sha(run/'returned/build/cfg_c_s4_dual')==manifest['binary_sha256']
new_sass=build/'returned/build/cfg_c_s4_dual.sass';old_sass=old/'returned/build/cfg_c_s4_dual.sass'
assert sha(new_sass)==sha(old_sass)
part=next(p for p in new_sass.read_text().split('Function : ') if p.startswith('_ZN7cutlass13device_kernel'))
symbol=part.splitlines()[0].strip();resources=[]
for root in (old,build):
    path=root/'returned/build/cfg_c_s4_dual.log';sha(path)
    section=path.read_text().split('Function properties for '+symbol)[1]
    resources.append(section.split('ptxas info    : Compile time =',1)[0])
assert resources[0]==resources[1]
for name in ('source/probes/v06_trace.hpp',
             'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
             'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp'):
    assert sha(build/name)==sha(old/name)
probe_path=build/'source/probes/r18.cu';sha(probe_path);probe=probe_path.read_text()
admission=next(line for line in probe.splitlines() if 'o.mode != "warm"' in line)
assert 'o.mode != "check-producer-tail"' in admission
assert 'o.mode!="setup" && o.mode!="warm" && o.mode!="check-producer-tail"' in probe
assert '!check_output.empty() && mode != "check-producer-tail"' in probe
assert 'if (o.mode == "check-producer-tail") {\n          CUDA_CHECK(cudaMemcpyToSymbol(r18_producer_tail_observer,&value,sizeof(value)));\n          return;' in probe
start=probe.index('    if (o.mode == "check-producer-tail" ||')
checks=probe[start:probe.index('    warm_sequence();',start)]
steps=['unsigned flag=0;','set_check_flag(flag);','Gemm::run(launch_params), "baseline check run"',
       'cudaMemcpy(baseline.data()', 'gaps::fill_output_sentinel',
       'flag=o.mode == "check-producer-delay"', 'set_check_flag(flag);',
       'Gemm::run(launch_params), "swap check run"', 'cudaMemcpy(swapped.data()',
       'std::memcmp(&baseline[pos], &swapped[pos], sizeof(float))',
       'if (!o.check_output.empty() && different == 0 && nonfinite == 0)',
       'reinterpret_cast<char const*>(baseline.data()+size_t(i)*ldd)']
position=0
for step in steps:position=checks.index(step,position)+len(step)
assert 'for (int i = 0; i < 16;' not in checks
# Independent period-17 closed form, without importing the campaign reference.
table=[]
for row in range(17):
    values=[]
    for col in range(17):
        products=[((7*row+13*k)%17-8)*((5*k+11*col)%17-8) for k in range(17)]
        values.append((sum(products)*(4096//17)+sum(products[:4096%17]))/1024)
    table.append(values)
full_path=run/'returned/full-d.bin';sha(full_path)
data=array('f');data.frombytes(full_path.read_bytes())
if sys.byteorder!='little':data.byteswap()
assert len(data)==2304*3072
for row in range(2304):
    expected=array('f',(table[row%17][col%17] for col in range(3072)))
    assert data[row*3072:(row+1)*3072]==expected,row
sys.path.insert(0,str(run/'analysis_snapshot'))
from decode_cfg_c_producer_tail import decode
records=read(run/'returned/records.json')
assert [{k:row[k] for k in ('id','command')} for row in records]==manifest['samples']
context=read(run/'returned/assigned-context.json')
assert context['SLURM_JOB_ID']=='740088' and len(context['CUDA_VISIBLE_DEVICES'].split(','))==1
decoded=[];gpu=None;warm_packets=[]
for record in records:
    path=run/'returned'/record['raw']
    assert record['returncode']==0 and sha(path)==record['sha256']
    with gzip.open(path,'rt') as stream:packets=[json.loads(line) for line in stream]
    events={p['event']:p for p in packets};assert len(events)==len(packets)
    setup=events['setup'];gpu=gpu or setup['gpu_uuid'];assert setup['gpu_uuid']==gpu
    if record['id']=='full_D':
        check=events['producer_tail_check']
        assert check['elements']==len(data) and check['comparison_flag']==1
        assert check['different']==check['nonfinite']==0
        continue
    call=events['call'];flag=int(record['id'][-1]);assert setup['producer_tail_observer']==call['producer_tail_observer']==flag
    packet=decode(setup,call,gpu);assert packet['record_valid'] and not packet['errors']
    assert len(packet['records'])==132
    check=events['check']
    assert check['samples']==4096 and len(set(check['checked_indices']))==len(check['checked_values'])==4096
    assert not check['nonfinite'] and not check['padding_errors']
    for index,value in zip(check['checked_indices'],check['checked_values']):
        assert type(index) is int and 0<=index<len(data)
        assert value==table[(index//3072)%17][(index%3072)%17]
    decoded.append(dict(flag=flag,ctas=132,roles=264,prefix_calls=call['warmup_calls'],tail_writers=sum(r['producer_tail'] is not None for r in packet['records'])))
    warm_packets.append((setup,call))
# Reject paths of the exact frozen decoder; do not edit the decoder or raw.
setup,call=warm_packets[1]
bad=copy.deepcopy(setup);bad['trace_version']='cfg-c-producer-tail-v1'
try:decode(bad,call,gpu)
except ValueError:pass
else:raise AssertionError('v1 accepted')
bad=copy.deepcopy(call);bad['producer_tail_observer']=True
try:decode(setup,bad,gpu)
except ValueError:pass
else:raise AssertionError('bool flag accepted')
bad=copy.deepcopy(call);bad['trace'][15]=0
assert not decode(setup,bad,gpu)['record_valid']
setup,call=warm_packets[0];bad=copy.deepcopy(call);bad['trace'][14]=1
assert not decode(setup,bad,gpu)['record_valid']
report=dict(input_sha256=inputs,reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    gpu_uuid=gpu,allocation=context,host_full_D_entry_reachable=True,comparison_sequence='flag0/run/copy, resetD, flag1/run/copy, memcmp all logical values, export baseline',
    device_sass_byte_equal_to_740082=True,kernel_resource_text_byte_equal=True,source_files_verified=len(source_hashes),
    full_D_reference_exact_values=len(data),full_D_off_on_bit_equal_from_bound_host_check=True,
    warm_sample_reference_exact_values=8192,decoded_records=decoded,
    decoder_negative_checks=['v1 protocol','bool call flag','missing on endpoint','nonzero off endpoint'],
    observation_qualified=False,limits=['One full-D comparison and one warm process per flag; no timing or perturbation qualification.',
       'Only baseline full-D dump is saved. Off/on bit equality is supported by source-bound host memcmp and its raw result.',
       'Full-D correctness mode has no sixteen-call prefix; warm records each have sixteen same-dual-binary off calls.',
       'Actual GPU identity is retained without cross-card performance inference.'])
with (out/'report.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','limits')}))

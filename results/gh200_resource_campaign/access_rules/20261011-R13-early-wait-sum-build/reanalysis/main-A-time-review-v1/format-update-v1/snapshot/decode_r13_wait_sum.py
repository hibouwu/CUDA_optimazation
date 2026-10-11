#!/usr/bin/env python3
"""Strict native-cycle decoder for the fixed, unqualified R13 early-wait observer."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from analyze_r18 import reference

VERSION='r13-early-wait-sum-v1'
MASK=sum(1<<i for i in range(2,17))


def decode_trace(setup,call):
    expected=dict(config='cfg_b',m=2816,n=1536,k=4096,grid=[1,132,1],tile=[128,128,64],
        cluster=[1,1],stages=setup.get('stages'),trace_words=784,trace_tile_words=6,
        trace_version=VERSION,input_mode='dyadic',seed=17,swizzle=1,lda=4096,ldb=1536,ldd=1536,
        alloc_lda=4104,alloc_ldb=1536,storage_m=2816,storage_n=1536,input_map_m=2816,input_map_n=1536,
        zero_m=-1,zero_n=-1,evict=0,resolved_raster_order='along_n',raster_order='heuristic')
    if type(setup.get('stages')) is not int or setup['stages'] not in (4,6):
        raise ValueError('unsupported stage count')
    for key,value in expected.items():
        if setup.get(key)!=value:
            raise ValueError('fixed setup mismatch: '+key)
    flag=setup.get('wait_sum_observer')
    if type(flag) is not int or flag not in (0,1):
        raise ValueError('observer must be 0 or 1')
    history=dict(wait_sum_protocol=VERSION,wait_sum_observer=flag,wait_sum_prefix_flag=0,
        wait_sum_first_group=2,wait_sum_last_group=16,warmup_policy='common-sw1-16',
        warmup_fixed_calls=16,warmup_swizzle=1)
    for packet in (setup,call):
        if any(packet.get(key)!=value for key,value in history.items()):
            raise ValueError('observer or prefix identity mismatch')
    if setup.get('mode')!='warm' or call.get('label')!='warm' or call.get('target_swizzle')!=1:
        raise ValueError('expected one warm target')
    warm=call.get('warmup_us',[])
    if call.get('warmup_calls')!=16 or len(warm)!=16 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in warm):
        raise ValueError('incomplete common prefix')
    elapsed=call.get('elapsed_us')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<=0:
        raise ValueError('invalid target event')
    words=call['trace']
    if len(words)!=132*784 or any(type(v) is not int or not 0<=v<2**64 for v in words):
        raise ValueError('expected complete uint64 trace')
    if any(words[784:]):
        raise ValueError('non-target CTA wrote sparse trace')
    b=words[:784]
    allowed={1,13,14,15,16,17,18,19,20,21,22,23,24,31,32,34,35,36,37,38,39,40}
    if any(v for i,v in enumerate(b) if i not in allowed):
        raise ValueError('unexpected field or overflow')
    if b[31]!=0x52313309 or not 1<=b[1]<=132 or b[1]!=b[23] or b[13:15]!=[12,1] or b[40]!=flag:
        raise ValueError('marker, SM, work or target flag mismatch')
    mapping=dict(first_mma=21,commit1=20,full_wait_begin16=15,commit16=17,wait1_end16=18,
        full_wait_begin17=16,commit17=19,commit63=24,mma_return=32,main_end=22)
    clocks={name:b[index] for name,index in mapping.items()}
    ordered=list(clocks.values())
    if not all(ordered) or any(a>z for a,z in zip(ordered,ordered[1:])):
        raise ValueError('progress endpoint order mismatch')
    duration=b[17]-b[20]
    if not duration:
        raise ValueError('empty early window')
    if flag:
        if b[35]!=15 or b[36]!=MASK or b[39]:
            raise ValueError('missing, duplicated or invalid wait intervals')
        if not b[20]<=b[37]<=b[15]<=b[38]<=b[17] or b[34]>b[38]-b[37]:
            raise ValueError('cumulative waits escape the early window')
        wait=b[34]
        outside=duration-wait
    else:
        if any(b[34:40]):
            raise ValueError('disabled observer wrote wait fields')
        wait=outside=None
    return dict(cta=0,physical_thread=256,WG_local_output_ordinal=0,cta_output_ordinal=1,
        work=[11,0],sm=b[1]-1,stages=setup['stages'],observer=flag,
        early_cycles=duration,full_wait_cycles=wait,outside_wait_cycles=outside,
        full_wait_ns=None,outside_wait_ns=None,clocks=clocks,
        target_event_us=elapsed,completed_prefix_event_us=warm,
        scope='Same-thread recorded cycle partition. Outside includes observer arithmetic; off is unobserved, not zero wait.',
        hardware_ready_identified=False,observation_qualified=False,prediction_qualified=False)


def decode(events,expected_gpu):
    packets={packet['event']:packet for packet in events}
    required={'setup','call','input_zero_storage','check'}
    if len(packets)!=len(events) or set(packets)!=required:
        raise ValueError('expected setup, target call, empty input-region check and numerical check')
    witness=packets['input_zero_storage']
    # The frozen commands explicitly set --input-map-* and this fixed geometry
    # has no extra explicit-zero rows or columns. Do not discard this packet.
    if any(type(witness.get(key)) is not int or witness[key]!=0 for key in ('checked','nonzero')):
        raise ValueError('unexpected explicit-zero input region')
    setup,call,check=(packets[name] for name in ('setup','call','check'))
    if setup.get('gpu_uuid')!=expected_gpu:
        raise ValueError('wrong GPU')
    result=decode_trace(setup,call)
    indices,values=check.get('checked_indices',[]),check.get('checked_values',[])
    if (check.get('status')!='ok' or check.get('samples')!=4096 or check.get('nonfinite')!=0
            or check.get('padding_errors')!=0 or check.get('max_storage_reference_error')!=0
            or len(indices)!=4096 or len(values)!=4096 or len(set(indices))!=4096):
        raise ValueError('incomplete or failed numerical check')
    for index,value in zip(indices,values):
        if type(index) is not int or not 0<=index<2816*1536 or type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('invalid numerical witness')
        row,col=divmod(index,1536)
        if value!=reference(row%17,col%17,4096,17):
            raise ValueError('independent GEMM check failed')
    result.update(gpu_uuid=expected_gpu,independently_checked_values=4096,full_D_verified=False)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--expected-raw-sha256',required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    payload=args.input.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=args.expected_raw_sha256:
        raise ValueError('raw identity changed')
    text=gzip.decompress(payload).decode() if args.input.suffix=='.gz' else payload.decode()
    result=decode([json.loads(line) for line in text.splitlines()],args.gpu_uuid)
    result.update(raw_sha256=args.expected_raw_sha256,decoder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')

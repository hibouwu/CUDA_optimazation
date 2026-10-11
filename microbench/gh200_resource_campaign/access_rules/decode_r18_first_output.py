#!/usr/bin/env python3
"""Strict decoding only; file hashes, numerical checks and observer gates are separate."""
from analyze_r18_first_output import EVENTS, SELECTED_CTAS, first_output_windows
from v08_model import scheduled_work

VERSION='r18-first-output-v1'
WORDS=1318


def decode(setup, call, invocation_id):
    fixed=dict(config='cfg_a',m=960,n=2816,k=4096,grid=[132,1,1],tile=[128,128,64],
        cluster=[2,1],stages=6,trace_words=WORDS,trace_tile_words=10,trace_version=VERSION)
    if any(setup.get(k)!=v for k,v in fixed.items()):raise ValueError('first-output setup mismatch')
    flag=setup.get('first_output_observer')
    if type(flag) is not int or flag not in (0,1) or call.get('first_output_observer')!=flag:
        raise ValueError('first-output flag mismatch')
    for packet in (setup,call):
        if packet.get('first_output_warmup')!=0 or packet.get('warmup_policy')!='common-sw1-16':
            raise ValueError('first-output prefix identity mismatch')
    if call.get('warmup_calls')!=16:raise ValueError('sixteen prefix calls required')
    trace=call.get('trace')
    if not isinstance(trace,list) or len(trace)!=132*WORDS:
        raise ValueError('first-output trace capacity mismatch')
    if any(type(v) is not int or not 0<=v<2**64 for v in trace):
        raise ValueError('raw uint64 trace required')
    work=scheduled_work('cfg_a',960,2816,[132,1,1],1)
    records=[]
    for cta in range(132):
        b=trace[cta*WORDS:(cta+1)*WORDS]
        if cta not in SELECTED_CTAS:
            if any(b[1296:]):raise ValueError('unselected CTA wrote extra records')
            continue
        if b[10] or not b[2]:raise ValueError('overflow or missing SM identity')
        for role in (1,2):
            if b[10+role]!=b[2]:raise ValueError('consumer SM changed')
            head=16+(role-1)*640
            if b[head+4:head+6]!=[x+1 for x in work[cta][0]]:
                raise ValueError('first-output coordinate mismatch')
            raw=b[1296+(role-1)*11:1296+role*11]
            if raw[0]!=0x52313810+flag:raise ValueError('first-output marker mismatch')
            if not flag:
                if any(raw[1:]):raise ValueError('disabled observer recorded cuts')
                continue
            stamps={'FIRST':dict(clock64=b[head],globaltimer_ns=b[head+6]),
                    'MAIN':dict(clock64=b[head+1],globaltimer_ns=b[head+7])}
            for index,event in enumerate(EVENTS[1:-1]):
                stamps[event]=dict(clock64=raw[1+2*index],globaltimer_ns=raw[2+2*index])
            record=dict(invocation_id=invocation_id,config='cfg_a',cta=cta,role=role,
                physical_thread=role*128,tile_seq=0,sm_start=b[2]-1,sm_end=b[10+role]-1,
                stamps=stamps)
            records.append(dict(record=record,partition=first_output_windows(record)))
    if len(records)!=(24 if flag else 0):raise ValueError('first-output writer coverage mismatch')
    return dict(enabled=bool(flag),records=records,observation_qualified=False,
        limits='Decode/source-order checks only; numerical, binary, preparation and perturbation qualification required.')

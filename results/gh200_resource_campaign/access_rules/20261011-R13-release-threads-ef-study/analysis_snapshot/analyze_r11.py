#!/usr/bin/env python3
"""R11 independent value/work/timing replay from frozen process witnesses."""
import argparse
import csv
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct


def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def med(x):return statistics.median(x)
def cv(x):return statistics.pstdev(x)/statistics.mean(x) if len(x)>1 else 0.0


@lru_cache(None)
def recurrence(value,iterations,pair,multiplier=0):
    for _ in range(iterations):
        if pair==0:value+=6*(value&0xffffffff)
        else:value=(value*multiplier+3)&0xffffffff
    return value


@lru_cache(None)
def conversions(base,checksum,units):
    for unit in range(units):
        sign=-1 if unit%2 else 1
        for _ in range(16):
            base+=sign/256
            checksum=(checksum+struct.unpack('<H',struct.pack('<e',base))[0])&0xffffffff
    return bits(base),checksum


@lru_cache(None)
def dot(row,col):
    return sum((1+(row+2*k)%7)*(1+(col+3*k)%11) for k in range(16))/512


def replay(directory,record):
    def raw(name):
        with gzip.open(directory/(name+'.gz'),'rb') as stream:data=stream.read()
        expected=record['files'][name]
        if len(data)!=expected['bytes'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:
            raise ValueError('raw witness changed: '+str(directory/name))
        return struct.unpack('<'+'Q'*(len(data)//8),data)
    initial=raw('initial.u64');observed=raw('output.u64')
    pair,order,units=record['pair'],record['order'],record['units']
    group=128 if pair==2 else 32
    checked=0
    for t in range(record['threads']):
        local,role=t%group,t//group
        row=list(initial[t*129:(t+1)*129])
        # Independently reconstruct the opaque initial state; a corrupt fixture
        # cannot redefine the expected values.
        wanted=[0]*129
        if pair==2 and role==0:
            for j in range(64):wanted[j]=bits((local+1)/1024+j/32)
        for j in range(8):
            wanted[64+j]=bits((local+1)/1024+j/16) if pair==0 else 0
            wanted[72+j]=bits((local+1)/32+j/16) if pair==1 else 0
            wanted[80+j]=((local+1)<<32)+j+1 if pair==0 else 0
            wanted[88+j]=(local+1)*257+j+1 if pair!=0 else 0
        if row!=wanted:raise ValueError('incorrect initial data: '+str(directory))
        active_a=role==0 and order in (0,2,3,4,5)
        active_b=role==(1 if order==5 or record['b_role'] else 0) and order in (1,2,3,4,5)
        if active_a:
            if pair==0:
                for j in range(8):wanted[64+j]=bits((local+1)/1024+j/16+(units%2)*(local+1)*5/256)
            elif pair==1:
                for j in range(32):wanted[96+j]=(local+1)*256+j
            else:
                for j in range(64):
                    r=(local//32)*16+(local%32)//4+((j//2)%2)*8
                    c=(local%4)*2+j%2+(j//4)*8
                    wanted[j]=bits((local+1)/1024+j/32+(units%2)*128*dot(r,c))
        if active_b:
            for j in range(8):
                if pair==0:wanted[80+j]=recurrence(wanted[80+j],units*16,0)
                elif pair==1:wanted[72+j],wanted[88+j]=conversions((local+1)/32+j/16,wanted[88+j],units)
                else:wanted[88+j]=recurrence(wanted[88+j],units*16,2,local+17)
        sf=(2*units)/1024 if pair!=2 or role==0 else 0.;su=(1<<48)+2*units
        for j in range(8):
            sf=f32(sf+struct.unpack('<f',struct.pack('<I',wanted[64+j]))[0])
            sf=f32(sf+struct.unpack('<f',struct.pack('<I',wanted[72+j]))[0])
            su=(((su<<7)&0xffffffffffffffff)^(su>>5)^(wanted[88+j] if pair==2 else wanted[80+j]))
            if pair==2:su=(su+0x9e3779b9)&0xffffffffffffffff
            else:su=(su+wanted[88+j]+0x9e3779b9)&0xffffffffffffffff
        if pair==2:
            for j in range(64):sf=f32(sf+struct.unpack('<f',struct.pack('<I',wanted[j]))[0])
        if pair==1:
            for value in wanted[96:128]:
                su=((((su<<7)&0xffffffffffffffff)^(su>>5)^value)+0x9e3779b9)&0xffffffffffffffff
        wanted[128]=su^bits(sf)
        found=observed[t*129:(t+1)*129]
        for j,(a,b) in enumerate(zip(found,wanted)):
            if a!=b:raise ValueError(f'numeric mismatch {directory}: thread {t}, slot {j}, got {a}, want {b}')
        checked+=129
    starts=[];ends=[]
    for start,end,sm in record['stamps']:
        if start==0 or end<start or sm!=record['stamps'][0][2]:raise ValueError('bad same-SM timeline')
        starts.append(start);ends.append(end)
    if max(ends)-min(starts)!=record['elapsed_cycles']:raise ValueError('wrong envelope')
    return checked


def sass_check(root):
    text=(root/'build/sass.txt').read_text();functions=[]
    for body in text.split('Function : ')[1:]:
        name=body.splitlines()[0].strip();match=re.search(r'r11_probeILi([01])E',name)
        tensor=re.search(r'r11_tensorILi([0-6])E',name)
        if tensor:match=tensor
        if not match:continue
        pair=2 if tensor else int(match[1]);lines=[]
        for x in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^;]+);',body):lines.append((int(x[1],16),x[2]))
        op_names=['FFMA','IMAD.WIDE','IMAD','LDS.128','F2FP','F2F','HGMMA']
        counts={op:sum(bool(re.search(r'\b'+re.escape(op)+r'\b',s)) for _,s in lines) for op in op_names}
        loops=[]
        for addr,ins in lines:
            bra=re.search(r'\bBRA(?:\.\w+)*\s+0x([0-9a-f]+)',ins)
            if bra and int(bra[1],16)<addr:
                lo=int(bra[1],16);members=[s for a,s in lines if lo<=a<=addr]
                loops.append(dict(start=lo,end=addr,counts={op:sum(bool(re.search(r'\b'+re.escape(op)+r'\b',s)) for s in members) for op in op_names}))
        needed=['FFMA','IMAD.WIDE'] if pair==0 else ['LDS.128','F2F'] if pair==1 else ['HGMMA','IMAD']
        if not tensor or int(tensor[1]) in [0,2,3,4,5]:
            if not all(counts[x]>=8 for x in needed if not (tensor and int(tensor[1])==0 and x=='IMAD')):raise ValueError('missing operations: '+name)
        if any(re.search(r'\b(?:LDL|STL)\b',s) for _,s in lines):raise ValueError('spill/local instructions: '+name)
        clocks=[a for a,s in lines if 'SR_CLOCKLO' in s]
        if len(clocks)!=(4 if tensor else 2):raise ValueError('expected one input-dependent and one completion-dependent clock')
        if not any('LDS.64' in s for a,s in lines if a<clocks[0]):raise ValueError('no shared ready load before start')
        functions.append(dict(pair=pair,kernel=name,opcode_counts=counts,loops=loops,clock_addresses=clocks,
                              automatic_wgmma_waits=[s for _,s in lines if 'WARPGROUP.DEPBAR' in s],
                              mma_shapes=sorted(set(re.findall(r'HGMMA\.([^\s;]+)',body)))))
    if len(functions)!=9:raise ValueError('expected two scalar images plus seven tensor organizations')
    return functions


def analyze(root,output=None,sass_only=False,check_only=False):
    output=output or root/'analysis';output.mkdir(parents=True,exist_ok=True)
    facts=sass_check(root)
    (output/'sass-check.json').write_text(json.dumps(facts,indent=2)+'\n')
    if sass_only:print('R11 two scalar and seven tensor images checked');return
    rows=[];words=0
    for p in sorted((root/'samples').glob('*/*/result.json')):
        r=json.loads(p.read_text());words+=replay(p.parent,r);rows.append(r)
    (output/'checks.json').write_text(json.dumps(dict(processes=len(rows),checked_u64_words=words,numeric_errors=0),indent=2)+'\n')
    if check_only:print('R11 values:',len(rows),'processes,',words,'words');return
    cases=json.loads((root/'cases.json').read_text());conditions=[]
    for c in cases:
        rs=[r for r in rows if r['case_id']==c['id'] and r['label'].startswith('formal-')]
        if not rs:raise ValueError('missing formal case '+c['id'])
        times=[r['elapsed_cycles'] for r in rs]
        conditions.append(dict(case=c['id'],pair=c['pair'],order=c['order'],cycles=med(times),
          samples=len(rs),cv=cv(times),registers=rs[0]['registers'],
          qualified=cv(times)<=.05 and all(r['warmup_converged'] for r in rs),
          work_a=(32768 if c['order'] in (0,2,3,4,5) else 0),
          work_b=(32768 if c['order'] in (1,2,3,4,5) else 0)))
    joint=[]
    for pair in range(3):
        points={r['order']:r for r in conditions if r['pair']==pair}
        empty=[r['elapsed_cycles'] for r in rows if r['pair']==pair and r['label'].startswith('empty-')]
        matched=[r['elapsed_cycles'] for r in rows if r['pair']==pair and r['label'].startswith('matched-b-')]
        t0=med(empty);ta=points[0]['cycles'];tb=points[1]['cycles'];tb_cross=med(matched)
        for order in [2,3,4,5]:
            reference_b=tb_cross if order==5 else tb
            joint.append(dict(pair=pair,order=order,cycles=points[order]['cycles'],a_only=ta,
                 b_only=reference_b,empty=t0,raw_sum=ta+reference_b,
                 common_cost_corrected_sum=ta+reference_b-t0,
                 max=max(ta,reference_b),relative_to_max=points[order]['cycles']/max(ta,reference_b)-1,
                 relative_to_corrected_sum=points[order]['cycles']/(ta+reference_b-t0)-1))
    with (output/'cases.csv').open('w',newline='') as s:
        w=csv.DictWriter(s,fieldnames=list(conditions[0]));w.writeheader();w.writerows(conditions)
    result=dict(resource_conditions=18,processes=len(rows),checked_words=words,conditions=conditions,
      joint_service=joint,unit='same_SM_clock64_cycles_CTA_envelope',
      work_definition='per participating path: 128 actions/unit, 256 units; scalar32 lanes, tensor128 lanes',
      qualification='conditioned_joint_service_not_physical_pipeline_partition',
      analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'rules.json').write_text(json.dumps(result,indent=2)+'\n')
    print('R11 formal conditions',len(conditions),'qualified',sum(c['qualified'] for c in conditions))
    for r in joint:print(r['pair'],r['order'],round(r['cycles']),f"vs max {r['relative_to_max']:.1%}")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--sass-only',action='store_true')
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    analyze(a.input.resolve(),a.output,a.sass_only,a.check_only)

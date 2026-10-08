#!/usr/bin/env python3
"""Independent R12 physical-layout, arithmetic, work and timing replay."""
import argparse,csv,gzip,hashlib,json,re,statistics,struct
from pathlib import Path


def half_bits(v):return struct.unpack('<H',struct.pack('<e',v))[0]
def float_bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def median(x):return statistics.median(x)
def cv(x):return statistics.pstdev(x)/statistics.mean(x) if len(x)>1 else 0.


def xor128(element):
    return element^((element>>3)&0x38)


def a_address(r,k,swizzle):
    q=r*64+k if swizzle else (r%8)*8+k%8+(r//8)*64+(k//8)*512
    return xor128(q) if swizzle else q


def b_address(n,k,swizzle):
    q=n%64+k*64+(n//64)*4096 if swizzle else n%8+(k%8)*8+(k//8)*64+(n//8)*512
    return 4096+(xor128(q) if swizzle else q)


def payload_address(row,col,swizzle):
    q=row*64+col
    return xor128(q) if swizzle else q


def replay(path,r):
    def raw(name,code):
        with gzip.open(path/(name+'.gz'),'rb') as f:data=f.read()
        meta=r['files'][name]
        if len(data)!=meta['bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256']:
            raise ValueError('raw witness mismatch: '+str(path/name))
        return struct.unpack('<'+code*(len(data)//struct.calcsize(code)),data)
    source=raw('source.u16','H');initial=raw('initial.f32','f')
    shared=raw('shared.u16','H');accum=raw('accum.f32','f')
    sw=r['swizzle'];pair=r['pair'];org=r['order'];units=r['units']
    do_a=org in [0,2,3,5];do_b=org in [1,2,3,4]
    has_tma=pair in [0,2] and do_a
    has_mma=pair==0 and do_b or pair==1 and do_a
    has_sts=pair in [1,2] and do_b
    if any(source[q]!=((q*13+17)^0x4321)&65535 for q in range(32768)):
        raise ValueError('incorrect global source fixture')
    expected=[0xdead]*77824
    for row in range(64):
        for k in range(64):expected[a_address(row,k,sw)]=half_bits((1+(row+2*k)%7)/16)
    for col in range(128):
        for k in range(64):expected[b_address(col,k,sw)]=half_bits((1+(col+3*k)%11)/32)
    if has_tma:
        for tile in range(4):
            for row in range(128):
                for col in range(64):
                    expected[12288+tile*8192+payload_address(row,col,sw)]=source[tile*8192+row*64+col]
    if has_sts:
        for row in range(512):
            for col in range(64):
                q=row*64+col
                expected[45056+payload_address(row,col,sw)]=((q*17+(units-1)*31)^0x1234)&65535
    for q,(a,b) in enumerate(zip(shared,expected)):
        if a!=b:raise ValueError(f'wrong shared value at half {q}: {path}: {a} != {b}')
    for t in range(128):
        for j in range(64):
            init=(t+1)/1024+j/32
            if initial[t*64+j]!=init:raise ValueError('incorrect accumulator fixture')
            row=(t//32)*16+(t%32)//4+((j//2)%2)*8
            col=(t%4)*2+j%2+(j//4)*8
            dot=sum((1+(row+2*k)%7)*(1+(col+3*k)%11) for k in range(16))/512
            value=init+(units*64*dot if has_mma else 0)
            if float_bits(accum[t*64+j])!=float_bits(value):
                raise ValueError(f'wrong accumulator {path}: thread{t} slot{j}: {accum[t*64+j]} != {value}')
    starts=[];ends=[]
    for start,end,sm in r['stamps']:
        if start==0 or end<start or sm!=r['stamps'][0][2]:raise ValueError('invalid timeline')
        starts.append(start);ends.append(end)
    if max(ends)-min(starts)!=r['elapsed_cycles']:raise ValueError('wrong cycle envelope')
    return dict(transport=65536*units if has_tma else 0,stores=65536*units if has_sts else 0,
                matrix_flop=2*64*128*16*64*units if has_mma else 0,checked_halves=77824,
                checked_accumulators=8192,start_skew=max(starts)-min(starts))


def sass_checks(root):
    text=(root/'build/sass.txt').read_text();facts=[]
    pattern=r'r12_probeILi([012])ELb([01])ELi([0-6])E'
    for body in text.split('Function : ')[1:]:
        name=body.splitlines()[0].strip();m=re.search(pattern,name)
        if not m:continue
        pair,sw,org=map(int,m.groups())
        lines=[s for s in body.splitlines() if re.search(r'/\*[0-9a-f]+\*/',s)]
        if any(re.search(r'\b(?:LDL|STL)\b',s) for s in lines):raise ValueError('local/spill: '+name)
        has_tma=pair in [0,2] and org in [0,2,3,5]
        has_mma=pair==0 and org in [1,2,3,4] or pair==1 and org in [0,2,3,5]
        has_sts=pair in [1,2] and org in [1,2,3,4]
        hgmma=sum('HGMMA' in s for s in lines);tma=sum('UTMALDG' in s for s in lines)
        sts128=sum('STS.128' in s for s in lines)
        if has_mma and hgmma!=8:raise ValueError('expected 8 MMA per loop: '+name)
        if has_tma and tma!=4:raise ValueError('expected 4 TMA requests per unit: '+name)
        if has_sts and sts128<1:raise ValueError('missing vector stores: '+name)
        waits=[s.strip() for s in lines if 'WARPGROUP.DEPBAR' in s]
        facts.append(dict(kernel=name,pair=pair,swizzle=sw,order=org,hgmma=hgmma,tma=tma,
                          sts128=sts128,waits=waits))
    if len(facts)!=42:raise ValueError('expected 24 kernels plus 18 matched phase/empty controls')
    return facts


def analyze(root,output=None,sass_only=False,check_only=False):
    out=output or root/'analysis';out.mkdir(parents=True,exist_ok=True)
    facts=sass_checks(root);(out/'sass-check.json').write_text(json.dumps(facts,indent=2)+'\n')
    if sass_only:print('R12 SASS:',len(facts),'kernels');return
    records=[];count=0
    for p in sorted((root/'samples').glob('*/*/result.json')):
        r=json.loads(p.read_text());work=replay(p.parent,r);r['work']=work;records.append(r);count+=work['checked_halves']+work['checked_accumulators']
    (out/'checks.json').write_text(json.dumps(dict(processes=len(records),checked_values=count,numeric_errors=0),indent=2)+'\n')
    if check_only:print('R12 exact replay:',len(records),'processes',count,'values');return
    cases=json.loads((root/'cases.json').read_text());rows=[]
    for c in cases:
        rs=[r for r in records if r['case_id']==c['id'] and r['label'].startswith('formal-')]
        if not rs:raise ValueError('missing formal '+c['id'])
        ts=[r['elapsed_cycles'] for r in rs]
        rows.append(dict(case=c['id'],pair=c['pair'],swizzle=c['swizzle'],order=c['order'],
            cycles=median(ts),cv=cv(ts),samples=len(rs),registers=rs[0]['registers'],
            qualified=cv(ts)<=.05 and all(r['warmup_converged'] for r in rs),
            **{k:rs[0]['work'][k] for k in ['transport','stores','matrix_flop']}))
    groups=[]
    for pair in range(3):
        for sw in range(2):
            rr={r['order']:r for r in rows if r['pair']==pair and r['swizzle']==sw}
            ta,tb,serial,parallel=[rr[i]['cycles'] for i in range(4)]
            groups.append(dict(pair=pair,swizzle=sw,a=ta,b=tb,serial=serial,concurrent=parallel,
                raw_sum=ta+tb,raw_max=max(ta,tb),concurrent_vs_serial=parallel/serial-1,
                concurrent_vs_max=parallel/max(ta,tb)-1))
    with (out/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(resource_conditions=24,processes=len(records),checked_values=count,conditions=rows,
                joint_service=groups,unit='SM_clock64_cycles_one_CTA',scope='independent_regions_384_threads',
                physical_SMEM_ports_proven=False,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'rules.json').write_text(json.dumps(result,indent=2)+'\n')
    print('R12 conditions',len(rows),'qualified',sum(r['qualified'] for r in rows))
    for g in groups:print(g)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--sass-only',action='store_true')
    p.add_argument('--check-only',action='store_true');a=p.parse_args();analyze(a.input.resolve(),a.output,a.sass_only,a.check_only)

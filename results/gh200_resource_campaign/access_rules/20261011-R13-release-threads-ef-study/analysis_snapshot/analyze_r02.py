#!/usr/bin/env python3
"""Independent R02 witness/timing/SASS replay. No runner work-count functions imported."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct


def bits(value):
    return struct.unpack('<I',struct.pack('<f',value))[0]


def replay(path, record):
    def raw(name):
        data=gzip.open(path/(name+'.gz'),'rb').read()
        identity=record['files'][name]
        if len(data)!=identity['bytes'] or hashlib.sha256(data).hexdigest()!=identity['sha256']:
            raise ValueError('witness hash/size mismatch: '+str(path/name))
        return data
    values=struct.unpack('<'+'Q'*(record['threads']*49),raw('output.u64'))
    inputs=struct.unpack('<16f',raw('input.f32'))
    wanted=tuple([1.25+(i/64 if record['witness'] else 0) for i in range(8)]+
                 [1/128+(i%3/1024 if record['witness'] else 0) for i in range(8)])
    if inputs != wanted:
        raise ValueError('unexpected runtime source witness')
    family,pool,steps=record['family'],record['pool'],record['steps']
    mixed=family>=3
    fc=0 if family==2 else 8 if mixed else pool
    ic=pool if family==2 else 8 if family==3 else 0
    checked=0
    for t in range(record['threads']):
        af=not mixed or t//32==0
        ab=not mixed or t//32==(1 if record['split'] else 0)
        f=[0.0]*32;u=[0]*32;lds=[0]*32
        for j in range(fc):
            # Exact dyadic arithmetic in integer units, independent of CUDA replay.
            numerator=t+1+64*j
            f[j]=numerator/1024
            if af:
                if family==0:
                    if record['source']==2 and pool==1:
                        increment=(steps//8)*sum(inputs[q]*inputs[8+q] for q in range(8))
                    else:
                        q=j if record['source']==2 else 0
                        increment=(steps//fc)*(1.25 if record['source']==0 else inputs[q])*inputs[8+q]
                    f[j]+=increment
                else:f[j]+=(steps//fc)*inputs[8]
        for j in range(ic):
            lane=t%32 if mixed else t
            u[j]=((lane+1)<<32)+j+(1 if family==3 else 0)
            if ab:
                if family==3:
                    for _ in range(steps//ic):u[j]+=6*(u[j]&0xffffffff)
                else:u[j]+=(steps//ic)*(lane+17)*3
        if family==4 and ab:lds=[(t%32+1)*256+j for j in range(32)]
        checksum=sum(u)^bits(sum(f))
        if family==4:checksum=(sum(lds))^bits(sum(f))
        if checksum==0:raise ValueError('consumer timer witness may not be zero')
        expected=[0]*49
        for j in range(32):
            if family==2:expected[j]=u[j]
            elif family==3:expected[j]=bits(f[j]) if j<8 else u[j-8] if j<16 else 0
            elif family==4:expected[j]=lds[j]
            else:expected[j]=bits(f[j])
        if family==4:
            for j in range(8):expected[32+j]=bits(f[j])
        else:
            for j in range(16):expected[32+j]=bits(inputs[j])
        expected[48]=checksum
        observed=values[t*49:(t+1)*49]
        for j,(a,b) in enumerate(zip(observed,expected)):
            if a!=b:raise ValueError(f'wrong retained result: {path}, lane {t}, slot {j}: {a} != {b}')
        checked+=49
    starts=[];progress=[];ends=[]
    for start,mid,end,sm in record['stamps']:
        if sm!=record['sm'] or end<start or (record['intermediate'] and not start<=mid<=end):
            raise ValueError('invalid same-SM event ordering')
        starts.append(start);progress.append(mid);ends.append(end)
    if max(ends)-min(starts)!=record['consumed_cycles']:
        raise ValueError('wrong consumer envelope')
    if record['intermediate'] and max(progress)-min(starts)!=record['progress_cycles']:
        raise ValueError('wrong producer envelope')
    return checked


def sass_checks(path):
    text=path.read_text()
    result=[]
    expression=r'r02_probeILi(\d+)ELi(\d+)ELi(\d+)ELb([01])ELi(\d+)E'
    for body in text.split('Function : ')[1:]:
        name=body.splitlines()[0].strip()
        match=re.search(expression,name)
        if not match:continue
        family,source,pool,split,position=map(int,match.groups())
        instructions=[]
        for line in body.splitlines():
            m=re.search(r'/\*([0-9a-f]+)\*/\s+([^;]+);',line)
            if m:instructions.append((int(m[1],16),m[2].strip()))
        loops=[]
        for address,ins in instructions:
            b=re.search(r'\bBRA(?:\.\w+)*\s+0x([0-9a-f]+)',ins)
            if b and int(b[1],16)<address:
                target=int(b[1],16)
                selected=[s for a,s in instructions if target<=a<=address]
                counts={op:sum(bool(re.search(r'\b'+re.escape(op)+r'\b',s)) for s in selected)
                        for op in ['FFMA','FADD','IMAD.WIDE','LDS.128','IADD3']}
                loops.append(dict(start=target,end=address,counts=counts,instructions=selected))
        needed=['FFMA'] if family==0 else ['FADD'] if family==1 else ['IADD3'] if family==2 else ['FADD','IMAD.WIDE'] if family==3 else ['FADD','LDS.128']
        if family==2:
            producer=next((loop for loop in loops if loop['counts']['IMAD.WIDE']==64
                           or loop['counts']['IADD3']>=128),None)
        else:
            producer=next((loop for loop in loops
                           if all(loop['counts'][op]==64 for op in needed)),None)
        if producer is None:
            raise ValueError(f'cannot establish 64 target actions in producer loop: {name}; loops={[(l["start"],l["counts"]) for l in loops]}')
        clocks=[(a,s) for a,s in instructions if 'SR_CLOCKLO' in s]
        if len(clocks)!=3:
            raise ValueError('unexpected timer count: '+name)
        before_start=[s for a,s in instructions if a<clocks[0][0]]
        if not any('LDS.64' in s for s in before_start[-12:]) or not any('ISETP.NE' in s for s in before_start[-12:]):
            raise ValueError('first timer lacks shared-gate-dependent readiness: '+name)
        end_address=clocks[-1][0]
        preceding=[s for a,s in instructions if producer['end']<a<end_address]
        # ISETP must read the dependent final checksum before the last clock.
        # ptxas may make the clock unconditional and select its value afterward;
        # the checksum-dependent instruction still precedes it in the issue stream.
        if not any('ISETP.NE' in s for s in preceding):
            raise ValueError('missing checksum-dependent observation before end clock: '+name)
        if any(re.search(r'\b(?:LDL|STL)\b',s) for _,s in instructions):
            raise ValueError('local/spill instruction in RF probe: '+name)
        ffmas=[s for s in producer['instructions'] if re.search(r'\bFFMA\b',s)]
        pairs=[];destinations=[]
        for ins in ffmas:
            operands=ins.split('FFMA',1)[1].strip().split(',')
            destinations.append(operands[0].strip())
            pairs.append(tuple(x.strip().replace('.reuse','') for x in operands[1:3]))
        row=dict(kernel=f'{family}_{source}_{pool}_{split}_{position}',
                 name=name,producer_loop_start=producer['start'],producer_loop_end=producer['end'],
                 loop_counts=producer['counts'],clock_addresses=[a for a,_ in clocks],
                 checksum_observation=preceding[-10:],register_source_pairs=len(set(pairs)),
                 ffma_destination_registers=len(set(destinations)),
                 reuse_annotation_count=sum(s.count('.reuse') for s in producer['instructions']))
        if family==0 and source==2 and len(set(pairs))!=8:
            raise ValueError('eight-pair source coordinate changed in SASS: '+name)
        if family==0 and source in (0,1) and len(set(pairs))!=1:
            raise ValueError('single-pair source coordinate changed in SASS: '+name)
        result.append(row)
    if len(result)!=28:raise ValueError(f'expected 28 specializations, found {len(result)}')
    return result


def cv(values):
    return statistics.pstdev(values)/statistics.mean(values) if len(values)>1 else 0.0


def workload(record):
    threads,steps,family=record['threads'],record['steps'],record['family']
    if record['preloaded']:
        return dict(flop=0,integer_op=0,logical_load_bytes=0,
                    note='preloaded consumer control; no production in measured window')
    if family==0:return dict(flop=threads*steps*2,integer_op=0,logical_load_bytes=0)
    if family==1:return dict(flop=threads*steps,integer_op=0,logical_load_bytes=0)
    if family==2:return dict(flop=0,integer_op=threads*steps,logical_load_bytes=0)
    return dict(flop=32*steps,integer_op=32*steps*2 if family==3 else 0,
                logical_load_bytes=32*steps*16 if family==4 else 0)


def analyze(root,check_only=False,sass_only=False,output=None):
    output=output or root/'analysis'
    output.mkdir(parents=True,exist_ok=True)
    binary=json.loads((root/'build/binary.json').read_text())
    if hashlib.sha256((root/'build/sass.txt').read_bytes()).hexdigest()!=binary['sass_sha256']:
        raise ValueError('SASS identity changed')
    sass=sass_checks(root/'build/sass.txt')
    if sass_only:
        output.mkdir(exist_ok=True)
        (output/'sass-check.json').write_text(json.dumps(sass,indent=2)+'\n')
        print('R02 SASS checked:',len(sass),'specializations');return
    records=[];checked=0
    for p in sorted((root/'samples').glob('*/*/result.json')):
        record=json.loads(p.read_text());checked+=replay(p.parent,record);records.append(record)
    output.mkdir(exist_ok=True)
    (output/'checks.json').write_text(json.dumps(dict(processes=len(records),checked_u64_words=checked,
                   sass_specializations=sass),indent=2)+'\n')
    if check_only:
        print('independent R02 replay:',len(records),'processes,',checked,'words,',len(sass),'SASS specializations')
        return
    cases=json.loads((root/'cases.json').read_text());rows=[];rules=[]
    for case in cases:
        mine=[r for r in records if r['case_id']==case['id']]
        formal=[r for r in mine if r['label'].startswith('formal-')]
        if not formal:raise ValueError('missing formal samples: '+case['id'])
        def paired_tail_delta(r):
            control=next(c for c in mine if c['label']==r['label'].replace('formal-','preload-'))
            a=[s[2]-s[1] for s in r['stamps']]
            b=[s[2]-s[1] for s in control['stamps']]
            return statistics.median(x-y for x,y in zip(a,b))
        perturbations=[]
        for r in formal:
            final=next((c for c in mine if c['label']==r['label'].replace('formal-','final-')),None)
            if final:perturbations.append(r['consumed_cycles']/final['consumed_cycles']-1)
        progress=[r['progress_cycles'] for r in formal]
        consumed=[r['consumed_cycles'] for r in formal]
        perturbation=statistics.median(perturbations) if perturbations else None
        final=[r for r in mine if r['label'].startswith('final-')]
        if len(final)!=len(formal):raise ValueError('missing matched final-only samples')
        if any(r['kernel']!=formal[0]['kernel'] for r in final):
            raise ValueError('plain and trace do not use the same machine-code kernel')
        plain_consumed=[r['consumed_cycles'] for r in final]
        qualified=all(r['warmup_converged'] for r in formal) and max(cv(progress),cv(consumed))<=.05
        plain_qualified=all(r['warmup_converged'] for r in final) and cv(plain_consumed)<=.05
        progress_qualified=qualified and not formal[0]['preloaded'] and (perturbation is not None and abs(perturbation)<=.05)
        work=workload(formal[0]);time=statistics.median(progress)
        tails=[paired_tail_delta(r) for r in formal] if perturbations and not formal[0]['preloaded'] else []
        row=dict(case=case['id'],processes=len(formal),registers=formal[0]['registers'],
                 progress_cycles=time,consumed_cycles=statistics.median(plain_consumed),
                 traced_consumed_cycles=statistics.median(consumed),plain_cv=cv(plain_consumed),
                 plain_registers=final[0]['registers'],
                 progress_cv=cv(progress),consumed_cv=cv(consumed),trace_perturbation=perturbation,
                 matched_consumer_tail_delta=statistics.median(tails) if tails and progress_qualified else None,
                 progress_qualified=progress_qualified,consumer_qualified=plain_qualified,
                 **{k:v for k,v in work.items() if k!='note'})
        rows.append(row)
        rules.append(dict(condition=case,measurement=row,scope='same_SM_one_CTA_envelope_SM_cycles',
                          work=work,sass_kernel=formal[0]['kernel'],
                          evidence=[str(Path('samples')/case['id']/r['label']) for r in formal],
                          limitations=['not physical RF bandwidth or port capacity',
                                       'consumer covers retained destinations, not overwritten results',
                                       'pool size also changes dependency distance and compiler allocation']))
    with (output/'cases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(resource_conditions=len(rows),processes=len(records),checked_u64_words=checked,
                 qualified_progress_conditions=sum(r['progress_qualified'] for r in rows),
                 qualified_consumer_conditions=sum(r['consumer_qualified'] for r in rows),
                 rules=rules,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'rules.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# R02 原始条件服务','',f'{len(rows)} 条件，{len(records)} 进程，{checked} 个 uint64 输出独立核对。',
            '', '| 条件 | reg/thread | 推进 cycle | 消费就绪 cycle | 打点扰动 | 消费尾差 cycle | 资格 |',
            '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        pert='—' if r['trace_perturbation'] is None else f"{r['trace_perturbation']:.2%}"
        tail='—' if r['matched_consumer_tail_delta'] is None else f"{r['matched_consumer_tail_delta']:.1f}"
        report.append(f"| {r['case']} | {r['registers']} | {r['progress_cycles']:.1f} | {r['consumed_cycles']:.1f} | {pert} | {tail} | {'推进/消费' if r['progress_qualified'] else '消费' if r['consumer_qualified'] else '不合格'} |")
    (output/'report.md').write_text('\n'.join(report)+'\n')
    print('R02 conditions',len(rows),'qualified progress',summary['qualified_progress_conditions'],
          'qualified consumer',summary['qualified_consumer_conditions'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--sass-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();analyze(args.input.resolve(),args.check_only,args.sass_only,args.output)

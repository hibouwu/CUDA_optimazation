#!/usr/bin/env python3
"""R16 residency independent packed input, output and per-SM timeline replay."""
import argparse,csv,gzip,hashlib,json,re,statistics,struct
from functools import lru_cache
from pathlib import Path

@lru_cache(None)
def expected_dot(block,row,col,tiles,large):
    total=0
    for tile in range(tiles):
        aa=(row+block+tile)%7 if large else row%7
        bb=(col+2*block+tile)%11 if large else col%11
        total+=sum((1+(aa+2*k)%7)*(1+(bb+3*k)%11) for k in range(64))
    return total/512


def replay(path,r):
    def raw(name,kind):
        with gzip.open(path/(name+'.gz'),'rb') as stream:data=stream.read()
        meta=r['files'][name]
        if len(data)!=meta['bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256']:raise ValueError('witness hash/size changed')
        return struct.unpack('<'+kind*(len(data)//struct.calcsize(kind)),data)
    out=raw('accum.f32','f');last=raw('last.u16','H')
    for block in range(r['blocks']):
        for t in range(128):
            for j in range(32):
                row=(t//32)*16+(t%32)//4+((j//2)%2)*8;col=(t%4)*2+j%2+(j//4)*8
                want=(t+1)/1024+j/32+expected_dot(block%77,row%7,col%11,r['tiles'],r['large'])
                if out[(block*128+t)*32+j]!=want:raise ValueError(f'wrong TC result: {path} block{block} lane{t} slot{j}')
        for q in range(8192):
            b=q>=4096;x=(q%4096)//64;k=q%64;tile=r['tiles']-1 if r['large'] else 0
            bb=block if r['large'] else 0
            val=(1+(x+3*k+2*bb+tile)%11)/32 if b else (1+(x+2*k+bb+tile)%7)/16
            bits=struct.unpack('<H',struct.pack('<e',val))[0];index=(q%4096)^(((q%4096)>>3)&0x38)
            offset=block*8192+(4096 if b else 0)+index
            if last[offset]!=bits:raise ValueError('wrong final tile/generation: '+str(path))
    starts=[];ends=[];per_sm={}
    for s in r['stamps']:
        if s[0]==0 or s[1]<s[0] or s[4]!=s[5]:raise ValueError('invalid SM-local interval')
        starts.append(s[2]);ends.append(s[3]);per_sm.setdefault(s[4],[]).append((s[2],s[3]))
    if max(ends)-min(starts)!=r['elapsed_ns']:raise ValueError('wrong whole-GPU envelope')
    overlap=[]
    for sm,windows in per_sm.items():
        events=sorted([(a,1) for a,b in windows]+[(b,-1) for a,b in windows],key=lambda x:(x[0],x[1]))
        active=0;peak=0
        for _,delta in events:active+=delta;peak=max(peak,active)
        overlap.append(peak)
    return dict(words=len(out)+len(last),observed_sms=len(per_sm),window_overlap_max=max(overlap),
                note='execution-window overlap is not full CTA residency')


def sass(root):
    text=(root/'build/sass.txt').read_text();facts=[]
    for body in text.split('Function : ')[1:]:
        name=body.splitlines()[0].strip();m=re.search(r'resident16ILi([124])E',name)
        if not m:continue
        if re.search(r'\b(?:LDL|STL)\b',body):raise ValueError('spill/local access in residency probe')
        h=len(re.findall(r'\bHGMMA\b',body));copies=len(re.findall(r'\bUBLKCP\b',body))
        if h!=4:raise ValueError('expected four MMA per K tile')
        if copies!=1:raise ValueError('expected one 16KiB bulk request per input tile')
        facts.append(dict(stage=int(m[1]),kernel=name,hgmma=h,bulk_copies=copies,
                          waits=[s.strip() for s in body.splitlines() if 'WARPGROUP.DEPBAR' in s]))
    if len(facts)!=3:raise ValueError('expected three stage kernels')
    return facts


def analyze(root,output=None,sass_only=False,check_only=False):
    cases_path=root/'cases.json'
    if (root/'identity.json').exists() or (cases_path.exists() and json.loads(cases_path.read_text())[0]['subset']=='quota'):
        from analyze_r16_quota import analyze as analyze_quota
        return analyze_quota(root,output,sass_only,check_only)
    dest=output or root/'analysis';dest.mkdir(parents=True,exist_ok=True)
    facts=sass(root);(dest/'sass-check.json').write_text(json.dumps(facts,indent=2)+'\n')
    if sass_only:print('R16 residency SASS checked');return
    records=[];count=0
    for p in sorted((root/'samples').glob('*/*/result.json')):
        r=json.loads(p.read_text());r['replay']=replay(p.parent,r);records.append(r);count+=r['replay']['words']
    (dest/'checks.json').write_text(json.dumps(dict(processes=len(records),checked_values=count),indent=2)+'\n')
    if check_only:print('R16 exact replay',len(records),count);return
    cases=json.loads((root/'cases.json').read_text());rows=[]
    for c in cases:
        rr=[r for r in records if r['case_id']==c['id'] and r['label'].startswith('formal-')]
        if not rr:raise ValueError('missing formal case')
        values=[r['elapsed_ns'] for r in rr];cv=statistics.pstdev(values)/statistics.mean(values)
        flop=2*64*64*64*32*rr[0]['blocks'];bytes_=16384*32*rr[0]['blocks']
        rows.append(dict(case=c['id'],stage=c['stage'],large=int(c['source']=='independent_large'),
            resident=c['target_occupancy'],ns=statistics.median(values),cv=cv,processes=len(rr),
            qualified=cv<=.05 and all(r['warmup_converged'] for r in rr),
            producer_flop=flop,logical_input_bytes=bytes_,source_bytes=rr[0]['source_bytes'],
            registers=rr[0]['registers'],occupancy_limit=rr[0]['occupancy_limit'],
            observed_window_overlap=max(r['replay']['window_overlap_max'] for r in rr)))
    with (dest/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(subset='residency',conditions=rows,processes=len(records),checked_values=count,
                unit='globaltimer_ns_whole_GPU',source_core='64x64x64_F16_F32',
                scope='2*SMs_CTA_grid; API capacity and window overlap distinguished')
    (dest/'rules.json').write_text(json.dumps(result,indent=2)+'\n');print('R16 residency conditions',len(rows),'qualified',sum(r['qualified'] for r in rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--sass-only',action='store_true');p.add_argument('--check-only',action='store_true')
    a=p.parse_args();analyze(a.input.resolve(),a.output,a.sass_only,a.check_only)

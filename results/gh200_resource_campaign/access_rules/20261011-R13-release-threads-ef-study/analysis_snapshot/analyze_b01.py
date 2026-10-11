#!/usr/bin/env python3
"""Replay the four missing B01 SMEM and large-working-set TMA coordinates."""
import argparse,gzip,hashlib,json,re,statistics,struct
from pathlib import Path


def replay(folder,r):
    checked=0
    for name,meta in r['files'].items():
        digest=hashlib.sha256();size=0
        with gzip.open(folder/(name+'.gz'),'rb') as stream:
            if name=='transport.u16':
                slots=1 if r['input'] else r['slots']
                offset=73*(r['slots']-1) if r['input'] else r['generation']
                expected=struct.pack('<8192H',*[((q*17+19+offset)%65536) for q in range(8192)])
                for _ in range(slots):
                    data=stream.read(16384);digest.update(data);size+=len(data)
                    if data!=expected:raise ValueError('wrong TMA transport/generation')
                checked+=slots*8192
            else:
                data=stream.read();digest.update(data);size=len(data);values=struct.unpack('<'+'I'*(size//4),data)
                if name=='stored.u32':expected=[((r['iterations']-1)*31+i+7)%2**32 for i in range(r['warps']*1024)]
                else:
                    expected=[]
                    for t in range(r['warps']*32):
                        warp,lane=divmod(t,32)
                        subtotal=sum(17*((warp*8+slot)*128+lane*4+j)+19 for slot in range(8) for j in range(4))
                        expected.append((subtotal*r['iterations'])%2**32)
                if tuple(expected)!=values:raise ValueError('wrong independent SMEM values/sink')
                checked+=len(values)
            if stream.read(1):raise ValueError('extra witness bytes')
        if size!=meta['bytes'] or digest.hexdigest()!=meta['sha256']:raise ValueError('witness hash/size changed')
    if r['first_sm']!=r['last_sm'] or r['local_bytes'] or r['cycles']<=0:raise ValueError('invalid scope/resource')
    warm=r['warmup'];converged=8<=len(warm)<=30 and statistics.pstdev(warm[-5:])/statistics.mean(warm[-5:])<=.02
    if converged!=r['warmup_converged']:raise ValueError('warmup flag inconsistent')
    return checked


def sass(root):
    facts={}
    for name in ['smem','tma']:
        text=(root/'build'/f'{name}.sass').read_text();kernels=[]
        for body in text.split('Function : ')[1:]:
            symbol=body.splitlines()[0]
            if not ('joint_smem' in symbol or 'tensor_large' in symbol):continue
            if re.search(r'\b(?:LDL|STL)\b',body):raise ValueError('spilled B01')
            kernels.append(dict(kernel=symbol,lds128=len(re.findall(r'\bLDS\.128',body)),sts128=len(re.findall(r'\bSTS\.128',body)),
                                tensor_copies=len(re.findall(r'\bUTMA(?:LDG|STG)\.2D',body)),clocks=len(re.findall('SR_CLOCKLO',body))))
        if not kernels:raise ValueError('missing B01 kernels')
        if name=='smem' and (kernels[0]['lds128']!=8 or kernels[0]['sts128']!=8):raise ValueError('SMEM vector loop changed')
        if name=='tma' and (len(kernels)!=2 or any(k['tensor_copies']!=1 for k in kernels)):raise ValueError('tensor copy count changed')
        facts[name]=kernels
    return facts


def analyze(root,output=None):
    for manifest in ['source_hashes.json','build/hashes.json']:
        for name,h in json.loads((root/manifest).read_text()).items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('artifact changed: '+name)
    facts=sass(root);rows=[];total=0;processes=0
    for case in json.loads((root/'cases.json').read_text()):
        records=[]
        for p in (root/'samples'/case['id']).glob('*/result.json'):
            r=json.loads(p.read_text());total+=replay(p.parent,r);processes+=1
            if p.parent.name.startswith('formal-'):records.append(r)
        if len(records)<3:raise ValueError('missing independent samples')
        r=records[0];cycles=[r['cycles'] for r in records]
        bytes_=r['warps']*32*r['iterations']*8*32 if r['kind']=='smem' else r['slots']*16384
        variation=statistics.pstdev(cycles)/statistics.mean(cycles)
        rows.append(dict(case=case['id'],processes=len(records),cycles=statistics.median(cycles),cv=variation,
         logical_bytes=bytes_,bytes_per_cycle=statistics.median(bytes_/x for x in cycles),
         different_addresses_bytes=r['warps']*8192 if r['kind']=='smem' else r['slots']*16384,
         qualified=variation<=.05 and (variation<=.01 or len(records)>=10) and all(r['warmup_converged'] for r in records),
         scope='one_CTA_clock64',completion='read_sink_and_CTA_join' if r['kind']=='smem' else 'mbarrier_or_wait_group0_then_CTA_join'))
    dest=output or root/'analysis';dest.mkdir(parents=True,exist_ok=True)
    (dest/'rules.json').write_text(json.dumps(dict(conditions=rows,processes=processes,checked_values=total,sass=facts),indent=2)+'\n')
    print('B01',len(rows),'conditions',total,'values')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();analyze(a.input.resolve(),a.output)

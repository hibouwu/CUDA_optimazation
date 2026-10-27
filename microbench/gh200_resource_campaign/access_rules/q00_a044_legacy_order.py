#!/usr/bin/env python3
"""以原宿主/设备时标或冻结采样循环恢复 legacy 的实际子轮次顺序。"""
import argparse, collections, copy, json, random, re
from pathlib import Path
from q00_a044_remeasure import read, save
from q00_a044_legacy_stdout import Text

CAM=Path('results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement')

def external_orders():
    groups=[]
    for filename in ('legacy-r04-order-r5.json','legacy-v01-order-r5.json','legacy-extra-order-r5.json'):
        if not (CAM/filename).exists():continue
        for key,value in read(CAM/filename).items():
            groups.append((key,value['records'],filename))
    for key,value in read(CAM/'legacy-r02-program-order-r5.json').items():
        records=value['program_order']
        extra=[r for r in value['same_SM_clock_order'] if (r.get('record') if isinstance(r,dict) else r) not in records]
        records += [(r['record'] if isinstance(r,dict) else r) for r in extra]
        groups.append((key,records,'legacy-r02-program-order-r5.json'))
    value=read(CAM/'legacy-gpu-and-r03-order-r5.json')
    for key,rows in value['globaltimer_sequences'].items():
        groups.append((key,[r['record'] for r in rows],'legacy-gpu-and-r03-order-r5.json:globaltimer_sequences'))
    for key,rows in value['r03_program_sequences'].items():
        groups.append((key,rows,'legacy-gpu-and-r03-order-r5.json:r03_program_sequences'))
    return groups


def r10(data):
    root=Path(data['run']);groups=[]
    for protocol in root.glob('protocol-*.json'):
        p=read(protocol);role=protocol.stem.removeprefix('protocol-');cases=p['cases'];rng=random.Random(p['seed']);records=[]
        for trial in range(p['processes_per_mode']):
            configs=sorted({c['config'] for c in cases});rng.shuffle(configs)
            for config in configs:
                ordered=[c for c in cases if c['config']==config];rng.shuffle(ordered)
                for case in ordered:
                    modes=['plain','trace'];rng.shuffle(modes)
                    for mode in modes:records.append(str(root/'samples'/role/case['id']/mode/f'{trial:02}'/'command.json'))
        groups.append((str(root)+'/'+role,records,str(root/'source/run_r10.py')+':173-195'))
    return groups


def r15(data):
    root=Path(data['run']);MODES=('reg_smem','tma_only','full_output','background','serial','concurrent')
    matrix=[dict(id=f'r15_{kib}k_b{buffers}_{MODES[mode]}',kib=kib,buffers=buffers,mode=mode) for kib in (64,128) for buffers in (1,2) for mode in (range(6) if buffers==1 else (2,4,5))]
    present={r['origin']['record'] for r in data['rows']};groups=[]
    for role in ('representative','formal'):
        if not any('/'+role+'-' in p for p in present):continue
        cases=[c for c in matrix if c['kib']==64 and c['buffers']==1 and c['mode'] in (2,5)] if role=='representative' else matrix
        rng=random.Random(20261007);records=[]
        def emit(case,trial,traced):
            name=f'{role}-{trial:02}-'+('trace' if traced else 'plain')
            records.append(str(root/'samples'/case['id']/name/'command.json'))
        for trial in range(3):
            blocks=[];pending=list(cases)
            while pending:
                c=pending.pop(0);companion=next((q for q in pending if q['kib']==c['kib'] and q['buffers']==c['buffers'] and {q['mode'],c['mode']}=={4,5}),None);block=[c]
                if companion is not None:pending.remove(companion);block.append(companion)
                rng.shuffle(block);blocks.append(block)
            rng.shuffle(blocks)
            for c in [c for block in blocks for c in block]:
                variants=[False,True];rng.shuffle(variants)
                for traced in variants:emit(c,trial,traced)
        if role=='representative':
            for trial in range(3,10):
                for c in cases:
                    variants=[False,True];rng.shuffle(variants)
                    for traced in variants:emit(c,trial,traced)
        else:
            for c in cases:
                for trial in range(3,10):
                    for traced in (False,True):emit(c,trial,traced)
        groups.append((str(root)+'/'+role,[p for p in records if p in present],str(root/'source/run_r15.py')+':143-208'))
    return groups


def m11(data,index):
    root=Path(data['run']);records=[]
    if index==45:
        package=root/'run-20261011T210745Z/continuation-20261012/package';output=package/'development'
        records.append(str(output/'device.process.json'))
        cases=[f'read_w4_stride{s}' for s in (1,2,4,8,16,32)]+['read_w4_broadcast','read_w8_stride1','read_w16_stride1','write_w4_stride1']
        for trial in range(10):
            for case in cases:
                if trial>=3 and case!='read_w4_broadcast':continue
                records.append(str(output/'compatibility'/(case+f'-{trial:02}')/'measure.process.json'))
        for op in (0,1):
            for width in (4,16):records.append(str(output/'compatibility'/f'r03-o{op}-b{width}'/'measure.process.json'))
        for trial in range(10):records.append(str(output/f'grain-{trial:02}.process.json'))
        cases=read(package/'development-cases.json')+read(package/'control-cases.json');seed=2026101130;attempt=False
        source=package/'source/run_r03_lds_sts.py'
    else:
        parent=root/'run-20261012T110219Z' if index==49 else root;output=parent/'measurements';package=parent/'package'
        records.append(str(output/'device.process.json'));attempt=True
        cases=read(package/('minimal-measurement-cases.json' if index==49 else 'freeze/holdout-cases.json'))
        seed={47:2026101320,48:2026101420,49:2026101130}[index]
        source=package/'execution-source'/('run_r03_lane_contrast.py' if index==49 else 'run_r03_holdout.py')
    rng=random.Random(seed)
    for trial in range(3):
        ordered=cases.copy();rng.shuffle(ordered)
        for case in ordered:
            folder=output/'samples'/case['id']/f'trial-{trial:02}'
            if attempt:
                for a in (0,1):
                    path=folder/f'attempt-{a:02}'/'measure.process.json'
                    if path.exists():records.append(str(path))
            else:records.append(str(folder/'measure.process.json'))
    return [(str(output),records,str(source))]


def restore(data,index,external):
    rows=data['rows']
    if data['ordering']=='recorded process timestamp':
        return [dict(subrun=data['run'],order_evidence=data['ordering'],rows=rows)], []
    lookup={r['origin']['record']:r for r in rows};assert len(lookup)==len(rows)
    groups=[]
    if not rows:return groups
    if data['ordering']=='recorded process timestamp':
        groups=[(data['run'],[r['origin']['record'] for r in rows],data['ordering'])]
    elif index in (8,10,13):
        stamps=[]
        for row in rows:
            value=json.loads(Text(Path(row['origin']['original_stdout'])).read_text());s=value['stamps'];stamps.append((min(r[2] for r in s),max(r[3] for r in s),row['origin']['record']))
        stamps.sort();assert all(a[1]<b[0] for a,b in zip(stamps,stamps[1:]))
        groups=[(data['run'],[x[2] for x in stamps],data['run']+'/source/probes/r13.cu:15,338-340 globaltimer cols2/3')]
    elif index in (20,21):groups=r10(data)
    elif index==33:groups=r15(data)
    elif index in (45,47,48,49):groups=m11(data,index)
    else:
        groups=[(name,[r for r in records if r in lookup],source) for name,records,source in external if any(r in lookup for r in records)]
    output=[];seen=set()
    for name,records,source in groups:
        actual=[]
        for record in records:
            if record not in lookup:continue
            assert record not in seen,record;seen.add(record);actual.append(lookup[record])
        if actual:output.append(dict(subrun=name,order_evidence=source,rows=actual))
    missing=set(lookup)-seen
    return output,sorted(missing)


def main(args):
    external=external_orders();unresolved=[];allgroups=[]
    for source in read(args.input/'sources.json'):
        index=int(source['id'].split('-')[1]);data=read(args.input/(source['id']+'.json'))
        if not data['rows']:continue
        groups,missing=restore(data,index,external)
        for group in groups:group.update(source_id=source['id'],run=data['run'],labels=data['labels'])
        allgroups.extend(groups)
        if missing:unresolved.append(dict(source_id=source['id'],run=data['run'],records=missing))
        print(source['id'],len(data['rows'])-len(missing),len(missing),flush=True)
    args.output.mkdir(exist_ok=False)
    save(args.output/'groups.json',allgroups);save(args.output/'unresolved.json',unresolved)
    save(args.output/'summary.json',dict(groups=len(allgroups),ordered_calls=sum(len(g['rows']) for g in allgroups),unresolved_calls=sum(len(g['records']) for g in unresolved)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

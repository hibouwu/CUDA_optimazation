#!/usr/bin/env python3
"""V07: V06 event recurrence plus matched-window R18 increments, all on the calibration card.

Boundary increments apply to corresponding physical tiles, separately for first/later
round and observed class. Two K points define an interpolation hypothesis; V07 alone
assesses its transfer. No average cycle/Ktile is added as a second absolute slope.
"""
import gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path
import v06_model as base
import v06_run as common
from analyze_r18 import replay,tile_class


def convert(root,kind,destination,qualified_only=False):
    """Derived V06-format views, after checking original six-field traces and witnesses."""
    destination.mkdir(parents=True,exist_ok=True);(destination/'samples').mkdir(exist_ok=True)
    rows=[r for r in json.loads((root/'cases.json').read_text()) if r['set']==kind]
    setup_rows=json.loads((root/'static_setup.json').read_text())
    common.write_json(destination/'static_setup.json',[dict(case_id=r['case'],**r['setup']) for r in setup_rows])
    evidence=[];records=[];eligibility=[]
    for row in rows:
        observations=[]
        for path in sorted((root/'samples'/row['id']).glob('*.json')):
            r=json.loads(path.read_text())
            if r['returncode'] and 'last five warmups CV exceeds 2%' in r['stderr']:continue
            if r['returncode']:raise ValueError('unresolved calibration process failure')
            observed=replay(root,r,row)
            observations.append((r,observed))
        plain=[v['elapsed_us'] for r,v in observations if r['variant']=='plain']
        stamped=[v['elapsed_us'] for r,v in observations if r['variant']=='stamped']
        if len(plain)!=10 or len(stamped)!=10:raise ValueError('V07 requires ten processes per variant')
        perturbation=statistics.median(stamped)/statistics.median(plain)-1
        pcv=statistics.pstdev(plain)/statistics.mean(plain);tcv=statistics.pstdev(stamped)/statistics.mean(stamped)
        eligible=abs(perturbation)<=.05 and pcv<=.05 and tcv<=.05
        eligibility.append(dict(case=row['id'],qualified=eligible,perturbation=perturbation,plain_cv=pcv,trace_cv=tcv))
        for r,observed in observations:
            if observed['ctas']:
                setup=next(x['setup'] for x in setup_rows if x['case']==row['id'])
                predicted=base.scheduled_work(row['config'],row['m'],row['n'],setup['grid'])
                if [c['work'] for c in observed['ctas']]!=predicted:raise ValueError('scheduler model differs from recorded coordinates')
            with gzip.open(root/r['raw'],'rt') as stream:events=[json.loads(l) for l in stream if l.strip()]
            for ev in events:
                if ev['event']=='call' and ev['trace']:
                    source=ev['trace'];out=[]
                    for offset in range(0,len(source),784):
                        head=source[offset:offset+16];out+=head
                        for t in range(128):out+=source[offset+16+6*t:offset+16+6*t+4]
                    ev['trace']=out
                elif ev['event']=='setup':ev.update(trace_words=528,trace_version='derived-v06-view-of-validated-r18')
            raw=Path('derived')/row['id']/Path(r['raw']).name;(destination/raw).parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(destination/raw,'wt') as stream:
                for ev in events:stream.write(json.dumps(ev)+'\n')
            evidence.append(dict(derived=str(raw),derived_sha256=common.sha(destination/raw),original_raw=r['raw'],original_sha256=r['raw_sha256']))
            if eligible or not qualified_only:
                records.append(dict(case=row['id'],config=row['config'],variant=r['variant'],trial=r['trial'],returncode=r['returncode'],
                 check='ok',padding_errors=0,max_err=0,check_samples=4096,elapsed_us=r['elapsed_us'],raw=str(raw),
                 start_ns=r['host_start_ns'],end_ns=r['host_stop_ns']))
    (destination/'samples'/f'{kind}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    common.write_json(destination/f'{kind}-eligibility.json',eligibility)
    common.write_json(destination/f'{kind}-derivation.json',dict(converter_sha256=common.sha(Path(__file__)),files=evidence))
    return eligibility


def boundary_models(r18_analysis):
    """Qualified paired differences only; keep cells with insufficient K support explicit."""
    rows=json.loads((r18_analysis/'rules.json').read_text())['conditions'];lookup={r['id']:r for r in rows}
    groups=defaultdict(list)
    for p in json.loads((r18_analysis/'paired-tile-increments.json').read_text()):
        row=lookup[p['case']]
        if row['kind']=='explicit_zero':continue
        key='|'.join(map(str,[row['config'],row['kind'],p['round'],p['cls']]))
        groups[key].append(p)
    result={}
    for key,points in groups.items():
        good=sorted([p for p in points if p['qualified']],key=lambda p:p['k'])
        if len({p['k'] for p in good})<2:
            result[key]=dict(status='insufficient_qualified_K_support',cases=[p['case'] for p in points]);continue
        lo,hi=good[0],good[-1];x0,x1=lo['k']/64,hi['k']/64
        b=(hi['delta_cycles']-lo['delta_cycles'])/(x1-x0);a=lo['delta_cycles']-b*x0
        result[key]=dict(status='frozen_interpolation_hypothesis_not_hardware_constant',a=a,b=b,kt_min=x0,kt_max=x1,
                         cases=[lo['case'],hi['case']],source_window_cycles=[lo['delta_cycles'],hi['delta_cycles']])
    return result


def predict(row,grid,params,clock,fixed,boundary):
    cfg=row['config'];schedule=base.CONFIGS[cfg]['schedule'];work=base.scheduled_work(cfg,row['m'],row['n'],grid)
    kt=base.cdiv(row['k'],64);affected=defaultdict(set);unknown=[]
    kind=row['kind']
    active_kind=kind in ['oob','partial_even','partial_odd'] and cfg in ['cfg_a','cfg_c']
    if active_kind:
        for tiles in work:
            for j,(mi,ni) in enumerate(tiles):
                if tile_class(row,mi,ni)!='other':affected[j].add(ni if cfg=='cfg_a' else mi)
    cycles=[];details=[]
    for cta,tiles in enumerate(work):
        additions=[]
        for j,(mi,ni) in enumerate(tiles):
            extra=0
            if active_kind:
                cls=tile_class(row,mi,ni);transverse=ni if cfg=='cfg_a' else mi
                if cls=='other' and transverse in affected[j]:cls='same_column' if cfg=='cfg_a' else 'same_row'
                key='|'.join(map(str,[cfg,kind,j,cls]));rule=boundary.get(key,{})
                if 'a' in rule and rule['kt_min']<=kt<=rule['kt_max']:extra=rule['a']+rule['b']*kt
                else:unknown.append(key)
            additions.append(extra)
        c,stages=base.cta_cycles(params,schedule,len(tiles),kt,detail=True)
        # Cooperative tile chain is sequential. A matched mainloop-window increment
        # propagates once through the chain; do not add an absolute slope/intercept.
        if additions and stages is not None and schedule=='cooperative':
            total=sum(additions);excess_change=params['xk']*total
            c+=total+excess_change;stages=dict(stages)
            stages['mainloop']+=total;stages['max_cta_excess']+=excess_change
        cycles.append(c);details.append(stages)
    critical=max(range(len(cycles)),key=cycles.__getitem__);cmax=cycles[critical];stages=details[critical]
    phi=sum(cycles)/(base.SMS*cmax);mu=stages['mainloop']/cmax
    dbytes=base.dram_bytes(cfg,row['m'],row['n'],row['k'],work);window=cmax/1600
    for _ in range(200):
        f=base.clock_ghz(clock,phi,math.log(window),dbytes/(window*1e-6)/1e12,mu)
        if f<=0:raise ValueError('clock model outside positive domain')
        window=.5*window+.5*cmax/(f*1000)
    return dict(row=row,grid=grid,critical_cta=critical,critical_cycles=cmax,stage_cycles=stages,
                clock_ghz=f,predicted_us=fixed+window,window_us=window,fixed_us=fixed,phi=phi,mu=mu,
                boundary_unknown_cells=sorted(set(unknown)),work=work,per_cta_cycles=cycles,
                scope='swizzle1, FP16/F32; boundary interpolation hypothesis, measured-card clock calibration')

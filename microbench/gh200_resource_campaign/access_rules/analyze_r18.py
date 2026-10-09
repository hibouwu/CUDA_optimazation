#!/usr/bin/env python3
"""Independent numeric replay and actual CTA tile/event analysis for R18/R19."""
import argparse,gzip,json,math,statistics
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import v06_model as model
import v06_run as common


@lru_cache(None)
def reference(row,col,k,seed=17):
    total=sum(((row*7+t*13+seed*3)%17-8)*((t*5+col*11+seed*5)%17-8) for t in range(17))
    remainder=sum(((row*7+t*13+seed*3)%17-8)*((t*5+col*11+seed*5)%17-8) for t in range(k%17))
    return (total*(k//17)+remainder)/1024


def random_references(indices,m,n,k,seed):
    """Regenerate logical FP16 inputs independently of the device-side output checker."""
    import numpy as np

    def values(rows,cols,is_a):
        x=np.uint32(seed)^np.uint32(0xa511e9b3 if is_a else 0x63d83595)
        x=x^(rows*np.uint32(0x9e3779b9))^(cols*np.uint32(0x85ebca6b))
        x=(x^(x>>16))*np.uint32(0x7feb352d)
        x=(x^(x>>15))*np.uint32(0x846ca68b)
        x=x^(x>>16)
        return ((x>>8).astype(np.float32)*np.float32(2**-23)-np.float32(1)).astype(np.float16).astype(np.float64)

    positions=np.asarray(indices,dtype=np.int64)
    rows=(positions//n).astype(np.uint32);cols=(positions%n).astype(np.uint32)
    inner=np.arange(k,dtype=np.uint32)[None,:]
    expected=[];tolerances=[]
    for start in range(0,len(indices),32):
        a=values(rows[start:start+32,None],inner,True)
        b=values(inner,cols[start:start+32,None],False)
        products=a*b
        expected.extend(products.sum(axis=1).tolist())
        tolerances.extend((2**-20+2**-21*np.abs(products).sum(axis=1)).tolist())
    return expected,tolerances


def replay(root,record,row,retain_roles=False):
    raw=root/record['raw']
    if common.sha(raw)!=record['raw_sha256']:raise ValueError('raw process identity changed')
    with gzip.open(raw,'rt') as stream:events={ev['event']:ev for line in stream if line.strip() for ev in [json.loads(line)]}
    setup,call,check=(events[k] for k in ['setup','call','check'])
    for key in ['m','n','k','lda','ldb','ldd','zero_m','zero_n','storage_m','storage_n','swizzle']:
        if setup[key]!=row[key]:raise ValueError('setup coordinate changed: '+key)
    mode=row.get('input_mode','dyadic');seed=row.get('seed',17)
    if setup.get('input_mode','dyadic')!=mode or setup.get('seed',17)!=seed:
        raise ValueError('input mode or seed changed')
    if setup.get('requested_sm_count',0)!=row.get('sm_count',0):raise ValueError('requested SM count changed')
    for key in ['alloc_lda','alloc_ldb','input_map_m','input_map_n']:
        if key in row and setup.get(key)!=row[key]:raise ValueError('input allocation changed: '+key)
    if 'input_map_m' in row or 'input_map_n' in row:
        if (setup['initialized_input_rows'],setup['initialized_input_columns'])!=(row['storage_m'],row['storage_n']):
            raise ValueError('matched input initialization extent changed')
        expected_zero=(max(0,row['storage_m']-row['m'])*row['k'] if row['zero_m']==row['m'] else 0)
        expected_zero+=(row['k']*max(0,row['storage_n']-row['n']) if row['zero_n']==row['n'] else 0)
        witness=events['input_zero_storage']
        if witness['checked']!=expected_zero or witness['nonzero']:
            raise ValueError('explicit zero storage witness failed')
    if check['status']!='ok' or check['padding_errors'] or len(check['checked_values'])!=4096:
        raise ValueError('incorrect or incomplete GEMM check')
    if len(set(check['checked_indices']))!=4096:raise ValueError('duplicate sampled values')
    if mode=='random':
        if setup['check_atol']!=2**-20 or setup['check_sum_abs_rtol']!=2**-21:
            raise ValueError('random input tolerance changed')
        expected_values,tolerances=random_references(check['checked_indices'],row['m'],row['n'],row['k'],seed)
    elif mode not in ['dyadic','zero']:
        raise ValueError('unknown input mode')
    zero_count=0
    for q,(index,value) in enumerate(zip(check['checked_indices'],check['checked_values'])):
        i,j=divmod(index,row['n'])
        if not 0<=i<row['m']:raise ValueError('invalid check position')
        zero=mode=='zero' or (row['zero_m']>=0 and i>=row['zero_m']) or (row['zero_n']>=0 and j>=row['zero_n'])
        zero_count+=zero
        expected=0 if zero else expected_values[q] if mode=='random' else reference(i%17,j%17,row['k'],seed)
        tolerance=(2**-20 if zero else tolerances[q]) if mode=='random' else 0
        if value is None or not math.isfinite(value) or abs(value-expected)>tolerance:
            raise ValueError('wrong GEMM output at '+str((i,j)))
    if row['kind']=='explicit_zero' and zero_count<1:raise ValueError('zero panel unchecked')
    warm=call['warmup_us'];warm_cv=statistics.pstdev(warm[-5:])/statistics.mean(warm[-5:])
    if not 8<=len(warm)<=30 or warm_cv>.02:raise ValueError('warmup did not converge')
    result=dict(case=row['id'],variant=record['variant'],trial=record['trial'],elapsed_us=call['elapsed_us'],checked_values=4096,zero_values=zero_count,ctas=[])
    if record['variant'] in ['plain','wide']:return result
    tile_words=setup.get('trace_tile_words',6)
    words=call['trace'];ctas=math.prod(setup['grid']);width=16+2*64*tile_words
    if len(words)!=ctas*width:raise ValueError('wrong trace size')
    dual=setup['trace_version']=='r18-dual-clock-events'
    if tile_words not in [6,10] or (dual and tile_words!=10):raise ValueError('wrong tile layout')
    schedule='pingpong' if row['config']=='cfg_b' else 'cooperative'
    seen=[]
    light=setup['trace_version']=='r18-light-events'
    inferred_work=model.scheduled_work(row['config'],row['m'],row['n'],setup['grid']) if light else None
    for c in range(ctas):
        head=words[c*width:(c+1)*width]
        if head[10] or not head[0] or not head[2]:raise ValueError('missing/overflow CTA trace')
        roles=[];coords=[]
        for role in range(2):
            end,ns,count=head[4+3*role:7+3*role]
            if count>64 or head[11+role]!=head[2]:raise ValueError('invalid count/SM-local clock boundary')
            tiles=[];tiles_ns=[];work=[]
            for j in range(count):
                x=16+(role*64+j)*tile_words;tile=head[x:x+4];mi,ni=head[x+4:x+6]
                if light:
                    if schedule!='cooperative' or count!=len(inferred_work[c]):raise ValueError('light work-count mismatch')
                    mi,ni=(q+1 for q in inferred_work[c][j])
                if not all(tile) or min(mi,ni)<=0 or not tile[0]<=tile[1]<=tile[2]<=tile[3]<=end:
                    raise ValueError('invalid tile events/coordinates')
                tiles.append(tile);work.append((mi-1,ni-1))
                if dual:
                    times=head[x+6:x+10]
                    if not head[1]<=head[13]<=times[0]<=times[1]<=times[2]<=times[3]<=ns:
                        raise ValueError('invalid direct globaltimer tile events')
                    tiles_ns.append(times)
            roles.append(dict(final_c=end,final_ns=ns,tiles=tiles,tiles_ns=tiles_ns));coords.append(work)
        if schedule=='cooperative':
            if coords[0]!=coords[1]:raise ValueError('consumer work mismatch')
            work=coords[0]
        else:
            work=[coords[j%2][j//2] for j in range(len(coords[0])+len(coords[1]))]
        entry=dict(entry_c=head[0],entry_ns=head[1],smid=head[2]-1,prod_c=head[3],roles=roles)
        tiles,end_c,end_ns=model.cta_timeline(entry,schedule)
        cta=dict(cta=c,entry_c=head[0],entry_ns=head[1],sm=head[2]-1,prod_c=head[3],
                 end_c=end_c,end_ns=end_ns,tiles=tiles,work=work)
        if dual:
            # Reuse the same role merge with ns-valued endpoints; no clock interpolation.
            ns_roles=[dict(tiles=r['tiles_ns'],final_c=r['final_ns'],final_ns=r['final_ns']) for r in roles]
            cta['tiles_ns']=model.cta_timeline(dict(roles=ns_roles),schedule)[0]
            cta['prod_ns']=head[13]
        if retain_roles:
            cta['roles']=roles
            cta['role_work']=coords
        result['ctas'].append(cta)
        seen+=work
    if len(seen)!=len(set(seen)):raise ValueError('duplicated logical work tiles')
    tm=256 if row['config'].startswith('cfg_c') else 128;tn=128
    required={(i,j) for i in range(model.cdiv(row['m'],tm)) for j in range(model.cdiv(row['n'],tn))}
    if not required<=set(seen):raise ValueError('actual scheduler omitted real tile')
    return result


def tile_class(row,mi,ni):
    axis=0 if row['config'].startswith('cfg_a') else 1
    q=(mi,ni)[axis];extent=row['m'] if axis==0 else row['n'];tile=128
    if q*tile>=extent:return 'oob'
    zero=row['zero_m'] if axis==0 else row['zero_n']
    if zero>=0 and q*tile>=zero:return 'explicit_zero'
    if not row['config'].endswith('1') and (q^1)*tile>=extent:return 'partner'
    if zero>=0 and (q^1)*tile>=zero:return 'zero_partner'
    if q==extent//tile and extent%tile:return 'partial'
    return 'other'


def analyze(root,output=None):
    common.verify(root)
    for name,h in json.loads((root/'build/sass_hashes.json').read_text()).items():
        if common.sha(root/'build'/name)!=h:raise ValueError('SASS changed')
    dest=output or root/'analysis';dest.mkdir(parents=True,exist_ok=True)
    rows=json.loads((root/'cases.json').read_text());results=[];classes=[];tails=[];paired=[];all_records={};increments=[];rejected=[]
    for row in rows:
        recs=[]
        for path in sorted((root/'samples'/row['id']).glob('*.json')):
            record=json.loads(path.read_text())
            if record['returncode']:
                if 'last five warmups CV exceeds 2%' in record['stderr']:
                    rejected.append(dict(path=str(path.relative_to(root)),reason='warmup_not_converged_no_output_check',record=record));continue
                raise ValueError('failed process remains unqualified')
            recs.append(replay(root,record,row))
        plain=[r['elapsed_us'] for r in recs if r['variant']=='plain'];trace=[r['elapsed_us'] for r in recs if r['variant']=='stamped']
        if not plain or len(plain)!=len(trace):raise ValueError('missing paired plain/trace')
        all_records[row['id']]=recs
        disturbance=statistics.median(trace)/statistics.median(plain)-1
        trace_cv=statistics.pstdev(trace)/statistics.mean(trace)
        cv=statistics.pstdev(plain)/statistics.mean(plain)
        qualified=len(plain)>=10 and cv<=.05
        results.append(dict(**row,plain_us=statistics.median(plain),plain_cv=cv,processes=len(plain),
                            perturbation=disturbance,trace_cv=trace_cv,plain_qualified=qualified,trace_qualified=qualified and abs(disturbance)<=.05 and trace_cv<=.05))
        per_class=defaultdict(list)
        for rec in recs:
            if rec['variant']!='stamped':continue
            active=[c for c in rec['ctas'] if c['tiles']]
            start=min(c['entry_ns'] for c in rec['ctas']);end=max(c['end_ns'] for c in active)
            critical=max(active,key=lambda c:c['end_ns'])
            durations=[c['end_c']-c['entry_c'] for c in active]
            tails.append(dict(case=row['id'],trial=rec['trial'],critical_cta=critical['cta'],critical_sm=critical['sm'],
                critical_work=critical['work'],sm_distribution=dict(Counter(c['sm'] for c in rec['ctas'])),critical_cycles=critical['end_c']-critical['entry_c'],median_cta_cycles=statistics.median(durations),
                max_cta_cycles=max(durations),critical_entry_skew_ns=critical['entry_ns']-start,
                entry_spread_ns=max(c['entry_ns'] for c in rec['ctas'])-start,envelope_ns=end-start,
                tiles_count_histogram=dict(Counter(len(c['tiles']) for c in rec['ctas'])),
                critical_mainloop_cycles=[t[1]-t[0] for t in critical['tiles']],
                critical_epilogue_cycles=[t[3]-t[2] for t in critical['tiles']],
                critical_permit_cycles=[t[2]-t[1] for t in critical['tiles']],
                critical_handoff_cycles=[b[0]-a[1] for a,b in zip(critical['tiles'],critical['tiles'][1:])]))
            groups=defaultdict(list)
            for c in active:
                for j,((mi,ni),t) in enumerate(zip(c['work'],c['tiles'])):
                    cls=tile_class(row,mi,ni) if row['config']!='cfg_b' else 'all'
                    groups[(j,cls)].append(t[1]-t[0])
            for key,vals in groups.items():per_class[key].append(statistics.median(vals))
        for (j,cls),vals in sorted(per_class.items()):
            classes.append(dict(case=row['id'],round=j,cls=cls,mainloop_cycles=statistics.median(vals),
                                stddev=statistics.pstdev(vals),processes=len(vals),qualified=qualified and abs(disturbance)<=.05 and trace_cv<=.05))
    for row in results:
        if row['kind'] not in ['oob','explicit_zero'] or row['config'].endswith('1'):continue
        base=next(x for x in results if x['group']==row['group'] and x['kind']=='ordinary' and x['config']==row['config'])
        paired.append(dict(case=row['id'],baseline=base['id'],delta_plain_us=row['plain_us']-base['plain_us'],
                           relative_plain=row['plain_us']/base['plain_us']-1,
                           qualified=row['plain_qualified'] and base['plain_qualified']))
    for row in results:
        if row['kind'] not in ['oob','explicit_zero','partial_even','partial_odd'] or row['config'].endswith('1'):continue
        base=next(x for x in results if x['group']==row['group'] and x['kind']=='ordinary' and x['config']==row['config'])
        controls={r['trial']:r for r in all_records[base['id']] if r['variant']=='stamped'}
        deltas=defaultdict(list)
        for rec in all_records[row['id']]:
            if rec['variant']!='stamped':continue
            control=controls[rec['trial']]
            if [c['work'] for c in rec['ctas']]!=[c['work'] for c in control['ctas']]:
                raise ValueError('boundary comparison changed physical tile list/CTA assignment')
            current=defaultdict(list)
            # Boundary columns/rows reached in each round; distinguish affected
            # neighbors from other CTAs, instead of pooling every valid tile.
            affected=defaultdict(set)
            for c in rec['ctas']:
                for j,(mi,ni) in enumerate(c['work']):
                    if tile_class(row,mi,ni)!='other':affected[j].add(ni if row['config']=='cfg_a' else mi)
            for c,b in zip(rec['ctas'],control['ctas']):
                for j,((mi,ni),t,bt) in enumerate(zip(c['work'],c['tiles'],b['tiles'])):
                    cls=tile_class(row,mi,ni)
                    transverse=ni if row['config']=='cfg_a' else mi
                    if cls=='other' and transverse in affected[j]:cls='same_column' if row['config']=='cfg_a' else 'same_row'
                    current[j,cls].append((t[1]-t[0])-(bt[1]-bt[0]))
            for key,values in current.items():deltas[key].append(statistics.median(values))
        for (j,cls),values in sorted(deltas.items()):
            increments.append(dict(case=row['id'],baseline=base['id'],round=j,cls=cls,k=row['k'],
             delta_cycles=statistics.median(values),stddev=statistics.pstdev(values),process_deltas=values,
             qualified=row['trace_qualified'] and base['trace_qualified'],
             scope='same physical coordinates/CTA assignment; per-process paired difference'))
    result=dict(conditions=results,rejected_processes=rejected,checked_values=sum(r['processes']*2*4096 for r in results),
                plain_qualified=sum(r['plain_qualified'] for r in results),trace_qualified=sum(r['trace_qualified'] for r in results),
                paired_complete_time=paired,analyzer_sha256=common.sha(Path(__file__)),scope='full traces record actual CUTLASS coordinates; light R18 traces infer unchanged static mapping checked against full cohort; traces failing 5% perturbation do not yield cycle parameters')
    for name,data in [('rules.json',result),('classes.json',classes),('tails.json',tails),('paired-tile-increments.json',increments)]:common.write_json(dest/name,data)
    print('conditions',len(results),'plain qualified',result['plain_qualified'],'trace qualified',result['trace_qualified'])

def analyze_v08_padding(root, output):
    """Separate V08F padding observations by round and total CTA work; no causal fit."""
    import v08_model
    med = statistics.median
    rows = json.loads((root / 'cases.json').read_text())
    cases = []
    for row in rows:
        if row['swizzle'] != 8 or not any(label in row['id'] for label in ('_padM_', '_padN_', '_padMN_')):
            continue
        pooled, groups, differences = defaultdict(list), defaultdict(list), defaultdict(list)
        for path in sorted((root / 'samples' / row['id']).glob('stamped-*.json')):
            record = json.loads(path.read_text())
            if record['returncode']:
                continue
            observed = replay(root, record, row)
            current = defaultdict(list)
            for cta in observed['ctas']:
                for j, (tile, coord) in enumerate(zip(cta['tiles'], cta['work'])):
                    cls = v08_model.tile_class(row['config'], row, *coord)
                    cycles = tile[1] - tile[0]
                    current[j, len(cta['tiles']), cls].append(cycles)
                    if j:
                        pooled[cls].append(cycles)
            for (j, count, cls), values in current.items():
                groups[j, count, cls].append(dict(trial=record['trial'], windows=len(values), cycles=med(values)))
                if cls != 'in' and (j, count, 'in') in current:
                    differences[j, count, cls].append(dict(trial=record['trial'],
                        cycles=med(values) - med(current[j, count, 'in'])))
        kt = v08_model.cdiv(row['k'], 64)
        strata = []
        for (j, count, cls), values in sorted(groups.items()):
            deltas = differences[j, count, cls]
            strata.append(dict(round=j, cta_tiles=count, cls=cls, processes=values,
                cycles_per_ktile=med(v['cycles'] for v in values) / kt,
                delta_processes=deltas,
                delta_per_ktile=med(v['cycles'] for v in deltas) / kt if deltas else None,
                delta_stddev_per_ktile=statistics.pstdev(v['cycles'] / kt for v in deltas) if deltas else None))
        timings = defaultdict(list)
        for path in (root / 'samples' / row['id']).glob('*.json'):
            record = json.loads(path.read_text())
            if not record['returncode']:
                timings[record['variant']].append(record['elapsed_us'])
        perturbation = med(timings['stamped']) / med(timings['plain']) - 1
        cases.append(dict(**row, kt=kt, perturbation=perturbation,
            trace_qualified=(min(len(timings[v]) for v in ('plain', 'stamped')) >= 10
                and abs(perturbation) <= .05
                and max(statistics.pstdev(timings[v]) / statistics.mean(timings[v]) for v in ('plain', 'stamped')) <= .05),
            pooled_later_tiles={cls: dict(windows=len(v), cycles=med(v),
                delta_per_ktile=(med(v) - med(pooled['in'])) / kt) for cls, v in sorted(pooled.items())},
            strata=strata))
    output.mkdir(parents=True, exist_ok=False)
    common.write_json(output / 'padding.json', dict(cases=cases, analyzer_sha256=common.sha(Path(__file__)),
        scope='Posthoc V08F. Groups share round and CTA work count, not coordinates or absolute event time. No physical service attribution or frozen prediction.'))
    print('padding cases', len(cases), '->', output)


def analyze_input_map_pairs(root, output):
    """Matched logical output/work lists; only the input descriptor bounds differ."""
    from analyze_r13_sm import select_attempts
    from v08_model import tile_class as logical_tile_class
    med = statistics.median
    common.verify(root)
    rows = json.loads((root / 'cases.json').read_text())
    setups = {s['case']: s['setup'] for s in json.loads((root / 'static_setup.json').read_text())}
    env = json.loads((root / 'environment.json').read_text())
    observations, summaries, all_hashes = {}, [], {}
    variants = ('plain', 'wide', 'stamped', 'dual')
    for row in rows:
        selected, failed, ignored = select_attempts(root, row['id'], variants)
        if ignored or any(sum(x['record']['variant'] == v for x in selected) != 10 for v in variants):
            raise ValueError('ten successful processes per variant required')
        if setups[row['id']]['gpu_uuid'] != env['gpu'].split(',')[0]:
            raise ValueError('input-map cases are not on the recorded GPU')
        by_variant = {v: {} for v in variants}
        for item in selected:
            rec = item['record']; obs = replay(root, rec, row)
            by_variant[rec['variant']][rec['trial']] = obs
            all_hashes[rec['raw']] = rec['raw_sha256']
        observations[row['id']] = by_variant
        times = {v: [by_variant[v][t]['elapsed_us'] for t in range(10)] for v in variants}
        summaries.append(dict(condition=row, median_us={v: med(t) for v,t in times.items()},
            cv={v: statistics.pstdev(t)/statistics.mean(t) for v,t in times.items()},
            wide_over_plain=med(times['wide'])/med(times['plain'])-1,
            dual_over_wide=med(times['dual'])/med(times['wide'])-1,
            dual_over_wide_pairs=[b/a-1 for a,b in zip(times['wide'],times['dual'])],
            failed_attempts=failed))
    pairs = []
    fixed = ('config','m','n','k','lda','ldb','ldd','storage_m','storage_n','alloc_lda','alloc_ldb',
             'zero_m','zero_n','swizzle','sm_count','input_mode','seed')
    for group in sorted({r['group'] for r in rows}):
        control = next(r for r in rows if r['group']==group and r['input_path']=='oob')
        mapped = next(r for r in rows if r['group']==group and r['input_path']=='address_zero')
        if any(control[k] != mapped[k] for k in fixed):
            raise ValueError('paired source-path conditions changed another variable')
        if setups[control['id']]['grid'] != setups[mapped['id']]['grid']:
            raise ValueError('input descriptor changed the output grid')
        a,b = observations[control['id']], observations[mapped['id']]
        time_pairs = {v: [b[v][t]['elapsed_us']/a[v][t]['elapsed_us']-1 for t in range(10)] for v in variants}
        strata = defaultdict(list)
        kt = (control['k']+63)//64
        for trial in range(10):
            ac,bc = a['dual'][trial]['ctas'], b['dual'][trial]['ctas']
            if [c['work'] for c in ac] != [c['work'] for c in bc]:
                raise ValueError('paired descriptor path changed CTA work lists')
            groups = defaultdict(list)
            for x,y in zip(ac,bc):
                for j,coord in enumerate(x['work']):
                    cls = logical_tile_class(control['config'],control,*coord)
                    xc,yc = x['tiles'][j],y['tiles'][j]
                    xn,yn = x['tiles_ns'][j],y['tiles_ns'][j]
                    groups[j,len(x['work']),cls].append(((xc[1]-xc[0])/kt,(yc[1]-yc[0])/kt,
                                                        (xn[1]-xn[0])/kt,(yn[1]-yn[0])/kt))
            for key,values in groups.items():
                means = [statistics.fmean(v[i] for v in values) for i in range(4)]
                strata[key].append(dict(trial=trial,windows=len(values),oob_cycle_per_kt=means[0],
                    address_zero_cycle_per_kt=means[1],oob_ns_per_kt=means[2],address_zero_ns_per_kt=means[3],
                    delta_cycle_per_kt=means[1]-means[0],delta_ns_per_kt=means[3]-means[2]))
        groups = []
        for (j,T,cls),values in sorted(strata.items()):
            groups.append(dict(j=j,T=T,logical_class=cls,processes=values,
                median_delta_cycle_per_kt=med(x['delta_cycle_per_kt'] for x in values),
                median_delta_ns_per_kt=med(x['delta_ns_per_kt'] for x in values)))
        pairs.append(dict(group=group,control=control['id'],address_zero=mapped['id'],
            complete_time_relative_pairs=time_pairs,
            complete_time_relative_median={v:med(x) for v,x in time_pairs.items()},groups=groups))
    output.mkdir(parents=True,exist_ok=False)
    common.write_json(output/'input-map-pairs.json',dict(environment=env,cases=summaries,pairs=pairs,
        raw_sha256=all_hashes,analyzer_sha256=common.sha(Path(__file__)),
        scope='Address-zero minus TMA OOB with the same logical output, storage, grid and CTA coordinates. '
              'Stage windows are dual-observer measurements; logical pad classes do not change when the input map expands. '
              'These are paired process comparisons, not physical traffic counts or a frozen prediction.'))
    print('input-map pairs',len(pairs),'successful processes',len(all_hashes),'->',output)


def boundary_geometry(row,coord):
    """Logical output footprint, independent of input descriptor padding."""
    tm,tn,_=model.CONFIGS[row['config']]['tile']
    mi,ni=coord
    mr=max(0,min(tm,row['m']-mi*tm));nr=max(0,min(tn,row['n']-ni*tn))
    return 'oob' if not mr or not nr else 'partial' if mr<tm or nr<tn else 'real'


def maximum_set(values):
    top=max(values)
    return [i for i,v in enumerate(values) if abs(v-top)<1e-7]


def critical_replacements(ctas,prediction):
    """Diagnostic counterfactuals, never model input or a revised candidate set."""
    entry0=min(c['entry_ns'] for c in ctas)
    entries=[c['entry_ns']-entry0 for c in ctas]
    durations=[c['end_ns']-c['entry_ns'] for c in ctas]
    cycles=[c['end_c']-c['entry_c'] for c in ctas]
    ends=[e+d for e,d in zip(entries,durations)]
    pred=prediction['cta_duration_ns'];frozen=prediction['critical_candidates']
    if len(pred)!=len(ctas):raise ValueError('prediction and observed CTA counts differ')
    with_entry=[e+d for e,d in zip(entries,pred)]
    last=maximum_set(ends);local=maximum_set(durations)
    representative=prediction['critical_cta']
    return dict(observed_last=last,observed_longest_ns=local,observed_longest_cycles=maximum_set(cycles),
        frozen_candidates=frozen,frozen_representative=representative,
        frozen_hit=bool(set(last)&set(frozen)),candidate_count=len(frozen),
        entry_only_candidates=maximum_set(with_entry),
        entry_only_hit=bool(set(last)&set(maximum_set(with_entry))),
        internal_only_candidates=local,internal_only_hit=bool(set(last)&set(local)),
        entry_spread_ns=max(entries),observed_envelope_ns=max(ends),
        frozen_envelope_ns=max(pred),entry_only_envelope_ns=max(with_entry),
        internal_only_envelope_ns=max(durations),
        representative_duration_deficit=1-durations[representative]/max(durations[i] for i in last),
        representative_finish_deficit_ns=max(ends)-ends[representative],
        best_frozen_finish_deficit_ns=max(ends)-max(ends[i] for i in frozen),
        ctas=[dict(cta=c['cta'],sm=c['sm'],entry_offset_ns=e,duration_ns=d,duration_cycles=cy,
                   end_offset_ns=en,prediction_ns=p,work=c['work'])
              for c,e,d,cy,en,p in zip(ctas,entries,durations,cycles,ends,pred)])


def boundary_stage_rows(row,observed,prediction=None,source=''):
    """Same-role paired endpoints; merged CTA residuals are labeled separately."""
    rows=[]
    for c in observed['ctas']:
        for role,(r,work) in enumerate(zip(c['roles'],c['role_work'])):
            for j,(cy,ns,coord) in enumerate(zip(r['tiles'],r['tiles_ns'],work)):
                for stage,i,k in [('L',0,1),('E',2,3)]:
                    rows.append(dict(case=row['id'],trial=observed['trial'],source=source,cta=c['cta'],role=role,
                        j=j,T=len(work),phase='first' if j==0 else 'later',geometry=boundary_geometry(row,coord),
                        mi=coord[0],ni=coord[1],stage=stage,cycles=cy[k]-cy[i],ns=ns[k]-ns[i],
                        prediction_ns=None,measurement='same_role'))
            if work:
                rows.append(dict(case=row['id'],trial=observed['trial'],source=source,cta=c['cta'],role=role,
                    j=len(work)-1,T=len(work),phase='tail',geometry=boundary_geometry(row,work[-1]),
                    mi=work[-1][0],ni=work[-1][1],stage='tail',cycles=r['final_c']-r['tiles'][-1][3],
                    ns=r['final_ns']-r['tiles_ns'][-1][3],prediction_ns=None,measurement='same_role'))
        if prediction is not None:
            pe=prediction['cta_events'][c['cta']]['events']
            for j,(ns,coord,p) in enumerate(zip(c['tiles_ns'],c['work'],pe)):
                for stage,i,k in [('L',0,1),('E',2,3)]:
                    rows.append(dict(case=row['id'],trial=observed['trial'],source=source,cta=c['cta'],role=-1,
                        j=j,T=len(c['work']),phase='first' if j==0 else 'later',geometry=boundary_geometry(row,coord),
                        mi=coord[0],ni=coord[1],stage=stage,cycles=None,ns=ns[k]-ns[i],
                        prediction_ns=p[k]-p[i],measurement='merged_cta'))
            if c['work']:
                coord=c['work'][-1]
                rows.append(dict(case=row['id'],trial=observed['trial'],source=source,cta=c['cta'],role=-1,
                    j=len(c['work'])-1,T=len(c['work']),phase='tail',geometry=boundary_geometry(row,coord),
                    mi=coord[0],ni=coord[1],stage='tail',cycles=None,
                    ns=c['end_ns']-max(t[3] for t in c['tiles_ns']),
                    prediction_ns=prediction['cta_events'][c['cta']]['tail'],measurement='merged_cta'))
    return rows


def stage_process_summary(rows):
    """Reduce correlated CTA windows inside each process before case statistics."""
    grouped=defaultdict(list)
    keys=('case','trial','measurement','role','phase','geometry','stage')
    for r in rows:grouped[tuple(r[k] for k in keys)].append(r)
    processes=[]
    for key,group in sorted(grouped.items()):
        value=dict(zip(keys,key),windows=len(group),ns=statistics.median(r['ns'] for r in group))
        cycles=[r['cycles'] for r in group if r['cycles'] is not None]
        if cycles:value['cycles']=statistics.median(cycles)
        predicted=[r for r in group if r['prediction_ns'] is not None]
        if predicted:
            value['prediction_ns']=statistics.median(r['prediction_ns'] for r in predicted)
            value['error_ns']=statistics.median(r['prediction_ns']-r['ns'] for r in predicted)
            rel=[r['prediction_ns']/r['ns']-1 for r in predicted if r['ns']>0]
            if rel:value['relative_error']=statistics.median(rel)
        processes.append(value)
    return processes


def old_boundary_pairs(base):
    """Reuse existing fixed-output input-map interventions, with role endpoints."""
    from analyze_r13_sm import select_attempts
    summaries=[];provenance={}
    fixed=('config','m','n','k','lda','ldb','ldd','storage_m','storage_n','alloc_lda','alloc_ldb',
           'zero_m','zero_n','swizzle','sm_count','input_mode','seed')
    for name in ('20261009-R18-first-fill-job738707','20261009-R18-first-fill-job738865'):
        root=base/name;rows=json.loads((root/'cases.json').read_text());cache={}
        for row in rows:
            selected,_,_=select_attempts(root,row['id'],('dual','plain'))
            calls={};times=defaultdict(list)
            for item in selected:
                rec=item['record'];obs=replay(root,rec,row,retain_roles=rec['variant']=='dual')
                times[rec['variant']].append(obs['elapsed_us'])
                provenance[str(root/rec['raw'])]=rec['raw_sha256']
                if rec['variant']=='dual':
                    calls[rec['trial']]=(obs,stage_process_summary(boundary_stage_rows(row,obs,source=str(root/rec['raw']))))
            cache[row['id']]=(calls,{v:statistics.median(t) for v,t in times.items()})
        for group in sorted({r['group'] for r in rows}):
            a=next(r for r in rows if r['group']==group and r['input_path']=='oob')
            b=next(r for r in rows if r['group']==group and r['input_path']=='address_zero')
            if any(a[k]!=b[k] for k in fixed):raise ValueError('input-map pair changed another factor')
            ca,ta=cache[a['id']];cb,tb=cache[b['id']];deltas=[]
            for trial in sorted(ca):
                oa,sa=ca[trial];ob,sb=cb[trial]
                if [c['work'] for c in oa['ctas']]!=[c['work'] for c in ob['ctas']]:raise ValueError('work changed')
                lookup={(r['role'],r['phase'],r['geometry'],r['stage']):r for r in sb}
                for r in sa:
                    key=r['role'],r['phase'],r['geometry'],r['stage'];s=lookup[key]
                    deltas.append(dict(trial=trial,role=key[0],phase=key[1],geometry=key[2],stage=key[3],
                        oob_ns=r['ns'],address_zero_ns=s['ns'],delta_ns=s['ns']-r['ns'],
                        oob_cycles=r['cycles'],address_zero_cycles=s['cycles'],delta_cycles=s['cycles']-r['cycles']))
            summaries.append(dict(root=str(root),group=group,oob=a,address_zero=b,
                plain_relative=tb['plain']/ta['plain']-1,dual_relative=tb['dual']/ta['dual']-1,
                oob_plain_us=ta['plain'],address_zero_plain_us=tb['plain'],processes=deltas))
    return dict(pairs=summaries,provenance=provenance,
        scope='Same output and allocation, changed input descriptor; independent processes grouped by trial, not one shared invocation. Positive delta means address-zero slower. No physical write-completion claim.')


def failure_diagnostics(base,output,calls_factory):
    """V09 post-hoc endpoint/geometry analysis. Does not modify predictions."""
    import csv
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    processes=[];stages=[];provenance={}
    selected=('v09d_cfg_a_P1_partial_first_and_later','v09d_cfg_c_P2_partial_later_sw8',
              'v09d_cfg_a_G3_A16','v09d_cfg_a_G3_base')
    for call in calls_factory(base):
        row=call['row'];obs=call['observed'];pred=call['prediction'];rec=call['record']
        if not pred.get('cta_duration_ns'):continue
        source=call.get('source',str(Path(base)/'20261009-V09-heldout-job739011'/rec['raw']))
        value=critical_replacements(obs['ctas'],pred)
        value.update(case=row['id'],trial=call['trial'],source=source,raw_sha256=rec['raw_sha256'])
        if row['id'] not in selected:value.pop('ctas')
        elif row['config']!='cfg_b':
            for c,detail,pp in zip(obs['ctas'],value['ctas'],pred['cta_events']):
                ns=c['tiles_ns'];pe=pp['events']
                actual=dict(prefix=ns[0][0]-c['entry_ns'],L=sum(t[1]-t[0] for t in ns),
                    E=sum(t[3]-t[2] for t in ns),tail=c['end_ns']-ns[-1][3])
                actual['gaps']=detail['duration_ns']-sum(actual.values())
                predicted=dict(prefix=pe[0][0],L=sum(t[1]-t[0] for t in pe),
                    E=sum(t[3]-t[2] for t in pe),tail=pp['tail'])
                predicted['gaps']=detail['prediction_ns']-sum(predicted.values())
                detail['observed_components_ns']=actual;detail['predicted_components_ns']=predicted
        processes.append(value);provenance[source]=rec['raw_sha256']
        if row['id'] in selected:stages.extend(boundary_stage_rows(row,obs,pred,source))
    stage_summary=stage_process_summary(stages)
    grouped=defaultdict(list)
    for p in processes:grouped[p['case']].append(p)
    cases=[]
    for case,group in sorted(grouped.items()):
        value=dict(case=case,processes=len(group),candidate_count=group[0]['candidate_count'])
        for k in ('frozen_hit','entry_only_hit','internal_only_hit'):value[k]=sum(p[k] for p in group)
        for k in ('entry_spread_ns','observed_envelope_ns','frozen_envelope_ns','entry_only_envelope_ns',
                  'internal_only_envelope_ns','representative_duration_deficit','representative_finish_deficit_ns','best_frozen_finish_deficit_ns'):
            value[k]=statistics.median(p[k] for p in group)
        cases.append(value)
    common.write_json(output/'C-processes.json',dict(processes=processes,provenance=provenance))
    common.write_json(output/'C-summary.json',dict(cases=cases,stages=stage_summary,
        scope='Development diagnostics: fixed frozen candidates; entry-only replacement adds measured entry offsets to predicted duration; internal-only removes entry offsets from measured duration. These are observed counterfactuals, not forecasts. Same-role stage endpoints; merged CTA model residuals kept separate.'))
    with gzip.open(output/'C-stages.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stages[0]));writer.writeheader();writer.writerows(stages)
    common.write_json(output/'C-old-input-pairs.json',old_boundary_pairs(Path(base)))
    return dict(cases=len(cases),processes=len(processes),stage_rows=len(stages))


def paired_boundary_windows(row,control,mapped,control_source='',mapped_source=''):
    """Match CTA/role/work within two independent processes of one trial."""
    if control['trial']!=mapped['trial']:raise ValueError('unmatched trial')
    if [[c['work'],c['role_work']] for c in control['ctas']]!=[[c['work'],c['role_work']] for c in mapped['ctas']]:
        raise ValueError('descriptor pair changed work coordinates')
    left=boundary_stage_rows(row,control,source=control_source)
    right=boundary_stage_rows(row,mapped,source=mapped_source)
    keys=('cta','role','j','stage')
    lookup={tuple(r[k] for k in keys):r for r in right}
    if len(lookup)!=len(left):raise ValueError('descriptor pair has different role windows')
    paired=[]
    for a in left:
        b=lookup[tuple(a[k] for k in keys)]
        cm,cn=model.CONFIGS[row['config']]['cluster']
        m0,n0=(a['mi']//cm)*cm,(a['ni']//cn)*cn
        cluster_classes={boundary_geometry(row,(mi,ni)) for mi in range(m0,m0+cm) for ni in range(n0,n0+cn)}
        cluster_boundary='partial' if 'partial' in cluster_classes else 'oob' if 'oob' in cluster_classes else 'interior'
        paired.append(dict(trial=a['trial'],cta=a['cta'],role=a['role'],j=a['j'],T=a['T'],cluster_boundary=cluster_boundary,
            mi=a['mi'],ni=a['ni'],phase=a['phase'],geometry=a['geometry'],stage=a['stage'],
            oob_ns=a['ns'],address_zero_ns=b['ns'],delta_ns=b['ns']-a['ns'],
            oob_cycles=a['cycles'],address_zero_cycles=b['cycles'],delta_cycles=b['cycles']-a['cycles'],
            oob_source=control_source,address_zero_source=mapped_source))
    return paired


def control_diagnostics(root,output):
    """Same-card P1 input-map intervention; no old-card frozen parameters."""
    import csv
    from analyze_r13_sm import select_attempts
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    common.verify(root)
    rows=[r for r in json.loads((root/'cases.json').read_text()) if r['group']=='r18_partial_map']
    a=next(r for r in rows if r['input_map_m']==960);b=next(r for r in rows if r['input_map_m']==1024)
    if {k:v for k,v in a.items() if k not in ('id','input_map_m')}!={k:v for k,v in b.items() if k not in ('id','input_map_m')}:
        raise ValueError('P1 descriptor intervention changed another field')
    setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
    fixed=('grid','allocated_a_elements','allocated_b_elements','storage_m','storage_n',
           'initialized_input_rows','initialized_input_columns','gpu_uuid','stages','smem')
    if any(setups[a['id']][k]!=setups[b['id']][k] for k in fixed):raise ValueError('P1 setup mismatch')
    cache={};cases=[];provenance={};last_ctas=[]
    for row in rows:
        selected,failed,_=select_attempts(root,row['id'],('plain','dual'))
        variants={v:{} for v in ('plain','dual')};records={}
        for item in selected:
            rec=item['record'];obs=replay(root,rec,row,retain_roles=rec['variant']=='dual')
            variants[rec['variant']][rec['trial']]=obs
            source=str(root/rec['raw']);provenance[source]=rec['raw_sha256'];records[rec['variant'],rec['trial']]=source
            if rec['variant']=='dual':
                ctas=obs['ctas'];earliest=min(c['entry_ns'] for c in ctas);last=maximum_set([c['end_ns'] for c in ctas])
                details=[]
                for i in last:
                    c=ctas[i];ns=c['tiles_ns']
                    details.append(dict(cta=i,sm=c['sm'],work=c['work'],geometry=[boundary_geometry(row,w) for w in c['work']],
                        entry_offset_ns=c['entry_ns']-earliest,duration_ns=c['end_ns']-c['entry_ns'],
                        prefix_ns=ns[0][0]-c['entry_ns'],L_ns=[t[1]-t[0] for t in ns],E_ns=[t[3]-t[2] for t in ns],
                        tail_ns=c['end_ns']-ns[-1][3]))
                last_ctas.append(dict(case=row['id'],trial=rec['trial'],source=source,
                    envelope_ns=max(c['end_ns'] for c in ctas)-earliest,
                    entry_spread_ns=max(c['entry_ns'] for c in ctas)-earliest,last=details))
        if any(set(variants[v])!=set(range(10)) for v in variants):raise ValueError('P1 needs ten successful processes per variant')
        times={v:[variants[v][t]['elapsed_us'] for t in range(10)] for v in variants}
        med={v:statistics.median(t) for v,t in times.items()}
        cases.append(dict(case=row['id'],condition=row,median_us=med,
            cv={v:statistics.pstdev(t)/statistics.mean(t) for v,t in times.items()},
            dual_over_plain=med['dual']/med['plain']-1,times_us=times,failed_attempts=failed))
        cache[row['id']]=(variants,records)
    ac,ar=cache[a['id']];bc,br=cache[b['id']];windows=[]
    for trial in range(10):
        windows.extend(paired_boundary_windows(a,ac['dual'][trial],bc['dual'][trial],ar['dual',trial],br['dual',trial]))
    groupkeys=('trial','role','phase','geometry','cluster_boundary','stage','j','T')
    grouped=defaultdict(list)
    for w in windows:grouped[tuple(w[k] for k in groupkeys)].append(w)
    processes=[]
    for key,group in sorted(grouped.items()):
        value=dict(zip(groupkeys,key),windows=len(group))
        for metric in ('oob_ns','address_zero_ns','delta_ns','oob_cycles','address_zero_cycles','delta_cycles'):
            value[metric]=statistics.median(r[metric] for r in group)
        processes.append(value)
    groups=defaultdict(list)
    for p in processes:groups[tuple(p[k] for k in groupkeys[1:])].append(p)
    summary=[]
    for key,group in sorted(groups.items()):
        value=dict(zip(groupkeys[1:],key),processes=len(group))
        for metric in ('oob_ns','address_zero_ns','delta_ns','oob_cycles','address_zero_cycles','delta_cycles'):
            value[metric]=statistics.median(r[metric] for r in group)
        value['delta_ns_min']=min(r['delta_ns'] for r in group);value['delta_ns_max']=max(r['delta_ns'] for r in group)
        summary.append(value)
    contrasts={v:dict(oob_us=statistics.median(ac[v][t]['elapsed_us'] for t in range(10)),
        address_zero_us=statistics.median(bc[v][t]['elapsed_us'] for t in range(10)),
        paired_relative=[bc[v][t]['elapsed_us']/ac[v][t]['elapsed_us']-1 for t in range(10)]) for v in ('plain','dual')}
    for v in contrasts:contrasts[v]['relative']=contrasts[v]['address_zero_us']/contrasts[v]['oob_us']-1
    result=dict(root=str(root),environment=json.loads((root/'environment.json').read_text()),cases=cases,
        contrasts=contrasts,stage_summary=summary,stage_processes=processes,critical_ctas=last_ctas,
        raw_sha256=provenance,analyzer_sha256=common.sha(Path(__file__)),
        scope='Same-card, same-output P1 descriptor intervention. Positive delta means valid-address zero slower than descriptor OOB. Independent processes paired by trial and coordinate; statistics reduce correlated CTA windows within each process first. Same-role cycles/ns and merged last-CTA durations are separate. No old-card model parameter or physical write completion inferred.')
    common.write_json(output/'C-controls.json',result)
    with gzip.open(output/'C-controls-windows.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
    return dict(cases=2,processes=40,paired_windows=len(windows),contrasts=contrasts,stage_summary=summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--v08-padding',action='store_true')
    p.add_argument('--input-map-pairs',action='store_true')
    a=p.parse_args()
    if a.input_map_pairs:
        if a.output is None:p.error('--input-map-pairs requires a new --output directory')
        analyze_input_map_pairs(a.input.resolve(),a.output)
    elif a.v08_padding:
        if a.output is None:p.error('--v08-padding requires a new --output directory')
        analyze_v08_padding(a.input.resolve(),a.output)
    else:analyze(a.input.resolve(),a.output)

#!/usr/bin/env python3
"""V05 offline, phase-separated scoring against the original frozen prediction."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
from v05_predict import scheduled_work

EVENTS = ('work', 'first_mma', 'main_end', 'epi_permit', 'store_return',
          'source_release', 'next_start', 'load_return')
CAPACITY, WORDS = 32, 24
CTA_WORDS = 4 + 3 * CAPACITY * WORDS


@lru_cache(maxsize=None)
def reference(k):
    # Direct K-length integer dot products, independent of the runner's period sum.
    return [[sum(((r*7+t*13)%17-8)*((t*5+c*11)%17-8)
                 for t in range(k))/1024 for c in range(17)] for r in range(17)]


def check_values(setup, check):
    positions, values = check['checked_indices'], check['checked_values']
    if len(positions)!=4096 or len(set(positions))!=4096 or len(values)!=4096:
        raise ValueError('sample identity/count mismatch')
    ref = reference(setup['k'])
    for index, value in zip(positions, values):
        if not 0 <= index < setup['m']*setup['n']:
            raise ValueError('check index outside output')
        r,c = divmod(index, setup['n'])
        if value is None or value != ref[r%17][c%17]:
            raise ValueError('independent integer reference mismatch')
    if check['padding_errors'] != 0: raise ValueError('padding overwritten')


def decode(setup, trace):
    ctas = math.prod(setup['grid']); pair = setup['trace_pair']
    if setup['trace_version']!=6 or len(trace)!=ctas*CTA_WORDS:
        raise ValueError('trace format mismatch')
    work = scheduled_work(setup); rows = []
    for cta in range(ctas):
        block = trace[cta*CTA_WORDS:(cta+1)*CTA_WORDS]
        if pair!=3 and cta!=0:
            if any(block): raise ValueError('unselected CTA wrote trace')
            continue
        if not work[cta]:
            if any(block): raise ValueError('empty CTA wrote trace')
            continue
        if pair==3:
            data = block[4:4+WORDS]
            if block[0]!=1 or block[1]!=1 or block[3] or block[2]!=data[3]+1:
                raise ValueError('critical endpoints or SM identity missing')
            if tuple(data[:3])!=work[cta][0]: raise ValueError('critical initial tile mismatch')
            start,end = data[4],data[14]
            if not 0<start<=end or not 0<data[5]<=data[15]:
                raise ValueError('critical event order')
            allowed={0,1,2,3,4,5,14,15}
            if any(v for i,v in enumerate(data) if i not in allowed) or any(block[4+WORDS:]):
                raise ValueError('critical trace contains extra events')
            rows.append(dict(cta=cta,pair=pair,critical_cta=end-start))
            continue
        expected = [work[cta],work[cta],work[cta]]
        if setup['config']=='cfg_b': expected=[work[cta],work[cta][::2],work[cta][1::2]]
        if block[:3]!=[len(x) for x in expected] or block[3]:
            raise ValueError('role counts/overflow mismatch')
        smids=set()
        for role,coords in enumerate(expected):
            for seq in range(CAPACITY):
                data=block[4+(role*CAPACITY+seq)*WORDS:4+(role*CAPACITY+seq+1)*WORDS]
                if seq>=len(coords):
                    if any(data): raise ValueError('unused tile record not zero')
                    continue
                if tuple(data[:3])!=coords[seq]: raise ValueError('scheduler order mismatch')
                smids.add(data[3])
                required=[]
                if pair==2: required=['work'] + (['first_mma'] if role else [])
                elif role and pair==0: required=['first_mma','main_end']
                elif role and pair==1:
                    required=['main_end','epi_permit']
                    if setup['config']=='cfg_b' or role==2: required+=['source_release']
                row=dict(cta=cta,role=role,seq=seq,coord=coords[seq],pair=pair)
                for i,event in enumerate(EVENTS):
                    cycle,ns=data[4+2*i:6+2*i]
                    if event in required:
                        if cycle<=0 or ns!=0: raise ValueError('missing cycle or unexpected globaltimer')
                        row[event]=cycle
                    elif cycle or ns: raise ValueError('unexpected event')
                times=[row[event] for event in required]
                if times!=sorted(times): raise ValueError('event order violation')
                rows.append(row)
        if len(smids)!=1: raise ValueError('mixed SM clock domains')
    return rows


def measures(rows):
    if not rows: raise ValueError('empty observations')
    pair=rows[0]['pair']
    if pair==3: return dict(critical_cta=max(r['critical_cta'] for r in rows))
    tiles=defaultdict(list); values=defaultdict(list)
    for row in rows: tiles[row['coord']].append(row)
    timeline=[]
    final_coord=max((r for r in rows if r.get('role')==0),key=lambda r:r['seq'])['coord']
    for coord,tile in tiles.items():
        consumers=[r for r in tile if r['role']]
        if pair==0:
            first=min(r['first_mma'] for r in consumers)
            end=max(r['main_end'] for r in consumers)
            values['mainloop'].append(end-first); timeline.append((first,end))
        elif pair==1:
            end=max(r['main_end'] for r in consumers)
            epi=max(r['epi_permit'] for r in consumers)
            release=max(r['source_release'] for r in consumers if 'source_release' in r)
            if coord==final_coord:
                values['epi_permission_wait']=[epi-end]
                values['output']=[release-epi]
        else:
            producer=next(r for r in tile if r['role']==0)
            if producer['seq']==0:
                values['supply'].append(min(r['first_mma'] for r in consumers)-producer['work'])
    if pair==0:
        timeline.sort()
        values['handoff']=[b[0]-a[1] for a,b in zip(timeline,timeline[1:])]
    return {name:statistics.median(x) for name,x in values.items() if x}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); run=a.input; a.output.mkdir(parents=True,exist_ok=False)
    prediction=json.loads((run/'predictions.json').read_text())
    binding=json.loads((run/'prediction_binding.json').read_text())
    digest=hashlib.sha256((run/'predictions.json').read_bytes()).hexdigest()
    if binding['sha256']!=digest: raise ValueError('prediction digest mismatch')
    for key,name in (('source_manifest_sha256','source_hashes.json'),
                     ('binary_manifest_sha256','build/binary_hashes.json'),
                     ('static_setup_sha256','static_setup.json'),
                     ('environment_sha256','environment.json')):
        if hashlib.sha256((run/name).read_bytes()).hexdigest()!=prediction.get(key):
            raise ValueError('frozen identity mismatch: '+name)
    for name in ('source_hashes.json','build/binary_hashes.json'):
        for path,h in json.loads((run/name).read_text()).items():
            if hashlib.sha256((run/path).read_bytes()).hexdigest()!=h:
                raise ValueError('artifact changed: '+path)
    if not 0 < binding['frozen_unix_ns'] < binding['first_sample_unix_ns']:
        raise ValueError('invalid first-sample/freeze order')
    case_map={c['id']:c for c in prediction['cases']}
    setup_map={s['case_id']:s for s in json.loads((run/'static_setup.json').read_text())}
    if len(case_map)!=18 or len(setup_map)!=18:raise ValueError('18 unique cases required')
    pair_ids={'main':0,'output':1,'supply':2,'critical':3}
    observed=defaultdict(list); durations=defaultdict(dict); failures=[]; seen=set()
    for path in sorted((run/'samples').rglob('result.json')):
        try:
            result=json.loads(path.read_text())
            if result['status']!='measured': raise ValueError('failed sample')
            raw_path=path.parent/'stdout.txt.gz'
            raw=result['raw']
            if (raw['path']!=raw_path.name or raw['bytes']!=raw_path.stat().st_size
                    or raw['sha256']!=hashlib.sha256(raw_path.read_bytes()).hexdigest()):
                raise ValueError('raw identity mismatch')
            with gzip.open(path.parent/'stdout.txt.gz','rt') as f:
                events=[json.loads(line) for line in f if line.strip()]
            setup=next(x for x in events if x.get('event')=='setup')
            call=next(x for x in events if x.get('event')=='call')
            check=next(x for x in events if x.get('event')=='check')
            if check['status']!='ok' or check['samples']!=4096:
                raise ValueError('output correctness evidence missing')
            warm=call['warmup_us']
            if (not 8<=len(warm)<=30 or any(not math.isfinite(v) or v<=0 for v in warm)
                    or statistics.pstdev(warm[-5:])/statistics.mean(warm[-5:])>.02):
                raise ValueError('warmup evidence not converged')
            check_values(setup,check)
            case=result['case_id'].split('_',1)[1]
            kind='trace' if setup['traced'] else 'plain'
            observation=path.parent.name.split('-')[0]
            if observation not in pair_ids or result.get('set')!=observation:
                raise ValueError('observation label mismatch')
            if result.get('returncode')!=0 or type(result['trial']) is not int or not 0<=result['trial']<10:
                raise ValueError('process/trial identity')
            expected_case=case_map[case]; static=setup_map[case]
            for key in ('config','m','n','k','lda','ldb','ldd'):
                if setup[key]!=expected_case[key]:raise ValueError('case setup mismatch: '+key)
            for key in ('grid','tile','cluster','stages','threads','smem','trace_tile_capacity'):
                if setup[key]!=static[key]:raise ValueError('static setup mismatch: '+key)
            if setup['traced'] and setup['trace_pair']!=pair_ids[observation]:
                raise ValueError('trace belongs to another observation pair')
            if not setup['traced'] and call['trace']:
                raise ValueError('plain unexpectedly traced')
            if not math.isfinite(call['elapsed_us']) or call['elapsed_us']<=0:
                raise ValueError('invalid event duration')
            if result['started_unix_ns']<binding['first_sample_unix_ns']:
                raise ValueError('sample predates frozen binding')
            key=(case,result['trial'],observation,kind)
            if key in seen:raise ValueError('duplicate process identity')
            seen.add(key)
            durations[(case,result['trial'],observation)][kind]=call['elapsed_us']
            if setup['traced']: observed[case].append(measures(decode(setup,call['trace'])))
        except (ValueError,KeyError,StopIteration) as exc:
            failures.append(dict(path=str(path),error=str(exc)))
    scores=[]
    for case in prediction['cases']:
        name=case['id']; rows=observed[name]
        for phase,reference in prediction['predictions'][name].items():
            actual=[r[phase] for r in rows if phase in r]
            applicable = phase!='handoff' or len(prediction['work_assignment'][name]['cta_work'][0])>1
            if reference.get('applicable',True)!=applicable:
                failures.append(dict(case=name,phase=phase,error='applicability mismatch'))
                continue
            if not applicable:
                if actual:failures.append(dict(case=name,phase=phase,error='unexpected handoff observation'))
                scores.append(dict(case=name,phase=phase,predicted_cycles=reference['cycles'],
                    measured_cycles=None,absolute_relative_error=None,applicable=False,
                    small_phase=False,event_relation_ok=True,reason='one tile on observed CTA; no handoff'))
                continue
            if len(actual)!=10:
                failures.append(dict(case=name,phase=phase,error='expected 10 observations'))
                continue
            median=statistics.median(actual); expected=reference['cycles']
            error=abs(expected-median)/max(abs(median),1)
            relation_ok = phase!='handoff' or (expected<0)==(median<0)
            if abs(median)<512 and not relation_ok:
                failures.append(dict(case=name,phase=phase,error='small handoff overlap/order prediction mismatch'))
            scores.append(dict(case=name,phase=phase,predicted_cycles=expected,
                measured_cycles=median,absolute_relative_error=error,
                small_phase=abs(median)<512,applicable=True,event_relation_ok=relation_ok))
    phases={}
    for phase in ('supply','mainloop','output','handoff','epi_permission_wait','critical_cta'):
        group=[r for r in scores if r['phase']==phase and r['applicable'] and not r['small_phase']]
        errors=[r['absolute_relative_error'] for r in group]
        phases[phase]=dict(scored_cases=len(errors),
            median=statistics.median(errors) if errors else None,max=max(errors) if errors else None,
            pass_threshold=(statistics.median(errors)<=.1 and max(errors)<=.2) if errors
                else len([r for r in scores if r['phase']==phase])==18,
            not_applicable_cases=[r['case'] for r in scores if r['phase']==phase and not r['applicable']],
            small_phase_rule='validated event order/overlap sign; excluded from percentage threshold')
    pairs=[dict(case=c,trial=t,observation=observation,relative=p['trace']/p['plain']-1)
           for (c,t,observation),p in durations.items() if set(p)=={'plain','trace'}]
    trace_ok=len(pairs)==720 and all(abs(p['relative'])<=.05 for p in pairs)
    stability=[]
    for case in case_map:
        for observation in pair_ids:
            for kind in ('plain','trace'):
                values=[p[kind] for (name,trial,obs),p in durations.items()
                        if name==case and obs==observation and kind in p]
                cv=statistics.pstdev(values)/statistics.mean(values) if len(values)==10 else None
                stability.append(dict(case=case,observation=observation,kind=kind,cv=cv))
                if cv is None or cv>.05:
                    failures.append(dict(case=case,observation=observation,kind=kind,
                                         error='missing samples or CV above 5%',cv=cv))
    times=[]
    for case in prediction['cases']:
        values=[p['plain'] for (name,trial,observation),p in durations.items()
                if name==case['id'] and observation=='main' and 'plain' in p]
        if len(values)==10:
            expected=prediction['microseconds'][case['id']]['predicted_us']
            actual=statistics.median(values)
            times.append(dict(case=case['id'],predicted_us=expected,measured_us=actual,
                              absolute_relative_error=abs(expected-actual)/actual))
    report=dict(microseconds=times,stability=stability,status='pass' if not failures and trace_ok and
        all(p['pass_threshold'] for p in phases.values()) else 'not_passed',
        prediction_sha256=digest,phases=phases,scores=scores,trace_pairs=pairs,
        trace_qualified=trace_ok,failures=failures,
        boundaries='supply=first producer work to first MMA; output=final tile permission to source-tail return; '
                   'critical CTA=first producer work to final source release, not kernel exit')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],failures=len(failures))))


if __name__=='__main__': main()

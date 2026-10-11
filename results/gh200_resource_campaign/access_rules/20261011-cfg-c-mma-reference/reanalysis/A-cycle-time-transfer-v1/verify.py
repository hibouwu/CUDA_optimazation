#!/usr/bin/env python3
"""Independently bind reference/target integers and labeled time-error identities."""
import copy,gzip,hashlib,json
from fractions import Fraction as Q
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'transfer.json').read_text())
for filename,digest in report['input_sha256'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
assert hashlib.sha256((ROOT/'analyze.py').read_bytes()).hexdigest()==report['analysis_sha256']
reference_root=next(Path(p).parent for p in report['input_sha256'] if p.endswith('/20261011-cfg-c-mma-reference/manifest.json'))
records=json.loads((reference_root/'returned/records.json').read_text())
long_cycles=[];long_ns=[];reference_begin=[];reference_end=[]
for record in records:
    original=json.loads((reference_root/'returned/samples'/record['id']/'stdout.json').read_text())
    ns0,ns1,c0,c1,sm=original['stamps'][0]
    assert sm==124 and original['elapsed']==c1-c0
    reference_begin.append(ns0);reference_end.append(ns1)
    expected=next(r for r in report['references'] if r['id']==record['id'])
    assert expected['cycles']==c1-c0 and expected['ns']==ns1-ns0
    assert Q(expected['observed_cycle_per_ns'])==Q(c1-c0,ns1-ns0)
    if record['iterations']==1024:long_cycles.append(c1-c0);long_ns.append(ns1-ns0)
cycle=Q(st.median(long_cycles));duration=Q(st.median(long_ns))
candidate=report['candidate'];ratio=Q(2*256*128*4096,2*64*128*16*16*1024)
assert ratio==Q(candidate['work_ratio'])==Q(1,16)
pc=cycle*ratio;rate=cycle/duration;pn=pc/rate
assert (Q(candidate['reference_cycles']),Q(candidate['reference_ns']))==(cycle,duration)
assert Q(candidate['reference_cycle_per_ns'])==rate
assert Q(candidate['predicted_common_cycle'])==pc and Q(candidate['predicted_common_ns'])==pn

lookup={};target_begin=[];target_end=[]
for run in ('20261011-cfg-c-producer-tail-control','20261011-cfg-c-producer-tail-off-off'):
    root=next(Path(p).parent for p in report['input_sha256'] if p.endswith('/'+run+'/manifest.json'))
    records=json.loads((root/'returned/records.json').read_text())
    for record in records:
        with gzip.open(root/'returned'/record['raw'],'rt') as stream:
            events={q['event']:q for line in stream if line.strip() for q in [json.loads(line)]}
        trace=events['call']['trace'];arm=record.get('arm',record['flag'])
        for cta in range(132):
            h=cta*1296
            assert trace[h+2]==trace[h+11]==trace[h+12]
            target_begin.append(trace[h+1]);target_end.extend((trace[h+5],trace[h+8]))
            assert trace[h+6]==trace[h+9]
            for j in range(trace[h+6]):
                offsets=[h+16+wg*640+j*10 for wg in (0,1)]
                start=min(offsets,key=lambda k:trace[k]);end=max(offsets,key=lambda k:trace[k+1])
                key=(run,record['pair'],arm,cta,j)
                assert key not in lookup
                lookup[key]=(trace[end+1]-trace[start],trace[end+7]-trace[start+6],trace[h+2]-1)
assert len(lookup)==8640
with gzip.open(ROOT/'window-errors.json.gz','rt') as stream:rows=json.load(stream)


def verify_row(row):
    key=(row['run'],row['pair'],row['arm'],row['cta'],row['ordinal'])
    c,n,sm=lookup[key]
    assert (row['cycle'],row['ns'],row['sm'])==(c,n,sm)
    ec,ef,en=Q(row['cycle_error']),Q(row['conditional_conversion_error']),Q(row['composed_window_error'])
    assert ec==pc/c-1 and ef==Q(c,n)/rate-1 and en==pn/n-1
    assert (1+ec)*(1+ef)==1+en


assert len(rows)==8640 and len({(r['run'],r['pair'],r['arm'],r['cta'],r['ordinal']) for r in rows})==8640
for row in rows:verify_row(row)
same_sm=sum(r['sm']==124 for r in rows);assert same_sm==80
for group in report['groups']:
    selected=[r for r in rows if r['run']==group['run'] and r['ordinal']==group['ordinal'] and (group['selection']=='all_SM' or r['sm']==124)]
    for field,metrics in group['metrics'].items():
        values=[Q(r[field]) for r in selected]
        assert metrics['n']==len(values)
        assert metrics['median']==float(st.median(values))
        assert metrics['maximum_absolute']==float(max(map(abs,values)))
negative=[]
wrong=copy.deepcopy(rows[0]);wrong['cycle_error'],wrong['conditional_conversion_error']=wrong['conditional_conversion_error'],wrong['cycle_error']
assert (1+Q(wrong['cycle_error']))*(1+Q(wrong['conditional_conversion_error']))==1+Q(wrong['composed_window_error'])
try:verify_row(wrong)
except AssertionError:negative.append('swapped labeled terms rejected despite product closure')
else:raise AssertionError('wrong term labels accepted')
wrong=copy.deepcopy(rows[0]);wrong['ns']+=32
wrong['conditional_conversion_error']=str(Q(wrong['cycle'],wrong['ns'])/rate-1)
wrong['composed_window_error']=str(pn/wrong['ns']-1)
assert (1+Q(wrong['cycle_error']))*(1+Q(wrong['conditional_conversion_error']))==1+Q(wrong['composed_window_error'])
try:verify_row(wrong)
except AssertionError:negative.append('changed timer delta rejected despite recalculated closed terms')
else:raise AssertionError('wrong native timer delta accepted')
chronology=dict(target_first_entry_ns=min(target_begin),target_last_recorded_final_ns=max(target_end),
    reference_first_ns=min(reference_begin),reference_last_ns=max(reference_end),
    reference_starts_after_last_target_recorded_endpoint_ns=min(reference_begin)-max(target_end))
assert chronology['reference_starts_after_last_target_recorded_endpoint_ns']>0
result=dict(passed=True,input_hashes=len(report['input_sha256']),reference_calls=6,raw_target_calls=40,
    exact_native_window_identities=len(lookup),labeled_multiplicative_closures=len(rows),
    same_reference_SM_target_windows=same_sm,negative_checks=negative,recorded_globaltimer_chronology=chronology,
    scope='Exact recorded differences and retrospective transfer arithmetic; no hardware simultaneous-read or physical-rate proof.')
(ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

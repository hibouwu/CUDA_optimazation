#!/usr/bin/env python3
"""Verify raw endpoint identities and scalar minimax certificates independently."""
import copy,gzip,hashlib,json
from fractions import Fraction as Q
from pathlib import Path

ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'report.json').read_text())
for filename,digest in report['input_sha256'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
assert hashlib.sha256((ROOT/'analyze.py').read_bytes()).hexdigest()==report['analysis_sha256']
with gzip.open(ROOT/'windows.json.gz','rt') as stream:rows=json.load(stream)
with gzip.open(ROOT/'endpoint-selections.json.gz','rt') as stream:endpoints=json.load(stream)
assert len(rows)==12960 and len(endpoints)==8640

packets={}
for run in {r['run'] for r in rows}:
    root=next(Path(p).parent for p in report['input_sha256'] if p.endswith('/'+run+'/manifest.json'))
    records=json.loads((root/'returned/records.json').read_text())
    for record in records:
        with gzip.open(root/'returned'/record['raw'],'rt') as stream:
            events={q['event']:q for line in stream if line.strip() for q in [json.loads(line)]}
        key=(run,record['pair'],record.get('arm',record['flag']))
        assert key not in packets
        packets[key]=events['call']['trace']
assert len(packets)==40


def native(trace,cta,ordinal,wg):
    head=cta*1296
    assert trace[head+11]==trace[head+12]==trace[head+2]
    assert trace[head+6]==trace[head+9]>ordinal
    at=head+16+wg*640+ordinal*10
    return trace[at:at+10]


for row in rows:
    work=None
    for arm in (0,1):
        trace=packets[row['run'],row['pair'],arm]
        slots=[native(trace,row['cta'],row['ordinal'],wg) for wg in (0,1)]
        assert slots[0][4:6]==slots[1][4:6]
        assert row['work']==[v-1 for v in slots[0][4:6]]
        if row['scope']=='WG':
            slot=slots[row['wg']];cycles=slot[1]-slot[0];ns=slot[7]-slot[6]
        else:
            start=min(range(2),key=lambda w:slots[w][0]);end=max(range(2),key=lambda w:slots[w][1])
            assert row['clock_labels'+str(arm)]==[start,end]
            cycles=slots[end][1]-slots[start][0];ns=slots[end][7]-slots[start][6]
        assert row['c'+str(arm)]==cycles>0 and row['n'+str(arm)]==ns>0
for row in endpoints:
    slots=[native(packets[row['run'],row['pair'],row['arm']],row['cta'],row['ordinal'],wg) for wg in (0,1)]
    start=min(range(2),key=lambda w:slots[w][0]);end=max(range(2),key=lambda w:slots[w][1])
    assert row['span_cycles']==max(s[1] for s in slots)-min(s[0] for s in slots)
    assert row['ns_envelope_extra_ns']==max(s[7] for s in slots)-min(s[6] for s in slots)-(slots[end][7]-slots[start][6])


def verify_bound(values,bound):
    q=[Q(r['n0']*r['c1'],r['c0']) for r in values];n=[r['n1'] for r in values]
    absolute=bound['absolute'];delta=Q(absolute['radius_ns']);alpha=Q(absolute['alpha'])
    assert alpha>0 and delta>=0 and max(abs(alpha*x-y) for x,y in zip(q,n))==delta
    i,j=absolute['witness_indices']
    lower=abs(n[i]*q[j]-n[j]*q[i])/(q[i]+q[j])
    assert lower==delta
    assert Q(absolute['alpha_lower'])==max((y-delta)/x for x,y in zip(q,n))
    assert Q(absolute['alpha_upper'])==min((y+delta)/x for x,y in zip(q,n))
    assert Q(absolute['alpha_lower'])<=alpha<=Q(absolute['alpha_upper'])
    z=[x/y for x,y in zip(q,n)];relative=bound['relative']
    lo,hi=min(z),max(z);r=Q(relative['alpha']);floor=Q(relative['floor'])
    assert z[relative['minimum_z_index']]==lo and z[relative['maximum_z_index']]==hi
    assert r==2/(lo+hi) and floor==(hi-lo)/(hi+lo)
    assert max(abs(r*x-1) for x in z)==floor


for bound in report['bounds']:
    values=rows[bound['row_offset']:bound['row_offset']+bound['n']]
    assert all((r['run'],r['pair'],r['scope'])==(bound['run'],bound['pair'],bound['scope']) for r in values)
    verify_bound(values,bound)
negative=[]
original=report['bounds'][0];values=rows[original['row_offset']:original['row_offset']+original['n']]
wrong=copy.deepcopy(original);wrong['absolute']['alpha']=str(Q(wrong['absolute']['alpha'])+Q(1,100))
try:verify_bound(values,wrong)
except AssertionError:negative.append('wrong oracle scale rejected')
else:raise AssertionError('corrupt scale accepted')
wrong=copy.deepcopy(original);wrong['absolute']['witness_indices']=[0,0]
try:verify_bound(values,wrong)
except AssertionError:negative.append('wrong lower-bound witness rejected')
else:raise AssertionError('corrupt witness accepted')
wrong=copy.deepcopy(original);wrong['relative']['floor']='0'
try:verify_bound(values,wrong)
except AssertionError:negative.append('wrong relative floor rejected')
else:raise AssertionError('corrupt relative floor accepted')
result=dict(passed=True,input_hashes=len(report['input_sha256']),raw_calls=len(packets),
    exact_raw_window_identities=len(rows),endpoint_selection_identities=len(endpoints),
    exact_minimax_certificates=len(report['bounds']),negative_checks=negative,
    scope='Raw same-endpoint arithmetic and conditional oracle certificates only; no physical clock or observer qualification.')
proxy_file=ROOT/'proxy-gap.json'
if proxy_file.exists():
    proxy=json.loads(proxy_file.read_text())
    for filename,digest in proxy['input_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
    for item in proxy['pairs']:
        bound=next(b for b in report['bounds'] if b['run']==item['run'] and b['pair']==item['pair'] and b['scope']=='WG')
        values=rows[bound['row_offset']:bound['row_offset']+bound['n']]
        errors=[Q(bound['alpha_last_prefix'])*Q(r['n0']*r['c1'],r['c0']*r['n1'])-1 for r in values]
        assert Q(item['prefix_maximum_relative_residual'])==max(map(abs,errors))
        assert item['optimal_maximum_relative_residual']==bound['relative']['floor']
    result['prefix_proxy_pair_checks']=len(proxy['pairs'])
(ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

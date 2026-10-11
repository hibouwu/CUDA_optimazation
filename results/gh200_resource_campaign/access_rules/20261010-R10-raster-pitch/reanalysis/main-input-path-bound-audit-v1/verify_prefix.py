"""Check every reported initial envelope against unpruned FIFO affine terms."""
from fractions import Fraction as F
import hashlib,itertools,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent;repo=root.parents[5]
sys.path.insert(0,str(repo/'microbench/gh200_resource_campaign/access_rules'))
from supply_model import cfg_b_source_program
from analyze_r10_granule_overlap import source_granules
r=json.loads((root/'B-report-snapshot.json').read_text());p=r['parameters']
original=root.parent/'B-20261011-source-event-transfer-v3/B-r10-event-transfer.json'
assert hashlib.sha256(original.read_bytes()).hexdigest()==r['input_sha256'][str(original.resolve())]
context=json.loads(original.read_text())['contexts'][r['case']]
program=cfg_b_source_program(context['row'],context['setup'])
q=lambda x:F.from_float(x) if isinstance(x,float) else F(x)
lower,upper=F(r['price_interval']['lower']),F(r['price_interval']['upper'])
op,pg,lat,dest,cost=map(q,(p['operand_gap_ns'],p['producer_gap_ns'],p['read_latency_ns'],p['destination_service_ns'],p['service_ns']/128))
units={};source_frontier=F(0);prefix_work=F(0);arrivals={}
local=[[] for _ in range(126)];last_source=[F(0)]*126
actual={};demand_count=0
assert 5*(2*op+pg)+2*op<lat  # No initial unit can expire.
for g in range(6):
    for atom,work in enumerate((F(1),F(1,2),F(1,2))):
        t=g*(2*op+pg)+atom*op
        for cta,groups in enumerate(program['source_keys']):
            key=groups[g][atom]
            requested={(key[0],i) for i in source_granules(key,128)}
            fresh=requested-units.keys()
            if fresh:
                source_frontier=max(source_frontier,t)+len(fresh)*cost
                for u in fresh:units[u]=source_frontier+lat
            ready=max(units[u] for u in requested)
            assert ready>=last_source[cta];last_source[cta]=ready
            # Closed queue formula: max_j(t_j + sigma*(Q_i-Q_{j-1})).
            # Equal arrival times need only the smallest Q_{j-1}, since sigma>0.
            if t not in arrivals:arrivals[t]=prefix_work
            prefix_work+=work
            path_terms=[(prefix_work-before,arrival+lat) for arrival,before in arrivals.items()]
            admission=[(F(0),ready)]+path_terms
            local[cta]=[(a,b+dest*work) for a,b in local[cta]+admission]
            if atom==2:actual[cta,g]=list(local[cta])
            demand_count+=1
envelope_checks=0;segment_checks=0
for e in r['initial_envelopes']:
    reported=[(F(x['slope']),F(x['intercept'])) for x in e['lines']]
    candidates=actual[e['cta'],e['group']]
    points={lower,upper}
    for (a,b),(c,d) in itertools.combinations(reported,2):
        if a!=c:
            x=(d-b)/(a-c)
            if lower<x<upper:points.add(x)
    points=sorted(points)
    for left,right in zip(points,points[1:]):
        mid=(left+right)/2
        active=max(reported,key=lambda ab:ab[0]*mid+ab[1])
        # One original term attains the reported line over the whole segment.
        assert active in candidates
        for x in (left,right):
            predicted=active[0]*x+active[1]
            assert max(a*x+b for a,b in candidates)==predicted
        segment_checks+=1
    envelope_checks+=1
assert demand_count==r['initial_prefix_demands']==2268
result=dict(envelopes=envelope_checks,affine_segments_checked=segment_checks,
    exact_initial_demands=demand_count,all_envelopes_equal_on_closed_interval=True,
    B_snapshot_sha256=hashlib.sha256((root/'B-report-snapshot.json').read_bytes()).hexdigest(),
    method='Unpruned closed request FIFO terms and local FIFO recursion; each reported segment has an actual attaining line and all terms lie below it at both endpoints.',
    scope='Rational initial-prefix formulas only; final delivered source binding still required.')
(root/'prefix-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

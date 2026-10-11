"""Explain the two already-selected minimum-margin witnesses, with no fitting."""
from fractions import Fraction as Q
import hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent
base=out.parents[2];repo=base.parents[2]
code=repo/'microbench/gh200_resource_campaign/access_rules'
sys.path.insert(0,str(code))
from supply_model import cfg_b_source_program,shared_stage_events
from analyze_r10_granule_overlap import source_granules
inputs={}
def read(p):
    inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
original=read(out.parent/'B-20261011-source-event-transfer-v3/B-r10-event-transfer.json')
fixed=read(out.parent/'main-input-path-v1/B-input-path.json')
endpoints=read(out.parent/'main-input-path-endpoints-v1/report.json')
selected=read(out.parent/'main-input-path-endpoints-v1/verification.json')['endpoint_summary']
rows=[]
for witness in selected:
    context=original['contexts'][witness['limiting_case']]
    row,setup=context['row'],context['setup'];p=fixed['parameters']
    program=cfg_b_source_program(row,setup)
    keys={key for groups in program['source_keys'] for group in groups for key in group}
    units={key:tuple((key[0],i) for i in source_granules(key,128)) for key in keys}
    price=witness['price_ns']
    model=shared_stage_events(row['k']//64,setup['stages'],len(program['output_counts']),
        source_keys=program['source_keys'],output_counts=program['output_counts'],
        share_inflight_sources=True,source_address_units=units,
        source_unit_service_ns=p['service_ns']/128,input_min_latency_ns=p['read_latency_ns'],
        input_path_service_ns=price,trace_requests=True,**p)
    cta=witness['cta'];g=witness['group'];actor=model['ctas'][cta]
    assert g==1
    prior=next(r for r in endpoints['rows'] if r['case']==witness['limiting_case'] and r['endpoint']==witness['endpoint'])
    assert actor['starts'][0]==prior['ctas'][cta]['first_ns']
    margin=actor['starts'][0]+g*p['compute_ns']-actor['ready'][g]
    assert margin==witness['minimum_margin_ns']
    demands=[d for d in model['input_demands'] if d['cta']==cta and d['index']<=g]
    deliveries={d['demand_id']:d for d in model['deliveries'] if d['cta']==cta and d['index']<=g}
    details=[]
    for d in demands:
        delivery=deliveries[d['demand_id']]
        path=d['input_path_end_ns']+p['read_latency_ns'];source=d['source_ready_ns']
        assert max(source,path)==d['destination_arrival_ns']
        details.append(dict(demand=d,delivery=delivery,
            winning_admission_branch='source' if source>path else 'request_path' if path>source else 'tie',
            path_plus_latency_ns=path,local_queue_wait_ns=delivery['start_ns']-d['destination_arrival_ns']))
    for group in (0,1):
        assert max(deliveries[d['demand_id']]['ready_ns'] for d in demands if d['index']==group)==actor['ready'][group]
    rows.append(dict(witness=witness,margin_ns=margin,ready_gap_ns=actor['ready'][1]-actor['ready'][0],
        first_ns=actor['starts'][0],first_ready_ns=actor['ready'][0],details=details,
        aggregate_nominal_group_path_ns=264*price,
        aggregate_group_compute_headroom_ns=288-264*price))
    source_granules.cache_clear()
for name in ('supply_model.py','analyze_r10_granule_overlap.py'):
    path=code/name;inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
result=dict(input_sha256=inputs,witnesses=rows,
    upper_endpoint_capacity_headroom_exact=str(Q(288)-264*Q(fixed['selection']['upper_exact'])),
    scope='Two frozen endpoint witnesses only; aggregate path throughput is not a sufficient no-stall proof.')
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:
    print(r['witness']['endpoint'],r['margin_ns'],r['ready_gap_ns'],
        [(d['demand']['index'],d['demand']['kind'],d['winning_admission_branch'],d['local_queue_wait_ns']) for d in r['details']])

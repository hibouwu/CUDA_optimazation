"""Two prescribed endpoints, not a price search or an interval proof."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
out = Path(__file__).resolve().parent
root = out.parents[2]
repo = root.parents[2]
code = repo / 'microbench/gh200_resource_campaign/access_rules'
sys.path.insert(0, str(code))
from supply_model import cfg_b_source_program, shared_stage_events
from analyze_r10_granule_overlap import source_granules

inputs = {}
def read(path):
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

fixed = read(out.parent / 'main-input-path-v1/B-input-path.json')
original = read(out.parent / 'B-20261011-source-event-transfer-v3/B-r10-event-transfer.json')
parameters = fixed['parameters']
rows = []
for endpoint in ('lower_exact', 'upper_exact'):
    exact = Fraction(fixed['selection'][endpoint])
    price = float(exact)
    for case, context in original['contexts'].items():
        row, setup = context['row'], context['setup']
        assert setup['gpu_uuid'] == 'GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'
        program = cfg_b_source_program(row, setup)
        keys = {k for groups in program['source_keys'] for group in groups for k in group}
        units = {k: tuple((k[0], i) for i in source_granules(k, 128)) for k in keys}
        kt = row['k'] // 64
        model = shared_stage_events(kt, setup['stages'], len(program['output_counts']),
            source_keys=program['source_keys'], output_counts=program['output_counts'],
            share_inflight_sources=True, source_address_units=units,
            source_unit_service_ns=parameters['service_ns']/128,
            input_min_latency_ns=parameters['read_latency_ns'],
            input_path_service_ns=price, trace_requests=True, **parameters)
        ctas = []
        for cta, actor in enumerate(model['ctas']):
            start = actor['starts'][0]
            # Exclude group 0, whose ready is the FIRST anchor itself. For
            # later groups, compare supply readiness with ideal TC service.
            margins = [start+i*parameters['compute_ns']-actor['ready'][i]
                for i in range(1, kt)]
            idle = [max(0., actor['starts'][i]-actor['service_frontier'][i-1])
                for i in range(1, kt)]
            ctas.append(dict(cta=cta, ready_margin_ns=margins,
                post_FIRST_idle_ns=idle, L_ns=actor['main_ends'][0]-start,
                first_ns=start, first_ready_ns=actor['ready'][0]))
        rows.append(dict(endpoint=endpoint, exact_price=str(exact), price_ns=price,
            case=case, kt=kt, ctas=ctas, demands=model['input_demand_count'],
            minimum_ready_margin_ns=min(min(c['ready_margin_ns']) for c in ctas),
            maximum_idle_ns=max(max(c['post_FIRST_idle_ns']) for c in ctas),
            maximum_L_floor_difference_ns=max(abs(c['L_ns']-kt*parameters['compute_ns']) for c in ctas)))
        source_granules.cache_clear()
        print(json.dumps({k:v for k,v in rows[-1].items() if k!='ctas'}), flush=True)

for name in ('supply_model.py','analyze_r10_granule_overlap.py'):
    p=code/name;inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(input_sha256=inputs, runs=len(rows),
    first_CTA_windows=sum(len(r['ctas']) for r in rows),
    logical_demands=sum(r['demands'] for r in rows), rows=rows,
    minimum_ready_margin_ns=min(r['minimum_ready_margin_ns'] for r in rows),
    maximum_idle_ns=max(r['maximum_idle_ns'] for r in rows),
    scope='Two fixed endpoints only; no interior continuity, event-order stability, monotonicity, or interval certificate.',
    observation_qualified=False, prediction_qualified=False)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

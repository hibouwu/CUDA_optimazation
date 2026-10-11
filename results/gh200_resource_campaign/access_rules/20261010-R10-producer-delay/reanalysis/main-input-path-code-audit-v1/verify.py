"""Offline snapshot audit: old default identity and independent FIFO formula."""
from fractions import Fraction as Q
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
identity = json.loads((root / 'source-identity.json').read_text())

def module(name):
    path = root / (name + '_supply.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == identity[name + '_sha256']
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

old, new = module('old'), module('new')
rows = []
for stages, counts, sharing, latency in itertools.product(
        (2, 4), ([1, 2, 3], [2, 2, 2]), (False, True), (0., 8.)):
    args = dict(service_ns=4., read_latency_ns=32., compute_ns=64.,
        output_latency_ns=16., producer_gap_ns=2., operand_gap_ns=4.,
        destination_service_ns=2., destination_latency_ns=4.,
        input_min_latency_ns=latency, output_counts=counts, trace_requests=True,
        diagnostic_producer_offsets_ns=[0., 8., 40.],
        share_inflight_sources=sharing)
    if sharing:
        args['source_keys'] = [[[(a, i) for a in range(3)]
            for i in range(4 * count)] for count in counts]
    baseline = old.shared_stage_events(4, stages, 3, **args)
    assert new.shared_stage_events(4, stages, 3, **args) == baseline
    assert new.shared_stage_events(4, stages, 3, input_path_service_ns=0., **args) == baseline
    result = new.shared_stage_events(4, stages, 3, input_path_service_ns=8., **args)
    demands = result['input_demands']
    assert len(demands) == 3 * 4 * sum(counts)
    assert [d['arrival_ns'] for d in demands] == sorted(d['arrival_ns'] for d in demands)
    # Closed FIFO max-plus expression, independent of the implementation's
    # mutable frontier: end_i = max_j(arrival_j + sum_{k=j..i} duration_k).
    prefix = [Q(0)]
    for d in demands:
        prefix.append(prefix[-1] + 8 * Q(d['work']))
    for i, d in enumerate(demands):
        end = max(Q(x['arrival_ns']) + prefix[i + 1] - prefix[j]
            for j, x in enumerate(demands[:i + 1]))
        assert Q(d['input_path_end_ns']) == end
        assert Q(d['input_path_start_ns']) == end - 8 * Q(d['work'])
        assert Q(d['destination_arrival_ns']) == max(
            Q(d['source_ready_ns']) + 4, end + Q(latency))
    assert Q(result['input_path_service']['busy_ns']) == prefix[-1]
    assert result['input_path_service']['work'] == 2 * 4 * sum(counts)
    for actor in result['ctas']:
        for group in range(stages, len(actor['issued_a'])):
            assert actor['issued_a'][group] >= actor['released'][group - stages]
    rows.append(dict(stages=stages, outputs=counts, sharing=sharing,
        minimum_latency_ns=latency, full_old_result_equal=True,
        demand_formula_checks=len(demands), merged_demands=result['input_merged_count']))

report = dict(cases=len(rows), default_full_dictionary_comparisons=2*len(rows),
    independent_demand_checks=sum(r['demand_formula_checks'] for r in rows), rows=rows,
    source_identity=identity, scope='Synthetic implementation audit only; no measured accuracy or hardware attribution.',
    observation_qualified=False, prediction_qualified=False)
(root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_identity')}))

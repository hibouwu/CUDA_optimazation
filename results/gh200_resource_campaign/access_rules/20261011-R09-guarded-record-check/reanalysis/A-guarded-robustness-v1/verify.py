#!/usr/bin/env python3
"""Rebuild joint-boundary graphs and check saved proofs without solving."""
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
from r09_guarded_clock_robustness import bracket_graph, cycle_lower_bound, verify_bracket_witness
from r09_shared_clock import verify_certificate

root = Path(__file__).resolve().parent
with gzip.open(root / 'robustness.json.gz', 'rt') as stream:
    report = json.load(stream)
for name, digest in report['input_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
stamps = {int(sm): points for sm, points in report['original_stamps'].items()}
certificates = points_verified = 0
for case in report['cases']:
    epsilon, cap = case['endpoint_slack_ns'], case['cap']
    previous = Fraction(0)
    for step in case['separating_cycles']:
        graph = bracket_graph(stamps, epsilon, previous, cap)
        assert json.loads(json.dumps(graph)) == step['graph']
        verify_certificate(graph, step['certificate'])
        # Check the cycle's affine bound directly, without the helper.
        ids = step['certificate']['negative_cycle_edge_ids']
        count = sum(edge in graph['cycle_relaxed_edge_ids'] for edge in ids)
        bound = sum(graph['edges'][edge][2] for edge in ids)
        required = previous - Fraction(bound, graph['cycle_scale'] * count)
        assert required == Fraction(*step['necessary_clock_cycles_fraction'])
        assert required == cycle_lower_bound(graph, step['certificate']) and required > previous
        previous = required
        certificates += 1
    minimum = Fraction(*case['minimum_extra_clock_cycles_fraction'])
    assert minimum == previous and case['attained']
    graph = bracket_graph(stamps, epsilon, minimum, cap)
    assert json.loads(json.dumps(graph)) == case['feasible_graph']
    verify_certificate(graph, case['feasible_certificate'])
    verification = verify_bracket_witness(stamps, graph, case['feasible_certificate'])
    assert verification == case['original_bracket_verification']
    points_verified += verification['points_verified']
    certificates += 1
for row in report['containment_frontier']:
    inner, outer = row['inner'], row['outer']
    for window in (inner, outer):
        i, j = window['point_indices']
        a, b = stamps[window['sm']][i], stamps[window['sm']][j]
        assert a == window['begin'] and b == window['end']
        assert window['cycles_lo'] == max(0, b[0] - a[2])
        assert window['cycles_hi'] == b[2] - a[0]
    gap = inner['cycles_lo'] - outer['cycles_hi']
    margin = min(inner['begin_ns'] - outer['begin_ns'], outer['end_ns'] - inner['end_ns'])
    assert gap == row['cycle_gap'] and margin >= 2 * row['endpoint_allowance_ns']
    assert Fraction(gap, 4) == Fraction(str(row['sufficient_cycle_allowance_open_upper']))
assert points_verified == 10560
print(json.dumps(dict(solver_called=False, exact_boundary_cases=20, certificates_verified=certificates,
                     original_bracket_inversions_verified=points_verified, contained_pairs_verified=len(report['containment_frontier']))))

#!/usr/bin/env python3
"""Rebuild saved interval graphs and verify proofs without a solver call."""
import gzip
import hashlib
import json
from pathlib import Path
from r09_shared_clock import clock_graph, verify_certificate

root = Path(__file__).resolve().parent
with gzip.open(root / 'audit.json.gz', 'rt') as stream:
    report = json.load(stream)
for name, digest in report['input_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
stamps = {int(sm): values for sm, values in report['original_stamps'].items()}
certificates = inversions = 0


def check(points, saved, certificate, epsilon, cap):
    global certificates, inversions
    graph = clock_graph({sm: [(before, n) for before, n, after in values]
                         for sm, values in points.items()}, epsilon, cap)
    scale = graph['cap_fraction'][1] if graph['cap_fraction'] else 1
    edge = (len(graph['time_nodes_ns']) - 1) * (2 if cap is not None else 1)
    for sm in sorted(points):
        for before, n, after in points[sm]:
            u, v, unused = graph['edges'][edge]
            graph['edges'][edge] = (u, v, (after - graph['clock_origins'][sm]) * scale)
            edge += 2
    assert json.loads(json.dumps(graph)) == saved
    verify_certificate(graph, certificate)
    certificates += 1
    if not certificate['feasible']:
        return
    times = graph['time_nodes_ns']
    curve = certificate['potential_scaled_cycles']
    positions = {n: index for index, n in enumerate(times)}
    for sm, values in points.items():
        for before, n, after in values:
            relative = n - graph['timer_origin_ns']
            left, right = positions[relative - epsilon], positions[relative + epsilon]
            offset = curve[graph['sm_nodes'][sm]]
            low = (before - graph['clock_origins'][sm]) * scale + offset
            high = (after - graph['clock_origins'][sm]) * scale + offset
            # A continuous monotone piecewise-linear curve crosses this
            # nonempty intersection within the saved endpoint-time bracket.
            assert max(low, curve[left]) <= min(high, curve[right])
            inversions += 1


for branch, result in report['branches'].items():
    cap = 1.98 if branch == 'capped' else None
    upper = result['minimum_integer_additional_endpoint_allowance_ns']
    check(stamps, result['feasible_graph'], result['feasible_certificate'], upper, cap)
    check(stamps, result['infeasible_graph'], result['infeasible_certificate'], upper - 1, cap)
    check(stamps, result['zero_graph'], result['zero_certificate'], 0, cap)
for result in report['independent_local_zero_allowance']:
    sm = result['sm']
    check({sm: stamps[sm]}, result['graph'], result['certificate'], 0, 1.98)
assert certificates == 138 and inversions == 1584
print(json.dumps(dict(solver_called=False, certificates_verified=certificates,
                     original_bracket_intersections_verified=inversions)))

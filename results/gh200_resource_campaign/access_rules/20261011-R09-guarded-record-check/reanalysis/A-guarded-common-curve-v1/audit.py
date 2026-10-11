#!/usr/bin/env python3
"""Conditional common varying curve for the existing guarded smoke brackets."""
from bisect import bisect_left
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

from r09_shared_clock import clock_graph, solve_clock_graph, verify_certificate

out = Path(__file__).resolve().parent
run = out.parents[1]


def bracket_graph(stamps, epsilon, cap):
    """Cb<=offset+F(t)<=Ca for some t in [N-epsilon,N+epsilon]."""
    for points in stamps.values():
        for before, timer, after in points:
            if any(type(value) is not int for value in (before, timer, after)) or after < before:
                raise ValueError('ordered native integer bracket required')
    graph = clock_graph({sm: [(before, timer) for before, timer, after in points]
                         for sm, points in stamps.items()}, epsilon, cap)
    scale = graph['cap_fraction'][1] if graph['cap_fraction'] else 1
    edge = (len(graph['time_nodes_ns']) - 1) * (2 if cap is not None else 1)
    for sm in sorted(stamps):
        for before, timer, after in stamps[sm]:
            u, v, previous = graph['edges'][edge]
            assert previous == (before - graph['clock_origins'][sm]) * scale
            graph['edges'][edge] = (u, v, (after - graph['clock_origins'][sm]) * scale)
            edge += 2
    assert edge == len(graph['edges'])
    return graph


def verify(stamps, graph, certificate):
    verify_certificate(graph, certificate)
    if not certificate['feasible']:
        return dict(negative_closed_cycle_verified=True)
    times, potentials = graph['time_nodes_ns'], certificate['potential_scaled_cycles']
    curve = potentials[:len(times)]
    scale = graph['cap_fraction'][1] if graph['cap_fraction'] else 1
    cap_num = graph['cap_fraction'][0] if graph['cap_fraction'] else None
    time_index = {value: index for index, value in enumerate(times)}
    maximum_error = Fraction(0)
    count = 0
    for sm, points in stamps.items():
        for before, timer, after in points:
            n = timer - graph['timer_origin_ns']
            epsilon = graph['endpoint_slack_ns']
            left, right = time_index[n - epsilon], time_index[n + epsilon]
            offset = potentials[graph['sm_nodes'][sm]]
            low = (before - graph['clock_origins'][sm]) * scale + offset
            high = (after - graph['clock_origins'][sm]) * scale + offset
            target = max(low, curve[left])
            assert target <= min(high, curve[right])
            index = bisect_left(curve, target, left, right + 1)
            if curve[index] == target:
                adjusted_time = Fraction(times[index])
            else:
                adjusted_time = times[index - 1] + Fraction(target - curve[index - 1], curve[index] - curve[index - 1]) * (times[index] - times[index - 1])
            assert low <= target <= high and abs(adjusted_time - n) <= epsilon
            maximum_error = max(maximum_error, abs(adjusted_time - n))
            count += 1
    for i in range(len(times) - 1):
        delta = curve[i + 1] - curve[i]
        assert delta >= 0
        if cap_num is not None:
            assert delta <= cap_num * (times[i + 1] - times[i])
    return dict(original_brackets_inverted=count, maximum_constructed_additional_timer_shift_ns=float(maximum_error))


def minimum(stamps, cap):
    def solve(epsilon):
        graph = bracket_graph(stamps, epsilon, cap)
        return graph, solve_clock_graph(graph)
    zero_graph, zero = solve(0)
    if zero['feasible']:
        upper, upper_graph, upper_proof = 0, zero_graph, zero
        lower_graph = lower_proof = None
    else:
        lower, upper = 0, 1
        lower_graph, lower_proof = zero_graph, zero
        while True:
            upper_graph, upper_proof = solve(upper)
            if upper_proof['feasible']:
                break
            lower, lower_graph, lower_proof = upper, upper_graph, upper_proof
            upper *= 2
        while upper - lower > 1:
            middle = (upper + lower) // 2
            graph, proof = solve(middle)
            if proof['feasible']:
                upper, upper_graph, upper_proof = middle, graph, proof
            else:
                lower, lower_graph, lower_proof = middle, graph, proof
    result = dict(minimum_integer_additional_endpoint_allowance_ns=upper,
                  real_infimum_bracket_ns=[max(0, upper - 1), upper],
                  real_lower_bound_open=cap is not None and upper > 0,
                  real_bound_kind='attained minimum' if cap is not None else 'infimum; attainment not established',
                  feasible_graph=upper_graph, feasible_certificate=upper_proof,
                  infeasible_graph=lower_graph, infeasible_certificate=lower_proof,
                  zero_graph=zero_graph, zero_certificate=zero)
    result['original_bracket_verification'] = verify(stamps, upper_graph, upper_proof)
    if lower_graph is not None:
        verify(stamps, lower_graph, lower_proof)
    verify(stamps, zero_graph, zero)
    return result


# Exact-width reduction and an analytic interval case check the changed edges.
test = {0: [(10**18, 10**18, 10**18), (10**18 + 200, 10**18 + 100, 10**18 + 200)]}
for epsilon in (0, 32, 50):
    assert bracket_graph(test, epsilon, 1) == clock_graph({0: [(a, n) for a, n, z in test[0]]}, epsilon, 1)
assert minimum(test, 1)['minimum_integer_additional_endpoint_allowance_ns'] == 50
broad = {0: [(10**18, 10**18, 10**18 + 100), (10**18 + 200, 10**18 + 100, 10**18 + 200)]}
assert minimum(broad, 1)['minimum_integer_additional_endpoint_allowance_ns'] == 0

previous = run / 'reanalysis/A-smoke-boundary-audit-v1/audit.json'
bound = json.loads(previous.read_text())
inputs = dict(bound['input_sha256'])
inputs[str(previous)] = hashlib.sha256(previous.read_bytes()).hexdigest()
cap_path = run.parent / '20261010-R09-R13-R18-failure-controls/reanalysis/A-local-clock-consistency-v1/cap-evidence.json'
cap_evidence = json.loads(cap_path.read_text())
assert cap_evidence['clock_intercept_cap_ghz'] == 1.98
inputs[str(cap_path)] = hashlib.sha256(cap_path.read_bytes()).hexdigest()
inputs[cap_evidence['source']] = cap_evidence['sha256']
for path, digest in inputs.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
assert bound['actual_raw_complete_heads_valid'] and bound['actual_packets_valid']
with gzip.open(run / 'returned/raw/guarded_bracket.jsonl.gz', 'rt') as stream:
    events = {packet['event']: packet for packet in map(json.loads, stream)}
trace = events['call']['trace']
stamps = {}
for cta in range(132):
    b = trace[1552 * cta:1552 * (cta + 1)]
    sm = b[2] - 1
    assert sm not in stamps and b[11] == b[12] == b[2]
    stamps[sm] = []
    for role in range(2):
        offset = 16 + 768 * role
        stamps[sm].extend([(b[offset], b[offset + 6], b[offset + 10]),
                           (b[offset + 1], b[offset + 7], b[offset + 11])])
branches = {name: minimum(stamps, cap) for name, cap in [('capped', 1.98), ('uncapped', None)]}
locals_ = []
for sm, points in stamps.items():
    graph = bracket_graph({sm: points}, 0, 1.98)
    certificate = solve_clock_graph(graph)
    assert certificate['feasible']
    locals_.append(dict(sm=sm, graph=graph, certificate=certificate,
                        verification=verify({sm: points}, graph, certificate)))
for path in (Path(__file__), out / 'r09_shared_clock.py'):
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
report = dict(schema='gh200-r09-guarded-common-curve/1', input_sha256=inputs, original_stamps=stamps,
              model='Cb<=offset_sm+F(t)<=Ca; t in [N-epsilon,N+epsilon]; common nonnegative continuous F, independent SM offsets.',
              branches=branches, independent_local_zero_allowance=locals_,
              original_phenomenon_inference='inconclusive', observation_qualified=False,
              limits=['Original guarded smoke only. Hardware sampling enclosure is an assumption, not a verified fact.',
                      'Extra endpoint allowance is a compatibility variable after using cycle brackets, not measured timer precision or the128ns total-window gate.',
                      'The1.98 cap remains the declared old r2 bound; uncapped branch separately tests nonnegative common evolution.',
                      'No raw clock64 subtraction between SMs, no physical frequency identification, target fit, new GPU or local holdout.',
                      'A local curve witness is conditioned on the observed target brackets and cannot be used as a free predictor.'])
target = out / 'audit.json.gz'
if target.exists():
    with gzip.open(target, 'rt') as stream:
        assert json.load(stream) == json.loads(json.dumps(report))
else:
    target.write_bytes(gzip.compress((json.dumps(report, indent=2) + '\n').encode(), mtime=0))
print(json.dumps({name: value['minimum_integer_additional_endpoint_allowance_ns'] for name, value in branches.items()}))
print('132 independent local zero-allowance witnesses;6common certificates;1584 original bracket inversions')

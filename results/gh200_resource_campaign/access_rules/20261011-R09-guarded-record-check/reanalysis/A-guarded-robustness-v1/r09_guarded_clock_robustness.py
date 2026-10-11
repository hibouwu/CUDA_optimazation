#!/usr/bin/env python3
"""Exact joint timer/cycle allowance for conditional guarded clock curves.

No archive IO or prediction. The hardware sampling-enclosure assumption is
relaxed explicitly; allowances are compatibility variables, not measurements.
"""
from bisect import bisect_left
from fractions import Fraction

from r09_shared_clock import clock_graph, solve_clock_graph, verify_certificate


def bracket_graph(stamps, endpoint_slack_ns, extra_clock_cycles, cap=1.98):
    """Cb-kappa <= offset_sm+F(t) <= Ca+kappa, |t-N|<=epsilon.

    epsilon is an integer ns allowance, kappa an exact nonnegative rational
    number of native cycles outside EACH recorded clock bracket. SM offsets
    are free. Only the existing graph solver/certificate checker is reused;
    its exact-point inversion is not valid for these interval constraints.
    """
    if type(extra_clock_cycles) is not int and not isinstance(extra_clock_cycles, Fraction):
        raise ValueError('exact integer or rational cycle allowance required')
    kappa = Fraction(extra_clock_cycles)
    if kappa < 0:
        raise ValueError('nonnegative extra cycle allowance required')
    for points in stamps.values():
        for stamp in points:
            if len(stamp) != 3 or any(type(value) is not int for value in stamp) or stamp[2] < stamp[0]:
                raise ValueError('ordered integer clock/timer/clock bracket required')
    graph = clock_graph({sm: [(before, n) for before, n, after in points] for sm, points in stamps.items()},
                        endpoint_slack_ns, cap)
    base_scale = graph['cap_fraction'][1] if graph['cap_fraction'] else 1
    scale = base_scale * kappa.denominator
    graph['edges'] = [(u, v, bound * kappa.denominator) for u, v, bound in graph['edges']]
    edge = (len(graph['time_nodes_ns']) - 1) * (2 if cap is not None else 1)
    relaxed = []
    for sm in sorted(stamps):
        for before, n, after in stamps[sm]:
            origin = graph['clock_origins'][sm]
            upper = (after - origin + kappa) * scale
            lower = (-before + origin + kappa) * scale
            assert upper.denominator == lower.denominator == 1
            u, v, unused = graph['edges'][edge]
            graph['edges'][edge] = (u, v, int(upper))
            u, v, unused = graph['edges'][edge + 1]
            graph['edges'][edge + 1] = (u, v, int(lower))
            relaxed.extend((edge, edge + 1))
            edge += 2
    assert edge == len(graph['edges'])
    graph.update(cycle_scale=scale, extra_clock_cycles_fraction=[kappa.numerator, kappa.denominator],
                 cycle_relaxed_edge_ids=relaxed)
    return graph


def verify_bracket_witness(stamps, graph, certificate):
    """Verify all original intervals and construct times by exact inversion."""
    verify_certificate(graph, certificate)
    if not certificate['feasible']:
        raise ValueError('feasible certificate required for inversion')
    times = graph['time_nodes_ns']
    potential = certificate['potential_scaled_cycles']
    curve = potential[:len(times)]
    positions = {n: index for index, n in enumerate(times)}
    scale = graph['cycle_scale']
    kappa = Fraction(*graph['extra_clock_cycles_fraction'])
    epsilon = graph['endpoint_slack_ns']
    cap = Fraction(*graph['cap_fraction']) if graph['cap_fraction'] else None
    maximum_time = Fraction(0)
    maximum_clock = Fraction(0)
    count = 0
    for sm, points in stamps.items():
        origin = graph['clock_origins'][sm]
        offset = potential[graph['sm_nodes'][sm]]
        for before, n, after in points:
            relative = n - graph['timer_origin_ns']
            left, right = positions[relative - epsilon], positions[relative + epsilon]
            low = (before - origin - kappa) * scale + offset
            high = (after - origin + kappa) * scale + offset
            target = max(low, curve[left])
            if target > min(high, curve[right]):
                raise ValueError('original cycle/time bracket has empty intersection')
            index = bisect_left(curve, target, left, right + 1)
            if curve[index] == target:
                adjusted = Fraction(times[index])
            else:
                adjusted = times[index - 1] + Fraction(target - curve[index - 1], curve[index] - curve[index - 1]) * (times[index] - times[index - 1])
            native_cycle = Fraction(target - offset, scale) + origin
            clock_error = max(Fraction(0), before - native_cycle, native_cycle - after)
            if abs(adjusted - relative) > epsilon or clock_error > kappa:
                raise ValueError('constructed witness exceeds declared allowances')
            maximum_time = max(maximum_time, abs(adjusted - relative))
            maximum_clock = max(maximum_clock, clock_error)
            count += 1
    for i in range(len(times) - 1):
        dc = Fraction(curve[i + 1] - curve[i], scale)
        if dc < 0 or (cap is not None and dc > cap * (times[i + 1] - times[i])):
            raise ValueError('curve violates nonnegative capped evolution')
    return dict(points_verified=count, maximum_constructed_timer_shift_ns=float(maximum_time),
                maximum_constructed_extra_clock_cycles=float(maximum_clock))


def cycle_lower_bound(graph, certificate):
    """One negative cycle yields an exact necessary lower bound on kappa."""
    verify_certificate(graph, certificate)
    if certificate['feasible']:
        raise ValueError('negative cycle required')
    relaxed = set(graph['cycle_relaxed_edge_ids'])
    ids = certificate['negative_cycle_edge_ids']
    count = sum(edge in relaxed for edge in ids)
    if not count:
        raise ValueError('negative cycle has no cycle-allowance-dependent edges')
    bound = sum(graph['edges'][edge][2] for edge in ids)
    kappa = Fraction(*graph['extra_clock_cycles_fraction'])
    required = kappa - Fraction(bound, graph['cycle_scale'] * count)
    if required <= kappa:
        raise ValueError('separating cycle did not increase the necessary bound')
    return required


def minimum_clock_allowance(stamps, endpoint_slack_ns, cap=1.98):
    """Exact attained minimum kappa for FIXED epsilon, with graph proofs.

    At fixed epsilon the time nodes are fixed and every endpoint edge is
    affine in kappa. A negative cycle supplies a necessary lower bound; solve
    at that rational bound until feasible. The last lower bound plus feasible
    potential proves optimality, even without a rate cap. This is distinct
    from changing epsilon, whose uncapped infimum need not be attained.
    """
    kappa = Fraction(0)
    steps = []
    while True:
        graph = bracket_graph(stamps, endpoint_slack_ns, kappa, cap)
        certificate = solve_clock_graph(graph)
        if certificate['feasible']:
            verification = verify_bracket_witness(stamps, graph, certificate)
            return dict(endpoint_slack_ns=endpoint_slack_ns, cap=cap,
                        minimum_extra_clock_cycles_fraction=[kappa.numerator, kappa.denominator],
                        minimum_extra_clock_cycles=float(kappa), attained=True,
                        separating_cycles=steps, feasible_graph=graph, feasible_certificate=certificate,
                        original_bracket_verification=verification)
        required = cycle_lower_bound(graph, certificate)
        steps.append(dict(graph=graph, certificate=certificate,
                          necessary_clock_cycles_fraction=[required.numerator, required.denominator]))
        kappa = required

#!/usr/bin/env python3
"""Conditional GT-counter anchors and conservative frozen-candidate checks.

This is a C-owned diagnostic adapter, not a public trace reader or time model.
Ordering declarations are caller-supplied assumptions, never hardware proof.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

from model_snapshot.output_model import dependency_bounds


MARKER = re.compile(r'r([01])\.w([01])\.t([0-9]+)\.(FIRST|MAIN)\Z')
UINT64_MAX = (1 << 64) - 1


def canonical_sha256(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def require_uint(value, name, positive=False):
    if type(value) is not int or value < int(positive) or value > UINT64_MAX:
        raise ValueError('unsigned integer required: ' + name)


def maximum_interval(intervals):
    """Maximum of marginal intervals; None denotes an unbounded endpoint."""
    if not intervals:
        raise ValueError('empty endpoint list')
    lower_values = []
    upper_values = []
    upper_unbounded = False
    for interval in intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            raise ValueError('two endpoints required')
        low, high = interval
        for value in (low, high):
            if value is not None and type(value) is not int:
                raise ValueError('counter-domain endpoints must be integers or None')
        if low is not None and high is not None and low > high:
            raise ValueError('reversed interval')
        if low is not None:
            lower_values.append(low)
        if high is None:
            upper_unbounded = True
        else:
            upper_values.append(high)
    return [max(lower_values) if lower_values else None,
            None if upper_unbounded else max(upper_values)]


def project_first_main(graph, contract, packet, shared_origin_gt_counter_ns=None):
    """Bind complete FIRST/MAIN coverage of one two-rank cluster/invocation.

    Sparse refers to lifecycle coverage: no invented output/exit/visibility reads.
    GT values are translated by one shared integer origin before propagation.
    Zero-lag source ordering adds constraints; no service price is estimated.
    """
    if contract.get('schema') != 'C-gt-lifecycle-contract/1' or packet.get('schema') != 'C-gt-lifecycle-packet/1':
        raise ValueError('unsupported contract or packet schema')
    family = contract.get('graph_family')
    if family == 'cfg_c':
        if graph.get('schema') != 'gh200-cfg-c-rank-dependencies/1':
            raise ValueError('unsupported cfg_c source graph')
    elif family == 'cfg_a':
        # The pinned cfg_a constructor predates a schema field. Its exact
        # graph hash is required; do not relabel or mutate the shared graph.
        if graph.get('schema') is not None or 'transfers' not in graph or 'producer_observation_ports' not in graph:
            raise ValueError('unsupported pinned cfg_a source graph')
    else:
        raise ValueError('explicit graph family required')
    if contract.get('graph_sha256') != canonical_sha256(graph):
        raise ValueError('source graph identity differs')
    if packet.get('contract_sha256') != canonical_sha256(contract):
        raise ValueError('frozen contract identity differs')
    for name in ('invocation', 'device_uuid', 'binary_sha256'):
        if not isinstance(contract.get(name), str) or not contract[name]:
            raise ValueError('missing identity: ' + name)
        if packet.get(name) != contract[name]:
            raise ValueError('packet identity differs: ' + name)
    for name in ('binary_sha256', 'ordering_review_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', contract.get(name, '')):
            raise ValueError('missing frozen hash: ' + name)
    if contract.get('actual_capture_order_assumed') is not True or contract.get('monotone_gt_reference_assumed') is not True:
        raise ValueError('explicit conditional sampling assumptions required')
    if packet.get('data_origin') not in ('synthetic_test', 'measured_unqualified'):
        raise ValueError('explicit data origin required')

    expected_nodes = {name for name in graph['nodes'] if MARKER.fullmatch(name)}
    bindings = contract.get('anchor_bindings')
    if not expected_nodes or not isinstance(bindings, dict) or set(bindings) != expected_nodes:
        raise ValueError('exact FIRST/MAIN source-marker coverage required')
    rank_ctas = contract.get('rank_ctas')
    if not isinstance(rank_ctas, list) or len(rank_ctas) != 2 or rank_ctas[0] == rank_ctas[1]:
        raise ValueError('two distinct rank CTA identities required')
    for cta in rank_ctas:
        require_uint(cta, 'rank CTA')
    require_uint(contract.get('sm_count'), 'SM count', positive=True)
    coordinates = {}
    for node, binding in bindings.items():
        rank, wg, ordinal, event = MARKER.fullmatch(node).groups()
        rank, wg, ordinal = int(rank), int(wg), int(ordinal)
        if not isinstance(binding, dict) or set(binding) != {'cta', 'physical_thread', 'coord'}:
            raise ValueError('exact marker binding fields required')
        if type(binding['cta']) is not int or binding['cta'] != rank_ctas[rank]:
            raise ValueError('rank/CTA identity differs')
        if type(binding['physical_thread']) is not int or binding['physical_thread'] != 128*(wg+1):
            raise ValueError('consumer WG/physical-thread identity differs')
        coord = binding['coord']
        if not isinstance(coord, list) or len(coord) != 2:
            raise ValueError('two output coordinates required')
        for value in coord:
            require_uint(value, 'output coordinate')
        key = rank, ordinal
        if key in coordinates and coord != coordinates[key]:
            raise ValueError('same output marker coordinates disagree')
        coordinates[key] = coord

    records = packet.get('records')
    if not isinstance(records, list) or len(records) != len(expected_nodes):
        raise ValueError('record coverage differs')
    anchors = {}
    sm_by_cta = {}
    native_by_thread = {}
    widths = []
    origin = None
    for record in records:
        if not isinstance(record, dict) or set(record) != {'node','invocation','device_uuid','binary_sha256','cta',
                           'physical_thread','coord','sm','gt_before','cycles','gt_after','overflow'}:
            raise ValueError('exact record fields required')
        node = record['node']
        if not isinstance(node, str) or node not in bindings or node in anchors:
            raise ValueError('unknown or duplicate marker')
        for name in ('invocation', 'device_uuid', 'binary_sha256'):
            if record[name] != contract[name]:
                raise ValueError('record identity differs: ' + name)
        for name in ('cta', 'physical_thread', 'coord'):
            if record[name] != bindings[node][name] or type(record[name]) is not type(bindings[node][name]):
                raise ValueError('record binding differs: ' + name)
        # list equality alone would accept True as a coordinate equal to 1.
        for value in record['coord']:
            require_uint(value, 'record output coordinate')
        require_uint(record['sm'], 'SM')
        if record['sm'] >= contract['sm_count']:
            raise ValueError('SM identity outside declared device range')
        for name in ('gt_before', 'cycles', 'gt_after'):
            require_uint(record[name], name, positive=True)
        if record['overflow'] is not False:
            raise ValueError('overflow record is invalid')
        if record['gt_before'] > record['gt_after']:
            raise ValueError('reversed GT bracket')
        cta = record['cta']
        if cta in sm_by_cta and sm_by_cta[cta] != record['sm']:
            raise ValueError('same CTA has inconsistent SM identity')
        sm_by_cta[cta] = record['sm']
        rank, wg, ordinal, event = MARKER.fullmatch(node).groups()
        sequence = 2*int(ordinal) + int(event == 'MAIN')
        key = cta, record['physical_thread']
        native_by_thread.setdefault(key, []).append((sequence, record['cycles']))
        anchors[node] = [record['gt_before'], record['gt_after']]
        widths.append(record['gt_after'] - record['gt_before'])
        origin = record['gt_before'] if origin is None else min(origin, record['gt_before'])
    for values in native_by_thread.values():
        ordered = sorted(values)
        for previous, current in zip(ordered, ordered[1:]):
            if current[1] < previous[1]:
                raise ValueError('same-thread native cycle order regresses')
    if shared_origin_gt_counter_ns is not None:
        require_uint(shared_origin_gt_counter_ns, 'shared GT origin')
        if shared_origin_gt_counter_ns > origin:
            raise ValueError('shared origin must not exceed any anchor before value')
        origin = shared_origin_gt_counter_ns
    for node in anchors:
        anchors[node] = [value-origin for value in anchors[node]]
    bounds = dependency_bounds(graph, anchors=anchors)
    consumer_final = {}
    for rank, cta in enumerate(rank_ctas):
        consumer_final[str(cta)] = maximum_interval([bounds[f'r{rank}.w{wg}.FINAL'] for wg in (0,1)])
    return dict(schema='C-gt-lifecycle-projection/1', graph_sha256=canonical_sha256(graph),
        contract_sha256=canonical_sha256(contract), data_origin=packet['data_origin'],
        invocation=packet['invocation'], device_uuid=packet['device_uuid'], binary_sha256=packet['binary_sha256'],
        ordering_review_sha256=contract['ordering_review_sha256'],
        sampling_order_status='caller_assumption_bound_to_review_hash; not hardware proof',
        time_domain='target_globaltimer_counter', shared_origin_gt_counter_ns=origin,
        anchor_count=len(anchors), max_bracket_width_gt_counter_ns=max(widths), sm_by_cta=sm_by_cta,
        anchors_gt_counter_ns=anchors, bounds_gt_counter_ns=bounds,
        completion_intervals_gt_counter_ns=dict(consumer_FINAL_by_CTA=consumer_final,
            cluster_source_safe_exit=bounds['source_safe_exit'], cluster_cuda_complete=bounds['cuda_complete']),
        no_per_CTA_cuda_endpoint_in_source_graph=True, service_prices_fitted=False,
        qualification=dict(hardware_sampling_enclosure=False, physical_ns_accuracy=False,
                           observation=False, lifecycle_prediction=False, whole_gemm=False))


def last_candidate_certificate(endpoint_intervals, frozen_candidates, endpoint_scope, endpoint_contexts):
    """Sufficient interval certificate only; marginal possibility is a superset.

    Caller supplies the complete declared endpoint universe and its scope. No
    candidate expansion, omission, or equivalence of FINAL/exit/CUDA is implied.
    """
    if endpoint_scope not in ('consumer_FINAL', 'source_safe_exit', 'cuda_complete'):
        raise ValueError('explicit completion scope required')
    if not isinstance(endpoint_intervals, dict) or not endpoint_intervals:
        raise ValueError('complete declared endpoint universe required')
    if not isinstance(endpoint_contexts, dict) or set(endpoint_contexts) != set(endpoint_intervals):
        raise ValueError('exact endpoint domain contexts required')
    context = None
    for declared in endpoint_contexts.values():
        if not isinstance(declared, dict) or set(declared) != {
                'invocation','device_uuid','binary_sha256','shared_origin_gt_counter_ns'}:
            raise ValueError('complete endpoint domain identity required')
        for name in ('invocation','device_uuid'):
            if not isinstance(declared[name], str) or not declared[name]:
                raise ValueError('missing endpoint domain identity')
        if not re.fullmatch(r'[0-9a-f]{64}',declared['binary_sha256']):
            raise ValueError('missing endpoint binary hash')
        require_uint(declared['shared_origin_gt_counter_ns'],'endpoint shared origin')
        if context is not None and declared != context:
            raise ValueError('endpoint invocation/device/binary/origin domains differ')
        context = declared
    for cta, interval in endpoint_intervals.items():
        require_uint(cta, 'endpoint identity')
        maximum_interval([interval])
    if not isinstance(frozen_candidates, list) or not frozen_candidates:
        raise ValueError('nonempty frozen candidate list required')
    for cta in frozen_candidates:
        require_uint(cta, 'candidate identity')
    if len(set(frozen_candidates)) != len(frozen_candidates) or set(frozen_candidates)-set(endpoint_intervals):
        raise ValueError('duplicate or unknown frozen candidate')
    possible = []
    for cta, (low, high) in endpoint_intervals.items():
        other_lows = [value[0] for other,value in endpoint_intervals.items()
                      if other != cta and value[0] is not None]
        if high is None or not other_lows or high >= max(other_lows):
            possible.append(cta)
    possible.sort()
    unique = None
    for cta, (low, high) in endpoint_intervals.items():
        other_uppers = [value[1] for other,value in endpoint_intervals.items() if other != cta]
        if low is not None and all(value is not None and low > value for value in other_uppers):
            unique = cta
    outside_lows = [value[0] for cta,value in endpoint_intervals.items()
                    if cta not in frozen_candidates and value[0] is not None]
    candidate_uppers = [endpoint_intervals[cta][1] for cta in frozen_candidates]
    miss_gap = None
    if outside_lows and all(value is not None for value in candidate_uppers):
        gap = max(outside_lows) - max(candidate_uppers)
        if gap > 0:
            miss_gap = gap
    return dict(schema='C-frozen-last-candidate-certificate/1', endpoint_scope=endpoint_scope, domain_context=dict(context),
        declared_endpoint_universe=sorted(endpoint_intervals), frozen_candidates=list(frozen_candidates),
        possible_by_marginal_bounds=possible, possible_set_is_only_a_superset=True,
        candidate_coverage_certified=set(possible) <= set(frozen_candidates),
        unique_last_certified=unique, guaranteed_candidate_miss_gap_gt_counter_ns=miss_gap,
        max_endpoint_interval_gt_counter_ns=maximum_interval(list(endpoint_intervals.values())),
        assumptions='complete declared universe; valid joint source constraints and conditional marginal bounds',
        qualification=dict(observation=False, critical_CTA=False, lifecycle_prediction=False, whole_gemm=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('graph', 'contract', 'packet', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--shared-origin-gt-counter-ns',type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    graph_bytes = args.graph.read_bytes()
    if args.graph.suffix == '.gz':
        graph_bytes = gzip.decompress(graph_bytes)
    result = project_first_main(json.loads(graph_bytes), json.loads(args.contract.read_bytes()),
                                json.loads(args.packet.read_bytes()),args.shared_origin_gt_counter_ns)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

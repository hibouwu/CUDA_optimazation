#!/usr/bin/env python3
"""Review only alias semantics and aggregate/thread/visibility boundaries."""
from collections import defaultdict, deque
import gzip
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
RUN = BASE.parent.parent
sys.path.insert(0, str(BASE / 'source'))
import analyze_r10_lifecycle_composition as target
import legacy_D_binding as source_D


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ancestors(nodes, target_node):
    found = set()
    queue = deque(nodes[target_node])
    while queue:
        node = queue.popleft()
        if node not in found:
            found.add(node)
            queue.extend(nodes[node])
    return found


def require_order(nodes, before, after):
    if before not in ancestors(nodes, after):
        raise AssertionError(f'Missing required precedence: {before} -> {after}')


def symbolic_extra_order(nodes, before, after):
    """An acyclic extension is only a source-constraint witness, not a time trace."""
    changed = {node: list(parents) for node, parents in nodes.items()}
    changed[after].append(before)
    order = target.topological_order(changed)
    assert order.index(before) < order.index(after)
    return {'before': before, 'after': after, 'acyclic': True,
            'interpretation': 'Symbolic precedence witness only; no time, physical sampling, or GPU feasibility is asserted.'}


def main():
    destination = BASE / 'review.json'
    if destination.exists():
        raise FileExistsError('Review output is immutable; use a new version')
    main_dir = RUN / 'reanalysis/main-lifecycle-composition-v1'
    d_dir = RUN / 'reanalysis/C-20261011-legacy-D-lifecycle-v1/data-final-v3'
    input_file = RUN / 'reanalysis/main-input-composition-v1/required-order.json.gz'
    inputs = {str(path): sha(path) for path in [
        BASE / 'source/analyze_r10_lifecycle_composition.py',
        BASE / 'source/analyze_r10_input_dependency_composition.py',
        BASE / 'source/legacy_D_binding.py', input_file,
        d_dir / 'cfg_b_ldb3000-D-lifecycle-ledger.json.gz', Path(__file__)]}
    assert inputs[str(BASE / 'source/legacy_D_binding.py')] == 'cd122a1ae0dc277e31849a5a7a7d7b66621e8f50dd35789097d3b5ef42fb0178'
    assert inputs[str(input_file)] == 'e094d4f8b9ce4072a75b61c596f6f0a19838a4ddf1f7154a6fd3022f7a6dac3a'
    bound = json.load(gzip.open(d_dir / 'cfg_b_ldb3000-D-lifecycle-ledger.json.gz', 'rt'))['outputs']
    input_nodes = json.load(gzip.open(input_file, 'rt'))['nodes']
    alias_definitions = [
        ['math_order.wait.return_all128', 'mma_order_wait_return_all',
         'Same math-stage wait return maximum over the owning128 threads. Initial/opposite-phase and prior-WG notification retain their original meaning.'],
        ['math_order.arrive_all128', 'mma_order_arrive_all',
         'Same math-stage128 arrival completion after mma() return, before own mma_tail.'],
        ['mma_tail.return_all128', 'mma_tail_return_all',
         'Same owning128-thread tail return maximum after wait0 and last input release emission.'],
        ['epilogue_order.wait.return_all128', 'epilogue_order_wait_return_all',
         'Same epilogue-stage wait return maximum, after own thread tail and prior-WG128 epilogue arrivals.'],
        ['store_tail.return_all128', 'store_tail_return_all',
         'Same128-thread per-output store_tail return maximum; includes issuing-path read0. Does not require nonissuing threads to wait for all other thread returns.'],
        ['epilogue_order.arrive_all128', 'epilogue_order_arrive_all',
         'Same128 epilogue-stage arrival completion, after per-thread store_tail and pingpong state advance.'],
    ]
    cases = []
    for cta in [0,107,108]:
        outputs = [row for row in bound if row['cta'] == cta]
        selected_input = {node: parents for node, parents in input_nodes.items() if node.startswith(f'c{cta}.')}
        d_graph = source_D.make_cta_graph(outputs)
        graph = target.compose_cta(cta, selected_input, d_graph, outputs)
        saved = main_dir / f'cta{cta}-required-lifecycle.json.gz'
        saved_graph = json.load(gzip.open(saved, 'rt'))
        inputs[str(saved)] = sha(saved)
        assert graph == saved_graph, 'Composed target differs from main saved graph'
        aliases = graph['aliases']
        for row in outputs:
            j = row['output_ordinal']
            for d_name, input_name, definition in alias_definitions:
                assert aliases[f'cta{cta}.j{j}.{d_name}'] == f'c{cta}.o{j}.{input_name}'
        assert aliases[f'cta{cta}.input.producer_tail_return'] == f'c{cta}.input_producer_exit'
        assert len(aliases) == 6 * len(outputs) + 1

        # An exact union of the frozen input constraints and renamed C constraints
        # introduces no additional aggregate-to-individual dependency.
        expected_edges = {(parent, child) for child, parents in selected_input.items() for parent in parents}
        expected_edges.update((aliases.get(a,a), aliases.get(b,b)) for a,b,rule in d_graph['edges'])
        actual_edges = {(parent, child) for child, parents in graph['nodes'].items() for parent in parents}
        assert actual_edges == expected_edges
        external_threads = [node['id'] for node in d_graph['nodes']
                            if node['kind'] in ('external_input_port','input_composition_port') and node['id'] not in aliases]
        assert graph['unresolved_thread_input_ports'] == external_threads
        assert all('.t' in name for name in external_threads)
        assert graph['event_times'] is None and graph['service_durations'] is None
        assert graph['qualification'] is False

        nodes = graph['nodes']
        shared = set(aliases.values())
        manufactured = [(a,b) for a,b in actual_edges - {(a,b) for a,b in expected_edges}
                        if a in shared and '.t' in b]
        assert manufactured == []
        checks = []
        for row in outputs:
            j = row['output_ordinal']
            p = f'cta{cta}.j{j}'
            m = f'c{cta}.o{j}'
            for t in range(128):
                require_order(nodes, f'{p}.mma_tail.return.t{t}', f'{p}.epilogue_order.wait.return.t{t}')
                require_order(nodes, f'{p}.store_tail.return.t{t}', f'{p}.epilogue_order.arrive.t{t}')
                require_order(nodes, f'{p}.epilogue_order.arrive.t{t}', f'{m}.epilogue_order_arrive_all')
            checks.append({'output':j, 'thread_tail_wait_and_tail_arrival_preserved':True,
                           'all128_means_max_or_arrival_completion':True})
            tail_ancestors = ancestors(nodes, f'{m}.store_tail_return_all')
            source_ancestors = ancestors(nodes, graph['source_lifetime_requirement'])
            cuda_ancestors = ancestors(nodes, graph['CUDA_completion_requirement'])
            for box in row['D_boxes_source_order']:
                q = box['subtile_ordinal']
                visible = f'{p}.q{q}.global_valid_writes_visible'
                if box['global_valid_write_obligation_required']:
                    assert visible in cuda_ancestors and visible not in source_ancestors and visible not in tail_ancestors
                    assert visible not in aliases and visible not in aliases.values()
                else:
                    assert visible not in nodes
        require_order(nodes, f'c{cta}.input_producer_exit', graph['source_lifetime_requirement'])
        require_order(nodes, graph['source_lifetime_requirement'], graph['CUDA_completion_requirement'])
        # Explicitly extend the graph with orders that a fabricated all-thread
        # barrier, a serialized next-WG MMA, or an early D-visible claim forbids.
        early_arrival = f'cta{cta}.j0.epilogue_order.arrive.t96'
        final_issuer_read = f'cta{cta}.j0.q7.source_read_complete.t0'
        witnesses = [
            symbolic_extra_order(nodes, early_arrival, final_issuer_read),
            symbolic_extra_order(nodes, f'cta{cta}.j2.math_order.wait.return.t96', final_issuer_read),
            symbolic_extra_order(nodes, f'c{cta}.o1.mma_order_wait_return_all', f'c{cta}.o0.store_tail_return_all'),
        ]
        valid_box = next(box for box in outputs[0]['D_boxes_source_order'] if box['global_valid_write_obligation_required'])
        visible = f"cta{cta}.j0.q{valid_box['subtile_ordinal']}.global_valid_writes_visible"
        witnesses.append(symbolic_extra_order(nodes, graph['source_lifetime_requirement'], visible))
        # The all128 exits still need the issuer read even in these witnesses.
        require_order(nodes, final_issuer_read, f'c{cta}.o0.store_tail_return_all')
        require_order(nodes, final_issuer_read, f'c{cta}.o0.epilogue_order_arrive_all')
        cases.append({'cta':cta, 'output_endpoint_checks':checks,
                      'six_aliases_semantically_matched':True,
                      'producer_alias_is_input_lifetime_only':True,
                      'saved_graph_matches_exact_union':True,
                      'unresolved_individual_input_ports_preserved':True,
                      'D_visibility_independent_from_source_exit':True,
                      'symbolic_order_witnesses':witnesses})
    script = BASE / 'source/analyze_r10_lifecycle_composition.py'
    current = Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation/microbench/gh200_resource_campaign/access_rules/analyze_r10_lifecycle_composition.py')
    assert sha(current) == sha(script), 'Target changed while reviewing; rebind to a new version'
    report = {'schema':'C-legacy-lifecycle-alias-review/1',
              'scope':'compose_cta only: six all128 alias meanings, aggregate/thread distinction, input producer exit and independent D-visible obligations. No recount, full source-manifest replay, prices, GPU or new model.',
              'reviewed_script_sha256':sha(script),
              'alias_definitions':alias_definitions,
              'producer_alias_definition':'Logical input producer_tail completion requirement. No equality of physical return timestamps or all384-thread/CUDA completion is asserted.',
              'cases':cases,'findings':[],
              'qualification':{'source_alias_review_passed':True,'physical_joint_execution_verified':False,
                               'observation_qualified':False,'time_prediction_qualified':False,
                               'independent_local_holdout_passed':False},
              'limits':['The union preserves necessary precedence. Individual thread input endpoints remain symbolic; max equalities and hardware timing are not solved.',
                        'Acyclic order extensions are symbolic counterexamples to stronger inferred order, not measured or physically certified executions.',
                        'Main completion remains an external obligation. No finite upper bound or last-finisher prediction follows.'],
              'input_sha256':inputs}
    destination.write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps({'report':str(destination),'reviewed_script_sha256':sha(script),
                      'CTAs_reviewed':[row['cta'] for row in cases], 'findings':[],
                      'qualification':report['qualification']},indent=2))


if __name__ == '__main__':
    main()

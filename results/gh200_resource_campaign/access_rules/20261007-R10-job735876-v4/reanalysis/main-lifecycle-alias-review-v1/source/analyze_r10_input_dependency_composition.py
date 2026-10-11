"""Compose required legacy input ordering; missing service durations remain unknown."""
import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path


def topological_order(nodes):
    children = {node: [] for node in nodes}
    degree = {}
    for node, parents in nodes.items():
        if len(set(parents)) != len(parents):
            raise ValueError('duplicate prerequisite: ' + node)
        degree[node] = len(parents)
        for parent in parents:
            if parent not in nodes:
                raise ValueError('missing prerequisite: ' + parent)
            children[parent].append(node)
    ready = deque(node for node in nodes if not degree[node])
    ordered = []
    while ready:
        node = ready.popleft()
        ordered.append(node)
        for child in children[node]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    if len(ordered) != len(nodes):
        raise ValueError('composed ordering has a cycle')
    return ordered


def compose(ports, lifecycle):
    if ports['schema'] != 'gh200-cfg-b-boundary-completion-ports/1':
        raise ValueError('bound legacy completion ports required')
    if ports['shape'] != [2600, 3000, 2000] or ports['scheduler']['stages'] != 6:
        raise ValueError('this composition is bound to the original R10 geometry')
    counts = ports['scheduler']['output_counts']
    nodes = {}
    fragments = {}
    for fragment in ports['fragments']:
        cta = fragment['cta']
        group = fragment['ring_state']['group']
        if (cta, group) in fragments:
            raise ValueError('duplicate input group')
        fragments[cta, group] = fragment
        for node, parents in fragment['nodes'].items():
            if node in nodes:
                raise ValueError('duplicate input node')
            nodes[node] = list(parents)
    output_ports = []
    producer_exits = []
    consumer_tails = []
    release_records = []
    tail_certificates = []

    def add(node, parents):
        if node in nodes:
            raise ValueError('duplicate added node: ' + node)
        nodes[node] = parents
        return node

    for cta, outputs in enumerate(counts):
        source = lifecycle['lifecycle_by_CTA_output_count'][str(outputs)]
        if len(source['groups']) != 32 * outputs:
            raise ValueError('lifecycle group count differs')
        root = add(f'c{cta}.initialized', [])
        def group_node(group, name):
            return f'c{cta}.g{group}.{name}'
        def output_node(output, name):
            return f'c{cta}.o{output}.{name}'

        for group in range(outputs * 32):
            fragment = fragments[cta, group]
            state = fragment['ring_state']
            if state != {key: source['groups'][group][key] for key in state}:
                raise ValueError('input and consumer ring bindings differ')
            acquire = fragment['external_empty_acquired_port']
            nodes[acquire] = [root if group < 6 else group_node(group - 6, 'empty_complete')]
            if group:
                nodes[acquire] += [group_node(group - 1, atom + '.submit') for atom in ('B0', 'B1')]
            output, q = divmod(group, 32)
            if q == 0:
                predecessors = [root] if output == 0 else [output_node(output - 1, 'mma_order_arrive_all')]
                if output >= 2:
                    predecessors.append(output_node(output - 2, 'epilogue_order_arrive_all'))
                prior = add(output_node(output, 'mma_order_wait_return_all'), predecessors)
            elif q == 1:
                prior = group_node(group - 1, 'commit_all')
            else:
                prior = group_node(group - 2, 'release_emit_all')
            full_wait = add(group_node(group, 'full_wait_return_all'), [fragment['full_output_port'], prior])
            commit = add(group_node(group, 'commit_all'), [full_wait])
            add(group_node(group, 'mma_retired'), [commit])
            if q:
                wait = add(group_node(group, 'wait1_return_all'), [commit, group_node(group - 1, 'mma_retired')])
                add(group_node(group - 1, 'release_emit_all'), [wait])
                add(group_node(group - 1, 'empty_complete'), [group_node(group - 1, 'release_emit_all')])
                release_records.append((cta, group - 1, 'normal_wait1'))
            if q == 31:
                returned = add(output_node(output, 'mma_return_all'), [group_node(group - 1, 'release_emit_all')])
                arrived = add(output_node(output, 'mma_order_arrive_all'), [returned])
                tail_wait = add(output_node(output, 'wait0_return_all'), [arrived, group_node(group, 'mma_retired')])
                emitted = add(group_node(group, 'release_emit_all'), [tail_wait])
                add(group_node(group, 'empty_complete'), [emitted])
                tail_return = add(output_node(output, 'mma_tail_return_all'), [emitted])
                consumer_tails.append(tail_return)
                release_records.append((cta, group, 'mma_tail_wait0'))
                epilogue_predecessors = [tail_return]
                if output:
                    epilogue_predecessors.append(output_node(output - 1, 'epilogue_order_arrive_all'))
                epilogue_wait = add(output_node(output, 'epilogue_order_wait_return_all'), epilogue_predecessors)
                store_tail = add(output_node(output, 'store_tail_return_all'), [epilogue_wait])
                epilogue_arrive = add(output_node(output, 'epilogue_order_arrive_all'), [store_tail])
                output_ports.append(dict(cta=cta, output_ordinal=output, consumer_wg=output % 2,
                    entry=epilogue_wait, unrefined_completion=store_tail, same_WG_permission=epilogue_arrive))
        previous = add(f'c{cta}.producer_tail_entry', [group_node(outputs * 32 - 1, atom + '.submit') for atom in ('B0', 'B1')])
        tail_groups = []
        for step, item in enumerate(source['producer_tail_waits']):
            group = item['release_group']
            tail_groups.append(group)
            previous = add(f'c{cta}.producer_tail_wait{step}_return', [previous, group_node(group, 'empty_complete')])
        producer_exits.append(add(f'c{cta}.input_producer_exit', [previous]))
        tail_certificates.append(dict(cta=cta, groups=tail_groups,
            earlier_groups_required_by_reacquire=list(range(outputs * 32 - 6))))

    ordered = topological_order(nodes)
    if len(release_records) != len(fragments) or len(set((c, g) for c, g, _ in release_records)) != len(fragments):
        raise ValueError('a group has missing or duplicated release ownership')
    return dict(schema='legacy-R10-required-input-order/1', nodes=nodes,
        output_ports=output_ports, input_producer_exits=producer_exits,
        consumer_mma_tail_returns=consumer_tails, producer_exit_certificates=tail_certificates,
        node_count=len(nodes), edge_count=sum(map(len, nodes.values())), topological_nodes=len(ordered),
        releases=len(release_records), service_durations=None,
        event_scope='Consumer *_all events are maxima over all128 participating threads, not a recorded leader timestamp.',
        interpretation='Necessary software ordering only; output interiors and all service durations are unrefined. No zero-lag execution or source-server independence is inferred.',
        kernel_completion=None, global_D_visibility=None, observation_qualified=False, prediction_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', required=True, type=Path)
    parser.add_argument('--lifecycle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    hashes = {}
    def read(path, compressed=False):
        raw = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(gzip.decompress(raw) if compressed else raw)
    completion = read(args.completion / 'B-boundary-completion.json')
    lifecycles = read(args.lifecycle / 'B-boundary-lifecycle.json')
    for report in (completion, lifecycles):
        for name, expected in report['input_sha256'].items():
            actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError('upstream input SHA mismatch: ' + name)
            hashes[name] = actual
    cases = []
    common_graph = None
    for row in completion['cases']:
        ports = read(args.completion / row['port_file'], True)
        if hashes[str((args.completion / row['port_file']).resolve())] != row['port_sha256']:
            raise ValueError('completion artifact SHA mismatch')
        lifecycle = next(case for case in lifecycles['cases'] if case['case'] == row['case'])
        graph = compose(ports, lifecycle)
        if common_graph is None:
            common_graph = graph
        elif graph != common_graph:
            raise ValueError('expected same required ordering across the three original pitches')
        cases.append(dict(case=row['case'], groups=graph['releases'], nodes=graph['node_count'], edges=graph['edge_count']))
    data = gzip.compress(json.dumps(common_graph, separators=(',', ':')).encode(), mtime=0)
    (args.output / 'required-order.json.gz').write_bytes(data)
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = dict(cases=cases, identical_required_order_across_pitches=True,
        graph_sha256=hashlib.sha256(data).hexdigest(), input_sha256=hashes,
        limits=['Graph equality is not timing equality: address-dependent service and cross-CTA sharing remain unpriced.',
                'Store-tail/output order are necessary open interfaces pending the legacy epilogue source binding.',
                'Unknown durations are None, not zero. No full-time, physical endpoint or critical-CTA prediction is emitted.'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(cases))


if __name__ == '__main__':
    main()

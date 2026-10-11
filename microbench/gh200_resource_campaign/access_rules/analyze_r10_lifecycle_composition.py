"""Join archived legacy input/D requirements without assigning event times."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

from analyze_r10_input_dependency_composition import topological_order


def compose_cta(cta, input_nodes, output_graph, outputs):
    aliases = {f'cta{cta}.input.producer_tail_return': f'c{cta}.input_producer_exit'}
    endpoints = {
        'math_order.wait.return_all128': 'mma_order_wait_return_all',
        'math_order.arrive_all128': 'mma_order_arrive_all',
        'mma_tail.return_all128': 'mma_tail_return_all',
        'epilogue_order.wait.return_all128': 'epilogue_order_wait_return_all',
        'store_tail.return_all128': 'store_tail_return_all',
        'epilogue_order.arrive_all128': 'epilogue_order_arrive_all',
    }
    for output in outputs:
        j = output['output_ordinal']
        for output_name, input_name in endpoints.items():
            aliases[f'cta{cta}.j{j}.{output_name}'] = f'c{cta}.o{j}.{input_name}'
    output_names = {node['id'] for node in output_graph['nodes']}
    if not set(aliases) <= output_names or not set(aliases.values()) <= set(input_nodes):
        raise ValueError('missing shared lifecycle endpoint')
    nodes = {node: list(parents) for node, parents in input_nodes.items()}
    for node in output_graph['nodes']:
        name = aliases.get(node['id'], node['id'])
        if name not in nodes:
            nodes[name] = []
        if node['time_price'] is not None:
            raise ValueError('only unpriced output requirements may be joined')
    for before, after, _ in output_graph['edges']:
        before, after = aliases.get(before, before), aliases.get(after, after)
        if before == after:
            raise ValueError('endpoint alias creates a self edge')
        if before not in nodes[after]:
            nodes[after].append(before)
    topological_order(nodes)
    unresolved = [node['id'] for node in output_graph['nodes']
                  if node['kind'] in ('external_input_port', 'input_composition_port')
                  and node['id'] not in aliases]
    return dict(nodes=nodes, aliases=aliases, unresolved_thread_input_ports=unresolved,
        source_lifetime_requirement=f'cta{cta}.source_lifetimes_closed',
        CUDA_completion_requirement=f'cta{cta}.cuda_kernel_completion_requirement',
        event_times=None, service_durations=None, qualification=False)


def check_composition(cta, graph, outputs):
    nodes = graph['nodes']
    def ancestors(node):
        pending = list(nodes[node])
        found = set()
        while pending:
            parent = pending.pop()
            if parent not in found:
                found.add(parent)
                pending.extend(nodes[parent])
        return found
    source_end = ancestors(graph['source_lifetime_requirement'])
    CUDA_end = ancestors(graph['CUDA_completion_requirement'])
    assert f'c{cta}.input_producer_exit' in source_end
    assert source_end <= CUDA_end
    checks = Counter()
    for output in outputs:
        j = output['output_ordinal']
        prefix = f'cta{cta}.j{j}'
        input_prefix = f'c{cta}.o{j}'
        tail = ancestors(input_prefix + '.store_tail_return_all')
        assert input_prefix + '.mma_tail_return_all' in tail
        assert input_prefix + '.mma_tail_return_all' in ancestors(input_prefix + '.epilogue_order_wait_return_all')
        checks['own_input_tail_precedes_output_wait_and_store_tail'] += 1
        for box in output['D_boxes_source_order']:
            q = box['subtile_ordinal']
            for t in range(32):
                read = f'{prefix}.q{q}.source_read_complete.t{t}'
                assert read in tail and read in source_end
                checks['issuing_thread_source_read_paths_closed'] += 1
            visible = f'{prefix}.q{q}.global_valid_writes_visible'
            if box['global_valid_write_obligation_required']:
                assert visible in CUDA_end and visible not in source_end
                assert visible not in tail
                checks['valid_D_visibility_separate_from_source_lifetime'] += 1
            else:
                assert visible not in nodes
                checks['zero_valid_boxes_without_fictitious_valid_write'] += 1
        if j + 1 < len(outputs):
            next_math = ancestors(f'c{cta}.o{j+1}.mma_order_wait_return_all')
            assert input_prefix + '.store_tail_return_all' not in next_math
            checks['next_WG_MMA_not_serialized_after_prior_D_tail'] += 1
            next_epilogue = ancestors(f'c{cta}.o{j+1}.epilogue_order_wait_return_all')
            assert input_prefix + '.epilogue_order_arrive_all' in next_epilogue
            checks['next_WG_epilogue_keeps_prior_all128_arrivals'] += 1
        if j >= 2:
            previous_own = f'c{cta}.o{j-2}.epilogue_order_arrive_all'
            assert previous_own in ancestors(input_prefix + '.mma_order_wait_return_all')
            checks['same_WG_next_MMA_keeps_own_epilogue_permission'] += 1
    return dict(checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--D-data', dest='d_data', type=Path, required=True)
    parser.add_argument('--D-module', dest='d_module', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    hashes = {}
    def read(path, compressed=False):
        data = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return json.loads(gzip.decompress(data) if compressed else data)
    input_summary = read(args.input / 'summary.json')
    input_graph = read(args.input / 'required-order.json.gz', True)
    assert hashes[str((args.input / 'required-order.json.gz').resolve())] == input_summary['graph_sha256']
    output_report = read(args.d_data / 'C-legacy-D-lifecycle.json')
    for report in (input_summary, output_report):
        for name, expected in report['input_sha256'].items():
            actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError('upstream input identity changed: ' + name)
            hashes[name] = actual
    module_sha = hashlib.sha256(args.d_module.read_bytes()).hexdigest()
    expected_module = [value for name, value in output_report['input_sha256'].items() if name.endswith('/analyze.py')]
    if expected_module != [module_sha]:
        raise ValueError('output graph function differs from the bound report')
    hashes[str(args.d_module.resolve())] = module_sha
    spec = importlib.util.spec_from_file_location('legacy_D_binding', args.d_module)
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    shared_outputs = None
    for item in output_report['payloads']:
        payload = read(args.d_data / item['file'], True)
        assert hashes[str((args.d_data / item['file']).resolve())] == item['sha256']
        if shared_outputs is None:
            shared_outputs = payload['outputs']
        elif payload['outputs'] != shared_outputs:
            raise ValueError('D outputs differ across original pitches')
    by_cta = defaultdict(list)
    input_by_cta = defaultdict(dict)
    for output in shared_outputs:
        by_cta[output['cta']].append(output)
    for name, parents in input_graph['nodes'].items():
        cta = int(name.split('.', 1)[0][1:])
        input_by_cta[cta][name] = parents
    if sorted(by_cta) != list(range(132)) or set(by_cta) != set(input_by_cta):
        raise ValueError('incomplete CTA coverage')
    rows, totals, checks = [], Counter(), Counter()
    for cta in sorted(by_cta):
        outputs = by_cta[cta]
        graph = compose_cta(cta, input_by_cta[cta], binding.make_cta_graph(outputs), outputs)
        checks.update(check_composition(cta, graph, outputs))
        encoded = json.dumps(graph, sort_keys=True, separators=(',', ':')).encode()
        counts = dict(nodes=len(graph['nodes']), edges=sum(map(len, graph['nodes'].values())),
            shared_endpoint_aliases=len(graph['aliases']), unresolved_thread_input_ports=len(graph['unresolved_thread_input_ports']))
        totals.update(counts)
        rows.append(dict(cta=cta, outputs=len(outputs), **counts, graph_sha256=hashlib.sha256(encoded).hexdigest()))
        if cta in (0, 107, 108):
            (args.output / f'cta{cta}-required-lifecycle.json.gz').write_bytes(gzip.compress(encoded, mtime=0))
    for path in (Path(__file__), Path(__file__).with_name('analyze_r10_input_dependency_composition.py')):
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(schema='legacy-R10-composed-lifecycle-requirements/1', CTAs=rows,
        totals=dict(totals), source_order_checks=dict(checks), input_sha256=hashes,
        source_lifetime_and_CUDA_requirements_separate=True,
        numerical_timing_support=False, all_event_times_and_services_unknown=True,
        limitations=['Alias only matching all128 aggregate endpoints and input producer exit.',
                    'Thread-level input ports remain symbolic. Aggregation does not manufacture individual thread timestamps.',
                    'Max-envelope constraints are necessary inequalities, not zero-time synchronization or a full joint execution model.',
                    'D visibility and CUDA completion are external unpriced obligations. No finite upper bound or last-CTA prediction is supplied.'])
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(CTAs=len(rows), totals=dict(totals), checks=dict(checks)), indent=2))


if __name__ == '__main__':
    main()

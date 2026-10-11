"""Check reachability consequences of the composed source requirements."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
summary = json.loads((here / 'summary.json').read_text())
for name, expected in summary['input_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
raw = (here / 'required-order.json.gz').read_bytes()
assert hashlib.sha256(raw).hexdigest() == summary['graph_sha256']
graph = json.loads(gzip.decompress(raw))
nodes = graph['nodes']

def ancestors(node):
    pending = list(nodes[node])
    result = set()
    while pending:
        parent = pending.pop()
        if parent not in result:
            result.add(parent)
            pending.extend(nodes[parent])
    return result

counts = Counter(item['cta'] for item in graph['output_ports'])
checks = Counter()
for cta, outputs in sorted(counts.items()):
    total_groups = outputs * 32
    # The producer exit requires all prior empty generations, including those
    # reached only through a later ordinary acquire, not just its final six.
    exit_predecessors = ancestors(f'c{cta}.input_producer_exit')
    for group in range(total_groups):
        assert f'c{cta}.g{group}.empty_complete' in exit_predecessors
        checks['every_used_empty_generation_required_at_producer_exit'] += 1
    for output in range(outputs):
        first = output * 32
        last = first + 31
        # Full and committed-group completion are distinct prerequisites.
        tail_predecessors = ancestors(f'c{cta}.o{output}.mma_tail_return_all')
        assert all(f'c{cta}.g{group}.mma_retired' in tail_predecessors for group in range(first, last + 1))
        checks['own_last_committed_group_completed_before_tail_return'] += 1
        checks['all_output_committed_groups_completed_before_tail_return'] += 32
        if output >= 2:
            predecessors = ancestors(f'c{cta}.o{output}.mma_order_wait_return_all')
            assert f'c{cta}.o{output-2}.epilogue_order_arrive_all' in predecessors
            checks['same_WG_output_reuse_requires_own_epilogue'] += 1
        if output + 1 == outputs:
            continue
        tail = f'c{cta}.g{last}.empty_complete'
        # Next WG can pass the MMA order stage before the prior tail. Its sixth
        # initial acquire still needs that tail's final empty generation.
        assert tail not in ancestors(f'c{cta}.o{output+1}.mma_order_wait_return_all')
        checks['next_MMA_order_does_not_require_prior_tail'] += 1
        for q in range(6):
            predecessors = ancestors(f'c{cta}.g{last+1+q}.empty_acquired')
            assert (tail in predecessors) == (q == 5)
            checks['boundary_acquire_without_prior_tail' if q < 5 else 'boundary_sixth_acquire_requires_prior_tail'] += 1

zero_fill = 0
completion_dir = here.parent / 'B-20261011-boundary-completion-v2'
completion = json.loads((completion_dir / 'B-boundary-completion.json').read_text())
case = completion['cases'][0]
ports = json.loads(gzip.decompress((completion_dir / case['port_file']).read_bytes()))
for fragment in ports['fragments']:
    full_ancestors = ancestors(fragment['full_output_port'])
    for atom in fragment['atoms']:
        if atom['source_valid_bytes'] == 0:
            assert atom['valid_available'] is None and atom['valid_completion'] is None
            assert atom['fill_completion'] in full_ancestors
            assert atom['submit'] in full_ancestors
            zero_fill += 1
assert zero_fill == 672
assert graph['service_durations'] is None
assert graph['kernel_completion'] is None and graph['global_D_visibility'] is None
assert graph['prediction_qualified'] is False and graph['observation_qualified'] is False
result = dict(input_hashes_verified=len(summary['input_sha256']), CTAs=len(counts), outputs=sum(counts.values()),
              checks=dict(checks), zero_source_fill_paths_preserved=zero_fill,
              unknown_services_and_output_endpoints_preserved=True,
              scope='Reachability of required source ordering, not timing feasibility or prediction.')
(here / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

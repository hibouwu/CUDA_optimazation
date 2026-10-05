#!/usr/bin/env python3
"""Read-only, independently enumerated position/coverage/atlas compatibility audit.

No author_draft import; no extractor rerun; no files written. Runtime C++
semantics and arbitrary template instantiations are deliberately not certified.
"""
from __future__ import annotations
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from tree_sitter import Language, Parser
import tree_sitter_cpp

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from reconcile_candidates import top_items

EXPECTED_FILES = {'include/cute/algorithm/' + name + '.hpp' for name in ('clear', 'fill', 'axpby')}
AST_CATEGORIES = {
    'function_definition': 'function', 'namespace_definition': 'namespace',
    'alias_declaration': 'local_declaration', 'declaration': 'local_declaration',
    'lambda_expression': 'lambda', 'call_expression': 'call_expression',
    'compound_literal_expression': 'value_initialization', 'binary_expression': 'operator_expression',
    'assignment_expression': 'operator_expression', 'conditional_expression': 'operator_expression',
    'update_expression': 'operator_expression', 'for_statement': 'control_flow',
    'if_statement': 'control_flow', 'return_statement': 'return_statement',
    'preproc_include': 'include', 'preproc_call': 'pragma',
}


def walk(node):
    yield node
    for child in node.named_children:
        yield from walk(child)


def checked_source(path):
    assert path.startswith('include/'), path
    return (ROOT / 'snapshot' / path).read_bytes()


def check_range(s, raw_field=None):
    raw = checked_source(s['path']); a, b = s['start_byte'], s['end_byte']
    assert 0 <= a < b <= len(raw), s
    assert raw[:a].count(b'\n') + 1 == s['start_line'], s
    assert raw[:b-1].count(b'\n') + 1 == s['end_line'], s
    if raw_field:
        assert raw[a:b].decode() == s[raw_field], s
    return raw[a:b].decode()


def verify(data, ledger=True, atlas=True):
    nodes = {n['id']: n for n in data['nodes']}; edges = {e['id']: e for e in data['edges']}
    assert len(nodes) == len(data['nodes']) and len(edges) == len(data['edges']), 'duplicate identity'
    assert set(data['scope']['paths']) == EXPECTED_FILES
    assert not data['global_completion_claimed'] and not data['runtime_execution_claimed']
    for record in data['scope']['files']:
        raw = checked_source(record['path'])
        assert hashlib.sha256(raw).hexdigest() == record['sha256'], record['path']
        assert len(raw) == record['bytes'] and len(raw.splitlines()) == record['line_count']
    for e in edges.values():
        assert e['source'] in nodes and e['target'] in nodes, e['id']
        assert e.get('evidence'), e['id']
        for ev in e['evidence']:
            lines = checked_source(ev['path']).decode().splitlines()
            assert 1 <= ev['start_line'] <= ev['end_line'] <= len(lines), e['id']
            assert not ev.get('quote') or ev['quote'] in '\n'.join(lines[ev['start_line']-1:ev['end_line']]) + '\n', e['id']
            if 'start_byte' in ev:
                check_range(ev)
        if e['relation'] == 'calls':
            c = e['callsite']; actual = check_range(c, 'raw_source_expression')
            assert actual == e['source_expression'] == c['source_expression'], e['id']
            assert c['caller'] == e['source'], e['id']
            assert e.get('evaluation') == 'potentially_evaluated', e['id']
            assert nodes[e['target']]['kind'] in {'api', 'binding'}, e['id']
    for view in data['views']:
        assert len(set(view['node_ids'])) == len(view['node_ids'])
        assert len(set(view['edge_ids'])) == len(view['edge_ids'])
        assert set(view['node_ids']) <= nodes.keys() and set(view['edge_ids']) <= edges.keys()
        for eid in view['edge_ids']:
            assert {edges[eid]['source'], edges[eid]['target']} <= set(view['node_ids']), (view['id'], eid)
    for n in nodes.values():
        if 'signature_range' in n:
            assert check_range(n['signature_range']).rstrip() == n['signature']
        if 'manual_declaration' in n:
            m = n['manual_declaration']; assert check_range(m['signature_range']).rstrip() == m['raw_signature']
        for key in ('source_declaration', 'source_range'):
            if key in n:
                check_range(n[key], 'raw')
    obligations = data['coverage']['obligations']
    assert len({x['id'] for x in obligations}) == len(obligations)
    assert not data['coverage']['unclassified_source_obligations']
    actual = set(); recorded = set(); calls = set(); function_counts = Counter()
    for o in obligations:
        s = o['source_range']; check_range(s, 'raw')
        assert s['path'] in EXPECTED_FILES and o['status'] == 'source_reviewed'
        assert set(o['graph_refs']) <= (nodes.keys() | edges.keys()), o['id']
        if o['category'] == 'include':
            target = 'include/' + re.search(r'<([^>]+)>', s['raw']).group(1)
            assert o['target_path'] == target and o['relation'] == 'file_dependency'
            assert (ROOT / 'snapshot' / target).is_file()
        if o['syntax_kind'] in AST_CATEGORIES:
            recorded.add((s['path'], s['start_byte'], s['end_byte'], o['syntax_kind']))
        if o['category'] == 'call_expression':
            assert o['graph_refs'], o['id']
            for ref in o['graph_refs']:
                e = edges[ref]
                if e['relation'] == 'calls':
                    c = e['callsite']
                else:
                    assert e['evaluation'] == 'unevaluated_decltype'; c = e['expression_range']
                assert (c['path'], c['start_byte'], c['end_byte']) == (s['path'], s['start_byte'], s['end_byte']), o['id']
    for path in sorted(EXPECTED_FILES):
        raw = checked_source(path)
        # This independently reproduces byte-preserving masks and checks that
        # all physical default values still have separately audited records.
        mask = bytearray(raw)
        for regex in (rb'\bCUTE_HOST_DEVICE\b', rb'\bCUTE_UNROLL\b', rb'(?<=p) = \{\}'):
            for m in re.finditer(regex, raw):
                mask[m.start():m.end()] = b' ' * (m.end() - m.start())
        tree = Parser(Language(tree_sitter_cpp.language())).parse(bytes(mask))
        assert not tree.root_node.has_error, path
        for n in walk(tree.root_node):
            if n.type in AST_CATEGORIES:
                actual.add((path, n.start_byte, n.end_byte, n.type))
            if n.type == 'call_expression':
                calls.add((path, n.start_byte, n.end_byte))
            if n.type == 'function_definition':
                function_counts[path] += 1
        defaults = [m for m in re.finditer(rb'(?<=p = )\{\}', raw)]
        defaults_recorded = [o for o in obligations if o['category'] == 'default_argument' and o['source_range']['path'] == path]
        assert {(m.start(), m.end()) for m in defaults} == {(o['source_range']['start_byte'], o['source_range']['end_byte']) for o in defaults_recorded}
    assert actual == recorded, {'missing': sorted(actual-recorded), 'invented': sorted(recorded-actual)}
    call_inventory = {(o['source_range']['path'], o['source_range']['start_byte'], o['source_range']['end_byte'])
                      for o in obligations if o['category'] == 'call_expression'}
    assert calls == call_inventory and len(calls) == 20, len(calls)
    assert function_counts == {'include/cute/algorithm/clear.hpp': 2, 'include/cute/algorithm/fill.hpp': 4, 'include/cute/algorithm/axpby.hpp': 2}
    assert Counter(o['category'] for o in obligations) == data['coverage']['counts']
    # Per-source call identity must survive same-line repeated x(i)/y(i).
    for name in ('x', 'y'):
        duplicates = [e['callsite'] for e in edges.values() if e['relation'] == 'calls'
                      and e['source_expression'] == name + '(i)' and e['callsite']['start_line'] == 89]
        assert len(duplicates) == 2 and duplicates[0]['start_byte'] != duplicates[1]['start_byte']
    dispatch = [e for e in edges.values() if e['source_expression'] == 'detail::fill(tensor, value, prefer<1>{})' and e['relation'] == 'calls']
    assert len(dispatch) == 2 and dispatch[0]['target'] != dispatch[1]['target']
    assert dispatch[0]['callsite'] == dispatch[1]['callsite'], 'two conditional targets must retain one physical dispatch site'
    assert sum(o['category'] == 'default_argument' for o in obligations) == 2
    assert sum(o['category'] == 'local_declaration' for o in obligations) == 4
    assert sum(o['category'] == 'lambda' for o in obligations) == 1
    fixed = [n for n in nodes.values() if n.get('semantic_class') in {'builtin_int_update', 'fixed_type_value_initialization'}]
    assert len(fixed) == 6
    for n in fixed:
        assert n['resolution'] == 'source_proven' and not n['dependency_parameters'], n['id']
        assert all(e['resolution'] == 'source_proven' for e in edges.values() if e['relation'] == 'evaluates' and e['target'] == n['id'])
    containers = [n for n in nodes.values() if n.get('semantic_class') == 'source_control_container']
    assert len(containers) == 4 and all(n['resolution'] == 'source_proven' for n in containers)
    assert all(e['relation'] != 'calls' for e in edges.values() if e['target'] in {n['id'] for n in containers})
    macros = [o for o in obligations if o['category'] == 'macro_use' and o['source_range']['raw'] == 'CUTE_GCC_UNREACHABLE']
    assert len(macros) == 1 and len(macros[0]['expansion_variants']) == 3
    for variant in macros[0]['expansion_variants']:
        edge = edges[variant['edge_id']]
        assert edge['relation'] == 'expands_to' and edge['target'] == variant['target']
        assert edge['evaluation'] == 'macro_expansion_only_not_runtime_call'
    assert sorted(str(v['expanded_spelling']) for v in macros[0]['expansion_variants']) == ['', 'None', '__builtin_unreachable()']
    # Every global source declaration in the complete three-header scope must
    # be represented; local declarations are not manufactured ledger entries.
    if ledger:
        wanted = {n['declaration_occurrence_id']: n for n in nodes.values() if n.get('declaration_occurrence_id')}
        wanted_namespace = {o['declaration_occurrence_id'] for n in nodes.values() for o in n.get('occurrence_refs', [])}
        seen = set(); in_scope = set(); parameters = []
        for key, o, item in top_items(ROOT / 'data/declarations.json', arrays=('occurrences',)):
            if not item or key != 'occurrences':
                continue
            oid = o['declaration_occurrence_id']
            if o['path'] in EXPECTED_FILES:
                in_scope.add(oid)
                if o['kind'] == 'function':
                    owner = next(n['id'] for n in nodes.values() if n.get('declaration_occurrence_id') == oid)
                    for group in o['template_parameters']:
                        parameters.extend({'owner': owner, 'category': 'template_parameter', **p} for p in group['parameters'])
                    parameters.extend({'owner': owner, 'category': 'function_parameter', **p} for p in o['parameters'])
            if oid in wanted:
                n = wanted[oid]; seen.add(oid)
                assert n['entity_id'] == o['entity_id'] and n['signature'] == o['raw_signature'] and n['qualified_name'] == o['qualified_name']
        assert seen == wanted.keys()
        assert in_scope == {oid for oid, n in wanted.items() if n['path'] in EXPECTED_FILES} | wanted_namespace
        assert len(in_scope) == 12
        expected = sorted(parameters, key=lambda x: (x['owner'], x['category'], x['start_byte']))
        observed = sorted(data['coverage']['parameter_obligations'], key=lambda x: (x['owner'], x['category'], x['start_byte']))
        assert expected == observed
        assert Counter(x['category'] for x in observed) == {'template_parameter': 30, 'function_parameter': 22}
        for p in observed:
            check_range(p, 'raw')
    if atlas:
        from build_atlas import enrich_nodes, validate_evidence, diagram_source
        copied_nodes = copy.deepcopy(nodes); copied_edges = copy.deepcopy(edges); issues = []
        scope = json.loads((ROOT / 'data/scope.json').read_text())
        enrich_nodes(copied_nodes, scope, issues)
        validate_evidence(copied_edges, scope, issues)
        assert not issues, issues
        for view in data['views']:
            rendered = diagram_source(view['title'], view['node_ids'], view['edge_ids'], copied_nodes, copied_edges)
            assert rendered.startswith('@startuml') and rendered.rstrip().endswith('@enduml')
    return {'status': 'PASS', 'scope_files': 3, 'source_global_declaration_occurrences': 12,
            'function_templates': 8, 'call_expression_sites': 20, 'potentially_evaluated_call_sites': 18,
            'unevaluated_call_expression_sites': 2, 'nodes': len(nodes), 'edges': len(edges),
            'views': len(data['views']), 'coverage_counts': data['coverage']['counts'],
            'ledger_verified': ledger, 'atlas_enrichment_and_evidence_verified': atlas,
            'not_verified': ['C++ template instantiation', 'GPU runtime', 'numerical correctness', 'performance']}


if __name__ == '__main__':
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DIRECTORY / 'relations.json'
    data = json.loads(target.read_text())
    print(json.dumps(verify(data), ensure_ascii=False, indent=2))

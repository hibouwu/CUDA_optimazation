"""Source-mapped target spelling branches for three fixed library macro families.

This is not C++ lookup and not a general preprocessor. The include-prefix trace
accounts for visible fixed-source define/undef order, including provider tails
and pragma-once recursion. Unsupported/local mutations remain pending. External
headers are explicit boundaries; command-line macro environments are not run.
"""
from __future__ import annotations

from bisect import bisect_right
import copy
from functools import lru_cache
import hashlib
import importlib.util
import itertools
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys

_lexer_spec = importlib.util.spec_from_file_location('_using_target_expansion_scan_candidates',
                                                   Path(__file__).with_name('scan_candidates.py'))
_lexer = importlib.util.module_from_spec(_lexer_spec)
sys.modules[_lexer_spec.name] = _lexer
_lexer_spec.loader.exec_module(_lexer)
SourceScan, lex = _lexer.SourceScan, _lexer.lex

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = {
    'CUTE_STL_NAMESPACE': ('include/cute/config.hpp',
        '99d1ab9b25680fe19cd804e654ce3e7fcb3b3c03b06b81aff11d65254d296c41', '__CUDACC_RTC__'),
    'CUTLASS_CMATH_NAMESPACE': ('include/cutlass/detail/helper_macros.hpp',
        'be3a5a1a94af0ceb540a8e7343f1cfc0bbace6177bf02446f62b193c377fc3d2', '__CUDA_ARCH__'),
    'CUTLASS_STL_NAMESPACE': ('include/cutlass/platform/platform.h',
        '50b1b50334716ce30894a0816f36c11f6f0fe92ed3cdfe6938d1c8da912e291e', '__CUDACC_RTC__'),
}
TYPE_ROLE = 'target_expression_only_no_cpp_lookup'
INCLUDE_MACRO_PROVIDER = ('include/cutlass/cutlass.h',
    '3f83df4d3164afea3a865a5e93d7c1aff3cfdd7c604319571a94332b4e9ec8da')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _safe_path(path):
    return isinstance(path, str) and bool(path) and not path.startswith('/') and '..' not in PurePosixPath(path).parts


def _default_reader(path):
    if not _safe_path(path):
        return None
    file = ROOT / 'snapshot' / path
    return file.read_bytes() if file.is_file() else None


def _span(path, raw, a, b):
    starts = [0] + [m.end() for m in re.finditer(b'\n', raw)]
    return {'path': path, 'start_byte': a, 'end_byte': b,
            'start_line': bisect_right(starts, a), 'end_line': bisect_right(starts, max(a, b - 1)),
            'raw_sha256': _sha(raw[a:b])}


@lru_cache(maxsize=384)
def _scan(path, raw):
    scan = SourceScan(path, raw)
    scan.scan_preprocessor()
    candidates = {c['candidate_id']: c for c in scan.candidates}
    directives = {d['candidate_id']: d for d in scan.directives}

    def conditions(ids):
        out = []
        for ident in ids:
            branch = candidates[ident]
            directive = directives[branch['directive_id']]
            a, b = directive['byte_range']
            out.append({'expression': branch['predicate'], 'constant_false': branch['constant_false'],
                        'branch_id': ident, 'directive': directive['directive'],
                        'directive_path': path, 'directive_line': directive['line_range'][0],
                        'directive_span': _span(path, raw, a, b),
                        'evaluation_point': {'path': path, 'start_byte': a,
                            'meaning': 'Macro state when this directive is evaluated, not automatically the use-point state'},
                        'directive_source': raw[a:b].decode('utf-8', 'replace')})
        return out

    definitions = {d['directive_id']: d for d in scan.definitions}
    result = []
    for directive in scan.directives:
        a, b = directive['byte_range']
        row = {'command': directive['directive'], 'expression': directive['expression'],
               'source_range': _span(path, raw, a, b), 'conditions': conditions(directive['conditions']),
               'raw': raw[a:b].decode('utf-8', 'replace')}
        tokens = directive['_tokens']
        if row['command'] in {'define', 'undef'} and tokens:
            row['name'] = tokens[0].text
        if directive['candidate_id'] in definitions:
            d = definitions[directive['candidate_id']]
            body = ''.join(t.text for t in d['_tokens'])
            row.update(function_like=d['function_like'], body_tokens=[t.text for t in d['_tokens']],
                       parameter_tokens=d['parameter_tokens'],
                       body_spelling=body, body_range=_span(path, raw, *d['body_byte_range']))
        if row['command'] == 'include' and tokens:
            if len(tokens) == 1 and tokens[0].kind == 'literal' and tokens[0].text.startswith('"'):
                row.update(header=tokens[0].text[1:-1], include_form='quoted')
            elif tokens[0].text == '<' and tokens[-1].text == '>':
                row.update(header=''.join(t.text for t in tokens[1:-1]), include_form='angle')
        result.append(row)
    regions = [(r['byte_range'][0], r['byte_range'][1], conditions(r['conditions'])) for r in scan.regions]
    return {'directives': result, 'regions': regions, 'issues': scan.issues,
            'pragma_once': any(d['command'] == 'pragma' and d['expression'].strip() == 'once'
                               and not d['conditions'] for d in result)}


def _conditions_at(parsed, byte):
    return copy.deepcopy(next((c for a, b, c in parsed['regions'] if a <= byte < b), []))


def _strip_outer(text):
    while text.startswith('(') and text.endswith(')'):
        level = 0
        for i, char in enumerate(text):
            level += (char == '(') - (char == ')')
            if level == 0:
                break
        if i != len(text) - 1:
            break
        text = text[1:-1]
    return text


def _defined_atom(expression):
    expression = re.sub(r'\s+', '', expression)
    positive = True
    while True:
        expression = _strip_outer(expression)
        if not expression.startswith('!'):
            break
        positive = not positive
        expression = expression[1:]
    match = re.fullmatch(r'defined\(?([A-Za-z_]\w*)\)?', expression)
    return (match[1], positive) if match else None


def _combine(groups):
    result, seen, values = [], set(), {}
    for group in groups:
        for condition in group:
            atom = _defined_atom(condition['expression'])
            # These built-in configuration atoms may be correlated only while
            # the trace proves no source define/undef changes them. Other
            # guards (notably !defined(CUTLASS_STL_NAMESPACE)) describe their
            # directive-entry state, which its own body can immediately change.
            if atom and atom[0] in {'__CUDACC_RTC__', '__CUDA_ARCH__'}:
                name, value = atom
                if name in values and values[name] != value:
                    return None
                values[name] = value
            key = (condition.get('directive_path'), condition.get('directive_line'), condition['expression'])
            if key not in seen:
                result.append(copy.deepcopy(condition))
                seen.add(key)
    return result


def _trace(path, source, before, names, reader):
    """Walk actual textual include order; no definition after the use is read."""
    files, reads, events, boundaries, problems = {}, {path: source}, [], [], []
    once_seen, active = set(), set()

    def read(candidate):
        if not _safe_path(candidate):
            return None
        if candidate not in reads:
            try:
                value = reader(candidate)
            except OSError as error:
                problems.append({'reason': 'source_reader_error', 'path': candidate, 'error': str(error)})
                value = None
            if value is not None and not isinstance(value, bytes):
                raise TypeError('source_reader must return bytes or None')
            reads[candidate] = value
        return reads[candidate]

    def visit(current, raw, limit, chain, parent_conditions):
        if len(files) >= 512 and current not in files:
            problems.append({'reason': 'include_trace_budget_exceeded', 'path': current})
            return
        parsed = _scan(current, raw)
        if parsed['pragma_once'] and current in once_seen:
            return
        if current in active:
            if parsed['pragma_once']:
                return  # Same active inclusion, not a claim about other configurations.
            problems.append({'reason': 'unproved_include_cycle', 'path': current, 'include_chain': chain})
            return
        if parsed['pragma_once'] and not parent_conditions:
            once_seen.add(current)
        active.add(current)
        files[current] = {'path': current, 'sha256': _sha(raw), 'prefix_end_byte': limit,
                          'pragma_once': parsed['pragma_once']}
        if parsed['issues']:
            problems.append({'reason': 'preprocessor_lexical_review_required', 'path': current,
                             'issues': copy.deepcopy(parsed['issues'])})
        for directive in parsed['directives']:
            if directive['source_range']['start_byte'] >= limit:
                break
            conditions = _combine([parent_conditions, directive['conditions']])
            if conditions is None:
                continue
            inactive = any(c.get('constant_false') for c in conditions)
            command = directive['command']
            if command in {'define', 'undef'} and directive.get('name') in names:
                events.append({**copy.deepcopy(directive), 'conditions': conditions,
                               'include_conditions': copy.deepcopy(parent_conditions),
                               'include_chain': copy.deepcopy(chain), 'source_sha256': _sha(raw),
                               'inactive': inactive, 'sequence_index': len(events)})
            if command not in {'include', 'include_next', 'import'} or inactive:
                continue
            if command != 'include':
                problems.append({'reason': 'unsupported_include_directive', 'evidence': directive})
                continue
            header = directive.get('header')
            if not header:
                # Spelling alone is insufficient: a local override can turn
                # CUDA_STD_HEADER(...) into a fixed-library mutating include.
                visible = [e for e in events if e['name'] == 'CUDA_STD_HEADER' and not e['inactive']]
                fixed = len(visible) == 1 and all(
                    e['command'] == 'define' and not e['conditions'] and e.get('function_like')
                    and e.get('parameter_tokens') == ['header'] and e.get('body_spelling') == '<cuda/std/header>'
                    and e['source_range']['path'] == INCLUDE_MACRO_PROVIDER[0]
                    and e['source_sha256'] == INCLUDE_MACRO_PROVIDER[1] for e in visible)
                kind = ('external_generated_standard_header' if fixed and re.fullmatch(
                    r'CUDA_STD_HEADER\s*\(\s*[A-Za-z_][A-Za-z_0-9./]*\s*\)', directive['expression'])
                    else 'unresolved_generated_include')
                item = {'classification': kind, 'evidence': copy.deepcopy(directive),
                        'include_macro_definition_evidence': copy.deepcopy(visible)}
                boundaries.append(item)
                if kind == 'unresolved_generated_include':
                    problems.append({'reason': kind, 'evidence': directive})
                continue
            candidates = ['include/' + header]
            if directive.get('include_form') == 'quoted':
                candidates.insert(0, posixpath.normpath(str(PurePosixPath(current).parent / header)))
            resolved = next(((candidate, value) for candidate in dict.fromkeys(candidates)
                             if (value := read(candidate)) is not None), None)
            if resolved is None:
                item = {'classification': 'missing_library_source' if header.startswith(('cute/', 'cutlass/'))
                        else 'external_header', 'header': header, 'evidence': copy.deepcopy(directive)}
                boundaries.append(item)
                if item['classification'] == 'missing_library_source':
                    problems.append({'reason': 'missing_library_include_source', 'evidence': directive})
                continue
            target, child = resolved
            edge = {'path': current, 'target_path': target, 'source_range': directive['source_range'],
                    'source_expression': directive['expression'], 'conditions': conditions}
            visit(target, child, len(child), chain + [edge], conditions)
        active.remove(current)

    visit(path, source, before, [], [])
    return {'events': events, 'files': list(files.values()), 'boundaries': boundaries, 'problems': problems}


def _pending(reason, name, evidence=()):
    return {'validation_status': 'pending', 'expanded_replacement': None, 'macro_name': name,
            'conditions': [], 'reason': reason, 'macro_definition_chain': list(evidence),
            'missing_external_bindings': []}


def _alternatives(name, trace):
    active = [e for e in trace['events'] if e.get('name') == name and not e['inactive']]
    if name not in PROVIDERS:
        return [_pending('unsupported_target_macro', name, active)] if active else None
    path, expected_sha, config = PROVIDERS[name]
    if not active:
        return [_pending('provider_not_proven_before_target', name)]
    mutations = [e for e in active if e['command'] != 'define' or e['source_range']['path'] != path]
    if mutations:
        reason = 'macro_undefined_or_redefined_in_include_prefix'
        return [_pending(reason, name, active)]
    if any(e['source_sha256'] != expected_sha for e in active):
        return [_pending('fixed_provider_source_changed', name, active)]
    if any(e['include_conditions'] for e in active):
        return [_pending('conditional_provider_visibility', name, active)]
    bodies = {'': True, 'std': False} if name == 'CUTLASS_CMATH_NAMESPACE' else {'cuda::std': True, 'std': False}
    if len(active) != 2 or {e['body_spelling'] for e in active} != set(bodies) or any(e['function_like'] for e in active):
        return [_pending('unexpected_fixed_macro_definition_pair', name, active)]
    alternatives = []
    for event in active:
        expected = {(config, bodies[event['body_spelling']])}
        if name == 'CUTLASS_STL_NAMESPACE':
            expected.add((name, False))
        if {_defined_atom(c['expression']) for c in event['conditions']} != expected:
            return [_pending('fixed_macro_guard_evidence_changed', name, active)]
        alternatives.append({'validation_status': 'source_proven', 'macro_name': name,
                             'expanded_replacement': event['body_spelling'], 'conditions': event['conditions'],
                             'macro_definition_chain': [event], 'missing_external_bindings': []})
    if name == 'CUTLASS_STL_NAMESPACE':
        guard = next(c for c in active[0]['conditions'] if _defined_atom(c['expression']) == (name, False))
        positive = {**guard, 'expression': 'defined(CUTLASS_STL_NAMESPACE)',
                    'branch_id': 'derived-complement:' + guard['branch_id'],
                    'derived_from_branch_id': guard['branch_id'],
                    'derived_from': 'Complement of the fixed provider external-override guard'}
        alternatives.append({'validation_status': 'pending', 'macro_name': name,
                             'expanded_replacement': None, 'conditions': [positive],
                             'reason': 'external_override_binding_required', 'macro_definition_chain': active,
                             'missing_external_bindings': [{'macro_name': name,
                                 'required': 'Externally supplied replacement tokens and their use-point rescanning'}]})
    return alternatives


def _render(path, source, a, b, tokens, choices):
    pieces, mapping, cursor, virtual = [], [], a, 0

    def append(start, end, replacement=None, macro=None):
        nonlocal virtual
        content = source[start:end] if replacement is None else replacement.encode()
        pieces.append(content)
        mapping.append({'virtual_start_byte': virtual, 'virtual_end_byte': virtual + len(content),
                        'source_range': _span(path, source, start, end),
                        'mapping': 'macro_token_replacement' if macro else 'exact_source_slice',
                        **({'macro_name': macro, 'expanded_replacement': replacement} if macro else {})})
        virtual += len(content)

    for token in tokens:
        if token.kind != 'identifier' or token.text not in choices:
            continue
        if cursor < token.start:
            append(cursor, token.start)
        append(token.start, token.end, choices[token.text], token.text)
        cursor = token.end
    if cursor < b:
        append(cursor, b)
    return b''.join(pieces).decode('utf-8', 'replace'), mapping


def expand_using_target(path, source: bytes, target_span, source_reader=None):
    """Return physical-source-backed expansion spellings, never target entities.

    target_span is a half-open physical byte range in source (optional path and
    line fields are checked). source_reader takes a repository-relative path and
    returns bytes or None; omitting it reads the immutable snapshot directory.
    Known override uncertainty is an explicit pending variant, not a silently
    missing third branch. No filesystem writes occur.
    """
    if not _safe_path(path) or not isinstance(source, bytes) or not isinstance(target_span, dict):
        raise ValueError('Expected a safe source path, bytes and physical target_span')
    a, b = target_span.get('start_byte'), target_span.get('end_byte')
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(source):
        raise ValueError('Invalid physical target byte range')
    span = _span(path, source, a, b)
    if any(k in target_span and target_span[k] != span[k] for k in ('path', 'start_line', 'end_line', 'raw_sha256')):
        raise ValueError('Target range metadata disagrees with physical source')
    tokens, _, lexical_issues = lex(source)
    selected = [t for t in tokens if a <= t.start and t.end <= b]
    if not selected or any(t.start < a < t.end or t.start < b < t.end for t in tokens):
        raise ValueError('Target range must contain whole source tokens')
    parsed = _scan(path, source)
    if any(d['source_range']['start_byte'] <= a < d['source_range']['end_byte'] for d in parsed['directives']):
        raise ValueError('A target expression cannot be a preprocessor directive')
    source_conditions = _conditions_at(parsed, a)
    names = {t.text for t in selected if t.kind == 'identifier'}
    source_atoms = {atom[0] for c in source_conditions if (atom := _defined_atom(c['expression']))}
    controls = {'__CUDACC_RTC__', '__CUDA_ARCH__'} | source_atoms
    trace = _trace(path, source, a, names | {'std', 'cuda', 'CUDA_STD_HEADER'} | controls, source_reader or _default_reader)
    changed_controls = [e for e in trace['events'] if e['name'] in controls and not e['inactive']]
    if changed_controls:
        trace['problems'].append({'reason': 'condition_macro_state_changed_in_source_prefix',
                                  'evidence': changed_controls})
    alternatives, pending = {}, []
    for name in sorted(names):
        values = _alternatives(name, trace)
        if values is not None:
            alternatives[name] = values
    # Namespace replacement tokens themselves must not hide another known macro.
    if alternatives and any(e['name'] in {'std', 'cuda'} and not e['inactive'] for e in trace['events']):
        trace['problems'].append({'reason': 'replacement_token_requires_macro_rescan',
                                  'evidence': [e for e in trace['events'] if e['name'] in {'std', 'cuda'}]})
    if lexical_issues:
        trace['problems'].append({'reason': 'source_lexical_review_required', 'issues': lexical_issues})
    variants, excluded = [], []
    combinations = itertools.product(*alternatives.values()) if alternatives else [()]
    for index, combination in enumerate(combinations):
        if index >= 64:
            pending.append({'reason': 'variant_budget_exceeded', 'category': 'source_contract_pending'}); break
        conditions = _combine([source_conditions, *(c['conditions'] for c in combination)])
        if conditions is None:
            excluded.append({'reason': 'contradictory_defined_macro_conditions',
                             'conditions': [source_conditions, *(c['conditions'] for c in combination)]})
            continue
        failures = [c for c in combination if c['validation_status'] == 'pending']
        failures += trace['problems']
        contract_status = ('parameterized_external_binding' if failures and all(
            c.get('reason') == 'external_override_binding_required' for c in failures)
            else 'source_contract_pending' if failures else 'source_proven')
        definitions = [d for choice in combination for d in choice['macro_definition_chain']]
        replacement = {c['macro_name']: c['expanded_replacement'] for c in combination}
        text, mapping = (None, []) if failures else _render(path, source, a, b, selected, replacement)
        words = [t.text for t in lex(text.encode())[0]] if text is not None else []
        identity = repr((path, a, b, _sha(source[a:b]), conditions, replacement)).encode()
        variant = {'variant_id': 'using-target:' + _sha(identity)[:24], 'conditions': conditions,
                   'expanded_spelling': text, 'expanded_root_qualified': words[:2] == [':', ':'] if text is not None else None,
                   'validation_status': 'pending' if failures else 'source_proven',
                   'source_contract_status': contract_status,
                   'macro_definition_chain': definitions, 'macro_bindings': replacement,
                   'source_mapping': mapping, 'missing_external_bindings': [x for c in combination for x in c['missing_external_bindings']],
                   'inactive': any(c.get('constant_false') for c in conditions),
                   'condition_satisfiability': 'Not solved generally; only literal false and stable defined-macro contradictions are recognized',
                   'pending_reasons': [c.get('reason', 'macro_review_required') for c in failures]}
        variants.append(variant)
        pending.extend({'variant_id': variant['variant_id'], 'reason': reason, 'category': contract_status}
                       for reason in variant['pending_reasons'])
    diagnostics = [dict(p, source_range=span) for p in pending if p['category'] == 'source_contract_pending']
    result = {'schema_version': 1, 'role': TYPE_ROLE, 'source_expression': source[a:b].decode('utf-8', 'replace'),
            'source_range': span, 'source_preprocessor_conditions': source_conditions,
            'root_qualified_in_source': [t.text for t in selected[:2]] == [':', ':'],
            'variants': variants, 'pending': pending, 'diagnostics': diagnostics,
            'source_contract_status': 'source_contract_pending' if diagnostics else
                'source_proven_with_parameterized_external_binding' if pending else 'source_proven',
            'excluded_condition_combinations': excluded,
            'proof': trace, 'completeness': {
                'all_source_branches_accounted_for': not trace['problems'] and all(
                    c.get('reason') in (None, 'external_override_binding_required') for values in alternatives.values() for c in values),
                'all_expansion_spellings_proven': bool(variants) and not pending,
                'cpp_lookup_performed': False,
                'external_environment_boundary': 'Only supplied/fixed source is traced; external headers and command-line macros are not executed. Explicit override guards remain variants.'}}
    return copy.deepcopy(result)  # Cached scans cannot be mutated by consumers.

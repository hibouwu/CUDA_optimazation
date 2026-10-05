#!/usr/bin/env python3
"""Physical using syntax for the fixed CUTLASS snapshot, without C++ lookup.

No owner, target kind/entity, inheritance or final accessibility is inferred.
The established source lexer preserves original byte offsets across line
splicing and hides comment/literal contents. This module does not read files,
expand macros, run the extractor or mutate the candidate denominator.
"""
from bisect import bisect_right
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

_LEXER_PATH = Path(__file__).resolve().with_name('scan_candidates.py')
_LEXER_NAME = '_using_syntax_source_lexer_' + hashlib.sha256(str(_LEXER_PATH).encode()).hexdigest()[:12]
_LEXER_SPEC = importlib.util.spec_from_file_location(_LEXER_NAME, _LEXER_PATH)
_LEXER = importlib.util.module_from_spec(_LEXER_SPEC)
sys.modules[_LEXER_NAME] = _LEXER  # dataclass annotations need their defining module
_LEXER_SPEC.loader.exec_module(_LEXER)
lex = _LEXER.lex

SCHEMA_VERSION = 1
TYPE_SPECIFIER_WORDS = {
    'void', 'bool', 'char', 'char8_t', 'char16_t', 'char32_t', 'wchar_t',
    'short', 'int', 'long', 'signed', 'unsigned', 'float', 'double', 'auto',
    'decltype', 'const', 'volatile', 'typename', 'struct', 'class', 'union', 'enum',
}


def _hash(data):
    return hashlib.sha256(data).hexdigest()


class _Source:
    def __init__(self, path, source, commit):
        if not isinstance(source, bytes):
            raise TypeError('source must be original bytes')
        self.path, self.source, self.commit = path, source, commit
        self.lines = [0] + [m.end() for m in re.finditer(b'\n', source)]
        self.tokens, self.comments, self.lexical_issues = lex(source)
        self.diagnostics = []

    def text(self, start, end):
        return self.source[start:end].decode('utf-8', 'replace')

    def span(self, start, end):
        return {'path': self.path, 'start_byte': start, 'end_byte': end,
                'start_line': bisect_right(self.lines, start),
                'end_line': bisect_right(self.lines, max(start, end - 1))}

    def token_span(self, tokens):
        return self.span(tokens[0].start, tokens[-1].end) if tokens else None

    def token_evidence(self, token):
        return self.span(token.start, token.end) | {
            'raw': self.text(token.start, token.end), 'logical_token': token.text}

    def identity(self, family, start, end):
        # kind/terminal name/owner/target binding are intentionally absent.
        payload = [SCHEMA_VERSION, family, self.commit, self.path, start, end,
                   _hash(self.source[start:end])]
        return ('using_src_' if family == 'written_using' else 'namespace_alias_src_') + _hash(
            json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode())[:24]

    def diagnostic(self, category, start, end, *, source_id=None, message):
        item = {'category': category, 'source_id': source_id, 'message': message,
                'blocks_using_source_contract': True, **self.span(start, end)}
        item['diagnostic_id'] = 'using_diag_' + _hash(json.dumps(
            item, ensure_ascii=False, sort_keys=True).encode())[:24]
        self.diagnostics.append(item)
        return item['diagnostic_id']

    def statement_end(self, start_index):
        """Find the physical terminator, ignoring semicolons inside groups.

        Braced/lambda type expressions in alias RHS may contain semicolons.
        Angle brackets do not shelter statement terminators: comparisons and
        shift operators are deliberately not a generic template parser here.
        """
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        for index in range(start_index + 1, len(self.tokens)):
            word = self.tokens[index].text
            if word in ('(', '[', '{'):
                stack.append(word)
            elif word in pairs:
                if stack and stack[-1] == pairs[word]:
                    stack.pop()
                elif not stack:
                    # Do not consume the rest of a containing class/function
                    # after a missing semicolon. This remains a source error.
                    return index, False, 'unbalanced_statement_group'
                else:
                    return index, False, 'mismatched_statement_group'
            elif word == ';' and not stack:
                return index, True, None
        return len(self.tokens), False, 'unterminated_using_statement'

    def statement_record(self, family, keyword_index, end_index, terminated):
        begin = self.tokens[keyword_index].start
        end = self.tokens[end_index].end if terminated else (
            self.tokens[end_index].start if end_index < len(self.tokens) else len(self.source))
        ident = self.identity(family, begin, end)
        return {'source_using_id' if family == 'written_using' else 'source_namespace_alias_id': ident,
                'family': family, 'commit': self.commit, 'path': self.path,
                'source_range': self.span(begin, end), 'raw_signature': self.text(begin, end),
                'raw_sha256': _hash(self.source[begin:end]),
                'keyword_range': self.token_evidence(self.tokens[keyword_index]),
                'terminator_range': self.token_evidence(self.tokens[end_index]) if terminated else None,
                'syntax_status': 'source_shape_classified' if terminated else 'malformed',
                'diagnostic_refs': [], 'declared_name': None, 'declared_name_range': None,
                'attribute_ranges': [], 'typename_range': None,
                'target_source_expression': None, 'target_source_range': None,
                'target_token_spellings': [], 'target_name_source_expression': None,
                'target_name_source_range': None, 'leading_global_scope_range': None,
                'qualifier_segments': [], 'scope_separator_ranges': [],
                'terminal_name': None, 'terminal_name_range': None,
                'terminal_kind': None, 'constructor_spelling_candidate': False,
                'constructor_candidate_evidence': None,
                'semantic_resolution': 'not_attempted_source_syntax_only'}

    def attribute_after_name(self, tokens):
        index = 1
        attributes = []
        while [t.text for t in tokens[index:index + 2]] == ['[', '[']:
            start = index
            depth = 0
            while index < len(tokens):
                word = tokens[index].text
                depth += (word == '[') - (word == ']')
                index += 1
                if depth == 0:
                    break
            if depth:
                return index, attributes, False
            attributes.append(self.token_span(tokens[start:index]))
        return index, attributes, True

    def split_qualified(self, tokens):
        """Split only outer ::; template/function/array interiors stay whole.

        Qualified-id paths are the fixed import/directive forms. A general
        alias type expression may not be a qualified-id and returns a normal
        non-name result, preserving its complete RHS elsewhere.
        """
        if not tokens:
            return None, 'empty_target'
        leading = tokens[:2] if [t.text for t in tokens[:2]] == [':', ':'] else []
        cursor = 2 if leading else 0
        begin = cursor
        angle = 0
        groups = []
        parts, separators = [], []
        matching = {')': '(', ']': '[', '}': '{'}
        while cursor < len(tokens):
            word = tokens[cursor].text
            if not groups and angle == 0 and tokens[begin].text == 'operator':
                # The operator-id is the terminal component. Its '<', '>' or
                # comma spellings are not template brackets/list separators.
                cursor = len(tokens)
                break
            if word in ('(', '[', '{'):
                groups.append(word)
            elif word in matching:
                if not groups or groups[-1] != matching[word]:
                    return None, 'unbalanced_qualified_target'
                groups.pop()
            elif not groups:
                if word == '<':
                    angle += 1
                elif word == '>':
                    if not angle:
                        return None, 'non_qualified_type_expression'
                    angle -= 1
                elif angle == 0 and [t.text for t in tokens[cursor:cursor + 2]] == [':', ':']:
                    if cursor == begin:
                        return None, 'empty_qualifier_segment'
                    parts.append(tokens[begin:cursor]);separators.append(tokens[cursor:cursor + 2])
                    cursor += 2;begin = cursor;continue
                elif angle == 0 and word == ',':
                    return None, 'using_declarator_list_outside_fixed_shapes'
            cursor += 1
        if groups or angle:
            return None, 'unbalanced_or_expression_template_arguments'
        if begin == len(tokens):
            return None, 'empty_terminal_name'
        parts.append(tokens[begin:])
        for part in parts[:-1]:
            if not self.qualifier_head(part):
                return None, 'non_qualified_type_expression'
        terminal = parts[-1]
        if len(terminal) == 1 and terminal[0].kind == 'identifier' and terminal[0].text not in TYPE_SPECIFIER_WORDS:
            terminal_name, terminal_kind = terminal[0].text, 'identifier'
        elif terminal[0].text == 'operator' and ''.join(t.text for t in terminal[1:]) in (
                '=', '+', '-', '*', '/', '%', '^', '&', '|', '~', '!', '<', '>', '+=', '-=', '*=', '/=', '%=',
                '^=', '&=', '|=', '<<', '>>', '>>=', '<<=', '==', '!=', '<=', '>=', '<=>', '&&', '||',
                '++', '--', ',', '->*', '->', '()', '[]', 'new', 'new[]', 'delete', 'delete[]'):
            terminal_name = 'operator' + (' ' if terminal[1].text in ('new', 'delete') else '') + ''.join(t.text for t in terminal[1:])
            terminal_kind = 'operator_function_id'
        elif terminal[0].text != 'decltype' and self.qualifier_head(terminal):
            # Useful for an alias RHS such as std::pair<int, N::T>;
            # record the whole unqualified template-id, not its last argument.
            terminal_name = self.text(terminal[0].start, terminal[-1].end)
            terminal_kind = 'template_id'
        else:
            return None, 'non_qualified_type_expression'
        return {'leading': leading, 'parts': parts, 'separators': separators,
                'terminal_name': terminal_name, 'terminal_kind': terminal_kind}, None

    @staticmethod
    def qualifier_head(tokens):
        if not tokens or tokens[0].kind != 'identifier':
            return None
        if tokens[0].text in TYPE_SPECIFIER_WORDS and tokens[0].text != 'decltype':
            return None
        if len(tokens) == 1:
            return tokens[0]
        if tokens[0].text == 'decltype' and len(tokens) >= 3 and tokens[1].text == '(' and tokens[-1].text == ')':
            return tokens[0]
        if len(tokens) >= 3 and tokens[1].text == '<' and tokens[-1].text == '>':
            return tokens[0]
        return None

    def target(self, record, tokens, *, strict):
        record['target_source_range'] = self.token_span(tokens)
        record['target_source_expression'] = self.text(tokens[0].start, tokens[-1].end) if tokens else ''
        record['target_token_spellings'] = [t.text for t in tokens]
        name_tokens = tokens
        if name_tokens and name_tokens[0].text == 'typename':
            record['typename_range'] = self.token_evidence(name_tokens[0])
            name_tokens = name_tokens[1:]
        record['target_name_source_range'] = self.token_span(name_tokens)
        record['target_name_source_expression'] = self.text(name_tokens[0].start, name_tokens[-1].end) if name_tokens else ''
        split, reason = self.split_qualified(name_tokens)
        if not split:
            record['target_form'] = 'unclassified_qualified_target' if strict else 'type_expression'
            record['target_name_analysis'] = {'status': 'unsupported' if strict else 'not_a_simple_qualified_name', 'reason': reason}
            if strict:
                record['kind'] = 'unsupported'
                record['syntax_status'] = 'unsupported_source_shape'
                region = record['target_source_range'] or record['source_range']
                record['diagnostic_refs'].append(self.diagnostic(reason, region['start_byte'], region['end_byte'],
                    source_id=record.get('source_using_id') or record.get('source_namespace_alias_id'),
                    message='Target source preserved, but this shape is outside fixed qualified-name handling; no owner/target is inferred.'))
            return
        if split['leading']:
            record['leading_global_scope_range'] = self.token_span(split['leading'])
        for part in split['parts'][:-1]:
            head = self.qualifier_head(part)
            decltype_qualifier = head.text == 'decltype'
            record['qualifier_segments'].append({'source_range': self.token_span(part),
                'source_expression': self.text(part[0].start, part[-1].end),
                'token_spellings': [t.text for t in part],
                'syntax_kind': 'decltype_specifier' if decltype_qualifier else 'template_id' if len(part) > 1 else 'identifier',
                'head_name': None if decltype_qualifier else head.text,
                'head_name_range': None if decltype_qualifier else self.token_evidence(head)})
        record['scope_separator_ranges'] = [self.token_span(s) for s in split['separators']]
        record['terminal_name'] = split['terminal_name']
        record['terminal_kind'] = split['terminal_kind']
        record['terminal_name_range'] = self.token_span(split['parts'][-1])
        record['target_form'] = 'qualified_id' if split['separators'] or split['leading'] else 'unqualified_id'
        record['target_name_analysis'] = {'status': 'source_segments_recorded', 'reason': None}
        if record['kind'] == 'import' and not record['typename_range'] and record['qualifier_segments'] and split['terminal_kind'] == 'identifier':
            preceding = record['qualifier_segments'][-1]
            if preceding['head_name'] == record['terminal_name']:
                record['constructor_spelling_candidate'] = True
                record['constructor_candidate_evidence'] = {
                    'rule': 'terminal_identifier_equals_immediately_preceding_qualifier_head',
                    'qualifier_name_range': preceding['head_name_range'],
                    'terminal_name_range': record['terminal_name_range'],
                    'semantic_constructor_classification': 'not_attempted_requires_owner_and_base_binding'}

    def using(self, index):
        end, terminated, error = self.statement_end(index)
        record = self.statement_record('written_using', index, end, terminated)
        tail = self.tokens[index + 1:end]
        record['kind'] = 'unsupported'
        if not terminated:
            record['diagnostic_refs'].append(self.diagnostic(error, record['source_range']['start_byte'], record['source_range']['end_byte'],
                source_id=record['source_using_id'], message='Written using has no trustworthy complete statement terminator; partial source is retained.'))
        if not tail:
            record['diagnostic_refs'].append(self.diagnostic('empty_using_statement', record['source_range']['start_byte'], record['source_range']['end_byte'],
                source_id=record['source_using_id'], message='using has no declarator or nominated target.'))
            record['syntax_status'] = 'malformed'
            return record
        if tail[0].text == 'namespace':
            record['kind'] = 'directive'
            record['namespace_keyword_range'] = self.token_evidence(tail[0])
            self.target(record, tail[1:], strict=True)
        elif tail[0].text == 'enum':
            record['unsupported_form'] = 'using_enum_not_present_in_fixed_snapshot'
            record['enum_keyword_range'] = self.token_evidence(tail[0])
            self.target(record, tail[1:], strict=False)
            record['syntax_status'] = 'unsupported_source_shape'
            record['diagnostic_refs'].append(self.diagnostic('using_enum_outside_fixed_shapes', tail[0].start, tail[-1].end,
                source_id=record['source_using_id'], message='using enum is retained but is not supported by this fixed-snapshot source layer.'))
        else:
            after_name, attributes, balanced = self.attribute_after_name(tail)
            if balanced and tail[0].kind == 'identifier' and [t.text for t in tail[after_name:after_name + 1]] == ['=']:
                record['kind'] = 'alias'
                record['declared_name'] = tail[0].text
                record['declared_name_range'] = self.token_evidence(tail[0])
                record['attribute_ranges'] = attributes
                record['alias_equal_range'] = self.token_evidence(tail[after_name])
                self.target(record, tail[after_name + 1:], strict=False)
                if not tail[after_name + 1:]:
                    record['syntax_status'] = 'malformed'
                    record['diagnostic_refs'].append(self.diagnostic('empty_alias_target', tail[after_name].start, tail[after_name].end,
                        source_id=record['source_using_id'], message='Alias has no target type expression.'))
            else:
                record['kind'] = 'import'
                self.target(record, tail, strict=True)
        if not terminated:
            record['kind'] = 'unsupported';record['syntax_status'] = 'malformed'
        return record

    def namespace_alias(self, index):
        # This pass records only the written namespace-alias grammar; it does
        # not count arbitrary namespace declarations or using-directives.
        if index and self.tokens[index - 1].text == 'using':
            return None
        if index + 2 >= len(self.tokens) or self.tokens[index + 1].kind != 'identifier' or self.tokens[index + 2].text != '=':
            return None
        end, terminated, error = self.statement_end(index)
        record = self.statement_record('written_namespace_alias', index, end, terminated)
        record['kind'] = 'namespace_alias'
        name = self.tokens[index + 1]
        record['declared_name'] = name.text;record['declared_name_range'] = self.token_evidence(name)
        record['alias_equal_range'] = self.token_evidence(self.tokens[index + 2])
        self.target(record, self.tokens[index + 3:end], strict=True)
        if not terminated:
            record['kind'] = 'unsupported';record['syntax_status'] = 'malformed'
            record['diagnostic_refs'].append(self.diagnostic(error, record['source_range']['start_byte'], record['source_range']['end_byte'],
                source_id=record['source_namespace_alias_id'], message='Namespace alias has no trustworthy complete terminator.'))
        return record


def scan_source_usings(path, source: bytes, commit):
    """Return all physical written using statements plus separate namespace aliases.

    target_source_expression includes a written leading ``typename``; the
    target_name_source_expression excludes that keyword for qualifier/terminal
    segmentation. Neither field is an expanded C++ lookup result. terminal_name
    is null for a general alias RHS that is not a simple qualified-id.
    """
    view = _Source(path, source, commit)
    for issue in view.lexical_issues:
        a, b = issue['byte_range']
        view.diagnostic('source_lexical_' + issue['kind'], a, b,
            message='Source lexer reported malformed input; no successful using-source contract is inferred from this region.')
    sources, namespace_aliases = [], []
    for index, token in enumerate(view.tokens):
        if token.text == 'using':
            sources.append(view.using(index))
        elif token.text == 'namespace':
            record = view.namespace_alias(index)
            if record:
                namespace_aliases.append(record)
    counts = Counter(s['kind'] for s in sources)
    return {'schema_version': SCHEMA_VERSION, 'commit': commit, 'path': path,
            'source_sha256': _hash(source), 'sources': sources,
            'namespace_aliases': namespace_aliases,
            'counts': {'written_using': len(sources), **{kind: counts[kind] for kind in ('alias', 'import', 'directive', 'unsupported')},
                       'namespace_alias': len(namespace_aliases),
                       'constructor_spelling_candidates': sum(s['constructor_spelling_candidate'] for s in sources)},
            'diagnostics': view.diagnostics,
            'scope_or_target_resolution_performed': False}

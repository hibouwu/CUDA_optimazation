#!/usr/bin/env python3
"""Source-checked reconciliation of immutable lexical candidates, NOT API coverage.

No declaration extractor, parser, namespace resolver, or macro expander is imported.
The independent scanner's byte lexer is used only to inspect immutable source.
The input JSON arrays are streamed; a temporary SQLite index stores only the
declaration fields needed by this pass. At most one source file's candidates and
occurrences are reconciled in memory. Every original candidate produces one row.

Mapping to an enclosing namespace/class/body is explicitly forbidden. A mapped
declaration needs compatible kind/name, a verified signature/declarator span, and
coverage of the candidate's significant tokens. Function-body non-API decisions
need a source-verified callable body; local type scopes are excluded independently.
Uncertainty stays pending, even when the declaration extractor reported no error.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reconciliation_lexer", Path(__file__).with_name("scan_candidates.py"))
LEXER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LEXER
SPEC.loader.exec_module(LEXER)
CALLABLE = {"function", "method", "constructor", "destructor", "operator"}
TYPES = {"class_specifier", "struct_specifier", "union_specifier", "enum"}
VALUES = {"variable", "constant", "member", "member_constant", "variable_template", "enumerator"}
ALIASES = {"alias", "typedef", "using_declaration", "namespace_alias"}
SCAFFOLD = {";", "{", "}", ",", "public", "private", "protected", ":"}
CONTROL = {"if", "for", "while", "switch", "catch", "sizeof", "alignof", "alignas", "decltype", "noexcept", "static_assert", "requires", "return", "constexpr"}
RESERVED_CALLABLE_NAMES = CONTROL | {"else", "nullptr", "true", "false", "class", "struct", "namespace", "int", "float", "double", "char", "bool", "void", "auto", "template", "typename", "friend", "const", "static"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class JSONStream:
    """Small stdlib streaming reader for top-level object / itemwise arrays."""
    def __init__(self, stream, chunk_size=1024 * 1024):
        self.stream, self.chunk_size = stream, chunk_size
        self.buffer, self.pos, self.eof = "", 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        self.buffer = self.buffer[self.pos:]
        self.pos = 0
        data = self.stream.read(self.chunk_size)
        self.eof = not data
        self.buffer += data

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ""
            self.fill()

    def expect(self, char):
        if self.peek() != char:
            raise ValueError(f"Expected JSON {char!r}, found {self.peek()!r}")
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                self.pos = end
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.fill()


def top_items(path, arrays=(), chunk_size=1024 * 1024):
    """Yield (key, value, array_item); unrequested arrays are decoded/discarded."""
    with open(path, encoding="utf-8") as stream:
        reader = JSONStream(stream, chunk_size)
        reader.expect("{")
        first = True
        while reader.peek() != "}":
            if not first:
                reader.expect(",")
            first = False
            key = reader.value()
            reader.expect(":")
            if reader.peek() == "[":
                reader.expect("[")
                initial = True
                while reader.peek() != "]":
                    if not initial:
                        reader.expect(",")
                    initial = False
                    value = reader.value()
                    if key in arrays:
                        yield key, value, True
                reader.expect("]")
            else:
                yield key, reader.value(), False
        reader.expect("}")
        if reader.peek():
            raise ValueError("Trailing data after JSON document")


def span(value):
    return [value["start_byte"], value["end_byte"]] if value else None


def slim_occurrence(o):
    return {
        "id": o["declaration_occurrence_id"], "entity_id": o["entity_id"],
        "source_id": o.get("source_occurrence_id"),
        "kind": o["kind"], "name": o["name"], "qualified_name": o["qualified_name"],
        "range": [o["start_byte"], o["end_byte"]],
        "signature": span(o.get("signature_range")), "declarator": span(o.get("declarator_range")),
        "body": span(o.get("body_range")), "syntax": span(o.get("syntax_node_range")),
        "raw_signature_sha256": sha(o.get("raw_signature", "").encode()),
        "parse_status": o.get("parse_status"), "attributes": o.get("attributes", []),
        "alignment_specifiers":o.get('alignment_specifiers',[]),
        "scope_review_required":o.get('scope_review_required',False),
        "scope_integrity_refs":o.get('scope_integrity_refs',[]),
        "parameter_count": len(o.get("parameters", [])),
        "condition_lines": [c.get("directive_line") for c in o.get("preprocessor_conditions", [])],
        "conditions": o.get("preprocessor_conditions", []),
        "macro_origin": o.get("macro_origin"),
        "conditional_origin": o.get("conditional_declaration_origin"),
        "target_type": o.get("target_type"),
        "initializer": o.get("initializer"), "declared_type": o.get("declared_type"),
        "return_type": o.get("return_type"), "qualifiers": o.get("qualifiers", []),
        "prefix_specifiers": o.get("prefix_specifiers", []),
        "parameters": [{"range": span(p), "name": p.get("name"), "type": p.get("type"), "default": p.get("default")} for p in o.get("parameters", [])],
        "is_bitfield": o.get("is_bitfield", False), "bitfield_source_id": o.get("bitfield_source_id"),
        "display_name": o.get("display_name"),
        "bit_width": o.get("bit_width"), "bit_width_range": span(o.get("bit_width_range")),
        "colon_range": span(o.get("colon_range")), "bitfield_origin": o.get("bitfield_origin"),
        "expanded_signature": o.get("expanded_signature"),
        "namespace_choices": o.get("namespace_choices"), "namespace_conditions": o.get("namespace_conditions"),
        "expanded_namespace_component": o.get("expanded_namespace_component"),
        "name_resolution": o.get("name_resolution", {}),
        "qualified_name_resolution": o.get("qualified_name_resolution"),
        "scope_chain": [{k: s.get(k) for k in ("name", "kind", "entity_id", "name_resolution")} for s in o.get("scope_chain", [])],
    }


def token_spelling(text):
    return tuple(t.text for t in LEXER.lex(text.encode())[0])


def condition_form(expression):
    """Normalize redundant boolean parentheses without erasing grouping."""
    text = "".join(token_spelling(expression))
    while text.startswith("(") and text.endswith(")"):
        depth, closes_early = 0, False
        for index, char in enumerate(text):
            depth += char == "("
            depth -= char == ")"
            if depth == 0 and index != len(text) - 1:
                closes_early = True
                break
        if closes_early:
            break
        text = text[1:-1]
    for operator in ("||", "&&"):
        depth, start, parts, index = 0, 0, [], 0
        while index < len(text):
            depth += text[index] == "("
            depth -= text[index] == ")"
            if not depth and text.startswith(operator, index):
                parts.append(condition_form(text[start:index]))
                index += 2
                start = index
            else:
                index += 1
        if parts:
            return (operator, *parts, condition_form(text[start:]))
    if text.startswith("!") and not text.startswith("!="):
        return ("!", condition_form(text[1:]))
    defined = re.fullmatch(r"defined\(?([A-Za-z_]\w*)\)?", text)
    return ("defined", defined[1]) if defined else ("atom", text)


def condition_line(condition):
    return condition.get("directive_line", condition.get("directive_span", {}).get("start_line"))


def condition_matches(condition, proof):
    location = condition.get("directive_span", {})
    return (condition_line(condition) == proof["line"]
            and condition_form(condition.get("expression", "")) == condition_form(proof["expression"])
            and (not location.get("path") or location["path"] == proof["path"])
            and (not condition.get("directive_path") or condition["directive_path"] == proof["path"])
            and (not location or (location.get("start_byte"), location.get("end_byte")) == tuple(proof.get("span", (None, None)))))


def proof_key(proof):
    return (proof["path"], proof["line"], condition_form(proof["expression"]))


class SourceProof:
    """Fresh lexical evidence; never calls namespace or declaration resolvers."""
    def __init__(self, path, data):
        self.path, self.data = path, data
        self.scan = LEXER.SourceScan(path, data)
        self.scan.scan_preprocessor()
        self.directives = []
        for original in self.scan.directives:
            start, end = original["byte_range"]
            line_start = data.rfind(b"\n", 0, start) + 1
            if not data[line_start:start].strip():
                start = line_start
            self.directives.append({**original, "physical_range": [start, end]})
        self.directive_ids = {d["candidate_id"]: d for d in self.directives}
        self.branches = {c["candidate_id"]: c for c in self.scan.candidates if c["kind"] == "preprocessor_branch"}

    def condition(self, expression, directive):
        return {"path": self.path, "line": directive["line_range"][0],
                "span": directive["physical_range"], "expression": expression}

    def conditions_at(self, byte):
        carrier = next((d for d in self.directives if d["physical_range"][0] <= byte < d["physical_range"][1]), None)
        if carrier is None:
            carrier = next((r for r in self.scan.regions if r["byte_range"][0] <= byte < r["byte_range"][1]), None)
        result = []
        for reference in (carrier or {}).get("conditions", []):
            branch = self.branches[reference]
            result.append(self.condition(branch["predicate"], self.directive_ids[branch["directive_id"]]))
        return result

    def definition(self, record):
        """Verify claimed definition bounds, token body, parameters and guards."""
        matches = []
        for actual in self.scan.definitions:
            a, b = actual["byte_range"]
            start, end = record.get("start_byte", -1), record.get("end_byte", -1)
            if actual["name"] != record.get("name") or not 0 <= start <= a < end <= b:
                continue
            if self.data[start:a].strip() or self.data[end:b].strip():
                continue
            raw_params = actual["parameter_tokens"]
            params = None if not actual["function_like"] else ["".join(piece) for piece in self._comma_parts(raw_params)]
            x, y = actual["body_byte_range"]
            if params == record.get("parameters") and token_spelling(self.data[x:y].decode("utf-8", "replace")) == token_spelling(record.get("body", "")):
                guards = self.conditions_at(a)
                claimed = record.get("preprocessor_conditions", [])
                if all(any(condition_matches(c, g) for c in claimed) for g in guards) and all(any(condition_matches(c, g) for g in guards) for c in claimed):
                    matches.append((actual, guards))
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _comma_parts(tokens):
        parts = [[]]
        for token in tokens:
            if token == ",": parts.append([])
            else: parts[-1].append(token)
        return [] if parts == [[]] else parts

    def conditional_projections(self, start, end, budget=64):
        """Enumerate all physical text branches in ONE conditional declaration."""
        directives = [d for d in self.directives if start <= d["physical_range"][0] < end]
        def predicate(d):
            command, expression = d["directive"], d["expression"]
            return expression if command == "if" else ("!" if command == "ifndef" else "") + f"defined({expression})"
        def combine(left, right):
            if len(left) * len(right) > budget:
                raise ValueError("conditional projection branch budget exceeded")
            return [(a + b, ac + bc) for a, ac in left for b, bc in right]
        def sequence(cursor, index, stops):
            variants = [([], [])]
            while index < len(directives):
                d = directives[index]
                a, b = d["physical_range"]
                if d["directive"] in stops:
                    return combine(variants, [([(cursor, a)] if cursor < a else [], [])]), index, a
                if d["directive"] not in {"if", "ifdef", "ifndef"}:
                    raise ValueError("unsupported directive in conditional declaration")
                variants = combine(variants, [([(cursor, a)] if cursor < a else [], [])])
                prior, alternatives, has_else = [predicate(d)], [], False
                body, index, _ = sequence(b, index + 1, {"elif", "else", "endif"})
                alternatives.extend((spans, [self.condition(prior[0], d)] + conditions) for spans, conditions in body)
                while index < len(directives) and directives[index]["directive"] in {"elif", "else"}:
                    branch = directives[index]
                    negated = "!(" + " || ".join("(" + p + ")" for p in prior) + ")"
                    has_else = branch["directive"] == "else"
                    active = negated if has_else else negated + " && (" + branch["expression"] + ")"
                    if not has_else: prior.append(branch["expression"])
                    body, index, _ = sequence(branch["physical_range"][1], index + 1, {"elif", "else", "endif"})
                    alternatives.extend((spans, [self.condition(active, branch)] + conditions) for spans, conditions in body)
                if index >= len(directives) or directives[index]["directive"] != "endif":
                    raise ValueError("unclosed conditional declaration")
                closing = directives[index]
                if not has_else:
                    alternatives.append(([], [self.condition("!(" + " || ".join("(" + p + ")" for p in prior) + ")", closing)]))
                variants = combine(variants, alternatives)
                cursor, index = closing["physical_range"][1], index + 1
            return combine(variants, [([(cursor, end)] if cursor < end else [], [])]), index, end
        variants, _, _ = sequence(start, 0, set())
        outer = self.conditions_at(start)
        return [(segments, outer + conditions) for segments, conditions in variants]


class IntervalIndex:
    def __init__(self, pairs):
        self.items = sorted((a, b, value) for a, b, value in pairs)
        self.starts = [x[0] for x in self.items]
        maximum = -1
        self.prefix_end = []
        for _, end, _ in self.items:
            maximum = max(maximum, end)
            self.prefix_end.append(maximum)

    def overlapping(self, a, b):
        index = bisect_left(self.starts, b) - 1
        result = []
        while index >= 0 and self.prefix_end[index] > a:
            start, end, value = self.items[index]
            if start < b and a < end:
                result.append(value)
            index -= 1
        return result


def head_kind(tokens):
    """Conservative declaration-shape guard, not a replacement C++ parser."""
    text = [t.text for t in tokens]
    paren = angle = 0
    for index, word in enumerate(text):
        if paren == angle == 0:
            if word == "namespace":
                return "namespace"
            if word in {"struct", "class", "union", "enum"}:
                return "type"
            if word in {"using", "typedef"}:
                return "alias"
            if word == "operator":
                return "callable"
            if word == "=":
                return "value"
            if word == "(" and index and re.fullmatch(r"[A-Za-z_]\w*", text[index - 1]) and text[index - 1] not in CONTROL and (index + 1 == len(text) or text[index + 1] not in {"*", "&"}):
                return "callable"
        if word == "<":
            angle += 1
        elif word == ">" and angle:
            angle -= 1
        elif word == "(":
            paren += 1
        elif word == ")" and paren:
            paren -= 1
    return "undetermined"


def compatible(kind, expected):
    if expected == "namespace":
        return kind in {"namespace", "namespace_alias"}
    if expected == "type":
        return kind in TYPES
    if expected == "alias":
        return kind in ALIASES | TYPES
    if expected == "callable":
        return kind in CALLABLE
    if expected == "value":
        return kind in VALUES
    return kind in CALLABLE | TYPES | VALUES | ALIASES | {"namespace"}


class Reconciler:
    def __init__(self, path, source, candidates, occurrences, diagnostics=(), macro_definitions=None, file_record=None, source_reader=None):
        self.path, self.source, self.candidates = path, source, candidates
        self.occurrences, self.diagnostics = occurrences, list(diagnostics)
        self._invalid_cache = {}
        self._name_anchors = {}
        self._proof_views, self._binding_infos, self._conditional_infos, self._bitfield_infos = {}, {}, {}, {}
        self.source_reader = source_reader
        self.macro_definitions = macro_definitions or {}
        self.file_record = file_record or {}
        if len({o["id"] for o in occurrences}) != len(occurrences):
            raise ValueError("Duplicate declaration instance ID must not collapse namespace/conditional variants")
        source_identities = {}
        for o in occurrences:
            if o.get("source_id"):
                identity = (tuple(o["range"]), o["kind"], o["name"], o.get("bitfield_source_id"))
                if o["source_id"] in source_identities and source_identities[o["source_id"]] != identity:
                    raise ValueError("One source occurrence ID points to different physical declarations")
                source_identities[o["source_id"]] = identity
        self.tokens, _, self.lexical_issues = LEXER.lex(source)
        self.starts = [t.start for t in self.tokens]
        self.by_id = {c["candidate_id"]: c for c in candidates}
        self.directive_lines = {c["candidate_id"]: c["line_range"][0] for c in candidates if c["kind"] == "preprocessor_directive"}
        self.branch_lines = {c["candidate_id"]: self.directive_lines.get(c["directive_id"]) for c in candidates if c["kind"] == "preprocessor_branch"}
        self.signatures = IntervalIndex((o["signature"][0], o["signature"][1], index) for index, o in enumerate(occurrences) if o.get("signature"))
        self.diag_index = IntervalIndex((d["start_byte"], max(d["end_byte"], d["start_byte"] + 1), i) for i, d in enumerate(self.diagnostics))
        self.invocations = IntervalIndex((c["byte_range"][0], c["byte_range"][1], i) for i, c in enumerate(candidates)
                                        if c["kind"] == "macro_invocation" and c.get("origin") == "source")
        self.macros = defaultdict(list)
        self.namespace_owners = defaultdict(list)
        self.type_owners = defaultdict(list)
        self.conditional_groups = defaultdict(list)
        for o in occurrences:
            if o["kind"] == "namespace":
                self.namespace_owners[o["entity_id"]].append(o)
            if o["kind"] in TYPES:
                self.type_owners[o["entity_id"]].append(o)
            if o.get("conditional_origin"):
                self.conditional_groups[o["conditional_origin"].get("source_declaration_id")].append(o)
            if o.get("macro_origin"):
                inv = o["macro_origin"]["invocation"]
                self.macros[(inv["start_byte"], inv["end_byte"])].append(o)
        self.braces = self.lexical_braces()
        self.delimiters = self.lexical_delimiters()
        self.scope_body_openings = [c["byte_range"][1] - 1 for c in candidates
                                   if c["kind"] == "syntax_interval" and c.get("terminator") == "{"
                                   and head_kind(self.source_tokens(*c["byte_range"])) in {"type", "namespace"}]
        body_pairs = []
        for index, o in enumerate(occurrences):
            if o["kind"] in CALLABLE and o.get("body") and not o.get("macro_origin"):
                a, b = o["body"]
                if self.callable_header_verified(o) and self.braces.get(a) == b:
                    body_pairs.append((a, b, index))
        self.bodies = IntervalIndex(body_pairs)
        # Recognize possible local type braces from raw candidates even if the
        # declaration extractor entirely omitted the type and its members.
        self.type_bodies = IntervalIndex(
            (c["byte_range"][1] - 1, self.braces[c["byte_range"][1] - 1], i)
            for i, c in enumerate(candidates)
            if c["kind"] == "syntax_interval" and c.get("terminator") == "{"
            and c["byte_range"][1] - 1 in self.braces
            and head_kind(self.source_tokens(*c["byte_range"])) == "type")
        self.expression_contexts = self.build_expression_contexts()
        self.expressions = IntervalIndex((e["range"][0], e["range"][1], i) for i, e in enumerate(self.expression_contexts))

    def source_tokens(self, a, b):
        return self.tokens[bisect_left(self.starts, a):bisect_left(self.starts, b)]

    def lexical_braces(self):
        directives = sorted(c["byte_range"] for c in self.candidates if c["kind"] == "preprocessor_directive")
        index, stack, pairs = 0, [], {}
        for token in self.tokens:
            while index < len(directives) and directives[index][1] <= token.start:
                index += 1
            if index < len(directives) and directives[index][0] <= token.start < directives[index][1]:
                continue
            if token.text == "{":
                stack.append(token.start)
            elif token.text == "}" and stack:
                pairs[stack.pop()] = token.end
        return pairs

    def lexical_delimiters(self):
        directives = sorted(c["byte_range"] for c in self.candidates if c["kind"] == "preprocessor_directive")
        index, stack, pairs = 0, [], {}
        closing = {")": "(", "]": "[", "}": "{"}
        for token in self.tokens:
            while index < len(directives) and directives[index][1] <= token.start:
                index += 1
            if index < len(directives) and directives[index][0] <= token.start < directives[index][1]:
                continue
            if token.text in {"(", "[", "{"}:
                stack.append(token)
            elif token.text in closing:
                if stack and stack[-1].text == closing[token.text]:
                    pairs[stack.pop().start] = token.end
                else:
                    stack.clear()  # Recovery cannot match across a broken delimiter.
        return pairs

    def callable_header_verified(self, o):
        if o["kind"] not in CALLABLE or not o.get("body") or o.get("macro_origin"):
            return False
        name = (o.get("name") or "").split("::")[-1]
        if not name or name in RESERVED_CALLABLE_NAMES:
            return False
        words = self.source_tokens(*o["signature"])
        if words and words[0].text in {"if", "else", "for", "while", "return", "switch"}:
            return False
        binding_errors = set(self.binding_info(o)["errors"])
        invalid = set(self.invalid_occurrence(o)) - binding_errors - {"macro_dependent_owner_not_materialized"}
        if not invalid:
            return o["parse_status"] == "parsed"
        if invalid != {"occurrence_not_cleanly_parsed"}:
            return False
        a, b = o["body"]
        relevant = self.blockers(*o["range"])
        # An explicit body diagnostic does not invalidate a byte-exact function
        # header. No diagnostic or a header/cross-boundary diagnostic is NOT proof.
        return bool(relevant) and all(a < d["start_byte"] <= d["end_byte"] <= b for d in relevant)

    def nested_type_between(self, start, a, b):
        return any(start <= self.candidates[i]["byte_range"][1] - 1 < a for i in self.type_bodies.overlapping(a, b))

    def signature_verified(self, o):
        if not compatible(o["kind"], head_kind(self.source_tokens(*o["signature"]))):
            return False
        return not self.invalid_occurrence(o) or self.callable_header_verified(o)

    def declaration_name_anchor(self, o):
        if o["id"] not in self._name_anchors:
            name = self.source_name(o)
            if not name:
                self._name_anchors[o["id"]] = None
                return None
            expected = token_spelling(name)
            values = self.source_tokens(*(o.get("declarator") or o.get("syntax") or o["signature"]))
            spellings = [t.text for t in values]
            hit = next((i for i in range(len(values)) if tuple(spellings[i:i + len(expected)]) == expected), None)
            self._name_anchors[o["id"]] = None if hit is None else (values[hit].start, values[hit + len(expected) - 1].end)
        return self._name_anchors[o["id"]]

    def build_expression_contexts(self):
        contexts = []
        for index, o in enumerate(self.occurrences):
            if o.get("macro_origin") or o.get("conditional_origin") or not self.signature_verified(o):
                continue
            start, end = o["signature"]
            tokens = self.source_tokens(start, end)
            anchor = self.declaration_name_anchor(o)
            if not anchor:
                continue
            type_expression = o.get("declared_type") if o["kind"] in VALUES else o.get("return_type") if o["kind"] in CALLABLE else None
            if type_expression:
                expected = token_spelling(type_expression)
                before = [t for t in tokens if t.end <= anchor[0]]
                actual = [t.text for t in before]
                hits = [i for i in range(len(before)) if tuple(actual[i:i + len(expected)]) == expected]
                if expected and hits:
                    i = hits[-1]
                    contexts.append({"range": [before[i].start, before[i + len(expected) - 1].end], "owner": index, "kind": "declared_or_return_type_expression"})
            if o["kind"] in ALIASES and o.get("target_type"):
                equal = next((t for t in tokens if t.text == "=" and t.start >= anchor[1]), None)
                stop = tokens[-1].start if tokens and tokens[-1].text == ";" else end
                if equal and token_spelling(self.source[equal.end:stop].decode()) == token_spelling(o["target_type"]):
                    contexts.append({"range": [equal.end, stop], "owner": index, "kind": "alias_type_expression"})
            if o["kind"] in VALUES and o.get("initializer") and head_kind(tokens) != "callable":
                payload = o["initializer"].encode()
                position = self.source.find(payload, anchor[1], end)
                if position >= 0:
                    prefix = token_spelling(self.source[anchor[1]:position].decode())
                    if prefix == ("=",) or (not prefix and payload.startswith(b"{")):
                        contexts.append({"range": [position, position + len(payload)], "owner": index, "kind": "data_initializer_expression"})
            if o["kind"] in CALLABLE:
                for parameter in o.get("parameters", []):
                    bounds = parameter.get("range")
                    if not bounds or not start <= bounds[0] <= bounds[1] <= end:
                        continue
                    contexts.append({"range": bounds, "owner": index, "kind": "parameter_type_or_default_expression"})
                if o["kind"] == "constructor" and o.get("declarator"):
                    after = [t for t in tokens if t.start >= o["declarator"][1]]
                    if after and after[0].text == ":":
                        contexts.append({"range": [after[0].end, end], "owner": index, "kind": "constructor_initializer_expression"})
        return contexts

    def expression_mapping(self, c):
        a, b = c["byte_range"]
        tokens = [t for t in self.source_tokens(a, b) if t.text not in SCAFFOLD]
        if not tokens or self.blockers(a, b) or self.unexpanded_declaration_macro(a, b):
            return None
        matched, categories = [], set()
        for index in self.expressions.overlapping(a, b):
            context = self.expression_contexts[index]
            x, y = context["range"]
            owner = self.occurrences[context["owner"]]
            if all(x <= t.start and t.end <= y for t in tokens) and not self.nested_type_between(x, a, b):
                matched.append(owner)
                categories.add(context["kind"])
        if matched and not self.binding_completeness(matched):
            return self.row(c, "classified_non_api", "phase2_expression", "exact_expression_subrange_of_verified_declaration_signature", matched,
                            classification="declaration_subexpression", expression_contexts=sorted(categories), relationship_status="pending_phase2")
        return None

    def unexpanded_declaration_macro(self, a, b):
        hints = {"operator", "namespace", "struct", "class", "enum", "using", "typedef"}
        for index in self.invocations.overlapping(a, b):
            c = self.candidates[index]
            if not a <= c["byte_range"][0] < c["byte_range"][1] <= b:
                continue
            definitions = [self.macro_definitions[ref] for ref in c.get("definition_candidates", []) if ref in self.macro_definitions]
            if not any(hints.intersection(d.get("declaration_generation_hints", [])) for d in definitions):
                continue
            # The exact legacy static_assert compatibility helper remains a
            # separate macro obligation. Its source statement is a constraint,
            # never permission to dismiss arbitrary declaration-generating macros.
            if c.get("name") == "static_assert":
                expected = token_spelling("typedef int __platform_cat(AsSeRt, __LINE__)[(__e) ? 1 : -1]")
                verified = bool(definitions)
                for definition in definitions:
                    try:
                        view = self.proof_view(definition["path"])
                        x, y = definition["body_byte_range"]
                        verified &= definition["path"] == "include/cutlass/platform/platform.h" and token_spelling(view.data[x:y].decode()) == expected
                    except (KeyError, ValueError, OSError):
                        verified = False
                if verified:
                    continue
            if not self.macros.get(tuple(c["byte_range"])):
                return True
        return False

    def signature_token_mapping(self, c):
        a, b = c["byte_range"]
        tokens = self.source_tokens(a, b)
        name = c.get("name")
        matches, role = [], None
        for index in self.signatures.overlapping(a, b):
            o = self.occurrences[index]
            if o.get("macro_origin") or not self.signature_verified(o) or not o["signature"][0] <= a < b <= o["signature"][1] or self.nested_type_between(o["signature"][0], a, b):
                continue
            if name == "constexpr" and c.get("form") == "object_like" and "constexpr" in o.get("qualifiers", []) + o.get("prefix_specifiers", []):
                matches.append(o); role = "language_declaration_specifier"
            elif name == "alignas" and o["kind"] in VALUES | TYPES and any("alignas" in q for q in o.get("qualifiers", [])):
                matches.append(o); role = "alignment_attribute_expression"
            elif name and o["kind"] == "namespace" and self.source_name(o) == name:
                matches.append(o); role = "namespace_declaration_name_token"
            elif c["kind"] == "syntax_interval" and all(t.text in {"(", ")", "[", "]", "<", ">", ",", ";", "{", "}"} for t in tokens):
                matches.append(o); role = "signature_delimiter_continuation"
        if matches and not self.binding_completeness(matches):
            return self.row(c, "mapped_occurrences" if role == "namespace_declaration_name_token" else "classified_non_api",
                            "phase1_declaration" if role == "namespace_declaration_name_token" else "phase1_declaration_component",
                            "exact_syntax_component_of_verified_signature", matches, classification=role)
        return None

    def proof_view(self, path):
        if path not in self._proof_views:
            if path == self.path:
                data = self.source
            elif self.source_reader is not None:
                data = self.source_reader(path)
            else:
                raise ValueError("No verified source reader for definition evidence: " + path)
            self._proof_views[path] = SourceProof(path, data)
        return self._proof_views[path]

    @staticmethod
    def alignment_conditions_match(claimed, actual):
        return (len(claimed) == len(actual)
                and all(any(condition_matches(c, p) for c in claimed) for p in actual)
                and all(any(condition_matches(c, p) for p in actual) for c in claimed))

    def alignment_span_verified(self, record):
        data = self.proof_view(record['path']).data
        a, b = record['start_byte'], record['end_byte']
        return (0 <= a <= b <= len(data)
                and record.get('start_line') == data[:a].count(b'\n') + 1
                and record.get('end_line') == data[:max(a, b - 1)].count(b'\n') + 1)

    def alignment_proof_errors(self, c, attribute, owner):
        """Fixed CUTE alignment proof; never invokes the header/extractor adapter."""
        errors = []
        a, b = c['byte_range']
        tokens = self.source_tokens(a, b)
        if (len(tokens) < 4 or tokens[0].text != 'CUTE_ALIGNAS' or tokens[1].text != '('
                or tokens[-1].text != ')' or self.delimiters.get(tokens[1].start) != b):
            return ['alignment_invocation_not_exact_balanced_source']
        expression_span = attribute['expression_span']
        expected_expression_span = [tokens[1].end, tokens[-1].start]
        expression = self.source[expected_expression_span[0]:expected_expression_span[1]].decode('utf-8')
        if (attribute.get('kind') != 'cute_requested_alignment' or attribute.get('path') != self.path
                or expression_span.get('path') != self.path
                or c.get('raw_sha256') != sha(self.source[a:b])
                or not self.alignment_span_verified(attribute) or not self.alignment_span_verified(expression_span)
                or span(attribute) != [a, b] or expression_span.get('path') != self.path
                or span(expression_span) != expected_expression_span
                or attribute.get('alignment_expression') != expression or not expression.strip()
                or attribute.get('semantic_spelling') != self.source[a:b].decode('utf-8')
                or attribute.get('alignment_unit') != 'bytes'
                or not attribute.get('value_evaluation', '').startswith('not_evaluated')):
            errors.append('alignment_attribute_or_expression_not_exact_source')
        # Actual fixed uses have a single identifier or integer. Do not invent
        # semantics for comma lists or newly encountered expression forms.
        if len(self.source_tokens(*expected_expression_span)) != 1:
            errors.append('alignment_expression_outside_reviewed_fixed_forms')
        local = self.proof_view(self.path)
        if not self.alignment_conditions_match(attribute.get('conditions', []), local.conditions_at(a)):
            errors.append('alignment_call_conditions_not_source_verified')
        hint = attribute['owner_hint']
        class_keys = {'struct_specifier': 'struct', 'class_specifier': 'class', 'union_specifier': 'union'}
        expected_kind = 'type' if owner['kind'] in class_keys else 'field' if owner['kind'] in {'member', 'member_constant'} else None
        name_anchor = self.declaration_name_anchor(owner)
        if (expected_kind is None or hint.get('kind') != expected_kind
                or hint.get('syntax_kind') != (owner['kind'] if expected_kind == 'type' else 'field_declaration')
                or hint['declaration_span'].get('path') != self.path
                or not self.alignment_span_verified(hint['declaration_span'])
                or span(hint['declaration_span']) != owner.get('syntax')
                or attribute.get('owner_occurrence_id') != owner['id']
                or attribute.get('owner_entity_id') != owner['entity_id']
                or not owner['signature'][0] <= a < b <= owner['signature'][1]
                or name_anchor is None or b > name_anchor[0]):
            errors.append('alignment_owner_identity_range_or_name_not_verified')
        else:
            prefix = tuple(t.text for t in self.source_tokens(owner['syntax'][0], a))
            if expected_kind == 'type' and prefix != (class_keys[owner['kind']],):
                errors.append('alignment_not_immediately_after_class_key')
            if expected_kind == 'field' and (prefix or hint.get('syntax_kind') != 'field_declaration'):
                errors.append('alignment_not_fixed_field_prefix')
        errors.extend(self.invalid_occurrence(owner))
        if self.blockers(a, b):
            errors.append('alignment_has_blocking_source_diagnostic')

        provider_path = 'include/cute/container/alignment.hpp'
        provider = self.proof_view(provider_path)
        variants = attribute['conditional_expansions']
        bodies = {'__align__(n)', 'alignas(n)'}
        if len(variants) != 2 or {v['definition']['body'] for v in variants} != bodies:
            errors.append('alignment_definition_alternatives_incomplete')
        allowed_definitions = set()
        for variant in variants:
            definition = variant['definition']
            checked = provider.definition({**definition, 'preprocessor_conditions': variant.get('conditions', [])})
            if (definition.get('path') != provider_path or definition.get('name') != 'CUTE_ALIGNAS'
                    or not self.alignment_span_verified(definition)
                    or definition.get('parameters') != ['n'] or definition.get('body') not in bodies
                    or variant.get('definition_source_sha256') != sha(provider.data) or checked is None):
                errors.append('alignment_definition_range_body_or_condition_not_verified')
                continue
            actual, guards = checked
            for condition in variant.get('conditions', []):
                location = condition.get('directive_span', {})
                if (not self.alignment_span_verified(location) or condition.get('constant_false', False)
                        or not any(d['physical_range'] == span(location) and d['directive'] == condition.get('directive')
                                   for d in provider.directives)):
                    errors.append('alignment_definition_condition_location_not_verified')
            wanted = ('defined', '__CUDACC__') if definition['body'] == '__align__(n)' else ('!', ('defined', '__CUDACC__'))
            if len(guards) != 1 or condition_form(guards[0]['expression']) != wanted:
                errors.append('alignment_definition_not_the_two_cudacc_branches')
            if variant.get('expanded_spelling') != definition['body'].replace('(n)', '(' + expression + ')'):
                errors.append('alignment_expanded_spelling_not_parameter_substitution')
            allowed_definitions.add(actual['byte_range'][0])

        chain = attribute['definition_include_chain']
        if self.path == provider_path:
            if chain:
                errors.append('alignment_provider_self_chain_not_empty')
        elif not chain or chain[0].get('path') != self.path or chain[-1].get('target_path') != provider_path:
            errors.append('alignment_definition_include_chain_missing')
        previous = self.path
        checked_views = {self.path: (local, a), provider_path: (provider, len(provider.data))}
        for index, edge in enumerate(chain):
            if edge.get('path') != previous:
                return errors + ['alignment_include_chain_source_path_disagrees']
            view = self.proof_view(edge['path'])
            matches = [d for d in view.directives if d['directive'] == 'include'
                       and d['line_range'] == [edge['start_line'], edge['end_line']]]
            directive = matches[0] if len(matches) == 1 else None
            literal = re.fullmatch(r'[<"]([^>"]+)[>"](?:\s*//[^\n]*)?', directive['expression']) if directive else None
            if (edge['path'] != previous or not literal or 'include/' + literal[1] != edge.get('target_path')
                    or view.conditions_at(directive['physical_range'][0])
                    or index == 0 and directive['physical_range'][1] > a):
                return errors + ['alignment_include_edge_not_unconditional_exact_source']
            previous = edge.get('target_path')
            if edge['path'] != self.path:
                checked_views[edge['path']] = (view, len(view.data))
        # Bounded to the actual source and cited include chain. A local change
        # invalidates the fixed provider contract; no general macro execution.
        for path, (view, before) in checked_views.items():
            for directive in view.directives:
                if directive['physical_range'][0] >= before:
                    continue
                if directive['directive'] not in {'define', 'undef'} or not re.match(r'(?:CUTE_ALIGNAS|__align__)\b', directive['expression']):
                    continue
                if path == provider_path and directive['directive'] == 'define' and directive['physical_range'][0] in allowed_definitions:
                    continue
                errors.append('alignment_local_macro_binding_changed')
        return errors

    def alignment_mapping(self, c):
        if c.get('name') != 'CUTE_ALIGNAS' or c.get('origin') == 'macro_body':
            return None
        a, b = c['byte_range']
        matches, metadata, errors = [], [], []
        for o in self.occurrences:
            for attribute in o.get('alignment_specifiers', []):
                if attribute.get('kind') == 'cute_requested_alignment' and span(attribute) == [a, b]:
                    try:
                        errors.extend(self.alignment_proof_errors(c, attribute, o))
                    except (KeyError, TypeError, ValueError, OSError, IndexError, AttributeError):
                        errors.append('alignment_metadata_or_source_proof_unavailable')
                    matches.append(o)
                    metadata.append(attribute)
        if not matches:
            # An explicit unconditional undef can expose an ordinary C++
            # function with this spelling. Let normal declaration/call rules
            # inspect it, rather than claiming an alignment attribute.
            view = self.proof_view(self.path)
            prior = [d for d in view.directives if d['physical_range'][0] < a
                     and d['directive'] in {'define', 'undef'} and re.match(r'CUTE_ALIGNAS\b', d['expression'])]
            if prior and prior[-1]['directive'] == 'undef' and not view.conditions_at(prior[-1]['physical_range'][0]):
                return None
            errors.append('alignment_owner_attribute_metadata_missing')
        header = self.file_record.get('syntax_analysis', {}).get('header_analysis', {})
        records = [v for v in header.get('alignment_attributes', [])
                   if v.get('kind') == 'cute_requested_alignment' and span(v) == [a, b]]
        if len(records) != 1:
            errors.append('alignment_file_attribute_record_missing_or_ambiguous')
        else:
            record = records[0]
            if (set(record.get('owner_occurrence_ids', [])) != {o['id'] for o in matches}
                    or set(record.get('owner_entity_ids', [])) != {o['entity_id'] for o in matches}):
                errors.append('alignment_file_owner_bindings_incomplete')
            keys = ('attribute_id', 'semantic_spelling', 'alignment_expression', 'expression_span', 'conditions',
                    'conditional_expansions', 'definition_include_chain', 'owner_hint', 'alignment_unit', 'value_evaluation')
            if any(any(m.get(k) != record.get(k) for k in keys) for m in metadata):
                errors.append('alignment_occurrence_and_file_provenance_disagree')
        errors.extend(self.binding_completeness(matches))
        return self.row(c, 'pending' if errors else 'classified_non_api', 'phase1_declaration_attribute',
                        'source_proven_cute_alignment_attribute' if not errors else 'cute_alignment_attribute_proof_incomplete',
                        matches, errors, classification='requested_alignment_attribute', alignment_specifiers=metadata)

    def namespace_proof(self, resolution, owner):
        status = resolution.get("status")
        result = {"errors": [], "conditions": [], "choices": {}, "expected_choices": {},
                  "spelling": resolution.get("source_alias_spelling", resolution.get("spelling")),
                  "parameterized": status == "parameterized_macro_binding", "resolution": resolution}
        errors = result["errors"]
        if status not in {"literal_macro_binding", "parameterized_macro_binding"}:
            errors.append("namespace_binding_requires_explicit_source_proof")
            return result
        chain = resolution.get("definition_chain", [])
        if not chain or not result["spelling"]:
            errors.append("namespace_binding_definition_chain_missing")
            return result
        verified = []
        for definition in chain:
            try:
                view = self.proof_view(definition["path"])
                proof = view.definition(definition)
            except (KeyError, ValueError, OSError):
                proof = None
            if not proof:
                errors.append("namespace_definition_range_body_parameters_or_guard_not_source_verified")
            else:
                verified.append((definition, proof[1]))
                if definition["path"] == self.path and definition["end_byte"] > owner["range"][0]:
                    errors.append("namespace_definition_does_not_precede_owner")
        if len(verified) != len(chain):
            return result
        if chain[0]["name"] != result["spelling"]:
            errors.append("namespace_source_spelling_does_not_start_definition_chain")
        signature_tokens = [t.text for t in self.source_tokens(*owner["signature"])]
        if result["spelling"] not in signature_tokens:
            errors.append("namespace_macro_spelling_absent_from_physical_header")
        local = self.proof_view(self.path)
        for directive in local.directives:
            if directive["physical_range"][0] >= owner["range"][0]:
                continue
            if directive["directive"] == "undef" and directive["expression"].strip() == result["spelling"]:
                conditions = local.conditions_at(directive["physical_range"][0])
                if not any(condition_form(p["expression"]) == condition_form("0") for p in conditions):
                    cutoff = chain[0]["end_byte"] if chain[0]["path"] == self.path else -1
                    if directive["physical_range"][0] >= cutoff:
                        errors.append("namespace_binding_lifetime_after_undef_not_proven")
        if chain[0]["path"] != self.path and any(d["name"] == result["spelling"] and d["byte_range"][0] < owner["range"][0] for d in local.scan.definitions):
            errors.append("imported_namespace_binding_has_unchecked_local_shadow")
        guards = list({proof_key(g): g for _, gs in verified for g in gs}.values())
        key, choice = resolution.get("binding_key"), resolution.get("choice_key")
        special_cutlass = result["spelling"] == "cutlass" and any(d.get("parameters") is not None for d in chain)
        if special_cutlass:
            required = [("cutlass", None, "mkcutlassnamespace(cutlass_, CUTLASS_NAMESPACE)"),
                        ("mkcutlassnamespace", ["pre", "ns"], "concat_tok(pre, ns)"),
                        ("concat_tok", ["a", "b"], "a ## b")]
            helper = "include/cutlass/detail/helper_macros.hpp"
            if len(chain) != 3 or any(d["path"] != helper or d["name"] != name or d.get("parameters") != params or token_spelling(d["body"]) != token_spelling(body) for d, (name, params, body) in zip(chain, required)):
                errors.append("namespace_token_paste_chain_not_verified")
            if len(guards) != 1 or condition_form(guards[0]["expression"]) != condition_form("defined(CUTLASS_NAMESPACE)"):
                errors.append("namespace_customization_guard_not_verified")
            if key != helper + "::cutlass":
                errors.append("namespace_binding_key_not_tied_to_definition")
            result["expected_choices"][key] = {"default_namespace", "custom_namespace"}
            if status == "parameterized_macro_binding":
                expression = 'rescan(token_paste("cutlass_", expand(CUTLASS_NAMESPACE)))'
                if choice != "custom_namespace" or resolution.get("is_literal_namespace") is not False or token_spelling(resolution.get("scope_expression", "")) != token_spelling(expression):
                    errors.append("parameterized_namespace_expression_not_derived_from_macro_chain")
                missing = " ".join(resolution.get("missing_bindings", []))
                if "CUTLASS_NAMESPACE" not in missing or "rescan" not in missing or not resolution.get("requirements"):
                    errors.append("parameterized_namespace_missing_binding_or_result_constraint_absent")
                components = ["<cutlass_namespace(CUTLASS_NAMESPACE)>"]
                result["conditions"] = guards
            else:
                if choice != "default_namespace" or resolution.get("expanded_namespace") != "cutlass" or resolution.get("missing_bindings") != []:
                    errors.append("namespace_default_choice_not_verified")
                components = ["cutlass"]
                result["conditions"] = [{**g, "expression": "!(" + g["expression"] + ")"} for g in guards]
        elif status == "literal_macro_binding":
            if any(d.get("parameters") is not None for d in chain) or any(token_spelling(chain[i]["body"]) != (chain[i + 1]["name"],) for i in range(len(chain) - 1)):
                errors.append("literal_namespace_alias_chain_not_verified")
            final = chain[-1]["body"].strip()
            if not re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", final) or resolution.get("expanded_namespace") != final or resolution.get("missing_bindings") != []:
                errors.append("literal_namespace_result_not_equal_to_replacement_tokens")
            components = final.split("::")
            result["conditions"] = guards
            if result["spelling"] == "CUTE_STL_NAMESPACE" and chain[0]["path"] == "include/cute/config.hpp":
                expected_key = "include/cute/config.hpp::CUTE_STL_NAMESPACE"
                actual = {self.proof_view(chain[0]["path"]).data[d["body_byte_range"][0]:d["body_byte_range"][1]].decode().strip() for d in self.proof_view(chain[0]["path"]).scan.definitions if d["name"] == "CUTE_STL_NAMESPACE"}
                if actual != {"std", "cuda::std"} or key != expected_key or choice != final or not guards:
                    errors.append("stl_namespace_alternatives_or_guard_not_verified")
                result["expected_choices"][key] = {"std", "cuda::std"}
            elif key or choice:
                errors.append("unmodeled_namespace_choice_key_requires_review")
        else:
            errors.append("unmodeled_parameterized_namespace_requires_review")
            components = []
        index = owner.get("expanded_namespace_component")
        if (index is not None and (not 0 <= index < len(components) or owner["name"] != components[index])) or (index is None and owner["name"] not in components):
            errors.append("bound_namespace_component_name_not_equal_to_proven_result")
        if key:
            result["choices"][key] = choice
        return result

    def binding_info(self, o):
        if o["id"] in self._binding_infos:
            return self._binding_infos[o["id"]]
        info = {"errors": [], "conditions": [], "choices": {}, "expected_choices": {}, "proofs": []}
        owners = []
        for scope in o.get("scope_chain", []):
            resolution = scope.get("name_resolution") or {}
            if resolution.get("status", "source_name") == "source_name":
                continue
            matching = [p for p in self.namespace_owners.get(scope.get("entity_id"), []) if p["name"] == scope.get("name") and p["range"][0] <= o["range"][0] <= o["range"][1] <= p["range"][1] and p.get("name_resolution") == resolution]
            if not matching:
                info["errors"].append("bound_namespace_scope_has_no_matching_physical_owner_occurrence")
            else:
                owners.append(min(matching, key=lambda p: p["range"][1] - p["range"][0]))
        if o["kind"] == "namespace" and o.get("name_resolution", {}).get("status", "source_name") != "source_name":
            owners.append(o)
        for owner in owners:
            proof = self.namespace_proof(owner["name_resolution"], owner)
            info["proofs"].append(proof)
            info["errors"].extend(proof["errors"])
            for key, value in proof["choices"].items():
                if key in info["choices"] and info["choices"][key] != value:
                    info["errors"].append("conflicting_namespace_choices_for_one_binding_key")
                if key not in info["choices"]:
                    info["conditions"].extend(proof["conditions"])
                info["choices"][key] = value
            if not proof["choices"]:
                info["conditions"].extend(proof["conditions"])
            info["expected_choices"].update(proof["expected_choices"])
        if o.get("qualified_name_resolution") not in {None, "source_name"} and not owners:
            info["errors"].append("qualified_namespace_status_has_no_positive_binding_origin")
        if o.get("qualified_name_resolution") == "parameterized_macro_binding" and not any(p["parameterized"] for p in info["proofs"]):
            info["errors"].append("parameterized_namespace_status_without_parameterized_source_expression")
        if owners and (not o.get("source_id") or o["source_id"] == o["id"]):
            info["errors"].append("physical_source_and_binding_instance_identity_not_separated")
        if o.get("namespace_choices") is not None and dict(o["namespace_choices"]) != info["choices"]:
            info["errors"].append("recorded_namespace_choices_disagree_with_verified_scope_bindings")
        self._binding_infos[o["id"]] = info
        return info

    def conditional_info(self, o):
        if o["id"] in self._conditional_infos:
            return self._conditional_infos[o["id"]]
        origin = o.get("conditional_origin") or {}
        if origin.get('kind')=='function_header_variant':
            info=self.conditional_function_info(o)
            self._conditional_infos[o['id']]=info
            return info
        info = {"errors": [], "segments": [], "conditions": [], "variant_key": None, "expected_variants": []}
        if origin.get("kind") != "alias":
            info["errors"].append("conditional_declaration_form_requires_dedicated_proof:" + str(origin.get("kind")))
            self._conditional_infos[o["id"]] = info
            return info
        errors = info["errors"]
        try:
            physical = origin["declaration_span"]
            start, end = physical["start_byte"], physical["end_byte"]
            if physical["path"] != self.path or [start, end] != o["signature"] or not 0 <= start < end <= len(self.source):
                raise ValueError("conditional declaration physical range mismatch")
            if origin.get("kind") != "alias" or o["kind"] != "alias" or origin.get("name") != o["name"] or not origin.get("source_declaration_id") or not origin.get("conditional_variant_id"):
                raise ValueError("conditional declaration identity/kind mismatch")
            if not o.get("source_id") or o["source_id"] == o["id"]:
                errors.append("physical_source_and_binding_instance_identity_not_separated")
            if origin.get("semantic_spelling") != self.source[start:end].decode("utf-8", "replace"):
                raise ValueError("conditional semantic spelling is not original source")
            virtual = origin["virtual_source"].encode()
            cursor, pieces = 0, []
            for segment in origin["segments"]:
                piece = segment["physical_span"]
                a, b = piece["start_byte"], piece["end_byte"]
                if piece["path"] != self.path or not start <= a <= b <= end or segment.get("mapping") != "exact_source_slice" or segment["virtual_start_byte"] != cursor:
                    raise ValueError("conditional segment location/sequence mismatch")
                payload = self.source[a:b]
                cursor += len(payload)
                if segment["virtual_end_byte"] != cursor:
                    raise ValueError("conditional segment byte length mismatch")
                pieces.append(payload)
                if a != b:
                    info["segments"].append((a, b))
            if b"".join(pieces) != virtual or (o.get("expanded_signature") or "").rstrip() != virtual.decode().rstrip():
                raise ValueError("conditional virtual/expanded text differs from exact source pieces")
            words = token_spelling(virtual.decode())
            if len(words) < 5 or words[:3] != ("using", o["name"], "=") or words[-1] != ";" or token_spelling(origin.get("target_type") or "") != words[3:-1] or token_spelling(o.get("target_type") or "") != words[3:-1]:
                raise ValueError("conditional alias target is not the reconstructed type expression")
            if origin.get("status") != "parsed" or origin.get("parse_errors"):
                errors.append("conditional_variant_still_has_parse_gap")
            projections = self.proof_view(self.path).conditional_projections(start, end)
            def key(segments, conditions):
                return (tuple(segments), tuple(sorted((proof_key(c) for c in conditions), key=repr)))
            info["expected_variants"] = [key(s, c) for s, c in projections]
            for segments, conditions in projections:
                claimed = origin.get("conditions", [])
                if tuple(segments) == tuple(info["segments"]) and all(any(condition_matches(c, p) for c in claimed) for p in conditions) and all(any(condition_matches(c, p) for p in conditions) for c in claimed):
                    info["conditions"], info["variant_key"] = conditions, key(segments, conditions)
                    break
            if info["variant_key"] is None:
                errors.append("conditional_segments_or_guards_do_not_match_any_source_branch")
        except (KeyError, TypeError, ValueError, UnicodeError) as error:
            errors.append("conditional_piecewise_source_proof_failed:" + str(error))
        self._conditional_infos[o["id"]] = info
        return info

    def conditional_function_info(self,o):
        """Verify selected header text AND the untouched shared body separately."""
        origin=o['conditional_origin']
        info={'errors':[],'segments':[],'conditions':[],'variant_key':None,'expected_variants':[], 'is_root_function':False}
        try:
            outer,signature=origin['declaration_span'],origin['signature_span']
            start,end=outer['start_byte'],outer['end_byte'];hs,he=signature['start_byte'],signature['end_byte']
            if outer['path']!=self.path or signature['path']!=self.path or start!=hs or not 0<=start<he<end<=len(self.source):
                raise ValueError('function declaration/header source bounds differ')
            if origin['semantic_spelling']!=self.source[hs:he].decode() or origin.get('status')!='header_parsed':
                raise ValueError('function source header is changed or unparsed')
            body=origin.get('body_span')
            if body:
                if body['path']!=self.path or [body['start_byte'],body['end_byte']]!=[he,end] or self.braces.get(he)!=end:
                    raise ValueError('shared function body is not the full balanced source body')
            elif self.source[he:end]!=b';':raise ValueError('prototype terminator is not exact')
            cursor=0;pieces=[];headers=[];header_segments=[];tails=[]
            for segment in origin['segments']:
                physical=segment['physical_span'];a,b=physical['start_byte'],physical['end_byte']
                if physical['path']!=self.path or not start<=a<=b<=end or segment.get('mapping')!='exact_source_slice' or segment['virtual_start_byte']!=cursor:
                    raise ValueError('function source segment sequence is invalid')
                piece=self.source[a:b];cursor+=len(piece)
                if segment['virtual_end_byte']!=cursor:raise ValueError('function source segment byte length differs')
                pieces.append(piece)
                if segment['role']=='header':
                    if b>he:raise ValueError('header segment overlaps shared body')
                    headers.append(piece)
                    if a<b:header_segments.append((a,b))
                elif segment['role']in {'shared_body','prototype_terminator'}:tails.append((a,b))
                else:raise ValueError('unknown function source segment role')
            if tails!=[(he,end)] or b''.join(pieces)!=origin['virtual_source'].encode() or b''.join(headers)!=origin['virtual_signature'].encode():
                raise ValueError('virtual function text loses/changes a header or shared body piece')
            projections=self.proof_view(self.path).conditional_projections(hs,he)
            def key(spans,conditions):return(tuple(spans),tuple(sorted((proof_key(c)for c in conditions),key=repr)))
            info['expected_variants']=[key(s,c)for s,c in projections]
            for spans,conditions in projections:
                claimed=origin.get('conditions',[])
                if tuple(spans)==tuple(header_segments)and all(any(condition_matches(c,p)for c in claimed)for p in conditions)and all(any(condition_matches(c,p)for p in conditions)for c in claimed):
                    info['variant_key']=key(spans,conditions);info['conditions']=conditions;break
            if info['variant_key']is None:raise ValueError('function header branch or guard differs from independent source branches')
            info['segments']=header_segments
            name=origin.get('signature',{}).get('name')
            words=list(token_spelling(origin['virtual_signature']))
            if not name or not re.fullmatch(r'[A-Za-z_]\w*',name)or head_kind(LEXER.lex(origin['virtual_signature'].encode())[0])!='callable':
                raise ValueError('conditional header is not a verified simple named callable')
            # A top-level name followed by (...) rejects a function-pointer
            # object and cannot be borrowed from a call nested in decltype.
            nesting=0;anchors=[]
            for i,word in enumerate(words):
                if nesting==0 and word==name and i+1<len(words)and words[i+1]=='(':anchors.append(i)
                if word in {'(','['}:nesting+=1
                elif word in {')',']'}:nesting-=1
            if len(anchors)!=1:raise ValueError('callable name has no unique top-level declaration anchor')
            root=(o['kind']in CALLABLE and o.get('name')==name and (o.get('body')==[he,end]if body else o['range'][1]==end and o['signature'][0]<he))
            info['is_root_function']=root
            if root:
                if o['signature'][1]!=(he if body else end)or o['range'][1]!=end:raise ValueError('root function signature/body is truncated or mixed')
                expected=origin['virtual_signature'].rstrip()+(''if body else';')
                if token_spelling(o.get('expanded_signature')or'')!=token_spelling(expected):raise ValueError('expanded function signature differs from source header')
                prefix=words[:anchors[0]]
                cleaned=[];i=0
                while i<len(prefix):
                    if prefix[i]=='__attribute__'and i+1<len(prefix)and prefix[i+1]=='(':
                        end=i+2;balance=1
                        while end<len(prefix)and balance:
                            balance+=(prefix[end]=='(')-(prefix[end]==')');end+=1
                        if balance or not any(tuple(prefix[i:end])==token_spelling(q)for q in o.get('qualifiers',[])):
                            raise ValueError('GNU declaration attribute is unbalanced or not retained in qualifiers')
                        i=end
                    else:cleaned.append(prefix[i]);i+=1
                prefix=cleaned
                specifiers={'static','inline','constexpr','consteval','virtual','explicit','extern','friend',
                    'CUTLASS_HOST_DEVICE','CUTLASS_HOST','CUTLASS_DEVICE','CUTLASS_GLOBAL','CUTE_HOST_DEVICE','CUTE_HOST','CUTE_DEVICE',
                    '__host__','__device__','__global__','__forceinline__'}
                if 'template'in prefix or '->'in words:raise ValueError('template/trailing-return conditional header needs additional type proof')
                if tuple(w for w in prefix if w not in specifiers)!=token_spelling(o.get('return_type')or''):
                    raise ValueError('function return type is not the exact source type')
                depth=1;closing=anchors[0]+2
                while closing<len(words)and depth:
                    depth+=(words[closing]=='(')-(words[closing]==')');closing+=1
                if depth:raise ValueError('parameter list is unbalanced')
                parameters=[];part=[];depth=0
                for word in words[anchors[0]+2:closing-1]:
                    if word==','and depth==0:parameters.append(part);part=[];continue
                    part.append(word)
                    if word in {'(','[','{','<'}:depth+=1
                    elif word in {')',']','}','>'}:depth-=1
                if part:parameters.append(part)
                if len(parameters)!=len(o.get('parameters',[])):raise ValueError('function parameter count differs from source')
                for source_parameter,p in zip(parameters,o['parameters']):
                    equal=source_parameter.index('=')if '='in source_parameter else len(source_parameter)
                    type_tokens=source_parameter[:equal]
                    if p.get('name'):
                        hits=[i for i,w in enumerate(type_tokens)if w==p['name']]
                        if len(hits)!=1:raise ValueError('parameter name lacks a unique source anchor')
                        type_tokens=type_tokens[:hits[0]]+type_tokens[hits[0]+1:]
                    if tuple(type_tokens)!=token_spelling(p.get('type')or'')or tuple(source_parameter[equal+1:])!=token_spelling(p.get('default')or''):
                        raise ValueError('parameter type/default differs from source')
            else:
                if not body or not he<o['signature'][0]<=o['signature'][1]<end:raise ValueError('nested declaration is outside shared body')
                conditions=self.proof_view(self.path).conditions_at(o['signature'][0])
                info['conditions']=list({proof_key(c):c for c in info['conditions']+conditions}.values())
        except(KeyError,TypeError,ValueError,UnicodeError)as error:
            info['errors'].append('conditional_function_source_proof_failed:'+str(error))
        return info

    def source_name(self, o):
        resolution = o.get("name_resolution") or {}
        if o["kind"] == "namespace" and resolution.get("status") in {"literal_macro_binding", "parameterized_macro_binding"}:
            return resolution.get("source_alias_spelling", resolution.get("spelling", ""))
        return o["name"]

    def bitfield_errors(self, o):
        if o["id"] in self._bitfield_infos:
            return list(self._bitfield_infos[o["id"]])
        errors = []
        origin = o.get("bitfield_origin") or {}
        try:
            if o["kind"] not in {"member", "member_constant"} or not o.get("is_bitfield") or not o.get("bitfield_source_id") or origin.get("bitfield_source_id") != o["bitfield_source_id"]:
                raise ValueError("bitfield identity/kind is missing or inconsistent")
            declaration, declarator = origin["declaration"], origin["declarator"]
            for field in (declaration, declarator):
                a, b = field["start_byte"], field["end_byte"]
                if field["path"] != self.path or not 0 <= a <= b <= len(self.source) or field["raw"] != self.source[a:b].decode():
                    raise ValueError("bitfield physical text/range differs from source")
            if span(declaration) != o["signature"] or span(declarator) != o.get("declarator"):
                raise ValueError("bitfield occurrence ranges differ from standalone source origin")
            colon, width = origin["colon_range"], origin["bit_width_range"]
            c, ce = colon["start_byte"], colon["end_byte"]
            a, b = width["start_byte"], width["end_byte"]
            if colon["path"] != self.path or width["path"] != self.path or [c, ce] != o.get("colon_range") or [a, b] != o.get("bit_width_range"):
                raise ValueError("bitfield colon/width source location mismatch")
            if ce != c + 1 or self.source[c:ce] != b":" or not declaration["start_byte"] <= c < a < b <= declarator["end_byte"]:
                raise ValueError("bitfield colon or width bounds invalid")
            if self.source[a:b].decode() != o.get("bit_width") or origin.get("bit_width") != o.get("bit_width"):
                raise ValueError("bitfield width is not the exact source expression")
            if origin.get("name") != o.get("name"):
                raise ValueError("bitfield name disagrees with source origin")
            if o.get("name") is None:
                if origin.get("name_range") is not None or declarator["start_byte"] != c or origin.get("kind") != "anonymous_bitfield":
                    raise ValueError("anonymous bitfield has an invented name or wrong colon anchor")
            else:
                named = origin["name_range"]
                if named["path"] != self.path or self.source[named["start_byte"]:named["end_byte"]].decode() != o["name"] or named["end_byte"] > c:
                    raise ValueError("named bitfield name anchor invalid")
            owner = origin["owner"]
            bounds = span(owner["range"])
            compatible_owner = [p for scope in o.get("scope_chain", []) for p in self.type_owners.get(scope.get("entity_id"), [])
                                if p["kind"] == owner["kind"] and p.get("syntax") == bounds
                                and head_kind(self.source_tokens(*p["signature"])) == "type"
                                and self.braces.get(p["signature"][1]) == bounds[1]]
            if owner["kind"] not in {"struct_specifier", "class_specifier", "union_specifier"} or owner["range"]["path"] != self.path or not compatible_owner:
                raise ValueError("bitfield has no exact physical class/struct/union owner")
            if not bounds[0] < declaration["start_byte"] <= c < b < bounds[1]:
                raise ValueError("bitfield lies outside its source type owner")
            prefix_tokens = self.source_tokens(declaration["start_byte"], c)
            if origin.get("declarator_index", 0) == 0 and (head_kind(prefix_tokens) == "callable" or any(t.text in {"?", "="} for t in prefix_tokens)):
                raise ValueError("first bitfield colon could belong to a function header or initializer")
            if origin.get("initializer_range"):
                init = origin["initializer_range"]
                if init["start_byte"] < b or self.source[init["start_byte"]:init["end_byte"]].decode() != o.get("initializer"):
                    raise ValueError("bitfield initializer overlaps width or differs from source")
            elif o.get("initializer") is not None:
                raise ValueError("bitfield initializer has no separate source range")
        except (KeyError, ValueError, TypeError, UnicodeError) as error:
            errors.append("bitfield_source_proof_failed:" + str(error))
        self._bitfield_infos[o["id"]] = errors
        return list(errors)

    def condition_errors(self, o, candidate=None):
        if "conditions" not in o:  # Old fixture/schema compatibility only.
            expected = [self.branch_lines.get(ref) for ref in (candidate or {}).get("conditions", [])]
            return [] if candidate is None or expected == o.get("condition_lines", []) else ["candidate_and_occurrence_branch_disagree"]
        binding = self.binding_info(o)
        source_conditions = self.proof_view(self.path).conditions_at(o["signature"][0])
        if o.get("conditional_origin"):
            source_conditions = self.conditional_info(o)["conditions"]
        expected = source_conditions + binding["conditions"]
        observed = o["conditions"]
        errors = []
        if any(not any(condition_matches(c, p) for c in observed) for p in expected):
            errors.append("required_source_or_namespace_binding_condition_missing")
        if any(not any(condition_matches(c, p) for p in expected) for c in observed):
            errors.append("occurrence_condition_has_no_source_or_namespace_binding_proof")
        if o.get("namespace_conditions") is not None:
            if any(not any(condition_matches(c, p) for c in o["namespace_conditions"]) for p in binding["conditions"]) or any(not any(condition_matches(c, p) for p in binding["conditions"]) for c in o["namespace_conditions"]):
                errors.append("recorded_namespace_conditions_disagree_with_binding_proof")
        if candidate is not None and not o.get("conditional_origin"):
            expected_lines = [self.branch_lines.get(ref) for ref in candidate.get("conditions", [])]
            if expected_lines != [p["line"] for p in source_conditions]:
                errors.append("candidate_and_occurrence_local_branch_disagree")
        return errors

    def binding_completeness(self, occurrences):
        groups = defaultdict(list)
        for o in occurrences:
            key = (tuple(o["signature"]), o["kind"], o.get("bitfield_source_id") if o.get("is_bitfield") else self.source_name(o))
            groups[key].append(o)
        errors = []
        for members in groups.values():
            expected = {}
            for o in members:
                expected.update(self.binding_info(o)["expected_choices"])
            keys = sorted(expected)
            wanted = set(itertools.product(*(sorted(expected[k]) for k in keys)))
            actual = {tuple(self.binding_info(o)["choices"].get(k) for k in keys) for o in members}
            if wanted != actual:
                errors.append("namespace_binding_instance_alternatives_incomplete")
        return errors

    def conditional_mapping(self, c):
        if c["kind"] not in {"syntax_interval", "macro_invocation"}:
            return None
        a, b = c["byte_range"]
        if not any(t.text not in SCAFFOLD for t in self.source_tokens(a, b)):
            return None
        for group in self.conditional_groups.values():
            if all(o['conditional_origin'].get('kind')=='function_header_variant'for o in group):
                origin=group[0]['conditional_origin'];header=origin.get('signature_span',{})
                if not header.get('start_byte',-1)<=a<b<=header.get('end_byte',-1):continue
                roots=[o for o in group if self.conditional_info(o).get('is_root_function')]
                errors=[];selected=[];known=set();expected=set()
                for o in roots:
                    info=self.conditional_info(o);errors.extend(self.invalid_occurrence(o))
                    expected.update(info['expected_variants'])
                    if info['variant_key']is not None:known.add(info['variant_key'])
                    meaningful=[t for t in self.source_tokens(a,b)if t.text not in SCAFFOLD]
                    if meaningful and all(any(x<=t.start and t.end<=y for x,y in info['segments'])for t in meaningful):selected.append(o)
                if not roots or expected!=known:errors.append('conditional_function_root_variants_incomplete')
                for variant in expected:errors.extend(self.binding_completeness([o for o in roots if self.conditional_info(o)['variant_key']==variant]))
                if not selected:errors.append('candidate_tokens_not_covered_by_any_function_header_variant')
                if self.blockers(a,b):errors.append('overlapping_blocking_extraction_diagnostic')
                return self.row(c,'pending'if errors else'mapped_occurrences'if c['kind']=='syntax_interval'else'classified_non_api',
                    'phase1_conditional_function_header','header_branches_and_shared_body_verified_separately',selected,errors)
            if any(o["conditional_origin"].get("kind") != "alias" for o in group):
                contained = [o for o in group if o["range"][0] <= a < b <= o["range"][1]]
                if contained:
                    return self.row(c, "pending", "phase1_conditional_declaration_pending", "conditional_non_alias_form_not_silently_reinterpreted", contained,
                                    ["conditional_declaration_form_requires_dedicated_proof:" + str(o["conditional_origin"].get("kind")) for o in contained])
                continue
            origin = group[0]["conditional_origin"]
            outer = origin.get("declaration_span", {})
            if not outer.get("start_byte", -1) <= a < b <= outer.get("end_byte", -1):
                continue
            errors, selected, known_variants, expected_variants = [], [], set(), set()
            for o in group:
                info = self.conditional_info(o)
                errors.extend(self.invalid_occurrence(o))
                expected_variants.update(info["expected_variants"])
                if info["variant_key"] is not None:
                    known_variants.add(info["variant_key"])
                meaningful = [t for t in self.source_tokens(a, b) if t.text not in SCAFFOLD]
                if meaningful and all(any(x <= t.start and t.end <= y for x, y in info["segments"]) for t in meaningful):
                    selected.append(o)
            if expected_variants != known_variants:
                errors.append("conditional_source_branch_variants_incomplete")
            for variant in expected_variants:
                instances = [o for o in group if self.conditional_info(o)["variant_key"] == variant]
                errors.extend(self.binding_completeness(instances))
            if not selected:
                errors.append("candidate_tokens_not_covered_by_any_conditional_source_variant")
            if self.blockers(a, b):
                errors.append("overlapping_blocking_extraction_diagnostic")
            obligation = "phase1_conditional_declaration" if c["kind"] == "syntax_interval" else "phase1_conditional_type_expression"
            status = "pending" if errors else "mapped_occurrences" if c["kind"] == "syntax_interval" else "classified_non_api"
            return self.row(c, status, obligation, "exact_piecewise_source_variant_with_all_branches_and_binding_instances_verified", selected, errors,
                            source_declaration_id=origin.get("source_declaration_id"))
        return None

    def invalid_occurrence(self, o):
        if o["id"] in self._invalid_cache:
            return list(self._invalid_cache[o["id"]])
        errors = []
        if o.get("parse_status") != "parsed":
            errors.append("occurrence_not_cleanly_parsed")
        if o.get('scope_review_required'):
            errors.append('enclosing_scope_requires_source_review')
        if o.get("name") is None:
            errors.extend(self.bitfield_errors(o))
        elif o.get("is_bitfield"):
            errors.extend(self.bitfield_errors(o))
        if o.get("source_id") and o.get("name") is not None and o["qualified_name"] != "::".join([s["name"] for s in o.get("scope_chain", [])] + [o["name"]]):
            errors.append("qualified_name_disagrees_with_verified_scope_components")
        errors.extend(self.binding_info(o)["errors"])
        if o.get("conditional_origin"):
            errors.extend(self.conditional_info(o)["errors"])
        errors.extend(self.condition_errors(o))
        if o.get("qualified_name_resolution") == "macro_dependent" or o.get("name_resolution", {}).get("status") == "macro_dependent" or any(s.get("name_resolution", {}).get("status") == "macro_dependent" for s in o.get("scope_chain", []) if s.get("name_resolution")):
            errors.append("macro_dependent_owner_not_materialized")
        sig = o.get("signature")
        if not sig or not 0 <= sig[0] <= sig[1] <= len(self.source):
            errors.append("invalid_signature_range")
        elif not o.get("macro_origin"):
            if o["kind"] in TYPES | {"namespace"} and any(sig[0] <= opening < sig[1] for opening in self.scope_body_openings):
                errors.append("namespace_or_type_signature_improperly_contains_body")
            if o["kind"] in CALLABLE and o.get("body") and sig[1] > o["body"][0]:
                errors.append("callable_signature_improperly_contains_body")
            raw = self.source[sig[0]:sig[1]].decode("utf-8", "replace").rstrip()
            if sha(raw.encode()) != o.get("raw_signature_sha256"):
                errors.append("signature_not_equal_to_physical_source")
            name = self.source_name(o)
            if name and not name.startswith("<anonymous"):
                name_tokens = [t.text for t in LEXER.lex(name.encode())[0]]
                source_text = [t.text for t in self.source_tokens(*sig)]
                if name_tokens and not any(source_text[i:i + len(name_tokens)] == name_tokens for i in range(len(source_text))):
                    errors.append("occurrence_name_absent_from_own_signature")
        self._invalid_cache[o["id"]] = errors
        return list(errors)

    def row(self, c, status, obligation, rule, occurrences=(), reasons=(), **extra):
        a, b = c["byte_range"]
        instances = []
        for o in occurrences:
            if not o.get("source_id"):
                continue
            binding = self.binding_info(o)
            instances.append({"declaration_occurrence_id": o["id"], "source_occurrence_id": o["source_id"], "entity_id": o["entity_id"],
                              "namespace_choices": binding["choices"],
                              "namespace_binding_kind": "parameterized" if any(p["parameterized"] for p in binding["proofs"]) else "literal" if binding["proofs"] else "source_name",
                              "missing_bindings": sorted({missing for p in binding["proofs"] for missing in p["resolution"].get("missing_bindings", [])}),
                              "conditional_variant_id": (o.get("conditional_origin") or {}).get("conditional_variant_id")})
        return {"candidate_id": c["candidate_id"], "candidate_kind": c["kind"], "path": self.path,
                "byte_range": [a, b], "conditions": c.get("conditions", []),
                "status": status, "obligation": obligation,
                "occurrence_ids": sorted({o["id"] for o in occurrences}),
                "source_occurrence_ids": sorted({o["source_id"] for o in occurrences if o.get("source_id")}),
                "binding_instances": instances,
                "bitfields": [{"declaration_occurrence_id": o["id"], "bitfield_source_id": o["bitfield_source_id"], "name": o["name"],
                               "bit_width": o["bit_width"], "colon_range": o["colon_range"], "bit_width_range": o["bit_width_range"]} for o in occurrences if o.get("is_bitfield")],
                "evidence": {"rule": rule, "source_sha256": c["raw_sha256"],
                             "excerpt": self.source[a:min(b, a + 160)].decode("utf-8", "replace")},
                "pending_reasons": sorted(set(reasons)), **extra}

    def local_body_owner(self, a, b):
        bodies = [self.occurrences[i] for i in self.bodies.overlapping(a, b) if self.occurrences[i]["body"][0] < a and b <= self.occurrences[i]["body"][1]]
        if not bodies:
            return None
        owner = min(bodies, key=lambda o: o["body"][1] - o["body"][0])
        for index in self.type_bodies.overlapping(a, b):
            start = self.candidates[index]["byte_range"][1] - 1
            if owner["body"][0] < start < a:
                return None  # A member of a local class is not a local expression.
        return owner

    def blockers(self, a, b):
        return [self.diagnostics[i] for i in self.diag_index.overlapping(a, b) if self.diagnostics[i].get("blocks_phase_1", True)]

    def match_declarations(self, c, expected, callable_name=None):
        a, b = c["byte_range"]
        matches, errors = [], []
        candidate_text = [t.text for t in self.source_tokens(a, b)]
        condition_lines = [self.branch_lines.get(ref) for ref in c.get("conditions", [])]
        for index in self.signatures.overlapping(a, b):
            o = self.occurrences[index]
            if o.get("macro_origin") or o.get("conditional_origin") or o["kind"] == "macro_definition":
                continue
            if not compatible(o["kind"], expected):
                continue
            if callable_name and (o["kind"] not in CALLABLE or not o.get("name") or o["name"].split("::")[-1] != callable_name):
                continue
            if o.get("name") is None and (not o.get("colon_range") or not a <= o["colon_range"][0] < o["colon_range"][1] <= b):
                continue
            name = self.source_name(o)
            if name and not name.startswith("<anonymous"):
                name_tokens = [t.text for t in LEXER.lex(name.encode())[0]]
                if not any(candidate_text[i:i + len(name_tokens)] == name_tokens for i in range(len(candidate_text))):
                    continue  # Enclosing initializer/signature is not a name anchor.
                if o["id"] not in self._name_anchors:
                    declaration_tokens = self.source_tokens(*(o.get("declarator") or o.get("syntax") or o["signature"]))
                    spellings = [t.text for t in declaration_tokens]
                    hit = next((i for i in range(len(spellings)) if spellings[i:i + len(name_tokens)] == name_tokens), None)
                    self._name_anchors[o["id"]] = None if hit is None else (declaration_tokens[hit].start, declaration_tokens[hit + len(name_tokens) - 1].end)
                anchor = self._name_anchors[o["id"]]
                if not anchor or not a <= anchor[0] <= anchor[1] <= b:
                    continue  # Same spelling in a body is not the declaration name.
            invalid = self.invalid_occurrence(o)
            invalid.extend(self.condition_errors(o, c))
            if invalid:
                errors.extend(invalid)
                continue
            matches.append(o)
        tokens = [t for t in self.source_tokens(a, b) if t.text not in SCAFFOLD]
        uncovered = [t for t in tokens if not any(o["signature"][0] <= t.start and t.end <= o["signature"][1] for o in matches)]
        # Full statement signatures are shared by `int a,b;`. Their union is
        # insufficient evidence that BOTH declarators were extracted. Inspect
        # each top-level comma fragment against actual declarator subranges.
        paren = angle = bracket = 0
        commas = []
        for token in self.source_tokens(a, b):
            word = token.text
            if word == "," and paren == angle == bracket == 0:
                commas.append(token.start)
            elif word == "(": paren += 1
            elif word == ")": paren = max(0, paren - 1)
            elif word == "[": bracket += 1
            elif word == "]": bracket = max(0, bracket - 1)
            elif word == "<": angle += 1
            elif word == ">": angle = max(0, angle - 1)
        ranges = [o["declarator"] for o in matches if o.get("declarator")]
        comma_declarations = [o for o in matches if o["kind"] in (VALUES - {"enumerator"}) | (CALLABLE - {"constructor"}) | {"typedef"}]
        if commas and comma_declarations and any(not o.get("declarator") for o in comma_declarations):
            errors.append("comma_declarators_require_explicit_ranges")
        if commas and ranges and comma_declarations:
            if min(x[0] for x in ranges) > commas[0]:
                errors.append("first_comma_declarator_has_no_occurrence")
            after_comma = [t for t in tokens if t.start > commas[0]]
            missed = [t for t in after_comma if not any(x <= t.start and t.end <= y for x, y in ranges)]
            if missed:
                errors.append("comma_declarator_tokens_missing_from_declarator_ranges")
                uncovered.extend(t for t in missed if t not in uncovered)
        errors.extend(self.binding_completeness(matches))
        return matches, uncovered, errors

    def macro_mapping(self, c):
        a, b = c["byte_range"]
        generated = self.macros.get((a, b), [])
        if not generated:
            return None
        errors = [error for o in generated for error in self.invalid_occurrence(o)]
        errors.extend(error for o in generated for error in self.condition_errors(o, c))
        errors.extend(self.binding_completeness(generated))
        if self.blockers(a, b):
            errors.append("macro_invocation_has_blocking_diagnostic")
        kinds = Counter(o["kind"] for o in generated)
        definitions = {d["candidate_id"]: d for d in self.macro_definitions.values()}
        selected = []
        for o in generated:
            definition = o["macro_origin"]["definition"]
            matches = [definitions[ref] for ref in c.get("definition_candidates", []) if ref in definitions
                       and definitions[ref]["path"] == definition["path"]
                       and definitions[ref]["byte_range"][0] <= definition["start_byte"] < definitions[ref]["byte_range"][1]]
            if not matches:
                errors.append("macro_origin_has_no_frozen_definition_candidate")
            selected.extend(matches)
        required = set(hint for d in selected for hint in d.get("declaration_generation_hints", []))
        for hint, wanted in {"operator": {"operator"}, "struct": {"struct_specifier"}, "enum": {"enum"}, "namespace": {"namespace"}, "using": ALIASES, "typedef": {"typedef"}}.items():
            if hint in required and not any(kinds[k] for k in wanted):
                errors.append("macro_generated_role_missing:" + hint)
        expansions = [e for e in self.file_record.get("macro_expansions", []) if e.get("start_byte") == a and e.get("end_byte") == b]
        if expansions and {identifier for e in expansions for identifier in e.get("declaration_occurrence_ids", [])} != {o["id"] for o in generated}:
            errors.append("macro_expansion_occurrence_list_incomplete")
        if any(e.get("status") != "expanded_and_parsed" for e in expansions):
            errors.append("macro_expansion_not_complete")
        return self.row(c, "pending" if errors else "mapped_occurrences", "phase1_macro_generated_declaration",
                        "exact_invocation_and_definition_provenance_with_generated_kind_guard", generated, errors)

    def reconcile(self, c):
        a, b = c["byte_range"]
        if not 0 <= a <= b <= len(self.source) or sha(self.source[a:b]) != c["raw_sha256"]:
            raise ValueError("Candidate source range/hash differs from frozen snapshot: " + c["candidate_id"])
        kind = c["kind"]
        conditional = self.conditional_mapping(c)
        if conditional is not None:
            return conditional
        if kind in {"preprocessor_branch", "preprocessor_region"}:
            return self.row(c, "classified_non_api", "preprocessing_structure", "condition_or_region_container_not_a_declaration",
                            classification="preprocessor_container", condition_expression=c.get("predicate"),
                            constant_false=c.get("constant_false", False))
        if kind == "include_dependency":
            expression = c.get("operand", "")
            return self.row(c, "classified_non_api", "phase2_file_dependency", "include_connects_files_not_api_entities",
                            classification="include_dependency", include_classification=c["classification"],
                            header=c.get("header"), target_path=c.get("target_path"), operand=expression,
                            expression_macro="CUDA_STD_HEADER" if "CUDA_STD_HEADER" in expression else None,
                            relationship_status="pending_phase2")
        if kind == "preprocessor_directive" and c.get("directive") != "define":
            directive = c.get("directive")
            names = [t.text for t in self.source_tokens(a, b)]
            condition_calls = sum(name == "CUDA_ARCH_FAMILY" and following == "(" for name, following in zip(names, names[1:])) if directive in {"if", "elif"} else 0
            return self.row(c, "classified_non_api", "preprocessing_structure", "directive_not_standalone_cpp_api",
                            classification="preprocessor_" + str(directive),
                            condition_macro_uses={"CUDA_ARCH_FAMILY": condition_calls} if condition_calls else {})
        if kind == "macro_definition" or (kind == "preprocessor_directive" and c.get("directive") == "define"):
            matches = []
            expected_conditions = [self.branch_lines.get(ref) for ref in c.get("conditions", [])]
            for index in self.signatures.overlapping(a, b):
                o = self.occurrences[index]
                if o["kind"] == "macro_definition" and o["range"][0] >= a and o["range"][1] <= b and not self.condition_errors(o, c) and not self.invalid_occurrence(o):
                    matches.append(o)
            return self.row(c, "mapped_occurrences" if matches else "pending", "phase1_macro_definition",
                            "exact_macro_definition_occurrence_not_expansion_proof", matches,
                            [] if matches else ["macro_definition_occurrence_missing_or_invalid"])
        if kind == "macro_invocation":
            if c.get("origin") == "macro_body":
                return self.row(c, "classified_non_api", "phase1_macro_expansion_dependency", "replacement_list_token_not_a_separate_source_declaration",
                                classification="macro_replacement_list_reference", expansion_obligation="retained_at_source_invocations",
                                owner_candidate_id=c.get("owner_candidate_id"))
            alignment = self.alignment_mapping(c)
            if alignment is not None:
                return alignment
            expanded = self.macro_mapping(c)
            if expanded:
                return expanded
            definitions = [self.macro_definitions[ref] for ref in c.get("definition_candidates", []) if ref in self.macro_definitions]
            declaration_hints = {"operator", "namespace", "struct", "class", "enum", "using", "typedef"}
            if any(declaration_hints.intersection(d.get("declaration_generation_hints", [])) for d in definitions):
                return self.row(c, "pending", "phase1_macro_generated_declaration", "declaration_capable_macro_requires_generated_occurrences",
                                reasons=["declaration_macro_invocation_has_no_generated_entity"])
            operator_name = "operator()" if c["name"] == "operator" and token_spelling(self.source[a:b].decode()) == ("operator", "(", ")") else c["name"]
            matches, uncovered, errors = self.match_declarations(c, "callable", operator_name)
            if matches and not uncovered and not errors:
                return self.row(c, "mapped_occurrences", "phase1_declaration", "callable_spelling_is_its_own_declaration_not_a_call", matches)
            component = self.signature_token_mapping(c)
            if component is not None:
                return component
            expression = self.expression_mapping(c)
            if expression is not None:
                return expression
            for index in self.signatures.overlapping(a, b):
                o = self.occurrences[index]
                if c["name"] in o.get("attributes", []) and o["signature"][0] <= a <= b <= o["signature"][1] and not self.invalid_occurrence(o):
                    return self.row(c, "classified_non_api", "phase1_declaration_attribute", "exact_attribute_token_in_verified_declaration_signature", [o], classification="declaration_attribute_token")
            owner = self.local_body_owner(a, b)
            if owner and not self.blockers(a, b):
                return self.row(c, "classified_non_api", "phase2_expression", "lexical_invocation_inside_verified_callable_body_not_library_declaration", [owner],
                                classification="function_body_macro_use" if c.get("macro_spelling_known") else "function_body_callable_expression",
                                relationship_status="pending_phase2")
            return self.row(c, "pending", "phase1_declaration_or_macro", "unresolved_lexical_invocation_retained", reasons=errors or ["no_compatible_declaration_or_verified_non_api_context"])
        if kind != "syntax_interval":
            return self.row(c, "pending", "unclassified", "unknown_candidate_kind_never_dropped", reasons=["unknown_candidate_kind"])
        tokens = self.source_tokens(a, b)
        meaningful = [t for t in tokens if t.text not in SCAFFOLD]
        if not meaningful:
            return self.row(c, "classified_non_api", "syntax_structure", "delimiter_or_access_label_only", classification="syntax_scaffolding")
        expected = head_kind(tokens)
        matches, uncovered, errors = self.match_declarations(c, expected)
        blocking = self.blockers(a, b)
        # A namespace/type occurrence can account for its verified header only;
        # source tokens in its body are never accepted via its occurrence range.
        if matches and not uncovered and not errors and not blocking:
            return self.row(c, "mapped_occurrences", "phase1_declaration", "compatible_kind_and_all_significant_tokens_in_verified_signature_ranges", matches)
        component = self.signature_token_mapping(c)
        if component is not None:
            return component
        expression = self.expression_mapping(c)
        if expression is not None:
            return expression
        # Macro-generated source syntax also has an independent invocation row.
        invocations = [self.candidates[i] for i in self.invocations.overlapping(a, b)
                       if a <= self.candidates[i]["byte_range"][0] and self.candidates[i]["byte_range"][1] <= b]
        if invocations:
            generated = []
            covered = []
            for invocation in invocations:
                mapped = self.macro_mapping(invocation)
                if mapped and mapped["status"] == "mapped_occurrences":
                    generated.extend(self.macros[tuple(invocation["byte_range"])])
                    covered.append(invocation["byte_range"])
            if generated and all(any(x <= t.start and t.end <= y for x, y in covered) for t in meaningful):
                return self.row(c, "mapped_occurrences", "phase1_macro_generated_declaration", "syntax_exactly_accounts_for_verified_macro_invocations", generated)
        owner = self.local_body_owner(a, b)
        has_declaration_macro = self.unexpanded_declaration_macro(a, b)
        if owner and expected not in {"type", "namespace"} and not blocking and not has_declaration_macro:
            return self.row(c, "classified_non_api", "phase2_expression", "statement_inside_independently_brace_verified_callable_body_and_outside_local_type",
                            [owner], classification="function_local_alias" if expected == "alias" else "function_body_statement_or_local_data",
                            relationship_status="pending_phase2")
        if meaningful[0].text == "static_assert" and not blocking and not has_declaration_macro:
            return self.row(c, "classified_non_api", "phase1_constraint_expression", "static_assert_is_constraint_not_named_library_api",
                            classification="compile_time_assertion", relationship_status="pending_phase2")
        if uncovered:
            errors.append("significant_source_tokens_not_covered_by_compatible_declaration_signatures")
        if blocking:
            errors.append("overlapping_blocking_extraction_diagnostic")
        if has_declaration_macro:
            errors.append("source_interval_contains_unexpanded_declaration_macro")
        return self.row(c, "pending", "phase1_declaration_or_non_api_review", "source_obligation_not_silently_discarded", matches, errors or ["unclassified_source_interval"],
                        diagnostic_ids=[d["diagnostic_id"] for d in blocking],
                        uncovered_token_examples=[t.text for t in uncovered[:12]])

    def run(self):
        rows = [self.reconcile(c) for c in self.candidates]
        if len(rows) != len(self.candidates) or len({r["candidate_id"] for r in rows}) != len(rows):
            raise ValueError("Candidate cardinality/identity changed during reconciliation")
        return rows


def candidate_metadata(path):
    metadata, files, macros = {}, {}, {}
    for key, value, item in top_items(path, {"files", "candidates"}):
        if not item:
            metadata[key] = value
        elif key == "files":
            if value["path"] in files:
                raise ValueError("Duplicate candidate file")
            files[value["path"]] = value
        elif value["kind"] == "macro_definition":
            macros[value["candidate_id"]] = value
    return metadata, files, macros


def verify_frozen_candidates(path, source, candidates, macro_definitions, scope_paths):
    """Reproduce the frozen scanner per file, without declaration-extractor data.

    Counts in a damaged candidate ledger cannot redefine the denominator: even
    deleting a candidate and adjusting its summary is rejected against source.
    The global macro index is retained, so object-like uses defined in another
    header do not disappear during per-file reproduction.
    """
    macro_index = defaultdict(list)
    for definition in macro_definitions.values():
        macro_index[definition["name"]].append(definition["candidate_id"])
    scan = LEXER.SourceScan(path, source)
    scan.scan_preprocessor()
    scan.scan_includes(scope_paths)
    scan.scan_syntax(macro_index)
    for candidate in scan.candidates:
        candidate.pop("_tokens", None)
    expected = {c["candidate_id"]: c for c in scan.candidates}
    observed = {c["candidate_id"]: c for c in candidates}
    if len(observed) != len(candidates) or observed != expected:
        missing = sorted(set(expected) - set(observed))[:4]
        extra = sorted(set(observed) - set(expected))[:4]
        raise ValueError(f"Frozen candidate denominator/content differs from source rescan: {path}; missing={missing}; extra={extra}")


def stage_declarations(path, db):
    db.executescript("CREATE TABLE occurrences(path TEXT, id TEXT PRIMARY KEY, value TEXT); CREATE TABLE diagnostics(path TEXT, value TEXT); CREATE TABLE files(path TEXT PRIMARY KEY, value TEXT);")
    metadata = {}
    for key, value, item in top_items(path, {"occurrences", "diagnostics", "files"}):
        if not item:
            metadata[key] = value
        elif key == "occurrences":
            slim = slim_occurrence(value)
            db.execute("INSERT INTO occurrences VALUES(?,?,?)", (value["path"], slim["id"], json.dumps(slim, separators=(",", ":"))))
        elif key == "diagnostics":
            slim = {k: value[k] for k in ("diagnostic_id", "category", "start_byte", "end_byte", "blocks_phase_1")}
            db.execute("INSERT INTO diagnostics VALUES(?,?)", (value["path"], json.dumps(slim)))
        else:
            slim = {"path": value["path"], "sha256": value["sha256"],
                    "macro_expansions": [{k: e.get(k) for k in ("start_byte", "end_byte", "declaration_occurrence_ids", "status")} for e in value.get("macro_expansions", [])]}
            alignment = value.get('syntax_analysis', {}).get('header_analysis', {}).get('alignment_attributes', [])
            if alignment:
                slim['syntax_analysis'] = {'header_analysis': {'alignment_attributes': alignment}}
            db.execute("INSERT INTO files VALUES(?,?)", (value["path"], json.dumps(slim)))
    db.executescript("CREATE INDEX occurrences_path ON occurrences(path); CREATE INDEX diagnostics_path ON diagnostics(path);")
    db.commit()
    return metadata


def build(root=ROOT):
    candidate_path, declaration_path = root / "data/candidates.json", root / "data/declarations.json"
    inputs = {"candidates_sha256": file_sha(candidate_path), "declarations_sha256": file_sha(declaration_path),
              "reconciler_sha256": file_sha(Path(__file__)), "lexer_sha256": file_sha(Path(LEXER.__file__))}
    metadata, file_metadata, macros = candidate_metadata(candidate_path)
    scope = json.loads((root / "data/scope.json").read_bytes())
    if metadata["commit"] != scope["commit"] or set(file_metadata) != {f["path"] for f in scope["files"]}:
        raise ValueError("Candidate scope/commit differs from fixed manifest")
    if metadata.get("scanner_sha256") != inputs["lexer_sha256"] or metadata.get("source_scope_fingerprint") != file_sha(root / "data/scope.json"):
        raise ValueError("Frozen scanner/source-scope provenance differs from current inputs")
    target = root / "data/candidate-mapping.json"
    check_target = root / "data/candidate-mapping-checks.json"
    counters, obligations, rules, per_file = Counter(), Counter(), Counter(), []
    pending_obligations, pending_reasons = Counter(), Counter()
    seen_ids, processed_files = set(), set()
    special = Counter()
    with tempfile.TemporaryDirectory(prefix=".reconcile-", dir=root / "data") as work:
        db = sqlite3.connect(str(Path(work) / "declarations.sqlite"))
        declaration_meta = stage_declarations(declaration_path, db)
        declared_files = {x[0] for x in db.execute("SELECT path FROM files")}
        if declaration_meta.get("commit") != scope["commit"] or declared_files != set(file_metadata) or declaration_meta.get("is_subset"):
            raise ValueError("Declaration ledger does not contain the identical full fixed scope")
        output = Path(work) / "mapping.json"
        with output.open("w", encoding="utf-8") as stream:
            header = {"schema_version": 1, "commit": scope["commit"], "inputs": inputs,
                      "contract": "Every frozen lexical candidate is classified; lexical count is NOT an API denominator. Mapping is evidence, not semantic-completeness certification."}
            stream.write(json.dumps(header, ensure_ascii=False, separators=(",", ":"))[:-1] + ',"mappings":[')
            first = True

            def process(path, candidates):
                nonlocal first
                if path in processed_files:
                    raise ValueError("Candidates are not grouped uniquely by file")
                processed_files.add(path)
                source = (root / "snapshot" / path).read_bytes()
                if sha(source) != file_metadata[path]["source_sha256"]:
                    raise ValueError("Snapshot differs from candidate source: " + path)
                verify_frozen_candidates(path, source, candidates, macros, set(file_metadata))
                row = db.execute("SELECT value FROM files WHERE path=?", (path,)).fetchone()
                file_record = json.loads(row[0])
                if file_record["sha256"] != sha(source):
                    raise ValueError("Declaration source hash differs from snapshot: " + path)
                occurrences = [json.loads(r[0]) for r in db.execute("SELECT value FROM occurrences WHERE path=?", (path,))]
                diagnostics = [json.loads(r[0]) for r in db.execute("SELECT value FROM diagnostics WHERE path=?", (path,))]
                def snapshot_source(other_path):
                    if other_path not in file_metadata:
                        raise ValueError("Namespace proof points outside fixed source scope")
                    data = (root / "snapshot" / other_path).read_bytes()
                    if sha(data) != file_metadata[other_path]["source_sha256"]:
                        raise ValueError("Namespace definition source differs from fixed scope")
                    return data
                reconciler = Reconciler(path, source, candidates, occurrences, diagnostics, macros, file_record, snapshot_source)
                file_counts = Counter()
                for mapped in reconciler.run():
                    cid = mapped["candidate_id"]
                    if cid in seen_ids:
                        raise ValueError("Duplicate frozen candidate ID")
                    seen_ids.add(cid)
                    if not first:
                        stream.write(",")
                    first = False
                    stream.write(json.dumps(mapped, ensure_ascii=False, separators=(",", ":")))
                    counters[mapped["status"]] += 1
                    obligations[mapped["obligation"]] += 1
                    rules[mapped["evidence"]["rule"]] += 1
                    file_counts[mapped["status"]] += 1
                    if mapped["status"] == "pending":
                        pending_obligations[mapped["obligation"]] += 1
                        pending_reasons.update(mapped["pending_reasons"])
                    if mapped.get("expression_macro") == "CUDA_STD_HEADER":
                        special["CUDA_STD_HEADER_include_operands"] += 1
                    special["CUDA_ARCH_FAMILY_condition_uses"] += mapped.get("condition_macro_uses", {}).get("CUDA_ARCH_FAMILY", 0)
                if len(candidates) != sum(file_metadata[path]["candidate_counts"].values()):
                    raise ValueError("Per-file candidate denominator changed")
                per_file.append({"path": path, "candidate_count": len(candidates), "statuses": dict(file_counts),
                                 "occurrence_count": len(occurrences), "lexical_diagnostics": reconciler.lexical_issues})
                if len(per_file) % 100 == 0:
                    print(f"reconciled {len(per_file)}/{len(file_metadata)} files, candidates={len(seen_ids)}, pending={counters['pending']}", flush=True)

            current_path, batch = None, []
            for key, value, item in top_items(candidate_path, {"candidates"}):
                if not item or key != "candidates":
                    continue
                path = value["path"]
                if current_path is not None and path != current_path:
                    process(current_path, batch)
                    batch = []
                current_path = path
                batch.append(value)
            if current_path is not None:
                process(current_path, batch)
            # Empty source files still belong to the denominator.
            for path in sorted(set(file_metadata) - processed_files):
                process(path, [])
            expected = metadata["summary"]["candidate_count"]
            if len(seen_ids) != expected or processed_files != set(file_metadata):
                raise ValueError("Candidate/file denominator changed")
            checks = {"schema_version": 1, "inputs": inputs, "file_count": len(per_file), "candidate_count": len(seen_ids),
                      "frozen_candidate_count": expected, "candidate_identity_and_cardinality_preserved": True,
                      "statuses": dict(counters), "obligations": dict(obligations), "rules": dict(rules),
                      "pending_by_obligation": dict(pending_obligations), "pending_by_reason": dict(pending_reasons),
                      "all_candidates_reproduced_from_source_and_frozen_scanner": True,
                      "special_macro_classifications": dict(special),
                      "phase_1_passed": False,
                      "reason": "Lexical over-inclusion is not an API denominator; pending rows and semantic coverage require independent review. Phase-2 obligations are not declaration gaps."}
            stream.write('],"files":' + json.dumps(per_file, ensure_ascii=False, separators=(",", ":")))
            stream.write(',"checks":' + json.dumps(checks, ensure_ascii=False, separators=(",", ":")) + "}\n")
        db.close()
        if inputs["candidates_sha256"] != file_sha(candidate_path) or inputs["declarations_sha256"] != file_sha(declaration_path):
            raise ValueError("An input changed during reconciliation; output not published")
        if inputs["reconciler_sha256"] != file_sha(Path(__file__)) or inputs["lexer_sha256"] != file_sha(Path(LEXER.__file__)):
            raise ValueError("Reconciler/lexer changed during run; output not published")
        output.replace(target)
        checks["mapping_sha256"] = file_sha(target)
        temporary_checks = Path(work) / "checks.json"
        temporary_checks.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
        temporary_checks.replace(check_target)
    print(json.dumps({k: checks[k] for k in ("file_count", "candidate_count", "statuses", "special_macro_classifications", "phase_1_passed", "mapping_sha256")}, ensure_ascii=False))
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    build(parser.parse_args().root)

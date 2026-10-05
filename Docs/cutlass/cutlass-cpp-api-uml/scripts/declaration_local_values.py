#!/usr/bin/env python3
"""Prove selected ambiguous T name(value) declarations are local objects.

This is a deliberately bounded name-lookup proof, not a C++ semantic analyzer.
It independently builds lexical scopes from the original source (only directive
and known annotation spelling is whitespace-masked, with physical bytes kept).
An initializer identifier must bind to a function parameter, a direct member,
or a preceding block object. A visible type alias wins over an outer value.
Unknown names, conditional shadow ambiguity, extern, lambda and loop scopes stay
pending. Types/imports/dependent qualified names are never invented.

Core integration can cache LocalValueAnalyzer once per physical source file and
call classify_range(start_byte, end_byte). No extractor/reconciler is imported,
no source or inventory is written, and constructor/call/read/write resolution
remains a separate phase-2 obligation.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

from tree_sitter import Language, Parser
import tree_sitter_cpp


_spec = importlib.util.spec_from_file_location("local_value_source_lexer", Path(__file__).with_name("scan_candidates.py"))
_lex = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _lex
_spec.loader.exec_module(_lex)
LANGUAGE = Language(tree_sitter_cpp.language())
TYPE_NODES = {"class_specifier", "struct_specifier", "union_specifier", "enum_specifier"}
NAME_NODES = {"identifier", "field_identifier", "type_identifier"}
ANNOTATIONS = {"CUTLASS_HOST_DEVICE", "CUTLASS_DEVICE", "CUTLASS_HOST", "CUTLASS_GLOBAL", "CUTE_HOST_DEVICE", "CUTE_DEVICE", "CUTE_HOST",
               "CUTLASS_PRAGMA_UNROLL", "CUTLASS_PRAGMA_NO_UNROLL", "CUTE_UNROLL", "CUTE_NO_UNROLL", "__host__", "__device__", "__forceinline__"}
BUILTINS = {"void", "bool", "char", "signed", "unsigned", "short", "int", "long", "float", "double", "wchar_t", "char16_t", "char32_t"}
QUALIFIERS = {"const", "volatile", "constexpr", "static", "register", "thread_local", "*", "&", "&&"}
FORBIDDEN_OWNERS = {"if", "else", "for", "while", "constexpr", "sizeof", "decltype", "return"} | BUILTINS


def identity(prefix, *parts):
    data = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(data.encode()).hexdigest()[:24]


def children_field(node, name):
    return [c for i, c in enumerate(node.children) if node.field_name_for_child(i) == name]


def declared_name(node):
    if node is None:
        return None
    if node.type in NAME_NODES:
        return node
    if node.type in {"template_type", "qualified_identifier"}:
        return node.child_by_field_name("name")
    child = node.child_by_field_name("declarator")
    if child:
        return declared_name(child)
    for child in node.named_children:
        if child.type.endswith("declarator") or child.type in NAME_NODES:
            found = declared_name(child)
            if found:
                return found
    return None


def callable_declarator(node):
    if node is None:
        return None
    if node.type == "function_declarator":
        inside = node.child_by_field_name("declarator")
        return None if inside and inside.type == "parenthesized_declarator" else node
    child = node.child_by_field_name("declarator")
    if child:
        return callable_declarator(child)
    return next((found for c in node.named_children if c.type.endswith("declarator") for found in [callable_declarator(c)] if found), None)


@dataclass
class Scope:
    kind: str
    node: object
    parent: Scope | None = None
    name: str | None = None
    bindings: list[dict] = field(default_factory=list)


class LocalValueAnalyzer:
    def __init__(self, path: str, source: bytes, root_node=None):
        self.path, self.source = path, source
        self.lex = _lex.SourceScan(path, source)
        self.lex.scan_preprocessor()
        self.tokens = self.lex.tokens
        self.starts = [t.start for t in self.tokens]
        self.line_starts = [0] + [m.end() for m in re.finditer(b"\n", source)]
        self.branches = {c["candidate_id"]: c for c in self.lex.candidates if c["kind"] == "preprocessor_branch"}
        self.directives = {d["candidate_id"]: d for d in self.lex.directives}
        self.groups = self._branch_groups()
        self.normalizations = []
        normalized = bytearray(source)
        for d in self.lex.directives:
            self._mask(normalized, *d["byte_range"], "preprocessor_directive_retained_in_condition_ledger")
        for token in self.tokens:
            if token.kind == "identifier" and token.text in ANNOTATIONS:
                self._mask(normalized, token.start, token.end, "known_scope_neutral_annotation")
        if root_node is None:
            self.tree = Parser(LANGUAGE).parse(bytes(normalized))
            root_node = self.tree.root_node
        elif root_node.start_byte != 0 or root_node.end_byte > len(source):
            raise ValueError("Provided AST does not address the physical source")
        self.root_node = root_node
        self.brace_ends = self._brace_pairs()
        self.global_scope = Scope("translation_unit", root_node)
        self.candidates = []
        self._walk(root_node, self.global_scope)
        self.results = {}

    def text(self, node):
        return self.source[node.start_byte:node.end_byte].decode("utf-8", "replace") if node else ""

    def span(self, start, end):
        return {"path": self.path, "start_byte": start, "end_byte": end,
                "start_line": bisect_right(self.line_starts, start), "end_line": bisect_right(self.line_starts, max(start, end - 1))}

    def _mask(self, output, start, end, reason):
        output[start:end] = bytes(c if c in (10, 13) else 32 for c in output[start:end])
        self.normalizations.append({**self.span(start, end), "reason": reason})

    def _branch_groups(self):
        result, stack = {}, []
        for directive in self.lex.directives:
            command = directive["directive"]
            if command in {"if", "ifdef", "ifndef"}:
                stack.append(directive["candidate_id"])
            if command in {"if", "ifdef", "ifndef", "elif", "else"} and stack:
                for branch in self.branches.values():
                    if branch["directive_id"] == directive["candidate_id"]:
                        result[branch["candidate_id"]] = stack[-1]
            if command == "endif" and stack:
                stack.pop()
        return result

    def conditions(self, byte):
        carrier = next((d for d in self.lex.directives if d["byte_range"][0] <= byte < d["byte_range"][1]), None)
        if carrier is None:
            carrier = next((r for r in self.lex.regions if r["byte_range"][0] <= byte < r["byte_range"][1]), {})
        return list(carrier.get("conditions", []))

    def condition_records(self, conditions):
        return [{"branch_id": ref, "group_id": self.groups.get(ref), "expression": self.branches[ref]["predicate"],
                 "constant_false": self.branches[ref]["constant_false"],
                 "directive": self.span(*self.directives[self.branches[ref]["directive_id"]]["byte_range"])} for ref in conditions]

    def disjoint(self, left, right):
        return any(self.groups.get(a) == self.groups.get(b) and a != b for a in left for b in right)

    def _brace_pairs(self):
        excluded = [d["byte_range"] for d in self.lex.directives]
        stack, result = [], {}
        for token in self.tokens:
            if any(a <= token.start < b for a, b in excluded):
                continue
            if token.text == "{": stack.append(token.start)
            elif token.text == "}" and stack: result[stack.pop()] = token.end
        return result

    def _add(self, scope, name_node, category, kind, declaration, *, visible_after=None, declared_type=None):
        if name_node is None:
            return
        name = self.text(name_node)
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            return
        scope.bindings.append({"binding_id": identity("value-binding:", self.path, name_node.start_byte, category, kind),
                               "name": name, "category": category, "kind": kind,
                               "declaration": self.span(declaration.start_byte, declaration.end_byte),
                               "name_range": self.span(name_node.start_byte, name_node.end_byte),
                               "scope": {"kind": scope.kind, **self.span(scope.node.start_byte, scope.node.end_byte)},
                               "visible_after": declaration.end_byte if visible_after is None else visible_after,
                               "conditions": self.conditions(declaration.start_byte), "declared_type": declared_type,
                               "parse_error": bool(declaration.has_error)})

    def _shape(self, node):
        declarators = children_field(node, "declarator")
        if len(declarators) != 1:
            return None
        decl = declarators[0]
        name_node = declared_name(decl)
        if name_node is None:
            return None
        tokens = self.tokens[bisect_left(self.starts, name_node.end_byte):bisect_left(self.starts, node.end_byte)]
        if not tokens or tokens[0].text != "(" or tokens[-1].text != ";":
            return None
        depth, argument_start, args, closing = 0, tokens[0].end, [], None
        for token in tokens:
            if token.text == "(": depth += 1
            elif token.text == ")":
                depth -= 1
                if depth == 0:
                    if self.source[argument_start:token.start].strip() or args:
                        args.append((argument_start, token.start))
                    closing = token
                    break
            elif token.text == "," and depth == 1:
                args.append((argument_start, token.start)); argument_start = token.end
        if closing is None or self.source[closing.end:node.end_byte].strip() != b";":
            return None
        return {"node": node, "declarator": decl, "name_node": name_node,
                "type_range": (node.start_byte, name_node.start_byte),
                "argument_ranges": args, "initializer_range": (tokens[0].start, closing.end)}

    @staticmethod
    def chain(scope):
        while scope:
            yield scope
            scope = scope.parent

    def _walk_initializer_scopes(self, node, scope):
        if node.type == "lambda_expression":
            self._walk(node, scope)
            return
        for child in node.named_children:
            self._walk_initializer_scopes(child, scope)

    def _walk(self, node, scope):
        kind = node.type
        if kind in {"comment", "preproc_def", "preproc_function_def", "parameter_list", "template_parameter_list"}:
            return
        if kind == "template_declaration":
            child_scope = Scope("template", node, scope)
            parameters = node.child_by_field_name("parameters") or next((c for c in node.named_children if c.type == "template_parameter_list"), None)
            for parameter in parameters.named_children if parameters else []:
                name = parameter.child_by_field_name("name") or declared_name(parameter.child_by_field_name("declarator"))
                if name is None and parameter.type == "type_parameter_declaration":
                    name = next((c for c in parameter.named_children if c.type == "type_identifier"), None)
                category = "value" if parameter.type in {"parameter_declaration", "optional_parameter_declaration"} else "type"
                self._add(child_scope, name, category, "template_value_parameter" if category == "value" else "template_type_parameter", parameter, visible_after=node.start_byte)
            for child in node.named_children:
                if child != parameters: self._walk(child, child_scope)
            return
        if kind == "namespace_definition":
            body = node.child_by_field_name("body")
            child_scope = Scope("namespace", node, scope, self.text(node.child_by_field_name("name")))
            if body:
                for child in body.named_children: self._walk(child, child_scope)
            return
        if kind in TYPE_NODES:
            named = node.child_by_field_name("name")
            if named and named.type == "template_type": named = named.child_by_field_name("name")
            self._add(scope, named, "type", "class_or_enum_type", node, visible_after=node.start_byte)
            body = node.child_by_field_name("body")
            child_scope = Scope("class", node, scope, self.text(named))
            if body:
                for child in body.named_children: self._walk(child, child_scope)
            return
        if kind == "function_definition":
            function = callable_declarator(node.child_by_field_name("declarator"))
            name = declared_name(node.child_by_field_name("declarator"))
            self._add(scope, name, "callable", "function", node)
            child_scope = Scope("function", node, scope, self.text(name) or self.text(node.child_by_field_name("declarator")))
            parameters = function.child_by_field_name("parameters") if function else None
            for parameter in parameters.named_children if parameters else []:
                named = declared_name(parameter.child_by_field_name("declarator"))
                self._add(child_scope, named, "value", "function_parameter", parameter, visible_after=node.start_byte,
                          declared_type=self.text(parameter.child_by_field_name("type")))
            body = node.child_by_field_name("body")
            if body: self._walk(body, child_scope)
            return
        if kind in {"lambda_expression", "for_statement", "for_range_loop"}:
            child_scope = Scope("lambda" if kind == "lambda_expression" else "for", node, scope)
            for child in node.named_children: self._walk(child, child_scope)
            return
        if kind in {"if_statement", "switch_statement", "while_statement", "catch_clause"}:
            child_scope = Scope("catch" if kind == "catch_clause" else "control", node, scope)
            for child in node.named_children: self._walk(child, child_scope)
            return
        if kind == "compound_statement":
            child_scope = Scope("block", node, scope)
            for child in node.named_children: self._walk(child, child_scope)
            return
        if kind in {"alias_declaration", "type_definition"}:
            names = [node.child_by_field_name("name")] if kind == "alias_declaration" else [declared_name(d) for d in children_field(node, "declarator")]
            for name in names: self._add(scope, name, "type", "using_alias" if kind == "alias_declaration" else "typedef", node)
            return
        if kind in {"declaration", "field_declaration"}:
            type_node = node.child_by_field_name("type")
            if type_node and type_node.type in TYPE_NODES:
                self._walk(type_node, scope)
            shape = self._shape(node)
            in_function = any(s.kind == "function" for s in self.chain(scope))
            if shape and in_function:
                self.candidates.append({**shape, "scope": scope})
                self._walk_initializer_scopes(node, scope)
                return
            for declarator in children_field(node, "declarator"):
                name = declared_name(declarator)
                callable_node = callable_declarator(declarator)
                if callable_node:
                    self._add(scope, name, "callable", "function_declaration", node)
                elif scope.kind != "for":
                    self._add(scope, name, "value", "member_object" if scope.kind == "class" else "block_object" if in_function else "namespace_object",
                              node, declared_type=self.text(type_node))
            self._walk_initializer_scopes(node, scope)
            return
        for child in node.named_children:
            self._walk(child, scope)

    def _lookup(self, name, scope, position, candidate_conditions):
        crossed_class = False
        for current in self.chain(scope):
            options = [b for b in current.bindings if b["name"] == name
                       and (current.kind == "class" or b["visible_after"] <= position)
                       and not self.disjoint(b["conditions"], candidate_conditions)]
            if options:
                if any(not set(b["conditions"]).issubset(candidate_conditions) for b in options):
                    return None, "conditional_binding_or_shadow_not_proven_visible"
                categories = {b["category"] for b in options}
                if len(categories) != 1:
                    return None, "type_value_or_callable_shadow_ambiguity"
                chosen = max(options, key=lambda b: b["declaration"]["start_byte"])
                if chosen["parse_error"]:
                    return None, "binding_declaration_has_parse_error"
                if crossed_class and chosen["kind"] == "member_object":
                    return None, "outer_class_object_access_not_proven"
                if crossed_class and chosen["kind"] in {"function_parameter", "block_object", "proven_local_object"}:
                    return None, "enclosing_function_automatic_value_not_accessible_to_local_class"
                return chosen, None
            crossed_class |= current.kind == "class"
        return None, "identifier_has_no_visible_source_binding"

    def _macro_affects(self, name, position, conditions):
        for d in self.lex.definitions:
            if d["name"] != name or d["byte_range"][0] >= position or self.disjoint(d["conditions"], conditions):
                continue
            # A compatible definition needs actual preprocessing/lifetime proof;
            # this module never assumes a value binding overrides a macro.
            if any(self.branches[r]["constant_false"] and r not in conditions for r in d["conditions"]):
                continue
            return True
        return False

    def _type_binding(self, candidate, conditions):
        a, b = candidate["type_range"]
        words = [t.text for t in self.tokens[bisect_left(self.starts, a):bisect_left(self.starts, b)]]
        if "extern" in words:
            return None, "local_extern_declaration_not_reclassified"
        if any(re.fullmatch(r"[A-Za-z_]\w*", word) and self._macro_affects(word, a, conditions) for word in words):
            return None, "macro_can_change_target_type"
        if any(w in {"typename", "typedef", "using", "auto", "::", ":", "<", ">"} for w in words):
            return None, "dependent_qualified_or_unproven_target_type"
        names = [w for w in words if w not in QUALIFIERS]
        if names and all(w in BUILTINS for w in names):
            if names == ["void"] and "*" not in words:
                return None, "void_is_not_a_local_object_type"
            if any(self._macro_affects(w, a, conditions) for w in names):
                return None, "macro_can_change_target_type"
            return {"category": "type", "kind": "builtin_type", "spelling": self.source[a:b].decode(), "range": self.span(a, b)}, None
        if len(names) != 1 or not re.fullmatch(r"[A-Za-z_]\w*", names[0]):
            return None, "unsupported_target_type_spelling"
        if self._macro_affects(names[0], a, conditions):
            return None, "macro_can_change_target_type"
        binding, error = self._lookup(names[0], candidate["scope"], a, conditions)
        if error: return None, "target_type_" + error
        if binding["category"] != "type": return None, "target_name_is_not_a_proven_type"
        return binding, None

    def _argument_binding(self, bounds, candidate, conditions):
        a, b = bounds
        values = self.tokens[bisect_left(self.starts, a):bisect_left(self.starts, b)]
        result = {"expression": self.source[a:b].decode().strip(), "range": self.span(a, b)}
        if any(t.kind == "identifier" and self._macro_affects(t.text, candidate["node"].start_byte, conditions) for t in values):
            return result, "macro_can_change_initializer_identifier"
        if len(values) == 1 and (values[0].kind in {"number", "literal"} or values[0].text in {"true", "false", "nullptr"}):
            result["binding"] = {"category": "value", "kind": "literal", "range": self.span(values[0].start, values[0].end)}
            return result, None
        if len(values) != 1 or values[0].kind != "identifier":
            return result, "initializer_argument_expression_form_not_supported"
        name = values[0].text
        if name == self.text(candidate["name_node"]):
            return result, "self_initialization_name_lookup_not_assumed"
        if self._macro_affects(name, candidate["node"].start_byte, conditions):
            return result, "macro_can_change_initializer_identifier"
        binding, error = self._lookup(name, candidate["scope"], candidate["node"].start_byte, conditions)
        if error: return result, error
        result["binding"] = binding
        if binding["category"] != "value": return result, "initializer_identifier_is_type_or_callable_not_object"
        if binding["kind"] not in {"function_parameter", "member_object", "block_object", "proven_local_object", "template_value_parameter"}:
            return result, "value_binding_category_outside_supported_local_contract"
        return result, None

    def _classify(self, candidate):
        node, scope = candidate["node"], candidate["scope"]
        conditions = self.conditions(node.start_byte)
        scopes = list(self.chain(scope))
        owner = next((s for s in scopes if s.kind == "function"), None)
        reasons = []
        if owner is None:
            reasons.append("no_function_owner")
        else:
            body = owner.node.child_by_field_name("body")
            if not body or self.brace_ends.get(body.start_byte) != body.end_byte:
                reasons.append("function_body_scope_not_lexically_verified")
            if owner.name in FORBIDDEN_OWNERS:
                reasons.append("recovered_keyword_is_not_a_function_owner")
        if any(s.kind == "lambda" for s in scopes): reasons.append("lambda_capture_and_parameter_scope_pending")
        if any(s.kind == "for" for s in scopes): reasons.append("for_initializer_and_iteration_scope_pending")
        if any(s.kind == "catch" for s in scopes): reasons.append("catch_parameter_scope_pending")
        if node.has_error: reasons.append("candidate_declaration_has_parse_error")
        if not candidate["argument_ranges"]: reasons.append("empty_parentheses_are_not_value_initialization_proof")
        target, error = self._type_binding(candidate, conditions)
        if error: reasons.append(error)
        arguments = []
        for bounds in candidate["argument_ranges"]:
            argument, error = self._argument_binding(bounds, candidate, conditions)
            arguments.append(argument)
            if error: reasons.append(error)
        boundary_pending = any(s.kind in {"lambda", "for", "catch"} for s in scopes)
        candidate_type = None
        if boundary_pending:
            candidate_type, target = target, None
            for argument in arguments:
                if "binding" in argument:
                    argument["source_binding_candidate"] = argument.pop("binding")
                    argument["binding_status"] = "not_resolved_across_pending_scope_boundary"
        name = self.text(candidate["name_node"])
        identifier = identity("local-initialization:", self.path, node.start_byte, node.end_byte)
        result = {"candidate_id": identifier, "status": "pending" if reasons else "classified_non_api",
                  "classification": "unresolved_local_declaration" if reasons else "proven_local_object_initialization",
                  **self.span(node.start_byte, node.end_byte), "raw": self.text(node), "name": name,
                  "name_range": self.span(candidate["name_node"].start_byte, candidate["name_node"].end_byte),
                  "initializer_range": self.span(*candidate["initializer_range"]),
                  "target_type_binding": target, "argument_bindings": arguments,
                  "source_type_candidate": candidate_type,
                  "conditions": self.condition_records(conditions), "pending_reasons": sorted(set(reasons)),
                  "function_owner": {"name": owner.name, **self.span(owner.node.start_byte, owner.node.end_byte)} if owner else None,
                  "scope_chain": [{"kind": s.kind, "name": s.name, **self.span(s.node.start_byte, s.node.end_byte)} for s in reversed(scopes)],
                  "phase2_obligations": ["initialization/conversion call target", "value-use reads", "local object writes"],
                  "semantic_claim": "value-versus-type binding only; no construction, lifetime, or runtime-correctness proof"}
        if not reasons:
            result["local_object_id"] = identity("local-object:", self.path, candidate["name_node"].start_byte)
            self._add(scope, candidate["name_node"], "value", "proven_local_object", node,
                      declared_type=self.source[slice(*candidate["type_range"])].decode())
        return result

    def analyze(self):
        for candidate in sorted(self.candidates, key=lambda c: c["node"].start_byte):
            key = (candidate["node"].start_byte, candidate["node"].end_byte)
            if key not in self.results:
                self.results[key] = self._classify(candidate)
        return {"schema_version": 1, "path": self.path, "source_sha256": hashlib.sha256(self.source).hexdigest(),
                "normalizations": self.normalizations, "candidates": list(self.results.values()),
                "summary": {"candidate_count": len(self.results), "statuses": dict(Counter(c["status"] for c in self.results.values()))},
                "coverage_claim": "Only the listed direct-initialization candidates; no complete local-variable or library-API claim"}

    def classify_range(self, start_byte, end_byte):
        self.analyze()
        return self.results.get((start_byte, end_byte), {"status": "pending", **self.span(start_byte, end_byte),
                                                       "pending_reasons": ["range_not_a_supported_local_declaration"]})


def analyze_local_values(path, source, candidate_ranges=None, root_node=None):
    analyzer = LocalValueAnalyzer(path, source, root_node)
    result = analyzer.analyze()
    if candidate_ranges is not None:
        result["requested_classifications"] = [analyzer.classify_range(a, b) for a, b in candidate_ranges]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--path", help="Repository-relative provenance path")
    args = parser.parse_args()
    print(json.dumps(analyze_local_values(args.path or str(args.source), args.source.read_bytes()), ensure_ascii=False, indent=2))

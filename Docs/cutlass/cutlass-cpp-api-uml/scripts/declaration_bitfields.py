"""Byte-exact, standalone bitfield inventory and anonymous-name parser projection.

No preprocessing branch is selected and no source is rewritten. A lexical colon
ledger is built independently of the extraction generator, then reconciled with
the raw C++ tree. Unclassified colons remain explicit obligations. In particular,
Tree-sitter's recovery can call constructor initializers ``bitfield_clause``;
that node name alone is never sufficient evidence for a bitfield.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from tree_sitter import Language, Parser
import tree_sitter_cpp

LANGUAGE = Language(tree_sitter_cpp.language())
SCHEMA_VERSION = 1
_TOKEN = re.compile(rb"[A-Za-z_$\x80-\xff][A-Za-z_0-9$\x80-\xff]*|[0-9][A-Za-z_0-9.']*|::|->\*|->|\.\.\.|&&|\|\||<<|>>|<=|>=|==|!=|[^\s]")
_RAW = re.compile(rb'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(')
_QUOTED = re.compile(rb'(?:u8|u|U|L)?["\']')
_DIRECTIVE = re.compile(rb"^[ \t]*#[ \t]*([A-Za-z_]+)(?:[^\n]*\\\r?\n)*[^\n]*", re.M)
_NON_BITFIELD = {
    "conditional_expression": "conditional_expression_separator",
    "base_class_clause": "base_class_separator",
    "field_initializer_list": "constructor_initializer_separator",
    "labeled_statement": "label_separator",
    "case_statement": "case_separator",
    "for_range_loop": "range_for_separator",
    "gnu_asm_expression": "gnu_asm_separator",
}


def _identity(prefix, *parts):
    spelling = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(spelling.encode()).hexdigest()[:24]


def _walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _tokens(data):
    """Physical tokens, with comments/literals removed but byte coordinates intact."""
    result, cursor = [], 0
    while cursor < len(data):
        if data[cursor] in b" \t\r\n\v\f":
            cursor += 1
            continue
        if data.startswith(b"//", cursor):
            end = data.find(b"\n", cursor + 2)
            while end >= 0 and data[cursor:end].rstrip(b"\r").endswith(b"\\"):
                end = data.find(b"\n", end + 1)
            cursor = len(data) if end < 0 else end
            continue
        if data.startswith(b"/*", cursor):
            end = data.find(b"*/", cursor + 2)
            cursor = len(data) if end < 0 else end + 2
            continue
        raw = _RAW.match(data, cursor)
        if raw:
            closing = b")" + raw[1] + b'"'
            end = data.find(closing, raw.end())
            cursor = len(data) if end < 0 else end + len(closing)
            continue
        quote = _QUOTED.match(data, cursor)
        if quote:
            delimiter, end = data[quote.end() - 1], quote.end()
            while end < len(data):
                if data[end] == 92:
                    end += 2
                elif data[end] == delimiter:
                    end += 1
                    break
                else:
                    end += 1
            cursor = min(end, len(data))
            continue
        token = _TOKEN.match(data, cursor)
        if token:
            result.append((token.start(), token.end(), token[0]))
            cursor = token.end()
        else:
            cursor += 1
    return result


class _Source:
    def __init__(self, path, data):
        self.path, self.data = path, data
        self.lines = [0] + [m.end() for m in re.finditer(b"\n", data)]
        self.tokens = _tokens(data)
        self.token_starts = [t[0] for t in self.tokens]
        self.conditional_colons = set()
        nesting, questions = [], []
        for start, _, token in self.tokens:
            if token in {b"(", b"[", b"{"}:
                nesting.append(start)
            elif token in {b")", b"]", b"}"}:
                questions = [q for q in questions if q != tuple(nesting)]
                if nesting:
                    nesting.pop()
            elif token == b"?":
                questions.append(tuple(nesting))
            elif token == b":" and questions and questions[-1] == tuple(nesting):
                self.conditional_colons.add(start)
                questions.pop()
            elif token == b";":
                questions = [q for q in questions if q != tuple(nesting)]
        self.directives = []
        # Detect directives only when their # is a real token, never in a literal.
        hashes = {a for a, _, text in self.tokens if text == b"#"}
        for match in _DIRECTIVE.finditer(data):
            if data.find(b"#", match.start(), match.end()) in hashes:
                self.directives.append((match.start(), match.end(), match[1].decode()))

    def span(self, start, end):
        return {"path": self.path, "start_byte": start, "end_byte": end,
                "start_line": bisect_right(self.lines, start),
                "end_line": bisect_right(self.lines, max(start, end - 1))}

    def text(self, start, end):
        return self.data[start:end].decode("utf-8", "replace")

    def field(self, start, end):
        return {"raw": self.text(start, end), **self.span(start, end)}

    def conditions(self, byte):
        stack = []
        for start, end, command in self.directives:
            if start >= byte:
                break
            expression = re.sub(rb"^[ \t]*#[ \t]*[A-Za-z_]+", b"", self.data[start:end], count=1)
            expression = re.sub(rb"\\\r?\n", b"", expression).decode("utf-8", "replace").strip()
            evidence = {"directive": command, "directive_range": self.span(start, end)}
            if command in {"if", "ifdef", "ifndef"}:
                value = expression if command == "if" else ("!" if command == "ifndef" else "") + "defined(" + expression + ")"
                stack.append({"prior": [value], "active": {"expression": value, **evidence}})
            elif command in {"elif", "else"} and stack:
                excluded = "!(" + " || ".join("(" + x + ")" for x in stack[-1]["prior"]) + ")"
                stack[-1]["active"] = {"expression": excluded + (" && (" + expression + ")" if command == "elif" else ""), **evidence}
                if command == "elif":
                    stack[-1]["prior"].append(expression)
            elif command == "endif" and stack:
                stack.pop()
        return [item["active"] for item in stack]


def _owner(declaration):
    node = declaration.parent
    while node is not None:
        if node.type in {"struct_specifier", "class_specifier", "union_specifier"}:
            return node
        if node.type == "function_definition":
            return None
        node = node.parent
    return None


def _recovered_constructor(view, clause):
    declaration = clause.parent
    if declaration.type == "ERROR":
        declaration = declaration.parent
    if declaration is None or declaration.type != "field_declaration":
        return False
    owner = _owner(declaration)
    owner_name = owner.child_by_field_name("name") if owner else None
    if owner_name is None:
        return False
    if owner_name.type == "template_type":
        owner_name = owner_name.child_by_field_name("name")
    if owner_name is None:
        return False
    spelling = view.data[owner_name.start_byte:owner_name.end_byte]
    for node in _walk(declaration):
        if node.type == "function_declarator" and node.end_byte <= clause.start_byte:
            name = node.child_by_field_name("declarator")
            if name is not None and view.data[name.start_byte:name.end_byte] == spelling:
                return True
    return False


class _LexicalColonRoles:
    """Prove a non-field role without trusting recovered AST owner boundaries.

    Raw-tree or independently recognized class headers supply header anchors.
    Physical token balancing supplies real body bounds and the direct enclosing
    scope at a colon; recovered AST ownership is not trusted.
    Preprocessor directive text is excluded from balancing, not branch code.
    Every branch and every token in an actual body stays present in the source.
    """
    def __init__(self, view, root):
        self.view = view
        self.tokens = []
        directive_index = 0
        for token in view.tokens:
            while directive_index < len(view.directives) and view.directives[directive_index][1] <= token[0]:
                directive_index += 1
            if directive_index < len(view.directives):
                a, b, _ = view.directives[directive_index]
                if a <= token[0] < b:
                    continue
            self.tokens.append(token)
        self.index = {t[0]: i for i, t in enumerate(self.tokens)}
        self.macro_names = set()
        for a, b, command in view.directives:
            if command == "define":
                match = re.match(rb"[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", view.data[a:b])
                if match:
                    self.macro_names.add(match[1])
        self.pairs, self.parents = {}, {}
        stack = []
        closers = {b")": b"(", b"]": b"[", b"}": b"{"}
        for i, (_, _, token) in enumerate(self.tokens):
            self.parents[i] = tuple(stack)
            if token in {b"(", b"[", b"{"}:
                stack.append(i)
            elif token in closers:
                if stack and self.tokens[stack[-1]][2] == closers[token]:
                    opening = stack.pop()
                    self.pairs[opening] = i
                    self.pairs[i] = opening
                else:
                    # A crossed/unbalanced delimiter cannot establish scope.
                    stack.clear()
        self.classes, self.asm_arguments = {}, {}
        for node in _walk(root):
            if node.type not in {"struct_specifier", "class_specifier", "union_specifier"}:
                continue
            body, name = node.child_by_field_name("body"), node.child_by_field_name("name")
            if body is None or name is None:
                continue
            header_name = name
            if name.type == "template_type":
                name = name.child_by_field_name("name")
            if name is None:
                continue
            opening = self.index.get(body.start_byte)
            if opening is None or opening not in self.pairs or self.tokens[opening][2] != b"{":
                continue
            keyword = self.index.get(node.start_byte)
            if keyword is None or self.tokens[keyword][2] not in {b"struct", b"class", b"union"}:
                continue
            closing = self.pairs[opening]
            self.classes[opening] = {"name": view.data[name.start_byte:name.end_byte],
                                     "name_range": view.span(name.start_byte, name.end_byte),
                                     "header_name_end": header_name.end_byte,
                                     "keyword_range": view.span(*self.tokens[keyword][:2]),
                                     "body_range": view.span(self.tokens[opening][0], self.tokens[closing][1]),
                                     "header_basis": "raw_class_header_with_lexically_balanced_body",
                                     "closing_index": closing}
        # A malformed template/annotation parse can omit the outer class node
        # entirely. Recognize only a restricted physical class header grammar;
        # template parameters (`class T,` / `class T>` / `class T = ...`) and
        # elaborated type uses (`struct S *p`) cannot pass this state machine.
        for i, (_, _, token) in enumerate(self.tokens):
            if token not in {b"struct", b"class", b"union"} or i + 2 >= len(self.tokens):
                continue
            name_index, j = i + 1, i + 2
            if not re.fullmatch(rb"[A-Za-z_$\x80-\xff][A-Za-z_0-9$\x80-\xff]*", self.tokens[name_index][2]):
                continue
            header_name_end = self.tokens[name_index][1]
            if self.tokens[j][2] == b"<":
                angle, j = 1, j + 1
                while j < len(self.tokens) and angle:
                    item = self.tokens[j][2]
                    if item in {b"(", b"[", b"{"} and j in self.pairs:
                        j = self.pairs[j] + 1
                        continue
                    if item == b"<":
                        angle += 1
                    elif item in {b">", b">>"}:
                        angle -= len(item)
                    elif item == b";":
                        break
                    j += 1
                if angle != 0:
                    continue
                header_name_end = self.tokens[j - 1][1]
            if j < len(self.tokens) and self.tokens[j][2] == b"final":
                j += 1
            if j < len(self.tokens) and self.tokens[j][2] == b":":
                j += 1
                while j < len(self.tokens) and self.tokens[j][2] not in {b"{", b";", b"}", b"="}:
                    if self.tokens[j][2] in {b"(", b"["} and j in self.pairs:
                        j = self.pairs[j] + 1
                    else:
                        j += 1
            if j >= len(self.tokens) or self.tokens[j][2] != b"{" or j not in self.pairs:
                continue
            # The class body begins at the same physical enclosing scope as its
            # keyword; skipping a mismatched delimiter cannot create a class.
            if self.parents[j] != self.parents[i]:
                continue
            closing = self.pairs[j]
            self.classes[j] = {"name": self.tokens[name_index][2],
                               "name_range": view.span(*self.tokens[name_index][:2]),
                               "header_name_end": header_name_end,
                               "keyword_range": view.span(*self.tokens[i][:2]),
                               "body_range": view.span(self.tokens[j][0], self.tokens[closing][1]),
                               "header_basis": "physical_class_keyword_name_template_base_header",
                               "closing_index": closing}
        for i, (_, _, token) in enumerate(self.tokens):
            if token not in {b"asm", b"__asm", b"__asm__"}:
                continue
            j = i + 1
            while j < len(self.tokens) and self.tokens[j][2] in {
                    b"volatile", b"__volatile", b"__volatile__", b"inline", b"goto"}:
                j += 1
            if j < len(self.tokens) and self.tokens[j][2] == b"(" and j in self.pairs:
                self.asm_arguments[j] = {"keyword_range": view.span(*self.tokens[i][:2]),
                                         "argument_range": view.span(self.tokens[j][0], self.tokens[self.pairs[j]][1])}

    def _constructor(self, index, owner):
        tokens, pairs = self.tokens, self.pairs
        before = index - 1
        if before >= 0 and tokens[before][2] == b"noexcept":
            before -= 1
        elif before in pairs and tokens[before][2] == b")":
            opening = pairs[before]
            if opening and tokens[opening - 1][2] == b"noexcept":
                before = opening - 2
        if before < 0 or tokens[before][2] != b")" or before not in pairs:
            return None
        parameter_open = pairs[before]
        if parameter_open == 0 or tokens[parameter_open - 1][2] != owner["name"]:
            return None
        if tokens[parameter_open - 1][2] in self.macro_names:
            return None
        # Seeing ClassName(...) is not sufficient: a same-named macro could
        # expand to a bitfield. Require the actual mem-initializer-list plus a
        # following balanced function body, not merely a ')' before the colon.
        j, initializers = index + 1, []
        while j < owner["closing_index"]:
            begin = j
            if not re.fullmatch(rb"[A-Za-z_$\x80-\xff][A-Za-z_0-9$\x80-\xff]*", tokens[j][2]):
                return None
            angle = 0
            while j < owner["closing_index"]:
                token = tokens[j][2]
                if token == b"<":
                    angle += 1
                elif token in {b">", b">>"}:
                    angle -= len(token)
                if angle < 0 or token in {b";", b"?", b"=", b"}", b":"}:
                    return None
                if angle == 0 and token in {b"(", b"{"}:
                    break
                if angle == 0 and token == b",":
                    return None
                j += 1
            if j not in pairs or pairs[j] >= owner["closing_index"]:
                return None
            j = pairs[j] + 1
            initializers.append(self.view.span(tokens[begin][0], tokens[j - 1][1]))
            if j < len(tokens) and tokens[j][2] == b"...":
                j += 1
            if j < len(tokens) and tokens[j][2] == b",":
                j += 1
                continue
            if j < len(tokens) and tokens[j][2] == b"{" and j in pairs and pairs[j] < owner["closing_index"]:
                return {"classification": "lexically_verified_constructor_initializer_separator",
                        "evidence": {"owner_name_range": owner["name_range"], "owner_body_range": owner["body_range"],
                                     "owner_header_basis": owner["header_basis"],
                                     "constructor_name_range": self.view.span(*tokens[parameter_open - 1][:2]),
                                     "parameter_range": self.view.span(tokens[parameter_open][0], tokens[before][1]),
                                     "initializer_ranges": initializers,
                                     "function_body_range": self.view.span(tokens[j][0], tokens[pairs[j]][1]),
                                     "scope_basis": "Nearest physical enclosing brace is the named class body; balanced initializer arguments precede a separate function body"}}
            return None
        return None

    def macro_member_declarator(self, byte):
        index = self.index.get(byte)
        if index is None or index == 0 or self.tokens[index - 1][2] != b")" or index - 1 not in self.pairs:
            return False
        parents = self.parents[index]
        opening = self.pairs[index - 1]
        return bool(parents and parents[-1] in self.classes and opening
                    and self.tokens[opening - 1][2] in self.macro_names)

    def classify(self, byte):
        index = self.index.get(byte)
        if index is None:
            return None
        parents = self.parents[index]
        if parents and parents[-1] in self.asm_arguments:
            return {"classification": "lexically_verified_asm_argument_separator",
                    "evidence": {**self.asm_arguments[parents[-1]],
                                 "scope_basis": "Colon is directly within balanced asm argument parentheses, not within any operand expression"}}
        if parents and parents[-1] in self.classes:
            found = self._constructor(index, self.classes[parents[-1]])
            if found:
                return found
        # Raw grammar misses a legal decltype base specifier. Match only the
        # exact physical class-name ':' decltype '(' ... ')' '{' sequence.
        for opening, owner in self.classes.items():
            if self.tokens[index][0] < owner["header_name_end"] or index + 2 >= len(self.tokens):
                continue
            if self.tokens[index - 1][1] != owner["header_name_end"]:
                continue
            if self.tokens[index + 1][2] != b"decltype" or self.tokens[index + 2][2] != b"(":
                continue
            closing = self.pairs.get(index + 2)
            if closing is not None and closing + 1 == opening:
                return {"classification": "lexically_verified_decltype_base_separator",
                        "evidence": {"class_keyword_range": owner["keyword_range"], "class_name_range": owner["name_range"],
                                     "base_type_range": self.view.span(self.tokens[index + 1][0], self.tokens[closing][1]),
                                     "class_body_range": owner["body_range"]}}
        return None


def _parse_clause(view, clause):
    """Accept a physical member declarator, not a recovered constructor clause."""
    declaration = clause.parent
    if declaration.type == "ERROR":
        declaration = declaration.parent
    if declaration is None or declaration.type != "field_declaration":
        return None, "clause_not_in_field_declaration"
    if not view.data[declaration.start_byte:declaration.end_byte].rstrip().endswith(b";"):
        return None, "member_declaration_has_no_physical_semicolon"
    declarators = [child for i, child in enumerate(declaration.children)
                   if declaration.field_name_for_child(i) == "declarator"]
    previous = [node for node in declarators if node.end_byte <= clause.start_byte]
    if not previous:
        return None, "bitfield_declarator_not_recovered"
    declarator = previous[-1]
    if declarator.type not in {"field_identifier", "identifier"}:
        return None, "non_identifier_declarator_before_colon"
    if not declarator.is_missing:
        gap = [text for a, _, text in view.tokens if declarator.end_byte <= a < clause.start_byte]
        # Standard attributes are deliberately pending until independently
        # parsed; parentheses/ternaries in this gap often expose ctor recovery.
        if gap:
            return None, "tokens_between_name_and_bitfield_colon"
    owner = _owner(declaration)
    if owner is None:
        return None, "bitfield_owner_not_recovered"
    width_nodes = [node for node in clause.named_children if node.type != "comment"]
    if len(width_nodes) != 1 or width_nodes[0].has_error or width_nodes[0].is_missing:
        return None, "bit_width_expression_not_recovered"
    width = width_nodes[0]
    if any(node.type == "ERROR" for node in _walk(clause)):
        return None, "bit_width_contains_error"
    anonymous = declarator.is_missing or declarator.start_byte == declarator.end_byte
    # Recovery coordinates can lie in the preceding line comment; use the colon.
    start = clause.start_byte if anonymous else declarator.start_byte
    index = declarators.index(declarator)
    owner_id = _identity("bitfield-owner:", view.path, owner.start_byte, owner.type)
    identifier = _identity("bitfield-source:", owner_id, declaration.start_byte, clause.start_byte, index)
    owner_name = owner.child_by_field_name("name")
    record = {
        "bitfield_source_id": identifier,
        "source_declaration_id": _identity("bitfield-declaration:", view.path, declaration.start_byte),
        "declarator_index": index,
        "kind": "anonymous_bitfield" if anonymous else "bitfield",
        "is_bitfield": True, "name": None if anonymous else view.text(declarator.start_byte, declarator.end_byte),
        "name_range": None if anonymous else view.span(declarator.start_byte, declarator.end_byte),
        "declaration": view.field(declaration.start_byte, declaration.end_byte),
        "declarator": view.field(start, clause.end_byte),
        "colon_range": view.span(clause.start_byte, clause.start_byte + 1),
        "bit_width": view.text(width.start_byte, width.end_byte),
        "bit_width_range": view.span(width.start_byte, width.end_byte),
        "initializer": None, "initializer_range": None,
        "owner": {"source_id": owner_id, "kind": owner.type,
                  "name": view.text(owner_name.start_byte, owner_name.end_byte) if owner_name else None,
                  "range": view.span(owner.start_byte, owner.end_byte)},
        "parser_missing_name": bool(declarator.is_missing),
        "preprocessor_conditions": view.conditions(clause.start_byte),
        "semantic_validation": "not_performed",
        "status": "parsed_source_bitfield",
    }
    # A C++20 initializer is a separate sibling, never the bitfield_clause.
    until = next((n.start_byte for n in declarators if n.start_byte > clause.end_byte), declaration.end_byte)
    values = [child for i, child in enumerate(declaration.children)
              if declaration.field_name_for_child(i) == "default_value"
              and clause.end_byte <= child.start_byte < until]
    if values:
        value = values[0]
        record["initializer"] = view.text(value.start_byte, value.end_byte)
        record["initializer_range"] = view.span(value.start_byte, value.end_byte)
        record["declarator"] = view.field(start, value.end_byte)
    return record, None


def _initializer_edits(view, root):
    """Validate a whole member declaration before masking C++20 initializers.

    Some comma lists recover their second width as a generic ERROR node. Raw
    default_value siblings provide candidate initializer extents, not proof:
    after masking only those exact extents, an isolated class-member parse must
    recover every associated identifier/colon/width without any other error.
    """
    edits = []
    for declaration in _walk(root):
        if declaration.type != "field_declaration" or not declaration.has_error or _owner(declaration) is None:
            continue
        if not view.data[declaration.start_byte:declaration.end_byte].rstrip().endswith(b";"):
            continue
        declarators = [child for i, child in enumerate(declaration.children)
                       if declaration.field_name_for_child(i) == "declarator"]
        values = [child for i, child in enumerate(declaration.children)
                  if declaration.field_name_for_child(i) == "default_value" and not child.has_error]
        proposed = []
        for value in values:
            before = [d for d in declarators if d.end_byte <= value.start_byte]
            if not before:
                continue
            name = before[-1]
            if name.type not in {"field_identifier", "identifier"} or name.is_missing:
                continue
            between = view.tokens[bisect_left(view.token_starts, name.end_byte):bisect_left(view.token_starts, value.start_byte)]
            if not between or between[0][2] != b":":
                continue
            colon = between[0][0]
            if value.type == "initializer_list":
                start = value.start_byte
            elif between[-1][2] == b"=":
                start = between[-1][0]
            else:
                continue
            # A raw sibling does not justify deleting malformed width tokens.
            if start <= colon + 1 or any(t[2] in {b";", b"="} for t in between[1:] if t[0] < start):
                continue
            original = view.data[start:value.end_byte]
            replacement = re.sub(rb"[^\r\n]", b" ", original)
            proposed.append({"projection_id": _identity("bitfield-initializer-projection:", view.path, start),
                             **view.span(start, value.end_byte), "kind": "bitfield_initializer_parser_mask",
                             "parser_only": True, "expanded": original.decode("utf-8", "replace"),
                             "semantic_spelling": original.decode("utf-8", "replace"),
                             "parse_projection": replacement.decode("ascii"),
                             "colon_start_byte": colon,
                             "initializer_value": view.text(value.start_byte, value.end_byte),
                             "initializer_range": view.span(value.start_byte, value.end_byte),
                             "declarator_name": view.text(name.start_byte, name.end_byte),
                             "reason": "C++20 bitfield initializer grammar adapter; isolated complete member declaration and each width are checked after equal-length masking"})
        if not proposed:
            continue
        local = bytearray(view.data[declaration.start_byte:declaration.end_byte])
        for edit in proposed:
            a, b = edit["start_byte"] - declaration.start_byte, edit["end_byte"] - declaration.start_byte
            local[a:b] = edit["parse_projection"].encode()
        prefix = b"struct __BitfieldInitializerProbe { "
        probe = prefix + bytes(local) + b" };"
        tree = Parser(LANGUAGE).parse(probe)
        # Legal anonymous fields still require a parser-only temporary name;
        # tolerate only those precise missing nodes, not arbitrary recovery.
        errors = [n for n in _walk(tree.root_node) if n.type == "ERROR" or n.is_missing]
        if any(not (n.is_missing and n.type == "field_identifier" and n.next_named_sibling is not None
                    and n.next_named_sibling.type == "bitfield_clause") for n in errors):
            continue
        clauses = {n.start_byte - len(prefix) + declaration.start_byte: n
                   for n in _walk(tree.root_node) if n.type == "bitfield_clause"}
        verified = True
        for edit in proposed:
            clause = clauses.get(edit["colon_start_byte"])
            if clause is None or clause.has_error:
                verified = False
                break
            previous = clause.prev_named_sibling
            while previous is not None and previous.type == "comment":
                previous = previous.prev_named_sibling
            if previous is None or previous.type != "field_identifier" or probe[previous.start_byte:previous.end_byte].decode() != edit["declarator_name"]:
                verified = False
                break
        if verified:
            edits.extend(proposed)
    return edits


def analyze_bitfields(path: str, source: bytes) -> dict:
    """Inventory physical bitfields and provide source-mapped, parser-only edits.

    Candidate count includes every non-:: colon outside comments/literals. A
    pending candidate is not asserted to be a bitfield. Macro-definition colons
    are retained pending expansion, not silently omitted from this denominator.
    """
    view = _Source(path, source)
    raw_tree = Parser(LANGUAGE).parse(source)
    initializer_edits = _initializer_edits(view, raw_tree.root_node)
    projected = bytearray(source)
    for edit in initializer_edits:
        projected[edit["start_byte"]:edit["end_byte"]] = edit["parse_projection"].encode()
    tree = Parser(LANGUAGE).parse(bytes(projected)) if initializer_edits else raw_tree
    initializer_by_colon = {e["colon_start_byte"]: e for e in initializer_edits}
    clauses = {n.start_byte: n for n in _walk(tree.root_node) if n.type == "bitfield_clause"}
    records, edits, candidates, diagnostics = [], [], [], []
    lexical_roles = None
    for token_index, (start, end, spelling) in enumerate(view.tokens):
        if spelling != b":":
            continue
        candidate = {"candidate_id": _identity("bitfield-colon:", path, start),
                     **view.span(start, end), "status": "pending", "classification": None}
        directive = next((d for d in view.directives if d[0] <= start < d[1]), None)
        leaf = tree.root_node.descendant_for_byte_range(start, end)
        ancestors, node = [], leaf
        while node is not None:
            ancestors.append(node)
            node = node.parent
        record, failure = (None, None)
        if start in view.conditional_colons:
            candidate.update(status="excluded_non_bitfield", classification="lexically_matched_conditional_separator")
        elif directive:
            candidate["classification"] = "preprocessor_colon_requires_expansion"
            candidate["directive"] = directive[2]
        elif start in clauses:
            record, failure = _parse_clause(view, clauses[start])
        if record is not None:
            if start in initializer_by_colon:
                edit = dict(initializer_by_colon[start])
                edit["bitfield_source_id"] = record["bitfield_source_id"]
                record["initializer"] = edit["initializer_value"]
                record["initializer_range"] = edit["initializer_range"]
                record["declarator"] = view.field(record["declarator"]["start_byte"], edit["initializer_range"]["end_byte"])
                edits.append(edit)
            records.append(record)
            candidate.update(status="bitfield", classification="physical_member_bitfield",
                             bitfield_source_id=record["bitfield_source_id"])
            if record["parser_missing_name"]:
                temporary = "__codex_parser_anon_bitfield_" + record["bitfield_source_id"].split(":")[1]
                edits.append({"projection_id": _identity("bitfield-projection:", path, start),
                              **view.span(start, start), "kind": "anonymous_bitfield_parser_name",
                              "parser_only": True, "expanded": "", "semantic_spelling": "",
                              "parse_projection": temporary + " ",
                              "bitfield_source_id": record["bitfield_source_id"],
                              "temporary_field_name": temporary,
                              "restored_name": None,
                              "restoration": "Match this inserted span and source ID, restore name=null; do not globally replace strings",
                              "reason": "Grammar requires a name for anonymous bitfield; insertion anchored at physical colon. Parser projection is not compilable semantic source, especially for zero-width fields"})
        elif not directive and candidate["status"] == "pending":
            non_field = next((n for n in ancestors if n.type in _NON_BITFIELD), None)
            macro_guard = False
            if (non_field is None or non_field.type == "field_initializer_list") and token_index and view.tokens[token_index - 1][2] == b")" and any(d[2] == "define" for d in view.directives):
                if lexical_roles is None:
                    lexical_roles = _LexicalColonRoles(view, tree.root_node)
                macro_guard = lexical_roles.macro_member_declarator(start)
            if macro_guard:
                candidate["classification"] = "macro_generated_declarator_requires_expansion"
            elif non_field is not None:
                candidate.update(status="excluded_non_bitfield", classification=_NON_BITFIELD[non_field.type])
            elif token_index and view.tokens[token_index - 1][2] in {b"public", b"protected", b"private"}:
                candidate.update(status="excluded_non_bitfield", classification="access_label")
            elif any(n.type == "enum_specifier" and (n.child_by_field_name("body") is None or
                     end <= n.child_by_field_name("body").start_byte) for n in ancestors):
                candidate.update(status="excluded_non_bitfield", classification="enum_underlying_type")
            elif any(n.type == "function_definition" and n.child_by_field_name("body") is not None
                     and end <= n.child_by_field_name("body").start_byte for n in ancestors):
                candidate.update(status="excluded_non_bitfield", classification="function_header_non_bitfield_colon")
            elif start in clauses and _recovered_constructor(view, clauses[start]):
                # Require a matching class/constructor name. Merely seeing `)`
                # is unsafe: FIELD(x):3 can be a real macro-generated bitfield.
                candidate.update(status="excluded_non_bitfield", classification="recovered_constructor_initializer_separator")
            else:
                candidate["classification"] = failure or "colon_role_unresolved"
                if lexical_roles is None:
                    lexical_roles = _LexicalColonRoles(view, tree.root_node)
                proof = lexical_roles.classify(start)
                if proof is not None:
                    candidate.update(status="excluded_non_bitfield", classification=proof["classification"],
                                     classification_evidence=proof["evidence"])
        candidate["raw_tree_ancestor_types"] = [n.type for n in ancestors[:8]]
        candidates.append(candidate)
        if candidate["status"] == "pending":
            diagnostics.append({"category": "bitfield_candidate_pending", "blocks_phase_1": True,
                                **view.span(start, end), "candidate_id": candidate["candidate_id"],
                                "message": candidate["classification"]})
    return {"schema_version": SCHEMA_VERSION, "path": path,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "raw_parse_has_error": raw_tree.root_node.has_error,
            "raw_syntax_error_count": sum(n.type == "ERROR" or n.is_missing for n in _walk(raw_tree.root_node)),
            "bitfields": records, "projection_edits": edits, "candidates": candidates,
            "diagnostics": diagnostics,
            "summary": {"colon_candidates": len(candidates), "bitfields": len(records),
                        "anonymous_bitfields": sum(r["name"] is None for r in records),
                        "unique_anonymous_source_ids": len({r["bitfield_source_id"] for r in records if r["name"] is None}),
                        "parser_name_insertions": sum(e["kind"] == "anonymous_bitfield_parser_name" for e in edits), "pending_candidates": len(diagnostics),
                        "candidate_classifications": dict(Counter(c["classification"] for c in candidates))},
            "coverage_claim": "Physical source colon ledger only; pending candidates and macro bodies are not asserted covered; no compiler or API completeness claim"}


def analyze_macro_bitfield_patterns(path: str, source: bytes) -> list[dict]:
    """Independent definition-body shape probe; never closes physical candidates.

    This does not expand a callsite, substitute parameters, or make API entities.
    Each physical macro replacement containing ':' is tested as a class member
    body. Only preprocessing line splices are removed, with piecewise source
    maps. '#'/'##' and unsupported declaration macros remain unproved.
    """
    view = _Source(path, source)
    reports = []
    for start, end, command in view.directives:
        if command != "define":
            continue
        match = re.match(rb"[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", source[start:end])
        if match is None:
            continue
        name = match[1].decode()
        body_start = start + match.end()
        if source[body_start:body_start + 1] == b"(":
            depth, cursor = 1, body_start + 1
            while cursor < end and depth:
                if source[cursor:cursor + 1] == b"(":
                    depth += 1
                elif source[cursor:cursor + 1] == b")":
                    depth -= 1
                cursor += 1
            if depth:
                continue
            body_start = cursor
        while body_start < end and source[body_start] in b" \t":
            body_start += 1
        physical_colons = [view.span(a, b) for a, b, text in view.tokens if body_start <= a < end and text == b":"]
        if not physical_colons:
            continue
        chunks, segments, virtual_cursor, source_cursor = [], [], 0, body_start
        for splice in re.finditer(rb"\\\r?\n", source[body_start:end]):
            a, b = body_start + splice.start(), body_start + splice.end()
            if source_cursor < a:
                chunks.append(source[source_cursor:a])
                segments.append((virtual_cursor, virtual_cursor + a - source_cursor, source_cursor, a))
                virtual_cursor += a - source_cursor
            source_cursor = b
        if source_cursor < end:
            chunks.append(source[source_cursor:end])
            segments.append((virtual_cursor, virtual_cursor + end - source_cursor, source_cursor, end))
        prefix, suffix = b"struct __MacroBitfieldProbe {\n", b"\n};"
        virtual = prefix + b"".join(chunks) + suffix
        probe = analyze_bitfields(path + "::macro:" + name + "@" + str(start), virtual)
        adapted = virtual
        for edit in sorted(probe["projection_edits"], key=lambda e: e["start_byte"], reverse=True):
            adapted = adapted[:edit["start_byte"]] + edit["parse_projection"].encode() + adapted[edit["end_byte"]:]
        tree = Parser(LANGUAGE).parse(adapted)
        owner = next((n for n in tree.root_node.named_children if n.type == "struct_specifier"), None)
        body = owner.child_by_field_name("body") if owner else None
        direct_shape = (not tree.root_node.has_error and body is not None
                        and len([n for n in tree.root_node.named_children if n.type != "comment"]) == 1
                        and all(n.type in {"field_declaration", "comment"} for n in body.named_children))

        def physical_ranges(a, b):
            a, b = a - len(prefix), b - len(prefix)
            return [view.span(x + max(a, u) - u, x + min(b, v) - u)
                    for u, v, x, _ in segments if max(a, u) < min(b, v)]

        patterns = []
        if direct_shape and not probe["diagnostics"]:
            for field in probe["bitfields"]:
                if field["owner"]["name"] != "__MacroBitfieldProbe":
                    continue
                colon = physical_ranges(field["colon_range"]["start_byte"], field["colon_range"]["end_byte"])
                if len(colon) != 1:
                    continue
                name_range = field["name_range"]
                width = field["bit_width_range"]
                patterns.append({"pattern_id": _identity("macro-bitfield-pattern:", path, start, colon[0]["start_byte"]),
                                 "name_template": field["name"], "bit_width_template": field["bit_width"],
                                 "colon_range": colon[0],
                                 "name_ranges": physical_ranges(name_range["start_byte"], name_range["end_byte"]) if name_range else [],
                                 "bit_width_ranges": physical_ranges(width["start_byte"], width["end_byte"]),
                                 "initializer_template": field["initializer"]})
        reports.append({"macro_definition_id": _identity("macro-bitfield-definition:", path, start),
                        "name": name, "definition_range": view.span(start, end),
                        "replacement_range": view.span(body_start, end),
                        "source_colon_candidates": physical_colons,
                        "virtual_source": virtual.decode("utf-8", "replace"),
                        "source_segments": [{"virtual_start_byte": a + len(prefix), "virtual_end_byte": b + len(prefix),
                                             "physical_range": view.span(x, y)} for a, b, x, y in segments],
                        "status": "confirmed_class_member_bitfield_pattern" if patterns else "unproved_class_member_bitfield_pattern",
                        "bitfield_patterns": patterns,
                        "changes_physical_candidate_classification": False,
                        "coverage_claim": "Definition-body syntax shape only; no callsite, argument substitution, generated entity, or semantic validity is inferred"})
    return reports


def audit_scope(scope_path, snapshot):
    """Read all manifest members, verify byte hashes, retain per-file findings."""
    manifest = json.loads(Path(scope_path).read_text())
    files, totals, anonymous = [], Counter(), set()
    for entry in manifest["files"]:
        source = (Path(snapshot) / entry["path"]).read_bytes()
        if hashlib.sha256(source).hexdigest() != entry["sha256"]:
            raise ValueError("Fixed source hash mismatch: " + entry["path"])
        result = analyze_bitfields(entry["path"], source)
        files.append(result)
        for key, value in result["summary"].items():
            if isinstance(value, int):
                totals[key] += value
        anonymous.update(r["bitfield_source_id"] for r in result["bitfields"] if r["name"] is None)
    return {"schema_version": SCHEMA_VERSION, "commit": manifest["commit"],
            "files_scanned": len(files), "files_with_bitfields": sum(bool(f["bitfields"]) for f in files),
            "files_with_pending_candidates": sum(bool(f["diagnostics"]) for f in files),
            "totals": dict(totals), "unique_anonymous_source_ids": len(anonymous), "files": files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=Path(__file__).resolve().parents[1] / "data/scope.json")
    parser.add_argument("--snapshot", type=Path, default=Path(__file__).resolve().parents[1] / "snapshot")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    report = audit_scope(args.scope, args.snapshot)
    if args.summary:
        report["files"] = [{"path": f["path"], **f["summary"],
                            "pending_locations": [{k: d[k] for k in ("start_byte", "start_line", "message")}
                                                  for d in f["diagnostics"]]} for f in report["files"]
                           if f["bitfields"] or f["diagnostics"]]
    print(json.dumps(report, ensure_ascii=False, indent=2))

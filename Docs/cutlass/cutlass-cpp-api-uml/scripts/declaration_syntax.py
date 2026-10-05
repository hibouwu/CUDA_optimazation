"""Loss-aware syntax projections for declaration extraction (not C++ rewriting).

The caller must keep semantic_spelling as the API text. parse_projection is
only a grammar adapter, never an executable/API replacement. Conditional
aliases return every source branch as a separately mapped declaration variant.
"""
from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache
import hashlib
import importlib.util
import itertools
from pathlib import Path
import re
import sys

from tree_sitter import Language, Parser
import tree_sitter_cpp

_spec = importlib.util.spec_from_file_location("declaration_syntax_macros", Path(__file__).with_name("macro_expansion.py"))
_macros = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _macros
_spec.loader.exec_module(_macros)
_projection_spec=importlib.util.spec_from_file_location('syntax_source_projection',Path(__file__).with_name('declaration_projection.py'))
_projection=importlib.util.module_from_spec(_projection_spec)
_projection_spec.loader.exec_module(_projection)
LANGUAGE = Language(tree_sitter_cpp.language())


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _id(prefix, path, start, extra=""):
    return prefix + hashlib.sha256(f"{path}:{start}:{extra}".encode()).hexdigest()[:24]


class Source:
    def __init__(self, path, data):
        self.path, self.data = path, data
        self.text = data.decode("latin1")  # Byte-transparent lexical scanning.
        self.visible = _macros.mask_comments_literals(self.text)
        self.line_starts = [0] + [m.end() for m in re.finditer("\n", self.text)]
        self.directives = []
        for match in re.finditer(r"^[ \t]*#[ \t]*([A-Za-z_]+)((?:[^\n]*\\\r?\n)*[^\n]*)", self.visible, re.M):
            command = match[1]
            end = match.end() + (match.end() < len(self.text) and self.text[match.end()] == "\n")
            raw = self.text[match.start():end]
            expression = re.sub(r"^[ \t]*#[ \t]*[A-Za-z_]+", "", raw, count=1)
            expression = re.sub(r"\\\r?\n", "", expression).strip()
            self.directives.append({"command": command, "expression": expression,
                                    **self.span(match.start(), end)})
        self.code = list(self.visible)
        for directive in self.directives:
            for index in range(directive["start_byte"], directive["end_byte"]):
                if self.code[index] not in "\r\n":
                    self.code[index] = " "
        self.code = "".join(self.code)

    def span(self, start, end):
        return {"path": self.path, "start_byte": start, "end_byte": end,
                "start_line": bisect_right(self.line_starts, start),
                "end_line": bisect_right(self.line_starts, max(start, end - 1))}

    def spelling(self, start, end):
        return self.data[start:end].decode("utf-8", "replace")

    def conditions_at(self, byte):
        stack = []
        for directive in self.directives:
            if directive["start_byte"] >= byte:
                break
            command, expression = directive["command"], directive["expression"]
            if command in {"if", "ifdef", "ifndef"}:
                expression = expression if command == "if" else ("!" if command == "ifndef" else "") + f"defined({expression})"
                stack.append({"prior": [expression], "active": condition(expression, directive)})
            elif command in {"elif", "else"} and stack:
                frame = stack[-1]
                excluded = "!(" + " || ".join(f"({item})" for item in frame["prior"]) + ")"
                frame["active"] = condition(excluded + (f" && ({expression})" if command == "elif" else ""), directive)
                if command == "elif":
                    frame["prior"].append(expression)
            elif command == "endif" and stack:
                stack.pop()
        return [frame["active"] for frame in stack]


def condition(expression, directive):
    return {"expression": expression, "directive": directive["command"],
            "directive_span": {k: directive[k] for k in ("path", "start_byte", "end_byte", "start_line", "end_line")},
            "constant_false": expression.strip() == "0"}


def _declaration_end(source, start):
    stack = []
    close = {")": "(", "]": "[", "}": "{"}
    for index in range(start, len(source.code)):
        char = source.code[index]
        if char in "([{":
            stack.append(char)
        elif char in close:
            if stack and stack[-1] == close[char]:
                stack.pop()
            else:
                return None
        elif char == ";" and not stack:
            return index + 1
    return None


def _condition_tree(source, start, end):
    directives = [d for d in source.directives if start <= d["start_byte"] < end]
    position = 0

    def sequence(cursor, stops):
        nonlocal position
        nodes = []
        while position < len(directives):
            directive = directives[position]
            command = directive["command"]
            if command in stops:
                if cursor < directive["start_byte"]:
                    nodes.append({"text_span": (cursor, directive["start_byte"])})
                return nodes, directive["start_byte"]
            if command not in {"if", "ifdef", "ifndef"}:
                raise ValueError(f"Unsupported directive inside declaration: #{command}")
            if cursor < directive["start_byte"]:
                nodes.append({"text_span": (cursor, directive["start_byte"])})
            expression = directive["expression"] if command == "if" else ("!" if command == "ifndef" else "") + f"defined({directive['expression']})"
            prior, branches = [expression], []
            position += 1
            branch_nodes, _ = sequence(directive["end_byte"], {"elif", "else", "endif"})
            branches.append({"condition": condition(expression, directive), "nodes": branch_nodes})
            has_else = False
            while position < len(directives) and directives[position]["command"] in {"elif", "else"}:
                branch_directive = directives[position]
                excluded = "!(" + " || ".join(f"({item})" for item in prior) + ")"
                is_else = branch_directive["command"] == "else"
                branch_expression = excluded if is_else else excluded + f" && ({branch_directive['expression']})"
                if not is_else:
                    prior.append(branch_directive["expression"])
                has_else = has_else or is_else
                position += 1
                branch_nodes, _ = sequence(branch_directive["end_byte"], {"elif", "else", "endif"})
                branches.append({"condition": condition(branch_expression, branch_directive), "nodes": branch_nodes})
            if position >= len(directives) or directives[position]["command"] != "endif":
                raise ValueError("Conditional declaration has no matching #endif")
            closing = directives[position]
            if not has_else:
                excluded = "!(" + " || ".join(f"({item})" for item in prior) + ")"
                branches.append({"condition": condition(excluded, closing), "nodes": []})
            position += 1
            cursor = closing["end_byte"]
            nodes.append({"branches": branches})
        if cursor < end:
            nodes.append({"text_span": (cursor, end)})
        return nodes, end

    nodes, _ = sequence(start, set())
    return nodes


def _expand_tree(nodes, budget=64):
    results = [([], [])]
    for node in nodes:
        options = [([node["text_span"]], [])] if "text_span" in node else [
            (spans, [branch["condition"]] + conditions)
            for branch in node["branches"]
            for spans, conditions in _expand_tree(branch["nodes"], budget)
        ]
        if len(results) * len(options) > budget:
            raise ValueError(f"Conditional declaration exceeds {budget} source variants; not truncated")
        results = [(a + b, ac + bc) for (a, ac), (b, bc) in itertools.product(results, options)]
    return results


def conditional_aliases(source):
    variants, diagnostics = [], []
    parser = Parser(LANGUAGE)
    raw_tree = parser.parse(source.data)
    template_lists = [n for n in _walk(raw_tree.root_node) if n.type == 'template_parameter_list' and not n.has_error]
    for match in re.finditer(r"\busing\s+([A-Za-z_]\w*)\s*=", source.code):
        end = _declaration_end(source, match.end())
        if end is None:
            continue
        directives = [d for d in source.directives if match.start() <= d["start_byte"] < end]
        if not directives:
            continue
        start = match.start()
        # Raw recovery can place a valid template parameter list under ERROR
        # when #if interrupts the alias target. Keep that prefix with the
        # declaration, verified by an isolated complete alias parse.
        for params in template_lists:
            if params.end_byte > start or source.code[params.end_byte:start].strip():
                continue
            prefix = re.search(r'\btemplate\s*$', source.code[:params.start_byte])
            if prefix is None:
                continue
            probe = source.data[prefix.start():start] + b'using __alias_probe = int;'
            probe_tree = parser.parse(probe)
            if not probe_tree.root_node.has_error and len(probe_tree.root_node.named_children) == 1 and probe_tree.root_node.named_children[0].type == 'template_declaration':
                start = prefix.start()
                break
        declaration_span = source.span(start, end)
        declaration_id = _id("conditional-declaration:", source.path, start)
        try:
            choices = _expand_tree(_condition_tree(source, start, end))
        except ValueError as error:
            diagnostics.append({"category": "conditional_declaration_projection_pending", **declaration_span,
                                "message": str(error), "blocks_phase_1": True})
            continue
        for index, (spans, branch_conditions) in enumerate(choices):
            pieces, segments, cursor = [], [], 0
            for a, b in spans:
                piece = source.data[a:b]
                pieces.append(piece)
                segments.append({"virtual_start_byte": cursor, "virtual_end_byte": cursor + len(piece),
                                 "physical_span": source.span(a, b), "mapping": "exact_source_slice"})
                cursor += len(piece)
            virtual_bytes = b"".join(pieces)
            tree = parser.parse(virtual_bytes)
            parse_errors = [{"node_type": n.type, "is_missing": n.is_missing,
                             "virtual_start_byte": n.start_byte, "virtual_end_byte": n.end_byte}
                            for n in _walk(tree.root_node) if n.type == "ERROR" or n.is_missing]
            aliases = [n for n in _walk(tree.root_node) if n.type == "alias_declaration"]
            variants.append({"source_declaration_id": declaration_id,
                             "conditional_variant_id": _id("conditional-variant:", source.path, match.start(), str(index)),
                             "kind": "alias", "name": match[1], "variant_index": index,
                             "declaration_span": declaration_span,
                             "semantic_spelling": source.spelling(start, end),
                             "virtual_source": virtual_bytes.decode("utf-8", "replace"),
                             "segments": segments, "conditions": source.conditions_at(match.start()) + branch_conditions,
                             "target_type": source_text(virtual_bytes, aliases[0].child_by_field_name("type")) if len(aliases) == 1 else None,
                             "parse_errors": parse_errors, "status": "parsed" if not parse_errors and len(aliases) == 1 else "parse_pending"})
            if parse_errors or len(aliases) != 1:
                diagnostics.append({"category": "conditional_declaration_variant_parse_pending", **declaration_span,
                                    "conditional_variant_id": variants[-1]["conditional_variant_id"],
                                    "message": "A source branch did not yield one complete alias; branch retained", "blocks_phase_1": True})
    return variants, diagnostics


def source_text(data, node):
    return data[node.start_byte:node.end_byte].decode("utf-8", "replace") if node else None


@lru_cache(maxsize=1)
def _default_macro_sources():
    root=Path(__file__).resolve().parents[1]/'snapshot'
    paths=('include/cutlass/detail/helper_macros.hpp','include/cute/config.hpp')
    return {p:(root/p).read_bytes()for p in paths if (root/p).exists()}


def assertion_macro_edits(source,macro_sources):
    config=macro_sources.get('include/cute/config.hpp')
    if not config:return [],[]
    definitions={d.name:d for d in _macros.definitions('include/cute/config.hpp',config)}
    expected={'CUTE_STATIC_ASSERT':'static_assert','CUTE_STATIC_V':'decltype(x)::value',
              'CUTE_STATIC_ASSERT_V':'static_assert(decltype(x)::value,##__VA_ARGS__)'}
    edits=[];diagnostics=[]
    for match in re.finditer(r'\b(CUTE_STATIC_ASSERT_V|CUTE_STATIC_ASSERT|CUTE_STATIC_V)\b',source.code):
        name=match[1];d=definitions.get(name)
        if d is None or re.sub(r'\s+','',d.body)!=expected[name]:
            diagnostics.append({'category':'assertion_macro_definition_pending',**source.span(match.start(),match.end()),'message':'Unexpected static assertion/value macro definition','blocks_phase_1':True});continue
        end=match.end();args=[]
        if name=='CUTE_STATIC_ASSERT':expanded='static_assert'
        else:
            opening=end
            while opening<len(source.code)and source.code[opening].isspace():opening+=1
            if opening>=len(source.code)or source.code[opening]!='(':continue
            try:args,end=_macros.arguments(source.text,opening)
            except ValueError as error:
                diagnostics.append({'category':'assertion_macro_arguments_pending',**source.span(match.start(),end),'message':str(error),'blocks_phase_1':True});continue
            if not args or name=='CUTE_STATIC_V'and len(args)!=1:
                diagnostics.append({'category':'assertion_macro_arity_pending',**source.span(match.start(),end),'message':'Invalid macro argument count','blocks_phase_1':True});continue
            expanded='decltype('+args[0]+')::value'
            if name=='CUTE_STATIC_ASSERT_V':expanded='static_assert('+expanded+(', '+', '.join(args[1:])if len(args)>1 else '')+')'
        nested=dependent_type_brace_edits(Source(source.path+'#static-macro',expanded.encode()))
        projected=project_for_parser(expanded.encode(),nested).decode()
        edits.append({'projection_id':_id('syntax-projection:',source.path,match.start(),'static-macro'),
            'kind':'static_assertion_macro'if name!='CUTE_STATIC_V'else'static_value_macro',
            **source.span(match.start(),end),'semantic_spelling':source.spelling(match.start(),end),
            'expanded':expanded,'parse_projection':projected,'parser_only':False,
            'macro_definition':vars(d),'arguments':args,'nested_parser_adaptations':nested,
            'conditions':source.conditions_at(match.start()),
            'restoration_contract':'Known macro semantics preserved; nested grammar adaptations are parser-only'})
    # The containing macro expansion owns nested macro invocation spelling.
    accepted=[]
    for e in sorted(edits,key=lambda x:(x['start_byte'],-x['end_byte'])):
        if not any(a['start_byte']<=e['start_byte']and e['end_byte']<=a['end_byte']for a in accepted):accepted.append(e)
    return accepted,diagnostics


def keyword_macro_edits(source, macro_sources):
    definitions = []
    for path, data in macro_sources.items():
        definition_source = Source(path, data)
        for definition in _macros.definitions(path, data):
            if definition.name == "CUTLASS_CONSTEXPR_IF_CXX17":
                definitions.append({"definition": vars(definition), "expanded": definition.body,
                                    "conditions": definition_source.conditions_at(definition.start_byte)})
    edits, diagnostics = [], []
    for match in re.finditer(r"\bCUTLASS_CONSTEXPR_IF_CXX17\b", source.code):
        span = source.span(match.start(), match.end())
        if {d["expanded"] for d in definitions} != {"constexpr", ""}:
            diagnostics.append({"category": "conditional_keyword_macro_definition_pending", **span,
                                "message": "Expected both fixed constexpr/empty definitions; no replacement guessed", "blocks_phase_1": True})
            continue
        edits.append({"projection_id": _id("syntax-projection:", source.path, match.start(), "keyword"),
                      "kind": "conditional_keyword_macro", **span,
                      "semantic_spelling": source.spelling(match.start(), match.end()),
                      "expanded": source.spelling(match.start(), match.end()), "parse_projection": "constexpr",
                      "parser_only": True, "variants": definitions,
                      "conditions": source.conditions_at(match.start()),
                      "restoration_contract": "Retain macro spelling and both conditional keyword variants; parser choice is not an active language-mode selection"})
    return edits, diagnostics


def dependent_type_brace_edits(source):
    """Parenthesize the type only; typename and every initializer token survive.

    The projection uses Tree-sitter's GNU compound-literal production. Original
    C++17 functional type construction remains the semantic/API spelling.
    """
    edits = []
    for match in re.finditer(r"\btypename\s+", source.code):
        cursor, depth = match.end(), 0
        while cursor < len(source.code):
            char = source.code[cursor]
            if char == "<":
                depth += 1
            elif char == ">":
                if depth == 0:
                    break  # Closing a surrounding template, not this type.
                depth -= 1
            elif depth == 0 and char == "{":
                type_end = cursor
                while type_end > match.start() and source.code[type_end - 1].isspace():
                    type_end -= 1
                type_spelling = source.spelling(match.start(), type_end)
                # No variable declarator may intervene between dependent type and braces.
                visible_type = source.code[match.end():type_end]
                if "::" not in visible_type or re.search(r"\b[A-Za-z_]\w*\s+[A-Za-z_]\w*\s*$", visible_type):
                    break
                original = source.spelling(match.start(), type_end)
                edits.append({"projection_id": _id("syntax-projection:", source.path, match.start(), "dependent-type"),
                              "kind": "dependent_type_braced_construction", **source.span(match.start(), type_end),
                              "semantic_spelling": original, "expanded": original,
                              "parse_projection": "(" + type_spelling + ")", "parser_only": True,
                              "conditions": source.conditions_at(match.start()),
                              "restoration_contract": "Added type parentheses select parser production only; preserve original dependent_type and braced initializer as actual C++ functional construction"})
                break
            elif depth == 0 and (char in ",;=()&*}\n+-/!|%?[]" or char == ":" and not
                                  (cursor + 1 < len(source.code) and source.code[cursor + 1] == ":" or cursor > 0 and source.code[cursor - 1] == ":")):
                break
            cursor += 1
    return edits


def braced_default_edits(source):
    """Preserve braced defaults as source AST data; shield grammar with an ID.

    Temporary identifiers MUST NOT become parameter defaults or API names.
    This adapter changes no declared type, and the original initializer_list is
    the only semantic default value returned by this module.
    """
    parser = Parser(LANGUAGE)
    # CUDA annotation spelling can otherwise hide the enclosing parameter list.
    # This is only a detector projection; emitted edits still address raw bytes.
    detector_source = bytearray(source.data)
    for annotation in re.finditer(r"\b(?:CUTE_HOST_DEVICE|CUTE_HOST|CUTE_DEVICE|CUTLASS_HOST_DEVICE|CUTLASS_HOST|CUTLASS_DEVICE|CUTLASS_GLOBAL|__host__|__device__|__global__|__forceinline__)\b", source.code):
        detector_source[annotation.start():annotation.end()] = b" " * (annotation.end() - annotation.start())
    tree = parser.parse(bytes(detector_source))
    parameter_ranges, recovery_ranges, ambiguous_defaults = [], [], []
    known_values = set()
    for node in _walk(tree.root_node):
        if node.type in {"declaration", "parameter_declaration"}:
            for child in node.named_children:
                declarator = child.child_by_field_name("declarator") if child.type == "init_declarator" else child
                if declarator and declarator.type == "identifier" and child != node.child_by_field_name("type"):
                    known_values.add(source_text(source.data, declarator))
    for node in _walk(tree.root_node):
        if node.type == "parameter_list":
            parameter_ranges.append((node.start_byte, node.end_byte))
        # T&& x={} is sometimes misread as a direct-initialized declaration.
        if node.type == "declaration":
            for declarator in node.named_children:
                if declarator.type == "init_declarator":
                    value = declarator.child_by_field_name("value")
                    if value and value.type == "argument_list":
                        recovery_ranges.append((value.start_byte, value.end_byte))
        if node.type == "optional_parameter_declaration" and node.child_by_field_name("declarator") is None:
            type_node, default_node = node.child_by_field_name("type"), node.child_by_field_name("default_value")
            if type_node and type_node.type == "type_identifier" and source_text(source.data, type_node) in known_values and default_node:
                ambiguous_defaults.append((default_node.start_byte, default_node.end_byte))
    edits = []
    for match in re.finditer(r"(?<![=!<>])=\s*\{", source.code):
        opening = source.code.find("{", match.start(), match.end())
        if any(a <= opening < b for a, b in ambiguous_defaults):
            continue  # Known value assignment, not an unnamed type parameter.
        if not any(a < opening < b for a, b in parameter_ranges):
            recovered = False
            for a, b in recovery_ranges:
                if not a < opening < b:
                    continue
                prefix = source.code[a + 1:match.start()].rsplit(",", 1)[-1].strip()
                # A genuine initializer assignment `S x(y={})` has no parameter
                # type. Only recover the grammar's T&& name / T* name cases.
                if re.search(r"(?:&&?|\*)\s*[A-Za-z_]\w*\s*$", prefix) and re.match(r"[A-Za-z_]\w*(?:::|\s|[<&*])", prefix):
                    recovered = True
            if not recovered:
                continue
        depth, cursor = 1, opening + 1
        while cursor < len(source.code) and depth:
            depth += (source.code[cursor] == "{") - (source.code[cursor] == "}")
            cursor += 1
        if depth:
            continue
        name = f"__codex_parser_braced_default_{opening}"
        original = source.spelling(opening, cursor)
        edits.append({"projection_id": _id("syntax-projection:", source.path, opening, "braced-default"),
                      "kind": "braced_parameter_default", **source.span(opening, cursor),
                      "semantic_spelling": original, "expanded": original, "parse_projection": name,
                      "parser_only": True, "temporary_identifier": name,
                      "semantic_node": {"kind": "initializer_list", "spelling": original},
                      "detection_basis": "parameter_list_or_direct_initializer_recovery_after_equal_length_CUDA_annotation_mask",
                      "conditions": source.conditions_at(opening),
                      "restoration_contract": "Restore parameter.default to semantic_spelling and never expose temporary_identifier as an API name or value"})
    return edits


def unnamed_pointer_parameter_edits(source):
    """Give the parser, not the API, a name for an abstract pointer default.

    Tree-sitter-cpp 0.23.4 recovers `T* = nullptr` as a broken declaration and
    can consume the enclosing namespace's closing brace. A lexical candidate
    alone is insufficient: the named detector must yield a clean optional
    function parameter, not an assignment, macro argument, NTTP or pointer-to-
    function object. The emitted zero-width edit has empty semantic spelling.
    """
    candidates=list(re.finditer(r'\*[ \t\r\n]*(?:(?:const|volatile)\s+)*(?<![=!<>])=(?!=)',source.code))
    if not candidates:return []
    masked=bytearray(source.data)
    for annotation in re.finditer(r'\b(?:CUTE_HOST_DEVICE|CUTE_HOST|CUTE_DEVICE|CUTLASS_HOST_DEVICE|CUTLASS_HOST|CUTLASS_DEVICE|CUTLASS_GLOBAL|__host__|__device__|__global__|__forceinline__)\b',source.code):
        masked[annotation.start():annotation.end()]=b' '*(annotation.end()-annotation.start())
    parser=Parser(LANGUAGE);original=parser.parse(bytes(masked))
    known_good=[n for n in _walk(original.root_node)if n.type=='optional_parameter_declaration'and not n.has_error
                and n.parent and n.parent.type=='parameter_list']
    planned=[]
    for match in candidates:
        at=match.end()-1
        if source.code[at-1]=='*':continue  # `*=` is one operator, never an abstract pointer plus a default.
        if any(n.start_byte<=at<n.end_byte for n in known_good):continue
        name=f'__codex_parser_unnamed_fnparam_{at}'
        while name in source.code:name+='_'
        planned.append({'at':at,'name':name,'insert':' '+name+' '})
    if not planned:return []
    pieces=[];cursor=shift=0
    for candidate in planned:
        at=candidate['at'];pieces.extend([bytes(masked[cursor:at]),candidate['insert'].encode()])
        candidate['name_start']=at+shift+1;shift+=len(candidate['insert']);cursor=at
    pieces.append(bytes(masked[cursor:]));detector=b''.join(pieces);tree=parser.parse(detector)
    identifiers={n.start_byte:n for n in _walk(tree.root_node)if n.type in {'identifier','field_identifier'}}
    edits=[]
    for candidate in planned:
        name=identifiers.get(candidate['name_start'])
        if name is None:continue
        parameter=name
        while parameter and parameter.type not in {'optional_parameter_declaration','parameter_list','template_parameter_list'}:
            parameter=parameter.parent
        if parameter is None or parameter.type!='optional_parameter_declaration'or parameter.has_error:continue
        parameters=parameter.parent
        if parameters is None or parameters.type!='parameter_list':continue
        function=parameters.parent
        if function is None or function.type!='function_declarator':continue
        declarator=function.child_by_field_name('declarator')
        if declarator is None or declarator.type=='parenthesized_declarator':continue
        owner=function.parent
        while owner and owner.type.endswith('declarator'):owner=owner.parent
        if owner is None or owner.type not in {'declaration','field_declaration','function_definition'}:continue
        # Default arguments in a function pointer type are not repaired into
        # valid APIs. A named function/constructor declaration is required.
        if owner.child_by_field_name('type')is None:
            scope=owner.parent
            while scope and scope.type not in {'class_specifier','struct_specifier','translation_unit'}:scope=scope.parent
            if scope is None or scope.type=='translation_unit':continue
            typename=scope.child_by_field_name('name')
            if typename is None or detector[typename.start_byte:typename.end_byte]!=detector[declarator.start_byte:declarator.end_byte]:continue
        at=candidate['at']
        edits.append({'projection_id':_id('syntax-projection:',source.path,at,'unnamed-pointer-parameter'),
            'kind':'unnamed_pointer_parameter_default',**source.span(at,at),'parser_only':True,
            'expanded':'','semantic_spelling':'','parse_projection':candidate['insert'],
            'temporary_identifier':candidate['name'],'conditions':source.conditions_at(at),
            'detection_basis':'clean_optional_function_parameter_after_grammar_only_name_insertion',
            'restoration_contract':'Original parameter remains unnamed; exact type/default/signature and physical scope boundaries are preserved by SourceProjection'})
    return edits


def analyze_syntax(path: str, source: bytes, macro_sources=None):
    """Return all-branch declaration variants and independently mapped edits.

    Core integration order: extract conditional alias variants with their source
    owner's context, shield their original span, apply other projection edits,
    parse, then restore physical ranges and semantic fields using these records.
    Edits inside a conditional alias are omitted from the whole-file edit list;
    analyze each virtual_source separately if it contains further syntax issues.
    """
    view = Source(path, source)
    variants, diagnostics = conditional_aliases(view)
    macro_sources=_default_macro_sources()if macro_sources is None else macro_sources
    keyword_edits, keyword_diagnostics = keyword_macro_edits(view,macro_sources)
    diagnostics.extend(keyword_diagnostics)
    assertion_edits,assertion_diagnostics=assertion_macro_edits(view,macro_sources)
    diagnostics.extend(assertion_diagnostics)
    pointer_edits=unnamed_pointer_parameter_edits(view)
    if pointer_edits:
        # Discover braced defaults after repairing the parameter-list grammar;
        # otherwise error recovery may hide another default in the same list.
        # Emitted edits still address original bytes and preserve original
        # semantic spelling. The temporary detector name never enters API data.
        detector_projection=_projection.SourceProjection(path,source,pointer_edits)
        defaults=braced_default_edits(Source(path,detector_projection.projected))
        defaults=[detector_projection.map_data(edit)for edit in defaults]
        for edit in defaults:edit['projection_id']=_id('syntax-projection:',path,edit['start_byte'],'braced-default')
    else:defaults=braced_default_edits(view)
    nested_syntax=dependent_type_brace_edits(view)+defaults+pointer_edits
    nested_syntax=[e for e in nested_syntax if not any(a['start_byte']<=e['start_byte']and e['end_byte']<=a['end_byte']for a in assertion_edits)]
    candidates = keyword_edits + assertion_edits + nested_syntax
    alias_ranges = {(v["declaration_span"]["start_byte"], v["declaration_span"]["end_byte"]) for v in variants}
    edits = []
    for edit in sorted(candidates, key=lambda e: (e["start_byte"], e["end_byte"])):
        if any(a <= edit["start_byte"] and edit["end_byte"] <= b for a, b in alias_ranges):
            continue
        if edits and edit["start_byte"] < edits[-1]["end_byte"]:
            diagnostics.append({"category": "overlapping_syntax_projection_pending", **view.span(edit["start_byte"], edit["end_byte"]),
                                "message": "Overlapping grammar adaptations retained for explicit reconciliation", "blocks_phase_1": True})
            continue
        edits.append(edit)
    return {"schema_version": 1, "path": path, "source_sha256": hashlib.sha256(source).hexdigest(),
            "generator_fingerprint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "conditional_declaration_variants": variants, "projection_edits": edits, "diagnostics": diagnostics,
            "coverage_claim": "Only listed constructs; no implication that other syntax errors or declaration candidates are covered"}


def project_for_parser(source: bytes, edits):
    """Convenience for fixtures; core should use its source-map implementation."""
    chunks, cursor = [], 0
    for edit in sorted(edits, key=lambda e: e["start_byte"]):
        if edit["start_byte"] < cursor:
            raise ValueError("Overlapping edits")
        chunks.extend((source[cursor:edit["start_byte"]], edit["parse_projection"].encode()))
        cursor = edit["end_byte"]
    chunks.append(source[cursor:])
    return b"".join(chunks)

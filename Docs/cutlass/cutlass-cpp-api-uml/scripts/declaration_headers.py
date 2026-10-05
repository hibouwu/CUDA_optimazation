"""Mapped conditional function headers and CUDA alignment parser adapters.

This module never selects a compilation configuration. Conditional header
variants carry the unchanged shared body and every header condition. Alignment
adapters retain the original __align__ spelling and requested expression; GNU
attribute syntax is only a parser projection, not a target-compiler selection.
"""
from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path
import re

from tree_sitter import Language, Parser
import tree_sitter_cpp

_spec = importlib.util.spec_from_file_location("header_syntax_source", Path(__file__).with_name("declaration_syntax.py"))
_syntax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_syntax)
LANGUAGE = Language(tree_sitter_cpp.language())
ANNOTATIONS = re.compile(r"\b(?:CUTLASS_HOST_DEVICE|CUTLASS_HOST|CUTLASS_DEVICE|CUTLASS_GLOBAL|CUTE_HOST_DEVICE|CUTE_HOST|CUTE_DEVICE|__host__|__device__|__global__|__forceinline__)\b")


def _nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _identity(prefix, path, start, extra=""):
    return prefix + hashlib.sha256(f"{path}:{start}:{extra}".encode()).hexdigest()[:24]


def _gap(source, start, end, category, message, **fields):
    return {"category": category, **source.span(start, end), "blocks_phase_1": True,
            "message": message, "semantic_spelling": source.spelling(start, end), **fields}


def _groups(source):
    stack, groups = [], []
    for directive in source.directives:
        command = directive["command"]
        if command in {"if", "ifdef", "ifndef"}:
            stack.append(directive)
        elif command == "endif" and stack:
            opening = stack.pop()
            groups.append((opening, directive))
    return sorted(groups, key=lambda group: group[0]["start_byte"])


def _prefix_start(code, before):
    """Nearest prior declaration delimiter, respecting balanced inner syntax."""
    stack = []
    pairs = {")": "(", "]": "["}
    for index in range(before - 1, -1, -1):
        char = code[index]
        if char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif not stack and char in ";{}":
            return index + 1
    return 0


def _next_nonspace(code, position):
    while position < len(code) and code[position].isspace():
        position += 1
    return position


def _top_level_declaration_boundary(fragment, prefix):
    depth = max(0, prefix.count("(") - prefix.count(")"))
    brackets = 0
    for char in fragment:
        if char == "(": depth += 1
        elif char == ")": depth = max(0, depth - 1)
        elif char == "[": brackets += 1
        elif char == "]": brackets = max(0, brackets - 1)
        elif not depth and not brackets and char in ";{}": return True
    return False


def _matching_brace(code, opening):
    depth = 0
    for position in range(opening, len(code)):
        depth += (code[position] == "{") - (code[position] == "}")
        if depth == 0:
            return position + 1
    return None


def _declarator_name(node):
    if node is None:
        return None
    if node.type in {"identifier", "field_identifier", "operator_name", "destructor_name", "qualified_identifier"}:
        return node
    child = node.child_by_field_name("declarator")
    if child is not None:
        return _declarator_name(child)
    for child in node.named_children:
        if child.type in {"parameter_list", "comment", "attribute_specifier", "type_qualifier"}:
            continue
        if child.type.endswith("declarator"):
            result = _declarator_name(child)
            if result is not None:
                return result
    return None


def _function_declarator(node):
    if node is None:
        return None
    if node.type == "function_declarator":
        inner = node.child_by_field_name("declarator")
        if inner is not None and inner.type == "parenthesized_declarator":
            return None
        return node
    child = node.child_by_field_name("declarator")
    if child is None:
        child = next((c for c in node.named_children if c.type.endswith("declarator")), None)
    return _function_declarator(child)


def map_virtual_range(segments, start, end):
    """Return discontiguous exact source slices; never invent contiguous text."""
    mapped = []
    for segment in segments:
        a, b = segment["virtual_start_byte"], segment["virtual_end_byte"]
        lo, hi = max(start, a), min(end, b)
        if lo < hi:
            physical = segment["physical_span"]
            mapped.append({"path": physical["path"], "start_byte": physical["start_byte"] + lo - a,
                           "end_byte": physical["start_byte"] + hi - a,
                           "mapping": "exact_source_slice"})
    return mapped


def _parse_signature(header_bytes):
    """Parse a declaration-form header; synthetic ';' is explicitly separate."""
    probe = bytearray(header_bytes + b";")
    visible = _syntax._macros.mask_comments_literals(header_bytes.decode("latin1"))
    adaptations = []
    for match in ANNOTATIONS.finditer(visible):
        probe[match.start():match.end()] = b" " * (match.end() - match.start())
        adaptations.append({"virtual_start_byte": match.start(), "virtual_end_byte": match.end(),
                            "spelling": match[0], "transformation": "equal_length_CUDA_annotation_mask"})
    parser = Parser(LANGUAGE)
    tree = parser.parse(bytes(probe))
    errors = [{"node_type": n.type, "is_missing": n.is_missing,
               "virtual_start_byte": n.start_byte, "virtual_end_byte": n.end_byte}
              for n in _nodes(tree.root_node) if n.type == "ERROR" or n.is_missing]
    declarations = [n for n in tree.root_node.named_children if n.type in {"declaration", "template_declaration"}]
    candidates = []
    for declaration in declarations:
        actual = declaration
        if declaration.type == "template_declaration":
            actual = next((c for c in declaration.named_children if c.type == "declaration"), None)
        if actual is None:
            continue
        declarator = actual.child_by_field_name("declarator")
        function = _function_declarator(declarator)
        if function is not None:
            candidates.append((declaration, actual, declarator, function))
    if len(candidates) != 1:
        return {"status": "signature_parse_pending", "parse_errors": errors,
                "reason": "Header did not yield exactly one function declarator", "parse_adaptations": adaptations}
    declaration, actual, declarator, function = candidates[0]
    text = lambda node: header_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace") if node else None
    name_node = _declarator_name(declarator)
    if name_node is None:
        return {"status": "signature_parse_pending", "parse_errors": errors,
                "reason": "Function declarator has no recoverable name", "parse_adaptations": adaptations}
    parameters = []
    parameter_list = function.child_by_field_name("parameters")
    if parameter_list is not None:
        for parameter in parameter_list.children:
            if not parameter.is_named and parameter.type != "..." or parameter.type == "comment":
                continue
            param_name = _declarator_name(parameter.child_by_field_name("declarator"))
            default = parameter.child_by_field_name("default_value")
            type_end = default.start_byte if default is not None else parameter.end_byte
            type_bytes = header_bytes[parameter.start_byte:type_end]
            if param_name is not None:
                a, b = param_name.start_byte - parameter.start_byte, param_name.end_byte - parameter.start_byte
                type_bytes = type_bytes[:a] + type_bytes[b:]
            type_spelling = re.sub(r"/\*.*?\*/|//[^\n]*", " ", type_bytes.decode("utf-8", "replace"), flags=re.S).rstrip().rstrip("=").rstrip()
            parameters.append({"name": text(param_name), "raw": text(parameter), "type": type_spelling,
                               "default": text(default), "virtual_start_byte": parameter.start_byte,
                               "virtual_end_byte": parameter.end_byte})
    type_node = actual.child_by_field_name("type")
    return_type = (text(type_node) or "") + " " + header_bytes[declarator.start_byte:name_node.start_byte].decode("utf-8", "replace")
    return_cv = [text(c) for c in actual.named_children if c.type == "type_qualifier" and text(c) in {"const", "volatile"}]
    return_type = " ".join(return_cv + [return_type.strip()]).strip() or None
    trailing = next((c for c in function.named_children if c.type == "trailing_return_type"), None)
    if trailing is not None:
        return_type = text(trailing).removeprefix("->").strip()
    template = declaration.child_by_field_name("parameters") if declaration.type == "template_declaration" else None
    qualifiers = [text(c) for c in actual.named_children if c.type in {"storage_class_specifier", "type_qualifier", "function_specifier"}]
    qualifiers.extend(text(c) for c in function.named_children if c.type in {"type_qualifier", "ref_qualifier", "noexcept", "requires_clause"})
    return {"status": "parsed" if not errors else "signature_parse_pending", "parse_errors": errors,
            "name": text(name_node), "return_type": return_type, "parameters": parameters,
            "template_parameters_raw": text(template), "qualifiers": qualifiers,
            "attributes": [x["spelling"] for x in adaptations], "parse_adaptations": adaptations,
            "synthetic_parse_terminator": {"virtual_start_byte": len(header_bytes), "spelling": ";"}}


def conditional_function_headers(source):
    variants, diagnostics, covered = [], [], []
    for opening, closing in _groups(source):
        if any(a <= opening["start_byte"] < b for a, b in covered):
            continue
        prefix_start = _prefix_start(source.code, opening["start_byte"])
        prefix_start = min(_next_nonspace(source.code, prefix_start), opening["start_byte"])
        # Access labels belong to the enclosing class, not to the function's
        # return type or template prefix. Comments/directives are already
        # whitespace in source.code, so this preserves the physical start.
        access=re.match(r'(?:public|protected|private)\s*:\s*',source.code[prefix_start:opening['start_byte']])
        if access:
            prefix_start+=access.end()
        fragment = source.code[opening["end_byte"]:closing["start_byte"]]
        prefix = source.code[prefix_start:opening["start_byte"]]
        if re.search(r"\busing\s+[A-Za-z_]\w*\s*=|\btypedef\b", prefix):
            continue  # Conditional type aliases belong to declaration_syntax.
        if re.match(r"\s*(?:if|else|for|while|switch|return|throw|asm|__asm__)\b", prefix + fragment):
            continue  # Conditions inside statements/inline assembly are not API heads.
        # Completed statements/types inside a conditional are not interrupted
        # declaration heads. They remain in the ordinary all-branch inventory.
        if _top_level_declaration_boundary(fragment, prefix):
            if prefix.strip() and _parse_signature(source.data[prefix_start:opening['start_byte']] + b' __header_probe()')["status"] == "parsed":
                diagnostics.append(_gap(source, prefix_start, closing["end_byte"], "conditional_function_branch_boundary_pending",
                                        "A shared declaration prefix has branch-local terminators or bodies; this module does not silently discard those header variants"))
            continue
        if not re.search(r"\b(?:[A-Za-z_]\w*|operator\s*[^\w\s]+)\s*\(", prefix + fragment):
            continue
        suffix = _next_nonspace(source.code, closing["end_byte"])
        # Complete name/parameter branches followed by a shared body are the
        # bounded supported form. A shared ')' or cv/noexcept suffix is allowed
        # if isolated parsing below proves the reconstructed header complete.
        terminator = suffix
        depth = 0
        while terminator < len(source.code):
            char = source.code[terminator]
            if char == "(": depth += 1
            elif char == ")": depth = max(0, depth - 1)
            elif depth == 0 and char in "{;": break
            elif depth == 0 and char == "}": break
            elif depth == 0 and char == ":" and not (terminator > 0 and source.code[terminator - 1] == ":" or terminator + 1 < len(source.code) and source.code[terminator + 1] == ":"):
                break
            terminator += 1
        if terminator >= len(source.code) or source.code[terminator] not in "{;":
            diagnostics.append(_gap(source, prefix_start, closing["end_byte"], "conditional_function_header_pending",
                                    "No safely delimited common function body/prototype terminator; constructor initializers and split bodies are not guessed"))
            continue
        body_start = terminator if source.code[terminator] == "{" else None
        declaration_end = _matching_brace(source.code, body_start) if body_start is not None else terminator + 1
        if declaration_end is None:
            diagnostics.append(_gap(source, prefix_start, len(source.data), "conditional_function_body_boundary_pending",
                                    "The shared function body does not have an independently balanced closing brace"))
            continue
        try:
            choices = _syntax._expand_tree(_syntax._condition_tree(source, prefix_start, terminator))
        except ValueError as error:
            diagnostics.append(_gap(source, prefix_start, terminator, "conditional_function_header_pending", str(error)))
            continue
        declaration_id = _identity("conditional-function:", source.path, prefix_start)
        local_variants = []
        for index, (spans, branch_conditions) in enumerate(choices):
            pieces, segments, cursor = [], [], 0
            for a, b in spans:
                piece = source.data[a:b]
                pieces.append(piece)
                segments.append({"virtual_start_byte": cursor, "virtual_end_byte": cursor + len(piece),
                                 "physical_span": source.span(a, b), "mapping": "exact_source_slice", "role": "header"})
                cursor += len(piece)
            header_bytes = b"".join(pieces)
            signature = _parse_signature(header_bytes)
            suffix_bytes = source.data[terminator:declaration_end]
            segments.append({"virtual_start_byte": cursor, "virtual_end_byte": cursor + len(suffix_bytes),
                             "physical_span": source.span(terminator, declaration_end), "mapping": "exact_source_slice",
                             "role": "shared_body" if body_start is not None else "prototype_terminator"})
            virtual_source = header_bytes + suffix_bytes
            for parameter in signature.get("parameters", []):
                parameter["source_segments"] = map_virtual_range(segments, parameter["virtual_start_byte"], parameter["virtual_end_byte"])
            variant = {"source_declaration_id": declaration_id,
                       "conditional_variant_id": _identity("header-variant:", source.path, prefix_start, str(index)),
                       "kind": "function_header_variant", "variant_index": index,
                       "declaration_span": source.span(prefix_start, declaration_end),
                       "signature_span": source.span(prefix_start, terminator),
                       "semantic_spelling": source.spelling(prefix_start, terminator),
                       "virtual_signature": header_bytes.decode("utf-8", "replace"),
                       "virtual_source": virtual_source.decode("utf-8", "replace"), "segments": segments,
                       "conditions": source.conditions_at(prefix_start) + branch_conditions,
                       "body_span": source.span(body_start, declaration_end) if body_start is not None else None,
                       "body_virtual_range": {"start_byte": cursor, "end_byte": len(virtual_source)} if body_start is not None else None,
                       "body_status": "retained_for_core_extraction_all_source_branches" if body_start is not None else "no_body_prototype",
                       "signature": signature, "status": "header_parsed" if signature["status"] == "parsed" else "header_parse_pending"}
            local_variants.append(variant)
            if signature["status"] != "parsed":
                diagnostics.append(_gap(source, prefix_start, terminator, "conditional_function_variant_parse_pending",
                                        "One complete header variant did not parse; variant and original source retained",
                                        conditional_variant_id=variant["conditional_variant_id"], parse_errors=signature["parse_errors"]))
        variants.extend(local_variants)
        covered.append((prefix_start, terminator))
    return variants, diagnostics


def alignment_attributes(source):
    edits, attributes, diagnostics = [], [], []
    definitions = _syntax._macros.definitions(source.path, source.data)
    for match in re.finditer(r"\b__align__\s*\(", source.code):
        opening = source.code.find("(", match.start(), match.end())
        try:
            arguments, end = _syntax._macros.arguments(source.text, opening)
        except ValueError as error:
            diagnostics.append(_gap(source, match.start(), len(source.data), "alignment_arguments_pending", str(error)))
            continue
        expression_start, expression_end = opening + 1, end - 1
        expression = source.spelling(expression_start, expression_end)
        attribute = {"attribute_id": _identity("alignment:", source.path, match.start()),
                     "kind": "cuda_requested_alignment", **source.span(match.start(), end),
                     "semantic_spelling": source.spelling(match.start(), end),
                     "alignment_expression": expression,
                     "expression_span": source.span(expression_start, expression_end),
                     "conditions": source.conditions_at(match.start()), "alignment_unit": "bytes",
                     "value_evaluation": "not_evaluated; target compiler and template bindings remain authoritative"}
        attributes.append(attribute)
        prior = [d for d in definitions if d.name == "__align__" and d.start_byte < match.start()]
        if prior:
            diagnostics.append(_gap(source, match.start(), end, "alignment_macro_override_pending",
                                    "A local __align__ definition may change the attribute contract; no built-in replacement assumed",
                                    candidate_definitions=[vars(d) for d in prior]))
            continue
        if len(arguments) != 1 or not arguments[0].strip():
            diagnostics.append(_gap(source, match.start(), end, "alignment_arguments_pending",
                                    "Expected exactly one alignment expression; no argument discarded"))
            continue
        if not re.search(r"\b(?:class|struct|union)\s*$", source.code[:match.start()]):
            diagnostics.append(_gap(source, match.start(), end, "alignment_context_pending",
                                    "Only alignment immediately following a class-key is adapted in this bounded implementation"))
            continue
        edits.append({"projection_id": _identity("header-projection:", source.path, match.start(), "alignment"),
                      "kind": "cuda_alignment_attribute", **source.span(match.start(), end),
                      "semantic_spelling": attribute["semantic_spelling"], "expanded": attribute["semantic_spelling"],
                      "parse_projection": "__attribute__((aligned(" + expression + ")))",
                      "parser_only": True, "alignment_attribute": attribute,
                      "restoration_contract": "Retain original CUDA alignment attribute and requested expression; GNU attribute production is a parser adapter, not an unconditional target-compiler expansion"})
    return edits, attributes, diagnostics


@lru_cache(maxsize=1)
def _cute_alignment_provider():
    path='include/cute/container/alignment.hpp'
    file=Path(__file__).resolve().parents[1]/'snapshot'/path
    if not file.is_file():return None
    raw=file.read_bytes();view=_syntax.Source(path,raw)
    definitions=[d for d in _syntax._macros.definitions(path,raw)if d.name=='CUTE_ALIGNAS']
    expected={'__align__(n)','alignas(n)'}
    if len(definitions)!=2 or any(d.parameters!=['n']for d in definitions)or {re.sub(r'\s+','',d.body)for d in definitions}!=expected:return None
    return {'path':path,'source':raw,'sha256':hashlib.sha256(raw).hexdigest(),
            'definitions':[{'definition':vars(d),'conditions':view.conditions_at(d.start_byte)}for d in definitions]}


@lru_cache(maxsize=1024)
def _alignment_include_edges(path):
    file=Path(__file__).resolve().parents[1]/'snapshot'/path
    if not file.is_file():return ()
    view=_syntax.Source(path,file.read_bytes());edges=[]
    for directive in view.directives:
        match=re.fullmatch(r'[<"]([^>"]+)[>"](?:\s*//[^\n]*)?',directive['expression'])if directive['command']=='include'else None
        if match and not view.conditions_at(directive['start_byte']):
            target='include/'+match[1]
            edges.append({'path':path,'start_line':directive['start_line'],'end_line':directive['end_line'],'target_path':target})
    return tuple(edges)


def _alignment_include_proof(source,before,provider):
    if source.path==provider['path']:return []
    first=[]
    for directive in source.directives:
        if directive['start_byte']>=before:break
        match=re.fullmatch(r'[<"]([^>"]+)[>"](?:\s*//[^\n]*)?',directive['expression'])if directive['command']=='include'else None
        if match and not source.conditions_at(directive['start_byte']):
            first.append({'path':source.path,'start_line':directive['start_line'],'end_line':directive['end_line'],'target_path':'include/'+match[1]})
    queue=[(edge['target_path'],[edge])for edge in first];seen=set()
    while queue:
        path,chain=queue.pop(0)
        if path==provider['path']:return chain
        if path in seen:continue
        seen.add(path)
        queue.extend((e['target_path'],chain+[e])for e in _alignment_include_edges(path))
    return None


def cute_alignment_attributes(source):
    """Fixed CUTE_ALIGNAS semantics, with separate parser spelling and owners."""
    matches=list(re.finditer(r'\bCUTE_ALIGNAS\s*\(',source.code))
    if not matches:return [],[],[],[]
    provider=_cute_alignment_provider();edits=[];attributes=[];diagnostics=[];non_attributes=[]
    local=[d for d in source.directives if d['command']in {'define','undef'}and re.match(r'(?:CUTE_ALIGNAS|__align__)\b',d['expression'])]
    for match in matches:
        opening=source.code.find('(',match.start(),match.end())
        try:arguments,end=_syntax._macros.arguments(source.text,opening)
        except ValueError as error:
            diagnostics.append(_gap(source,match.start(),len(source.data),'alignment_arguments_pending',str(error)));continue
        prior=[d for d in local if d['start_byte']<match.start()]
        if provider and source.path==provider['path']and source.data==provider['source']:
            prior=[d for d in prior if not re.match(r'CUTE_ALIGNAS\b',d['expression'])]
        latest=next((d for d in reversed(prior)if re.match(r'CUTE_ALIGNAS\b',d['expression'])),None)
        if latest and latest['command']=='undef'and not source.conditions_at(latest['start_byte']):
            non_attributes.append({'name':'CUTE_ALIGNAS',**source.span(match.start(),end),
                'classification':'explicitly_undefined_not_assumed_alignment_macro','undef':latest})
            continue
        if prior:
            diagnostics.append(_gap(source,match.start(),end,'alignment_macro_override_pending',
                'Local definition or conditional undef changes CUTE_ALIGNAS/__align__; fixed macro contract is not assumed',directives=prior));continue
        if provider is None:
            diagnostics.append(_gap(source,match.start(),end,'alignment_macro_definition_pending','Fixed CUTE_ALIGNAS definition pair is unavailable or changed'));continue
        chain=_alignment_include_proof(source,match.start(),provider)
        if chain is None:
            diagnostics.append(_gap(source,match.start(),end,'alignment_macro_availability_pending',
                'No unconditional preceding include chain to the verified CUTE_ALIGNAS definition; no macro semantics guessed'));continue
        if len(arguments)!=1 or not arguments[0].strip():
            diagnostics.append(_gap(source,match.start(),end,'alignment_arguments_pending','Expected exactly one alignment expression'));continue
        expression=source.spelling(opening+1,end-1)
        variants=[]
        for record in provider['definitions']:
            definition=record['definition'];body=definition['body']
            spelling=re.sub(r'\bn\b',lambda _:expression,body)
            variants.append({'variant_id':_identity('alignment-expansion:',source.path,match.start(),body),
                'definition':definition,'definition_source_sha256':provider['sha256'],
                'conditions':record['conditions'],'expanded_spelling':spelling})
        attribute={'attribute_id':_identity('alignment:',source.path,match.start()),'kind':'cute_requested_alignment',
            **source.span(match.start(),end),'semantic_spelling':source.spelling(match.start(),end),
            'alignment_expression':expression,'expression_span':source.span(opening+1,end-1),'alignment_unit':'bytes',
            'conditions':source.conditions_at(match.start()),'conditional_expansions':variants,'definition_include_chain':chain,
            'value_evaluation':'not_evaluated; expression and compiler-specific validity remain authoritative'}
        attributes.append(attribute)
        edits.append({'projection_id':_identity('header-projection:',source.path,match.start(),'cute-alignment'),
            'kind':'cute_alignment_attribute',**source.span(match.start(),end),'parser_only':True,
            'semantic_spelling':attribute['semantic_spelling'],'expanded':attribute['semantic_spelling'],
            'parse_projection':'__attribute__((aligned('+expression+')))',
            'alignment_attribute':attribute,'restoration_contract':'Requested alignment and both definition conditions retained; GNU attribute is only parser syntax'})
    if not edits:return [],attributes,diagnostics,non_attributes
    projection=_syntax._projection.SourceProjection(source.path,source.data,edits)
    probe=bytearray(projection.projected)
    visible=_syntax._macros.mask_comments_literals(probe.decode('latin1'))
    for annotation in ANNOTATIONS.finditer(visible):probe[annotation.start():annotation.end()]=b' '*(annotation.end()-annotation.start())
    tree=Parser(LANGUAGE).parse(bytes(probe));accepted=[]
    for edit in edits:
        a,b=edit['projected_start_byte'],edit['projected_end_byte']
        node=next((n for n in _nodes(tree.root_node)if n.type=='attribute_specifier'and n.start_byte==a and n.end_byte==b and not n.has_error),None)
        owner=node.parent if node else None
        kind='type'if owner and owner.type in {'class_specifier','struct_specifier','union_specifier'}else'field'if owner and owner.type=='field_declaration'else None
        type_node=owner.child_by_field_name('type')if owner else None
        if kind=='field'and (type_node is None or b>type_node.start_byte):kind=None
        if kind is None:
            diagnostics.append(_gap(source,edit['start_byte'],edit['end_byte'],'alignment_context_pending',
                'CUTE_ALIGNAS is not a proven class-key attribute or field-prefix attribute; no declaration or effect inferred'));continue
        attribute=edit['alignment_attribute'];attribute['owner_hint']={'kind':kind,'syntax_kind':owner.type,
            'declaration_span':projection.span(owner.start_byte,owner.end_byte)}
        accepted.append(edit)
    return accepted,attributes,diagnostics,non_attributes


def analyze_headers(path: str, source: bytes):
    view = _syntax.Source(path, source)
    variants, diagnostics = conditional_function_headers(view)
    edits, attributes, alignment_diagnostics = alignment_attributes(view)
    cute_edits,cute_attributes,cute_diagnostics,non_attributes=cute_alignment_attributes(view)
    edits.extend(cute_edits);attributes.extend(cute_attributes);alignment_diagnostics.extend(cute_diagnostics)
    diagnostics.extend(alignment_diagnostics)
    return {"schema_version": 1, "path": path, "source_sha256": hashlib.sha256(source).hexdigest(),
            "generator_fingerprint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "conditional_declaration_variants": variants, "projection_edits": edits,
            "alignment_attributes": attributes, "non_alignment_spellings":non_attributes,"diagnostics": diagnostics,
            "coverage_claim": "Conditional headers are independently parsed; shared bodies remain mandatory core extraction work. Only listed alignment positions are adapted."}

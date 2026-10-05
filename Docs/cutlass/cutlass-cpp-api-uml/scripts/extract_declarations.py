#!/usr/bin/env python3
"""Loss-aware, configuration-independent declaration inventory of the fixed snapshot.

This is a source inventory, not a C++ compiler or an assertion of semantic completeness.
No preprocessing branch is selected. Every parser error and unexpanded declaration
macro remains an explicit diagnostic. Offsets always address the original bytes.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from tree_sitter import Language, Parser
import tree_sitter_cpp

# Also support importlib-based unit tests without changing the caller's sys.path.
import importlib.util
import sys
_macro_spec = importlib.util.spec_from_file_location("cutlass_macro_expansion", Path(__file__).with_name("macro_expansion.py"))
_macro_module = importlib.util.module_from_spec(_macro_spec)
sys.modules[_macro_spec.name] = _macro_module
_macro_spec.loader.exec_module(_macro_module)
_projection_spec = importlib.util.spec_from_file_location("cutlass_declaration_projection", Path(__file__).with_name("declaration_projection.py"))
_projection_module = importlib.util.module_from_spec(_projection_spec)
_projection_spec.loader.exec_module(_projection_module)
_namespace_spec = importlib.util.spec_from_file_location("cutlass_namespace_bindings", Path(__file__).with_name("namespace_bindings.py"))
_namespace_module = importlib.util.module_from_spec(_namespace_spec)
_namespace_spec.loader.exec_module(_namespace_module)
_syntax_spec = importlib.util.spec_from_file_location("cutlass_declaration_syntax", Path(__file__).with_name("declaration_syntax.py"))
_syntax_module = importlib.util.module_from_spec(_syntax_spec)
_syntax_spec.loader.exec_module(_syntax_module)
_bitfield_spec = importlib.util.spec_from_file_location('cutlass_declaration_bitfields',Path(__file__).with_name('declaration_bitfields.py'))
_bitfield_module = importlib.util.module_from_spec(_bitfield_spec)
_bitfield_spec.loader.exec_module(_bitfield_module)
_headers_spec=importlib.util.spec_from_file_location('cutlass_declaration_headers',Path(__file__).with_name('declaration_headers.py'))
_headers_module=importlib.util.module_from_spec(_headers_spec)
_headers_spec.loader.exec_module(_headers_module)
_scope_spec=importlib.util.spec_from_file_location('cutlass_scope_integrity',Path(__file__).with_name('declaration_scope_integrity.py'))
_scope_module=importlib.util.module_from_spec(_scope_spec)
_scope_spec.loader.exec_module(_scope_module)
_using_syntax_spec=importlib.util.spec_from_file_location('cutlass_using_syntax',Path(__file__).with_name('using_syntax.py'))
_using_syntax_module=importlib.util.module_from_spec(_using_syntax_spec)
_using_syntax_spec.loader.exec_module(_using_syntax_module)
_using_interface_spec=importlib.util.spec_from_file_location('cutlass_using_interfaces',Path(__file__).with_name('using_interfaces.py'))
_using_interface_module=importlib.util.module_from_spec(_using_interface_spec)
_using_interface_spec.loader.exec_module(_using_interface_module)

LANGUAGE = Language(tree_sitter_cpp.language())
SCHEMA_VERSION = 1
TYPE_NODES = {"class_specifier", "struct_specifier", "union_specifier"}
DECL_NODES = {"declaration", "field_declaration", "function_definition"}
NAME_NODES = {"identifier", "field_identifier", "type_identifier", "operator_name",
              "destructor_name", "qualified_identifier", "template_function",
              "dependent_name", "operator_cast", "structured_binding_declarator"}
# This allowlist removes only spelling whose interface role is an annotation/pragma.
# Original spelling and exact mask spans remain in files[].normalizations.
ANNOTATIONS = {
    "CUTLASS_HOST_DEVICE", "CUTLASS_DEVICE", "CUTLASS_HOST", "CUTLASS_GLOBAL",
    "CUTLASS_PRAGMA_UNROLL", "CUTLASS_PRAGMA_NO_UNROLL", "CUTE_HOST_DEVICE",
    "CUTE_DEVICE", "CUTE_HOST", "CUTE_HOST_RTC", "CUTE_UNROLL", "CUTE_NO_UNROLL",
    "__host__", "__device__", "__global__", "__forceinline__", "__noinline__",
    "__shared__", "__constant__", "__managed__",
    "CUTLASS_LAMBDA_FUNC_INLINE",
}
ANNOTATION_RE = re.compile(rb"\b(?:" + b"|".join(x.encode() for x in sorted(ANNOTATIONS)) + rb")\b")
TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z_0-9$]*|\d+(?:\.\d+)?|::|&&|\.\.\.|->|==|!=|<=|>=|<<|>>|[^\s]")
PARSER_NAME_RE = re.compile(r"\b__codex_parser_unnamed_nttp_[0-9]+\b")


def digest(prefix: str, value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(data.encode()).hexdigest()[:24]


def canonical(value: str, substitutions: dict[str, str] | None = None) -> str:
    value = re.sub(r"/\*.*?\*/|//[^\n]*", " ", value, flags=re.S)
    tokens = TOKEN_RE.findall(value)
    if substitutions:
        tokens = [substitutions.get(t, t) for t in tokens]
    return " ".join(tokens)


def children_field(node, name):
    return [child for i, child in enumerate(node.children) if node.field_name_for_child(i) == name]


def descendants(node, *, include_anonymous=False):
    stack = [node]
    while stack:
        item = stack.pop()
        yield item
        stack.extend(reversed(item.children if include_anonymous else item.named_children))


@dataclass(frozen=True)
class Context:
    scopes: tuple[dict, ...] = ()
    access: str | dict = "public"
    templates: tuple[dict, ...] = ()
    template_prefix_start: int | None = None
    declaration_prefix_start: int | None = None
    in_function: bool = False
    linkage: str | None = None
    friend: bool = False
    binding_conditions: tuple[dict, ...] = ()
    namespace_choices: tuple[tuple[str,str], ...] = ()
    scope_reviews: tuple[dict, ...] = ()


class Extractor:
    def __init__(self, commit: str):
        self.commit = commit
        self.parser = Parser(LANGUAGE)
        self.entities: dict[str, dict] = {}
        self.occurrences: list[dict] = []
        self.diagnostics: list[dict] = []
        self.files: list[dict] = []
        self.using_sources: list[dict] = []
        self.using_interfaces: list[dict] = []
        self._extraction_depth=0

    def extract(self, path: str, source: bytes, initial_context=None, expand_macros=True, project_constraints=True, project_syntax=True, semantic_reader=None, project_bitfields=True) -> dict:
        outer=self._extraction_depth==0
        first_occ=len(self.occurrences);first_diag=len(self.diagnostics)
        self._extraction_depth+=1
        try:
            record=self._extract_impl(path,source,initial_context,expand_macros,project_constraints,project_syntax,semantic_reader,project_bitfields)
        finally:self._extraction_depth-=1
        if outer and not getattr(self,'class_fragment_context',False)and '#'not in path:
            inventory=_using_syntax_module.scan_source_usings(path,source,self.commit)
            result=_using_interface_module.build_using_interfaces(self.commit,path,source,inventory,record.get('using_contexts',[]),self.occurrences[first_occ:])
            record['using_source_ids']=[s['source_using_id']for s in result['sources']]
            record['using_interface_ids']=[i['using_interface_id']for i in result['interfaces']]
            record['using_source_counts']=inventory['counts']
            record['namespace_alias_sources']=inventory['namespace_aliases']
            self.using_sources.extend(result['sources']);self.using_interfaces.extend(result['interfaces'])
            for diagnostic in result['diagnostics']:
                self.diagnostic(diagnostic['category'],start=diagnostic['start_byte'],end=diagnostic['end_byte'],
                    message=diagnostic['message'],using_source_id=diagnostic.get('source_using_id'))
            record['diagnostic_count']=len(self.diagnostics)-first_diag
            if result['diagnostics']:record['status']='extraction_error_or_pending'
        return record

    def _extract_impl(self, path: str, source: bytes, initial_context=None, expand_macros=True, project_constraints=True, project_syntax=True, semantic_reader=None, project_bitfields=True) -> dict:
        if project_bitfields:
            return self.extract_bitfields(path,source,initial_context,expand_macros,project_constraints,project_syntax,semantic_reader)
        if project_syntax:
            syntax = _syntax_module.analyze_syntax(path, source)
            headers=_headers_module.analyze_headers(path,source)
            syntax['header_analysis']=headers
            syntax['projection_edits']+=headers['projection_edits']
            syntax['conditional_declaration_variants']+=headers['conditional_declaration_variants']
            syntax['diagnostics']+=headers['diagnostics']
            if syntax['projection_edits'] or syntax['conditional_declaration_variants'] or syntax['diagnostics']:
                return self.extract_syntax_projection(path, source, syntax, initial_context, expand_macros, project_constraints, semantic_reader)
        if project_constraints:
            traits_path = Path(__file__).resolve().parents[1] / "snapshot/include/cute/util/type_traits.hpp"
            if traits_path.exists():
                edits = _macro_module.constraint_expansions(path, source, traits_path.read_bytes())
                if edits:
                    for index, edit in enumerate(edits):
                        # tree-sitter-cpp 0.23.4 rejects a valid unnamed pointer
                        # non-type template parameter with a default. Give only
                        # the parser a temporary name; never expose it as API.
                        temporary = f"__codex_parser_unnamed_nttp_{index}"
                        edit["parse_projection"] = re.sub(r"\*\s*=", "* " + temporary + " =", edit["expanded"], count=1)
                        if edit["parse_projection"] != edit["expanded"]:
                            edit["parser_adaptation"] = {"temporary_parameter_name": temporary,
                                                         "reason": "Grammar repair for unnamed pointer NTTP; API name restored to null and actual expanded spelling unchanged"}
                    projection = _projection_module.SourceProjection(path, source, edits)
                    first_occ, first_diag = len(self.occurrences), len(self.diagnostics)
                    record = self.extract(path, projection.projected, initial_context, expand_macros, project_constraints=False, project_syntax=False,
                                          semantic_reader=projection.semantic_reader(semantic_reader),project_bitfields=False)
                    for index in range(first_occ, len(self.occurrences)):
                        occurrence = projection.map_data(self.occurrences[index])
                        if "macro_origin" not in occurrence:
                            sig = occurrence["signature_range"]
                            occurrence["expanded_signature"] = PARSER_NAME_RE.sub("", occurrence["raw_signature"])
                            occurrence["raw_signature"] = source[sig["start_byte"]:sig["end_byte"]].decode("utf-8", "replace").rstrip()
                            occurrence["signature_origin"] = "physical_source_with_source_mapped_constraint_expansion"
                        self.occurrences[index] = occurrence
                    for index in range(first_diag, len(self.diagnostics)):
                        self.diagnostics[index] = projection.map_data(self.diagnostics[index])
                    record = projection.map_data(record)
                    record["sha256"] = hashlib.sha256(source).hexdigest()
                    record["bytes"] = len(source)
                    record["constraint_expansions"] = edits
                    record["raw_parse_has_error"] = self.parser.parse(source).root_node.has_error
                    self.files[-1] = record
                    self.restore_context(projection, path, source)
                    record['scope_integrity']=self.scope_integrity
                    return record
        self.path, self.source = path, source
        self.semantic_reader = semantic_reader or (lambda a, b: source[a:b])
        self.parse_initial_context=initial_context
        self.line_starts = [0] + [m.end() for m in re.finditer(b"\n", source)]
        self.lines = source.splitlines(keepends=True)
        self.condition_by_line, self.pp_regions = self.preprocessor_conditions()
        self.raw_tree = self.parser.parse(source)
        self.normalizations = self.annotation_masks()
        self.normalization_ends = [m["end_byte"] for m in self.normalizations]
        self.macro_result = _macro_module.expand_file(path, source) if expand_macros else {"expansions": [], "diagnostics": []}
        self.macro_placeholders = [{**self.span(e["start_byte"], e["end_byte"]),
                                    "spelling": e["name"], "invocation_id": e["invocation_id"],
                                    "transformation": "equal_length_empty_declaration_placeholder",
                                    "reason": "Declaration macro is separately expanded and source-mapped; this preserves surrounding C++ scope"}
                                   for e in self.macro_result["expansions"]]
        masked = bytearray(source)
        for item in self.normalizations + self.macro_placeholders:
            for pos in range(item["start_byte"], item["end_byte"]):
                if masked[pos] not in (10, 13):
                    masked[pos] = 32
            if item.get("invocation_id"):
                masked[item["start_byte"]] = ord(";")
        self.tree = self.parser.parse(bytes(masked))
        self.body_preprocessor_repairs = []
        original_functions = {n.start_byte: n.child_by_field_name("body") for n in descendants(self.tree.root_node)
                              if n.type == "function_definition" and n.child_by_field_name("body")}
        # Recovery can prematurely close a function and misread later `else if
        # constexpr` as namespace declarations. Independently balance lexical
        # braces (comments/literals/directives excluded) before trusting bounds.
        lexical = _macro_module.mask_comments_literals(bytes(masked).decode("latin1"))
        lexical = re.sub(r"^[ \t]*#[^\n]*(?:\\\r?\n[^\n]*)*", lambda match: re.sub(r"[^\r\n]", " ", match[0]), lexical, flags=re.M)
        brace_stack, brace_ends = [], {}
        for match in re.finditer(r"[{}]", lexical):
            if match[0] == "{":
                brace_stack.append(match.start())
            elif brace_stack:
                brace_ends[brace_stack.pop()] = match.end()
        outer_functions = {}
        for start, body in sorted(original_functions.items()):
            end = brace_ends.get(body.start_byte, body.end_byte)
            if not any(a <= start < b for a, b in outer_functions.values()):
                outer_functions[start] = (body.start_byte, end)
        faulty_bodies = [(a, b) for start, (a, b) in outer_functions.items()
                         if original_functions[start].has_error or original_functions[start].end_byte != b]
        repair_regions = [r for r in self.pp_regions if any(a <= r["start_byte"] and r["end_byte"] <= b for a, b in faulty_bodies)]
        if repair_regions:
            repaired_bytes = bytearray(masked)
            for region in repair_regions:
                for pos in range(region["start_byte"], region["end_byte"]):
                    if repaired_bytes[pos] not in (10, 13):
                        repaired_bytes[pos] = 32
            candidate = self.parser.parse(bytes(repaired_bytes))
            before = sum(n.type == "ERROR" or n.is_missing for n in descendants(self.tree.root_node, include_anonymous=True))
            after = sum(n.type == "ERROR" or n.is_missing for n in descendants(candidate.root_node, include_anonymous=True))
            candidate_functions = {n.start_byte: n.child_by_field_name("body") for n in descendants(candidate.root_node)
                                   if n.type == "function_definition" and n.child_by_field_name("body")}
            boundaries_preserved = all(start in candidate_functions and
                                       bounds == (candidate_functions[start].start_byte, candidate_functions[start].end_byte)
                                       for start, bounds in outer_functions.items())
            if after < before and boundaries_preserved:
                self.tree = candidate
                self.body_preprocessor_repairs = [{**r, "transformation": "equal_length_condition_directive_whitespace_in_function_body",
                                                  "reason": "All branch code and original conditions retained; function headers preserved and body extents independently checked with lexical brace balance",
                                                  "parse_errors_before": before, "parse_errors_after": after}
                                                 for r in repair_regions]
        start_occ, start_diag = len(self.occurrences), len(self.diagnostics)
        self.covered_nodes: set[tuple[int, int, str]] = set()
        self.function_bodies = []
        self.context_intervals = []
        self.access_event_cache = {}
        self.non_declaration_macro_uses = []
        self.current_using_contexts=[]
        self.scope_integrity=_scope_module.analyze_scope_integrity(path,source,self.tree)
        self.untrusted_scope_regions=[]
        for finding in self.scope_integrity['diagnostics']+self.scope_integrity['uncertainties']:
            span=finding.get('expected_body_range')or finding.get('parsed_body_range')or finding
            self.untrusted_scope_regions.append({'path':path,'start_byte':min(finding.get('start_byte',span['start_byte']),finding.get('declaration_range',span)['start_byte']),
                'end_byte':max(finding.get('end_byte',span['end_byte']),span['end_byte'],finding.get('affected_tail',span)['end_byte']),
                'scope_id':finding.get('scope_id'),'status':finding.get('status','input_unverified'),
                'proven_mismatch':finding.get('blocks_phase_1',False)})
            if finding.get('blocks_phase_1'):
                self.diagnostic('scope_closure_mismatch',start=finding['start_byte'],end=finding['end_byte'],
                    message=finding['message'],scope_evidence=finding)
        if getattr(self,'class_fragment_context',False):
            wrapper=next((n for n in self.tree.root_node.named_children if n.type=='struct_specifier'and self.text(n.child_by_field_name('name'))=='__codex_parser_class_context'),None)
            # semantic_reader intentionally erases the wrapper name. Locate
            # the unique outer grammar shell by its raw parsed source instead.
            if wrapper is None:
                wrapper=next((n for n in self.tree.root_node.named_children if n.type=='struct_specifier'and n.child_by_field_name('name')and
                    self.source[n.child_by_field_name('name').start_byte:n.child_by_field_name('name').end_byte]==b'__codex_parser_class_context'),None)
            if wrapper and wrapper.child_by_field_name('body'):
                outside=[n for n in self.tree.root_node.named_children if n.type!='comment'and n!=wrapper]
                if outside:
                    self.diagnostic('macro_member_context_escape_pending',self.tree.root_node,
                        message='Macro replacement escapes its class-member context; whole-source scope expansion remains required',
                        unparsed_root_nodes=[{'kind':n.type,**self.span(n.start_byte,n.end_byte)}for n in outside])
                self.walk(wrapper.child_by_field_name('body'),initial_context or Context())
            else:
                self.diagnostic('macro_member_context_parse_pending',self.tree.root_node,message='Member fragment grammar context could not be recovered')
        else:
            self.walk(self.tree.root_node, initial_context or Context())
        self.record_parse_errors(self.tree.root_node, "normalized_source")
        self.pending_macros()
        macro_expansions = self.integrate_macros(start_diag) if expand_macros else []
        errors = [d for d in self.diagnostics[start_diag:] if d["blocks_phase_1"]]
        record = {
            "path": path, "sha256": hashlib.sha256(source).hexdigest(), "bytes": len(source),
            "raw_parse_has_error": self.raw_tree.root_node.has_error,
            "normalized_parse_has_error": self.tree.root_node.has_error,
            "normalizations": self.normalizations + self.macro_placeholders + self.body_preprocessor_repairs, "preprocessor_regions": self.pp_regions,
            "macro_expansions": macro_expansions,
            "scope_integrity": self.scope_integrity,
            "using_contexts":self.current_using_contexts,
            "non_declaration_macro_uses": self.non_declaration_macro_uses,
            "macro_replacement_non_declaration_uses": self.macro_result.get('non_declaration_invocations',[]),
            "occurrence_count": len(self.occurrences) - start_occ,
            "diagnostic_count": len(self.diagnostics) - start_diag,
            "status": "extraction_error_or_pending" if errors else "parsed_needs_independent_reconciliation",
        }
        self.files.append(record)
        return record

    def extract_bitfields(self,path,source,initial_context,expand_macros,project_constraints,project_syntax,semantic_reader):
        analysis=_bitfield_module.analyze_bitfields(path,source)
        identity_mapper=getattr(self,'source_identity_mapper',None)
        if identity_mapper:
            def field_location(span):
                location=getattr(semantic_reader,'origin_span',lambda a,b:{'path':path,'start_byte':a,'end_byte':b})(span['start_byte'],span['end_byte'])
                return identity_mapper(location['start_byte'],location['end_byte'])
            identities={}
            for field in analysis['bitfields']:
                owner=field_location(field['owner']['range']);declaration=field_location(field['declaration']);colon=field_location(field['colon_range'])
                old=field['bitfield_source_id']
                owner_id=_bitfield_module._identity('bitfield-owner:',owner['path'],owner['start_byte'],field['owner']['kind'])
                field['bitfield_source_id']=_bitfield_module._identity('bitfield-source:',owner_id,declaration['start_byte'],colon['start_byte'],field['declarator_index'])
                field['source_declaration_id']=_bitfield_module._identity('bitfield-declaration:',declaration['path'],declaration['start_byte'])
                field['owner']['source_id']=owner_id
                identities[old]=field['bitfield_source_id']
            for item in analysis['projection_edits']+analysis['candidates']:
                if item.get('bitfield_source_id')in identities:item['bitfield_source_id']=identities[item['bitfield_source_id']]
        # A conditional header copies its complete shared body into each
        # source variant. Do not copy parser-only names into those variants:
        # defer inner field adaptations until that body is parsed in context.
        header_groups={}
        if project_syntax and analysis['projection_edits']:
            for variant in _headers_module.analyze_headers(path,source)['conditional_declaration_variants']:
                header_groups.setdefault(variant['source_declaration_id'],[]).append(variant)
        deferred=[variants[0]['declaration_span']for variants in header_groups.values()if all(v['status']=='header_parsed'for v in variants)]
        direct_edits=[e for e in analysis['projection_edits']if not any(s['start_byte']<=e['start_byte']and e['end_byte']<=s['end_byte']for s in deferred)]
        analysis['edits_deferred_to_conditional_bodies']=[e for e in analysis['projection_edits']if e not in direct_edits]
        outer_origin=getattr(semantic_reader,'origin_span',None)
        self.bitfield_source_models={(outer_origin(b['colon_range']['start_byte'],b['colon_range']['end_byte'])['start_byte'] if outer_origin else b['colon_range']['start_byte']):b for b in analysis['bitfields']}
        first_occ,first_diag=len(self.occurrences),len(self.diagnostics)
        projection=_projection_module.SourceProjection(path,source,direct_edits) if direct_edits else None
        record=self.extract(path,projection.projected if projection else source,initial_context,expand_macros,
                            project_constraints,project_syntax,
                            projection.semantic_reader(semantic_reader) if projection else semantic_reader,False)
        if projection:
            for index in range(first_occ,len(self.occurrences)):
                self.occurrences[index]=projection.map_data(self.occurrences[index])
            for index in range(first_diag,len(self.diagnostics)):
                self.diagnostics[index]=projection.map_data(self.diagnostics[index])
            record=projection.map_data(record)
            self.restore_context(projection,path,source)
            record['scope_integrity']=self.scope_integrity
            edits=record.get('constraint_expansions',[])+[e for e in record.get('syntax_analysis',{}).get('projection_edits',[])if not e.get('parser_only')]
            for occurrence in self.occurrences[first_occ:]:
                if occurrence.get('macro_origin') or occurrence.get('conditional_declaration_origin'):continue
                sig=occurrence['signature_range'];a,b=sig['start_byte'],sig['end_byte']
                occurrence['raw_signature']=source[a:b].decode('utf-8','replace').rstrip()
                semantic=source[a:b]
                for edit in sorted(edits,key=lambda e:e['start_byte'],reverse=True):
                    if a<=edit['start_byte'] and edit['end_byte']<=b:
                        semantic=semantic[:edit['start_byte']-a]+edit['expanded'].encode()+semantic[edit['end_byte']-a:]
                occurrence['expanded_signature']=semantic.decode('utf-8','replace').rstrip()
                occurrence['signature_origin']='physical_source_with_bitfield_and_syntax_projection'
        models={b['bitfield_source_id']:b for b in analysis['bitfields']}
        for occurrence in self.occurrences[first_occ:]:
            model=models.get(occurrence.get('bitfield_source_id'))
            if model:
                occurrence['bitfield_origin']=model
                for field in ('name_range','colon_range','bit_width','bit_width_range','initializer','initializer_range'):
                    occurrence[field]=model[field]
                occurrence['declarator_range']={k:v for k,v in model['declarator'].items()if k!='raw'}
                occurrence['declarator']=model['declarator']['raw']
        patterns,pattern_colons=self.macro_bitfield_patterns(path,source,analysis)
        for item in analysis['diagnostics']:
            if item['start_byte'] in pattern_colons:
                continue  # Original candidate retained with the pattern proof below.
            self.diagnostic(item['category'],start=item['start_byte'],end=item['end_byte'],
                            message=item.get('message','Independent bitfield candidate remains unclassified'),bitfield_candidate=item)
        record.update(sha256=hashlib.sha256(source).hexdigest(),bytes=len(source),
                      raw_parse_has_error=analysis['raw_parse_has_error'],bitfield_analysis=analysis,macro_bitfield_patterns=patterns,
                      occurrence_count=len(self.occurrences)-first_occ,diagnostic_count=len(self.diagnostics)-first_diag)
        record['status']='extraction_error_or_pending'if any(d['blocks_phase_1']for d in self.diagnostics[first_diag:])else'parsed_needs_independent_reconciliation'
        self.files[-1]=record
        return record

    def macro_bitfield_patterns(self,path,source,analysis):
        pending={d['start_byte']for d in analysis['diagnostics']if d.get('message')=='preprocessor_colon_requires_expansion'}
        if not pending:return [],{}
        patterns=[];colons={}
        definitions=_macro_module.definitions(path,source)
        for report in _bitfield_module.analyze_macro_bitfield_patterns(path,source):
            if report['status']!='confirmed_class_member_bitfield_pattern':continue
            definition=next((d for d in definitions if d.start_byte==report['definition_range']['start_byte']and d.name==report['name']),None)
            if not definition:continue
            mapped=[]
            for field in report['bitfield_patterns']:
                physical=field['colon_range']['start_byte']
                if physical not in pending or source[physical:physical+1]!=b':':continue
                mapped.append({'physical_colon':field['colon_range'],'pattern_bitfield_source_id':field['pattern_id']})
            if not mapped:continue
            identifier=digest('macro_bitfields_',[path,definition.start_byte,definition.body,definition.parameters])
            proof={'pattern_id':identifier,'definition':vars(definition),'formal_parameters':definition.parameters,
                   'member_context_required':True,'is_concrete_api_instance':False,
                   'substitution_contract':'Replacement tokens form a member declaration pattern before argument substitution; each invocation is separately expanded and checked',
                   'definition_shape_proof':report,'classified_colons':mapped}
            patterns.append(proof)
            colons.update({m['physical_colon']['start_byte']:identifier for m in mapped})
        return patterns,colons

    def extract_class_fragment(self,path,source,context):
        prefix='struct __codex_parser_class_context {\n';suffix='\n};'
        projection=_projection_module.SourceProjection(path,source,[
            {'start_byte':0,'end_byte':0,'expanded':'','parse_projection':prefix,'semantic_spelling':''},
            {'start_byte':len(source),'end_byte':len(source),'expanded':'','parse_projection':suffix,'semantic_spelling':''}])
        self.class_fragment_context=True
        record=self.extract(path,projection.projected,initial_context=context,expand_macros=False,
                            semantic_reader=projection.semantic_reader())
        self.class_fragment_context=False
        for index,occurrence in enumerate(self.occurrences):
            mapped=projection.map_data(occurrence)
            sig=mapped['signature_range']
            if not occurrence.get('conditional_declaration_origin'):
                mapped['raw_signature']=source[sig['start_byte']:sig['end_byte']].decode('utf-8','replace').rstrip()
            origin=mapped.get('bitfield_origin',{})
            owner=origin.get('owner',{})
            if owner.get('name')=='__codex_parser_class_context':
                owner.update(name=context.scopes[-1]['name'],source_scope_entity_id=context.scopes[-1]['identity'],
                             range_is_fragment_grammar_context=True)
            self.occurrences[index]=mapped
        self.diagnostics=[projection.map_data(d)for d in self.diagnostics]
        record=projection.map_data(record)
        record.update(sha256=hashlib.sha256(source).hexdigest(),bytes=len(source),
                      parser_context_adaptation={'kind':'class_member_fragment','prefix':prefix,'suffix':suffix,
                          'scope_entity_id':context.scopes[-1]['identity'],'emits_wrapper_entity':False})
        self.files[-1]=record
        self.restore_context(projection,path,source)
        record['scope_integrity']=self.scope_integrity
        return record

    def restore_context(self, projection, path, source):
        intervals=[]
        for a,b,ctx in self.context_intervals:
            span=projection.span(a,b)
            mapped=replace(ctx,
                access=projection.map_data(ctx.access),
                scopes=tuple(projection.map_data(x)for x in ctx.scopes),
                templates=tuple(projection.map_data(x)for x in ctx.templates),
                binding_conditions=tuple(projection.map_data(x)for x in ctx.binding_conditions),
                template_prefix_start=None if ctx.template_prefix_start is None else projection.offset(ctx.template_prefix_start),
                declaration_prefix_start=None if ctx.declaration_prefix_start is None else projection.offset(ctx.declaration_prefix_start))
            intervals.append((span['start_byte'],span['end_byte'],mapped))
        self.context_intervals=intervals
        self.function_bodies=[(projection.offset(a),projection.offset(b,end=True))for a,b in self.function_bodies]
        if hasattr(self,'scope_integrity'):
            original_hash=self.scope_integrity['source_sha256']
            self.scope_integrity=projection.map_data(self.scope_integrity)
            self.scope_integrity.setdefault('projection_input_hashes',[]).append(original_hash)
            self.scope_integrity.update(source_sha256=hashlib.sha256(source).hexdigest(),coordinate_space='mapped_source_bytes')
        self.untrusted_scope_regions=projection.map_data(getattr(self,'untrusted_scope_regions',[]))
        self.current_using_contexts=projection.map_data(getattr(self,'current_using_contexts',[]))
        self.path,self.source=path,source
        self.line_starts=[0]+[m.end()for m in re.finditer(b'\n',source)]
        self.lines=source.splitlines(keepends=True)
        self.condition_by_line,self.pp_regions=self.preprocessor_conditions()

    def contexts_at(self,start,end,initial_context=None):
        containing=[(b-a,ctx)for a,b,ctx in self.context_intervals if a<=start and end<=b]
        minimum=min((size for size,_ in containing),default=0)
        contexts={digest('ctx_',[[s['identity']for s in ctx.scopes],ctx.binding_conditions]):ctx
                  for size,ctx in containing if size==minimum}
        return list(contexts.values()) or [initial_context or Context()]

    def extract_syntax_projection(self,path,source,syntax,initial_context,expand_macros,project_constraints,semantic_reader=None):
        groups={}
        for variant in syntax['conditional_declaration_variants']:
            groups.setdefault(variant['source_declaration_id'],[]).append(variant)
        accepted={k:v for k,v in groups.items()if all(x['status']in {'parsed','header_parsed'}for x in v)}
        edits=list(syntax['projection_edits'])
        for identifier,variants in accepted.items():
            span=variants[0]['declaration_span']
            function=variants[0]['kind']=='function_header_variant'
            edits.append({**span,'expanded':source[span['start_byte']:span['end_byte']].decode('utf-8','replace'),'parse_projection':';',
                          'kind':'conditional_function_placeholder'if function else'conditional_alias_placeholder','source_declaration_id':identifier})
        if not edits:
            record=self.extract(path,source,initial_context,expand_macros,project_constraints,False,semantic_reader,False)
            for d in syntax['diagnostics']:
                self.diagnostic(d['category'],start=d['start_byte'],end=d['end_byte'],message=d['message'])
            record['syntax_analysis']=syntax
            return record
        projection=_projection_module.SourceProjection(path,source,edits)
        first_occ,first_diag=len(self.occurrences),len(self.diagnostics)
        record=self.extract(path,projection.projected,initial_context,expand_macros,project_constraints,False,
                            projection.semantic_reader(semantic_reader),False)
        actual_constraints=record.get('constraint_expansions',[])
        actual_constraints=[projection.map_data(x)for x in actual_constraints]
        for index in range(first_occ,len(self.occurrences)):
            occurrence=projection.map_data(self.occurrences[index])
            parser_signature=occurrence.get('expanded_signature',occurrence['raw_signature'])
            if 'macro_origin'not in occurrence:
                sig=occurrence['signature_range'];a,b=sig['start_byte'],sig['end_byte']
                occurrence['raw_signature']=source[a:b].decode('utf-8','replace').rstrip()
                semantic=source[a:b]
                semantic_edits=actual_constraints+[e for e in syntax['projection_edits']if not e.get('parser_only',False)]
                for edit in sorted(semantic_edits,key=lambda e:e['start_byte'],reverse=True):
                    if a<=edit['start_byte'] and edit['end_byte']<=b:
                        semantic=semantic[:edit['start_byte']-a]+edit['expanded'].encode()+semantic[edit['end_byte']-a:]
                occurrence['expanded_signature']=semantic.decode('utf-8','replace').rstrip()
                occurrence['parser_projection_signature']=parser_signature
                occurrence['signature_origin']='physical_source_with_explicit_syntax_projection'
                conditional=[e for e in syntax['projection_edits']if e['kind']=='conditional_keyword_macro' and a<=e['start_byte']and e['end_byte']<=b]
                if conditional:
                    occurrence['conditional_specifiers']=conditional
                    if not re.search(r'\bconstexpr\b',_macro_module.mask_comments_literals(occurrence['raw_signature'])):
                        for field in ('qualifiers','prefix_specifiers'):
                            occurrence[field]=[v for v in occurrence.get(field,[])if v!='constexpr']
                alignment=[]
                for attribute in syntax.get('header_analysis',{}).get('alignment_attributes',[]):
                    if not a<=attribute['start_byte']<attribute['end_byte']<=b:continue
                    hint=attribute.get('owner_hint')
                    if hint:
                        allowed=occurrence['kind']in TYPE_NODES if hint['kind']=='type'else occurrence['kind']=='member'
                        expected=hint['declaration_span'];actual=occurrence.get('syntax_node_range',{})
                        if not allowed or any(actual.get(k)!=expected.get(k)for k in ('path','start_byte','end_byte')):continue
                    elif attribute['kind']=='cute_requested_alignment' or occurrence['kind']not in TYPE_NODES:continue
                    alignment.append({**attribute,'owner_occurrence_id':occurrence['declaration_occurrence_id'],
                                      'owner_entity_id':occurrence['entity_id']})
                if alignment:occurrence['alignment_specifiers']=alignment
            self.occurrences[index]=occurrence
        for index in range(first_diag,len(self.diagnostics)):
            self.diagnostics[index]=projection.map_data(self.diagnostics[index])
        record=projection.map_data(record)
        self.restore_context(projection,path,source)
        record['scope_integrity']=self.scope_integrity
        for attribute in syntax.get('header_analysis',{}).get('alignment_attributes',[]):
            if attribute['kind']!='cute_requested_alignment'or not attribute.get('owner_hint'):continue
            owners=[o for o in self.occurrences[first_occ:]if o.get('parse_status')=='parsed'and not o.get('scope_review_required')
                    and any(x['attribute_id']==attribute['attribute_id']for x in o.get('alignment_specifiers',[]))]
            attribute['owner_occurrence_ids']=[o['declaration_occurrence_id']for o in owners]
            attribute['owner_entity_ids']=sorted({o['entity_id']for o in owners})
            if not owners:
                self.diagnostic('alignment_owner_pending',start=attribute['start_byte'],end=attribute['end_byte'],
                    message='Alignment has parser syntax but no verified declaration owner; attribute is not discharged')
                continue
            for diagnostic in self.diagnostics[first_diag:]:
                if diagnostic['category']=='declaration_macro_expansion_pending'and (diagnostic['start_byte'],diagnostic['end_byte'])==(attribute['start_byte'],attribute['end_byte']):
                    diagnostic.update(blocks_phase_1=False,resolution='verified_alignment_attribute_with_exact_declaration_owner',
                        alignment_attribute_id=attribute['attribute_id'],owner_occurrence_ids=attribute['owner_occurrence_ids'])
        integrated=[]
        for identifier,variants in accepted.items():
            span=variants[0]['declaration_span']
            for context in self.contexts_at(span['start_byte'],span['end_byte'],initial_context):
                for variant in variants:
                    is_header=variant['kind']=='function_header_variant'
                    virtual={'invocation_id':variant['conditional_variant_id'],'path':path,
                             'name':variant['signature']['name']if is_header else variant['name'],'start_byte':span['start_byte'],'end_byte':span['end_byte'],
                             'virtual_source':variant['virtual_source'],'definition':None,'bindings':{}}
                    if is_header:virtual['conditional_source_segments']=variant['segments']
                    if is_header:virtual['enclosing_semantic_reader']=semantic_reader
                    variant_context=replace(context,binding_conditions=context.binding_conditions+tuple(variant['conditions']))
                    begin=len(self.occurrences)
                    result=self.integrate_macro_context(virtual,variant_context,first_diag)
                    for occurrence in self.occurrences[begin:]:
                        occurrence.pop('macro_origin',None)
                        occurrence['conditional_declaration_origin']=variant
                        occurrence['expanded_signature']=occurrence['raw_signature']
                        sig=occurrence['signature_range']
                        occurrence['raw_signature']=source[sig['start_byte']:sig['end_byte']].decode('utf-8','replace').rstrip() if is_header else variant['semantic_spelling'].rstrip()
                        occurrence['signature_origin']='conditional_source_variant_with_exact_piecewise_mapping'
                    integrated.append({'conditional_variant_id':variant['conditional_variant_id'],
                                       'declaration_occurrence_ids':result['declaration_occurrence_ids'],
                                       'status':result['status']})
        for d in syntax['diagnostics']:
            self.diagnostic(d['category'],start=d['start_byte'],end=d['end_byte'],message=d['message'])
        record.update({'sha256':hashlib.sha256(source).hexdigest(),'bytes':len(source),'using_contexts':self.current_using_contexts,
                       'raw_parse_has_error':self.parser.parse(source).root_node.has_error,
                       'syntax_analysis':syntax,'conditional_variant_integrations':integrated,
                       'occurrence_count':len(self.occurrences)-first_occ,'diagnostic_count':len(self.diagnostics)-first_diag})
        record['status']='extraction_error_or_pending' if any(d['blocks_phase_1']for d in self.diagnostics[first_diag:])else'parsed_needs_independent_reconciliation'
        self.files[-1]=record
        return record

    def text(self, node) -> str:
        return self.semantic_text(node.start_byte, node.end_byte) if node else ""

    def semantic_text(self, start, end):
        return self.semantic_reader(start, end).decode('utf-8', 'replace')

    def identity_location(self,start,end):
        location=getattr(self.semantic_reader,'origin_span',self.span)(start,end)
        mapper=getattr(self,'source_identity_mapper',None)
        return mapper(location['start_byte'],location['end_byte'])if mapper else location

    def span(self, start: int, end: int) -> dict:
        return {"path": self.path, "start_byte": start, "end_byte": end,
                "start_line": bisect.bisect_right(self.line_starts, start),
                "end_line": bisect.bisect_right(self.line_starts, max(start, end - 1))}

    def conditions(self, byte: int) -> list[dict]:
        line = bisect.bisect_right(self.line_starts, byte)
        return self.condition_by_line.get(line, [])

    def diagnostic(self, category, node=None, *, start=None, end=None, message="", blocks=True, **extra):
        if node is not None:
            start, end = node.start_byte, node.end_byte
        item = {"diagnostic_id": digest("diag_", [self.path, category, start, end, message]),
                "category": category, "blocks_phase_1": blocks,
                **self.span(start, end), "message": message,
                "source_excerpt": self.source[start:min(end, start + 1000)].decode("utf-8", "replace"),
                "preprocessor_conditions": self.conditions(start), **extra}
        # Namespace variants can visit one physical source error more than
        # once. The diagnostic is a source obligation, not a second API.
        if not any(d['diagnostic_id']==item['diagnostic_id']for d in self.diagnostics):
            self.diagnostics.append(item)

    def preprocessor_conditions(self):
        stack, by_line, regions = [], {}, []
        continued_until = 0
        for index, raw in enumerate(self.lines, 1):
            by_line[index] = [dict(x["active"]) for x in stack]
            if index <= continued_until:
                continue
            match = re.match(rb"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", raw)
            if not match:
                continue
            command = match[1].decode()
            expression = match[2].decode("utf-8", "replace").rstrip("\r\n")
            end_line = index
            while expression.rstrip().endswith("\\") and end_line < len(self.lines):
                expression = expression.rstrip()[:-1] + self.lines[end_line].decode("utf-8", "replace").rstrip("\r\n")
                end_line += 1
            continued_until = end_line
            expression = expression.strip()
            start = self.line_starts[index - 1]
            end = self.line_starts[end_line] if end_line < len(self.line_starts) else len(self.source)
            region = {"directive": command, "expression": expression, **self.span(start, end)}
            regions.append(region)
            if command in {"if", "ifdef", "ifndef"}:
                expr = expression if command == "if" else ("!" if command == "ifndef" else "") + "defined(" + expression + ")"
                active = {"expression": expr, "directive_line": index, "directive_path":self.path,"directive": command,
                          "constant_false": expr.strip() == "0"}
                stack.append({"branches": [expr], "active": active})
            elif command in {"elif", "else"} and stack:
                frame = stack[-1]
                exclusion = "!(" + " || ".join("(" + x + ")" for x in frame["branches"]) + ")"
                expr = exclusion + (" && (" + expression + ")" if command == "elif" else "")
                frame["active"] = {"expression": expr, "directive_line": index,"directive_path":self.path, "directive": command,
                                   "constant_false": False}
                if command == "elif":
                    frame["branches"].append(expression)
            elif command == "endif" and stack:
                stack.pop()
        return by_line, regions

    def annotation_masks(self):
        # Tree-sitter comment/string/preprocessor ranges protect literal spelling and definitions.
        protected = []
        for node in descendants(self.raw_tree.root_node):
            if node.type in {"comment", "string_literal", "raw_string_literal", "char_literal",
                             "preproc_def", "preproc_function_def", "preproc_include", "preproc_call"}:
                protected.append((node.start_byte, node.end_byte))
        protected.sort()
        masks, cursor = [], 0
        for match in ANNOTATION_RE.finditer(self.source):
            while cursor < len(protected) and protected[cursor][1] <= match.start():
                cursor += 1
            if cursor < len(protected) and protected[cursor][0] <= match.start() < protected[cursor][1]:
                continue
            masks.append({**self.span(match.start(), match.end()), "spelling": match[0].decode(),
                          "transformation": "equal_length_whitespace",
                          "reason": "explicit_annotation_or_pragma_allowlist; original_attributes_retained"})
        return masks

    def template_info(self, node):
        result = []
        params = node.child_by_field_name("parameters")
        if params:
            for index, child in enumerate(params.named_children):
                if child.type == "comment":
                    continue
                default = child.child_by_field_name("default_type") or child.child_by_field_name("default_value")
                declarator = child.child_by_field_name("declarator")
                name = self.declarator_name(declarator) if declarator else None
                if name is None:
                    names = [x for x in child.named_children if x.type in {"type_identifier", "identifier"}]
                    name = names[-1] if names else None
                raw = self.text(child)
                # Defaults are retained but excluded from entity identity.
                identity_end = default.start_byte if default else child.end_byte
                identity = self.semantic_text(child.start_byte,identity_end).rstrip().rstrip("=").rstrip()
                identity = PARSER_NAME_RE.sub("", identity)
                result.append({"index": index, "kind": child.type, "name": self.text(name) or None,
                               "raw": raw, "identity_raw": identity, "default": self.text(default) or None,
                               **self.span(child.start_byte, child.end_byte)})
        return {"raw": self.text(params), "parameters": result,
                "requires": [self.text(c) for c in node.named_children if c.type == "requires_clause"]}

    def substitutions(self, context):
        result = {}
        for depth, template in enumerate(context.templates):
            for index, param in enumerate(template["parameters"]):
                if param["name"]:
                    result[param["name"]] = f"$T{depth}_{index}"
        return result

    def qualified(self, name, context):
        prefix = "::".join(s["name"] for s in context.scopes)
        return prefix + "::" + name if prefix else name

    def access_at(self, body, byte):
        """C++ access is set by the last active label, including #if labels."""
        key = (body.start_byte, body.end_byte)
        if key not in self.access_event_cache:
            events = []
            inherited = self.conditions(body.start_byte)
            for item in descendants(body):
                if item.type != 'access_specifier': continue
                owner = item.parent
                while owner and owner.type != 'field_declaration_list': owner = owner.parent
                if owner != body: continue
                guards = [c for c in self.conditions(item.start_byte) if c not in inherited]
                events.append({'access':self.text(item), 'conditions':guards,
                               'source':self.span(item.start_byte,item.end_byte)})
            self.access_event_cache[key] = sorted(events,key=lambda e:e['source']['start_byte'])
        default = 'private' if body.parent.type == 'class_specifier' else 'public'
        parent_name=body.parent.child_by_field_name('name')
        if getattr(self,'class_fragment_context',False) and parent_name and self.source[parent_name.start_byte:parent_name.end_byte]==b'__codex_parser_class_context':
            default=self.parse_initial_context.access if self.parse_initial_context else default
        decisions = []
        for event in self.access_event_cache[key]:
            if event['source']['end_byte'] > byte: break
            if not event['conditions']:
                default = event['access']; decisions = []
            else:
                decisions.append(event)
        if not decisions: return default
        return {'kind':'last_active_access_label', 'selection':'first_true_condition_in_order',
                'ordered_cases':list(reversed(decisions)), 'otherwise':default}

    def record(self, node, kind, name, context, *, signature_end=None, identity=None, **fields):
        start = min(x for x in (node.start_byte, context.template_prefix_start, context.declaration_prefix_start) if x is not None)
        # An equal-length masked leading annotation is whitespace to the parser,
        # so the syntax node begins after it. Recover its original signature span.
        mask_index = bisect.bisect_right(self.normalization_ends, start) - 1
        while mask_index >= 0:
            mask = self.normalizations[mask_index]
            if self.source[mask["end_byte"]:start].strip():
                break
            start = mask["start_byte"]
            mask_index -= 1
        end = node.end_byte
        signature_end = signature_end if signature_end is not None else end
        raw = self.source[start:signature_end].decode("utf-8", "replace").rstrip()
        template_shape = [[canonical(p["identity_raw"], self.substitutions(context))
                           for p in t["parameters"]] for t in context.templates]
        qualified_name = self.qualified(name if name is not None else fields['display_name'], context)
        attrs = sorted(set(ANNOTATION_RE.findall(raw.encode())))
        attributes = [a.decode() for a in attrs]
        # static at namespace scope and anonymous namespaces are TU-local.
        internal = any(s.get("anonymous") for s in context.scopes) or fields.get("namespace_const_internal", False) or (
            kind in {"function", "variable", "constant", "variable_template"}
            and re.search(r"\bstatic\b", raw[:max(0, raw.find("{"))] if "{" in raw else raw)
            and not any(s["kind"] in TYPE_NODES for s in context.scopes))
        identity_kind = 'class_type' if kind in {'class_specifier','struct_specifier'} else kind
        source_location=self.identity_location(start,end)
        scope_findings=[s for s in getattr(self,'untrusted_scope_regions',[])if s['start_byte']<=node.start_byte<s['end_byte']]+list(context.scope_reviews)
        scope_refs=list({(s.get('path'),s.get('scope_id'),s['status']):{k:v for k,v in s.items()if k not in {'start_byte','end_byte'}}for s in scope_findings}.values())
        key = [identity_kind, [s["identity"] for s in context.scopes], canonical(name or '', self.substitutions(context)),
               identity, template_shape, source_location['path'] if internal else None, context.linkage]
        if scope_refs:key.append({'unverified_scope_identity':source_location,'scope_reviews':scope_refs})
        entity_id = digest("ent_", key)
        declarator_range=fields.get('declarator_range')
        source_declarator=self.identity_location(declarator_range['start_byte'],declarator_range['end_byte'])if declarator_range else None
        source_occurrence_id = digest("srcocc_", [source_location,kind,name,source_declarator])
        occurrence_id = digest("occ_", [source_occurrence_id, entity_id, context.binding_conditions])
        conditions = self.conditions(node.start_byte) + list(context.binding_conditions)
        owner = node.parent
        while owner and owner.type != 'field_declaration_list': owner = owner.parent
        access = self.access_at(owner,node.start_byte) if owner and not context.friend else context.access
        variant_id = digest("var_", [entity_id, conditions, attributes, access, fields.get("replacement") if kind == "macro_definition" else None])
        occurrence = {
            "declaration_occurrence_id": occurrence_id, "entity_id": entity_id, "variant_id": variant_id,
            "source_occurrence_id": source_occurrence_id,
            "kind": kind, "name": name, "qualified_name": qualified_name,
            **self.span(start, end), "syntax_node_range": self.span(node.start_byte, node.end_byte),
            "signature_range": self.span(start, signature_end), "raw_signature": raw,
            "scope_chain": [dict(s) for s in context.scopes], "access": 'conditional' if isinstance(access,dict) else access,
            "access_resolution": access if isinstance(access,dict) else {'kind':'unconditional','value':access},
            "template_parameters": list(context.templates), "preprocessor_conditions": conditions,
            "qualified_name_resolution": "parameterized_macro_binding" if any(s.get("name_resolution", {}).get("status") == "parameterized_macro_binding" for s in context.scopes) else
                "macro_dependent" if any(s.get("name_resolution", {}).get("status") == "macro_dependent" for s in context.scopes) else "source_name",
            "attributes": attributes, "linkage": context.linkage, "internal_linkage": bool(internal),
            "prefix_specifiers": re.findall(r"\b(?:friend|constexpr|consteval|inline|virtual|static|extern|explicit)\b",
                                             self.source[start:node.start_byte].decode("utf-8", "replace")),
            "friend": context.friend, "parse_status": "contains_parse_error" if node.has_error else "parsed",
            "is_definition": bool(node.child_by_field_name("body")), **fields,
        }
        if scope_refs:
            occurrence['scope_review_required']=True
            occurrence['scope_integrity_refs']=scope_refs
            if any(s['proven_mismatch']for s in scope_refs):
                occurrence['local_syntax_parse_status']=occurrence['parse_status']
                occurrence['parse_status']='scope_closure_mismatch'
        self.occurrences.append(occurrence)
        if entity_id not in self.entities:
            self.entities[entity_id] = {
                "entity_id": entity_id, "kind": kind, "name": name, "qualified_name": qualified_name,
                "identity_key": key, "identity_signature": identity,
                "declaration_occurrence_ids": [], "variant_ids": [],
                "merge_basis": "exact_scope_kind_canonical_signature_template_shape_and_linkage",
                "semantic_resolution": "unverified_scope_identity_quarantined"if scope_refs else"source_identity_not_compiler_resolved",
            }
        entity = self.entities[entity_id]
        entity["declaration_occurrence_ids"].append(occurrence_id)
        if variant_id not in entity["variant_ids"]:
            entity["variant_ids"].append(variant_id)
        return occurrence

    def scope_for(self, occurrence, *, anonymous=False):
        return {"name": occurrence["name"], "kind": occurrence["kind"],
                "entity_id": occurrence["entity_id"], "identity": occurrence["entity_id"], "anonymous": anonymous,
                "template_depth": len(occurrence["template_parameters"]),
                "name_resolution": occurrence.get("name_resolution", {"status": "source_name"})}

    def walk(self, node, context):
        typ = node.type
        self.context_intervals.append((node.start_byte, node.end_byte, context))
        if typ in {"comment", "preproc_include", "preproc_arg", "preproc_params", "parameter_list"}:
            return
        if typ == "template_declaration":
            info = self.template_info(node)
            nested = replace(context, templates=context.templates + (info,),
                             template_prefix_start=context.template_prefix_start if context.template_prefix_start is not None else node.start_byte)
            for child in node.named_children:
                if child.type not in {"template_parameter_list", "requires_clause"}:
                    self.walk(child, nested)
            return
        if typ == "namespace_definition":
            name_node, body = node.child_by_field_name("name"), node.child_by_field_name("body")
            location=self.identity_location(node.start_byte,node.end_byte)
            name = self.text(name_node) or f"<anonymous_namespace@{location['path']}>"
            source_parts = name.split("::") if name_node else [name]
            macro_definitions = _macro_module.definitions(self.path, self.source)
            for definition in macro_definitions:
                definition.preprocessor_conditions = self.conditions(definition.start_byte)
            contexts = [context]
            for source_index, source_part in enumerate(source_parts):
                following = []
                for parent in contexts:
                    binding = _namespace_module.resolve_namespace(source_part, self.path, node.start_byte, self.source,
                        macro_definitions, config_macro_definitions(), self.conditions)
                    alternatives = binding.get("alternatives")
                    if binding["status"] == "pending":
                        resolution = {"status": "macro_dependent", "macro_name": source_part,
                                      "candidate_definitions": binding.get("definitions", []), "missing_binding": binding["reason"]}
                        self.diagnostic("namespace_macro_expansion_pending", name_node or node,
                            message="Namespace macro component requires source-grounded binding", name_resolution=resolution)
                        alternatives = [{"components":[source_part], "conditions":[], "resolution":resolution}]
                    elif alternatives is None:
                        alternatives = [{"components":[source_part], "conditions":[], "resolution":{"status":"source_name"}}]
                    for alternative in alternatives:
                        resolution = alternative["resolution"]
                        choices = dict(parent.namespace_choices)
                        key, choice = resolution.get("binding_key"), resolution.get("choice_key")
                        if key in choices and choices[key] != choice:
                            continue  # Same immutable configuration binding cannot choose two branches.
                        new_binding = bool(key and key not in choices)
                        if key: choices[key] = choice
                        child_context = replace(parent,
                            namespace_choices=tuple(sorted(choices.items())),
                            binding_conditions=parent.binding_conditions + tuple(alternative["conditions"] if not key or new_binding else []))
                        for expanded_index, part in enumerate(alternative["components"]):
                            occurrence = self.record(node, "namespace", part, child_context,
                                signature_end=body.start_byte if body else node.end_byte, identity=canonical(part),
                                inline=any(c.type == "inline" for c in node.children), compound_namespace_spelling=name,
                                compound_namespace_component=source_index, expanded_namespace_component=expanded_index,
                                name_resolution=resolution)
                            child_context = replace(child_context,
                                scopes=child_context.scopes + (self.scope_for(occurrence, anonymous=not bool(name_node)),))
                        following.append(child_context)
                contexts = following
            for namespace_context in contexts:
                if body: self.walk(body, replace(namespace_context, template_prefix_start=None))
            return
        if typ in TYPE_NODES:
            marker = (node.start_byte, node.end_byte, typ,tuple(s['identity']for s in context.scopes),digest('cond_',context.binding_conditions))
            if marker in self.covered_nodes:
                return
            self.covered_nodes.add(marker)
            name_node, body = node.child_by_field_name("name"), node.child_by_field_name("body")
            location=self.identity_location(node.start_byte,node.end_byte)
            name = self.text(name_node) or f"<anonymous_{typ}@{location['path']}:{location['start_byte']}>"
            bases = [self.text(c) for c in node.named_children if c.type == "base_class_clause"]
            occurrence = self.record(node, typ, name, context,
                                     signature_end=body.start_byte if body else node.end_byte,
                                     identity=canonical(name, self.substitutions(context)), bases=bases,
                                     specialization="explicit" if context.templates and not context.templates[-1]["parameters"] else
                                     "partial" if name_node and name_node.type == "template_type" else
                                     "primary" if context.templates else "non_template")
            if body:
                nested = replace(context, scopes=context.scopes + (self.scope_for(occurrence, anonymous=not bool(name_node)),),
                                 access="private" if typ == "class_specifier" else "public", template_prefix_start=None)
                self.walk(body, nested)
            return
        if typ == "enum_specifier":
            name_node, body = node.child_by_field_name("name"), node.child_by_field_name("body")
            location=self.identity_location(node.start_byte,node.end_byte)
            name = self.text(name_node) or f"<anonymous_enum@{location['path']}:{location['start_byte']}>"
            occurrence = self.record(node, "enum", name, context, signature_end=body.start_byte if body else node.end_byte,
                                     identity=canonical(name), underlying_type=self.text(node.child_by_field_name("base")) or None,
                                     scoped=any(c.type in {"class", "struct"} for c in node.children))
            if body:
                for child in body.named_children:
                    if child.type == "enumerator":
                        enum_context = replace(context, scopes=context.scopes + (self.scope_for(occurrence),), template_prefix_start=None)
                        self.record(child, "enumerator", self.text(child.child_by_field_name("name")), enum_context,
                                    identity=self.text(child.child_by_field_name("name")),
                                    value=self.text(child.child_by_field_name("value")) or None)
            return
        if typ in {"alias_declaration", "type_definition", "namespace_alias_definition", "using_declaration"}:
            self.alias(node, context)
            return
        if typ in {"preproc_def", "preproc_function_def"}:
            name = self.text(node.child_by_field_name("name"))
            # Macros have file-independent names but each replacement is a variant.
            macro_context = replace(context, scopes=(), templates=(), template_prefix_start=None)
            occurrence = self.record(node, "macro_definition", name, macro_context, identity=name,
                                     replacement=self.text(node.child_by_field_name("value")),
                                     macro_parameters=self.text(node.child_by_field_name("parameters")), is_definition=True)
            return
        if typ == "friend_declaration":
            targets = [c for c in node.named_children if c.type in {'type_identifier','qualified_identifier','template_type','dependent_type'}]
            if len(targets)==1 and not any(c.type in DECL_NODES for c in node.named_children):
                target = targets[0]; name = self.text(target)
                class_key = next((c.type for c in node.children if c.type in {'class','struct','union'}),None)
                lexical = list(context.scopes)
                last_type = next((s for s in reversed(context.scopes)if s['kind']in TYPE_NODES),None)
                own_templates = context.templates[last_type.get('template_depth',0):] if last_type else context.templates
                if class_key and target.type=='type_identifier':
                    owner_scopes = tuple(s for s in context.scopes if s['kind']=='namespace')
                    lookup = 'innermost_namespace_elaborated_friend_declaration'
                    resolved_templates = own_templates
                    lookup_pending = None
                    current_conditions = self.conditions(node.start_byte)+list(context.binding_conditions)
                    # Existing enclosing class members take precedence over
                    # introducing a namespace type. Match physical declarations
                    # in enclosing scopes, not unrelated same-spelled entities.
                    for depth in range(len(context.scopes),-1,-1):
                        prefix = context.scopes[:depth]
                        previous = next((o for o in reversed(self.occurrences)if o['path']==self.path and o['start_byte']<node.start_byte and
                            o['kind']in TYPE_NODES and o['name']==name and
                            not any(c.get('constant_false')for c in o['preprocessor_conditions']) and
                            [s['identity']for s in o['scope_chain']]==[s['identity']for s in prefix]),None)
                        if previous:
                            extra_conditions=[c for c in previous['preprocessor_conditions']if c not in current_conditions]
                            if extra_conditions:
                                lookup_pending={'reason':'Prior type visibility varies across source conditions',
                                    'candidate_occurrence_ids':[previous['declaration_occurrence_id']], 'visibility_conditions':extra_conditions}
                                break
                            owner_scopes=prefix;lookup='prior_enclosing_type_declaration'
                            inherited_depth=prefix[-1].get('template_depth',0) if prefix else 0
                            resolved_templates=tuple(previous['template_parameters'][:inherited_depth])+own_templates
                            break
                        imports=[o for o in self.current_using_contexts if o['source_range']['path']==self.path and o['source_range']['start_byte']<node.start_byte and
                            o['syntax_form']=='using_declaration'and not o.get('namespace_directive')and
                            o.get('terminal_name')==name and not any(c.get('constant_false')for c in o['preprocessor_conditions']) and
                            [s['identity']for s in o['scope_chain']]==[s['identity']for s in prefix]]
                        if imports:
                            lookup_pending={'reason':'A using-declaration participates in elaborated friend type lookup',
                                'using_source_context_refs':[o['context_source_ref']for o in imports],
                                'target_expressions':[o['target_expression']for o in imports]}
                            break
                        enclosing=next((o for o in reversed(self.occurrences)if prefix and o['entity_id']==prefix[-1]['identity'] and o.get('bases')),None)
                        if enclosing:
                            lookup_pending={'reason':'Base class member lookup must precede namespace introduction',
                                'candidate_occurrence_ids':[enclosing['declaration_occurrence_id']], 'base_expressions':enclosing['bases']}
                            break
                    if lookup_pending:
                        occurrence=self.record(node,'friend_type_reference',name,replace(context,friend=True,declaration_prefix_start=node.start_byte),
                            identity=[self.path,node.start_byte,canonical(name)],target_type=name,
                            target_type_range=self.span(target.start_byte,target.end_byte),class_key=class_key,
                            lexical_scope_chain=lexical,is_definition=False,lookup_status='extraction_pending',lookup_details=lookup_pending)
                        self.diagnostic('friend_type_lookup_pending',node,message=lookup_pending['reason'],
                            declaration_occurrence_id=occurrence['declaration_occurrence_id'],lookup_details=lookup_pending)
                        return
                    target_context=replace(context,scopes=owner_scopes,templates=resolved_templates,access='public',friend=True,
                        declaration_prefix_start=node.start_byte)
                    self.record(node,class_key+'_specifier',name,target_context,identity=canonical(name,self.substitutions(target_context)),
                        is_definition=False,bases=[],specialization='primary' if own_templates else 'non_template',
                        lexical_scope_chain=lexical,friend_injection=lookup,
                        lookup_contract='Unqualified elaborated friend type lookup precedes namespace introduction; imports and dependent bindings require relation reconciliation')
                else:
                    # A simple-type friend or qualified/template target grants
                    # friendship; it does not introduce a new class declaration.
                    self.record(node,'friend_type_reference',name,replace(context,friend=True,declaration_prefix_start=node.start_byte),
                        identity=[self.path,node.start_byte,canonical(name)],target_type=name,
                        target_type_range=self.span(target.start_byte,target.end_byte),class_key=class_key,
                        lexical_scope_chain=lexical,is_definition=False,lookup_contract='Friend target reference; not a newly declared class')
                return
            for child in node.named_children:
                self.walk(child, replace(context, friend=True, declaration_prefix_start=node.start_byte))
            return
        if typ == "linkage_specification":
            value = self.text(node.child_by_field_name("value"))
            body = node.child_by_field_name("body")
            if body:
                self.walk(body, replace(context, linkage=value))
            return
        if typ in DECL_NODES:
            self.declaration(node, context)
            return
        # Scope local classes under their owning function and lexical block.
        if typ == "compound_statement" and context.in_function:
            location=self.identity_location(node.start_byte,node.end_byte)
            nested = replace(context, scopes=context.scopes + ({"name": f"<block@{location['start_byte']}>", "kind": "lexical_block",
                              "identity": f"{location['path']}:{location['start_byte']}", "anonymous": True},))
            for child in node.named_children:
                self.walk(child, nested)
            return
        active = context
        # Empty declaration placeholders have no named child. Preserve the
        # access interval itself, including the space occupied by a macro or
        # conditional declaration, rather than falling back to the class's
        # initial default access. Only this immediate field list can change it.
        if typ == 'field_declaration_list':
            self.access_at(node,node.start_byte)  # populate only this body's labels
            starts = [node.start_byte] + [e['source']['end_byte'] for e in self.access_event_cache[(node.start_byte,node.end_byte)]]
            for a,b in zip(starts,starts[1:]+[node.end_byte]):
                self.context_intervals.append((a,b,replace(context,access=self.access_at(node,a))))
            active = context
        for child in node.named_children:
            if child.type == "access_specifier":
                active = replace(active, access=self.text(child))
            else:
                self.walk(child, active)

    def capture_using_context(self,node,context):
        target=node.child_by_field_name('type')if node.type=='alias_declaration'else next((c for c in node.named_children if c.type!='comment'),None)
        target_text=self.text(target)
        terminal=target
        while terminal and terminal.child_by_field_name('name'):terminal=terminal.child_by_field_name('name')
        classes=[i for i,s in enumerate(context.scopes)if s['kind']in TYPE_NODES]
        functions=[i for i,s in enumerate(context.scopes)if s['kind']in {'function','method','constructor','destructor','operator'}]
        last_class=max(classes,default=-1);last_function=max(functions,default=-1)
        class_context=last_class>last_function
        if node.type=='alias_declaration'and not(functions and not class_context):
            item={'source_range':self.span(node.start_byte,node.end_byte),'syntax_form':node.type,
                'context_source_ref':self.identity_location(node.start_byte,node.end_byte),
                'syntax_status':'contains_parse_error'if node.has_error else'parsed',
                'preprocessor_conditions':self.conditions(node.start_byte)+list(context.binding_conditions)}
            self.current_using_contexts.append(item)
            return item
        owner_node=node.parent
        while owner_node and owner_node.type!='field_declaration_list':owner_node=owner_node.parent
        access=self.access_at(owner_node,node.start_byte)if class_context and owner_node else context.access if class_context else None
        blocks=[];parent=node.parent
        while parent:
            if parent.type=='compound_statement':blocks.append(self.span(parent.start_byte,parent.end_byte))
            parent=parent.parent
        scope_reviews=[s for s in getattr(self,'untrusted_scope_regions',[])if s['start_byte']<=node.start_byte<s['end_byte']]+list(context.scope_reviews)
        class_record=None
        if class_context:
            class_record=next((o for o in reversed(self.occurrences)if o['entity_id']==context.scopes[last_class]['entity_id']and o['path']==self.path and o['start_byte']<=node.start_byte<o['end_byte']),None)
        visible_aliases=[{k:o[k]for k in ('entity_id','declaration_occurrence_id','name','target_type','signature_range','preprocessor_conditions')}
            for o in self.occurrences if o['kind']=='alias'and o['path']==self.path and o['start_byte']<node.start_byte and
            [s['identity']for s in o['scope_chain']]==[s['identity']for s in context.scopes]]if class_context else[]
        item={'source_range':self.span(node.start_byte,node.end_byte),'syntax_form':node.type,
            'context_source_ref':self.identity_location(node.start_byte,node.end_byte),
            'syntax_status':'contains_parse_error'if node.has_error else'parsed',
            'scope_kind':'local_class'if class_context and functions else'class'if class_context else'block'if functions and len(blocks)>1 else'function'if functions else'namespace',
            'scope_chain':[dict(s)for s in context.scopes],
            'enclosing_function_ref':context.scopes[last_function]['entity_id']if functions else None,
            'lexical_blocks':blocks,'lexical_access':access,
            'preprocessor_conditions':self.conditions(node.start_byte)+list(context.binding_conditions),
            'namespace_choices':dict(context.namespace_choices),'scope_reviews':scope_reviews,
            'template_environment':list(context.templates),'target_expression':target_text,
            'terminal_name':self.text(terminal),'namespace_directive':bool(re.match(r'using\s+namespace\b',self.text(node))),
            'class_owner':None if class_record is None else {k:class_record[k]for k in ('entity_id','name','qualified_name','signature_range','bases')},
            'visible_class_aliases':visible_aliases}
        self.current_using_contexts.append(item)
        return item

    def alias(self, node, context):
        using_context=self.capture_using_context(node,context)if node.type in {'using_declaration','alias_declaration'}else None
        if node.type=='using_declaration':return
        if context.in_function and not any(s["kind"] in TYPE_NODES for s in context.scopes[context_function_boundary(context):]):
            return
        if node.type == "type_definition":
            type_node = node.child_by_field_name("type")
            if type_node and type_node.type in TYPE_NODES | {"enum_specifier"}:
                self.walk(type_node, context)
            for decl in children_field(node, "declarator"):
                name_node = self.declarator_name(decl)
                if name_node:
                    name = self.text(name_node)
                    self.record(node, "typedef", name, context, identity=canonical(name),
                                target_type=self.text(type_node), declarator=self.text(decl),
                                declarator_range=self.span(decl.start_byte, decl.end_byte))
            return
        name_node = node.child_by_field_name("name")
        name = self.text(name_node)
        occurrence=self.record(node, "namespace_alias" if node.type == "namespace_alias_definition" else "alias", name, context,
                        identity=canonical(name), target_type=self.text(node.child_by_field_name("type")) or
                        "".join(self.text(c) for c in node.named_children if c != name_node))
        if using_context is not None:
            using_context['alias_entity_ref']=occurrence['entity_id'];using_context['alias_occurrence_ref']=occurrence['declaration_occurrence_id']
            using_context['scope_review_required']=occurrence.get('scope_review_required',False)

    def declarator_name(self, node):
        if node is None:
            return None
        if node.type in NAME_NODES:
            return node
        child = node.child_by_field_name("declarator")
        if child:
            return self.declarator_name(child)
        for child in node.named_children:
            if child.type not in {"parameter_list", "attribute_specifier", "type_qualifier", "argument_list"}:
                found = self.declarator_name(child)
                if found:
                    return found
        return None

    def function_declarator(self, node):
        if node is None:
            return None
        if node.type == "operator_cast":
            return next((n for n in node.named_children if n.type == "abstract_function_declarator"), None)
        if node.type == "function_declarator":
            inner = node.child_by_field_name("declarator")
            if inner and inner.type == "parenthesized_declarator":
                return None  # pointer/reference-to-function object, not a callable declaration
            return node
        inner = node.child_by_field_name("declarator")
        if inner is None:
            inner = next((c for c in node.named_children if c.type.endswith("declarator") or c.type in NAME_NODES), None)
        return self.function_declarator(inner) if inner else None

    def parameter(self, node):
        if node.type in {"variadic_parameter", "..."}:
            return {"raw": self.text(node), "name": None, "type": "...", "default": None,
                    **self.span(node.start_byte, node.end_byte)}
        declarator = node.child_by_field_name("declarator")
        name_node = self.declarator_name(declarator)
        default = node.child_by_field_name("default_value")
        end = default.start_byte if default else node.end_byte
        type_spelling = self.semantic_text(node.start_byte,end)
        if name_node:
            type_spelling = self.semantic_text(node.start_byte,name_node.start_byte) + self.semantic_text(name_node.end_byte,end)
        type_spelling = type_spelling.rstrip().rstrip("=").rstrip()
        return {"raw": self.text(node), "name": self.text(name_node) or None, "type": type_spelling,
                "default": self.text(default) or None, **self.span(node.start_byte, node.end_byte)}

    def declaration(self, node, context):
        type_node = node.child_by_field_name("type")
        if type_node and type_node.type in TYPE_NODES | {"enum_specifier"}:
            self.walk(type_node, context)
        declarators = children_field(node, "declarator")
        for decl in declarators:
            name_node = self.declarator_name(decl)
            if not name_node:
                self.diagnostic("unidentified_declarator", decl, message="Declaration declarator has no recoverable name")
                continue
            if self.text(name_node) in {"constexpr", "consteval", "if", "else", "return", "typename", "const", "volatile", "static"}:
                self.diagnostic("invalid_identifier_from_error_recovery", node,
                                message="Parser recovery produced a C++ keyword as a declaration name; not accepted as an API")
                continue
            function = self.function_declarator(decl)
            is_local = context.in_function and not any(s["kind"] in TYPE_NODES for s in context.scopes[context_function_boundary(context):])
            if not function:
                if is_local:
                    continue
                name = self.text(name_node)
                bitfield = None
                # The clause is a sibling of its declarator, not its default.
                for index,child in enumerate(node.children):
                    if child==decl:
                        for sibling in node.children[index+1:]:
                            if sibling.type in {',',';'}:break
                            if sibling.type=='bitfield_clause':bitfield=sibling;break
                            if sibling.type=='ERROR':
                                bitfield=next((n for n in descendants(sibling)if n.type=='bitfield_clause'),None)
                                if bitfield:break
                        break
                bitfield_model=None
                if bitfield:
                    origin=getattr(self.semantic_reader,'origin_span',self.span)(bitfield.start_byte,bitfield.start_byte+1)
                    bitfield_model=self.bitfield_source_models.get(origin['start_byte'])
                    if not bitfield_model:
                        self.diagnostic('bitfield_declaration_unmatched',bitfield,
                            message='Recovered bitfield has no independently verified physical source entry')
                        continue
                    name=bitfield_model['name']
                raw = self.text(node)
                kind = "member" if any(s["kind"] in TYPE_NODES for s in context.scopes) else "variable"
                if context.template_prefix_start is not None:
                    kind = "variable_template"
                elif re.search(r"\b(?:constexpr|const)\b", raw):
                    kind = "constant" if kind == "variable" else "member_constant"
                default = decl.child_by_field_name("value")
                if not default:
                    # field_declaration stores initializer alongside each declarator.
                    for index, child in enumerate(node.children):
                        if child == decl:
                            for sibling in node.children[index + 1:]:
                                if sibling.type in {",", ";"}:
                                    break
                                if sibling.is_named and sibling.type not in {"comment","bitfield_clause"}:
                                    default = sibling
                                    break
                bitfield_fields={'is_bitfield':True,'bitfield_source_id':bitfield_model['bitfield_source_id'],
                    'display_name':name or '<anonymous bitfield '+bitfield_model['bitfield_source_id'].split(':')[-1]+'>',
                    'bit_width':bitfield_model['bit_width'],'bit_width_range':self.span(bitfield.named_children[0].start_byte,bitfield.named_children[0].end_byte),
                    'colon_range':self.span(bitfield.start_byte,bitfield.start_byte+1)} if bitfield_model else {}
                type_components=[c for c in node.named_children if c==type_node or c.type=='type_qualifier'and self.text(c)in {'const','volatile','__restrict','__restrict__'}]
                value_node=decl.child_by_field_name('value')
                type_end=value_node.start_byte if value_node else decl.end_byte
                abstract_declarator=(self.semantic_text(decl.start_byte,name_node.start_byte)+
                    self.semantic_text(name_node.end_byte,type_end)).rstrip().rstrip('=').rstrip()
                complete_type=' '.join(self.text(c)for c in sorted(type_components,key=lambda c:c.start_byte))
                if abstract_declarator.strip():complete_type+=' '+abstract_declarator.strip()
                self.record(node, kind, name, context, identity=bitfield_model['bitfield_source_id'] if bitfield_model and name is None else canonical(name, self.substitutions(context)),
                            declared_type=self.text(type_node), declarator=self.text(decl), initializer=self.text(default) or None,
                            declared_type_spelling=complete_type.strip(),
                            declared_type_role='declaration_type_specifier',
                            qualifiers=self.qualifiers(node), declarator_range=self.span(decl.start_byte, decl.end_byte),
                            **bitfield_fields,
                            namespace_const_internal=kind == "constant" and not context.templates and
                            not any(s["kind"] in TYPE_NODES for s in context.scopes) and
                            not any(q in {"extern", "inline", "volatile"} for q in self.qualifiers(node)) and
                            (not any(n.type == "pointer_declarator" for n in descendants(decl)) or
                             any(n.type == "pointer_declarator" and any(c.type == "type_qualifier" and self.text(c) == "const" for c in n.named_children)
                                 for n in descendants(decl))))
                continue
            if is_local:
                # Function-scope extern declarations are not independently callable local functions.
                self.diagnostic("function_scope_declaration_pending", node,
                                message="Function-scope callable declaration needs namespace-target reconciliation")
                continue
            lexical_context = context
            if context.friend:
                last_class = next((s for s in reversed(context.scopes) if s["kind"] in TYPE_NODES), None)
                namespace_scopes = tuple(s for s in context.scopes if s["kind"] == "namespace")
                own_templates = context.templates[last_class.get("template_depth", 0):] if last_class else context.templates
                context = replace(context, scopes=namespace_scopes, templates=own_templates, access="public")
            body = node.child_by_field_name("body")
            name = self.text(name_node)
            if name_node.type == "operator_cast":
                name = "operator " + self.text(name_node.child_by_field_name("type"))
            last_type = next((s for s in reversed(context.scopes) if s["kind"] in TYPE_NODES), None)
            bare_class_name = last_type["name"].split("<")[0].split("::")[-1] if last_type else None
            simple_name = name.split("::")[-1]
            kind = "constructor" if bare_class_name == simple_name else "destructor" if simple_name.startswith("~") else \
                   "operator" if "operator" in name else "method" if last_type else "function"
            params = function.child_by_field_name("parameters")
            parameters = [self.parameter(c) for c in params.children if c.is_named and c.type != "comment" or c.type == "..."] if params else []
            parameters=[dict(parameter,index=index)for index,parameter in enumerate(parameters)]
            qualifiers = [self.text(c) for c in function.named_children if c.type in
                          {"type_qualifier", "ref_qualifier", "noexcept", "throw_specifier", "requires_clause", "virtual_specifier"}]
            trailing = next((c for c in function.named_children if c.type == "trailing_return_type"), None)
            requires = [self.text(c) for c in node.named_children if c.type == "requires_clause"]
            requires.extend(x for t in context.templates for x in t["requires"])
            # Return declarator wrappers contain pointer/ref components outside function_declarator.
            return_prefix = self.semantic_text(decl.start_byte,name_node.start_byte)
            return_cv = [self.text(c) for c in node.named_children if c.type == "type_qualifier" and self.text(c) in {"const", "volatile", "__restrict", "__restrict__"}]
            declaration_specifiers=[q for q in self.qualifiers(node)if q not in {"const","volatile","__restrict","__restrict__"}]
            return_type = " ".join(return_cv + [self.text(type_node)]) + (" " + return_prefix if return_prefix.strip() else "")
            if trailing:
                return_type = self.text(trailing).removeprefix("->").strip()
            substitution = self.substitutions(context)
            identity = {"parameters": [canonical(p["type"], substitution) for p in parameters],
                        "cv_ref": [canonical(q, substitution) for q in qualifiers if q in {"const", "volatile", "&", "&&"}],
                        "requires": [canonical(r, substitution) for r in requires],
                        "dependent_return": canonical(return_type, substitution) if context.template_prefix_start is not None else None}
            occurrence = self.record(node, kind, name, context,
                                     signature_end=body.start_byte if body else node.end_byte,
                                     identity=identity, parameters=parameters, return_type=return_type.strip() or None,
                                     trailing_return_type=self.text(trailing) or None, qualifiers=qualifiers + declaration_specifiers,
                                     return_cv_qualifiers=return_cv,
                                     requires=requires, declarator_range=self.span(decl.start_byte, decl.end_byte),
                                     body_range=self.span(body.start_byte, body.end_byte) if body else None,
                                     lexical_scope_chain=list(lexical_context.scopes) if lexical_context.friend else None,
                                     friend_injection="innermost_enclosing_namespace" if lexical_context.friend else None)
            if body:
                self.function_bodies.append((body.start_byte, body.end_byte))
                nested = replace(context, scopes=context.scopes + (self.scope_for(occurrence),),
                                 in_function=True, template_prefix_start=None, declaration_prefix_start=None, friend=False)
                self.walk(body, nested)
            context = lexical_context
        if not declarators and not (type_node and type_node.type in TYPE_NODES | {"enum_specifier"}):
            self.diagnostic("declaration_without_declarator", node,
                            message="Parser produced a declaration without a named declarator; may be macro-generated")

    def qualifiers(self, node):
        return [self.text(c) for c in node.named_children if c.type in
                {"storage_class_specifier", "type_qualifier", "function_specifier", "attribute_specifier",
                 "attribute_declaration", "ms_declspec_modifier", "explicit_function_specifier"}]

    def record_parse_errors(self, root, origin):
        for node in descendants(root, include_anonymous=True):
            if node.type == "ERROR" or node.is_missing:
                ancestors, current = [], node.parent
                while current:
                    ancestors.append(current.type)
                    current = current.parent
                inside_body = any(a <= node.start_byte and node.end_byte <= b for a, b in self.function_bodies)
                self.diagnostic("parse_error" if node.type == "ERROR" else "missing_syntax", node,
                                message=f"Tree-sitter {node.type} in {origin}", origin=origin,
                                in_function_body=inside_body, ancestor_types=ancestors[:8],
                                blocks=True)

    def pending_macros(self):
        # Preserve all uppercase call expressions at declaration scope as pending;
        # independent candidate reconciliation will refine classification.
        for node in descendants(self.tree.root_node):
            if node.type != "call_expression":
                continue
            function = node.child_by_field_name("function")
            name = self.text(function)
            if not re.fullmatch(r"[A-Z_][A-Z_0-9]*", name) or name in ANNOTATIONS:
                continue
            # #include operands and #if predicates are not declarations, even
            # when the parser represents a function-like macro as a call.
            # Keep their exact positions for dependency/condition extraction.
            ancestor = node.parent
            include = None
            while ancestor:
                if ancestor.type == 'preproc_include':
                    include = ancestor
                    break
                ancestor = ancestor.parent
            directive = next((r for r in self.pp_regions if r['start_byte'] <= node.start_byte and node.end_byte <= r['end_byte']), None)
            if include or directive:
                self.non_declaration_macro_uses.append({**self.span(node.start_byte,node.end_byte),
                    'name': name, 'expression': self.text(node),
                    'classification': 'include_operand' if include else 'preprocessor_condition',
                    'evidence': self.span(include.start_byte,include.end_byte) if include else directive})
                continue
            if any(a <= node.start_byte and node.end_byte <= b for a, b in self.function_bodies):
                continue
            self.diagnostic("declaration_macro_expansion_pending", node,
                            message="Declaration-scope uppercase invocation is not expanded; generated entities are pending",
                            macro_name=name, arguments=self.text(node.child_by_field_name("arguments")))

    def integrate_macros(self, diagnostic_start):
        result = self.macro_result
        for item in result["diagnostics"]:
            self.diagnostic(item["kind"], start=item["start_byte"], end=item.get("end_byte", item["start_byte"]),
                            message=item.get("message", "Declaration macro expansion remains unresolved"),
                            macro_details=item)
        integrated = []
        for expansion in result["expansions"]:
            containing = [(b - a, ctx) for a, b, ctx in self.context_intervals
                          if a <= expansion["start_byte"] and expansion["end_byte"] <= b]
            minimum = min((size for size, _ in containing), default=0)
            contexts = {}
            for size, ctx in containing:
                if size == minimum:
                    contexts[digest('ctx_', [[s['identity'] for s in ctx.scopes], ctx.binding_conditions])] = ctx
            for context in contexts.values() or [Context()]:
                integrated.append(self.integrate_macro_context(expansion, context, diagnostic_start))
        return integrated

    def integrate_macro_context(self, expansion, context, diagnostic_start):
        callsite_reviews=[{k:v for k,v in s.items()if k not in {'start_byte','end_byte'}}for s in getattr(self,'untrusted_scope_regions',[])
                          if s['start_byte']<=expansion['start_byte']<s['end_byte']]
        context = replace(context, template_prefix_start=None, declaration_prefix_start=None,
                          scope_reviews=context.scope_reviews+tuple(callsite_reviews))
        virtual_path = self.path + "#" + expansion["invocation_id"]
        virtual = Extractor(self.commit)
        segments=expansion.get('conditional_source_segments')
        def conditional_span(start,end):
            parts=_headers_module.map_virtual_range(segments,start,end)
            if not parts:
                segment=next((s for s in segments if s['virtual_start_byte']<=start<s['virtual_end_byte']),segments[-1])
                point=segment['physical_span']['start_byte']+min(max(start-segment['virtual_start_byte'],0),segment['virtual_end_byte']-segment['virtual_start_byte'])
                parts=[{'path':self.path,'start_byte':point,'end_byte':point,'mapping':'exact_source_boundary'}]
            span=self.span(min(p['start_byte']for p in parts),max(p['end_byte']for p in parts))
            span['source_segments']=parts
            span['range_kind']='exact_contiguous_source'if len(parts)==1 else'covering_discontiguous_source'
            return span
        if segments:
            def conditional_identity(start,end):
                location=conditional_span(start,end)
                reader=expansion.get('enclosing_semantic_reader')
                if hasattr(reader,'origin_span'):location=reader.origin_span(location['start_byte'],location['end_byte'])
                inherited=getattr(self,'source_identity_mapper',None)
                return inherited(location['start_byte'],location['end_byte'])if inherited else location
            virtual.source_identity_mapper=conditional_identity
        if context.scopes and context.scopes[-1]['kind']in TYPE_NODES:
            virtual.extract_class_fragment(virtual_path,expansion['virtual_source'].encode(),context)
        else:
            virtual.extract(virtual_path, expansion["virtual_source"].encode(), initial_context=context, expand_macros=False)
        invocation_span = self.span(expansion["start_byte"], expansion["end_byte"])
        inherited_conditions = self.conditions(expansion["start_byte"])

        def map_ranges(value):
            if isinstance(value, list):
                return [map_ranges(x) for x in value]
            if not isinstance(value, dict):
                return value
            out = {k: map_ranges(v) for k, v in value.items()}
            if value.get("path") == virtual_path and "start_byte" in value:
                out["virtual_range"] = {k: value[k] for k in ("path", "start_byte", "end_byte", "start_line", "end_line") if k in value}
                out.update(conditional_span(value['start_byte'],value['end_byte'])if segments else invocation_span)
            if segments and value.get('directive_path')==virtual_path and 'directive_line'in value:
                virtual_byte=virtual.line_starts[min(value['directive_line']-1,len(virtual.line_starts)-1)]
                mapped_line=conditional_span(virtual_byte,virtual_byte)
                out.update(directive_path=self.path,directive_line=mapped_line['start_line'])
            return out

        occurrence_map = {}
        variant_map = {}
        for index, occurrence in enumerate(virtual.occurrences):
            mapped = map_ranges(occurrence)
            old_id = mapped["declaration_occurrence_id"]
            mapped["declaration_occurrence_id"] = digest("occ_", [expansion["invocation_id"], index, old_id])
            occurrence_map[old_id] = mapped["declaration_occurrence_id"]
            mapped["preprocessor_conditions"] = inherited_conditions + mapped["preprocessor_conditions"]
            variant_id = digest("var_", [mapped["entity_id"], mapped["preprocessor_conditions"], mapped["attributes"]])
            variant_map[mapped["variant_id"]] = variant_id
            mapped["variant_id"] = variant_id
            mapped["macro_origin"] = {"invocation_id": expansion["invocation_id"], "expansion_ordinal": index,
                                      "definition": expansion["definition"], "bindings": expansion["bindings"],
                                      "invocation": invocation_span}
            mapped["signature_origin"] = "virtual_macro_expansion; physical_range_points_to_invocation"
            self.occurrences.append(mapped)
        for entity_id, entity in virtual.entities.items():
            mapped = dict(entity)
            mapped["declaration_occurrence_ids"] = [occurrence_map[x] for x in entity["declaration_occurrence_ids"]]
            mapped["variant_ids"] = list(dict.fromkeys(variant_map.get(x, x) for x in entity["variant_ids"]))
            if entity_id not in self.entities:
                self.entities[entity_id] = mapped
            else:
                self.entities[entity_id]["declaration_occurrence_ids"].extend(mapped["declaration_occurrence_ids"])
                self.entities[entity_id]["variant_ids"] = list(dict.fromkeys(self.entities[entity_id]["variant_ids"] + mapped["variant_ids"]))
        for using_context in virtual.files[-1].get('using_contexts',[])if virtual.files else []:
            mapped=map_ranges(using_context)
            mapped['preprocessor_conditions']=inherited_conditions+mapped.get('preprocessor_conditions',[])
            if mapped.get('alias_occurrence_ref')in occurrence_map:mapped['alias_occurrence_ref']=occurrence_map[mapped['alias_occurrence_ref']]
            self.current_using_contexts.append(mapped)
        for diagnostic in virtual.diagnostics:
            mapped = map_ranges(diagnostic)
            mapped["diagnostic_id"] = digest("diag_", [expansion["invocation_id"], diagnostic["diagnostic_id"]])
            mapped["macro_invocation_id"] = expansion["invocation_id"]
            mapped["preprocessor_conditions"] = inherited_conditions + mapped["preprocessor_conditions"]
            self.diagnostics.append(mapped)
        scope_pending=any(o.get('scope_review_required')for o in virtual.occurrences)
        successful = bool(virtual.occurrences) and not any(d["blocks_phase_1"] for d in virtual.diagnostics)and not scope_pending
        if not virtual.occurrences:
            self.diagnostic("macro_expansion_without_entities", start=expansion["start_byte"], end=expansion["end_byte"],
                            message="Declaration macro expanded but yielded no entities; classification remains pending",
                            macro_invocation_id=expansion["invocation_id"])
        if successful:
            for diagnostic in self.diagnostics[diagnostic_start:]:
                if diagnostic.get("macro_invocation_id"):
                    continue
                if expansion["start_byte"] <= diagnostic["start_byte"] and diagnostic["end_byte"] <= expansion["end_byte"]:
                    if diagnostic["category"] in {"declaration_macro_expansion_pending", "parse_error", "missing_syntax", "declaration_without_declarator"}:
                        diagnostic["blocks_phase_1"] = False
                        diagnostic["resolved_by_macro_invocation_id"] = expansion["invocation_id"]
                        diagnostic["resolution"] = "Exact span covered by successfully parsed source-mapped macro expansion"
        return {**{k:v for k,v in expansion.items()if k!='enclosing_semantic_reader'}, "declaration_occurrence_ids": list(occurrence_map.values()),
                "context_scope_ids": [s['identity'] for s in context.scopes],
                "context_conditions": list(context.binding_conditions),
                "scope_integrity":map_ranges(virtual.files[-1].get('scope_integrity',{}))if virtual.files else {},
                "scope_integrity_source_kind":"virtual_macro_expansion_with_invocation_mapping",
                "status": "expanded_and_parsed" if successful else "expanded_scope_review_pending"if scope_pending else"expanded_parse_pending"}

    def result(self):
        return {"schema_version": SCHEMA_VERSION, "commit": self.commit,
                "parser": {"engine": "tree-sitter", "language": "tree-sitter-cpp",
                           "branch_policy": "raw_source_all_preprocessor_branches",
                           "normalization_policy": "explicit_annotation_allowlist_equal_length_whitespace"},
                "completion": {"phase_1_passed": False,
                               "reason": "Requires independent candidate reconciliation and zero declaration/macro gaps"},
                "summary": {"files": len(self.files), "entities": len(self.entities), "occurrences": len(self.occurrences),
                            "diagnostics": len(self.diagnostics),
                            "blocking_diagnostics": sum(d["blocks_phase_1"] for d in self.diagnostics),
                            "scope_review_occurrences":sum(bool(o.get('scope_review_required'))for o in self.occurrences),
                            "scope_integrity_uncertainties":sum(len(f.get('scope_integrity',{}).get('uncertainties',[]))for f in self.files),
                            "using_sources":len(self.using_sources),"using_interfaces":len(self.using_interfaces),
                            "using_source_kinds":dict(Counter(s['kind']for s in self.using_sources)),
                            "using_interface_forms":dict(Counter(i['semantic_form']for i in self.using_interfaces)),
                            "syntax_diagnostics_in_function_bodies": sum(d["category"] in {"parse_error", "missing_syntax"} and d.get("in_function_body", False) for d in self.diagnostics),
                            "syntax_diagnostics_outside_function_bodies": sum(d["category"] in {"parse_error", "missing_syntax"} and not d.get("in_function_body", False) for d in self.diagnostics),
                            "other_blocking_diagnostics": sum(d["blocks_phase_1"] and d["category"] not in {"parse_error", "missing_syntax"} for d in self.diagnostics),
                            "kinds": dict(sorted(Counter(o["kind"] for o in self.occurrences).items())),
                            "diagnostic_categories": dict(sorted(Counter(d["category"] for d in self.diagnostics).items()))},
                "entities": list(self.entities.values()), "occurrences": self.occurrences,
                "using_sources":self.using_sources,"using_interfaces":self.using_interfaces,
                "diagnostics": self.diagnostics, "files": self.files}


def context_function_boundary(context):
    return max((i + 1 for i, scope in enumerate(context.scopes)
                if scope["kind"] in {"function", "method", "constructor", "destructor", "operator"}), default=0)


@lru_cache(maxsize=1)
def config_macro_definitions():
    root = Path(__file__).resolve().parents[1] / "snapshot"
    result = []
    for path in ("include/cute/config.hpp", "include/cutlass/detail/helper_macros.hpp", "include/cutlass/cutlass.h"):
        candidate = root / path
        if candidate.exists():
            source = candidate.read_bytes()
            found = _macro_module.definitions(path, source)
            condition_reader = object.__new__(Extractor)
            condition_reader.path, condition_reader.source = path, source
            condition_reader.lines = source.splitlines(keepends=True)
            condition_reader.line_starts = [0] + [m.end() for m in re.finditer(b"\n", source)]
            condition_reader.condition_by_line, _ = condition_reader.preprocessor_conditions()
            for definition in found:
                definition.preprocessor_conditions = condition_reader.conditions(definition.start_byte)
            result.extend(found)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("data/declarations.json"))
    parser.add_argument("--limit", type=int, help="Diagnostic subset only; never a phase-completion run")
    parser.add_argument("--pretty", action="store_true", help="Indent the large artifact; default is compact lossless JSON")
    args = parser.parse_args()
    project = args.project.resolve()
    generator_paths = [Path(__file__).resolve(), Path(__file__).with_name("declaration_projection.py").resolve(),
                       Path(__file__).with_name("macro_expansion.py").resolve(),Path(__file__).with_name("namespace_bindings.py").resolve(),Path(__file__).with_name("declaration_syntax.py").resolve(),Path(__file__).with_name("declaration_bitfields.py").resolve(),Path(__file__).with_name("declaration_headers.py").resolve(),Path(__file__).with_name('declaration_scope_integrity.py').resolve()]
    start_hashes = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest() for path in generator_paths}
    scope = json.loads((project / "data/scope.json").read_text())
    extractor = Extractor(scope["commit"])
    entries = scope["files"][:args.limit] if args.limit else scope["files"]
    for index, entry in enumerate(entries, 1):
        source = (project / "snapshot" / entry["path"]).read_bytes()
        if hashlib.sha256(source).hexdigest() != entry["sha256"]:
            raise SystemExit(f"Snapshot hash mismatch: {entry['path']}")
        extractor.extract(entry["path"], source)
        if index % 50 == 0:
            print(f"parsed {index}/{len(entries)} files; occurrences={len(extractor.occurrences)} diagnostics={len(extractor.diagnostics)}", flush=True)
    result = extractor.result()
    result["scope_file_count"] = scope["file_count"]
    result["is_subset"] = len(entries) != scope["file_count"]
    end_hashes = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest() for path in generator_paths}
    changed_sources = [path for path in start_hashes if start_hashes[path] != end_hashes[path]]
    result["generator_provenance"] = {"source_hashes_at_start": start_hashes, "source_hashes_at_end": end_hashes,
                                      "sources_changed_during_run": changed_sources,
                                      "data_matches_end_source_files": not bool(changed_sources),
                                      "dependency_versions": {name: package_version(name) for name in ("tree-sitter", "tree-sitter-cpp")}}
    output = args.output if args.output.is_absolute() else project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=".declarations-", suffix=".json.tmp", delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            json.dump(result, stream, ensure_ascii=False, indent=2 if args.pretty else None,
                      separators=None if args.pretty else (",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}; phase_1_passed=false")


if __name__ == "__main__":
    main()

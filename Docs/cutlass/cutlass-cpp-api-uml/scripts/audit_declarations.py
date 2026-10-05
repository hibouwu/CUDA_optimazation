#!/usr/bin/env python3
"""Mechanical identity/provenance audit, not a claim of API completeness.

The lexical candidate reconciliation is a separate independent denominator.
This check catches duplicate identities, broken back-references, stale generator
provenance and signatures that no longer match their physical/virtual source.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from reconcile_candidates import top_items


def alignment_issues(occurrence, source):
    """Mechanical ownership/byte checks; semantic macro proof is independent."""
    issues=[]
    for attribute in occurrence.get('alignment_specifiers',[]):
        reasons=[];a,b=attribute['start_byte'],attribute['end_byte']
        signature=occurrence['signature_range']
        if not signature['start_byte']<=a<b<=signature['end_byte']:reasons.append('outside_signature')
        if attribute.get('path')!=occurrence['path']or source[a:b].decode('utf-8','replace')!=attribute.get('semantic_spelling'):reasons.append('source_spelling_mismatch')
        if attribute.get('owner_occurrence_id')!=occurrence['declaration_occurrence_id']or attribute.get('owner_entity_id')!=occurrence['entity_id']:reasons.append('owner_backreference_mismatch')
        hint=attribute.get('owner_hint')
        if hint:
            expected=hint['declaration_span'];actual=occurrence.get('syntax_node_range',{})
            if any(expected.get(k)!=actual.get(k)for k in ('path','start_byte','end_byte')):reasons.append('owner_syntax_range_mismatch')
            if hint['kind']=='field'and occurrence['kind']!='member':reasons.append('field_attribute_on_non_member')
            if hint['kind']=='type'and occurrence['kind']not in {'struct_specifier','class_specifier','union_specifier'}:reasons.append('type_attribute_on_non_type')
        expression=attribute.get('expression_span')
        if expression and (not a<=expression['start_byte']<expression['end_byte']<=b or source[expression['start_byte']:expression['end_byte']].decode('utf-8','replace')!=attribute['alignment_expression']):reasons.append('expression_source_mismatch')
        if reasons:issues.append({'kind':'alignment_source_owner_mismatch','id':occurrence['declaration_occurrence_id'],'attribute_id':attribute['attribute_id'],'reasons':reasons})
    return issues


def audit(project, declarations):
    scope = json.loads((project/'data/scope.json').read_text())
    source = {f['path']:(project/'snapshot'/f['path']).read_bytes() for f in scope['files']}
    issues = []
    for entry in scope['files']:
        if hashlib.sha256(source[entry['path']]).hexdigest()!=entry['sha256']:
            issues.append({'kind':'snapshot_manifest_mismatch','path':entry['path']})
    entities, occurrence_ids = {}, set()
    entity_occurrences, entity_variants = defaultdict(set), defaultdict(set)
    counts, files, metadata = Counter(), {}, {}
    virtual_occurrences = []
    for key,value,item in top_items(declarations,arrays=('entities','occurrences','files','diagnostics')):
        if not item:
            metadata[key] = value
            continue
        if key == 'entities':
            identifier = value['entity_id']
            if identifier in entities: issues.append({'kind':'duplicate_entity','id':identifier})
            entities[identifier] = value
        elif key == 'occurrences':
            identifier = value['declaration_occurrence_id']
            if identifier in occurrence_ids: issues.append({'kind':'duplicate_occurrence','id':identifier})
            occurrence_ids.add(identifier)
            entity_occurrences[value['entity_id']].add(identifier)
            entity_variants[value['entity_id']].add(value['variant_id'])
            path = value['path']; counts[path] += 1
            span = value['signature_range']; a,b = span['start_byte'],span['end_byte']
            if path not in source or not 0 <= a <= b <= len(source[path]):
                issues.append({'kind':'invalid_signature_range','id':identifier,'path':path})
                continue
            if value.get('macro_origin'):
                virtual_occurrences.append((identifier,path,value['macro_origin']['invocation_id'],
                    span.get('virtual_range'),value['raw_signature']))
            elif source[path][a:b].decode('utf-8','replace').rstrip() != value['raw_signature']:
                issues.append({'kind':'physical_signature_mismatch','id':identifier,'path':path,'span':[a,b]})
            issues.extend(alignment_issues(value,source[path]))
            # Only semantic fields, never the explicit parser-projection trace.
            semantic = {k:value.get(k) for k in ('name','qualified_name','return_type','trailing_return_type',
                'declared_type','declared_type_spelling','target_type','initializer','parameters','template_parameters')}
            def no_trace(v):
                if isinstance(v,dict):return {k:no_trace(x)for k,x in v.items()if k not in {'raw','expanded_raw','constraint_projection_range','virtual_range'}}
                if isinstance(v,list):return [no_trace(x)for x in v]
                return v
            if '__codex_parser_' in json.dumps(no_trace(semantic)):
                issues.append({'kind':'parser_placeholder_in_semantic_field','id':identifier,'path':path})
        elif key == 'files':
            if value['path'] in files: issues.append({'kind':'duplicate_file_record','path':value['path']})
            files[value['path']] = value
        elif key == 'diagnostics':
            if value.get('blocks_phase_1'): counts['blocking_diagnostics'] += 1
    for identifier,entity in entities.items():
        declared = entity['declaration_occurrence_ids']
        if len(set(declared)) != len(declared): issues.append({'kind':'duplicate_entity_occurrence_reference','id':identifier})
        if set(declared) != entity_occurrences[identifier]: issues.append({'kind':'entity_occurrence_backreference_mismatch','id':identifier})
        if set(entity['variant_ids']) != entity_variants[identifier]: issues.append({'kind':'entity_variant_backreference_mismatch','id':identifier})
        if '__codex_parser_' in json.dumps(entity['identity_key']): issues.append({'kind':'parser_placeholder_in_identity','id':identifier})
    for identifier in entity_occurrences.keys()-entities.keys(): issues.append({'kind':'unknown_entity','id':identifier})
    if set(files) != set(source): issues.append({'kind':'file_denominator_mismatch','missing':sorted(set(source)-set(files)),'extra':sorted(set(files)-set(source))})
    expansions = {}
    for path,record in files.items():
        if path in source and record['sha256'] != hashlib.sha256(source[path]).hexdigest(): issues.append({'kind':'file_hash_mismatch','path':path})
        if record['occurrence_count'] != counts[path]: issues.append({'kind':'file_occurrence_count_mismatch','path':path})
        for expansion in record.get('macro_expansions',[]): expansions[(path,expansion['invocation_id'])] = expansion['virtual_source'].encode()
    for identifier,path,invocation,span,raw in virtual_occurrences:
        virtual = expansions.get((path,invocation))
        if virtual is None or span is None or virtual[span['start_byte']:span['end_byte']].decode('utf-8','replace').rstrip() != raw:
            issues.append({'kind':'virtual_signature_mismatch','id':identifier,'path':path,'invocation_id':invocation})
    provenance = metadata.get('generator_provenance',{})
    expected_generators = {'scripts/'+name for name in ('extract_declarations.py','declaration_projection.py','declaration_syntax.py','macro_expansion.py','namespace_bindings.py','declaration_bitfields.py','declaration_headers.py','declaration_scope_integrity.py')}
    if set(provenance.get('source_hashes_at_end',{})) != expected_generators:
        issues.append({'kind':'generator_provenance_denominator_mismatch'})
    if metadata.get('commit') != scope.get('commit'):
        issues.append({'kind':'commit_mismatch'})
    if provenance.get('sources_changed_during_run') or not provenance.get('data_matches_end_source_files'):
        issues.append({'kind':'generator_changed_during_run'})
    for path,expected in provenance.get('source_hashes_at_end',{}).items():
        if not (project/path).is_file() or hashlib.sha256((project/path).read_bytes()).hexdigest() != expected: issues.append({'kind':'generator_changed_after_run','path':path})
    summary = metadata.get('summary',{})
    for field,actual in [('entities',len(entities)),('occurrences',sum(v for k,v in counts.items()if k in source)),('files',len(files)),('blocking_diagnostics',counts['blocking_diagnostics'])]:
        if summary.get(field) != actual: issues.append({'kind':'summary_mismatch','field':field,'actual':actual,'reported':summary.get(field)})
    return {'mechanical_audit_passed':not issues,'phase_1_passed':False,
            'coverage_claim':'Identity and source-position consistency only; candidate reconciliation and semantic review remain separate',
            'commit':metadata.get('commit'),'files':len(files),'entities':len(entities),'occurrences':len(occurrence_ids),
            'blocking_diagnostics':counts['blocking_diagnostics'],'issue_count':len(issues),'issues':issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--declarations',type=Path,default=Path('data/declarations.json'))
    parser.add_argument('--output',type=Path,default=Path('data/declaration-mechanical-checks.json'))
    args = parser.parse_args(); project = args.project.resolve()
    result = audit(project,project/args.declarations)
    (project/args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='issues'},ensure_ascii=False))
    raise SystemExit(0 if result['mechanical_audit_passed'] else 1)


if __name__ == '__main__': main()

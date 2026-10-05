#!/usr/bin/env python3
"""Join the reviewed two-header draft to the current canonical source ledger."""
import importlib.util
import copy
from collections import Counter
import json
from pathlib import Path

from build_atlas import ROOT,enrich_nodes,dump,source_href
from reconcile_candidates import file_sha,condition_form,SourceProof,condition_matches
from audit_declarations import alignment_issues


def verify_canonical_contract(authored,nodes,physical):
    """Compare exact source selectors and metadata, not just total counts."""
    def identity(span,kind,name,qualified):
        return (span['path'],span['start_byte'],span['end_byte'],kind,name,qualified)
    expected=[]
    for n in authored['nodes']:
        s=n.get('source_selector')
        if s and s['path']in authored['scope']['paths']:
            expected.append(identity(s['signature_range'],s['kind'],s['name'],s['qualified_name']))
        for s in n.get('source_selectors',[]):
            expected.append(identity(s['signature_range'],s['kind'],s['name'],'cute'))
    actual=[identity(o['signature_range'],o['kind'],o['name'],o['qualified_name'])for o in physical]
    if Counter(expected)!=Counter(actual)or len({o['declaration_occurrence_id']for o in physical})!=len(physical):
        raise ValueError('Physical declaration multiset, names or occurrence identities disagree with independent selectors')
    namespaces=[o for o in physical if o['kind']=='namespace']
    if len({o['entity_id']for o in namespaces})!=1 or nodes['alignment.namespace.cute'].get('full_name')!='cute':
        raise ValueError('Namespace identity or qualified name differs between the two files')
    types=[n for key,n in nodes.items()if key.startswith('alignment.type.aligned_')]
    if len(types)!=10 or len({n['entity_id']for n in types})!=10:
        raise ValueError('Primary template and nine specializations must have independent identities')
    originals={n['id']:n for n in authored['nodes']};occurrences={o['declaration_occurrence_id']:o for o in physical}
    for key,n in nodes.items():
        original=originals[key]
        if original.get('source_selector'):
            if n.get('full_name')!=original['qualified_name']:raise ValueError('Canonical qualified name differs: '+key)
            for field in ('return_type','attributes','qualifiers'):
                if field in original and n.get(field)!=original[field]:raise ValueError('Canonical declaration metadata differs: '+key+':'+field)
            if 'parameters'in original:
                for fields in [('name','type','default','raw')]:
                    if [tuple(p.get(f)for f in fields)for p in n.get('parameters',[])]!=[tuple(p.get(f)for f in fields)for p in original['parameters']]:
                        raise ValueError('Function parameter contract differs: '+key)
            if 'template_parameters'in original:
                parameters=[p for header in n.get('template_parameters',[])for p in header['parameters']]
                expected_parameters=original['template_parameters']
                fields=('name','raw','default')
                if [tuple(p.get(f)for f in fields)for p in parameters]!=[tuple(p.get(f)for f in fields)for p in expected_parameters]:
                    raise ValueError('Template parameter/default contract differs: '+key)
        if original.get('semantic_class')!='requested_alignment_attribute':continue
        owner=nodes[original['owner_ref']];span=original['source_range']
        matched=[a for a in owner.get('alignment_specifiers',[])if (a['path'],a['start_byte'],a['end_byte'])==(span['path'],span['start_byte'],span['end_byte'])]
        if len(matched)!=1:raise ValueError('Missing or duplicate canonical attribute owner: '+key)
        attribute=matched[0];occurrence=occurrences[owner['declaration_occurrence_id']]
        raw=(ROOT/'snapshot'/span['path']).read_bytes()
        if alignment_issues(occurrence,raw):raise ValueError('Invalid canonical alignment source or owner: '+key)
        if attribute.get('alignment_unit')!='bytes'or attribute.get('semantic_spelling')!=span['raw']or attribute.get('alignment_expression')!=original['alignment_expression']:
            raise ValueError('Alignment spelling, expression or unit differs: '+key)
        expression=attribute.get('expression_span',{});expected_expression=original['expression_range']
        if any(expression.get(f)!=expected_expression.get(f)for f in ('path','start_byte','end_byte')):
            raise ValueError('Alignment expression source range differs: '+key)
        variants=attribute.get('conditional_expansions',[])
        expected_variants={condition_form(v['condition']):v for v in original['expansion_variants']}
        if len(variants)!=2:raise ValueError('Both alignment definition branches are required: '+key)
        seen=[]
        for variant in variants:
            conditions=variant.get('conditions',[])
            if len(conditions)!=1:raise ValueError('Wrong alignment definition condition: '+key)
            condition=condition_form(conditions[0]['expression']);seen.append(condition)
            if condition not in expected_variants:raise ValueError('Unexpected alignment definition branch: '+key)
            expected_variant=expected_variants[condition];definition=variant['definition'];macro=originals[expected_variant['definition_ref']]
            if variant.get('expanded_spelling')!=expected_variant['expanded_spelling']or definition.get('body')!=macro['replacement']or definition.get('name')!='CUTE_ALIGNAS'or definition.get('parameters')!=['n']:
                raise ValueError('Alignment macro replacement or expansion differs: '+key)
            declaration=macro['signature_range'];provider=(ROOT/'snapshot'/declaration['path']).read_bytes()
            if definition.get('path')!=declaration['path']or definition.get('start_byte')!=declaration['start_byte']or provider[definition['start_byte']:definition['end_byte']]!=provider[declaration['start_byte']:declaration['end_byte']].rstrip(b'\r\n')or variant.get('definition_source_sha256')!=file_sha(ROOT/'snapshot'/declaration['path']):
                raise ValueError('Alignment definition provenance differs: '+key)
            proofs=SourceProof(declaration['path'],provider).conditions_at(definition['start_byte'])
            if len(proofs)!=1 or not condition_matches(conditions[0],proofs[0]):raise ValueError('Alignment condition location differs: '+key)
        if set(seen)!=set(expected_variants):raise ValueError('Alignment definition branch duplicated: '+key)


def main():
    draft=ROOT/'data/module-drafts/cute_alignment/relations.json'
    checker=ROOT/'data/module-drafts/cute_alignment/check_source.py'
    spec=importlib.util.spec_from_file_location('alignment_source_check',checker)
    review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
    authored=json.loads(draft.read_text());source_check=review.verify(authored);data=copy.deepcopy(authored)
    inputs={str(p.relative_to(ROOT)):file_sha(p)for p in (draft,checker,ROOT/'data/declarations.json',Path(__file__),
        ROOT/'scripts/build_atlas.py',ROOT/'scripts/reconcile_candidates.py',ROOT/'scripts/audit_declarations.py',ROOT/'scripts/scan_candidates.py')}
    nodes={n['id']:n for n in data['nodes']};issues=[]
    records=enrich_nodes(nodes,json.loads((ROOT/'data/scope.json').read_text()),issues)
    if issues:raise ValueError(issues)
    for n in nodes.values():
        if n.get('source_selector'):
            if n.get('manual_declaration')or n.get('declaration_status')!='linked_to_source_ledger':
                raise ValueError('Unresolved current canonical declaration: '+n['id'])
    scope=set(data['scope']['paths'])
    physical=[o for p in scope for o in records[p]]
    if len(physical)!=16 or any(o.get('scope_review_required')or o.get('parse_status')!='parsed'for o in physical):
        raise ValueError('Two-header declaration denominator differs from the independent 16-source inventory')
    verify_canonical_contract(authored,nodes,physical)
    resolved={n['declaration_occurrence_id']for n in nodes.values()if n.get('declaration_occurrence_id')}
    namespace=nodes['alignment.namespace.cute']
    namespace['source_declaration_occurrences']=[{
        'declaration_occurrence_id':o['declaration_occurrence_id'],'entity_id':o['entity_id'],
        'signature':o['raw_signature'],'signature_range':o['signature_range'],
        'path':o['path'],'start_line':o['signature_range']['start_line'],
        'source_url':source_href(o['path'],o['signature_range']['start_line'])}
        for o in physical if o['kind']=='namespace']
    resolved.update(o['declaration_occurrence_id']for o in namespace['source_declaration_occurrences'])
    if not {o['declaration_occurrence_id']for o in physical}<=resolved:
        raise ValueError('A physical declaration has no current atlas endpoint')
    for n in nodes.values():
        if n.get('semantic_class')!='requested_alignment_attribute':continue
        owner=nodes[n['owner_ref']];span=n['source_range']
        matched=[a for a in owner.get('alignment_specifiers',[])if (a['path'],a['start_byte'],a['end_byte'])==(span['path'],span['start_byte'],span['end_byte'])]
        if len(matched)!=1 or matched[0]['alignment_expression']!=n['alignment_expression']:
            raise ValueError('Authored alignment differs from canonical owner: '+n['id'])
        n['canonical_alignment_attribute_id']=matched[0]['attribute_id']
        n['owner_entity_id']=owner['entity_id'];n['owner_occurrence_id']=owner['declaration_occurrence_id']
    for edge in data['edges']:
        if edge['relation']=='expands_to'and nodes[edge['source']].get('semantic_class')=='requested_alignment_attribute':
            edge['macro_bindings']={'n':nodes[edge['source']]['alignment_expression']}
    for contract in data['contracts']:
        if contract['id']=='alignment.contract.types':contract.update(kind='type_representation',events=[])
        elif contract['id']=='alignment.contract.pointer_predicate':contract['kind']='value_predicate'
    data['status']='source_and_canonical_identity_reviewed_pending_use_acceptance'
    data['canonical_integration']={'inputs':inputs,'independent_source_check':source_check,
        'physical_declarations_linked':16,'current_manual_overrides':0,
        'alignment_owners_linked':10,'global_completion_claimed':False}
    for issue in data['issues']:
        if issue['id']=='alignment.issue.adapter_and_identity':
            issue.update(status='closed',resolution='All 16 physical declarations and ten attribute owners match the reviewed current canonical ledger; manual identity overrides retired.')
    for path,expected in inputs.items():
        if file_sha(ROOT/path)!=expected:raise ValueError('Input changed during stage: '+path)
    target=ROOT/'data/modules/cute_alignment/relations.json';dump(target,data)
    dump(ROOT/'audits/cute-alignment-canonical-integration.json',data['canonical_integration'])
    print(json.dumps({'staged':str(target.relative_to(ROOT)),'physical_declarations':16,
        'nodes':len(data['nodes']),'edges':len(data['edges']),'manual_overrides':0,
        'api_coverage_passed':False,'browser_acceptance':False},ensure_ascii=False))


if __name__=='__main__':main()

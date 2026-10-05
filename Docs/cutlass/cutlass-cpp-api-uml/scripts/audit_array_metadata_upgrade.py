#!/usr/bin/env python3
"""Verify a metadata-only upgrade did not alter source identities or signatures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from reconcile_candidates import top_items,file_sha


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def occurrence_key(value,new=False):
    value=dict(value);return_cv=value.pop('return_cv_qualifiers',None)
    for name in ('declared_type_spelling','declared_type_role'):value.pop(name,None)
    if value.get('parameters'):
        value['parameters']=[{k:v for k,v in p.items()if k!='index'}if isinstance(p,dict)else p for p in value['parameters']]
    qualifiers=value.pop('qualifiers',None)
    return fingerprint(value),qualifiers,return_cv


def compare(before,after):
    old_entities={};old_occurrences={};issues=[];counts=Counter();seen_entities=set();seen_occurrences=set()
    for key,value,item in top_items(before,arrays=('entities','occurrences')):
        if not item:continue
        if key=='entities':old_entities[value['entity_id']]=fingerprint(value)
        elif key=='occurrences':old_occurrences[value['declaration_occurrence_id']]=occurrence_key(value)
    for key,value,item in top_items(after,arrays=('entities','occurrences')):
        if not item:continue
        if key=='entities':
            identifier=value['entity_id'];seen_entities.add(identifier)
            if old_entities.get(identifier)!=fingerprint(value):issues.append({'kind':'entity_identity_changed','id':identifier})
        elif key=='occurrences':
            identifier=value['declaration_occurrence_id'];seen_occurrences.add(identifier)
            actual,qualifiers,return_cv=occurrence_key(value,True);old=old_occurrences.get(identifier)
            if old is None or old[0]!=actual:
                issues.append({'kind':'unplanned_occurrence_field_changed','id':identifier,'path':value['path'],'name':value['qualified_name']});continue
            expected=list(old[1])if old[1]is not None else None
            if return_cv is not None:
                for cv in return_cv:
                    if expected is None or cv not in expected:
                        issues.append({'kind':'return_cv_missing_from_old_combined_qualifiers','id':identifier});continue
                    reverse_index=expected[::-1].index(cv);expected.pop(len(expected)-reverse_index-1)
                for index,p in enumerate(value.get('parameters',[])):
                    if p.get('index')!=index:issues.append({'kind':'parameter_index_incorrect','id':identifier,'index':index})
                counts['callable_return_cv_records']+=1
            if qualifiers!=expected:issues.append({'kind':'unplanned_qualifier_difference','id':identifier,'old':old[1],'new':qualifiers,'return_cv':return_cv})
            if qualifiers!=old[1]:counts['qualifier_lists_corrected']+=1
            if 'declared_type_spelling'in value:counts['complete_declarator_spelling_records']+=1
    for identifier in old_entities.keys()-seen_entities:issues.append({'kind':'entity_removed','id':identifier})
    for identifier in old_occurrences.keys()-seen_occurrences:issues.append({'kind':'occurrence_removed','id':identifier})
    return {'metadata_upgrade_invariants_passed':not issues,'before_sha256':file_sha(before),'after_sha256':file_sha(after),
        'entity_count':len(seen_entities),'occurrence_count':len(seen_occurrences),'allowed_changes':dict(counts),
        'issue_count':len(issues),'issues':issues,
        'coverage_claim':'Identity/source preservation and planned metadata delta only; source semantics and completeness reviewed separately'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('before',type=Path);parser.add_argument('after',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=compare(args.before,args.after)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='issues'},ensure_ascii=False))
    raise SystemExit(0 if result['metadata_upgrade_invariants_passed']else 1)

#!/usr/bin/env python3
"""Compare semantic records without mistaking new trust flags for source edits."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

from reconcile_candidates import top_items,file_sha

FIELDS=('kind','name','qualified_name','raw_signature','access','attributes','return_type','trailing_return_type',
        'declared_type','declared_type_spelling','return_cv_qualifiers','target_type','initializer','parameters','template_parameters','preprocessor_conditions','bases','alignment_specifiers')
POSITION_KEYS={'path','start_byte','end_byte','start_line','end_line','directive_line','directive_path','entity_id',
               'expanded_raw','constraint_projection_range','virtual_range','source_segments','owner_entity_id','owner_occurrence_id'}

def clean(value):
    if isinstance(value,list):return [clean(v)for v in value]
    if isinstance(value,dict):return {k:clean(v)for k,v in value.items()if k not in POSITION_KEYS}
    return value

def inventory(path):
    counts=defaultdict(Counter);samples={};review=Counter();metadata={};scope_findings=[]
    for key,value,item in top_items(path,arrays=('occurrences','diagnostics')):
        if not item:metadata[key]=value;continue
        if key=='diagnostics'and value.get('category')=='scope_closure_mismatch':
            scope_findings.append(value)
        if key!='occurrences':continue
        signature=value['signature_range']
        record={'signature_bytes':[signature['start_byte'],signature['end_byte']],
                'semantics':clean({field:value.get(field)for field in FIELDS})}
        fingerprint=hashlib.sha256(json.dumps(record,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        counts[value['path']][fingerprint]+=1
        samples[(value['path'],fingerprint)]={'qualified_name':value['qualified_name'],'kind':value['kind'],
            'signature_start_line':signature['start_line'],'signature_end_line':signature['end_line']}
        if value.get('scope_review_required'):review[value['path']]+=1
    return counts,samples,review,metadata.get('summary',{}),scope_findings

def compare(before,after):
    old,old_samples,old_review,old_summary,old_findings=inventory(before)
    new,new_samples,new_review,new_summary,new_findings=inventory(after)
    changed=[]
    for path in sorted(old.keys()|new.keys()):
        removed=old[path]-new[path];added=new[path]-old[path]
        if removed or added:
            changed.append({'path':path,'before_occurrences':sum(old[path].values()),'after_occurrences':sum(new[path].values()),
                'removed_semantic_records':sum(removed.values()),'added_semantic_records':sum(added.values()),
                'removed_examples':[dict(old_samples[(path,key)],count=number)for key,number in list(removed.items())[:8]],
                'added_examples':[dict(new_samples[(path,key)],count=number)for key,number in list(added.items())[:8]]})
    return {'before_sha256':file_sha(before),'after_sha256':file_sha(after),'before_summary':old_summary,'after_summary':new_summary,
        'semantic_changed_files':changed,'scope_review_before':dict(old_review),'scope_review_after':dict(new_review),
        'scope_findings_before':old_findings,'scope_findings_after':new_findings,
        'policy':'Compares physical signature ranges and semantic fields; parser traces, IDs and new scope-trust metadata excluded. This is a change audit, not completeness proof.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('before',type=Path);parser.add_argument('after',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=compare(args.before,args.after);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'changed_files':len(result['semantic_changed_files']),
        'files':[x['path']for x in result['semantic_changed_files']],
        'after_scope_review_occurrences':sum(result['scope_review_after'].values())},ensure_ascii=False))

if __name__=='__main__':main()

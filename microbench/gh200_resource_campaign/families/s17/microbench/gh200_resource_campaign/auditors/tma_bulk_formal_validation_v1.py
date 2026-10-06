"""Finite ABI1 for capture=false final tile/ring checks; bytes replay separately."""
import hashlib,json,re
from auditors import tma_bulk,tma_bulk_formal_v1 as formal
from auditors.tma_bulk_validation import relative_path
from common.suite_io import require
ADAPTER_ID='tma_bulk_formal_validation_v1'
ADAPTER_ABI_VERSION=1
PROFILE_ID='formal_final_1_2_33_v1'
PROFILE_HASH='abe982155183e73ea54f52595107e8af0b58044b4d8e48e9dbe3ffe239301644'
validate_device=tma_bulk.validate_device
audit_sass=tma_bulk.audit_sass

def profile_check(profile):
    require(profile.get('id')==PROFILE_ID and hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':')).encode()).hexdigest()==PROFILE_HASH,'frozen formal-path group profile')

def validation_argv(binary_relative,case,profile,seed):
    profile_check(profile);tma_bulk.case_identity(case)
    require(type(seed) is int and seed==3,'formal-path top seed3')
    return [relative_path(binary_relative),'validate-only',case['id'],PROFILE_ID,'3']

def validate_validation(device,row,case,profile,seed):
    profile_check(profile);require(type(seed) is int and seed==3,'formal-path top seed3')
    fields={'schema_version','validation_schema_version','type','case_id','profile_id','seed','scope','threads','blocks','errors','target_launches','checks','resource_identity','performance_eligible','warmup_executed','pilot_executed'}
    require(set(row)==fields,'fixed ABI1 row fields')
    for field,value in [('schema_version',2),('validation_schema_version',1),('seed',3),('threads',128),('errors',0)]:require(type(row[field]) is int and row[field]==value,'fixed integer '+field)
    require(row['type']=='validation' and row['case_id']==case['id'] and row['profile_id']==PROFILE_ID and row['scope']==case['scope'],'formal-path identity')
    require(all(row[k] is False for k in ('performance_eligible','warmup_executed','pilot_executed')),'short cannot measure performance')
    blocks=formal.short_blocks(device,case,row);q=case['parameters']['payload_bytes'];g2s=case['parameters']['direction']=='gmem_to_smem';words=q//4
    require(len(row['target_launches'])==len(row['checks'])==3,'exact three paired launches')
    total=0
    for index,((length,_),launch,check) in enumerate(zip(formal.SHORT_PAIRS,row['target_launches'],row['checks'])):
        require(launch=={'launch_index':index,'iterations':length,'input_profile':'paired_nonuniform_final_tile','threads':128,'blocks':blocks} and all(type(launch[k]) is int for k in ('launch_index','iterations','threads','blocks')),'fixed paired launch')
        check_fields={'launch_index','reference_model','reference_sha256','comparison','tolerance_id','checked_elements','expected_elements','errors','completed','verified_CTA_ids','output_artifacts'}
        require(set(check)==check_fields and type(check['launch_index']) is int and check['launch_index']==index and check['reference_model']==profile['reference_identity']['model'] and check['reference_sha256']==profile['reference_identity']['sha256'] and check['comparison']=='exact' and check['tolerance_id'] is None,'fixed full reference')
        count=blocks*words*(1 if g2s else 32)+8;total+=count
        require(type(check['errors']) is int and check['errors']==0 and check['completed'] is True and type(check['checked_elements']) is int and type(check['expected_elements']) is int and check['checked_elements']==check['expected_elements']==count,'full checked words')
        require(check['verified_CTA_ids']==list(range(blocks)) and all(type(i) is int for i in check['verified_CTA_ids']),'all CTA IDs')
        prefix=f'formal_{index}_';layouts={prefix+'completion.u32le':[blocks,5,2],prefix+'stamps.u32le':[blocks,5,2],prefix+'guards.u32le':[2,4],prefix+('final_tile.u32le' if g2s else 'ring.u32le'):([blocks,words] if g2s else [blocks,32,words])}
        require(len(check['output_artifacts'])==4 and {a['path'] for a in check['output_artifacts']}==set(layouts),'complete distinct artifacts')
        for a in check['output_artifacts']:
            require(set(a)=={'path','sha256','dtype','shape','evidence_kind'} and relative_path(a['path']) in layouts and a['dtype']=='uint32' and a['evidence_kind']=='full_values' and a['shape']==layouts[a['path']] and all(type(n) is int for n in a['shape']) and re.fullmatch('[0-9a-f]{64}',a['sha256']),'lossless artifact metadata')
    return {'status':'pass','case_id':case['id'],'profile_id':PROFILE_ID,'target_launches':3,'checked_elements':total,'output_evidence_kind':'full_values','performance_eligible':False}

def validate_prior_evidence(evidence,request,frozen_refs):
    tma_bulk.case_identity(request['case']);profile_check(request['profile'])
    require(isinstance(evidence,dict) and isinstance(frozen_refs,dict),'prior evidence object')
    return {'status':'insufficient','case_id':request['case']['id'],'missing_requirements':['original84 capture=true checks do not qualify capture=false final drain; no prior reuse approved']}

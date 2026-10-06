"""Pure ABI1 for the reviewed finite S13 1/2-sweep profile; never runs GPU/I/O."""
from pathlib import PurePosixPath
import hashlib
import json
import re
from common.suite_io import require
from auditors.global_duplex import (audit_sass,validate_device,case_identity,
    validate_resources,expected_elements)

ADAPTER_ID='global_duplex_validation_v1'
ADAPTER_ABI_VERSION=1
PROFILE_ID='short_global_duplex_1_2'
PROFILE_CANONICAL_SHA256='1be4072529705bde1c8fb8d03b9627ece1dfee07bfe0c6f4bf82db5607ac4c7c'
REFERENCE_MODEL='global_duplex_modular_sweep_v1'
REFERENCE_SHA256='e6b29111204b1f379188737712aa75334cf9ae1c813a51f02efea44b20c5a696'


def profile_check(profile):
    require(isinstance(profile,dict) and hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':')).encode()).hexdigest()==PROFILE_CANONICAL_SHA256,'frozen finite S13 profile')


def relative_path(value):
    require(isinstance(value,str) and value,'relative binary/artifact path')
    path=PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in value and path.as_posix()==value and value!='.','path escape/noncanonical')
    return value


def validation_argv(binary_relative,case,profile,seed):
    profile_check(profile);case_identity(case)
    require(type(seed) is int and 0<=seed<=0xffffffff,'uint32 seed')
    return [relative_path(binary_relative),'validate-only',case['id'],PROFILE_ID,str(seed)]


def validate_validation(device,row,case,profile,seed):
    profile_check(profile);case_identity(case);validate_device(device)
    require(type(seed) is int and 0<=seed<=0xffffffff,'uint32 seed')
    required={'schema_version','validation_schema_version','type','case_id','profile_id','seed','scope','threads','blocks','errors','target_launches','checks','resource_identity','performance_eligible','warmup_executed','pilot_executed'}
    require(isinstance(row,dict) and set(row)==required,'fixed validation fields')
    for field,value in [('schema_version',2),('validation_schema_version',1),('seed',seed),('threads',256),('errors',0)]:
        require(type(row[field]) is int and row[field]==value,'validation integer '+field)
    require(row['type']=='validation' and row['case_id']==case['id'] and row['profile_id']==PROFILE_ID and row['scope']=='all_gpu','validation identity')
    require(all(row[k] is False for k in ('performance_eligible','warmup_executed','pilot_executed')),'short-only false flags')
    plan=validate_resources(row['resource_identity'],case,device);blocks=plan['blocks'];count=expected_elements(plan)
    require(type(row['blocks']) is int and row['blocks']==blocks,'original grid preserved')
    launches,checks=row['target_launches'],row['checks']
    require(isinstance(launches,list) and isinstance(checks,list) and len(launches)==len(checks)==2,'exactly two target/check entries')
    require(all(isinstance(c,dict) and type(c.get('launch_index')) is int for c in checks) and {c['launch_index'] for c in checks}=={0,1},'unique complete launch check IDs')
    indexed={c['launch_index']:c for c in checks}
    for index,length in enumerate((1,2)):
        launch=launches[index]
        require(isinstance(launch,dict) and all(type(launch.get(k)) is int for k in ('launch_index','iterations','threads','blocks')),'launch integer fields')
        require(launch=={'launch_index':index,'iterations':length,'input_profile':'address_seed_nonuniform','threads':256,'blocks':blocks},'fixed launch plan')
        check=indexed[index]
        require(set(check)=={'launch_index','reference_model','reference_sha256','comparison','tolerance_id','checked_elements','expected_elements','errors','completed','verified_CTA_ids','output_artifacts'},'check fields')
        require(check['reference_model']==REFERENCE_MODEL and check['reference_sha256']==REFERENCE_SHA256 and check['comparison']=='exact' and check['tolerance_id'] is None,'independent frozen exact reference')
        require(type(check['errors']) is int and check['errors']==0 and check['completed'] is True,'failed numeric/completion evidence')
        require(type(check['checked_elements']) is int and type(check['expected_elements']) is int and check['checked_elements']==check['expected_elements']==count,'all nonempty read checksums and destination words; empty write checksums excluded')
        ids=check['verified_CTA_ids'];require(isinstance(ids,list) and all(type(i) is int for i in ids) and len(ids)==blocks and set(ids)==set(range(blocks)),'all launched CTA IDs')
        require(check['output_artifacts']==[],'error_count_only; no offline reference replacement')
    return {'status':'pass','case_id':case['id'],'profile_id':PROFILE_ID,'target_launches':2,
            'checked_elements':2*count,'output_evidence_kind':'error_count_only','performance_eligible':False}


def validate_prior_evidence(evidence,request,frozen_refs):
    require(isinstance(evidence,dict) and isinstance(request,dict) and isinstance(frozen_refs,dict),'evidence/request/frozen refs objects')
    case=request.get('case');case_identity(case);profile_check(request.get('profile'))
    declared=evidence.get('source_artifacts_sha256',{})
    require(isinstance(declared,dict) and (not evidence or declared),'prior evidence artifact bindings')
    for name,digest in declared.items():
        relative_path(name);require(isinstance(digest,str) and re.fullmatch(r'[0-9a-f]{64}',digest) and frozen_refs.get(name)==digest,'corrupt prior evidence')
    return {'status':'insufficient','case_id':case['id'],'missing_requirements':['no independently approved S13 semantic reuse mapping; no prior numeric evidence imported']}

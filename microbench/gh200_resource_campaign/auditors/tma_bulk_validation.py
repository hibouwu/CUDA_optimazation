"""Pure ABI1 metadata adapter for four finite S14 single-target profiles.

Actual complete word values are separately replayed by audit_artifact_values;
ABI1 receives no file bytes. Core checks each declared artifact's size and hash.
"""
from pathlib import PurePosixPath
import hashlib,json,re
from common.suite_io import require
from auditors.tma_bulk import (audit_sass,validate_device,case_identity,profile_coordinate,
    validate_resources,expected_elements,artifact_layouts,audit_artifact_values,complete_profile_set)

ADAPTER_ID='tma_bulk_validation_v1'
ADAPTER_ABI_VERSION=1
PROFILE_HASHES={'bulk_short_1_seed0_v1': '79a978bec386052d7e01a009cc8649b77f6197f48440e8f9271df466f0a8662c', 'bulk_short_2_seed3_v1': '516c521c9581c45db69853d1609e697fcc1fb768da3b9876d51ea68bfb52756f', 'bulk_short_33_seed4294967295_v1': 'f8fe40d566db7923b1467ceabbd3d4c7cd754d0cb03c344265612b022fb9b9f7', 'bulk_source_release_one_request_v1': '72bbd00f2031b46887c42e577b1e8a2206420ec53b787809ccea8af5e35ee371'}
REFERENCE_SHA256='98d20cfdfbdf5fc627976764ce87ceba032c94f09341bdac0e41ac95ec7bcf7b'


def profile_check(profile):
    require(isinstance(profile,dict) and profile.get('id') in PROFILE_HASHES,'finite S14 profile object')
    require(hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':')).encode()).hexdigest()==PROFILE_HASHES[profile['id']],'frozen executable S14 profile')


def relative_path(value):
    require(isinstance(value,str) and value,'nonempty relative path')
    p=PurePosixPath(value);require(not p.is_absolute() and '..' not in p.parts and '\\' not in value and p.as_posix()==value and value!='.','relative artifact/binary path')
    return value


def validation_argv(binary_relative,case,profile,seed):
    profile_check(profile);profile_coordinate(profile,case,seed)
    return [relative_path(binary_relative),'validate-only',case['id'],profile['id'],str(seed)]


def validate_validation(device,row,case,profile,seed):
    profile_check(profile);length,release=profile_coordinate(profile,case,seed);validate_device(device)
    fields={'schema_version','validation_schema_version','type','case_id','profile_id','seed','scope','threads','blocks','errors','target_launches','checks','resource_identity','performance_eligible','warmup_executed','pilot_executed'}
    require(isinstance(row,dict) and set(row)==fields,'fixed validation row fields')
    for field,value in [('schema_version',2),('validation_schema_version',1),('seed',seed),('threads',128),('errors',0)]:require(type(row[field]) is int and row[field]==value,'fixed integer '+field)
    require(row['type']=='validation' and row['case_id']==case['id'] and row['profile_id']==profile['id'] and row['scope']==case['scope'],'S14 identity')
    require(all(row[k] is False for k in ('performance_eligible','warmup_executed','pilot_executed')),'no performance/extra work qualification')
    blocks=validate_resources(row['resource_identity'],case,profile,device,seed)
    require(type(row['blocks']) is int and row['blocks']==blocks,'original actual-resource grid')
    launches,checks=row['target_launches'],row['checks'];require(isinstance(launches,list) and isinstance(checks,list) and len(launches)==len(checks)==1,'one target and one check per independent profile/process')
    launch=launches[0];expected={'launch_index':0,'iterations':length,'input_profile':profile['input_profiles'][0],'threads':128,'blocks':blocks}
    require(isinstance(launch,dict) and all(type(launch.get(k)) is int for k in ('launch_index','iterations','threads','blocks')) and launch==expected,'fixed single launch fields; seed is top-level only')
    check=checks[0];require(isinstance(check,dict) and set(check)=={'launch_index','reference_model','reference_sha256','comparison','tolerance_id','checked_elements','expected_elements','errors','completed','verified_CTA_ids','output_artifacts'},'complete output check fields')
    require(type(check['launch_index']) is int and check['launch_index']==0 and check['reference_model']=='tma_bulk_ring_word_reference_v1' and check['reference_sha256']==REFERENCE_SHA256 and check['comparison']=='exact' and check['tolerance_id'] is None,'frozen exact reference')
    require(type(check['errors']) is int and check['errors']==0 and check['completed'] is True,'numeric or lifecycle failure')
    count=expected_elements(case,profile,blocks,seed)
    require(type(check['checked_elements']) is int and type(check['expected_elements']) is int and check['checked_elements']==check['expected_elements']==count,'every declared payload and guard word')
    ids=check['verified_CTA_ids'];require(isinstance(ids,list) and all(type(i) is int for i in ids) and len(ids)==blocks and set(ids)==set(range(blocks)),'all launched CTA participants')
    artifacts=check['output_artifacts'];layouts=artifact_layouts(case,profile,blocks,seed)
    require(isinstance(artifacts,list) and len(artifacts)==len(layouts),'complete full-value artifact set')
    seen=set()
    for item in artifacts:
        require(isinstance(item,dict) and set(item)=={'path','sha256','dtype','shape','evidence_kind'},'artifact metadata fields')
        path=relative_path(item['path']);require(path in layouts and path not in seen,'unique required artifact role');seen.add(path)
        require(item['dtype']=='uint32' and item['evidence_kind']=='full_values' and isinstance(item['shape'],list) and all(type(n) is int and n>0 for n in item['shape']) and item['shape']==layouts[path],'lossless uint32 shape/encoding')
        require(isinstance(item['sha256'],str) and re.fullmatch(r'[0-9a-f]{64}',item['sha256']),'complete file SHA256')
    return {'status':'pass','case_id':case['id'],'profile_id':profile['id'],'target_launches':1,'checked_elements':count,'output_evidence_kind':'full_values','performance_eligible':False}


def validate_prior_evidence(evidence,request,frozen_refs):
    require(isinstance(evidence,dict) and isinstance(request,dict) and isinstance(frozen_refs,dict),'prior evidence/request/ref maps')
    case_identity(request.get('case'));profile_check(request.get('profile'))
    declared=evidence.get('source_artifacts_sha256',{});require(isinstance(declared,dict) and (not evidence or declared),'prior evidence must bind artifacts')
    for path,digest in declared.items():
        relative_path(path);require(isinstance(digest,str) and re.fullmatch(r'[0-9a-f]{64}',digest) and frozen_refs.get(path)==digest,'corrupt prior evidence')
    return {'status':'insufficient','case_id':request['case']['id'],'missing_requirements':['no independently approved S14 prior numerical reuse mapping; feasibility compile is not GPU correctness']}

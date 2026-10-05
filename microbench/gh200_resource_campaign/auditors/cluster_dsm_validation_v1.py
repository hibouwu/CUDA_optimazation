"""S17 finite short ABI and independent output replay; actual B2 still fail closed."""
import re,hashlib,json,math
from common.suite_io import require
from auditors.async_copy import validate_device
from auditors import cluster_dsm_reference_v1 as ref
ADAPTER_ID='cluster_dsm_multicast_v1'
ADAPTER_ABI_VERSION=1
from auditors.cluster_dsm_sass_baseline_v1 import BASELINE
from auditors.cluster_dsm_sass_v1 import audit_sass
PROFILE_SHA='ec78181501e83bc0a36dde0822f9b359ac1b8101e64f74a021b13a69924bce98'

def profile_check(p):require(hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()==PROFILE_SHA,'S17 exact profile')

def coordinate(case):
 form=case['form'];c=case['cluster_size'];scope=case['scope']
 require(form in ref.FORMS and type(c)is int and c in (2,4,8) and scope in ('one_cluster','cluster_grid') and case['id']==f'{form}_c{c}_{scope}' and case['parameters']=={'form':form,'cluster_size':c} and case['launch']=={'scope':scope,'threads':128},'S17 finite coordinate')
 return form,c,scope

def validation_argv(binary,case,p,seed):
 from pathlib import PurePosixPath
 profile_check(p);coordinate(case);require(type(seed)is int and seed==3,'S17 topseed3');path=PurePosixPath(binary);require(not path.is_absolute() and '..' not in path.parts and path.as_posix()==binary and '\\' not in binary and binary!='.','S17 binary path');return [binary,'validate-only',case['id'],p['id'],'3']

def symbol(form,c):return f"s17_{'bulk_single' if form=='bulk_single_target' else 'bulk_all' if form=='bulk_all_targets' else form}_c{c}"

def resources(device,row,case):
 validate_device(device);form,c,scope=coordinate(case);x=row['resource_identity'];e=x['extensions'];b=ref.uint(row['blocks']);require(b>0 and b%c==0 and row['threads']==128 and type(row['threads'])is int and row['scope']==scope,'S17 cluster grid geometry');g=b//c
 for key in ('registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm'):ref.uint(x[key])
 require(x['kernel_symbol']==symbol(form,c) and x['kernel_symbol'] in BASELINE,'S17 actual B2 target baseline required')
 require({k:x[k] for k in ('registers_per_thread','static_smem_bytes','local_size_bytes')}=={k:BASELINE[x['kernel_symbol']][k] for k in ('registers_per_thread','static_smem_bytes','local_size_bytes')} and x['local_size_bytes']==0 and x['dynamic_smem_bytes']==0 and x['occupancy_limit_ctas_per_sm']==0,'S17 exact actual compiler resources, cluster occupancy separate')
 require(set(e)=={'cluster_launch_supported','potential_cluster_size','active_cluster_capacity','cluster_size','clusters','mask','opaque_token_words_preserved_domain_only','occupancy_is_upper_bound'},'S17 exact resource extensions')
 for key in ('cluster_launch_supported','potential_cluster_size','active_cluster_capacity','cluster_size','clusters','mask'):ref.uint(e[key])
 require(e['cluster_launch_supported']==1 and e['potential_cluster_size']>=c and 0<e['active_cluster_capacity']<=1024 and e['cluster_size']==c and e['clusters']==g and g==(1 if scope=='one_cluster' else e['active_cluster_capacity']) and e['mask']==ref.mask(form,c) and e['occupancy_is_upper_bound']is True,'S17 actual query/selected cluster configuration')
 require(e['opaque_token_words_preserved_domain_only']==[ref.ledger(form,c,g,i)['opaque_token_words'] for i,_ in ref.PAIRS],'S17 opaque token exact exclusion count')
 return form,c,g

def validate_validation(device,row,case,p,seed):
 profile_check(p);require(type(seed)is int and seed==3,'S17 seed3')
 for k in ('schema_version','validation_schema_version','seed','blocks','threads','errors'):ref.uint(row[k])
 require(row['schema_version']==2 and row['validation_schema_version']==1 and row['type']=='validation' and row['case_id']==case['id'] and row['profile_id']==p['id'] and row['seed']==3 and row['errors']==0,'S17 identity and errors')
 for k in ('performance_eligible','warmup_executed','pilot_executed'):require(row[k]is False,'S17 only short targets')
 form,c,g=resources(device,row,case);b=c*g;require(len(row['checks'])==len(row['target_launches'])==3,'S17 fixed3 launch group')
 checked=0
 for n,(i,local_seed) in enumerate(ref.PAIRS):
  launch={'launch_index':n,'iterations':i,'input_profile':'nonuniform_uint32_cluster_paired_seeds_v1','threads':128,'blocks':b}
  require(row['target_launches'][n]==launch and all(type(row['target_launches'][n][k])is int for k in ('launch_index','iterations','threads','blocks')),'S17 fixed ordered launch and perlaunch seed')
  ch=row['checks'][n];layout=ref.shapes(form,c,g,i);count=sum(math.prod(s) for s in layout.values())-ref.ledger(form,c,g,i)['opaque_token_words']
  require(ch['launch_index']==n and type(ch['launch_index'])is int and ch['completed']is True and type(ch['errors'])is int and ch['errors']==0 and ch['reference_model']==p['reference_identity']['model'] and ch['reference_sha256']==p['reference_identity']['sha256'] and ch['comparison']=='exact' and ch['tolerance_id']is None and ch['verified_CTA_ids']==list(range(b)) and all(type(v)is int for v in ch['verified_CTA_ids']),'S17 exact check roles')
  require(type(ch['checked_elements'])is int and type(ch['expected_elements'])is int and ch['checked_elements']==ch['expected_elements']==count,'S17 independent exact count')
  require(len(ch['output_artifacts'])==len(layout),'S17 complete output')
  for item,(leaf,shape) in zip(ch['output_artifacts'],layout.items()):require(item['path']==f'cluster_{n}_{leaf}.u32le' and item['shape']==shape and item['dtype']=='uint32' and item['evidence_kind']=='full_values' and re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,'S17 artifact identity/shape')
  checked+=count
 return {'status':'pass','case_id':case['id'],'profile_id':p['id'],'target_launches':3,'checked_elements':checked,'performance_eligible':False,'output_evidence_kind':'full_values'}

def audit_values(device,row,case,p,seed,arrays):
 validate_validation(device,row,case,p,seed);form,c,g=resources(device,row,case)
 for n,(i,s) in enumerate(ref.PAIRS):ref.audit(form,c,g,i,s,{leaf:arrays[f'cluster_{n}_{leaf}.u32le'] for leaf in ref.shapes(form,c,g,i)})
 return {'status':'pass','performance_eligible':False,'full_values_replayed':True}


def validate_prior_evidence(evidence,request,artifacts):
 return {'status':'insufficient','reason':'S17 new finite short evidence required; feasibility compile cannot grant B3'}

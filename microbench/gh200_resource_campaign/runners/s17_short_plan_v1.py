"""Finite42 short planning from a reviewed actual21-target capability query."""
import json,math
from common.suite_io import require
from auditors.async_copy import validate_device
from auditors.cluster_dsm_sass_baseline_v1 import BASELINE
from auditors.cluster_dsm_reference_v1 import FORMS,PAIRS,shapes
from auditors.cluster_dsm_validation_v1 import coordinate,symbol,profile_check


def capabilities(device,rows):
 validate_device(device);require(type(rows)is list and len(rows)==21,'S17 full21 capability rows')
 expected=[symbol(form,c) for c in (2,4,8) for form in FORMS];require([r['kernel_symbol'] for r in rows]==expected,'S17 actual source query ordering and exact21 targets')
 out={}
 fields={'type','kernel_symbol','cluster_size','cluster_launch_supported','potential_cluster_size','active_cluster_capacity','cuda_error','registers_per_thread','static_smem_bytes','local_size_bytes','occupancy_is_upper_bound'}
 for row in rows:
  require(set(row)==fields and row['type']=='cluster_capability' and row['occupancy_is_upper_bound']is True,'S17 fixed query schema')
  c=int(row['kernel_symbol'].rsplit('_c',1)[1]);require(type(row['cluster_size'])is int and row['cluster_size']==c,'S17 requested C')
  for k in ('cluster_launch_supported','potential_cluster_size','active_cluster_capacity','cuda_error','registers_per_thread','static_smem_bytes','local_size_bytes'):require(type(row[k])is int and -(2**31)<=row[k]<2**32,'S17 exact integer query fields')
  if row['cuda_error']!=0:status='query_error_before_target'
  else:
   require(row['cluster_launch_supported'] in (0,1),'S17 launch support boolean integer')
   if not row['cluster_launch_supported']:status='cluster_launch_not_supported'
   elif row['potential_cluster_size']<c:status='selected_C_not_supported'
   elif row['active_cluster_capacity']==0:status='zero_active_cluster_capacity'
   else:
    require(row['active_cluster_capacity']>0,'S17 nonnegative successful active capacity')
    actual={k:row[k] for k in ('registers_per_thread','static_smem_bytes','local_size_bytes')};expected_resource={k:BASELINE[row['kernel_symbol']][k] for k in actual};require(actual==expected_resource,'S17 actual queried compiler resource equals signed B2')
    status='supported_query_only' if row['active_cluster_capacity']<=1024 else 'protocol_G_bound_exceeded'
  out[row['kernel_symbol']]={'status':status,'query':row,'target_launches':3 if status=='supported_query_only' else 0,'GPU_numerical_qualified':False}
 return out


def plan(contract,profile,device,query_rows):
 profile_check(profile);require(len(contract['cases'])==42 and len({c['id'] for c in contract['cases']})==42,'S17 fixed42 contract');caps=capabilities(device,query_rows);points=[]
 for index,case in enumerate(contract['cases']):
  form,c,scope=coordinate(case);cap=caps[symbol(form,c)];g=(1 if scope=='one_cluster' else cap['query']['active_cluster_capacity']) if cap['target_launches'] else 0
  bytes_=sum(4*sum(math.prod(s) for s in shapes(form,c,g,i).values()) for i,_ in PAIRS) if g else 0
  points.append({'index':index,'case_id':case['id'],'profile_id':profile['id'],'seed':3,'form':form,'cluster_size':c,'scope':scope,'clusters':g,'blocks':c*g,'status':cap['status'],'target_launches':cap['target_launches'],'artifact_bytes_upper_bound':bytes_,'resource_query':cap['query'],'occupancy_is_upper_bound':True})
 valid=[p for p in points if p['target_launches']];maximum=max(valid,key=lambda p:p['artifact_bytes_upper_bound'])['index'] if valid else None
 return {'nominal_coordinates':42,'supported_coordinates':len(valid),'planned_target_launches':3*len(valid),'unsupported_or_query_error_coordinates':42-len(valid),'maximum_index':maximum,'maximum_artifact_bytes_upper_bound':max((p['artifact_bytes_upper_bound'] for p in valid),default=0),'total_artifact_bytes_upper_bound':sum(p['artifact_bytes_upper_bound'] for p in valid),'coordinates':points,'device_capability_requires_verified_query':True,'actual_residency_claim':False,'GPU_numerical_qualified':False}


def strict_query_rows(data):
 def pairs(items):
  out={}
  for k,v in items:require(k not in out,'duplicate query field');out[k]=v
  return out
 def nonfinite(x):raise ValueError('nonfinite query JSON')
 rows=[json.loads(line,object_pairs_hook=pairs,parse_constant=nonfinite) for line in data.decode('utf-8').splitlines()];require(len(rows)==22,'S17 device+full21 query lines');return rows[0],rows[1:]

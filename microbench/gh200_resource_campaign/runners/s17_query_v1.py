"""Read-only S17 cluster query. No target launches; reviewed binary and Slurm only."""
from pathlib import Path
import os,time,uuid
from common.suite_io import require,read_json,sha,validate_gate,bounded,process_ok,atomic_json,gpu_lock
from common.s16_quota_v2 import query_destination
from runners.s17_short_plan_v1 import strict_query_rows,capabilities
SUITE='results/gh200_resource_campaign/20261001-resource-suite-v2'
SOURCE_REVIEW='reviews/S17-short-source-B-review.json'


def query(repo,binary,output):
 from runners.environment import inspect_allocation,environment_identity,budget
 repo,output=query_destination(repo,output);name=output.relative_to(repo)
 require(len(name.parts)==3 and name.parts[0]=='deployment_receipts' and name.parts[1].startswith('query-') and name.parts[2]=='cluster-query.json','finite query output; never replace source or archive')
 binary=Path(binary).resolve();suite=repo/SUITE;review_path=suite/SOURCE_REVIEW
 review=validate_gate(review_path,repo,'S17','early-validation-source-B')
 approved_name=SUITE+'/implementation/target-compile-s17-short-r2/diagnostics/cluster_dsm_multicast/probe'
 require(binary==(repo/approved_name).resolve() and binary.is_file() and sha(binary)==review['gate_files'].get(approved_name),'S17 canonical actual reviewed binary only')
 allocation=inspect_allocation();require(name.parts[1]=='query-'+allocation['job'],'query directory belongs to current allocation');budget(allocation,30);environment=environment_identity(allocation);require('GH200' in environment['name'],'allocated GH200 query only')
 output.parent.mkdir(parents=True,exist_ok=True);queries=output.parent/(output.stem+'.queries');require(not queries.is_symlink(),'immutable query namespace regular');queries.mkdir(exist_ok=True)
 folder=queries/(str(time.time_ns())+'-'+str(os.getpid())+'-'+uuid.uuid4().hex);folder.mkdir();stdout=folder/'stdout';stderr=folder/'stderr'
 with gpu_lock(allocation['uuid']):receipt=bounded([str(binary),'cluster-device'],repo,stdout,stderr,30,uuid=allocation['uuid'])
 atomic_json(folder/'process.json',receipt)
 if not process_ok(receipt):
  atomic_json(folder/'query.json',{'status':'cluster_query_process_failed','source_review_sha256':sha(review_path),'receipt_sha256':sha(folder/'process.json'),'GPU_target_launches':0,'alias_published':False});raise ValueError('S17 query process failed; no target or alias')
 device,rows=strict_query_rows(stdout.read_bytes());require(device['uuid'].lower()==environment['uuid'].lower(),'actual query allocated UUID');caps=capabilities(device,rows)
 result={'schema_version':1,'status':'complete_actual_cluster_query_preserved','device':device,'capability_rows':rows,'capability_statuses':caps,'environment':environment,'allocation_job':allocation['job'],'source_review_sha256':sha(review_path),'binary_sha256':sha(binary),'stdout':{'path':str(stdout.relative_to(repo)),'sha256':sha(stdout)},'stderr':{'path':str(stderr.relative_to(repo)),'sha256':sha(stderr)},'receipt':receipt,'GPU_target_launches':0,'GPU_numerical_qualified':False}
 atomic_json(folder/'query.json',result);atomic_json(output,result);return result

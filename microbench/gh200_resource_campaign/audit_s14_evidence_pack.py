"""Actual complete original84 pack byte and word proof, pending independent B.

No GPU; fixed metadata descriptors are derived from original signed files and
checked against the independent closure. No success grants bridge qualification.
"""
import argparse,json,time,resource
from pathlib import Path
from common.suite_io import read_json,relative,require,sha
from common.packed_evidence import PackedEvidence,json_object
from common.s14_packed_observer import S14EvidenceObserver
from pack_s14_evidence import COVERAGE,REVIEW,CONTRACT,PROFILES,REVIEW_SHA,COVERAGE_SHA


def prepare_original84(repo,closure):
    require(closure['parent_review']=={'path':REVIEW,'sha256':REVIEW_SHA}
            and closure['parent_coverage']=={'path':COVERAGE,'sha256':COVERAGE_SHA},'signed finite original84 roots')
    coverage=read_json(relative(repo,COVERAGE));require(sha(relative(repo,COVERAGE))==COVERAGE_SHA,'original coverage bytes')
    descriptors={};small={REVIEW,COVERAGE,CONTRACT,PROFILES}
    small.update(context['review_path'] for context in closure['review_contexts'])
    for record in coverage['records']:
        root=record['run_path'];raw_path=root+'/attempts/attempt_00/raw.jsonl';require(sha(relative(repo,raw_path))==closure['members'][raw_path]['sha256'],'original raw descriptor bytes')
        rows=[json_object(line) for line in relative(repo,raw_path).read_bytes().splitlines()];require(len(rows)==2,'original complete raw');row=rows[1]
        spec=read_json(relative(repo,root+'/validation_spec.json'));contract=read_json(relative(repo,root+'/'+spec['contract_path']));case=next(c for c in contract['cases'] if c['id']==record['case_id']);q=case['parameters']['payload_bytes'];steps=row['target_launches'][0]['iterations'];release=record['profile_id']=='bulk_source_release_one_request_v1';blocks=row['blocks']
        for artifact in row['checks'][0]['output_artifacts']:
            name=root+'/attempts/attempt_00/'+artifact['path'];require(name not in descriptors and name in closure['members'],'full unique original artifact');member=closure['members'][name];require(member['sha256']==artifact['sha256'],'frozen original array identity')
            descriptors[name]={'run':root,'role':artifact['path'],'shape':artifact['shape'],'bytes':member['bytes'],'blocks':blocks,'words':q//4,'iterations':steps,'seed':record['seed'],'release':release,'g2s':case['parameters']['direction']=='gmem_to_smem'}
        for name in ['validation_spec.json','validation_manifest.json','attempts/attempt_00/raw.jsonl','attempts/attempt_00/receipt.json','environment/device.json','environment/initial.json','build/dependencies.json','build/shared_libraries.json','build/compile.stderr']:
            small.add(root+'/'+name)
        small.update(root+'/'+spec[k] for k in ['contract_path','profiles_path','source_path'])
    require(len(coverage['records'])==84 and len(descriptors)==312,'complete original84 descriptor matrix')
    total=sum(closure['members'][name]['bytes'] for name in small)
    require(total<=64*1024*1024 and all(closure['members'][name]['bytes']<=16*1024*1024 for name in small),'bounded required metadata cache')
    return descriptors,sorted(small)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--pack',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();repo=args.repo.resolve();pack=args.pack.resolve();out=args.output.resolve();require(not out.exists(),'keep previous evidence receipts')
    closure=read_json(pack/'closure.json');construction=read_json(pack/'construction.json');descriptors,small=prepare_original84(repo,closure);sass_members={record['run_path']+'/build/sass.stdout':closure['members'][record['run_path']+'/build/sass.stdout']['bytes'] for record in read_json(relative(repo,COVERAGE))['records']};observer=S14EvidenceObserver(descriptors,sass_members,read_json(relative(repo,CONTRACT)))
    bundle=PackedEvidence(pack/'original84.tar.xz',pack/'index.json',expected_index_sha256=construction['index_sha256'],expected_archive_sha256=construction['archive_sha256'],expected_closure=closure['members']);start=time.monotonic();verified=bundle.verify_all(small_objects=small,observer=observer);proof=observer.finish();values=proof['complete_values'];elapsed=time.monotonic()-start
    require(values['payload_guard_words']==2967828384 and values['lifecycle_words']==231192 and len(values['per_run'])==84,'all original numerical/control words')
    receipt={'status':'actual_original84_pack_and_full_values_passed_pending_independent_B','resolver_source_sha256':sha(repo/'microbench/gh200_resource_campaign/common/packed_evidence.py'),'observer_source_sha256':sha(repo/'microbench/gh200_resource_campaign/common/s14_packed_observer.py'),'values_source_sha256':sha(repo/'microbench/gh200_resource_campaign/common/s14_packed_values.py'),'archive_verification':verified,'complete_values':values,'target_facts':proof['target_facts'],'full_SASS_members_checked':proof['full_SASS_members_checked'],'descriptor_sha256':__import__('hashlib').sha256(json.dumps(descriptors,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'required_small_objects':small,'elapsed_seconds':elapsed,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'GPU_execution':False,'bridge_qualified':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({k:receipt[k] for k in ['status','archive_verification','elapsed_seconds','peak_rss_kib','GPU_execution','bridge_qualified']},indent=2))

if __name__=='__main__':main()

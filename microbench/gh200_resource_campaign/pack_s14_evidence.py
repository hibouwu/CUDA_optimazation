"""Build a complete historical S14 evidence closure and lossless XZ archive.

CPU only; reads the exact independently signed original84 collection. Source
files and old archives are never altered. This builder cannot grant bridge/B3.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tarfile

from common.suite_io import read_json, relative, require, sha, gate_mapping

SUITE='results/gh200_resource_campaign/20261001-resource-suite-v2'
REVIEW=SUITE+'/reviews/S14-early-validation-B3-review.json'
REVIEW_SHA='cf4cea9fb258a8b7885026f9d0e14707eaa9ef5fc4d7e87f389a8d9c8775771b'
COVERAGE=SUITE+'/reviews/evidence/S14-early-validation-B3/coverage.json'
COVERAGE_SHA='01ae1060d52255e4fecfadacecd9fcb57cb50597ca5a8a76d60d2c7bed75202f'
CONTRACT='microbench/gh200_resource_campaign/contracts/tma_bulk.json'
PROFILES='microbench/gh200_resource_campaign/contracts/tma_bulk_validation_profiles_v1.json'


def write_new(path,value):
    with path.open('x') as out:json.dump(value,out,sort_keys=True,indent=2);out.write('\n')


def make_closure(repo):
    review_path=relative(repo,REVIEW);coverage_path=relative(repo,COVERAGE)
    require(sha(review_path)==REVIEW_SHA and sha(coverage_path)==COVERAGE_SHA,'immutable original84 parent identity')
    review=read_json(review_path);coverage=read_json(coverage_path)
    require(review['status']=='pass' and review['authorization']['family_B3_qualified'] is True,'original B3 eligibility')
    members={};reviews_seen=set();run_roots=[]
    def add(name,want=None):
        path=relative(repo,name);digest=sha(path)
        require(want is None or digest==want,'original file changed: '+name)
        item={'bytes':path.stat().st_size,'sha256':digest}
        require(name not in members or members[name]==item,'same logical path changed')
        members[name]=item
    def add_review(name,logical_root=''):
        pair=(name,logical_root)
        if pair in reviews_seen:return
        reviews_seen.add(pair);add(name);record=read_json(relative(repo,name))
        for local,digest in gate_mapping(record).items():
            target=logical_root+local;add(target,digest)
            # A bound historical initial-review JSON is an immutable artifact,
            # not a request to resolve its former gates in today's live root.
            # Only review contexts explicitly declared below are interpreted.
    add_review(REVIEW)
    for named in ('S14-early-validation-source-B-review.json','S14-scratch-recovery-review.json'):
        add_review(SUITE+'/reviews/'+named)
    add(COVERAGE,COVERAGE_SHA);add(CONTRACT);add(PROFILES)
    contract=read_json(relative(repo,CONTRACT));profiles=read_json(relative(repo,PROFILES))
    expected={(c['id'],p['id'],p['required_seed']) for c in contract['cases'] for p in profiles['profiles'] if c['parameters']['direction'] in p['applicable_directions']}
    require(len(expected)==84 and len(coverage['records'])==84,'fixed84 expected coverage')
    found=set()
    for record in coverage['records']:
        key=(record['case_id'],record['profile_id'],record['seed']);require(key in expected and key not in found,'unique paired profile coordinate');found.add(key)
        root=record['run_path'];require(root==SUITE+'/tma_bulk/short-v1-'+key[0]+'--'+key[1],'original finite run namespace');run_roots.append(root)
        manifest_name=root+'/validation_manifest.json';add(manifest_name,record['manifest_sha256']);manifest=read_json(relative(repo,manifest_name))
        for local,digest in manifest.items():add(root+'/'+local,digest)
        # State is retained as immutable historical bytes, never used to grant
        # qualification or replace the signed B3+raw receipts.
        add(root+'/validation_state.json')
        spec=read_json(relative(repo,root+'/validation_spec.json'))
        require((spec['case_id'],spec['profile_id'],spec['seed'])==key,'original case spec coordinates')
        snapshot_manifest=read_json(relative(repo,root+'/snapshot/manifest.json'))
        require(all(manifest.get('snapshot/'+local)==digest for local,digest in snapshot_manifest.items()),'run/snapshot manifest closure agreement')
        for old in spec['reviews']:
            name=root+'/snapshot/suite/'+old['path'];add(name,old['sha256']);add_review(name,root+'/snapshot/repo/')
        rows=[json.loads(line) for line in relative(repo,root+'/attempts/attempt_00/raw.jsonl').read_text().splitlines()]
        require(len(rows)==2,'complete historical raw')
        for check in rows[1]['checks']:
            for artifact in check['output_artifacts']:
                local='attempts/attempt_00/'+artifact['path'];require(manifest.get(local)==artifact['sha256'],'raw artifact belongs to original manifest');add(root+'/'+local,artifact['sha256'])
    require(found==expected,'all84 finite profile coordinates present')
    return {'schema_version':1,'namespace':'repository_relative_v1','parent_review':{'path':REVIEW,'sha256':REVIEW_SHA},'parent_coverage':{'path':COVERAGE,'sha256':COVERAGE_SHA},'members':dict(sorted(members.items())),'roots':{'review':REVIEW,'coverage':COVERAGE,'contract':CONTRACT,'profiles':PROFILES,'runs':sorted(root+'/validation_manifest.json' for root in run_roots)},'member_count':len(members),'uncompressed_bytes':sum(m['bytes'] for m in members.values()),'largest_member_bytes':max(m['bytes'] for m in members.values()),'review_contexts_verified':len(reviews_seen),'review_contexts':[{'review_path':name,'gate_root':root or 'repository_root'} for name,root in sorted(reviews_seen)],'qualification':'closure_constructed_pending_independent_bridge_review'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--output',type=Path,required=True);parser.add_argument('--closure-only',action='store_true');args=parser.parse_args()
    repo=args.repo.resolve();output=args.output.resolve();require(not output.exists(),'preserve prior evidence pack');output.mkdir(parents=True)
    closure=make_closure(repo);write_new(output/'closure.json',closure)
    if args.closure_only:print(json.dumps({k:v for k,v in closure.items() if k not in ('members','roots','review_contexts')},indent=2));return
    archive=output/'original84.tar.xz'
    # Canonical regular-member paths and deterministic tar metadata. XZ delta4
    # is lossless and preserves all four bytes of every raw uint32 word.
    with archive.open('xb') as destination:
        process=subprocess.Popen(['xz','-T2','--delta=dist=4','--lzma2=preset=3'],stdin=subprocess.PIPE,stdout=destination)
        try:
            with tarfile.open(fileobj=process.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for name,item in closure['members'].items():
                    path=relative(repo,name);require(path.stat().st_size==item['bytes'],'file length changed during pack')
                    info=tarfile.TarInfo(name);info.size=item['bytes'];info.mode=0o644;info.uid=info.gid=0;info.mtime=0
                    with path.open('rb') as data:tar.addfile(info,data)
            process.stdin.close();require(process.wait()==0,'XZ compression failed')
        except BaseException:
            process.stdin.close();process.terminate();process.wait();raise
    # Verify every source byte after packaging; simultaneous changes invalidate
    # this pack construction, not the original B3 or old GPU results.
    for name,item in closure['members'].items():require(sha(relative(repo,name))==item['sha256'],'source changed during packaging: '+name)
    index={'schema_version':1,'namespace':'repository_relative_v1','archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'members':closure['members'],'roots':closure['roots']};write_new(output/'index.json',index)
    receipt={'status':'pack_generated_pending_independent_full_verification','archive_sha256':index['archive_sha256'],'index_sha256':sha(output/'index.json'),'closure_sha256':sha(output/'closure.json'),'archive_bytes':index['archive_bytes'],'uncompressed_bytes':closure['uncompressed_bytes'],'member_count':closure['member_count'],'GPU_execution':False,'bridge_qualified':False};write_new(output/'construction.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()

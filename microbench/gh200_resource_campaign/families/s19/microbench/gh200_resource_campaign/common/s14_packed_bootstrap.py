"""Initialize fixed original84 descriptor/code observer from pack bytes only.

The canonical closure archive sorts global contract/profiles/coverage before
any original84 artifact. A different order is explicitly rejected, never
quietly skips arrays. No original expanded tree or package code is executed.
"""
import hashlib
from common.suite_io import require
from common.packed_evidence import json_object
from common.s14_packed_observer import S14EvidenceObserver
from auditors.tma_bulk import artifact_layouts,validate_contract
from pack_s14_evidence import REVIEW,REVIEW_SHA,COVERAGE,COVERAGE_SHA,CONTRACT,PROFILES


def metadata_whitelist(closure):
    small={REVIEW,COVERAGE,CONTRACT,PROFILES}
    small.update(c['review_path'] for c in closure['review_contexts'])
    require(len(closure['roots']['runs'])==84,'finite84 run manifests')
    for manifest in closure['roots']['runs']:
        require(manifest.endswith('/validation_manifest.json'),'explicit regular-member run root')
        root=manifest.removesuffix('/validation_manifest.json')
        for local in ('validation_spec.json','validation_manifest.json','attempts/attempt_00/raw.jsonl','attempts/attempt_00/receipt.json','environment/device.json','environment/initial.json','build/dependencies.json','build/shared_libraries.json','build/compile.stderr'):
            small.add(root+'/'+local)
        small.update(root+'/snapshot/repo/'+name for name in (CONTRACT,PROFILES,'microbench/gh200_resource_campaign/probes/tma_bulk.cu'))
    require(all(name in closure['members'] and closure['members'][name]['bytes']<=16*1024*1024 for name in small),'complete bounded metadata whitelist')
    require(sum(closure['members'][name]['bytes'] for name in small)<=64*1024*1024,'bounded metadata aggregate')
    return sorted(small)


class PackedOriginal84Bootstrap:
    def __init__(self,closure):
        self.closure=closure;self.metadata={};self.buffers={};self.inner=None
        require(closure['roots']['review']==REVIEW and closure['roots']['coverage']==COVERAGE
                and closure['roots']['contract']==CONTRACT and closure['roots']['profiles']==PROFILES,'fixed original global roots')
        require(closure['members'][REVIEW]['sha256']==REVIEW_SHA and closure['members'][COVERAGE]['sha256']==COVERAGE_SHA,'immutable parent B3/coverage')

    def initialize(self):
        contract=validate_contract(self.metadata[CONTRACT]);profiles=self.metadata[PROFILES];coverage=self.metadata[COVERAGE]
        require(len(coverage['records'])==84,'complete fixed coverage')
        cases={c['id']:c for c in contract['cases']};byprofile={p['id']:p for p in profiles['profiles']};descriptors={};sass={}
        for record in coverage['records']:
            case=cases[record['case_id']];profile=byprofile[record['profile_id']];root=record['run_path'];blocks=record['blocks'];seed=record['seed'];steps=profile['target_iterations'][0]
            layouts=artifact_layouts(case,profile,blocks,seed);require(set(layouts)==set(record['artifacts_sha256']),'full original descriptor artifact set')
            for role,shape in layouts.items():
                name=root+'/attempts/attempt_00/'+role;member=self.closure['members'][name];require(member['sha256']==record['artifacts_sha256'][role],'descriptor original member binding')
                require(name not in descriptors,'no duplicate artifact descriptor')
                descriptors[name]={'run':root,'role':role,'shape':shape,'bytes':member['bytes'],'blocks':blocks,'words':case['parameters']['payload_bytes']//4,'iterations':steps,'seed':seed,'release':record['profile_id']=='bulk_source_release_one_request_v1','g2s':case['parameters']['direction']=='gmem_to_smem'}
            name=root+'/build/sass.stdout';sass[name]=self.closure['members'][name]['bytes']
        require(len(descriptors)==312 and len(sass)==84,'finite full artifact/code set')
        self.inner=S14EvidenceObserver(descriptors,sass,contract)

    def __call__(self,name,offset,chunk):
        if name in (CONTRACT,PROFILES,COVERAGE):
            require(name not in self.metadata,'bootstrap member repeated')
            data=self.buffers.setdefault(name,bytearray());require(offset==len(data),'contiguous bootstrap metadata');data.extend(chunk)
            size=self.closure['members'][name]['bytes'];require(size<=16*1024*1024 and len(data)<=size,'bounded bootstrap bytes')
            if len(data)==size:
                require(hashlib.sha256(data).hexdigest()==self.closure['members'][name]['sha256'],'bootstrap exact metadata SHA')
                self.metadata[name]=json_object(bytes(data));del self.buffers[name]
                if all(key in self.metadata for key in (CONTRACT,PROFILES,COVERAGE)):self.initialize()
        if self.inner is not None:self.inner(name,offset,chunk)
        elif '/tma_bulk/short-v1-' in name and '/attempts/attempt_00/' in name and name.endswith('.u32le'):
            raise ValueError('canonical bootstrap metadata must precede every original numerical array')

    def finish(self):
        require(self.inner is not None and not self.buffers,'complete package-only bootstrap')
        result=self.inner.finish();result['descriptor_source']='only package global contract/profile/immutable original coverage bytes';result['original_expanded_tree_required']=False
        return result

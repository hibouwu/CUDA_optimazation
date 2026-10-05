"""Fixed original84 descriptors from a public already-verified preparation view."""
from common.suite_io import require
from common.packed_evidence import json_object
from pack_s14_evidence import REVIEW,REVIEW_SHA,COVERAGE,COVERAGE_SHA,CONTRACT,PROFILES


def metadata_names(closure):
    names={REVIEW,COVERAGE,CONTRACT,PROFILES}
    names.update(c['review_path'] for c in closure['review_contexts'])
    require(len(closure['roots']['runs'])==84,'fixed84 run manifest roots')
    for manifest in closure['roots']['runs']:
        require(manifest.endswith('/validation_manifest.json'),'run root is an exact regular manifest member')
        root=manifest.removesuffix('/validation_manifest.json')
        names.update(root+'/'+local for local in ['validation_spec.json','validation_manifest.json','attempts/attempt_00/raw.jsonl','attempts/attempt_00/receipt.json','environment/device.json','environment/initial.json','build/dependencies.json','build/shared_libraries.json','build/compile.stderr'])
        names.update(root+'/snapshot/repo/'+local for local in [CONTRACT,PROFILES,'microbench/gh200_resource_campaign/probes/tma_bulk.cu'])
    require(all(name in closure['members'] and closure['members'][name]['bytes']<=16*1024*1024 for name in names),'bounded required metadata members')
    require(sum(closure['members'][name]['bytes'] for name in names)<=64*1024*1024,'aggregate cache bound')
    return sorted(names)


def prepare_descriptors(view,closure):
    require(view.file_sha256(REVIEW)==REVIEW_SHA and view.file_sha256(COVERAGE)==COVERAGE_SHA,'immutable original parent roots')
    coverage=view.read_json(COVERAGE);require(len(coverage['records'])==84,'fixed full coverage')
    descriptors={};sass={}
    for record in coverage['records']:
        root=record['run_path'];spec=view.read_json(root+'/validation_spec.json');contract=view.read_json(root+'/'+spec['contract_path']);case=next(c for c in contract['cases'] if c['id']==record['case_id'])
        rows=[json_object(line) for line in view.read_bytes(root+'/attempts/attempt_00/raw.jsonl').splitlines()];require(len(rows)==2,'whole original raw');row=rows[1]
        for artifact in row['checks'][0]['output_artifacts']:
            name=root+'/attempts/attempt_00/'+artifact['path'];member=closure['members'][name];require(name not in descriptors and member['sha256']==artifact['sha256']==record['artifacts_sha256'][artifact['path']],'complete unique original artifact identity')
            descriptors[name]={'run':root,'role':artifact['path'],'shape':artifact['shape'],'bytes':member['bytes'],'blocks':row['blocks'],'words':case['parameters']['payload_bytes']//4,'iterations':row['target_launches'][0]['iterations'],'seed':record['seed'],'release':record['profile_id']=='bulk_source_release_one_request_v1','g2s':case['parameters']['direction']=='gmem_to_smem'}
        name=root+'/build/sass.stdout';sass[name]=closure['members'][name]['bytes']
    require(len(descriptors)==312 and len(sass)==84,'full312 arrays and84 code members')
    return descriptors,sass,view.read_json(CONTRACT)

"""Real small tar from an actual CPU host run; GPU registry is a labeled fixture.

This never executes a CUDA kernel and never creates GPU/B3 qualification.
"""
from pathlib import Path
import hashlib,json,os,shutil,tarfile
from common.suite_io import bounded,process_ok
from runners.s16_short_controller import SUITE,coordinates
from runners.validation_diagnostic import POLICY
from auditors import tma_stage_request_validation_v1 as adapter
from test_tma_stage_request_v1 import StageTests,ROOT,CONTRACT,PROFILE


def payload(repo):
    StageTests.setUpClass()
    try:
        points=coordinates(CONTRACT,{'family':'tma_stage_request','profiles':[PROFILE]})
        point=next(p for p in points if p['case_id']=='gmem_to_smem_16kib_s1_r1_one_cta')
        run=SUITE+'/tma_stage_request/short-v1-'+point['case_id']+'--'+point['profile_id']
        root=repo/run;attempt=root/'attempts/attempt_00';attempt.mkdir(parents=True)
        (root/'binary').mkdir();binary=root/'binary/probe';shutil.copy2(StageTests.bin,binary)
        argv=[str(binary),'validate-only',point['case_id'],point['profile_id'],'3']
        old=dict(os.environ);os.environ['S16_LAUNCH_LOG']=str(repo/'CPU-fixture-launches')
        try:receipt=bounded(argv,attempt,attempt/'raw.jsonl',attempt/'stderr',30)
        finally:os.environ.clear();os.environ.update(old)
        assert process_ok(receipt),receipt
        raw=(attempt/'raw.jsonl').read_bytes();device,row=[json.loads(x) for x in raw.decode().splitlines()]
        case=next(c for c in CONTRACT['cases'] if c['id']==point['case_id']);result=adapter.validate_validation(device,row,case,PROFILE,3)
        encode=lambda obj:json.dumps(obj,sort_keys=True).encode()
        digest=lambda data:hashlib.sha256(data).hexdigest()
        contract_path='snapshot/repo/microbench/gh200_resource_campaign/contracts/tma_stage_request_short_v1.draft.json'
        profile_path='snapshot/repo/microbench/gh200_resource_campaign/contracts/tma_stage_request_validation_profiles_v1.draft.json'
        initial={'uuid':device['uuid'],'execution_uid':os.geteuid(),'host':'CPU-fixture-host','job':'CPU-fixture-job'}
        registration={'schema_version':2,'registry_id':'CPU-fixture-registration','gpu_uuid':device['uuid'].lower(),'state':'cleanup_confirmed','pid':receipt['pid'],'pgid':receipt['pgid'],'argv':argv,'controller':{'host':initial['host'],'boot_id':'CPU-fixture-boot','uid':os.geteuid()},'cleanup_receipt':dict(receipt),'purpose':'CPU fixture only; not a real GPU registration'}
        receipt['gpu_process_registration']=registration
        spec={'stage':'S16','family':'tma_stage_request','kind':'case_diagnostic','selection_scope':'case_diagnostic','performance_eligible':False,'family_B3_eligible':False,**{k:point[k] for k in ('case_id','profile_id','seed')},'contract_path':contract_path,'profiles_path':profile_path,'execution_root':str(root),'binary_sha256':digest(binary.read_bytes()),'fixture_kind':'actual CPU host output; no GPU'}
        evidence={'result':result,'receipt':'attempts/attempt_00/receipt.json','receipt_sha256':digest(encode(receipt)),'raw_sha256':digest(raw),'output_artifacts':{x['path']:x['sha256'] for c in row['checks'] for x in c['output_artifacts']}}
        data={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
        data.update({'validation_spec.json':encode(spec),'diagnostic_summary.json':encode({'status':'case_diagnostic_passed','case_id':point['case_id'],'performance_eligible':False,'family_B3_eligible':False,'evidence':evidence}),'validation_state.json':encode({'status':'case_diagnostic_passed'}),'environment/device.json':encode(device),'environment/initial.json':encode(initial),contract_path:encode(CONTRACT),profile_path:encode({'schema_version':1,'family':'tma_stage_request','profiles':[PROFILE]}),'snapshot/repo/'+POLICY:(ROOT.parents[1]/POLICY).read_bytes(),'attempts/attempt_00/receipt.json':encode(receipt),'attempts/attempt_00/raw.jsonl.registry.json':encode(registration),'attempts/attempt_00/raw.jsonl.active.json':encode({'pid':receipt['pid'],'pgid':receipt['pgid'],'argv':argv,'registry_id':registration['registry_id'],'gpu_uuid':device['uuid']})})
        shutil.rmtree(root)
        return point,run,data
    finally:StageTests.tearDownClass()


def pack(repo,point,run,data,identity,tag):
    """Mutations rebuild the entire closure so rejection cannot rely on stale SHA."""
    import io
    from common.suite_io import digest as identity_digest
    data=dict(data);data.pop('validation_manifest.json',None)
    digest=lambda b:hashlib.sha256(b).hexdigest()
    manifest={n:digest(b) for n,b in data.items() if n!='validation_state.json'}
    data['validation_manifest.json']=json.dumps(manifest,sort_keys=True).encode()
    folder=repo/tag;folder.mkdir();archive=folder/'run.tar.xz'
    members={run+'/'+n:{'bytes':len(b),'sha256':digest(b)} for n,b in data.items()}
    with tarfile.open(archive,'w:xz',preset=1) as stream:
        for name,b in data.items():info=tarfile.TarInfo(run+'/'+name);info.size=len(b);stream.addfile(info,io.BytesIO(b))
    ah=digest(archive.read_bytes());mh=digest(data['validation_manifest.json'])
    index={'schema_version':1,'namespace':'repository_relative_v1','archive_sha256':ah,'archive_bytes':archive.stat().st_size,'members':members,'roots':{'original_manifest':run+'/validation_manifest.json'}}
    closure={'schema_version':1,'run_path':run,'manifest_sha256':mh,'members':members}
    entry={'coordinate':point,'state':'archived_verified','run_path':run,'manifest_sha256':mh,'family_identity_sha256':identity_digest(identity)}
    for key,file,obj in [('index','index.json',index),('closure','closure.json',closure)]:p=folder/file;p.write_text(json.dumps(obj));entry[key]={'path':str(p.relative_to(repo)),'sha256':digest(p.read_bytes())}
    entry['archive']={'path':str(archive.relative_to(repo)),'sha256':ah}
    for key,location in [('archive_verification','remote'),('offhost_receipt','offhost')]:
        receipt={'schema_version':1,'status':'full_members_verified','location':location,'archive_sha256':ah,'index_sha256':entry['index']['sha256'],'closure_sha256':entry['closure']['sha256'],'run_path':run,'manifest_sha256':mh,'members':members,'member_count':len(members),'uncompressed_bytes':sum(x['bytes'] for x in members.values()),'verification':'complete_decode_all_members_sha256'}
        p=folder/(key+'.json');p.write_text(json.dumps(receipt));entry[key]={'path':str(p.relative_to(repo)),'sha256':digest(p.read_bytes())}
    return entry

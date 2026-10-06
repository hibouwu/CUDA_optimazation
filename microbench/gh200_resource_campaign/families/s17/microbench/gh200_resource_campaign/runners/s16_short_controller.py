"""Finite S16 controller A draft: CPU plan/footprint/recovery only.

Actual run stays disabled pending independent controller A and storage/source B.
The existing diagnostic and packed transports retain their own qualifications.
"""
from pathlib import Path
import argparse,json,time
from common.suite_io import require,read_json,sha,relative,process_ok,digest
from auditors.tma_stage_request_validation_v1 import coordinate,profile_check
from auditors.tma_stage_request_sass_baseline_v1 import BASELINE
SUITE='results/gh200_resource_campaign/20261001-resource-suite-v2'
CORE_CAMPAIGN='microbench/gh200_resource_campaign'
CONTRACT='contracts/tma_stage_request_short_v1.draft.json'
PROFILES='contracts/tma_stage_request_validation_profiles_v1.draft.json'
MANIFEST='contracts/tma_stage_request_validation_adapter_v1.draft.json'
MAX_FIRST='gmem_to_smem_16kib_s1_r4_all_gpu'
FAMILY='tma_stage_request'
DEFAULT_RESIDENT_BYTES=6*1024**3
METADATA_BYTES=64*1024**2

def coordinates(contract,profiles):
    require(contract['family']==profiles['family']==FAMILY,'S16 profile family')
    require(len(profiles['profiles'])==1,'S16 one grouped profile')
    profile=profiles['profiles'][0];profile_check(profile)
    require(profile['required_seed']==3 and profile['target_iterations']==[1,2,5,33],'S16 fixed4 launches and seed3')
    require(len(contract['cases'])==36 and len({c['id'] for c in contract['cases']})==36,'S16 fixed36 nominal coordinates')
    rows=[]
    for contract_index,case in enumerate(contract['cases']):
        coordinate(case)
        rows.append({'contract_index':contract_index,'case_id':case['id'],'profile_id':profile['id'],'seed':3})
    require(sum(c['case_id']==MAX_FIRST for c in rows)==1,'S16 max-first coordinate present')
    ordered=[c for c in rows if c['case_id']==MAX_FIRST]+[c for c in rows if c['case_id']!=MAX_FIRST]
    return [dict(row,index=index) for index,row in enumerate(ordered)]

def device_capacities(device):
    for key in ('sms','smem_per_sm_bytes','smem_per_cta_optin_bytes','registers_per_sm'):
        require(type(device[key])is int and device[key]>0,'S16 device capacity integer')

def artifact_bytes(case,profile,device):
    profile_check(profile);device_capacities(device);g,s,r,scope=coordinate(case)
    shared=s*r*16384+8*s+32
    registers=BASELINE[f"ts_{'g2s' if g else 's2g'}_s{s}_r{r}"]['registers_per_thread']
    if shared>device['smem_per_cta_optin_bytes']:
        return {'capacity_class':'resource_reject_before_launch','dynamic_smem_bytes':shared,
                'blocks_upper_bound':0,'target_launches':0,'artifact_bytes_upper_bound':0,
                'actual_occupancy_claim':False}
    occ=min(16,device['smem_per_sm_bytes']//shared,device['registers_per_sm']//(registers*128),2048//128)
    require(occ>0,'S16 unexpected zero necessary capacity; protocol review required')
    b=1 if scope=='one_cta' else device['sms']*min(4,occ)
    items=sum(profile['target_iterations']);launches=len(profile['target_iterations'])
    trace=b*items*r*16384;final=launches*b*s*r*16384
    ring=launches*(b*32*r*16384+32)
    lifecycle=b*items*8*2*4;counts=launches*b*12*2*4;stamps=launches*b*5*2*4
    return {'capacity_class':'necessary_capacity_only','dynamic_smem_bytes':shared,
            'registers_per_thread':registers,'blocks_upper_bound':b,
            'occupancy_necessary_capacity_upper_bound':occ,'target_launches':4,
            'trace_bytes':trace,'final_shared_slot_bytes':final,'ring_and_guard_bytes':ring,
            'lifecycle_bytes_including_opaque_token':lifecycle,'counts_bytes':counts,'stamp_bytes':stamps,
            'artifact_bytes_upper_bound':trace+final+ring+lifecycle+counts+stamps,
            'opaque_token_preserved_domain_only_words':2*b*items if g else 0,
            'transport_bytes':b*items*r*16384,'actual_occupancy_claim':False}

def footprint(contract,profiles,device):
    points=coordinates(contract,profiles);cases={c['id']:c for c in contract['cases']};profile=profiles['profiles'][0]
    rows=[c|artifact_bytes(cases[c['case_id']],profile,device) for c in points]
    legal=[c for c in rows if c['target_launches']];reject=[c for c in rows if not c['target_launches']]
    require(len(legal)==32 and len(reject)==4 and all('_s4_r4_' in c['case_id'] for c in reject),'S16 expected32legal/4actualcapacityreject; changed device capability requires review')
    require(rows[0]['artifact_bytes_upper_bound']==max(c['artifact_bytes_upper_bound'] for c in legal),'S16 bound max-first must remain largest legal process')
    return {'coordinates':rows,'nominal_cases':36,'legal_processes':32,'legal_target_launches':128,
            'capacity_reject_cases':4,'capacity_reject_target_launches':0,
            'total_artifact_bytes_upper_bound':sum(c['artifact_bytes_upper_bound'] for c in legal),
            'max_artifact_bytes_upper_bound':rows[0]['artifact_bytes_upper_bound'],
            'default_resident_window_bytes':DEFAULT_RESIDENT_BYTES,'actual_occupancy_claim':False}

def device_binding(repo,device_path,binding,allocation,source_review,source_review_path):
    """Pure admission check for a separately executed reviewed read-only query."""
    from runners.environment import environment_identity
    repo=Path(repo).resolve();device=read_json(device_path);env=environment_identity(allocation)
    require(binding['schema_version']==1 and binding['device_sha256']==sha(device_path),'S16 device query bytes')
    require(device['uuid'].lower()==env['uuid'].lower() and binding['environment']==env,'S16 actual allocated UUID/environment')
    require(binding['allocation_job']==allocation['job'],'S16 query bound to actual Slurm allocation')
    require(binding['source_review_sha256']==sha(source_review_path),'S16 actual source-B identity')
    approved=[h for n,h in source_review['gate_files'].items() if n.endswith('/diagnostics/tma_stage_request/probe')]
    require(len(approved)==1 and binding['binary_sha256']==approved[0],'S16 independently reviewed actual binary')
    stdout=binding['stdout'];require(set(stdout)=={'path','sha256'} and sha(relative(repo,stdout['path']))==stdout['sha256'] and read_json(relative(repo,stdout['path']))==device,'S16 preserved actual device stdout')
    receipt=binding['receipt'];require(type(receipt.get('returncode'))is int and process_ok(receipt) and type(receipt['timeout_seconds'])is int and receipt['timeout_seconds']==30 and len(receipt['argv'])==2 and receipt['argv'][1]=='device' and receipt['stdout_sha256']==stdout['sha256'],'S16 bounded read-only query only')
    binary=Path(receipt['argv'][0]);require(binary.is_absolute() and binary.is_file() and not binary.is_symlink() and sha(binary)==binding['binary_sha256'],'S16 actual query executable bytes')
    require(device['type']=='device' and device['cc']=='9.0' and 'GH200' in device['name'],'S16 raw GH200 device identity')
    registration=receipt['gpu_process_registration'];require(registration['state']=='cleanup_confirmed' and registration['gpu_uuid'].lower()==env['uuid'].lower() and registration['argv']==receipt['argv'],'S16 UUID query registration/cleanup')
    return device

def family_identity(environment,device_sha256,source_files,controller_files):
    return {'schema_version':1,'family':FAMILY,'environment':environment,'device_sha256':device_sha256,
            'source_files_sha256':source_files,'controller_files_sha256':controller_files}

def quota_budget(resident,next_artifacts,headroom,window=DEFAULT_RESIDENT_BYTES):
    for value in (resident,next_artifacts,headroom,window):require(type(value)is int and value>=0,'S16 byte budget integer')
    next_bytes=next_artifacts+METADATA_BYTES
    seal=resident+next_bytes+(resident+next_bytes)//100+METADATA_BYTES
    return {'allow':resident+next_bytes<=window and next_bytes+seal<=headroom,
            'estimated_new_bytes':next_bytes,'worst_case_seal_bytes':seal,'quota_headroom':headroom,
            'status':'quota_admit_before_target' if resident+next_bytes<=window and next_bytes+seal<=headroom else 'quota_checkpoint_before_next_target'}

def next_action(root,entry,repo,point,identity):
    if entry is not None:
        require(entry['coordinate']==point and entry['family_identity_sha256']==digest(identity),'S16 changed coordinate/source/controller/UUID cannot resume')
        if entry['state']=='resource_reject_before_launch':
            require('_s4_r4_' in point['case_id'] and entry['target_launches']==0 and not Path(root).exists(),'S16 capacity reject must be one of four S4R4 points with no run/target')
            return 'verify_capacity_terminal'
        if entry['state']=='archived_verified':
            validate_archived(entry,repo,point);return 'archive_audit'
        require(entry['state'] not in ('failed','started_unknown'),'S16 failed/uncertain started evidence requires investigation')
    root=Path(root)
    if not root.exists():
        require(entry is None or entry['state']=='pending','S16 missing measured run never permits new target')
        return 'run'
    require((root/'validation_state.json').is_file(),'S16 partial initialize requires offline recovery')
    state=read_json(root/'validation_state.json')['status'];attempts=list((root/'attempts').glob('attempt_*'))
    if state=='case_diagnostic_passed':return 'audit'
    if state in ('pending','checkpoint') and not attempts:return 'resume'
    raise ValueError('S16 started/failed attempt cannot rerun automatically')
def validate_archived(entry,repo,coordinate):
    require(entry['coordinate']==coordinate and entry['state']=='archived_verified','archived fixed coordinate')
    run_path=SUITE+'/tma_stage_request/short-v1-'+coordinate['case_id']+'--'+coordinate['profile_id']
    require(entry.get('run_path')==run_path,'archive must preserve fixed original run path')
    for key in ('archive','index','closure','offhost_receipt','archive_verification'):
        ref=entry[key];require(set(ref)=={'path','sha256'},'archive identity ref fields')
        require(sha(relative(repo,ref['path']))==ref['sha256'],'archived evidence identity changed')
    index=read_json(relative(repo,entry['index']['path']));closure=read_json(relative(repo,entry['closure']['path']))
    require(set(closure)=={'schema_version','run_path','manifest_sha256','members'} and closure['schema_version']==1,'finite archive closure schema')
    require(closure['run_path']==run_path and closure['manifest_sha256']==entry['manifest_sha256'],'original manifest identity')
    require(index['members']==closure['members'] and index['archive_sha256']==entry['archive']['sha256'],'full archive closure')
    from common.packed_evidence import PackedEvidence,logical_path
    from runners.validation_diagnostic import validation_envelope,POLICY
    from auditors import tma_stage_request_validation_v1 as adapter
    members=closure['members'];require(all(n.startswith(run_path+'/') for n in members),'one original run namespace')
    require(run_path+'/failure_summary.json' not in members and run_path+'/failure_manifest.json' not in members,'failed diagnostic artifacts cannot be archived as passed')
    attempt_roots={n[len(run_path+'/attempts/'):].split('/')[0] for n in members if n.startswith(run_path+'/attempts/')}
    require(len(attempt_roots)==1 and next(iter(attempt_roots))=='attempt_00','exactly one successful first attempt; no extra/failed/interrupted attempts')
    attempt='attempts/attempt_00';raw_name=run_path+'/'+attempt+'/raw.jsonl';receipt_name=run_path+'/'+attempt+'/receipt.json'
    registry_name=raw_name+'.registry.json';active_name=raw_name+'.active.json';stderr_name=run_path+'/'+attempt+'/stderr'
    names=[run_path+'/'+n for n in ('validation_manifest.json','validation_spec.json','diagnostic_summary.json','validation_state.json','environment/device.json','environment/initial.json')]
    registry_names=[n for n in members if n.endswith('.registry.json')]
    required=names+[raw_name,receipt_name,registry_name,active_name,stderr_name,run_path+'/binary/probe']
    require(all(n in members for n in required),'complete spec/device/single attempt/process evidence')
    bundle=PackedEvidence(relative(repo,entry['archive']['path']),relative(repo,entry['index']['path']),expected_index_sha256=entry['index']['sha256'],expected_archive_sha256=entry['archive']['sha256'],expected_closure=members)
    small=[n for n in members if n.endswith('.json') or n.endswith('.jsonl')]
    verified=bundle.verify_all(small_objects=small)
    # One complete decode transaction covers metadata and binary hashes. Numeric
    # arrays are preserved/hashed, but numerical B3 replay is not performed here.
    with bundle.verified_transaction(small) as view:
        manifest=view.read_json(names[0]);require(isinstance(manifest,dict) and manifest,'nonempty original manifest')
        require(view.file_sha256(names[0])==entry['manifest_sha256'],'bound manifest bytes')
        expected={names[0],names[3]}
        for name,digest_ in manifest.items():
            full=run_path+'/'+logical_path(name);expected.add(full)
            require(full in members and view.file_sha256(full)==digest_,'original manifest member changed')
        require(expected==set(members),'exact complete original manifest closure')
        spec=view.read_json(names[1]);summary=view.read_json(names[2]);state=view.read_json(names[3])
        require(spec['stage']=='S16' and spec['family']==FAMILY and spec['kind']=='case_diagnostic' and spec['selection_scope']=='case_diagnostic','S16 original diagnostic role')
        require(spec['performance_eligible']is False and spec['family_B3_eligible']is False,'archive cannot grant performance/B3')
        for key in ('case_id','profile_id','seed'):require(spec[key]==coordinate[key],'original spec coordinate')
        require(state['status']=='case_diagnostic_passed' and summary['status']=='case_diagnostic_passed' and summary['case_id']==coordinate['case_id'],'original success state')
        require(summary['performance_eligible']is False and summary['family_B3_eligible']is False,'summary diagnostic-only role')
        contract=view.read_json(run_path+'/'+logical_path(spec['contract_path']))
        profiles=view.read_json(run_path+'/'+logical_path(spec['profiles_path']))
        selected=coordinates(contract,profiles);require(coordinate in selected,'snapshot finite coordinate')
        case=next(c for c in contract['cases'] if c['id']==coordinate['case_id']);profile=profiles['profiles'][0]
        policy=view.read_json(run_path+'/snapshot/repo/'+POLICY)
        device=view.read_json(names[4]);initial=view.read_json(names[5])
        require(device['uuid'].lower()==initial['uuid'].lower(),'actual device matches original allocation')
        rows=strict_archive_rows(view.read_bytes(raw_name));require(rows[0]==device,'raw device equals original environment/device.json')
        row=rows[1]
        validation_envelope(row,case,profile,coordinate['seed'],policy)
        result=adapter.validate_validation(device,row,case,profile,coordinate['seed'])
        require(row['target_launches']==[{'launch_index':n,'iterations':i,'input_profile':'nonuniform_uint32_stage_request_pattern','threads':128,'blocks':row['blocks']} for n,i in enumerate((1,2,5,33))],'exact four ordered target metadata')
        receipt=view.read_json(receipt_name);execution=Path(spec['execution_root'])
        require(execution.is_absolute() and execution.as_posix().endswith('/'+run_path),'original execution namespace')
        require(view.file_sha256(run_path+'/binary/probe')==spec['binary_sha256'],'original diagnostic binary identity')
        argv=[str(execution/'binary/probe'),'validate-only',coordinate['case_id'],coordinate['profile_id'],str(coordinate['seed'])]
        require(receipt['argv']==argv and type(receipt['returncode'])is int and process_ok(receipt) and type(receipt['timeout_seconds'])is int and receipt['timeout_seconds']==30,'one successful bounded fixed target process')
        require(receipt['stdout_sha256']==view.file_sha256(raw_name) and receipt['stderr_sha256']==view.file_sha256(stderr_name),'process output hashes')
        registration=receipt['gpu_process_registration'];require(view.read_json(registry_name)==registration,'exact target registry bytes')
        active=view.read_json(active_name)
        for key in ('pid','pgid','argv','registry_id'):require(active[key]==registration[key],'historical launch record links to closed registration')
        require(active['gpu_uuid'].lower()==device['uuid'].lower(),'historical launch UUID')
        intervals=[];registry_ids=set();target_records=0
        for name in registry_names:
            reg=view.read_json(name);require(reg['state']=='cleanup_confirmed' and reg['gpu_uuid'].lower()==device['uuid'].lower(),'all archived GPU registrations closed on actual UUID')
            require(reg['registry_id'] not in registry_ids,'duplicate GPU registration');registry_ids.add(reg['registry_id'])
            clean=reg['cleanup_receipt'];require(type(clean['returncode'])is int and process_ok(clean),'failed/unconfirmed archived registered process')
            require(type(clean['pid'])is int and clean['pid']>0 and clean['pgid']==clean['pid'] and reg['pid']==clean['pid'] and reg['pgid']==clean['pgid'],'registered PID/PGID identity')
            require(type(clean['host_start_ns'])is int and type(clean['host_stop_ns'])is int and 0<=clean['host_start_ns']<=clean['host_stop_ns'],'registered process interval')
            require(reg['argv']==clean['argv'] and (reg['argv']==argv or reg['argv']==[argv[0],'device']),'only this target or original read-only device query')
            if reg['argv']==argv:target_records+=1;require(name==registry_name,'one target registration only')
            origin=reg['controller'];require(origin['uid']==initial['execution_uid'] and origin['host']==initial['host'] and origin['boot_id'],'original registration controller identity')
            intervals.append((origin['host'],origin['boot_id'],clean['host_start_ns'],clean['host_stop_ns']))
        require(target_records==1 and registration['cleanup_receipt']=={k:v for k,v in receipt.items() if k!='gpu_process_registration'},'exact successful target cleanup receipt')
        for n,first in enumerate(intervals):
            for second in intervals[n+1:]:
                require(first[:2]!=second[:2] or first[3]<=second[2] or second[3]<=first[2],'illegal overlapping registered GPU intervals')
        artifact_hashes={}
        for check in row['checks']:
            for item in check['output_artifacts']:
                require(set(item)=={'path','sha256','dtype','shape','evidence_kind'} and item['dtype']=='uint32' and all(type(x)is int and x>0 for x in item['shape']),'full uint32 artifact metadata')
                full=run_path+'/'+attempt+'/'+logical_path(item['path']);count=1
                for extent in item['shape']:count*=extent
                require(full in members and members[full]['bytes']==count*4 and view.file_sha256(full)==item['sha256'],'all declared artifact bytes/hash/shape')
                require(item['path'] not in artifact_hashes,'artifact alias between target checks');artifact_hashes[item['path']]=item['sha256']
        evidence=summary['evidence']
        require(evidence['result']==result and evidence['receipt']==attempt+'/receipt.json' and evidence['receipt_sha256']==view.file_sha256(receipt_name) and evidence['raw_sha256']==view.file_sha256(raw_name) and evidence['output_artifacts']==artifact_hashes,'summary matches exact single successful attempt')
    require(bundle.transaction_receipt['closed']is True,'full archive transaction closed')
    for key,location in [('archive_verification','remote'),('offhost_receipt','offhost')]:
        check=read_json(relative(repo,entry[key]['path']))
        require(set(check)=={'schema_version','status','location','archive_sha256','index_sha256','closure_sha256','run_path','manifest_sha256','members','member_count','uncompressed_bytes','verification'},'complete verification receipt schema')
        require(check['schema_version']==1 and check['status']=='full_members_verified' and check['location']==location and check['verification']=='complete_decode_all_members_sha256','explicit full member verification')
        for field,refkey in [('archive_sha256','archive'),('index_sha256','index'),('closure_sha256','closure')]:require(check[field]==entry[refkey]['sha256'],'receipt archive identity')
        require(check['run_path']==run_path and check['manifest_sha256']==entry['manifest_sha256'] and check['members']==members and type(check['member_count'])is int and check['member_count']==verified['members'] and type(check['uncompressed_bytes'])is int and check['uncompressed_bytes']==verified['uncompressed_bytes'],'complete original member receipt')
    return 'archived_verified'

def strict_archive_rows(data):
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate raw JSON field');out[k]=v
        return out
    def nonfinite(value):raise ValueError('nonfinite archived raw JSON')
    lines=data.decode('utf-8').splitlines();require(len(lines)==2,'exact device and validation rows')
    rows=[json.loads(line,object_pairs_hook=pairs,parse_constant=nonfinite) for line in lines]
    require(rows[0].get('type')=='device' and rows[1].get('type')=='validation','diagnostic raw roles')
    return rows

def quota_headroom(repo,path,env):
    receipt=read_json(path)
    require(set(receipt)=={'schema_version','status','queried_unix_ns','execution_uid','filesystem_root','hard_bytes','used_bytes','reserve_bytes','query_stdout'},'quota receipt exact fields')
    require(receipt['schema_version']==1 and receipt['status']=='current_quota_query_verified','quota query must be verified by reviewed wrapper')
    require(type(receipt['queried_unix_ns']) is int and 0<=time.time_ns()-receipt['queried_unix_ns']<=60_000_000_000,'quota receipt freshness60seconds')
    require(receipt['execution_uid']==env['execution_uid'],'quota belongs to same user')
    fs=Path(receipt['filesystem_root']).resolve();require(fs.is_absolute() and fs!=Path('/') and repo.resolve().is_relative_to(fs),'quota mount covers this deployment')
    for key in ('hard_bytes','used_bytes','reserve_bytes'):require(type(receipt[key]) is int and receipt[key]>=0,'quota byte integer')
    require(receipt['hard_bytes']>receipt['used_bytes'] and receipt['reserve_bytes']>=512*1024**2,'quota must retain at least512MiB logs/output margin')
    query=receipt['query_stdout'];require(set(query)=={'path','sha256'} and sha(relative(repo,query['path']))==query['sha256'],'exact bound original quota stdout')
    return receipt['hard_bytes']-receipt['used_bytes']-receipt['reserve_bytes']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','footprint','run'))
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=36)
    parser.add_argument('--device-json',type=Path)
    args=parser.parse_args();repo=args.repo.resolve();code=repo/CORE_CAMPAIGN
    contract=read_json(code/CONTRACT);profiles=read_json(code/PROFILES)
    require(0<=args.start<args.stop<=36,'S16 bounded half-open slice0..36')
    if args.command=='plan':
        for row in coordinates(contract,profiles)[args.start:args.stop]:print(json.dumps(row))
        return 0
    if args.command=='footprint':
        require(args.device_json is not None,'S16 offline footprint requires explicit device snapshot')
        print(json.dumps(footprint(contract,profiles,read_json(args.device_json)),indent=2));return 0
    raise ValueError('S16 controller A draft: execution disabled until independent A and storage/source B; no queries/targets started')
if __name__=='__main__':raise SystemExit(main())

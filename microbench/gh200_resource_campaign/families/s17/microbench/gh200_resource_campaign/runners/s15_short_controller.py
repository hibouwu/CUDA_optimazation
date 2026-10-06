"""Finite S15 short slice planner/controller; archive state prevents remeasurement.

No GPU is executed by plan/status/footprint. Actual run requires independently
reviewed controller gates, a valid Slurm allocation and bound device query.
"""
from pathlib import Path
import argparse,json,subprocess,sys,time
from common.suite_io import require,read_json,sha,atomic_json,file_lock,relative,validate_gate

SUITE='results/gh200_resource_campaign/20261001-resource-suite-v2'
CONTRACT='contracts/tma_tensor_2d_short_current_core_v3.draft.json'
PROFILES='contracts/tma_tensor_2d_validation_profiles_v1.draft.json'
MANIFEST='contracts/tma_tensor_2d_validation_adapter_current_core_v4.draft.json'
CORE_CAMPAIGN='microbench/gh200_resource_campaign'


def coordinates(contract,profiles):
    result=[{'index':i,'case_id':case['id'],'profile_id':profile['id'],'seed':profile['required_seed']} for i,(case,profile) in enumerate((c,p) for c in contract['cases'] for p in profiles['profiles'])]
    require(len(contract['cases'])==68 and len(result)==204 and len({(r['case_id'],r['profile_id'],r['seed']) for r in result})==204,'fixed68×3 unique S15 coordinates')
    require([p['required_seed'] for p in profiles['profiles']]==[0,3,4294967295] and [p['target_iterations'] for p in profiles['profiles']]==[[1],[2],[33]],'fixed paired profiles, never Cartesian extra seeds')
    return result


def artifact_bytes(case,profile,device):
    p=case['parameters'];q=p['payload_bytes'];width,height=p['box_dim'];pitch=p['row_stride_bytes'];dynamic=q+1056
    # Necessary capacities give a conservative launch upper bound, not an API
    # occupancy claim. Source-B actual registers40/static0/local0 are fixed.
    occ=min(16,device['smem_per_sm_bytes']//dynamic,device['registers_per_sm']//(40*128),2048//128)
    require(occ>=1,'case exceeds modeled necessary resource capacities')
    blocks=1 if case['scope']=='one_cta' else device['sms']*min(4,occ)
    iterations=profile['target_iterations'][0];allocation=blocks*32*height*pitch+256
    logical=physical=blocks*iterations*q
    completion=stamps=blocks*5*2*4
    return {'blocks_upper_bound':blocks,'occupancy_necessary_capacity_upper_bound':occ,'logical_bytes':logical,'physical_bytes':physical,'global_ring_padding_guard_bytes':allocation,'completion_bytes':completion,'stamp_bytes':stamps,'descriptor_bytes':128,'artifact_bytes_upper_bound':logical+physical+allocation+completion+stamps+128,'actual_occupancy_claim':False}


def validate_archived(entry,repo,coordinate):
    require(entry['coordinate']==coordinate and entry['state']=='archived_verified','archived fixed coordinate')
    run_path=SUITE+'/tma_tensor_2d/short-v1-'+coordinate['case_id']+'--'+coordinate['profile_id']
    require(entry.get('run_path')==run_path,'archive must preserve the fixed original run path')
    for key in ['archive','index','closure','offhost_receipt','archive_verification']:
        ref=entry[key];require(set(ref)=={'path','sha256'},'archive identity ref fields');require(sha(relative(repo,ref['path']))==ref['sha256'],'archived evidence identity changed')
    index=read_json(relative(repo,entry['index']['path']));closure=read_json(relative(repo,entry['closure']['path']))
    require(set(closure)=={'schema_version','run_path','manifest_sha256','members'} and closure['schema_version']==1,'finite run archive closure schema')
    require(closure['run_path']==run_path and closure['manifest_sha256']==entry.get('manifest_sha256'),'original run manifest identity')
    require(index['members']==closure['members'] and index['archive_sha256']==entry['archive']['sha256'],'full archive closure')
    from common.packed_evidence import PackedEvidence,logical_path
    bundle=PackedEvidence(relative(repo,entry['archive']['path']),relative(repo,entry['index']['path']),expected_index_sha256=entry['index']['sha256'],expected_archive_sha256=entry['archive']['sha256'],expected_closure=closure['members'])
    manifest_name=run_path+'/validation_manifest.json';spec_name=run_path+'/validation_spec.json';summary_name=run_path+'/diagnostic_summary.json';state_name=run_path+'/validation_state.json'
    raw_names=[n for n in closure['members'] if n.startswith(run_path+'/attempts/') and n.endswith('/raw.jsonl')]
    require(raw_names and all(n.startswith(run_path+'/') for n in closure['members']),'single complete original run namespace, no unrelated members')
    verified=bundle.verify_all(small_objects=[manifest_name,spec_name,summary_name,state_name]+raw_names)
    with bundle.verified_transaction([manifest_name,spec_name,summary_name,state_name]+raw_names) as view:
        require(view.file_sha256(manifest_name)==entry['manifest_sha256'],'bound original manifest bytes')
        manifest=view.read_json(manifest_name)
        require(isinstance(manifest,dict) and manifest,'original nonempty validation manifest')
        expected={manifest_name,state_name}
        for name,digest in manifest.items():
            name=logical_path(name);full=run_path+'/'+name;expected.add(full)
            require(full in closure['members'] and view.file_sha256(full)==digest,'original manifest member missing or changed')
        require(expected==set(closure['members']),'complete exact original run manifest closure')
        spec=view.read_json(spec_name)
        require(spec['stage']=='S15' and spec['family']=='tma_tensor_2d' and spec['kind']=='case_diagnostic','S15 diagnostic archive identity')
        for key in ('case_id','profile_id','seed'):require(spec[key]==coordinate[key],'archived spec coordinate mismatch')
        require(view.read_json(state_name)['status']=='case_diagnostic_passed','original diagnostic state');require(view.read_json(summary_name)['status']=='case_diagnostic_passed','original diagnostic final state')
        for name in raw_names:
            rows=[json.loads(line) for line in view.read_bytes(name).decode('utf-8').splitlines() if line.strip()]
            validations=[row for row in rows if row.get('type')=='validation']
            require(len(validations)==1,'fixed one-target S15 process record')
            for key in ('case_id','profile_id','seed'):require(validations[0][key]==coordinate[key],'archived raw coordinate mismatch')
    require(bundle.transaction_receipt['closed'] is True,'archive evidence transaction completed')
    for key,location in [('archive_verification','remote'),('offhost_receipt','offhost')]:
        receipt=read_json(relative(repo,entry[key]['path']))
        require(set(receipt)=={'schema_version','status','location','archive_sha256','index_sha256','closure_sha256','run_path','manifest_sha256','members','member_count','uncompressed_bytes','verification'},'full-member verification receipt exact schema')
        require(receipt['schema_version']==1 and receipt['status']=='full_members_verified' and receipt['location']==location and receipt['verification']=='complete_decode_all_members_sha256','explicit full-member verification status')
        for field,refkey in [('archive_sha256','archive'),('index_sha256','index'),('closure_sha256','closure')]:require(receipt[field]==entry[refkey]['sha256'],'verification receipt pack identity')
        require(receipt['run_path']==run_path and receipt['manifest_sha256']==entry['manifest_sha256'] and receipt['members']==closure['members'],'verification receipt run/member identity')
        require(type(receipt['member_count']) is int and receipt['member_count']==verified['members'] and type(receipt['uncompressed_bytes']) is int and receipt['uncompressed_bytes']==verified['uncompressed_bytes'],'verification receipt complete counts')
    return 'archived_verified'

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


def next_action(root,ledger_entry,repo,coordinate):
    if ledger_entry is not None:
        require(ledger_entry['coordinate']==coordinate,'ledger coordinate mismatch')
        if ledger_entry['state']=='archived_verified':
            validate_archived(ledger_entry,repo,coordinate)
            return 'archive_audit' if not root.exists() else 'audit'
        require(ledger_entry['state']!='failed','failed measurement cannot automatically rerun')
        if ledger_entry['state']=='started_unknown':
            require(root.exists() and (root/'validation_state.json').is_file() and read_json(root/'validation_state.json')['status'] in ('pending','checkpoint') and not list((root/'attempts').glob('attempt_*')),'uncertain started target cannot automatically rerun')
    if not root.exists():
        require(ledger_entry is None or ledger_entry['state']=='pending','missing measured directory does not permit new GPU launch')
        return 'run'
    state_path=root/'validation_state.json'
    require(state_path.is_file(),'partial initialize requires explicit offline recovery before launch')
    state=read_json(state_path)['status'];attempts=list((root/'attempts').glob('attempt_*'))
    if state=='case_diagnostic_passed':return 'audit'
    if state in ('pending','checkpoint') and not attempts:return 'resume'
    raise ValueError('existing failed/started evidence requires investigation: '+str(root))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['plan','run','footprint']);parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=204);parser.add_argument('--device-json',type=Path);parser.add_argument('--quota-receipt',type=Path);parser.add_argument('--max-resident-bytes',type=int,default=4*1024**3);args=parser.parse_args()
    repo=args.repo.resolve();code=repo/CORE_CAMPAIGN;suite=repo/SUITE;contract=read_json(code/CONTRACT);profiles=read_json(code/PROFILES);selected=coordinates(contract,profiles);require(0<=args.start<args.stop<=204,'slice0<=start<stop<=204');cases={c['id']:c for c in contract['cases']};byprofile={p['id']:p for p in profiles['profiles']}
    if args.command=='plan':
        for c in selected[args.start:args.stop]:print(json.dumps(c))
        return 0
    require(args.device_json is not None,'footprint/run requires exact bound device query JSON');device=read_json(args.device_json)
    rows=[c|artifact_bytes(cases[c['case_id']],byprofile[c['profile_id']],device) for c in selected]
    if args.command=='footprint':print(json.dumps({'coordinates':rows,'total_artifact_bytes_upper_bound':sum(c['artifact_bytes_upper_bound'] for c in rows),'max_artifact_bytes_upper_bound':max(c['artifact_bytes_upper_bound'] for c in rows)},indent=2));return 0
    # This controller stays disabled until its own independent A/B source gates.
    for phase,path in [('controller-A','reviews/S15-short-controller-A-review-r6.json'),('controller-source-B','reviews/S15-short-controller-source-B-review-r4.json')]:validate_gate(relative(suite,path),repo,'S15',phase)
    require(args.quota_receipt is not None,'current verified quota receipt required before target')
    from runners.environment import inspect_allocation,environment_identity
    allocation=inspect_allocation();env=environment_identity(allocation);require(device['uuid'].lower()==env['uuid'].lower(),'device query belongs to current allocated GPU')
    binding=read_json(args.device_json.with_suffix('.binding.json'));require(binding['device_sha256']==sha(args.device_json) and binding['environment']==env,'device evidence binding belongs to current family')
    require(sha(relative(repo,binding['stdout']['path']))==binding['stdout']['sha256'] and read_json(relative(repo,binding['stdout']['path']))==device,'actual device query stdout preserved')
    from common.suite_io import process_ok
    require(process_ok(binding['receipt']) and binding['receipt']['timeout_seconds']==30 and len(binding['receipt']['argv'])==2 and binding['receipt']['argv'][1]=='device' and binding['receipt']['stdout_sha256']==binding['stdout']['sha256'],'successful bounded read-only actual device query')
    source_review=relative(suite,'reviews/S15-early-validation-source-B-review-r4.json');require(binding['source_review_sha256']==sha(source_review),'device query source-B identity')
    reviewed=read_json(source_review)['gate_files'];require(binding['binary_sha256'] in [v for n,v in reviewed.items() if n.endswith('/diagnostics/tma_tensor_2d/probe')],'device query binary exact reviewed compile identity')
    ledger_path=suite/'tma_tensor_2d-short-v1-ledger.json';identity_path=suite/'tma_tensor_2d-short-v1-environment.json'
    with file_lock(suite/'.tma_tensor_2d-short-v1-controller.lock'):
        if identity_path.exists():require(read_json(identity_path)==env,'family device/toolchain changed before target')
        else:atomic_json(identity_path,env)
        identity={'contract_sha256':sha(code/CONTRACT),'profiles_sha256':sha(code/PROFILES),'adapter_sha256':sha(code/MANIFEST),'device_sha256':sha(args.device_json),'environment':env,'controller_sha256':sha(Path(__file__)),'probe_dependencies_sha256':read_json(code/MANIFEST)['files_sha256']}
        if ledger_path.exists():ledger=read_json(ledger_path);require(ledger['identity']==identity,'source/device change requires new reviewed campaign revision')
        else:ledger={'schema_version':1,'identity':identity,'entries':{}};atomic_json(ledger_path,ledger)
        resident=0
        for row in rows:
            root=suite/'tma_tensor_2d'/('short-v1-'+row['case_id']+'--'+row['profile_id'])
            if root.exists():resident+=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        for c in selected[args.start:args.stop]:
            root=suite/'tma_tensor_2d'/('short-v1-'+c['case_id']+'--'+c['profile_id']);key=str(c['index']);action=next_action(root,ledger['entries'].get(key),repo,c)
            if action=='archive_audit':print(json.dumps({'coordinate':c,'action':action,'GPU_execution':False}));continue
            estimate=rows[c['index']]['artifact_bytes_upper_bound']+64*1024**2
            if action in ('run','resume'):
                from runners.s15_storage import query_quota
                query_quota(repo,args.quota_receipt)
                headroom=quota_headroom(repo,args.quota_receipt,env)
                # Reserve room for lossless sealing as well as next complete raw.
                seal_bound=resident+estimate+(resident+estimate)//100+64*1024**2
                if resident+estimate>args.max_resident_bytes or estimate+seal_bound>headroom:
                    print(json.dumps({'status':'quota_checkpoint_before_next_target','next_coordinate':c,'resident_bytes':resident,'estimated_new_bytes':estimate,'worst_case_seal_bytes':seal_bound,'quota_headroom':headroom}));return 0
                ledger['entries'][key]={'coordinate':c,'state':'started_unknown','run_path':str(root.relative_to(repo))};atomic_json(ledger_path,ledger)
            argv=[sys.executable,'-B',str(code/'validate_suite.py'),action]
            if action=='run':argv+=['--suite',str(suite),'--output',str(root),'--contract',str(code/CONTRACT),'--profiles',str(code/PROFILES),'--adapter-manifest',str(code/MANIFEST),'--case',c['case_id'],'--profile',c['profile_id'],'--seed',str(c['seed'])]
            else:argv+=[str(root)]
            result=subprocess.run(argv,check=False)
            if result.returncode:ledger['entries'][key]={'coordinate':c,'state':'failed','run_path':str(root.relative_to(repo))};atomic_json(ledger_path,ledger);return result.returncode
            require(read_json(root/'diagnostic_summary.json')['status']=='case_diagnostic_passed','only successful complete diagnostics')
            ledger['entries'][key]={'coordinate':c,'state':'resident_passed','run_path':str(root.relative_to(repo)),'manifest_sha256':sha(root/'validation_manifest.json')};atomic_json(ledger_path,ledger)
            if action!='audit':resident+=estimate
    return 0

if __name__=='__main__':raise SystemExit(main())

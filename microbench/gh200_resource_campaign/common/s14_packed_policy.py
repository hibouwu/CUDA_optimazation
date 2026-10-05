"""Finite historical S14 original84 schema bridge; no executable dispatch/GPU.

Caller must supply a successfully closed public verified_transaction view and
complete observer proof from the same archive. This routine only derives an
unsigned normalized history index; independent bridge/core B remains mandatory.
"""
import hashlib,json,re
from common.suite_io import require,gate_mapping,DIMENSIONS
from common.packed_evidence import json_object
from common.family_b3 import compile_resources,environment_identity
from auditors.tma_bulk import validate_contract,complete_profile_set
from auditors.tma_bulk_sass_baseline import BASELINE
from auditors.tma_bulk_validation import validate_validation
from pack_s14_evidence import REVIEW,REVIEW_SHA,COVERAGE,COVERAGE_SHA,CONTRACT,PROFILES


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def historical_gate(view,path,stage,phase,gate_root=''):
    """Finite context gate check on the public active transaction view only."""
    review=view.read_json(path)
    require(review.get('schema_version')==2 and review.get('stage')==stage and review.get('phase')==phase
            and review.get('status')=='pass','historical gate identity/status')
    authors=set(re.findall(r'/root(?:/[a-zA-Z0-9_]+)?',review.get('implementer','')))
    require(review.get('reviewer') and review.get('implementer') and review['reviewer']!=review['implementer']
            and review['reviewer'] not in authors,'historical gate independent author identity')
    require(all(review.get('checks',{}).get(key,{}).get('status')=='pass' for key in DIMENSIONS),'all historical gate dimensions')
    require(not any(f.get('status') not in ('closed','resolved','fixed','verified') for f in review.get('findings',[])),'historical findings closed')
    members=view.inventory()
    for local,digest in gate_mapping(review).items():
        name=gate_root+local;require(name in members and members[name]['sha256']==digest,'historical gate explicit namespace/hash')
    return review


def validate_original84(view,closure,observer_proof):
    require(view.file_sha256(REVIEW)==REVIEW_SHA and view.file_sha256(COVERAGE)==COVERAGE_SHA,'immutable signed original84 roots')
    original=historical_gate(view,REVIEW,'S14','early-validation-B3')
    require(original['authorization']['family_B3_qualified'] is True and original['authorization']['performance_eligible'] is False,'original historical purpose')
    coverage=view.read_json(COVERAGE);contract=validate_contract(view.read_json(CONTRACT));profiles=view.read_json(PROFILES)
    require(original['coverage_record']=={'path':COVERAGE,'sha256':COVERAGE_SHA},'direct original coverage binding')
    require(coverage['formal_cases']==24 and coverage['short_receipts']==72 and coverage['source_release_receipts']==12 and coverage['target_launches']==84,'original finite counts')
    cases={c['id']:c for c in contract['cases']};profile_map={p['id']:p for p in profiles['profiles']}
    expected={(c['id'],p['id'],p['required_seed']) for c in contract['cases'] for p in profiles['profiles'] if c['parameters']['direction'] in p['applicable_directions']}
    require(len(expected)==84 and len(coverage['records'])==84,'fixed84 profile combinations')
    require(observer_proof['full_SASS_members_checked']==84 and observer_proof['complete_values']['payload_guard_words']==2967828384
            and observer_proof['complete_values']['lifecycle_words']==231192,'same full actual words and code proof')
    require(set(observer_proof['complete_values']['per_run'])=={r['run_path'] for r in coverage['records']},'observer covers original84 roots')
    facts=observer_proof['target_facts'];contexts={(REVIEW,'')}
    for name in ('S14-early-validation-source-B-review.json','S14-scratch-recovery-review.json'):
        contexts.add((REVIEW.rsplit('/',1)[0]+'/'+name,''))
    records=[];seen=set();common_identity=None;machine=None;payload=control=0
    for old in coverage['records']:
        key=(old['case_id'],old['profile_id'],old['seed']);require(key in expected and key not in seen,'unique original profile coordinate');seen.add(key)
        root=old['run_path'];require(root=='results/gh200_resource_campaign/20261001-resource-suite-v2/tma_bulk/short-v1-'+key[0]+'--'+key[1],'canonical historical run root')
        require(view.file_sha256(root+'/validation_manifest.json')==old['manifest_sha256'],'original manifest identity')
        manifest=view.read_json(root+'/validation_manifest.json')
        for local,digest in manifest.items():require(view.file_sha256(root+'/'+local)==digest,'full run closure member identity')
        spec=view.read_json(root+'/validation_spec.json');require((spec['case_id'],spec['profile_id'],spec['seed'])==key,'original spec coordinates')
        require(view.read_json(root+'/'+spec['contract_path'])==contract and view.file_sha256(root+'/'+spec['profiles_path'])==view.file_sha256(PROFILES),'original per-run contract/profiles')
        for dep in spec['reviews']:
            name=root+'/snapshot/suite/'+dep['path'];require(view.file_sha256(name)==dep['sha256'],'frozen review identity');contexts.add((name,root+'/snapshot/repo/'))
        lines=view.read_bytes(root+'/attempts/attempt_00/raw.jsonl').splitlines();require(len(lines)==2,'device and original validation rows');device,row=[json_object(line) for line in lines]
        require(device==coverage['device']==view.read_json(root+'/environment/device.json'),'original device facts')
        validated=validate_validation(device,row,cases[key[0]],profile_map[key[1]],key[2]);require(row['resource_identity']==old['resource_identity'] and row['blocks']==old['blocks'],'original resources and geometry')
        require(validated['checked_elements']==old['payload_guard_words_checked'],'original declared full word count')
        receipt=view.read_json(root+'/attempts/attempt_00/receipt.json');require(view.file_sha256(root+'/attempts/attempt_00/receipt.json')==old['receipt_sha256'],'receipt original hash')
        require(receipt['returncode']==0 and receipt['timed_out'] is False and receipt['cleanup_confirmed'] is True and receipt['timeout_seconds']==30,'original completed bounded process')
        require(receipt['argv']==[spec['execution_root']+'/binary/probe','validate-only',key[0],key[1],str(key[2])],'original exact command')
        registration=receipt['gpu_process_registration'];require(registration['gpu_uuid']==device['uuid'].lower() and registration['state']=='cleanup_confirmed'
            and receipt['pid']==registration['pid']==receipt['pgid']==registration['pgid'],'registered original GPU process')
        require(receipt['stdout_sha256']==view.file_sha256(root+'/attempts/attempt_00/raw.jsonl') and receipt['stderr_sha256']==view.file_sha256(root+'/attempts/attempt_00/stderr'),'process stream integrity')
        require(view.file_sha256(root+'/binary/probe')==spec['binary_sha256'] and view.file_sha256(root+'/build/sass.stdout')==spec['sass_sha256'],'actual binary/SASS identity')
        require(registration['cleanup_receipt']=={k:v for k,v in receipt.items() if k!='gpu_process_registration'},'original exact cleanup receipt')
        require(receipt['stderr_sha256']==hashlib.sha256(b'').hexdigest(),'original empty stderr')
        require(view.file_sha256(root+'/build/dependencies.json')==spec['dependencies_sha256'] and view.file_sha256(root+'/build/shared_libraries.json')==spec['shared_libraries_sha256'] and view.file_sha256(root+'/snapshot/manifest.json')==spec['snapshot_manifest_sha256'],'original dependency/library/snapshot bindings')
        for a in row['checks'][0]['output_artifacts']:require(view.file_sha256(root+'/attempts/attempt_00/'+a['path'])==a['sha256'],'all full value artifacts retained')
        per_run=observer_proof['complete_values']['per_run'][root];require(per_run['payload_guard_words']==validated['checked_elements'] and per_run['lifecycle_words']==old['lifecycle_words_checked'],'independent complete value accounting')
        current=facts[root+'/build/sass.stdout'];require(len(current)==18,'all actual targets')
        if machine is None:machine=current
        require(current==machine,'actual18 complete encoded target drift')
        resources=compile_resources(view.read_bytes(root+'/build/compile.stderr').decode(),current)
        selected=row['resource_identity'];require(resources[selected['kernel_symbol']]['registers_per_thread']==selected['registers_per_thread']
            and resources[selected['kernel_symbol']]['static_smem_bytes']==selected['static_smem_bytes'] and selected['local_size_bytes']==0,'actual compiled selected resources')
        env=view.read_json(root+'/environment/initial.json');dependencies=view.read_json(root+'/build/dependencies.json');libraries=view.read_json(root+'/build/shared_libraries.json')
        expected_semantic={'contract_sha256':view.file_sha256(CONTRACT),'profiles_sha256':view.file_sha256(PROFILES),'source_sha256':view.file_sha256(contract['source']),'reference_sha256':profile_map[key[1]]['reference_identity']['sha256'],'device_uuid':device['uuid'],'normalized_target_set_sha256':canonical_sha(BASELINE),'compiler_identity_sha256':canonical_sha({name:env[name] for name in ['compiler','driver','host','execution_uid']})}
        require(old['semantic_identity']==expected_semantic,'original semantic identity matches actual files/code/environment')
        require(view.file_sha256(root+'/'+spec['source_path'])==expected_semantic['source_sha256'],'exact frozen original source')
        require(view.file_sha256(profile_map[key[1]]['reference_identity']['path'])==expected_semantic['reference_sha256'],'original numerical reference bytes')
        identity={'device':device,'environment':environment_identity(env),'external_toolchain':dependencies['external_toolchain'],'libraries':libraries}
        if common_identity is None:common_identity=identity
        require(identity==common_identity,'actual device/toolchain/external/libraries across original84')
        for local in dependencies['local']:
            require(local.startswith('repo/'),'frozen local include namespace')
            frozen=root+'/snapshot/'+local
            # Historical code is resolved in its declared snapshot, never live.
            require('snapshot/'+local in manifest,'compiled local include retained')
        payload+=per_run['payload_guard_words'];control+=per_run['lifecycle_words']
        record=dict(old);record['status']='pass';record['full_value_replay']=True;records.append(record)
    require(seen==expected and payload==2967828384 and control==231192,'complete aggregate original84 counts')
    require(contexts=={(c['review_path'],'' if c['gate_root']=='repository_root' else c['gate_root']) for c in closure['review_contexts']},'exact declared historical review contexts')
    for path,root in sorted(contexts):
        review=view.read_json(path);historical_gate(view,path,review['stage'],review.get('phase','review'),root)
    complete_profile_set(contract,records)
    return {'schema_version':1,'kind':'s14_original84_packed_history','status':'original84_integrity_derived_pending_independent_bridge',
            'original_review_sha256':REVIEW_SHA,'original_coverage_sha256':COVERAGE_SHA,'profile_records':records,
            'case_count':24,'profile_processes':84,'target_launches':84,'payload_guard_words':payload,'lifecycle_words':control,
            'historical_contexts':len(contexts),'actual_target_facts':machine,'actual_device_environment':common_identity,
            'original_historical_B3_preserved':True,'bridge_qualified':False,'formal_sampling':False,'GPU_execution':False}

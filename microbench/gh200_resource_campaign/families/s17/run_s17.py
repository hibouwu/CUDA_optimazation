"""One finite S17 family: resource exclusions, short checks, own pilots, sampling."""
from pathlib import Path
import argparse
import json
import os
import random
import shutil
import sys

if sys.flags.optimize != 0:
    raise RuntimeError('S17 requires active assertions; Python optimization forbidden')

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'microbench/gh200_resource_campaign'))
from common.suite_io import atomic_json, bounded, digest, file_lock, gpu_lock, process_ok, read_json, require, sha, stats, validate_gate, verify_files
from runners.environment import inspect_allocation, budget, permission_fingerprint
from control import capability_matrix, symbol, check_short, check_measured, bind_process
from point_pack import make_pack, publish_pack, ArchiveLimitExceeded

SHORT_PACK_LIMIT = 16*1024**2
OTHER_PACK_LIMIT = 3*1024**2
STORAGE_RESERVE = 768*1024**2


def storage_budget():
    # Upper bounds admit all 42 coordinates and all three batches, without compression assumptions.
    retained=42*SHORT_PACK_LIMIT+(42+1260)*OTHER_PACK_LIMIT
    cases=read_json(ROOT/'cases.json')['cases']
    largest=max(raw_payload_bytes(case,1 if case['scope']=='one_cluster' else 1024,True) for case in cases)+16*1024**2
    return {'retained_packet_ceiling_bytes':retained,'largest_raw_reserve_bytes':largest,
        'safety_reserve_bytes':STORAGE_RESERVE,'required_free_bytes':retained+largest+STORAGE_RESERVE}


def raw_payload_bytes(case,clusters,short):
    from auditors.cluster_dsm_reference_v2 import shapes, PAIRS
    import math
    p=case['parameters'];C=p['cluster_size'];mode=p['mode'];B=clusters*C
    if short:
        return 4*sum(sum(math.prod(shape) for shape in shapes(p['form'],C,clusters,I).values()) for I,_ in PAIRS)
    W=4096 if mode>=5 else 1024 if mode<4 else 0
    return 4*(B*22+(B*(W+8) if W else 0)+(B*128 if mode<4 else 0)+(clusters*32*4096 if mode>=5 else 0))


def run(binary, output, cpu_fixture=False):
    gate=validate_gate(ROOT/'source-review.json',ROOT,'S17','full-value-family-run-source-B')
    require(gate['authorization']['formal_after_full_short_and_own_pilot'] is True,'conditional family source admission')
    manifest=read_json(ROOT/'source-manifest.json');verify_files(ROOT,manifest)
    binary=Path(binary).resolve();identity=sha(binary)
    require(identity==read_json(ROOT/'binary-identity.json')['probe'],'exact target binary')
    cases=read_json(ROOT/'cases.json')['cases'];require(len(cases)==42 and len({c['id'] for c in cases})==42,'original42 coordinates')
    protocol=read_json(ROOT/'microbench/gh200_resource_campaign/contracts/protocol.json')
    env=inspect_allocation();env['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip();output=Path(output)
    require(output.is_absolute() and output.parent==Path('/tmp') and output.name.startswith('codex-s17-node-')
            and not output.exists(),'fresh owned node namespace; existing run must be recovered')
    # These are enforced output ceilings, not assumed compression ratios.
    policy=storage_budget();retained_budget=policy['retained_packet_ceiling_bytes']
    require(shutil.disk_usage('/tmp').free>=policy['required_free_bytes'],
            'complete three-batch packet budget plus largest raw stage and reserve')
    with file_lock(ROOT.parent/'.s17-output.lock'),gpu_lock(env['uuid']):
        output.mkdir(mode=0o700);stage=output/'raw-stage';stage.mkdir(mode=0o700);(output/'packs').mkdir()
        atomic_json(output/'initial.json',env);atomic_json(output/'source-manifest.json',manifest)
        atomic_json(output/'source-review.json',gate);atomic_json(output/'binary-identity.json',{'probe':identity})
        atomic_json(output/'binary-path.json',{'probe':str(binary)})
        atomic_json(output/'storage-policy.json',{'short_pack_ceiling':SHORT_PACK_LIMIT,'other_pack_ceiling':OTHER_PACK_LIMIT,
            'complete_retained_budget_bytes':retained_budget,'reserve':STORAGE_RESERVE,'raw_stage_inside_export_root':True,
            'cap_failure_preserves_original_raw_and_pack':True,'no_compression_ratio_assumed_for_admission':True})
        query=output/'device-query';query.mkdir()
        receipt=bounded([str(binary),'device'],ROOT,query/'stdout',query/'stderr',30,uuid=env['uuid'])
        atomic_json(query/'receipt.json',receipt);require(process_ok(receipt),'own device query')
        device=read_json(query/'stdout');require(device['uuid'].lower()==env['uuid'].lower() and device['cc']=='9.0'
            and 'GH200' in device['name'],'own current GH200')
        require(('CPU_fixture' in device['name'])==cpu_fixture,'explicit CPU fixture boundary')
        atomic_json(output/'device.json',device)
        capabilities={}
        for name,measured in (('legacy-capabilities',False),('measured-capabilities',True)):
            folder=output/name;folder.mkdir()
            receipt=bounded([str(binary),'measured-cluster-device' if measured else 'cluster-device'],ROOT,folder/'stdout',folder/'stderr',30,uuid=env['uuid'])
            atomic_json(folder/'receipt.json',receipt);require(process_ok(receipt),'own capability query')
            rows=[json.loads(line) for line in (folder/'stdout').read_text().splitlines()]
            require(len(rows)==22 and rows[0]==device,'complete own21 capabilities plus device')
            capabilities[name]=capability_matrix(rows[1:],device,measured)
        atomic_json(output/'capabilities.json',capabilities)
        profile=output/'counter-capability';profile.mkdir()
        ncu=shutil.which('ncu')
        fingerprint=permission_fingerprint(env,ncu)
        cache=Path('/tmp')/('gh200-s17-counter-'+digest(fingerprint))
        if cache.exists():
            require(not cache.is_symlink() and cache.stat().st_uid==os.geteuid(),'owned counter capability cache')
            status=read_json(cache/'status.json')
            require(status['fingerprint']==fingerprint and status['state'] in ('available','permission_denied','tool_missing'),
                    'prior counter outcome unresolved; inspect original process before collection')
            for name,h in status.get('evidence_files',{}).items():
                source=cache/name
                require(source.is_file() and not source.is_symlink() and sha(source)==h,'complete cached counter evidence')
                shutil.copy2(source,profile/name)
            atomic_json(profile/'status.json',dict(status,reused=True))
        else:
            cache.mkdir(mode=0o700)
            atomic_json(cache/'status.json',{'state':'started_unknown','fingerprint':fingerprint})
        if not (profile/'status.json').exists() and ncu is None:
            status={'state':'tool_missing','fingerprint':fingerprint,'physical_traffic_proven':False}
            atomic_json(profile/'status.json',status);atomic_json(cache/'status.json',status)
        elif not (profile/'status.json').exists():
            require(capabilities['measured-capabilities']['s17m_local_read_c2']['supported'],
                    'counter permission representative target must be queried supported')
            budget(env,240)
            argv=[ncu,'--clock-control','none','--cache-control','none','--kernel-name','regex:s17m_local_read_c2',
                '--launch-count','1','--metrics','gpu__time_duration.sum','--csv',str(binary),
                'pilot-only','local_read_c2_one_cluster','128','3']
            counter_receipt=bounded(argv,profile,profile/'stdout',profile/'stderr',120,uuid=env['uuid'])
            atomic_json(profile/'receipt.json',counter_receipt)
            text=(profile/'stdout').read_text()+(profile/'stderr').read_text()
            require(counter_receipt['cleanup_confirmed'],'counter process cleanup unconfirmed; stop device collection')
            state='available' if process_ok(counter_receipt) else 'permission_denied' if 'ERR_NVGPUCTRPERM' in text else 'failed_unknown'
            status={'state':state,'fingerprint':fingerprint,'physical_traffic_proven':False,
                'permission_check_only':True,'not_a_formal_sample':True}
            status['evidence_files']={p.name:sha(p) for p in profile.iterdir() if p.is_file()}
            for name in status['evidence_files']:shutil.copy2(profile/name,cache/name)
            atomic_json(profile/'status.json',status);atomic_json(cache/'status.json',status)
            require(state!='failed_unknown','counter capability execution failed for unclassified reason')
        resources={};index=0;processes=set();previous_stop=None

        def execute(case,kind,iterations=None,seed=3,batch=0,trial=0):
            nonlocal index,previous_stop
            require(sha(binary)==identity,'binary unchanged before target')
            if kind=='short':argv=[str(binary),'validate-only',case['id'],'cluster_short_1_2_5_v1','3']
            else:argv=[str(binary),'pilot-only' if kind=='pilot' else 'formal-only',case['id'],str(iterations),str(seed)]
            p=case['parameters'];query=resources[case['id']]['query']
            G=1 if case['scope']=='one_cluster' else query['active_cluster_capacity']
            if kind=='short':
                legacy=capabilities['legacy-capabilities'][symbol(p['form'],p['cluster_size'],False)]
                require(legacy['supported'] is True,'legacy short target capability')
                G=1 if case['scope']=='one_cluster' else legacy['query']['active_cluster_capacity']
            raw_upper=raw_payload_bytes(case,G,kind=='short')
            raw_upper+=16*1024**2 # bounded raw JSON, identity, receipts and control metadata
            ceiling=SHORT_PACK_LIMIT if kind=='short' else OTHER_PACK_LIMIT
            require(shutil.disk_usage(output).free>=raw_upper+2*ceiling+STORAGE_RESERVE,'exact raw plus enforced packet peak budget')
            budget(env,240);folder=stage/('point%04d'%index);folder.mkdir()
            point={'index':index,'case_id':case['id'],'kind':kind,'iterations':iterations,'seed':seed,'batch':batch,'trial':trial,
                'binary_sha256':identity,'raw_upper_bound':raw_upper,'pack_ceiling':ceiling}
            atomic_json(folder/'point.json',dict(point,state='started_unknown'));atomic_json(output/(folder.name+'-state.json'),dict(point,state='started_unknown',raw_path=str(folder)))
            receipt=bounded(argv,folder,folder/'raw.jsonl',folder/'stderr',120,uuid=env['uuid']);atomic_json(folder/'receipt.json',receipt)
            require(sha(binary)==identity and process_ok(receipt),'target failed; preserve original files, no automatic retry')
            key=bind_process(folder,receipt,argv,env,cpu_fixture)
            require(cpu_fixture or key not in processes,'independent process identity');processes.add(key)
            require(previous_stop is None or previous_stop<=receipt['host_start_ns'],'ordered nonoverlapping processes');previous_stop=receipt['host_stop_ns']
            rows=[json.loads(s) for s in (folder/'raw.jsonl').read_text().splitlines()];require(len(rows)==2 and rows[0]==device,'complete own-device rows')
            row=rows[1]
            if kind=='short':result=check_short(folder,row,case,device,capabilities['legacy-capabilities'][symbol(case['parameters']['form'],case['parameters']['cluster_size'],False)])
            else:result=check_measured(folder,row,case,device,resources[case['id']],seed,iterations,kind=='pilot',protocol)
            atomic_json(folder/'observation.json',result);atomic_json(folder/'point.json',dict(point,state='complete_checked'))
            archive=stage/(folder.name+'.tar.xz')
            try:packet=make_pack(folder,archive,max_archive_bytes=ceiling)
            except ArchiveLimitExceeded:
                atomic_json(output/(folder.name+'-state.json'),dict(point,state='target_checked_pack_limit_checkpoint',raw_path=str(folder),partial_archive=str(archive)))
                atomic_json(output/'checkpoint.json',{'reason':'bounded_compressed_write_limit','point':point,
                    'raw_preserved':str(folder),'partial_archive_preserved':str(archive),'target_already_complete':True,'new_GPU_retry_forbidden':True})
                raise
            if packet['archive_bytes']>ceiling:
                atomic_json(output/'checkpoint.json',{'reason':'complete_packet_exceeds_frozen_storage_ceiling','point':point,
                    'raw_preserved':str(folder),'archive_preserved':str(archive),'new_GPU_retry_forbidden':True})
                raise RuntimeError('packet ceiling exceeded; recover original packet without resampling')
            destination=output/'packs'/archive.name;publish_pack(archive,destination,packet)
            atomic_json(output/(folder.name+'-state.json'),dict(point,state='verified_node_pack_pending_offhost',pack_identity=packet))
            require(folder.parent==stage and stage.parent==output and stage.stat().st_uid==os.geteuid(),'only own successful staging release')
            shutil.rmtree(folder);archive.unlink();index+=1
            print(json.dumps({'index':point['index'],'case':case['id'],'kind':kind}),flush=True)
            return row,result

        legal=[];excluded=[]
        for case in cases:
            p=case['parameters'];cap=capabilities['measured-capabilities'][symbol(p['form'],p['cluster_size'])]
            legacy=capabilities['legacy-capabilities'][symbol(p['form'],p['cluster_size'],False)]
            resources[case['id']]=cap
            require(not cap['supported'] or legacy['supported'],'supported measured target needs its full legacy short evidence')
            (legal if cap['supported'] else excluded).append(case)
        atomic_json(output/'resources.json',resources);atomic_json(output/'capacity-exclusions.json',excluded)
        reused=[]
        for case in legal:execute(case,'short')
        atomic_json(output/'prior-short-reuse.json',{'case_ids':reused,'performance_reused':False})
        resolved={}
        for case in legal:
            row,_=execute(case,'pilot',128,4294967295)
            resolved[case['id']]={'iterations':max(128,min(65536,int(128*20.0/max(row['event_ms'],.01)))),
                'pilot_event_ms':row['event_ms'],'pilot_iterations':128,'model':'event_ms_truncate_clamp20ms_v1'}
        atomic_json(output/'resolved-cases.json',resolved)
        by_id={case['id']:case for case in legal};active=sorted(by_id);values={key:[] for key in active}
        outcomes={key:{'status':'pending','batches':[]} for key in active}
        for batch in range(3):
            if not active:break
            rng=random.Random(protocol['shuffle_seed']+batch);order=[]
            for trial in range(10):
                keys=list(active);rng.shuffle(keys);order.extend((key,trial) for key in keys)
            atomic_json(output/('order-batch%d.json'%batch),order);current={key:[] for key in active};warm_failed=set()
            for key,trial in order:
                if key in warm_failed:continue
                _,result=execute(by_id[key],'formal',resolved[key]['iterations'],3+19*trial+1009*batch,batch,trial)
                if result['warmup_converged']:current[key].append(result['value']);values[key].append(result['value'])
                else:warm_failed.add(key)
            next_active=[]
            for key in active:
                merged=stats(values[key]) if len(values[key])>=2 else None
                batch_stats=stats(current[key]) if len(current[key])>=2 else None
                stable=len(current[key])==10 and batch_stats['cv']<=.05 and merged['cv']<=.05
                outcomes[key]['batches'].append({'batch':batch,'complete':len(current[key])==10,'warmup_failed':key in warm_failed,'stats':batch_stats})
                outcomes[key].update(merged=merged,status='stable' if stable else 'unstable_after_bounded_remeasurement' if batch==2 else 'pending')
                if not stable:next_active.append(key)
            active=next_active;atomic_json(output/'case-results.json',outcomes)
        verify_files(ROOT,manifest);require(sha(binary)==identity,'immutable binary closure')
        atomic_json(output/'summary.json',{'coordinate_count':42,'legal_cases':len(legal),'capacity_excluded_cases':len(excluded),
            'prior_short_reused':reused,'case_results':outcomes,'point_count':index,'qualified':False,'independent_C_required':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binary',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.binary,args.output)

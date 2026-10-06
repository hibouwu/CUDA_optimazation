"""One finite S20 family: resource exclusions, short checks, own pilots, sampling."""
from pathlib import Path
import argparse
import json
import os
import random
import shutil
import sys

if sys.flags.optimize != 0:
    raise RuntimeError('S20 requires active assertions; Python optimization forbidden')

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'microbench/gh200_resource_campaign'))
from common.suite_io import atomic_json, bounded, digest, file_lock, gpu_lock, process_ok, read_json, require, sha, stats, validate_gate, verify_files
from runners.environment import inspect_allocation, budget, permission_fingerprint
from control import check_capability, check_short, check_measured, bind_process
from point_pack import make_pack, publish_pack, ArchiveLimitExceeded

SHORT_PACK_LIMIT = 1024**2
OTHER_PACK_LIMIT = 256*1024
STORAGE_RESERVE = 768*1024**2


def profiles_for(case):
    result=['periodic_i1','periodic_i3']
    if case['parameters']['mode']!='compute':result.extend(['tagged_i1','tagged_i3'])
    if case['id']=='serial_s1_k1':result.extend(['row_witness','column_witness'])
    return result


def raw_payload_bytes(K,short_N=0):
    return K*16384+4096*4+128*4+32768*2+464+112*4+4+short_N*K*32768


def storage_budget():
    retained=380*SHORT_PACK_LIMIT+(90+3150)*OTHER_PACK_LIMIT
    largest=raw_payload_bytes(64,3)+16*1024**2
    return {'retained_packet_ceiling_bytes':retained,'largest_raw_reserve_bytes':largest,
        'safety_reserve_bytes':STORAGE_RESERVE,'required_free_bytes':retained+largest+STORAGE_RESERVE}


def run(binary, output, cpu_fixture=False):
    gate=validate_gate(ROOT/'source-review.json',ROOT,'S20','full-value-family-run-source-B')
    require(gate['authorization']['formal_after_full_short_and_own_pilot'] is True,'conditional family source admission')
    manifest=read_json(ROOT/'source-manifest.json');verify_files(ROOT,manifest)
    binary=Path(binary).resolve();identity=sha(binary)
    require(identity==read_json(ROOT/'binary-identity.json')['probe'],'exact target binary')
    cases=read_json(ROOT/'cases.json')['cases'];require(len(cases)==105 and len({c['id'] for c in cases})==105,'original105 coordinates')
    protocol=read_json(ROOT/'microbench/gh200_resource_campaign/contracts/protocol.json')
    env=inspect_allocation();env['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip();output=Path(output)
    require(output.is_absolute() and output.parent==Path('/tmp') and output.name.startswith('codex-s20-node-')
            and not output.exists(),'fresh owned node namespace; existing run must be recovered')
    # These are enforced output ceilings, not assumed compression ratios.
    policy=storage_budget();retained_budget=policy['retained_packet_ceiling_bytes']
    require(shutil.disk_usage('/tmp').free>=policy['required_free_bytes'],
            'complete three-batch packet budget plus largest raw stage and reserve')
    with file_lock(ROOT.parent/'.s20-output.lock'),gpu_lock(env['uuid']):
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
        resources={}
        for label,measured in (('legacy-capability',False),('measured-capability',True)):
            folder=output/label;folder.mkdir()
            receipt=bounded([str(binary),'measured-capability' if measured else 'capability','compute_s1_k1'],ROOT,folder/'stdout',folder/'stderr',30,uuid=env['uuid'])
            atomic_json(folder/'receipt.json',receipt);require(process_ok(receipt),'own target capability query')
            rows=[json.loads(line) for line in (folder/'stdout').read_text().splitlines()]
            require(len(rows)==2 and rows[0]==device,'own query device and exact target')
            resources[label]=check_capability(rows[1],device,measured)
        atomic_json(output/'resources.json',resources);resource=resources['measured-capability']
        profile=output/'counter-capability';profile.mkdir()
        ncu=shutil.which('ncu')
        fingerprint=permission_fingerprint(env,ncu)
        cache=Path('/tmp')/('gh200-s20-counter-'+digest(fingerprint))
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
            budget(env,240)
            argv=[ncu,'--clock-control','none','--cache-control','none','--kernel-name','regex:s20_pipeline_measured',
                '--launch-count','1','--metrics','gpu__time_duration.sum','--csv',str(binary),
                'pilot-only','compute_s1_k1','128','3']
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
        index=0;processes=set();previous_stop=None

        def execute(case,kind,iterations=None,seed=3,batch=0,trial=0,short_profile=None):
            nonlocal index,previous_stop
            require(sha(binary)==identity,'binary unchanged before target')
            if kind=='short':argv=[str(binary),'validate-only',case['id'],short_profile,'1']
            else:argv=[str(binary),'pilot-only' if kind=='pilot' else 'formal-only',case['id'],str(iterations),str(seed)]
            K=case['parameters']['k_tiles'];short_N=3 if short_profile and short_profile.endswith('_i3') else 1
            raw_upper=raw_payload_bytes(K,short_N if kind=='short' else 0)
            raw_upper+=16*1024**2 # bounded raw JSON, identity, receipts and control metadata
            ceiling=SHORT_PACK_LIMIT if kind=='short' else OTHER_PACK_LIMIT
            require(shutil.disk_usage(output).free>=raw_upper+2*ceiling+STORAGE_RESERVE,'exact raw plus enforced packet peak budget')
            budget(env,240);folder=stage/('point%04d'%index);folder.mkdir()
            point={'index':index,'case_id':case['id'],'kind':kind,'iterations':iterations,'seed':seed,'batch':batch,'trial':trial,
                'binary_sha256':identity,'raw_upper_bound':raw_upper,'pack_ceiling':ceiling,'short_profile':short_profile}
            atomic_json(folder/'point.json',dict(point,state='started_unknown'));atomic_json(output/(folder.name+'-state.json'),dict(point,state='started_unknown',raw_path=str(folder)))
            receipt=bounded(argv,folder,folder/'raw.jsonl',folder/'stderr',120,uuid=env['uuid']);atomic_json(folder/'receipt.json',receipt)
            require(sha(binary)==identity and process_ok(receipt),'target failed; preserve original files, no automatic retry')
            key=bind_process(folder,receipt,argv,env,cpu_fixture)
            require(cpu_fixture or key not in processes,'independent process identity');processes.add(key)
            require(previous_stop is None or previous_stop<=receipt['host_start_ns'],'ordered nonoverlapping processes');previous_stop=receipt['host_stop_ns']
            rows=[json.loads(s) for s in (folder/'raw.jsonl').read_text().splitlines()];require(len(rows)==2 and rows[0]==device,'complete own-device rows')
            row=rows[1]
            if kind=='short':result=check_short(folder,row,case,device,resources['legacy-capability'],short_profile)
            else:result=check_measured(folder,row,case,device,resource,seed,iterations,kind=='pilot',protocol)
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

        legal=cases;excluded=[];reused=[]
        atomic_json(output/'capacity-exclusions.json',excluded)
        for case in legal:
            for short_profile in profiles_for(case):
                execute(case,'short',seed=1,short_profile=short_profile)
        atomic_json(output/'prior-short-reuse.json',{'coordinates':[],'performance_reused':False})
        fit_cases=[case for case in legal if not case['parameters']['holdout']]
        pilot_ms={}
        for case in fit_cases:
            row,_=execute(case,'pilot',128,4294967295)
            pilot_ms[case['id']]=row['event_ms']
        maximum_ms=max(pilot_ms.values());common_N=max(128,min(65536,int(128*20.0/max(maximum_ms,.01))))
        calibration={'case_event_ms':pilot_ms,'pilot_iterations':128,'max_event_ms':maximum_ms,
            'common_iterations':common_N,'model':'common_repeat_count_from_fit90_20ms_v1','holdout_used_for_calibration':False}
        atomic_json(output/'calibration.json',calibration)
        resolved={case['id']:{'iterations':common_N,'model':calibration['model'],'holdout_used_for_calibration':False} for case in legal}
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
        atomic_json(output/'summary.json',{'coordinate_count':105,'legal_cases':len(legal),'capacity_excluded_cases':len(excluded),
            'prior_short_reused':reused,'case_results':outcomes,'point_count':index,'qualified':False,'independent_C_required':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binary',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.binary,args.output)

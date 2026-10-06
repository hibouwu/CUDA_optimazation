"""Read-only S20 archive replay: full values, launch history and all statistics."""
from pathlib import Path, PurePosixPath
import argparse
import json
import random
import sys

if sys.flags.optimize != 0:
    raise RuntimeError('S20 requires active assertions; Python optimization forbidden')
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'microbench/gh200_resource_campaign'))
from common.suite_io import canonical, read_json, require, sha, stats, validate_gate, verify_files
from auditors.suite import protocol_check
from point_pack import verify_pack
from control import check_capability, check_short, check_measured, bind_process
from run_s20 import SHORT_PACK_LIMIT, OTHER_PACK_LIMIT, profiles_for


def evidence(run, cpu_fixture=False):
    run=Path(run);env=read_json(run/'initial.json');device=read_json(run/'device.json')
    require(('CPU_fixture' in device['name'])==cpu_fixture and (env.get('CPU_fixture_only') is True)==cpu_fixture,
            'fixture requires explicit unqualified route')
    manifest=read_json(ROOT/'source-manifest.json');verify_files(ROOT,manifest)
    require(read_json(run/'source-manifest.json')==manifest,'original source closure')
    if not cpu_fixture:
        gate=validate_gate(ROOT/'source-review.json',ROOT,'S20','full-value-family-run-source-B')
        require(gate==read_json(run/'source-review.json'),'source review identity')
    else:
        require(read_json(run/'source-review.json').get('CPU_fixture_only') is True,'fixture review cannot enter normal route')
    require(read_json(run/'binary-identity.json')==read_json(ROOT/'binary-identity.json'),'original binary identity')
    binary_path=read_json(run/'binary-path.json')['probe']
    require(Path(binary_path).is_absolute(),'original absolute binary path')
    require(device['uuid'].lower()==env['uuid'].lower(),'original allocated device')
    protocol=read_json(ROOT/'microbench/gh200_resource_campaign/contracts/protocol.json');protocol_check(protocol)
    cases=read_json(ROOT/'cases.json')['cases'];by_id={c['id']:c for c in cases};require(len(by_id)==105,'all105 coordinates')
    resources={}
    for label,measured in (('legacy-capability',False),('measured-capability',True)):
        folder=run/label;rows=[json.loads(line) for line in (folder/'stdout').read_text().splitlines()]
        require(len(rows)==2 and rows[0]==device,'original exact target query')
        receipt=read_json(folder/'receipt.json')
        require(receipt['argv']==[binary_path,'measured-capability' if measured else 'capability','compute_s1_k1']
            and receipt['returncode']==0 and receipt['cleanup_confirmed'] is True
            and receipt['stdout_sha256']==sha(folder/'stdout') and receipt['stderr_sha256']==sha(folder/'stderr'),'original zero-target capability evidence')
        resources[label]=check_capability(rows[1],device,measured)
    require(resources==read_json(run/'resources.json'),'independent resource recomputation');resource=resources['measured-capability']
    decoded=[];keys=set();last_stop=None
    for state_path in sorted(run.glob('point*-state.json')):
        state=read_json(state_path)
        require(state['index']==len(decoded) and state['state']=='verified_node_pack_pending_offhost','no missing/unfinished point')
        case=by_id[state['case_id']];archive=run/'packs'/('point%04d.tar.xz'%state['index']);packet=state['pack_identity']
        require(state['binary_sha256']==read_json(run/'binary-identity.json')['probe'],'each point uses frozen binary')
        require(sha(archive)==packet['archive_sha256'] and archive.stat().st_size==packet['archive_bytes'], 'complete original pack')
        require(packet['archive_bytes']<= (SHORT_PACK_LIMIT if state['kind']=='short' else OTHER_PACK_LIMIT),'frozen packet limit')
        verify_pack(archive,packet['members'])
        with tempfile.TemporaryDirectory(prefix='s20-replay-point-') as temporary:
            directory=Path(temporary)
            with tarfile.open(archive) as source:
                for member in source:
                    p=PurePosixPath(member.name)
                    require(member.isfile() and not p.is_absolute() and '..' not in p.parts,'regular relative point member')
                    target=directory/member.name;target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open('xb') as stream:stream.write(source.extractfile(member).read())
            point=read_json(directory/'point.json')
            require(point==dict({k:v for k,v in state.items() if k not in ('state','pack_identity')},state='complete_checked'), 'inner/outer point identity')
            receipt=read_json(directory/'receipt.json')
            if state['kind']=='short':argv=[binary_path,'validate-only',case['id'],state['short_profile'],'1']
            else:argv=[binary_path,'pilot-only' if state['kind']=='pilot' else 'formal-only',case['id'],str(state['iterations']),str(state['seed'])]
            require(receipt['stdout_sha256']==sha(directory/'raw.jsonl') and receipt['stderr_sha256']==sha(directory/'stderr'),'original process outputs')
            key=bind_process(directory,receipt,argv,env,cpu_fixture)
            require(cpu_fixture or key not in keys,'independent process identity');keys.add(key)
            require(last_stop is None or last_stop<=receipt['host_start_ns'],'original ordered process windows');last_stop=receipt['host_stop_ns']
            actual,row=[json.loads(s) for s in (directory/'raw.jsonl').read_text().splitlines()]
            require(actual==device,'device stable across complete family')
            if state['kind']=='short':result=check_short(directory,row,case,device,resources['legacy-capability'],state['short_profile'])
            else:result=check_measured(directory,row,case,device,resource,state['seed'],state['iterations'],state['kind']=='pilot',protocol)
            require(result==read_json(directory/'observation.json'),'independent original observation recalculation')
            decoded.append({'point':state,'row':row,'result':result})
    legal=cases;excluded=[];require(read_json(run/'capacity-exclusions.json')==[],'no unproven resource exclusions')
    expected_all=[(c['id'],profile) for c in cases for profile in profiles_for(c)]
    reuse=[]
    require(read_json(run/'prior-short-reuse.json')=={'coordinates':[],'performance_reused':False},'no unproven prior reuse')
    expected_short=[pair for pair in expected_all if list(pair) not in reuse]
    offset=0;short=decoded[:len(expected_short)]
    require([(v['point']['case_id'],v['point']['short_profile']) for v in short]==expected_short
        and all(v['point']['kind']=='short' and v['point']['seed']==1 for v in short),'complete finite short matrix before pilots')
    fit_cases=[case for case in legal if not case['parameters']['holdout']]
    offset+=len(short);pilots=decoded[offset:offset+len(fit_cases)]
    require([v['point']['case_id'] for v in pilots]==[c['id'] for c in fit_cases]
        and all(v['point']['kind']=='pilot' and v['point']['iterations']==128 and v['point']['seed']==4294967295 for v in pilots),'90 fit-domain pilots before formal; holdout excluded')
    pilot_ms={value['point']['case_id']:value['row']['event_ms'] for value in pilots};maximum_ms=max(pilot_ms.values())
    common_N=max(128,min(65536,int(128*20.0/max(maximum_ms,.01))))
    require(read_json(run/'calibration.json')=={'case_event_ms':pilot_ms,'pilot_iterations':128,'max_event_ms':maximum_ms,
        'common_iterations':common_N,'model':'common_repeat_count_from_fit90_20ms_v1','holdout_used_for_calibration':False},'independent common length from fit domain only')
    resolution=read_json(run/'resolved-cases.json')
    require(resolution=={case['id']:{'iterations':common_N,'model':'common_repeat_count_from_fit90_20ms_v1','holdout_used_for_calibration':False} for case in legal},'all controls and holdout use same N')
    formal=decoded[offset+len(pilots):];require(all(v['point']['kind']=='formal' for v in formal),'no unexpected target kind')
    active=sorted(c['id'] for c in legal);values={key:[] for key in active};outcomes={key:{'status':'pending','batches':[]} for key in active};consumed=[]
    for batch in range(3):
        if not active:break
        rng=random.Random(protocol['shuffle_seed']+batch);order=[]
        for trial in range(10):
            group=list(active);rng.shuffle(group);order.extend((key,trial) for key in group)
        require(canonical(read_json(run/('order-batch%d.json'%batch)))==canonical(order),'fixed original random schedule')
        batch_points=[v for v in formal if v['point']['batch']==batch];lookup={(v['point']['case_id'],v['point']['trial']):v for v in batch_points}
        require(len(lookup)==len(batch_points),'no duplicate process sample')
        current={key:[] for key in active};failed=set()
        for key,trial in order:
            if key in failed:continue
            require((key,trial) in lookup,'missing scheduled process');v=lookup[key,trial];p=v['point'];consumed.append(p['index'])
            require(p['seed']==3+19*trial+1009*batch and p['iterations']==resolution[key]['iterations'],'unchanged N and original seed')
            if v['result']['warmup_converged']:current[key].append(v['result']['value']);values[key].append(v['result']['value'])
            else:failed.add(key)
        next_active=[]
        for key in active:
            merged=stats(values[key]) if len(values[key])>=2 else None
            batch_stats=stats(current[key]) if len(current[key])>=2 else None
            stable=len(current[key])==10 and batch_stats['cv']<=.05 and merged['cv']<=.05
            outcomes[key]['batches'].append({'batch':batch,'complete':len(current[key])==10,'warmup_failed':key in failed,'stats':batch_stats})
            outcomes[key].update(merged=merged,status='stable' if stable else 'unstable_after_bounded_remeasurement' if batch==2 else 'pending')
            if not stable:next_active.append(key)
        active=next_active
    require(consumed==[v['point']['index'] for v in formal],'all batches retained, no extra or selected-best samples')
    require(canonical(outcomes)==canonical(read_json(run/'case-results.json')),'all-case independent statistics')
    summary=read_json(run/'summary.json')
    require(summary=={'coordinate_count':105,'legal_cases':len(legal),'capacity_excluded_cases':len(excluded),
        'prior_short_reused':reuse,'case_results':outcomes,'point_count':len(decoded),'qualified':False,'independent_C_required':True},'complete exact summary')
    return {'coordinate_count':105,'legal_cases':len(legal),'excluded_cases':len(excluded),'formal_process_count':len(formal),
        'cases':outcomes,'all_saved_values_recomputed':True,'CPU_fixture_only':cpu_fixture,'GPU_executed':False,'qualified':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--cpu-fixture',action='store_true')
    args=parser.parse_args();print(json.dumps(evidence(args.run,args.cpu_fixture),indent=2))

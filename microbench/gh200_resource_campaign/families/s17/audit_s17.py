"""Read-only S17 archive replay: full values, launch history and all statistics."""
from pathlib import Path, PurePosixPath
import argparse
import json
import random
import sys

if sys.flags.optimize != 0:
    raise RuntimeError('S17 requires active assertions; Python optimization forbidden')
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'microbench/gh200_resource_campaign'))
from common.suite_io import canonical, read_json, require, sha, stats, validate_gate, verify_files
from auditors.suite import protocol_check
from point_pack import verify_pack
from control import capability_matrix, symbol, check_short, check_measured, bind_process
from run_s17 import SHORT_PACK_LIMIT, OTHER_PACK_LIMIT, raw_payload_bytes


def evidence(run, cpu_fixture=False):
    run=Path(run);env=read_json(run/'initial.json');device=read_json(run/'device.json')
    require(('CPU_fixture' in device['name'])==cpu_fixture and (env.get('CPU_fixture_only') is True)==cpu_fixture,
            'fixture requires explicit unqualified route')
    manifest=read_json(ROOT/'source-manifest.json');verify_files(ROOT,manifest)
    require(read_json(run/'source-manifest.json')==manifest,'original source closure')
    if not cpu_fixture:
        gate=validate_gate(ROOT/'source-review.json',ROOT,'S17','full-value-family-run-source-B')
        require(gate==read_json(run/'source-review.json'),'source review identity')
    else:
        require(read_json(run/'source-review.json').get('CPU_fixture_only') is True,'fixture review cannot enter normal route')
    require(read_json(run/'binary-identity.json')==read_json(ROOT/'binary-identity.json'),'original binary identity')
    binary_path=read_json(run/'binary-path.json')['probe']
    require(Path(binary_path).is_absolute(),'original absolute binary path')
    require(device['uuid'].lower()==env['uuid'].lower(),'original allocated device')
    protocol=read_json(ROOT/'microbench/gh200_resource_campaign/contracts/protocol.json');protocol_check(protocol)
    cases=read_json(ROOT/'cases.json')['cases'];by_id={c['id']:c for c in cases};require(len(by_id)==42,'all42 coordinates')
    capabilities={}
    for name,measured in (('legacy-capabilities',False),('measured-capabilities',True)):
        rows=[json.loads(line) for line in (run/name/'stdout').read_text().splitlines()]
        require(len(rows)==22 and rows[0]==device,'original complete capability matrix')
        receipt=read_json(run/name/'receipt.json')
        require(receipt['argv']==[binary_path,'measured-cluster-device' if measured else 'cluster-device']
                and receipt['stdout_sha256']==sha(run/name/'stdout') and receipt['stderr_sha256']==sha(run/name/'stderr')
                and receipt['returncode']==0 and receipt['cleanup_confirmed'] is True,'successful original zero-target capability query')
        capabilities[name]=capability_matrix(rows[1:],device,measured)
    require(capabilities==read_json(run/'capabilities.json'),'independent capability classification')
    resources={c['id']:capabilities['measured-capabilities'][symbol(c['parameters']['form'],c['parameters']['cluster_size'])] for c in cases}
    decoded=[];keys=set();last_stop=None
    for state_path in sorted(run.glob('point*-state.json')):
        state=read_json(state_path)
        require(state['index']==len(decoded) and state['state']=='verified_node_pack_pending_offhost','no missing/unfinished point')
        case=by_id[state['case_id']];archive=run/'packs'/('point%04d.tar.xz'%state['index']);packet=state['pack_identity']
        require(state['binary_sha256']==read_json(run/'binary-identity.json')['probe'],'each point uses frozen binary')
        require(sha(archive)==packet['archive_sha256'] and archive.stat().st_size==packet['archive_bytes'], 'complete original pack')
        require(packet['archive_bytes']<= (SHORT_PACK_LIMIT if state['kind']=='short' else OTHER_PACK_LIMIT),'frozen packet limit')
        verify_pack(archive,packet['members'])
        with tempfile.TemporaryDirectory(prefix='s17-replay-point-') as temporary:
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
            if state['kind']=='short':argv=[binary_path,'validate-only',case['id'],'cluster_short_1_2_5_v1','3']
            else:argv=[binary_path,'pilot-only' if state['kind']=='pilot' else 'formal-only',case['id'],str(state['iterations']),str(state['seed'])]
            require(receipt['stdout_sha256']==sha(directory/'raw.jsonl') and receipt['stderr_sha256']==sha(directory/'stderr'),'original process outputs')
            key=bind_process(directory,receipt,argv,env,cpu_fixture)
            require(cpu_fixture or key not in keys,'independent process identity');keys.add(key)
            require(last_stop is None or last_stop<=receipt['host_start_ns'],'original ordered process windows');last_stop=receipt['host_stop_ns']
            actual,row=[json.loads(s) for s in (directory/'raw.jsonl').read_text().splitlines()]
            require(actual==device,'device stable across complete family')
            if state['kind']=='short':result=check_short(directory,row,case,device,capabilities['legacy-capabilities'][symbol(case['parameters']['form'],case['parameters']['cluster_size'],False)])
            else:result=check_measured(directory,row,case,device,resources[case['id']],state['seed'],state['iterations'],state['kind']=='pilot',protocol)
            require(result==read_json(directory/'observation.json'),'independent original observation recalculation')
            decoded.append({'point':state,'row':row,'result':result})
    require(resources==read_json(run/'resources.json'),'full queried resource matrix')
    legal=[c for c in cases if resources[c['id']]['supported']];excluded=[c for c in cases if not resources[c['id']]['supported']]
    require(read_json(run/'capacity-exclusions.json')==excluded,'exact zero-target exclusions')
    require(all(capabilities['legacy-capabilities'][symbol(c['parameters']['form'],c['parameters']['cluster_size'],False)]['supported'] for c in legal),'every measured coordinate has supported legacy short')
    reuse=[]
    require(read_json(run/'prior-short-reuse.json')=={'case_ids':[],'performance_reused':False},'no unproven prior reuse')
    offset=0;expected_short=[c['id'] for c in legal]
    short=decoded[offset:offset+len(expected_short)]
    require([v['point']['case_id'] for v in short]==expected_short and all(v['point']['kind']=='short' for v in short),'complete own short matrix')
    offset+=len(short);pilots=decoded[offset:offset+len(legal)]
    require([v['point']['case_id'] for v in pilots]==[c['id'] for c in legal]
            and all(v['point']['kind']=='pilot' and v['point']['iterations']==128 and v['point']['seed']==4294967295 for v in pilots),'all own pilots before formal')
    resolution=read_json(run/'resolved-cases.json')
    for value in pilots:
        key=value['point']['case_id'];ms=value['row']['event_ms']
        require(resolution[key]=={'iterations':max(128,min(65536,int(128*20.0/max(ms,.01)))),
            'pilot_event_ms':ms,'pilot_iterations':128,'model':'event_ms_truncate_clamp20ms_v1'},'own pilot-derived N')
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
    require(summary=={'coordinate_count':42,'legal_cases':len(legal),'capacity_excluded_cases':len(excluded),
        'prior_short_reused':reuse,'case_results':outcomes,'point_count':len(decoded),'qualified':False,'independent_C_required':True},'complete exact summary')
    return {'coordinate_count':42,'legal_cases':len(legal),'excluded_cases':len(excluded),'formal_process_count':len(formal),
        'cases':outcomes,'all_saved_values_recomputed':True,'CPU_fixture_only':cpu_fixture,'GPU_executed':False,'qualified':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--cpu-fixture',action='store_true')
    args=parser.parse_args();print(json.dumps(evidence(args.run,args.cpu_fixture),indent=2))

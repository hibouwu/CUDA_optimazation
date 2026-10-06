"""S19 queried resource and measured observation checks."""
from pathlib import Path
import math,sys
if sys.flags.optimize:raise RuntimeError('S19 requires active assertions; Python optimization forbidden')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'microbench/gh200_resource_campaign'))
from common.suite_io import require,process_ok,read_json
from auditors.async_copy import validate_device
from auditors.s19_validation import validate_resources
from auditors.observation import validate_observation
from reference_formal import verify_values,coordinate


def check_capability(row,device):
    validate_device(device)
    require(row['type']=='s19_capability' and type(row['GPU_target_launches']) is int and row['GPU_target_launches']==0,'zero-target queried resource')
    validate_resources(row['resource_identity'],device)
    require(row['resource_identity']['registers_per_thread']==64,'actual unchanged instruction target register allocation')
    return row['resource_identity']


def check_measured(directory,row,case,device,resource,seed,iterations,pilot,protocol):
    validate_device(device)
    require(row['case_id']==case['id'] and row['iterations']==iterations and row['seed']==seed,'exact measured request')
    require(row['phase']==('pilot' if pilot else 'formal') and row['performance_eligible'] is (not pilot),'pilot/formal boundary')
    for key in ('kernel_symbol','registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm'):
        require(row[key]==resource[key] and (key=='kernel_symbol' or type(row[key]) is int),'same own queried target resources '+key)
    value=verify_values(directory,row)
    # SM count does not bound sparse physical SM identifiers; reference rejects unwritten IDs.
    if pilot:return {'pilot':True,'event_ms':row['event_ms'],'complete_values_verified':True,'qualified':False}
    mode,S,K=coordinate(case['id'],iterations,seed)
    fma=0 if mode=='transport' else 65536*K*iterations;epi=2048*iterations if mode=='output' else 0
    read=0 if mode=='compute' else 8192*K*iterations;write=4096*iterations if mode=='output' else 0
    work=read if mode=='transport' else fma+epi
    expected={'blocks':1,'work':work,'read':read,'write':write,'operations':iterations,
        'correctness':{'method':'S19_complete_input_output_digest_slots_guards_stamp_v1',
            'checked_elements':K*2048+1024+128+8192+96+10,'input_conditions':'periodic_nonuniform_exact_float_input;trace_null;oneCTA'}}
    result=validate_observation(row,dict(case,iterations=iterations),device,seed,protocol,expected)
    require(result['value']==value['primary_value'],'independent local cycle accounting')
    result['clock64_cycles_per_sequence']=(row['blocks_detail'][0]['stop_cycle']-row['blocks_detail'][0]['start_cycle'])/iterations
    return result


def check_short(directory,row,case,device,resource,profile_id):
    import hashlib,json,struct
    from auditors.s19_validation import validate_validation,audit_artifact_values
    contract=json.loads((ROOT/'microbench/gh200_resource_campaign/contracts/s19.json').read_text())
    legacy=next(c for c in contract['cases'] if c['id']==case['id'])
    profiles=json.loads((ROOT/'microbench/gh200_resource_campaign/contracts/s19_profiles.json').read_text())['profiles']
    profile=next(p for p in profiles if p['id']==profile_id)
    require(row['resource_identity']==resource,'short target binds own actual capability query')
    validate_validation(device,row,legacy,profile,1)
    arrays={}
    for item in row['checks'][0]['output_artifacts']:
        path=Path(directory)/item['path'];require(path.is_file() and not path.is_symlink(),'complete short regular artifact')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==item['sha256'],'original short SHA')
        count=item['shape'][0];require(type(count) is int and count>0 and len(raw)==count*4,'exact short full byte count')
        arrays[item['path']]=list(struct.unpack('<'+str(count)+'I',raw))
    return audit_artifact_values(legacy,profile,1,arrays,device)


def bind_process(directory, receipt, argv, env, cpu_fixture=False):
    require(process_ok(receipt) and receipt['argv']==argv and receipt['timeout_seconds']==120
            and receipt['signals']==[], 'original successful process command/timeout/cleanup')
    require(type(receipt['pid']) is int and receipt['pid']==receipt['pgid']>0
            and type(receipt['host_start_ns']) is int and type(receipt['host_stop_ns']) is int
            and receipt['host_stop_ns']>receipt['host_start_ns'], 'independent process identity')
    require((receipt.get('CPU_fixture_only') is True)==cpu_fixture,'explicit fixture boundary')
    if cpu_fixture: return
    reg=receipt['gpu_process_registration'];owner=reg['controller']
    require(type(reg['start_ticks']) is int and reg['start_ticks']>0,'registered process start ticks')
    require(reg['state']=='cleanup_confirmed' and reg['gpu_uuid']==env['uuid'].lower()
        and reg['argv']==argv and reg['pid']==receipt['pid'] and reg['pgid']==receipt['pgid'], 'registered GPU process')
    require(owner['host']==env['host'] and owner['uid']==env['execution_uid']
        and owner['boot_id']==env['boot_id'],'original acquisition controller (portable replay)')
    require(reg['cleanup_receipt']=={k:v for k,v in receipt.items() if k!='gpu_process_registration'},'complete cleanup receipt')
    require(read_json(Path(directory)/'raw.jsonl.registry.json')==reg,'archived original registration')
    active=read_json(Path(directory)/'raw.jsonl.active.json')
    require(active['host']==env['host'] and active['pid']==receipt['pid'] and active['pgid']==receipt['pgid'] and active['argv']==argv
        and active['start_ticks']==reg['start_ticks'] and active['registry_id']==reg['registry_id']
        and active['slurm_job']==env['job'] and active['gpu_uuid'].lower()==env['uuid'].lower(), 'active process allocation')
    return (owner['host'],owner['boot_id'],receipt['pid'],reg['start_ticks'])

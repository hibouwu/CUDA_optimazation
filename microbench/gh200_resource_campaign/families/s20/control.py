"""S20 short and measured targets bind their respective actual resource queries."""
from pathlib import Path
import hashlib,struct,sys
if sys.flags.optimize:raise RuntimeError('S20 requires active assertions; Python optimization forbidden')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'microbench/gh200_resource_campaign'))
from common.suite_io import require,read_json,process_ok
from auditors.async_copy import validate_device
from auditors.observation import validate_observation
from reference_formal import verify_values,coordinate,legacy


def check_capability(row,device,measured):
    validate_device(device)
    require(row['type']==('s20_measured_capability' if measured else 's20_capability')
        and type(row['GPU_target_launches']) is int and row['GPU_target_launches']==0,'own zero-target capability')
    require(row['case_id']=='compute_s1_k1','fixed query representative')
    r=row['resource_identity'];require(r['kernel_symbol']==('s20_pipeline_measured' if measured else 's20_pipeline_v1'),'own actual target symbol')
    for k in ('registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm'):
        require(type(r[k]) is int and r[k]>=0,'typed queried resource '+k)
    require(r['registers_per_thread']==128 and r['dynamic_smem_bytes']==65568 and r['local_size_bytes']==0
        and r['extensions']=={'metadata_bytes':464,'TMA_request_bytes':16384},'compiled registers and exact requested resource')
    require(all(type(v) is int for v in r['extensions'].values()),'typed resource extensions')
    shared=r['static_smem_bytes']+65568;occ=r['occupancy_limit_ctas_per_sm']
    require(shared<=device['smem_per_cta_optin_bytes'] and 0<occ<=16 and occ*shared<=device['smem_per_sm_bytes']
        and occ*128*128<=device['registers_per_sm'],'necessary own occupancy capacity; not actual residency')
    return r


def check_measured(directory,row,case,device,resource,seed,iterations,pilot,protocol):
    validate_device(device)
    require(row['case_id']==case['id'] and row['seed']==seed and row['iterations']==iterations,'exact actual command')
    require(row['phase']==('pilot' if pilot else 'formal') and row['performance_eligible'] is (not pilot)
        and row['warmup_executed'] is (not pilot),'requested pilot/formal role and warmup eligibility')
    for k in ('kernel_symbol','registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm'):
        require(row[k]==resource[k] and (k=='kernel_symbol' or type(row[k]) is int),'actual measured query binding '+k)
    value=verify_values(directory,row)
    if pilot:return {'pilot':True,'event_ms':row['event_ms'],'complete_values_verified':True,'qualified':False}
    c=coordinate(case['id'],iterations,seed);mma=0 if c.mode=='transport' else 524288*c.k_tiles*iterations;epi=8192*iterations if c.mode=='output' else 0
    read=0 if c.mode=='compute' else 16384*c.k_tiles*iterations;write=16384*iterations if c.mode=='output' else 0
    expected={'blocks':1,'work':read if c.mode=='transport' else mma+epi,'read':read,'write':write,'operations':iterations,
        'correctness':{'method':'S20_complete_input_output_digest_slots_metadata_guards_event_v1',
            'checked_elements':c.k_tiles*8192+4096+128+32768+116+112+1,'input_conditions':'periodic_exact_BF16_FP32;trace_null;oneCTA'}}
    result=validate_observation(row,dict(case,iterations=iterations),device,seed,protocol,expected)
    require(result['value']==value['primary_value'],'independent oneCTA cycle work')
    result['clock64_cycles_per_sequence']=(row['blocks_detail'][0]['stop_cycle']-row['blocks_detail'][0]['start_cycle'])/iterations
    return result


def check_short(directory,row,case,device,resource,profile_id):
    validate_device(device)
    require(row['resource_identity']==resource,'short target binds own legacy API resource query')
    require(row['case_id']==case['id'] and row['profile_id']==profile_id and type(row['seed']) is int and row['seed']==1,'exact short request')
    require(row['scope']=='one_cta' and type(row['blocks']) is int and row['blocks']==1
        and type(row['threads']) is int and row['threads']==128,'short oneCTA geometry')
    require(row['errors']==0 and type(row['errors']) is int and row['performance_eligible'] is False
        and row['warmup_executed'] is False and row['pilot_executed'] is False,'short only successful numerical evidence')
    require(row['type']=='validation' and type(row['schema_version']) is int and row['schema_version']==2
        and type(row['validation_schema_version']) is int and row['validation_schema_version']==1
        and type(row['GPU_target_launches']) is int and row['GPU_target_launches']==1,'exact short schema and launch count')
    c,witness=legacy.configuration(case['id'],profile_id,1)
    require(row['target_launches']==[{'launch_index':0,'iterations':c.iterations,'input_profile':profile_id,'threads':128,'blocks':1}],'original short launch')
    require(all(type(row['target_launches'][0][k]) is int for k in ('launch_index','iterations','threads','blocks')),'typed short launch fields')
    check=row['checks'][0];require(len(row['checks'])==1 and check['completed'] is True and type(check['errors']) is int and check['errors']==0
        and check['reference_sha256']=='38615e2cebbf4ef7c30ddefba886705f64346a131dce0a2ad4677f65954e8434','exact reference and completion')
    require(check['verified_CTA_ids']==[0] and type(check['verified_CTA_ids'][0]) is int,'short full CTA IDs')
    require(check['reference_model']=='s20_cpp_integer_matrix_v1' and check['comparison']=='exact'
        and check['tolerance_id'] is None and type(check['checked_elements']) is int and type(check['expected_elements']) is int,'exact typed check semantics')
    roles=[('input.u16le','uint16'),('output.u32le','uint32'),('digest.u32le','uint32'),('slots.u16le','uint16'),('trace_input.u16le','uint16'),('trace_c.u32le','uint32'),('metadata.u32le','uint32'),('guards.u32le','uint32'),('event.u32le','uint32')]
    require([(a['path'],a['dtype']) for a in check['output_artifacts']]==roles,'complete ordered nine original short roles')
    for a in check['output_artifacts']:
        p=Path(directory)/a['path'];require(p.is_file() and not p.is_symlink(),'regular complete short artifact')
        raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==a['sha256'] and len(raw)==a['bytes'],'raw short identity')
        width=2 if a['dtype']=='uint16' else 4 if a['dtype']=='uint32' else 0
        require(width>0 and len(a['shape'])==1 and type(a['shape'][0]) is int and a['shape'][0]*width==len(raw),'complete short shape')
    result=legacy.audit_raw(directory,c,witness)
    require(type(row['event_ms']) in (int,float) and row['event_ms']==result['event_ms'],'event field binds saved float bits')
    require(len(check['output_artifacts'])==9 and check['checked_elements']==check['expected_elements']==result['compared_elements']+116+112+1,'complete nine short artifact roles/counts')
    return result


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

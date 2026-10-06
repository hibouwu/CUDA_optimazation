"""S17 queried cluster capabilities and independently derived service demand."""
from pathlib import Path
import sys
import array
import hashlib
import math
import json
from reference_formal import FORMS,verify_values

if sys.flags.optimize:
    raise RuntimeError('S17 requires active assertions; Python optimization forbidden')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'microbench/gh200_resource_campaign'))
from common.suite_io import require, process_ok, read_json
from auditors.observation import validate_observation
from auditors.async_copy import validate_device
from auditors.cluster_dsm_validation_v2 import validate_validation, audit_values


def symbol(form,C,measured=True):
    name={'bulk_single_target':'bulk_single','bulk_all_targets':'bulk_all'}.get(form,form)
    return ('s17m_' if measured else 's17_')+name+'_c'+str(C)


def capability_matrix(rows,device,measured=True):
    validate_device(device)
    require(len(rows)==21,'complete21 queried form/cluster targets')
    expected={symbol(form,C,measured):(form,C) for C in (2,4,8) for form in FORMS}
    result={}
    for row in rows:
        key=row['kernel_symbol'];require(key in expected and key not in result,'exact unique target symbol')
        form,C=expected[key]
        require(row['type']=='cluster_capability' and row['cluster_size']==C
                and row['occupancy_is_upper_bound'] is True,'original cluster target/capability semantics')
        for field in ('cluster_size','cluster_launch_supported','potential_cluster_size','active_cluster_capacity',
                      'cuda_error','registers_per_thread','static_smem_bytes','local_size_bytes'):
            require(type(row[field]) is int and row[field]>=0,'typed capability '+field)
        require(row['cluster_launch_supported'] in (0,1),'queried cluster support')
        require(row['cuda_error']==0,'CUDA capability query error is not hardware unsupported; preserve original query and stop')
        legal=(row['cuda_error']==0 and row['cluster_launch_supported']==1 and row['potential_cluster_size']>=C
               and 0<row['active_cluster_capacity']<=1024 and row['local_size_bytes']==0)
        if legal:
            require(0<row['registers_per_thread']<=255 and row['static_smem_bytes']<=device['smem_per_cta_optin_bytes'],
                    'necessary supported target resources')
        result[key]={'query':row,'supported':legal,'form':form,'cluster_size':C,
                     'unsupported_reason':None if legal else 'actual_capability_or_resource_not_admitted',
                     'target_launches':0,'actual_residency_proven':False}
    require(set(result)==set(expected),'no capability target omitted')
    return result


def check_measured(directory,row,case,device,capability,seed,iterations,pilot,protocol):
    validate_device(device)
    p=case['parameters'];mode=p['mode'];C=p['cluster_size'];query=capability['query']
    require(capability['supported'] is True,'unsupported coordinate cannot launch')
    require(row['case_id']==case['id'] and row['seed']==seed and row['iterations']==iterations,
            'exact actual request')
    require(row['phase']==('pilot' if pilot else 'formal') and row['performance_eligible'] is (not pilot),
            'pilot/formal eligibility')
    G=1 if case['scope']=='one_cluster' else query['active_cluster_capacity'];B=G*C
    require(row['clusters']==G and row['blocks']==B and row['cluster_size']==C
            and row['kernel_symbol']==query['kernel_symbol'],'own queried cluster grid')
    for key in ('registers_per_thread','static_smem_bytes','local_size_bytes','potential_cluster_size','active_cluster_capacity'):
        require(row[key]==query[key],'unchanged queried target attribute '+key)
    require(row['dynamic_smem_bytes']==0 and row['occupancy_is_upper_bound'] is True,'static SMEM/capacity not residency')
    for detail in row['blocks_detail']:
        require(all(type(detail[key]) is int and detail[key]>=0 for key in
            ('block_id','smid','start_ns','stop_ns','start_cycle','stop_cycle')), 'typed physical CTA timer/detail fields')
    value=verify_values(directory,row,case['id'])
    require(type(row['event_ms']) in (int,float) and math.isfinite(row['event_ms'])
            and row['event_ms']>0 and row['stop_ns']>row['start_ns']
            and row['stop_ns']-row['start_ns']<=row['event_ms']*1.05e6,
            'finite complete-kernel event covers primary service')
    ids={d['smid'] for d in row['blocks_detail']};require(0<len(ids)<=device['sms'],'observed physical SM domain')
    if pilot:
        require(row['warmup_converged'] is False,'pilot does not claim warmup')
        return {'pilot':True,'event_ms':row['event_ms'],'complete_values_verified':True,'qualified':False}
    W=4096 if mode>=5 else 1024 if mode<4 else 0
    requested=B*iterations*4096 if mode<4 else G*iterations*16384 if mode>=5 else 0
    work=G*iterations if mode==4 else requested
    checked=B*22+(B*(W+8) if W else 0)+(B*128 if mode<4 else 0)+(G*32*4096 if mode>=5 else 0)
    expected={'blocks':B,'work':work,'read':requested if mode!=4 else 0,'write':requested if mode in (2,3) else 0,
        'operations':G*iterations if mode==4 or mode>=5 else B*iterations,
        'correctness':{'method':'S17_complete_final_tiles_sums_guards_and_cluster_exit_v1','checked_elements':checked,
            'input_conditions':'nonuniform_uint32_cluster_pattern;capture_false;all participants live through exit'}}
    result=validate_observation(row,dict(case,iterations=iterations),device,seed,protocol,expected)
    require(result['value']==value['primary_value'],'independent primary unit accounting')
    return result


class WordArray(array.array):
    """Preserve complete uint32 data and the prior reference list-slice interface."""
    def __getitem__(self,index):
        value=super().__getitem__(index)
        return list(value) if isinstance(index,slice) else value


def legacy_short_case(case):
    p=case['parameters']
    require(p['mode']==FORMS[p['form']], 'short/measured coordinate mode matches')
    return {'id':case['id'],'form':p['form'],'cluster_size':p['cluster_size'],
        'scope':case['scope'],'parameters':{'form':p['form'],'cluster_size':p['cluster_size']},
        'launch':{'scope':case['scope'],'threads':128}}


def bind_short_query(row,case,capability):
    require(capability['supported'] is True,'short target must have admitted legacy query')
    query=capability['query'];p=case['parameters'];C=p['cluster_size']
    resource=row['resource_identity'];extension=resource['extensions']
    require(query['kernel_symbol']==symbol(p['form'],C,False),'legacy query exact target')
    for key in ('kernel_symbol','registers_per_thread','static_smem_bytes','local_size_bytes'):
        require(resource[key]==query[key],'short target binds own legacy query '+key)
    for key in ('cluster_launch_supported','potential_cluster_size','active_cluster_capacity','cluster_size'):
        require(extension[key]==query[key],'short capability binds legacy query '+key)
    G=1 if case['scope']=='one_cluster' else query['active_cluster_capacity']
    require(extension['clusters']==G and row['blocks']==G*C and row['threads']==128 and row['scope']==case['scope'],
            'short grid derived from own legacy query')


def check_short(directory,row,case,device,capability):
    profile_path=ROOT/'microbench/gh200_resource_campaign/contracts/cluster_dsm_multicast_validation_profiles_v1.draft.json'
    profile=json.loads(profile_path.read_text())['profiles'][0]
    short_case=legacy_short_case(case)
    bind_short_query(row,case,capability)
    validate_validation(device,row,short_case,profile,3)
    arrays={}
    for check in row['checks']:
        for item in check['output_artifacts']:
            path=Path(directory)/item['path']
            require(path.is_file() and not path.is_symlink(),'complete short regular artifact')
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            require(h.hexdigest()==item['sha256'],'original short artifact SHA')
            count=1
            for size in item['shape']:
                require(type(size) is int and size>0,'typed complete short shape')
                count*=size
            require(path.stat().st_size==count*4,'exact short byte length including EOF')
            values=WordArray('I')
            with path.open('rb') as stream:values.fromfile(stream,count)
            if sys.byteorder!='little':values.byteswap()
            require(item['path'] not in arrays,'unique short artifact')
            arrays[item['path']]=values
    return audit_values(device,row,short_case,profile,3,arrays)


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

"""S16 launch, resource and independent final-state accounting."""
from pathlib import Path
import array
import hashlib
import json
import math
import sys
from reference_formal import verify_values

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'microbench/gh200_resource_campaign'))
from common.suite_io import require, read_json, process_ok
from auditors.observation import validate_observation
from auditors.tma_stage_request_validation_v1 import validate_validation, audit_values
from auditors.tma_stage_request_sass_baseline_v1 import BASELINE


class WordArray(array.array):
    """Compact complete data; keep the prior reader's list-slice interface."""
    def __getitem__(self, index):
        value=super().__getitem__(index)
        return list(value) if isinstance(index,slice) else value


def check_resources(row, case, device):
    p = case['parameters']
    symbol = f"ts_{'g2s' if p['direction']=='gmem_to_smem' else 's2g'}_s{p['software_stages']}_r{p['requests_per_item']}"
    shared = p['software_stages'] * p['requests_per_item'] * 16384 + 8*p['software_stages'] + 32
    require(row['type'] == 'resources' and row['case_id'] == case['id'] and row['target_launches'] == 0,
            'resource query only, no target launches')
    for key in ('target_launches', 'software_stages', 'requests_per_item', 'dynamic_smem_bytes',
                'static_smem_bytes', 'registers_per_thread', 'local_size_bytes', 'occupancy_limit_ctas_per_sm'):
        require(type(row[key]) is int and row[key] >= 0, 'typed resource: '+key)
    require(row['kernel_symbol'] == symbol and row['software_stages'] == p['software_stages']
            and row['requests_per_item'] == p['requests_per_item'] and row['dynamic_smem_bytes'] == shared,
            'original resource coordinate')
    require(row['static_smem_bytes'] == 0 and row['local_size_bytes'] == 0
            and row['registers_per_thread'] == BASELINE[symbol]['registers_per_thread'], 'actual compiled baseline')
    cap = device['smem_per_cta_optin_bytes']
    require(row['smem_per_cta_optin_bytes'] == cap and type(row['legal']) is bool, 'own device capacity')
    legal = shared <= cap
    require(row['legal'] is legal, 'capacity classification must not substitute coordinate')
    occ = row['occupancy_limit_ctas_per_sm']
    if legal:
        require(occ > 0 and occ*shared <= device['smem_per_sm_bytes']
                and occ*128*row['registers_per_thread'] <= device['registers_per_sm']
                and occ*128 <= 2048, 'necessary queried occupancy bounds')
        require(row['reason'] == 'legal_capacity_and_occupancy_bound', 'resource admission reason')
    else:
        require(occ == 0 and row['reason'] == 'shared_capacity_exceeded', 'zero-launch capacity exclusion')
    return legal


def check_short(directory, row, case, device):
    profile = read_json(ROOT / 'microbench/gh200_resource_campaign/contracts/tma_stage_request_validation_profiles_v1.draft.json')['profiles'][0]
    validate_validation(device, row, case, profile, 3)
    arrays = {}
    for check in row['checks']:
        for item in check['output_artifacts']:
            path = Path(directory) / item['path']
            require(path.is_file() and not path.is_symlink(), 'full short regular artifact')
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
            require(h.hexdigest() == item['sha256'], 'original short artifact SHA')
            count=1
            for size in item['shape']:
                require(type(size) is int and size>0,'typed short shape')
                count*=size
            require(path.stat().st_size==count*4,'exact full short bytes; no trailing bytes')
            values = WordArray('I')
            with path.open('rb') as stream: values.fromfile(stream, path.stat().st_size // 4)
            if sys.byteorder != 'little': values.byteswap()
            arrays[item['path']] = values
    return audit_values(device, row, case, profile, 3, arrays)


def check_measured(directory, row, case, device, resource, seed, iterations, pilot, protocol):
    require(row['case_id'] == case['id'] and row['seed'] == seed and row['iterations'] == iterations,
            'actual requested N/seed/case')
    require(row['phase'] == ('pilot' if pilot else 'formal') and row['performance_eligible'] is (not pilot),
            'pilot/formal distinction')
    for key in ('registers_per_thread', 'static_smem_bytes', 'dynamic_smem_bytes',
                'local_size_bytes', 'occupancy_limit_ctas_per_sm', 'kernel_symbol'):
        require(row[key] == resource[key], 'same queried target resource: '+key)
    blocks = 1 if case['scope']=='one_cta' else device['sms']*min(4,resource['occupancy_limit_ctas_per_sm'])
    require(row['blocks'] == blocks, 'own resource-derived grid')
    verify_values(directory, row, case['id'])
    ids = {d['smid'] for d in row['blocks_detail']}
    require(len(ids) == (1 if case['scope']=='one_cta' else device['sms']), 'actual SM coverage')
    elapsed = row['stop_ns']-row['start_ns']
    require(type(row['event_ms']) in (int,float) and math.isfinite(row['event_ms'])
            and elapsed > 0 and row['event_ms'] > 0 and elapsed <= row['event_ms']*1.05e6,
            'complete-kernel event covers primary window')
    if pilot:
        require(row['warmup_samples_ns'] == [] and row['warmup_converged'] is False, 'pilot has no warmup')
        return {'pilot': True, 'event_ms': row['event_ms'], 'qualified': False}
    p=case['parameters'];work=blocks*iterations*p['requests_per_item']*16384
    checked=blocks*32*p['requests_per_item']*4096+8+blocks*p['software_stages']*p['requests_per_item']*4096+blocks*34
    expected={'blocks':blocks,'work':work,'read':work if p['direction']=='gmem_to_smem' else 0,
        'write':0 if p['direction']=='gmem_to_smem' else work,'operations':blocks*iterations*p['requests_per_item'],
        'correctness':{'method':'S16_complete32slot_ring_finalSMEM_allCTA_counts_and_stamps_v1',
            'checked_elements':checked,'input_conditions':'nonuniform_uint32_stage_request_pattern;capture_false;32slot_ring;exact'}}
    return validate_observation(row,dict(case,iterations=iterations),device,seed,protocol,expected)


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

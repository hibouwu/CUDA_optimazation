"""Unregistered S14 formal-host auditor; no admission or GPU execution."""
from auditors import tma_bulk
from auditors.observation import validate_observation
from common.suite_io import require

METHOD='host_final_tile_or_full_ring_and_guards_lifecycle'
INPUTS='32-slot nonuniform modular word pattern; poison reset every launch; capture=false'
validate_device=tma_bulk.validate_device

def validate_trial(row,case,device,seed,protocol):
    p=tma_bulk.case_identity(case);q=p['payload_bytes'];g2s=p['direction']=='gmem_to_smem'
    occ=row.get('occupancy_limit_ctas_per_sm');regs=row.get('registers_per_thread')
    require(type(occ) is int and 1<=occ<=16 and type(regs) is int and 0<regs<=255,'actual formal resources')
    total=row.get('static_smem_bytes',-1)+row.get('dynamic_smem_bytes',-1)
    require(row.get('kernel_symbol')==tma_bulk.kernel_symbol(case) and row.get('dynamic_smem_bytes')==q+32
            and row.get('local_size_bytes')==0 and type(row.get('static_smem_bytes')) is int and row['static_smem_bytes']>=0
            and total<=device['smem_per_cta_optin_bytes'] and occ*total<=device['smem_per_sm_bytes']
            and occ*128*regs<=device['registers_per_sm'] and occ*128<=2048,'formal resource capacity')
    blocks=1 if case['scope']=='one_cta' else device['sms']*min(4,occ)
    n=row.get('iterations');require(type(n) is int and 1<=n<=65536,'calibrated host length domain')
    require(row.get('payload_bytes')==q and row.get('global_slots_per_cta')==32 and row.get('global_allocation_bytes')==tma_bulk.allocation_bytes(blocks,q)
            and row.get('capture_enabled') is False and row.get('completed_requests')==blocks*n and row.get('timeout_count')==0,'formal allocation/lifecycle/capture')
    amount=tma_bulk.work_count(blocks,n,q)
    expected={'blocks':blocks,'read':amount if g2s else 0,'write':0 if g2s else amount,'work':amount,'operations':blocks*n,
              'correctness':{'method':METHOD,'checked_elements':blocks*(q//4)*(1 if g2s else 32)+8,'input_conditions':INPUTS}}
    effective={**case,'iterations':n}
    return validate_observation(row,effective,device,seed,protocol,expected)

SHORT_PAIRS=((1,0),(2,3),(33,4294967295))

def short_blocks(device,case,row):
    tma_bulk.validate_device(device)
    resource=row.get('resource_identity',{});normalized={**resource,'extensions':dict(resource.get('extensions',{}))}
    require(normalized['extensions'].pop('capture_enabled',None) is False and normalized['extensions'].get('validation_role')=='formal_final_transport','formal-path explicit capture/role')
    normalized['extensions']['validation_role']='short_transport'
    blocks=tma_bulk.validate_resources(normalized,case,{'id':'bulk_short_1_seed0_v1'},device,0)
    require(type(row.get('blocks')) is int and row['blocks']==blocks,'formal-path actual-resource grid')
    return blocks


def audit_formal_path_values(device,case,row,arrays):
    """Independent final-word replay after callers verify artifact hashes.

    This checks the capture=false path only; it grants no family qualification.
    """
    p=tma_bulk.case_identity(case);g2s=p['direction']=='gmem_to_smem';words=p['payload_bytes']//4
    tma_bulk.validate_device(device)
    require(type(row.get('schema_version')) is int and row['schema_version']==2 and row.get('type')=='validation'
            and row.get('validation_schema_version')==1 and row.get('case_id')==case['id']
            and row.get('profile_id')=='formal_final_1_2_33_v1' and type(row.get('seed')) is int and row['seed']==3
            and row.get('warmup_executed') is False and row.get('pilot_executed') is False
            and row.get('performance_eligible') is False and type(row.get('errors')) is int and row['errors']==0
            and type(row.get('threads')) is int and row['threads']==128 and row.get('scope')==case['scope'],'formal-path short identity/no performance')
    blocks=short_blocks(device,case,row)
    require(len(row.get('checks',[]))==3 and len(row.get('target_launches',[]))==3 and len(arrays)==3,'three exact paired launches')
    for index,((length,seed),check,values) in enumerate(zip(SHORT_PAIRS,row['checks'],arrays)):
        launch=row['target_launches'][index]
        require(launch=={'launch_index':index,'iterations':length,'input_profile':'paired_nonuniform_final_tile','threads':128,'blocks':blocks}
                and all(type(launch.get(k)) is int for k in ('launch_index','iterations','threads','blocks'))
                and type(check.get('launch_index')) is int and check['launch_index']==index and check.get('completed') is True,'short fixed pair/complete')
        prefix=f'formal_{index}_'
        layouts={prefix+'completion.u32le':[blocks,5,2],prefix+'stamps.u32le':[blocks,5,2],prefix+'guards.u32le':[2,4],
                 prefix+('final_tile.u32le' if g2s else 'ring.u32le'):([blocks,words] if g2s else [blocks,32,words])}
        require(set(values)==set(layouts),'exact final/lifecycle/guard artifacts')
        descriptors=check.get('output_artifacts',[])
        require(len(descriptors)==len(layouts) and {a.get('path') for a in descriptors}==set(layouts),'declared artifact identity')
        for a in descriptors:
            require(a.get('dtype')=='uint32' and a.get('shape')==layouts[a['path']] and isinstance(a.get('sha256'),str) and len(a['sha256'])==64,'declared artifact shape/digest')
        parsed={name:tma_bulk._u32_values(values[name],shape) for name,shape in layouts.items()}
        require(list(parsed[prefix+'guards.u32le'])==tma_bulk.GUARDS,'short guard corruption')
        counts=tma_bulk._u64s(parsed[prefix+'completion.u32le'])
        require(check.get('checked_elements')==blocks*words*(1 if g2s else 32)+8,'short exact checked quantity')
        stamps=tma_bulk._u64s(parsed[prefix+'stamps.u32le'])
        for block in range(blocks):
            done,attempts,timeout,release,full=counts[block*5:block*5+5]
            require(done==length and timeout==0 and release==0 and full==(0 if g2s else length),'formal-path complete lifecycle')
            require(length<=attempts<2**64-1 if g2s else attempts==0,'formal-path attempt counter')
            start_ns,stop_ns,start_cycle,stop_cycle,smid=stamps[block*5:block*5+5]
            require(all(type(v) is int and 0<=v<2**64-1 for v in (start_ns,stop_ns,start_cycle,stop_cycle))
                    and type(smid) is int and 0<=smid<2**32-1 and start_ns<=stop_ns and start_cycle<=stop_cycle,'formal-path exact uint64 stamps/uint32 SMID')
            data=parsed[prefix+('final_tile.u32le' if g2s else 'ring.u32le')]
            if g2s:
                for word in range(words):
                    expected=(17*((block*32+(length-1)%32)*words+word)+seed)&0xffffffff
                    require(data[block*words+word]==expected,'capture=false final tile mismatch')
            else:
                for slot in range(32):
                    for word in range(words):
                        expected=(29*(block*words+word)+seed)&0xffffffff
                        if length<32 and slot>=length:expected^=0xffffffff
                        require(data[(block*32+slot)*words+word]==expected,'formal-path entire destination ring mismatch')
    return {'status':'pass','target_launches':3,'checked_elements':sum(c['checked_elements'] for c in row['checks']),
            'performance_eligible':False,'family_B3_eligible':False}

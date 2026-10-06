"""S14 finite contract, lossless full artifacts and target-scoped lowering checks."""
from __future__ import annotations
import hashlib
import json
import re
from common.suite_io import require
from auditors.memory_baseline import validate_device as base_device
from auditors.tma_bulk_sass_baseline import BASELINE

ADAPTER_ID='tma_bulk_v2'
SIZES=(1024,4096,8192,16384,32768,65536)
SHORT_PROFILES={'bulk_short_1_seed0_v1':(1,0),'bulk_short_2_seed3_v1':(2,3),'bulk_short_33_seed4294967295_v1':(33,4294967295)}
RELEASE_PROFILE='bulk_source_release_one_request_v1'
GUARDS=[0xd15ea5e0+i for i in range(8)]


def equal_json(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def case_identity(case):
    require(isinstance(case,dict) and isinstance(case.get('parameters'),dict),'S14 case/parameters')
    p=case['parameters'];direction=p.get('direction');q=p.get('payload_bytes');scope=case.get('scope');g2s=direction=='gmem_to_smem'
    require(direction in ('gmem_to_smem','smem_to_gmem') and type(q) is int and q in SIZES and scope in ('one_cta','all_gpu'),'finite S14 coordinate')
    require(case.get('id')==f'{direction}_{q//1024}kib_{scope}' and type(case.get('threads')) is int and case['threads']==128 and type(case.get('iterations')) is int and case['iterations']==32,'case identity/128threads/pilot32')
    expected={'direction':direction,'payload_bytes':q,'copies_per_iteration':1,'issuer_thread':0,'stages':1,'requests_per_commit_group':None if g2s else 1,
              'global_slots_per_cta':32,'shared_payload_bytes':q,'shared_control_bytes':32,'source_alignment_bytes':16,'destination_alignment_bytes':16,
              'mbarrier_alignment_bytes':8,'expected_arrivals':1 if g2s else None,'expected_transaction_bytes_per_phase':q if g2s else None,
              'formal_completion':'mbarrier_acquire_then_CTA_gate' if g2s else 'bulk_wait_group_0_then_CTA_gate',
              'ptx':'cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes' if g2s else 'cp.async.bulk.global.shared::cta.bulk_group',
              'tensor_map':False,'source_reuse_diagnostic':not g2s}
    require(equal_json(p,expected),'exact payload/arrival/tx/slot/primitive/lifecycle')
    require(case.get('launch')==({'kind':'one_cta'} if scope=='one_cta' else {'kind':'sms_capped_occupancy','maximum_ctas_per_sm':4}),'original scope geometry')
    require(case.get('capabilities')=={'cc':'9.0','device_name_contains':'GH200'},'target capability')
    return p


def validate_contract(contract):
    require(contract.get('schema_version')==2 and contract.get('family')=='tma_bulk' and contract.get('stage')=='S14' and contract.get('adapter_id')==ADAPTER_ID,'S14 identity')
    ids=[]
    for c in contract['cases']:
        case_identity(c);ids.append(c['id'])
        require(c.get('work_model')=='tma_bulk_completed_transport_v1' and c.get('work_unit')=='byte','transport Q counted once')
        require(c.get('coverage_policy')==('one_sm' if c['scope']=='one_cta' else 'exact_sms'),'formal scope coverage')
    require(len(ids)==24 and len(set(ids))==24,'all24 formal coordinates')
    return contract


def validate_device(device):
    base_device(device)
    for name in ('global_memory_bytes','l2_cache_bytes','smem_per_sm_bytes','smem_per_cta_optin_bytes','registers_per_sm'):
        require(type(device.get(name)) is int and device[name]>0,'positive device capacity '+name)
    return device


def profile_coordinate(profile,case,seed):
    p=case_identity(case);name=profile['id'];release=name==RELEASE_PROFILE
    require(name in SHORT_PROFILES or release,'finite S14 profile')
    length,wanted=(1,3) if release else SHORT_PROFILES[name]
    require(type(seed) is int and seed==wanted and (not release or p['direction']=='smem_to_gmem'),'paired profile/seed/direction')
    return length,release


def kernel_symbol(case,release=False):
    p=case_identity(case);kind='release' if release else 'g2s' if p['direction']=='gmem_to_smem' else 's2g'
    return f'tb_{kind}_q{p["payload_bytes"]}'


def allocation_bytes(blocks,payload):
    require(type(blocks) is int and blocks>0 and type(payload) is int and payload in SIZES,'allocation coordinate')
    amount=blocks*32*payload+32;require(amount<=2**64-1,'uint64 guarded allocation');return amount


def work_count(blocks,iterations,payload):
    allocation_bytes(blocks,payload);require(type(iterations) is int and 1<=iterations<=65536,'iteration domain')
    total=blocks*iterations*payload;require(total<=2**64-1,'uint64 transported byte count');return total


def artifact_layouts(case,profile,blocks,seed):
    length,release=profile_coordinate(profile,case,seed);words=case['parameters']['payload_bytes']//4
    layouts={'bulk_guards.u32le':[2,4],'bulk_completion.u32le':[blocks,5,2]}
    if release:layouts.update({'bulk_source_after.u32le':[blocks,words],'bulk_release_clocks.u32le':[blocks,3,2]})
    else:layouts['bulk_trace.u32le']=[blocks,length,words]
    if case['parameters']['direction']=='smem_to_gmem':layouts['bulk_ring.u32le']=[blocks,32,words]
    return layouts


def expected_elements(case,profile,blocks,seed):
    length,release=profile_coordinate(profile,case,seed);words=case['parameters']['payload_bytes']//4
    return blocks*33*words+8 if release else blocks*length*words+(blocks*32*words if case['parameters']['direction']=='smem_to_gmem' else 0)+8


def validate_resources(resource,case,profile,device,seed):
    length,release=profile_coordinate(profile,case,seed);p=case['parameters'];q=p['payload_bytes'];g2s=p['direction']=='gmem_to_smem'
    require(isinstance(resource,dict) and set(resource)=={'kernel_symbol','registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm','extensions'},'complete resource identity')
    for name in ('registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm'):
        require(type(resource[name]) is int and resource[name]>=0,'resource integer '+name)
    occ=resource['occupancy_limit_ctas_per_sm'];total=resource['static_smem_bytes']+resource['dynamic_smem_bytes']
    require(resource['kernel_symbol']==kernel_symbol(case,release) and 0<resource['registers_per_thread']<=255 and resource['dynamic_smem_bytes']==q+32 and resource['local_size_bytes']==0,'actual specialization/Q+32/no-spill')
    require(1<=occ<=16 and occ*128<=2048 and total<=device['smem_per_cta_optin_bytes'] and occ*total<=device['smem_per_sm_bytes'] and occ*128*resource['registers_per_thread']<=device['registers_per_sm'],'necessary shared/register/thread occupancy capacities')
    blocks=1 if case['scope']=='one_cta' else device['sms']*min(4,occ)
    allocation=allocation_bytes(blocks,q);capture=blocks*q*(1 if release else length);bookkeeping=blocks*(40+5*8+(3*8 if release else 0))
    require(allocation+capture+bookkeeping<=device['global_memory_bytes'],'necessary memory capacity without reducing32slots')
    kind='separate_source_release_then_full_wait' if release else p['formal_completion']
    require(equal_json(resource['extensions'],{'payload_bytes':q,'global_slots_per_cta':32,'global_allocation_bytes':allocation,'completion_kind':kind,'validation_role':'source_release' if release else 'short_transport'}),'frozen allocation/completion role')
    return blocks


def _u32_values(item,shape):
    require(isinstance(item,dict) and set(item)=={'shape','values'} and equal_json(item['shape'],shape),'artifact shape/value object')
    count=1
    for dim in shape:require(type(dim) is int and dim>0,'positive artifact shape');count*=dim
    values=item['values'];require(len(values)==count,'complete artifact word count')
    # values may be a zero-copy uint32 memoryview supplied by an offline reader.
    require(all(type(v) is int and 0<=v<=0xffffffff for v in values),'uint32 artifact words')
    return values


def _u64s(words):
    require(len(words)%2==0,'complete low/high pairs')
    return [words[i]|(words[i+1]<<32) for i in range(0,len(words),2)]


def audit_artifact_values(case,profile,seed,arrays):
    """Pure full-value replay. Caller reads and hash-checks files before passing arrays.

    ABI1 itself cannot receive file bytes. This separate pure check never grants
    a family gate and never treats a metadata-only ABI pass as value replay.
    """
    length,release=profile_coordinate(profile,case,seed)
    require(isinstance(arrays,dict) and 'bulk_completion.u32le' in arrays,'completion artifact required')
    shape=arrays['bulk_completion.u32le'].get('shape',[])
    require(isinstance(shape,list) and len(shape)==3 and shape[1:]==[5,2] and type(shape[0]) is int and shape[0]>0,'completion array shape')
    blocks=shape[0];layouts=artifact_layouts(case,profile,blocks,seed);require(set(arrays)==set(layouts),'exact full artifact set')
    parsed={name:_u32_values(arrays[name],layout) for name,layout in layouts.items()}
    require(list(parsed['bulk_guards.u32le'])==GUARDS,'global guard corruption')
    p=case['parameters'];words=p['payload_bytes']//4;g2s=p['direction']=='gmem_to_smem';counts=_u64s(parsed['bulk_completion.u32le'])
    for b in range(blocks):
        done,attempts,timeout,read_wait,full_wait=counts[5*b:5*b+5]
        require(done==length and timeout==0 and read_wait==int(release) and full_wait==(0 if g2s else length),'complete original request lifecycle')
        require(length<=attempts<2**64-1 if g2s else attempts==0,'mbarrier wait attempt evidence')
    if release:
        after=parsed['bulk_source_after.u32le'];times=_u64s(parsed['bulk_release_clocks.u32le'])
        for b in range(blocks):
            start,released,full=times[b*3:b*3+3];require(0<=start<=released<=full<2**64-1,'same-CTA clocks complete and ordered; zero gap allowed')
            for word in range(words):require(after[b*words+word]==((29*(b*words+word)+seed)&0xffffffff)^0xffffffff,'source complement not fully written')
    else:
        trace=parsed['bulk_trace.u32le']
        for b in range(blocks):
            for iteration in range(length):
                # Explicit 32-slot ownership; word arithmetic wraps, addresses do not.
                base=(b*32+(iteration%32))*words if g2s else b*words;scale=17 if g2s else 29
                for word in range(words):require(trace[(b*length+iteration)*words+word]==(scale*(base+word)+seed)&0xffffffff,'per-request complete word output')
    if not g2s:
        ring=parsed['bulk_ring.u32le'];visited={i%32 for i in range(length)}
        for b in range(blocks):
            for slot in range(32):
                for word in range(words):
                    value=(29*(b*words+word)+seed)&0xffffffff;expected=value if slot in visited else value^0xffffffff
                    require(ring[(b*32+slot)*words+word]==expected,'full ring including untouched poison')
    result={'status':'pass','case_id':case['id'],'profile_id':profile['id'],'blocks':blocks,
            'checked_elements':expected_elements(case,profile,blocks,seed),'full_value_replay':True,'performance_eligible':False}
    if release:
        result['source_release_cycles_per_CTA']=[times[3*b+1]-times[3*b] for b in range(blocks)]
        result['full_completion_cycles_per_CTA']=[times[3*b+2]-times[3*b] for b in range(blocks)]
        result['time_unit']='clock64_cycle/request/CTA'
        result['full_endpoint_includes_source_overwrite_and_diagnostic_barriers']=True
    return result


def complete_profile_set(contract,records):
    """Pure finite coverage check, not a scheduler and not B3 authorization."""
    validate_contract(contract);expected={(c['id'],p,s) for c in contract['cases'] for p,(_,s) in SHORT_PROFILES.items()}
    expected|={(c['id'],RELEASE_PROFILE,3) for c in contract['cases'] if c['parameters']['direction']=='smem_to_gmem'}
    require(isinstance(records,list) and len(records)==84,'all72 short plus12 source-release receipts')
    keys=[];identities=[]
    for record in records:
        key=(record.get('case_id'),record.get('profile_id'),record.get('seed'));keys.append(key)
        require(type(record.get('seed')) is int and key in expected and record.get('status')=='pass' and record.get('full_value_replay') is True,'verified exact profile/seed/full values')
        # Actual binary hashes remain per-receipt; separate snapshot build paths
        # can alter debug metadata. Required shared semantics are explicit.
        identity=record.get('semantic_identity');require(isinstance(identity,dict) and set(identity)=={'contract_sha256','profiles_sha256','reference_sha256','source_sha256','normalized_target_set_sha256','device_uuid','compiler_identity_sha256'},'full semantic identity')
        for name,value in identity.items():
            require(isinstance(value,str) and (re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',value) if name=='device_uuid' else re.fullmatch(r'[0-9a-f]{64}',value)),'semantic identity hash/UUID')
        identities.append(identity)
        require(isinstance(record.get('receipt_sha256'),str) and re.fullmatch(r'[0-9a-f]{64}',record['receipt_sha256']),'bound independent receipt')
    require(len(set(keys))==len(keys) and set(keys)==expected,'no missing/replaced/duplicate profile coordinate')
    require(len({r['receipt_sha256'] for r in records})==84,'independent receipt required for every profile process')
    require(all(equal_json(identities[0],v) for v in identities[1:]),'cross-profile semantic identity drift')
    return {'status':'validation_collected_pending_B3_review','formal_cases':24,'short_receipts':72,'source_release_receipts':12,'performance_eligible':False,'pilot_authorized':False}


def _ops(body):
    result=[]
    for line in body.splitlines():
        m=re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?)(?:/\*|$)',line)
        if m:result.append((int(m[1],16),m[2].strip().rstrip(';').strip()))
    require(result,'actual target SASS required');return result


def audit_sass(text,contract):
    """Target-scoped structural evidence; independent actual B2 remains required."""
    validate_contract(contract);parts=re.split(r'Function\s*:\s*',text)[1:];out=[]
    for q in SIZES:
        for mode in ('g2s','s2g','release'):
            symbol=f'tb_{mode}_q{q}';found=[part for part in parts if part.splitlines()[0].strip()==symbol];require(len(found)==1,'missing/ambiguous '+symbol)
            ops=_ops(found[0]);stream='\n'.join(f'{pc:04x}: {op}' for pc,op in ops)
            identity={'instruction_count':len(ops),'instructions_sha256':hashlib.sha256(stream.encode()).hexdigest()}
            require(identity==BASELINE[symbol],'target instruction identity drift; new independent B2 required: '+symbol)
            global_timers=[pc for pc,op in ops if 'SR_GLOBALTIMERLO' in op]
            bulk=[(pc,op) for pc,op in ops if 'UBLKCP.' in op]
            require(len(bulk)==1 and ('UBLKCP.S.G' in bulk[0][1] if mode=='g2s' else 'UBLKCP.G.S' in bulk[0][1]),'one proper bulk direction per target loop/request')
            require(not bulk[0][1].startswith('@!PT'),'never-executed bulk rejected')
            require(len(global_timers)>=2,'outer time endpoints; internal timeout timers retained')
            bars=[pc for pc,op in ops if op.startswith('BAR.SYNC')]
            require(any(pc<bulk[0][0] for pc in bars) and any(pc>bulk[0][0] for pc in bars),'CTA preparation/completion gates')
            if mode=='g2s':
                expected=[(pc,op) for pc,op in ops if 'SYNCS.ARRIVE.TRANS64.RED.A0TR' in op]
                arrived=[(pc,op) for pc,op in ops if 'SYNCS.ARRIVE.TRANS64.A1T0' in op]
                waited=[pc for pc,op in ops if 'SYNCS.PHASECHK.TRANS64' in op]
                invalid=[pc for pc,op in ops if 'SYNCS.CCTL.IV' in op]
                require(len(expected)==len(arrived)==len(invalid)==1 and expected[0][0]<bulk[0][0]<arrived[0][0]<min(waited),'expect/copy/one-arrival/token-wait order')
                require(any(pc>max(waited) for pc in bars) and max(waited)<invalid[0],'final rendezvous before invalidation')
                require(any('BPT.TRAP' in op for _,op in ops),'fatal timeout trap required')
            else:
                commits=[pc for pc,op in ops if op=='UTMACMDFLUSH'];waits=[pc for pc,op in ops if op=='DEPBAR.LE SB0, 0x0'];full=[pc for pc,op in ops if op=='CCTL.IVALL']
                require(len(commits)==len(full)==1 and len(waits)==(2 if mode=='release' else 1),'issuer commit/readwait/fullwait lowering')
                require(bulk[0][0]<commits[0]<min(waits)<=max(waits)<full[0],'bulk commit then complete wait')
                if mode=='release':
                    overwrite=[pc for pc,op in ops if re.search(r'\bSTS(?:\.|\s)',op) and min(waits)<pc<max(waits)]
                    require(overwrite and any(min(waits)<pc<min(overwrite) for pc in bars) and any(max(overwrite)<pc<max(waits) for pc in bars),'source overwrite only between read and full wait with CTA gates')
                    require(sum('SR_CLOCKLO' in op for _,op in ops)>=3,'three local diagnostic endpoints')
            require(not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)',op) for _,op in ops),'no local spill')
            out.append({'symbol':symbol,'instruction_identity':identity,'payload_bytes':q,'mode':mode,'bulk':bulk,'globaltimer_pcs':global_timers,'CTA_barrier_pcs':bars,
                        'requires_independent_actual_token_Q_control_and_timeout_review':True,'GPU_qualification':False})
    return out

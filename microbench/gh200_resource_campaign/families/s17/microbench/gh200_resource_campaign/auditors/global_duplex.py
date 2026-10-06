"""Independent S13 demand/address and conservative target-SASS checks."""
from __future__ import annotations
import re
from common.suite_io import require
from auditors.memory_baseline import validate_device as base_device, timed_loops
from auditors.observation import validate_observation

ADAPTER_ID='global_duplex_v2'
MODES=[('read','ca',1,0),('read','cg',1,0),('write',None,0,1),('copy','cg',1,1),
       ('independent','cg',1,1),('independent','cg',2,1),('independent','cg',4,1),
       ('independent','cg',1,2),('independent','cg',1,4)]
METHOD='all_nonempty_thread_read_checksums_and_all_destination_words'
INPUTS='read=uint32(17*word_index+seed); independent_write=uint32(29*word_index+seed); copy=source; poison each launch'


def case_identity(case):
    require(isinstance(case,dict) and isinstance(case.get('parameters'),dict),'S13 case/parameters')
    p=case['parameters'];mode=p.get('mode');cache=p.get('read_cache_policy');r=p.get('read_requests_per_group');w=p.get('write_requests_per_group');size=p.get('working_set_class')
    require(type(r) is int and type(w) is int and (mode,cache,r,w) in MODES and size in ('small','large'),'finite S13 matrix')
    prefix='read_'+cache if mode=='read' else 'write' if mode=='write' else 'dependent_copy' if mode=='copy' else f'independent_r{r}_w{w}'
    require(case.get('id')==prefix+'_'+size and case.get('threads')==256 and case.get('iterations')==16 and case.get('scope')=='all_gpu','S13 identity/length/participants')
    require(case.get('launch')=={'kind':'sms_capped_occupancy','maximum_ctas_per_sm':4},'frozen occupancy grid policy')
    expected={'mode':mode,'read_cache_policy':cache,'write_cache_policy':'wb' if w else None,
              'read_requests_per_group':r,'write_requests_per_group':w,'vector_bytes':16,'working_set_class':size,
              'requested_bytes_per_active_array':'floor(device.l2_cache_bytes/4)' if size=='small' else '4*device.l2_cache_bytes',
              'allocation_alignment':'blocks*threads*16','allocation_bytes_per_active_array':'ceil(requested/alignment)*alignment',
              'input_word':'uint32(17*word_index+seed)','independent_store_word':'uint32(29*word_index+seed)','copy_output':'exact source word'}
    require(p==expected,'frozen address/cache/input conditions')
    require(case.get('capabilities')=={'cc':'9.0','device_name_contains':'GH200'},'S13 target')
    return p


def validate_contract(contract):
    require(contract.get('schema_version')==2 and contract.get('stage')=='S13' and contract.get('family')=='global_duplex' and contract.get('adapter_id')==ADAPTER_ID,'S13 contract identity')
    ids=[]
    for case in contract['cases']:
        case_identity(case);ids.append(case['id'])
        require(case.get('work_model')=='global_requested_read_plus_write_v1' and case.get('work_unit')=='byte','S13 work model')
        require(case.get('metric')=={'numerator':'work_count','denominator':'elapsed_ns','scale':1,'unit':'GB/s_requested_payload/GPU'},'globaltimer requested byte rate')
        require(case.get('coverage_policy')=='exact_sms' and case.get('exportable') is True,'formal full GPU policy')
    require(len(ids)==18 and len(set(ids))==18,'all18 finite cases')
    require(contract.get('build')=={'compiler':'nvcc','flags':['-std=c++17','-O3','-lineinfo','-gencode','arch=compute_90a,code=sm_90a','-Xptxas=-v'],'include_dirs':[]},'S13 CUDA12.9 build flags')
    return contract


def validate_device(device):
    base_device(device)
    for field in ('l2_cache_bytes','global_memory_bytes','registers_per_sm','smem_per_sm_bytes','smem_per_cta_optin_bytes'):
        require(type(device.get(field)) is int and device[field]>0,'positive device capacity: '+field)
    return device


def kernel_symbol(case):
    p=case_identity(case);cache=p['read_cache_policy'] or 'wb'
    return f'gd_{p["mode"]}_{cache}_r{p["read_requests_per_group"]}_w{p["write_requests_per_group"]}'


def allocation(case,device,occupancy):
    p=case_identity(case);require(type(occupancy) is int and 1<=occupancy<=8,'256-thread occupancy upper')
    blocks=device['sms']*min(4,occupancy);T=blocks*256;requested=device['l2_cache_bytes']//4 if p['working_set_class']=='small' else device['l2_cache_bytes']*4
    require(requested>0,'requested array size')
    quantum=T*16;size=((requested+quantum-1)//quantum)*quantum;aggregate=size*(int(p['read_requests_per_group']>0)+int(p['write_requests_per_group']>0))
    require(aggregate<device['l2_cache_bytes'] if p['working_set_class']=='small' else size>=4*device['l2_cache_bytes'],'L2-relative actual allocation')
    require(aggregate+T*4+blocks*40<=device['global_memory_bytes'],'necessary GPU memory capacity')
    return {'blocks':blocks,'requested_array_bytes':requested,'array_bytes':size,'aggregate_array_bytes':aggregate,
            'read_requests_per_group':p['read_requests_per_group'],'write_requests_per_group':p['write_requests_per_group']}


def validate_resources(resource,case,device):
    require(isinstance(resource,dict) and set(resource)=={'kernel_symbol','registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm','extensions'},'complete S13 resources')
    for field in ('registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm'):
        require(type(resource[field]) is int and resource[field]>=0,'resource integer '+field)
    require(resource['kernel_symbol']==kernel_symbol(case) and 0<resource['registers_per_thread']<=255 and resource['static_smem_bytes']==1024 and resource['dynamic_smem_bytes']==resource['local_size_bytes']==0,'fixed kernel/drain/no spilling')
    occ=resource['occupancy_limit_ctas_per_sm'];plan=allocation(case,device,occ)
    require(occ*256*resource['registers_per_thread']<=device['registers_per_sm'] and occ*1024<=device['smem_per_sm_bytes'] and 1024<=device['smem_per_cta_optin_bytes'],'necessary occupancy capacities')
    require(resource['extensions']=={k:v for k,v in plan.items() if k!='blocks'} and all(type(v) is int for v in resource['extensions'].values()),'independent aligned allocation/ratio identity')
    return plan


def expected_elements(plan):
    return (plan['blocks']*256 if plan['read_requests_per_group'] else 0)+(plan['array_bytes']//4 if plan['write_requests_per_group'] else 0)


def validate_trial(row,case,device,seed,protocol):
    resource={key:row.get(key) for key in ('kernel_symbol','registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm')}
    resource['extensions']={key:row.get(key) for key in ('requested_array_bytes','array_bytes','aggregate_array_bytes','read_requests_per_group','write_requests_per_group')}
    plan=validate_resources(resource,case,device);size=plan['array_bytes'];p=case['parameters']
    rd=16*size*p['read_requests_per_group'];wr=16*size*p['write_requests_per_group']
    expected={'blocks':plan['blocks'],'read':rd,'write':wr,'work':rd+wr,'operations':(rd+wr)//16,
              'correctness':{'method':METHOD,'checked_elements':expected_elements(plan),'input_conditions':INPUTS}}
    return validate_observation(row,case,device,seed,protocol,expected)


def read_checksum(size,total_threads,thread,reads,iterations,seed):
    """Independent sum over owner vectors, without reproducing issue order."""
    require(size>0 and size%(16*total_threads)==0 and 0<=thread<total_threads and reads in (0,1,2,4) and iterations in (1,2,16),'reference coordinates')
    groups=size//(16*total_threads)
    # Sum each lane arithmetic progression; modular wrap is deferred to end.
    total=sum(17*(groups*(4*thread+lane)+4*total_threads*groups*(groups-1)//2)+groups*seed for lane in range(4))
    return (total*reads*iterations)&0xffffffff


def _instructions(body):
    out=[]
    for line in body.splitlines():
        match=re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?)(?:/\*|$)',line)
        if match:out.append((int(match[1],16),match[2].strip().rstrip(';').strip()))
    require(out,'actual SASS instructions required');return out


def _opcode(op):
    parts=op.split()
    return parts[1] if parts[0].startswith('@') else parts[0]


def _vector_registers(value,width=4):
    match=re.fullmatch(r'(R|UR)([0-9]+)(?:\.reuse)?',value.strip())
    require(match is not None,'explicit contiguous vector register operand')
    return [match[1]+str(int(match[2])+i) for i in range(width)]


def dataflow_evidence(ops,inner,drain_pc,copy):
    """Conservative register provenance, including loop-carried dependencies.

    Iterate the group loop until its register provenance reaches a fixed point;
    a store using the prior iteration's read sum must also be rejected. Physical
    addressing/control interpretation still receives independent source-B review.
    """
    tags={};load_lanes=set();stores={};drain_tags=None
    def walk(segment):
        nonlocal drain_tags
        for pc,op in segment:
            opcode=_opcode(op);rest=op.split(opcode,1)[1].strip()
            if opcode.startswith('LDG.') and inner[0]<=pc<=inner[1]:
                require('.128' in opcode,'four-word dataflow load')
                regs=_vector_registers(rest.split(',',1)[0])
                for lane,reg in enumerate(regs):tags[reg]={(pc,lane)};load_lanes.add((pc,lane))
                continue
            if opcode.startswith('STG.') and inner[0]<=pc<=inner[1]:
                regs=_vector_registers(rest.rsplit(',',1)[1]);sources=[set(tags.get(reg,set())) for reg in regs]
                if copy:
                    require(all(len(x)==1 for x in sources),'copy output must come from actual loaded lanes')
                    origins=[next(iter(x)) for x in sources]
                    require(len({x[0] for x in origins})==1 and [x[1] for x in origins]==[0,1,2,3],'copy lane order/data dependency')
                else:require(not any(sources),'independent store data depends on a load or prior read sum')
                stores[pc]={'pc':pc,'data_registers':regs,'load_origins':[sorted(x) for x in sources]}
                continue
            if pc==drain_pc:
                reg=rest.rsplit(',',1)[1].strip().removesuffix('.reuse')
                require(re.fullmatch(r'R[0-9]+',reg) or (reg=='RZ' and not load_lanes),'scalar checksum drain register')
                drain_tags=set(tags.get(reg,set()));continue
            operands=rest.split(',');dest=operands[0].strip().removesuffix('.reuse')
            if not re.fullmatch(r'(?:R|UR)[0-9]+',dest):continue
            origins=set()
            for reg in re.findall(r'\b(?:R|UR)[0-9]+\b',','.join(operands[1:])):origins|=tags.get(reg,set())
            if op.startswith('@'):origins|=tags.get(dest,set())
            width=2 if ('.64' in opcode or '.WIDE' in opcode or opcode.startswith('CS2R')) else 1
            for reg in _vector_registers(dest,width):tags[reg]=set(origins)
    walk([(pc,op) for pc,op in ops if pc<inner[0]])
    body=[(pc,op) for pc,op in ops if inner[0]<=pc<=inner[1]]
    for passes in range(1,257):
        before={k:set(v) for k,v in tags.items()};walk(body)
        if tags==before:break
    else:raise ValueError('group-loop provenance did not converge')
    walk([(pc,op) for pc,op in ops if pc>inner[1]])
    require(drain_tags is not None and load_lanes<=drain_tags,'every uint4 lane must feed the pre-stop checksum drain')
    return {'loaded_lanes':len(load_lanes),'drained_lanes':len(drain_tags),'loop_fixed_point_passes':passes,'stores':[stores[k] for k in sorted(stores)]}


def audit_sass(text,contract):
    """Reject incomplete lowering; actual source-B must still review PC dataflow.

    Width/cache/count/nested loop/fence/drain and lane dataflow checks use the
    observed CUDA12.9 forms. Independent source-B still reviews address ownership
    and uniform control against this concrete compile; no GPU claim is made.
    """
    functions=re.split(r'Function\s*:\s*',text)[1:];out=[]
    for case in contract['cases']:
        p=case_identity(case);symbol=kernel_symbol(case);matches=[f for f in functions if f.splitlines()[0].strip()==symbol]
        require(len(matches)==1,'missing/ambiguous actual S13 function '+symbol)
        ops=_instructions(matches[0]);loops=timed_loops(matches[0]);timers=[pc for pc,op in ops if 'SR_GLOBALTIMERLO' in op]
        require(len(timers)==2,'two timing boundaries')
        memory_loops=[loop for loop in loops if any(re.search(r'\b(?:LDG|STG)(?:\.|\s)',op) for _,op in loop[2])]
        require(len(memory_loops)==2,'exact sweep and iteration loops required; lowering needs independent review')
        inner=min(memory_loops,key=lambda x:x[1]-x[0]);outer=max(memory_loops,key=lambda x:x[1]-x[0])
        require(outer[0]<=inner[0]<inner[1]<outer[1],'nested complete group/iteration loop structure')
        require(not any(re.search(r'\b(?:LDS|STS)(?:\.|\s)',op) for _,op in inner[2]),'unexpected shared traffic in payload group loop')
        counts={}
        for token,wanted in [('LDG',p['read_requests_per_group']),('STG',p['write_requests_per_group'])]:
            positions=[(pc,op) for pc,op in inner[2] if re.search(r'\b'+token+r'(?:\.|\s)',op)]
            require(len(positions)==wanted,'actual group read/write ratio')
            require(all(not op.startswith('@') for _,op in positions),'payload LDG/STG must be unconditional in the reviewed lowering')
            require(all('.128' in _opcode(op) for _,op in positions),'full uint4 vector lowering')
            expected_opcode=('LDG.E.128.STRONG.SM' if p['read_cache_policy']=='ca' else 'LDG.E.128.STRONG.GPU') if token=='LDG' else 'STG.E.128.STRONG.SM'
            require(all(_opcode(op)==expected_opcode for _,op in positions),'reviewable CUDA12.9 vector/cache form')
            counts[token]=[{'pc':pc,'instruction':op} for pc,op in positions]
        barriers=[pc for pc,op in ops if op.startswith('BAR.SYNC')]
        require(any(timers[0]<pc<inner[0] for pc in barriers),'start CTA gate')
        drains=[(pc,op) for pc,op in ops if outer[1]<pc<timers[1] and re.search(r'\bSTS(?:\.|\s)',op)]
        require(len(drains)==1 and not drains[0][1].startswith('@') and any(drains[0][0]<pc<timers[1] for pc in barriers),'one unconditional result-dependent shared drain and completion CTA barrier')
        fences=[(pc,op) for pc,op in ops if outer[1]<pc<timers[1] and ('MEMBAR' in op or 'FENCE' in op)]
        require(len(fences)==(1 if p['write_requests_per_group'] else 0),'one writing completion fence / read-only no fence')
        if fences:
            require(fences[0][1]=='MEMBAR.SC.GPU','writing fence must be unconditional MEMBAR.SC.GPU')
            require(any(fences[-1][0]<pc<timers[1] for pc in barriers),'CTA barrier after write fence')
        require(not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)',op) for _,op in ops),'no local/spill')
        flow=dataflow_evidence([(pc,op) for pc,op in ops if timers[0]<pc<timers[1]],inner,drains[0][0],p['mode']=='copy')
        out.append({'dataflow':flow,'case_id':case['id'],'symbol':symbol,'group_loop':[inner[0],inner[1]],'iteration_loop':[outer[0],outer[1]],'accesses':counts,'barrier_pcs':barriers,'drain':drains,'fences':fences,
                    'requires_independent_actual_cache_and_dataflow_review':True,
                    'B2_qualification':'actual_lowering_checks_pending_independent_source_B_review'})
    return out

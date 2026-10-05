"""Independent S10 address, payload, logical instruction and SASS checks."""
from __future__ import annotations
import itertools
import re
from auditors.memory_baseline import validate_device, timed_loops
from auditors.observation import validate_observation
from common.suite_io import require

ADAPTER_ID='matrix_exchange_v2'
METHOD='full_fragments_checksum_output_and_padding_v1'
INPUTS='coordinate+seed matrix uint16; step-XOR independent stores; lane/warp/stream shuffle seed'


def validate_contract(contract):
    require(contract.get('schema_version')==2 and contract.get('stage')=='S10' and contract.get('family')=='matrix_exchange' and contract.get('adapter_id')==ADAPTER_ID,'S10 identity')
    seen=[]
    for case in contract['cases']:
        p=case['parameters'];mode=p['mode'];shuffle=mode=='shuffle';seen.append(case['id'])
        require(case['iterations']==8192 and case['scope']=='one_cta' and case['launch']=={'kind':'one_cta'},'fixed launch/iterations')
        require(case['coverage_policy']=='one_sm' and case['exportable'] is True,'scope qualification')
        require(case['capabilities']=={'cc':'9.0','device_name_contains':'GH200'},'target')
        if shuffle:
            t=case['threads'];streams=p['streams']
            require(t in (32,128) and streams in (1,4),'shuffle dimensions')
            require(case['id']==f'shuffle_t{t}_streams{streams}','shuffle id')
            require(p=={'mode':'shuffle','ptx':'shfl.sync.idx.b32','streams':streams,'batch':8,'source_lane':'(lane+1)%32','clamp':31,'membermask':4294967295,'each_stream_depends_on_previous_shuffle':True,'validation_steps':[1,3,33]},'shuffle conditions')
        else:
            n=p['matrices'];trans=p['transpose']
            require(mode in ('load','store','roundtrip') and n in (1,2,4) and type(trans) is bool and case['threads']==32,'matrix dimensions')
            require(case['id']==f'{mode}_x{n}_'+('trans' if trans else 'normal'),'matrix id')
            suffix=f'.sync.aligned.m8n8.x{n}'+('.trans' if trans else '')+'.shared.b16'
            require(p=={'mode':mode,'matrices':n,'transpose':trans,'load_ptx':'ldmatrix'+suffix if mode!='store' else None,'store_ptx':'stmatrix'+suffix if mode!='load' else None,'batch':8,'slots':8,'slot_bytes':512,'input_array_bytes':4096,'output_array_bytes':4096,'dynamic_smem_bytes':8192,'barrier_per_pair':mode=='roundtrip','roundtrip_load_and_store_same_transpose':mode=='roundtrip'},'matrix access conditions')
        require(case['work_unit']==('operation' if shuffle else 'byte'),'work unit')
        require(case['work_model']==('warp_shuffle_instruction_v1' if shuffle else 'warp_matrix_requests_v1'),'work model')
        require(case['metric']=={'numerator':'work_count','denominator':'cta_clock64_cycles','scale':1,'unit':'warp_instruction/clock64_cycle/CTA' if shuffle else 'B/clock64_cycle/CTA'},'metric')
    expected={f'{mode}_x{n}_{t}' for mode,n,t in itertools.product(('load','store','roundtrip'),(1,2,4),('normal','trans'))}|{f'shuffle_t{t}_streams{s}' for t,s in itertools.product((32,128),(1,4))}
    require(len(seen)==22 and set(seen)==expected,'finite22 matrix')
    require(contract['build']=={'compiler':'nvcc','flags':['-std=c++17','-O3','-lineinfo','-gencode','arch=compute_90a,code=sm_90a','-Xptxas=-v'],'include_dirs':[]},'build protocol')
    require('sass_contracts' in contract,'SASS contract required')
    return contract


def fragment_coordinate(lane,half,transpose):
    row,col=divmod(lane,4)[0],2*(lane%4)+half
    return (col,row) if transpose else (row,col)


def reference(case,length,seed):
    """Expected complete unsigned output independently from the C++ implementation."""
    p=case['parameters']
    if p['mode']=='shuffle':
        # Explicit permutation power: after length steps each lane reads length lanes ahead.
        return [(seed+17*((t%32+length)%32)+131*s+8191*(t//32))&0xffffffff for t in range(case['threads']) for s in range(p['streams'])]
    n=p['matrices'];packed=[];checksums=[]
    for lane in range(32):
        lane_values=[]
        for slot in range(8):
            for matrix in range(n):
                values=[]
                for half in range(2):
                    row,col=fragment_coordinate(lane,half,p['transpose'])
                    value=(1+row+11*col+97*matrix+997*slot+seed%8191)&65535
                    if p['mode']=='store':value^=length&65535
                    values.append(value)
                lane_values.append(values[0]+65536*values[1])
        packed.extend(lane_values)
        checksums.append(0 if p['mode']=='store' else (sum(lane_values)*length)&0xffffffff)
    destination=[0xdead]*2048
    if p['mode']!='load':
        for slot,matrix,row,col in itertools.product(range(8),range(n),range(8),range(8)):
            v=(1+row+11*col+97*matrix+997*slot+seed%8191)&65535
            if p['mode']=='store':v^=length&65535
            destination[slot*256+matrix*64+row*8+col]=v
    return checksums+packed+destination


def work(case):
    p=case['parameters'];shuffle=p['mode']=='shuffle'
    direction=0 if shuffle else case['iterations']*8*p['matrices']*8*8*2
    rd=direction if p['mode'] in ('load','roundtrip') else 0
    wr=direction if p['mode'] in ('store','roundtrip') else 0
    operations=case['iterations']*8*(p['streams']*(case['threads']//32) if shuffle else int(rd>0)+int(wr>0))
    checked=case['threads']*p['streams'] if shuffle else 32+32*8*p['matrices']+2048
    return {'blocks':1,'read':rd,'write':wr,'work':operations if shuffle else rd+wr,'operations':operations,'correctness':{'method':METHOD,'checked_elements':checked,'input_conditions':INPUTS}}


def validate_trial(row,case,device,seed,protocol):
    expected=work(case);p=case['parameters'];shuffle=p['mode']=='shuffle'
    for key in ('registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm'):
        require(type(row.get(key)) is int and row[key]>=0,'integer resource '+key)
    require(0<row['registers_per_thread']<=255 and row['local_size_bytes']==0,'registers/spill')
    require(row['dynamic_smem_bytes']==(0 if shuffle else 8192),'dynamic shared allocation')
    require(row['static_smem_bytes']>=4*case['threads']+16,'drain and timer storage')
    occ=row['occupancy_limit_ctas_per_sm'];require(1<=occ<=32,'occupancy range')
    total=row['static_smem_bytes']+row['dynamic_smem_bytes']
    require(total<=device['smem_per_cta_optin_bytes'] and occ*total<=device['smem_per_sm_bytes'],'shared capacity')
    require(occ*case['threads']*row['registers_per_thread']<=device['registers_per_sm'],'register capacity')
    require(row.get('nonuniform_short_checks')==([1,3,33] if shuffle else [1,2]) and row.get('short_check_errors')==0,'short validation')
    return validate_observation(row,case,device,seed,protocol,expected)


def audit_sass(text,contract):
    functions=re.split(r'Function\s*:\s*',text)[1:];out=[]
    for case in contract['cases']:
        p=case['parameters'];shuffle=p['mode']=='shuffle'
        symbol=(f'warp_exchangeILi{case["threads"]}ELi{p["streams"]}' if shuffle else f'matrix_exchangeILi{p["matrices"]}ELb{int(p["transpose"])}ELi'+str({'load':0,'store':1,'roundtrip':2}[p['mode']]))
        matches=[f for f in functions if symbol in f.splitlines()[0]]
        require(len(matches)==1,'missing/ambiguous function '+symbol)
        body=matches[0]
        require(not re.search(r'\b(?:LDL|STL)(?:\.|\s)',body),'unexpected local/spill')
        loops=sorted(timed_loops(body))
        require(len(loops)==(2 if shuffle else 1),'main loop and optional short-check tail required')
        if shuffle:
            require(loops[0][1]<loops[1][0],'nonoverlapping main and remainder loops')
        all_ops=[]
        for line in body.splitlines():
            match=re.match(r'\s*/\*([0-9a-fA-F]+)\*/(.*?)(?:/\*|$)',line)
            if match:all_ops.append((int(match[1],16),match[2].strip()))
        timers=[pc for pc,op in all_ops if 'SR_GLOBALTIMERLO' in op]
        require(len(timers)==2,'two globaltimer boundaries')
        barriers=[pc for pc,op in all_ops if 'BAR.SYNC' in op and not op.startswith('@')]
        require(any(timers[0]<pc<loops[0][0] for pc in barriers),'start CTA gate missing')
        drains=[pc for pc,op in all_ops if loops[-1][1]<pc<timers[1] and re.search(r'\bSTS(?:\.|\s)',op)]
        require(drains and any(max(drains)<pc<timers[1] for pc in barriers),'result drain and completion CTA gate missing')
        evidence=[]
        for index,(lo,hi,ops) in enumerate(loops):
            counts={token:sum(bool(re.search(r'\b'+token+r'(?:\.|\s)',op)) for _,op in ops) for token in ('LDSM','STSM','SHFL','LDS','STS')}
            expected={'LDSM':0 if shuffle or p['mode']=='store' else 8,
                      'STSM':0 if shuffle or p['mode']=='load' else 8,
                      'SHFL':p['streams']*(8 if index==0 else 1) if shuffle else 0,
                      'LDS':0,'STS':0}
            require(counts==expected,'timed instruction direction/count')
            for _,op in ops:
                if re.search(r'\b(?:LDSM|STSM)\.',op):
                    token=re.search(r'\b(?:LDSM|STSM)\.[A-Z0-9.]+',op)[0]
                    shape=re.fullmatch(r'(?:LDSM|STSM)\.16\.(M88|MT88)(?:\.(2|4))?',token)
                    require(shape is not None and int(shape[2] or 1)==p['matrices'],'matrix shape/count format')
                    require(('.MT88' in token)==p['transpose'],'matrix transpose format')
                if 'SHFL.' in op:require('SHFL.IDX' in op,'shuffle mode')
            evidence.append({'role':'short_check_remainder' if shuffle and index else 'main',
                             'start':lo,'stop':hi,'counts':counts})
        out.append({'case_id':case['id'],'symbol':symbol,'loops':evidence})
    return out

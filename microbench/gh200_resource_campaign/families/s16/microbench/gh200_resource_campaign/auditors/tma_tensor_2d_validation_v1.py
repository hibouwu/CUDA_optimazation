"""Finite S15 ABI1 and separate complete value replay. No runtime promotion."""
import hashlib,json,re
from common.suite_io import require
from auditors.tma_bulk import validate_device,_u64s
from auditors.tma_bulk_validation import relative_path
ADAPTER_ID='tma_tensor_2d_validation_v1';ADAPTER_ABI_VERSION=1
PAIRS={'tensor_short_1_seed0_v1':(1,0),'tensor_short_2_seed3_v1':(2,3),'tensor_short_33_seed4294967295_v1':(33,4294967295)}
PROFILE_HASHES={'tensor_short_1_seed0_v1': '135d371f38dbe8cd66d9da0b39da25a76593e18cae1eb569cc20921ed9937555', 'tensor_short_2_seed3_v1': '564d93d5aa16ca708b2582f68e666bba1e31406c3cd3f8fe93be01ed8be69798', 'tensor_short_33_seed4294967295_v1': '8c7fbc23676afe1c415ead59a184132c40f7f08c85d86940a468a89123076018'}
REFERENCE_SHA='58791bc1c67434b450af2be8115f9ae47335a9f2c9d45519ef27171483ed3ec6'

def geometry(case):
    p=case['parameters'];q=p['payload_bytes'];require(type(q) is int and q in (1024,4096,8192,16384,32768,65536),'S15 finite payload');w=128 if q==65536 else 64;h=q//(2*w);pitch=p['row_stride_bytes'];sw=p['swizzle']=='SW128';g2s=p['direction']=='gmem_to_smem'
    require(type(q) is int and q in (1024,4096,8192,16384,32768,65536) and type(pitch) is int and pitch in (2*w,2*w+16),'S15 finite Q/stride')
    require(p['swizzle'] in ('none','SW128') and (not sw or q<=32768 and pitch==128),'S15 SW128 box/pitch')
    layout='continuous_sw128' if sw else 'continuous_none' if pitch==2*w else 'padding_none'
    require(case['id']==f'{p["direction"]}_{q//1024}kib_{layout}_{case["scope"]}' and case['scope'] in ('one_cta','all_gpu') and p['direction'] in ('gmem_to_smem','smem_to_gmem'),'S15 finite coordinate')
    require(type(case['threads']) is int and case['threads']==128 and p['element_bytes']==2 and p['box_dim']==[w,h] and p['layout']==layout and p['dynamic_shared_bytes']==q+1056 and p['global_slots_per_cta']==32 and p['tensor_rank']==2 and p['tensor_map'] is True,'S15 original shape/allocation')
    return q,w,h,pitch,sw,g2s

def profile_check(profile):
    require(profile['id'] in PAIRS and hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':')).encode()).hexdigest()==PROFILE_HASHES[profile['id']],'frozen S15 profile')

def validation_argv(binary_relative,case,profile,seed):
    geometry(case);profile_check(profile);require(type(seed) is int and seed==PAIRS[profile['id']][1],'paired S15 seed')
    return [relative_path(binary_relative),'validate-only',case['id'],profile['id'],str(seed)]

def resource_plan(device,row,case,length):
    validate_device(device);q,w,h,pitch,sw,g2s=geometry(case);r=row['resource_identity'];fields={'kernel_symbol','registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm','extensions'}
    require(set(r)==fields,'S15 complete resources')
    for key in fields-{'kernel_symbol','extensions'}:require(type(r[key]) is int and r[key]>=0,'S15 resource integer')
    occ=r['occupancy_limit_ctas_per_sm'];regs=r['registers_per_thread'];total=r['static_smem_bytes']+r['dynamic_smem_bytes']
    symbol=f'tt_{"g2s" if g2s else "s2g"}_q{q}_{"sw128" if sw else "none"}'
    require(r['kernel_symbol']==symbol and 1<=occ<=16 and regs==40 and r['static_smem_bytes']==0 and r['local_size_bytes']==0 and r['dynamic_smem_bytes']==q+1056 and total<=device['smem_per_cta_optin_bytes'] and occ*total<=device['smem_per_sm_bytes'] and occ*128*regs<=device['registers_per_sm'] and occ*128<=2048,'S15 resource capacity/symbol')
    blocks=1 if case['scope']=='one_cta' else device['sms']*min(4,occ);rows=blocks*32*h;allocation=rows*pitch+256;capture=blocks*length*q
    require(rows<=2**31-1 and allocation+2*capture+blocks*80<=device['global_memory_bytes'],'S15 coordinates/memory capacity')
    require(type(row['blocks']) is int and row['blocks']==blocks,'S15 actual grid')
    e=r['extensions'];expected={'payload_bytes':q,'row_stride_bytes':pitch,'global_slots_per_cta':32,'global_allocation_bytes':allocation,'capture_bytes_per_array':capture,'tensor_rank':2,'global_dimensions':[w,rows],'box_dimensions':[w,h],'element_strides':[1,1],'swizzle':'SW128' if sw else 'none','descriptor_encode_status':0,'descriptor_alignment_bytes':64,'global_base_alignment_bytes':128,'shared_tile_alignment_bytes':1024,'data_type':'UINT16','interleave':'none','l2_promotion':'none','oob_fill':'none'}
    for key,value in expected.items():
        if type(value) is int:require(type(e.get(key)) is int,'S15 exact integer descriptor field '+key)
        elif isinstance(value,list):require(isinstance(e.get(key),list) and all(type(x) is int for x in e[key]),'S15 exact integer descriptor array '+key)
    require(set(e)==set(expected)|{'descriptor_sha256','global_base_address','descriptor_host_address'} and all(e[k]==v for k,v in expected.items()),'S15 exact descriptor inputs/return/payload')
    for name,alignment in [('global_base_address',128),('descriptor_host_address',64)]:require(type(e[name]) is int and 0<e[name]<2**64 and e[name]%alignment==0,'S15 actual descriptor/base alignment')
    require(isinstance(e['descriptor_sha256'],str) and re.fullmatch('[0-9a-f]{64}',e['descriptor_sha256']),'S15 descriptor SHA')
    return blocks,allocation

def layouts(case,blocks,length,allocation):
    _,w,h,_,_,_=geometry(case)
    return {'tensor_logical.u16le':('uint16',[blocks,length,h,w]),'tensor_physical.u16le':('uint16',[blocks,length,h,w]),'tensor_global.u16le':('uint16',[allocation//2]),'tensor_completion.u32le':('uint32',[blocks,5,2]),'tensor_stamps.u32le':('uint32',[blocks,5,2]),'tensor_descriptor.u8':('uint8',[128])}

def validate_validation(device,row,case,profile,seed):
    profile_check(profile);length,wanted=PAIRS[profile['id']];require(type(seed) is int and seed==wanted,'fixed S15 seed')
    fields={'schema_version','validation_schema_version','type','case_id','profile_id','seed','scope','threads','blocks','errors','target_launches','checks','resource_identity','performance_eligible','warmup_executed','pilot_executed'}
    require(set(row)==fields,'S15 fixed ABI fields')
    for key,value in [('schema_version',2),('validation_schema_version',1),('seed',seed),('threads',128),('errors',0)]:require(type(row[key]) is int and row[key]==value,'S15 fixed integer '+key)
    require(row['type']=='validation' and row['case_id']==case['id'] and row['profile_id']==profile['id'] and row['scope']==case['scope'] and all(row[k] is False for k in ('performance_eligible','warmup_executed','pilot_executed')),'S15 identity/no performance')
    blocks,allocation=resource_plan(device,row,case,length);q,_,_,_,_,_=geometry(case)
    require(row['target_launches']==[{'launch_index':0,'iterations':length,'input_profile':'nonuniform_uint16_tensor_pattern','threads':128,'blocks':blocks}] and all(type(row['target_launches'][0][k]) is int for k in ('launch_index','iterations','threads','blocks')) and len(row['checks'])==1,'single target launch')
    check=row['checks'][0];fields={'launch_index','reference_model','reference_sha256','comparison','tolerance_id','checked_elements','expected_elements','errors','completed','verified_CTA_ids','output_artifacts'}
    require(set(check)==fields and check['reference_model']=='tma_tensor_2d_word_reference_v1' and check['reference_sha256']==REFERENCE_SHA and check['comparison']=='exact' and check['tolerance_id'] is None and check['completed'] is True,'S15 exact reference/complete')
    count=2*blocks*length*(q//2)+allocation//2
    for key,value in [('launch_index',0),('errors',0),('checked_elements',count),('expected_elements',count)]:require(type(check[key]) is int and check[key]==value,'S15 complete quantity '+key)
    require(check['verified_CTA_ids']==list(range(blocks)) and all(type(i) is int for i in check['verified_CTA_ids']),'all S15 CTA IDs')
    wanted_layouts=layouts(case,blocks,length,allocation);artifacts=check['output_artifacts'];require(len(artifacts)==6 and {a['path'] for a in artifacts}==set(wanted_layouts),'complete S15 artifacts')
    for a in artifacts:
        require(set(a)=={'path','sha256','dtype','shape','evidence_kind'} and (a['dtype'],a['shape'])==wanted_layouts[relative_path(a['path'])] and all(type(n) is int and n>0 for n in a['shape']) and a['evidence_kind']=='full_values' and re.fullmatch('[0-9a-f]{64}',a['sha256']),'S15 artifact metadata')
    require(next(a['sha256'] for a in artifacts if a['path']=='tensor_descriptor.u8')==row['resource_identity']['extensions']['descriptor_sha256'],'descriptor artifact identity')
    return {'status':'pass','case_id':case['id'],'profile_id':profile['id'],'target_launches':1,'checked_elements':count,'output_evidence_kind':'full_values','performance_eligible':False}

def audit_sass(text,contract):
    from auditors.tma_tensor_2d_sass_v1 import audit_sass as actual
    return actual(text,contract)

def validate_prior_evidence(evidence,request,frozen_refs):
    geometry(request['case']);profile_check(request['profile'])
    return {'status':'insufficient','case_id':request['case']['id'],'missing_requirements':['S14 or feasibility compilation cannot qualify S15 descriptor/swizzle/lifecycle; no prior mapping approved']}

def audit_values(device,row,case,profile,seed,arrays):
    result=validate_validation(device,row,case,profile,seed);length,_=PAIRS[profile['id']];blocks,allocation=resource_plan(device,row,case,length);q,w,h,pitch,sw,g2s=geometry(case)
    expected_layouts=layouts(case,blocks,length,allocation);require(set(arrays)==set(expected_layouts),'complete S15 full arrays')
    values={}
    limits={'uint8':255,'uint16':65535,'uint32':2**32-1}
    for name,(dtype,shape) in expected_layouts.items():
        item=arrays[name];require(set(item)=={'shape','values'} and item['shape']==shape and all(type(n) is int for n in item['shape']),'S15 exact artifact shape')
        count=1
        for dim in shape:count*=dim
        v=item['values'];require(len(v)==count and all(type(x) is int and 0<=x<=limits[dtype] for x in v),'S15 exact integer artifact domain');values[name]=v
    descriptor=values['tensor_descriptor.u8'];require(hashlib.sha256(bytes(descriptor)).hexdigest()==row['resource_identity']['extensions']['descriptor_sha256'],'S15 opaque descriptor bytes identity')
    controls=_u64s(values['tensor_completion.u32le']);stamps=_u64s(values['tensor_stamps.u32le']);logical=values['tensor_logical.u16le'];physical=values['tensor_physical.u16le'];global_values=values['tensor_global.u16le'];elements=q//2
    for b in range(blocks):
        done,attempts,timeout,release,full=controls[b*5:b*5+5]
        require(done==length and timeout==0 and release==0 and full==(0 if g2s else length),'S15 complete lifecycle')
        require(length<=attempts<2**64-1 if g2s else attempts==0,'S15 actual wait count')
        start,stop,start_cycle,stop_cycle,smid=stamps[b*5:b*5+5]
        require(0<=start<=stop<2**64-1 and 0<=start_cycle<=stop_cycle<2**64-1 and 0<=smid<2**32-1,'S15 ordered complete stamps')
        for i in range(length):
            slot=i%32
            for y in range(h):
                for x in range(w):
                    n=y*w+x;at=(b*length+i)*elements+n;expected=(17*x+31*y+(73*slot if g2s else 0)+151*b+seed)&65535
                    require(logical[at]==expected,'S15 complete logical capture mismatch')
                    physical_n=y*64+((x//8)^(y%8))*8+x%8 if sw else n
                    require(physical[(b*length+i)*elements+physical_n]==expected,'S15 complete physical/swizzle capture mismatch')
    rows=blocks*32*h;stride=pitch//2
    for i in range(64):require(global_values[i]==0xa900+i and global_values[64+rows*stride+i]==0xa900+64+i,'S15 global guards')
    for row in range(rows):
        b=row//(32*h);slot=(row//h)%32;y=row%h
        for x in range(stride):
            if x<w:
                expected=(17*x+31*y+(73*slot if g2s else 0)+151*b+seed)&65535
                if not g2s and length<32 and slot>=length:expected^=65535
            else:expected=(0x7d00+row*11+x)&65535
            require(global_values[64+row*stride+x]==expected,'S15 whole global payload/padding/untouched slots')
    return {**result,'values_checked':True,'family_B3_eligible':False}

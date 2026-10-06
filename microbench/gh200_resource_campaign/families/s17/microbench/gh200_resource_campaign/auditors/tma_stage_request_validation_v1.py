"""Pure short ABI and full-value replay; no GPU, subprocess or filesystem access."""
import hashlib,json,re
from common.suite_io import require
from auditors.async_copy import validate_device
from auditors.tma_stage_request_sass_v1 import audit_sass
from auditors.tma_stage_request_sass_baseline_v1 import BASELINE
ADAPTER_ID='tma_stage_request_v1'
ADAPTER_ABI_VERSION=1
PROFILE_SHA='e2eb615584bd5ab16d302d9fbdac019766f2f2a32dcbe34abb1a76b65af69d17'
LENGTHS=(1,2,5,33)
W=4096

def uint(x,bits=64):
    require(type(x)is int and 0<=x<2**bits,'S16 uint domain');return x

def profile_check(p):
    require(hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()==PROFILE_SHA,'S16 fixed profile')

def coordinate(case):
    m=re.fullmatch(r'(gmem_to_smem|smem_to_gmem)_16kib_s([124])_r([124])_(one_cta|all_gpu)',case['id']);require(m is not None,'S16 finite case');g=m[1]=='gmem_to_smem';s=int(m[2]);r=int(m[3]);require(case['parameters']=={'direction':m[1],'software_stages':s,'requests_per_item':r,'payload_bytes':16384},'S16 case parameters');return g,s,r,m[4]

def validation_argv(binary,case,p,seed):
    from pathlib import PurePosixPath
    profile_check(p);coordinate(case);uint(seed,32);require(seed==3,'S16 seed3');path=PurePosixPath(binary);require(not path.is_absolute() and '..' not in path.parts and path.as_posix()==binary and '\\' not in binary and binary!='.','S16 binary path');return [binary,'validate-only',case['id'],p['id'],'3']

def shape_layout(b,i,s,r):
    return {'trace':([b,i,r,W],b*i*r*W),'final_slots':([b,s,r,W],b*s*r*W),'ring_guards':([b*32*r*W+8],b*32*r*W+8),'counts':([b,12,2],b*24),'lifecycle':([b,i,8,2],b*i*16),'stamps':([b,5,2],b*10)}

def resources(device,row,case):
    validate_device(device);g,s,r,scope=coordinate(case);b=uint(row['blocks'],32);require(b>0 and row['threads']==128 and type(row['threads'])is int and row['scope']==scope,'S16 geometry')
    x=row['resource_identity'];
    for key in ('registers_per_thread','occupancy_limit_ctas_per_sm','static_smem_bytes','dynamic_smem_bytes','local_size_bytes'):uint(x[key],32)
    for key,v in x['extensions'].items():
        if key=='capture_enabled':require(v is True,'S16 capture flag')
        elif key=='opaque_token_evidence':
            require(v['classification']=='preserved_domain_only' and v['exact_checked']is False,'S16 opaque evidence classification')
            require(all(type(x)is int and x>=0 for x in v['words_per_launch']),'S16 preserved counts')
        else:uint(v)
    regs=uint(x['registers_per_thread'],32);occ=uint(x['occupancy_limit_ctas_per_sm'],32);static=uint(x['static_smem_bytes'],32);require(regs==BASELINE[f"ts_{'g2s' if g else 's2g'}_s{s}_r{r}"]['registers_per_thread'] and 0<occ<=32 and x['local_size_bytes']==0,'S16 resources');shared=s*r*16384+8*s+32;require(static==0 and x['dynamic_smem_bytes']==shared and static+shared<=device['smem_per_cta_optin_bytes'],'S16 shared capacity');require(occ*(shared+static)<=device['smem_per_sm_bytes'] and occ*128*regs<=device['registers_per_sm'] and occ*128<=2048,'S16 necessary occupancy bound');require(b==(1 if scope=='one_cta' else device['sms']*min(4,occ)),'S16 grid');require(x['kernel_symbol']==f"ts_{'g2s' if g else 's2g'}_s{s}_r{r}",'S16 target');require(x['extensions']=={'payload_bytes':16384,'software_stages':s,'requests_per_item':r,'global_slots_per_cta':32,'global_allocation_bytes':b*32*r*16384+32,'capture_enabled':True,'opaque_token_evidence':{'classification':'preserved_domain_only','exact_checked':False,'words_per_launch':[2*b*i if g else 0 for i in LENGTHS]}},'S16 resource extension');return b,g,s,r

def validate_validation(device,row,case,p,seed):
    profile_check(p);uint(seed,32);require(seed==3,'S16 seed');
    for key in ('schema_version','validation_schema_version','seed','blocks','threads','errors'):uint(row[key],32)
    require(row['schema_version']==2 and row['validation_schema_version']==1 and row['type']=='validation' and row['case_id']==case['id'] and row['profile_id']==p['id'] and row['seed']==3,'S16 identity');require(type(row['errors'])is int and row['errors']==0,'S16 errors')
    for k in ('performance_eligible','warmup_executed','pilot_executed'):require(row[k]is False,'S16 forbidden performance work')
    b,g,s,r=resources(device,row,case);require(len(row['checks'])==4 and len(row['target_launches'])==4,'S16 fixed4launches')
    for n,i in enumerate(LENGTHS):
        for key in ('launch_index','iterations','threads','blocks'):uint(row['target_launches'][n][key],32)
        for key in ('launch_index','errors','checked_elements','expected_elements'):uint(row['checks'][n][key])
        require(all(type(x)is int for x in row['checks'][n]['verified_CTA_ids']),'S16 CTA integer IDs')
        require(row['target_launches'][n]=={'launch_index':n,'iterations':i,'input_profile':'nonuniform_uint32_stage_request_pattern','threads':128,'blocks':b},'S16 launch order');ch=row['checks'][n];require(ch['launch_index']==n and ch['completed']is True and ch['errors']==0 and type(ch['errors'])is int and ch['comparison']=='exact' and ch['tolerance_id']is None and ch['reference_model']==p['reference_identity']['model'] and ch['reference_sha256']==p['reference_identity']['sha256'] and ch['verified_CTA_ids']==list(range(b)),'S16 completion checks');layout=shape_layout(b,i,s,r);require(ch['checked_elements']==sum(v[1] for v in layout.values())-(2*b*i if g else 0) and ch['expected_elements']==ch['checked_elements'] and type(ch['checked_elements'])is int,'S16 independent checked count');items=ch['output_artifacts'];require(len(items)==6,'S16 six output components')
        for item,(leaf,(shape,count)) in zip(items,layout.items()):require(item['path']==f'stage_{n}_{leaf}.u32le' and item['dtype']=='uint32' and item['evidence_kind']=='full_values' and item['shape']==shape and re.fullmatch('[0-9a-f]{64}',item['sha256'])is not None,'S16 artifact identity')
    return {'status':'pass','case_id':case['id'],'profile_id':p['id'],'target_launches':4,'checked_elements':sum(x['checked_elements'] for x in row['checks']),'performance_eligible':False,'output_evidence_kind':'full_values'}

def audit_values(device,row,case,p,seed,arrays):
    validate_validation(device,row,case,p,seed);b,g,s,r=resources(device,row,case)
    def value(block,item,request,word):return ((17*((block*32+item%32)*r+request)*W+17*word+seed) if g else (29*((block*s+item%s)*r+request)*W+29*word+seed))&0xffffffff
    def u64(values,index):return uint(values[index],32)|(uint(values[index+1],32)<<32)
    for n,i in enumerate(LENGTHS):
        a={leaf:arrays[f'stage_{n}_{leaf}.u32le'] for leaf in shape_layout(b,i,s,r)}
        for leaf,(shape,count) in shape_layout(b,i,s,r).items():
            require(len(a[leaf])==count,'S16 actual array size');require(all(type(x)is int and 0<=x<2**32 for x in a[leaf]),'S16 actual uint32 words')
        # Full payload replay independently walks logical coordinates.
        for block in range(b):
            for item in range(i):
                for request in range(r):
                    off=((block*i+item)*r+request)*W;require(all(v==value(block,item,request,w) for w,v in enumerate(a['trace'][off:off+W])),'S16 full trace')
            for slot in range(s):
                last=slot+(i-1-slot)//s*s if slot<i else None
                for request in range(r):
                    off=((block*s+slot)*r+request)*W;require(all(v==(value(block,slot if last is None else last,request,w)^(0xffffffff if g and last is None else 0)) for w,v in enumerate(a['final_slots'][off:off+W])),'S16 final shared slots')
            for slot in range(32):
                last=slot+(i-1-slot)//32*32 if slot<i else None
                for request in range(r):
                    off=4+((block*32+slot)*r+request)*W;require(all(v==(value(block,slot if g or last is None else last,request,w)^(0xffffffff if not g and last is None else 0)) for w,v in enumerate(a['ring_guards'][off:off+W])),'S16 full global ring')
            expected=[i*r,0 if g else i,i if g else 0,i,i,i,max(0,i-s),0 if g else 1,0,None if g else 0,s if g else 0,s if g else 0]
            for field,e in enumerate(expected):
                v=u64(a['counts'],(block*12+field)*2);require((v>=i and v<2**64-1) if e is None else v==e,'S16 lifecycle counts')
            for item in range(i):
                expected=[item%s,item//s,None if g else 0,1,1,1,r,0 if g else item+1]
                for field,e in enumerate(expected):
                    v=u64(a['lifecycle'],((block*i+item)*8+field)*2);require(e is None or v==e,'S16 item lifecycle')
            t=[u64(a['stamps'],(block*5+k)*2) for k in range(5)];require(all(x<2**64-1 for x in t[:4]) and t[4]<2**32-1 and t[1]>=t[0] and t[3]>=t[2],'S16 complete stamps')
        require(a['ring_guards'][:4]==[0xd15ea5e0+k for k in range(4)] and a['ring_guards'][-4:]==[0xd15ea5e4+k for k in range(4)],'S16 guards')
    return {'status':'pass','full_values_replayed':True,'opaque_token_words_preserved_domain_only':sum(2*b*i for i in LENGTHS) if g else 0,'opaque_token_exact_checked':False,'performance_eligible':False}

def validate_prior_evidence(evidence,request,artifacts):
    return {'status':'insufficient','reason':'S16 requires new fixed short execution; prior receipt cannot grant B3'}


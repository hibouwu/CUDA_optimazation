"""Fixed cfg_c producer-tail packets; numeric, artifact and timing gates are external."""
import math
from v08_model import scheduled_work
VERSION='cfg-c-producer-tail-v1'
WORDS=1296

def equal(a,b):
    return type(a) is type(b) and (len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b)) if isinstance(b,list) else a==b)

def decode(setup,call,gpu_uuid):
    fixed=dict(config='cfg_c',m=2304,n=3072,k=4096,tile=[256,128,64],cluster=[1,2],grid=[66,2,1],
        stages=4,sm_count=132,swizzle=1,trace_words=WORDS,trace_tile_words=10,trace_version=VERSION,
        traced=True,scratch_bytes=132*WORDS*8,storage_m=2304,storage_n=3072,input_map_m=2304,input_map_n=3072,
        zero_m=-1,zero_n=-1,lda=4096,alloc_lda=4096,ldb=3072,alloc_ldb=3072,ldd=3072,
        input_mode='dyadic',seed=17,history_mode='off',gpu_uuid=gpu_uuid)
    if any(not equal(setup.get(k),v) for k,v in fixed.items()):raise ValueError('fixed producer-tail setup mismatch')
    flag=setup.get('producer_tail_observer')
    if type(flag) is not int or flag not in (0,1) or not equal(call.get('producer_tail_observer'),flag):
        raise ValueError('explicit matching observer flag required')
    for packet in (setup,call):
        if not equal(packet.get('producer_tail_prefix_flag'),0) or packet.get('warmup_policy')!='common-sw1-16':
            raise ValueError('prefix must disable new observer')
        for op in ('a','b'):
            ptr=packet.get('target_'+op+'_ptr')
            if type(ptr) is not int or ptr<=0 or not equal(packet.get('prefix_'+op+'_ptr'),ptr):
                raise ValueError('same-buffer prefix required')
    if any(setup['target_'+op+'_ptr']!=call['target_'+op+'_ptr'] for op in ('a','b')):
        raise ValueError('setup/call buffer mismatch')
    warm=call.get('warmup_us')
    if not equal(call.get('warmup_calls'),16) or not isinstance(warm,list) or len(warm)!=16 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in warm):
        raise ValueError('complete sixteen-call prefix required')
    trace=call.get('trace')
    if not isinstance(trace,list) or len(trace)!=132*WORDS or any(type(v) is not int or not 0<=v<2**64 for v in trace):
        raise ValueError('complete native uint64 trace required')
    work=scheduled_work('cfg_c',2304,3072,[66,2,1],1);records=[];errors=[]
    for cta in range(132):
        b=trace[cta*WORDS:(cta+1)*WORDS]
        try:
            if b[10] or not 1<=b[2]<=132 or b[11]!=b[2] or b[12]!=b[2]:raise ValueError('invalid fixed-device SM/overflow')
            if not 0<b[0]<=b[3] or not 0<b[1]<=b[13]:raise ValueError('invalid entry/producer start')
            if flag:
                if not b[3]<=b[14] or not b[13]<=b[15]:raise ValueError('missing/reversed producer-tail endpoint')
                tail=dict(cycles=b[14],ns=b[15])
            else:
                if b[14] or b[15]:raise ValueError('disabled observer wrote tail')
                tail=None
            roles=[]
            for wg in (0,1):
                count=b[6+wg*3];offset=16+wg*640
                if count!=len(work[cta]) or not count:raise ValueError('incomplete role work')
                final=b[4+wg*3:6+wg*3];previous=[b[3],b[13]];tiles=[]
                for j,coord in enumerate(work[cta]):
                    t=b[offset+j*10:offset+(j+1)*10]
                    if t[4:6]!=[v+1 for v in coord]:raise ValueError('role coordinate mismatch')
                    for values,start,end in ((t[:4],previous[0],final[0]),(t[6:10],previous[1],final[1])):
                        if not start<=values[0]<values[1]<=values[2]<=values[3]<=end:raise ValueError('invalid role endpoint order')
                    tiles.append(dict(work=coord,cycles=t[:4],ns=t[6:10]));previous=[t[3],t[9]]
                if any(b[offset+count*10:offset+640]):raise ValueError('unused tile records nonzero')
                roles.append(dict(wg=wg,physical_thread=128*(wg+1),tiles=tiles,final_cycles=final[0],final_ns=final[1]))
            records.append(dict(cta=cta,sm=b[2]-1,producer_tail=tail,roles=roles))
        except ValueError as e:errors.append(dict(cta=cta,error=str(e)))
    return dict(records=records,errors=errors,record_valid=not errors and len(records)==132,observation_qualified=False,
        limits='Fixed complete GH200 contract only. Tail and consumer FINAL have no imposed relative order. No D visibility or kernel completion claim; identity hashes and numerical/perturbation checks remain external.')

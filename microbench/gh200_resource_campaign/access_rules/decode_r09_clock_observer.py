"""Strict packet validation for trace12. Archive/binary binding stays with replay."""
import math
from r09_clock_brackets import observer_L_window
from v08_model import scheduled_work

VERSIONS={'plain':'none','matched_dual':'r09-matched-dual12/v1','guarded_bracket':'r09-guarded-dual12/v1'}

def matches(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,list):return len(actual)==len(expected) and all(matches(a,b) for a,b in zip(actual,expected))
    return actual==expected

def decode(setup,call,arm,gpu_uuid):
    if arm not in VERSIONS:raise ValueError('explicit supported arm required')
    fixed=dict(config='cfg_b',m=2816,n=1536,k=4096,tile=[128,128,64],cluster=[1,1],
        grid=[1,132,1],sm_count=132,stages=6,swizzle=1,lda=4096,ldb=1536,ldd=1536,alloc_lda=4104,alloc_ldb=1536,
        storage_m=2816,storage_n=1536,input_map_m=2816,input_map_n=1536,zero_m=-1,zero_n=-1,
        input_mode='dyadic',seed=17,trace_tile_words=12,trace_words=1552,scratch_bytes=1638912,
        trace_version=VERSIONS[arm],gpu_uuid=gpu_uuid)
    if any(not matches(setup.get(k),v) for k,v in fixed.items()):raise ValueError('observer setup mismatch')
    common=dict(r09_clock_observer_protocol='r09-clock-observer/v1',r09_clock_observer_arm=arm,
        plain_prefix_protocol='r09-plain-module-sw1-16/v1',plain_prefix_abi=1,
        plain_prefix_preparation='trace-memset-device-sync-per-call',plain_prefix_clear_bytes=1638912,
        warmup_policy='common-sw1-16')
    for packet in (setup,call):
        if any(not matches(packet.get(k),v) for k,v in common.items()):raise ValueError('protocol/arm/prefix preparation mismatch')
        for operand in ('a','b'):
            before=packet.get('prefix_'+operand+'_ptr');after=packet.get('target_'+operand+'_ptr')
            if type(before) is not int or before<=0 or before!=after:
                raise ValueError('within-process prefix/target buffers differ')
    for operand in ('a','b'):
        if setup['target_'+operand+'_ptr']!=call['target_'+operand+'_ptr']:
            raise ValueError('setup/call buffer identities differ')
    if call.get('plain_prefix_same_buffers') is not True or call.get('plain_prefix_calls')!=16 or call.get('warmup_calls')!=16:
        raise ValueError('sixteen same-buffer prefix calls required')
    for field,count in (('plain_prefix_inter_call_us',16),('warmup_us',16)):
        values=call.get(field)
        if not isinstance(values,list) or len(values)!=count or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
            raise ValueError('complete finite prefix timing record required')
    trace=call.get('trace')
    if arm=='plain':
        if setup.get('traced') is not False or trace!=[]:raise ValueError('plain must have no device trace')
        return dict(arm=arm,records=[],record_valid=True,observation_qualified=False)
    if setup.get('traced') is not True or not isinstance(trace,list) or len(trace)!=132*1552:
        raise ValueError('observing trace12 capacity mismatch')
    if any(type(v) is not int or not 0<=v<2**64 for v in trace):raise ValueError('native uint64 trace required')
    work=scheduled_work('cfg_b',2816,1536,[1,132,1],1);records=[];errors=[]
    for cta in range(132):
        b=trace[cta*1552:(cta+1)*1552]
        for role in (1,2):
            try:
                if b[10] or not 1<=b[2]<=setup['sm_count'] or b[10+role]!=b[2]:raise ValueError('overflow/missing/out-of-range/different SM')
                if any(b[i]<=0 for i in (0,1,3,13)):raise ValueError('missing entry/producer endpoint')
                if b[3]<b[0] or b[13]<b[1]:raise ValueError('producer precedes entry')
                if len(work[cta])!=2 or b[6+(role-1)*3]!=1:raise ValueError('one output per original role required')
                offset=16+(role-1)*64*12;tile=b[offset:offset+12]
                if any(b[offset+12:offset+64*12]):raise ValueError('unexpected extra output records')
                window=observer_L_window(tile,arm)
                if tile[0]<b[3] or tile[6]<b[13]:raise ValueError('first MMA precedes producer')
                if window['work_coordinate']!=list(work[cta][role-1]):raise ValueError('role work mismatch')
                final=b[4+(role-1)*3:6+(role-1)*3]
                if final[0]<tile[3] or final[1]<tile[9]:raise ValueError('final precedes epilogue completion')
                records.append(dict(cta=cta,role=role,physical_thread=128*role,sm=b[2]-1,window=window))
            except ValueError as error:errors.append(dict(cta=cta,role=role,error=str(error)))
    return dict(arm=arm,records=records,errors=errors,record_valid=not errors and len(records)==264,
        observation_qualified=False,limits='Packet validation only; source/binary/module/raw SHA binding and hardware enclosure/perturbation qualification remain external.')

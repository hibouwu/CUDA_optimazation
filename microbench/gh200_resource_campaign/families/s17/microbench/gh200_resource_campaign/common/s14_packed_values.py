"""Code-owned complete S14 historical uint32 chunk observer; no GPU.

Each role/shape descriptor must be derived from the fixed84 signed raw records
by the finite policy. The resolver independently hashes every entire member.
This observer checks every numerical word with bounded chunks, not sampling.
"""
import numpy as np
from common.suite_io import require

PAYLOAD_ROLES={'bulk_guards.u32le','bulk_trace.u32le','bulk_ring.u32le','bulk_source_after.u32le'}
LIFECYCLE_ROLES={'bulk_completion.u32le','bulk_release_clocks.u32le'}


class S14WordObserver:
    def __init__(self,descriptors):
        require(isinstance(descriptors,dict) and descriptors,'fixed full S14 artifact descriptors')
        self.descriptors=descriptors
        self.states={}
        for name,d in descriptors.items():
            require(d['role'] in PAYLOAD_ROLES|LIFECYCLE_ROLES,'finite S14 artifact role')
            count=1
            for n in d['shape']:require(type(n) is int and n>0,'artifact shape domain');count*=n
            require(d['bytes']==count*4,'lossless uint32 shape bytes')
            require(d['blocks']>0 and d['words'] in (256,1024,2048,4096,8192,16384)
                    and d['iterations'] in (1,2,33) and d['seed'] in (0,3,4294967295),'fixed original84 word domain')
            self.states[name]={'offset':0,'words':0,'remainder':b'','control':[]}

    def __call__(self,name,offset,chunk):
        if name not in self.descriptors:return
        d=self.descriptors[name];s=self.states[name]
        require(type(offset) is int and offset==s['offset'],'contiguous complete artifact stream')
        require(isinstance(chunk,bytes) and 0<len(chunk)<=1048576,'bounded observer byte chunk')
        s['offset']+=len(chunk);require(s['offset']<=d['bytes'],'artifact exceeds known length')
        combined=s['remainder']+chunk;usable=len(combined)//4*4;s['remainder']=combined[usable:]
        values=np.frombuffer(combined[:usable],dtype='<u4');begin=s['words'];s['words']+=len(values)
        role=d['role']
        if role in LIFECYCLE_ROLES:
            require(len(s['control'])+len(values)<=d['blocks']*10,'bounded lifecycle record')
            s['control'].extend(map(int,values));return
        index=np.arange(begin,begin+len(values),dtype=np.uint64);w=d['words'];steps=d['iterations'];seed=d['seed']
        if role=='bulk_guards.u32le':expected=(0xd15ea5e0+index).astype(np.uint32)
        elif role=='bulk_source_after.u32le':expected=(29*index+seed).astype(np.uint32)^np.uint32(0xffffffff)
        elif role=='bulk_trace.u32le':
            block=index//(steps*w);phase=(index//w)%steps;word=index%w
            coordinate=(block*32+phase%32)*w+word if d['g2s'] else block*w+word
            expected=((17 if d['g2s'] else 29)*coordinate+seed).astype(np.uint32)
        else:
            block=index//(32*w);slot=(index//w)%32;word=index%w
            base=(29*(block*w+word)+seed).astype(np.uint32)
            expected=np.where((steps>=32)|(slot<steps),base,base^np.uint32(0xffffffff))
        if not np.array_equal(values,expected):
            at=int(np.flatnonzero(values!=expected)[0]);raise ValueError(f'S14 full packed word differs: {name} word={begin+at} actual={int(values[at])} expected={int(expected[at])}')

    def finish(self):
        payload=control=0;per_run={}
        for name,d in self.descriptors.items():
            s=self.states[name];require(s['offset']==d['bytes'] and s['words']*4==d['bytes'] and not s['remainder'],'missing or truncated complete artifact: '+name)
            role=d['role'];count=s['words'];entry=per_run.setdefault(d['run'],{'payload_guard_words':0,'lifecycle_words':0,'artifact_count':0});entry['artifact_count']+=1
            if role in PAYLOAD_ROLES:payload+=count;entry['payload_guard_words']+=count;continue
            values=s['control'];require(len(values)%2==0,'whole low/high uint64 pairs')
            pairs=[values[i]|(values[i+1]<<32) for i in range(0,len(values),2)]
            require(len(pairs)==d['blocks']*(5 if role=='bulk_completion.u32le' else 3),'full lifecycle pair layout')
            for b in range(d['blocks']):
                if role=='bulk_completion.u32le':
                    done,attempts,timeout,readwait,fullwait=pairs[5*b:5*b+5]
                    require(done==d['iterations'] and timeout==0 and readwait==int(d['release'])
                            and fullwait==(0 if d['g2s'] else d['iterations']),'complete request lifecycle')
                    require(d['iterations']<=attempts<2**64-1 if d['g2s'] else attempts==0,'actual wait attempts')
                else:
                    start,released,full=pairs[3*b:3*b+3];require(start<=released<=full<2**64-1,'complete same-CTA source-release clocks')
            control+=count;entry['lifecycle_words']+=count
        return {'status':'full_values_match_fixed_original84_reference','payload_guard_words':payload,
                'lifecycle_words':control,'artifact_count':len(self.descriptors),'per_run':per_run,
                'sampling':False,'GPU_execution':False,'bridge_qualified':False}

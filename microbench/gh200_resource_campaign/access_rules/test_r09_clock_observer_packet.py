import copy,unittest
from decode_r09_clock_observer import decode,VERSIONS
from v08_model import scheduled_work

def packet(arm='guarded_bracket'):
    s=dict(config='cfg_b',m=2816,n=1536,k=4096,tile=[128,128,64],cluster=[1,1],grid=[1,132,1],sm_count=132,stages=6,
        swizzle=1,lda=4096,ldb=1536,ldd=1536,alloc_lda=4104,alloc_ldb=1536,storage_m=2816,storage_n=1536,
        input_map_m=2816,input_map_n=1536,zero_m=-1,zero_n=-1,input_mode='dyadic',seed=17,
        trace_tile_words=12,trace_words=1552,scratch_bytes=1638912,trace_version=VERSIONS[arm],gpu_uuid='synthetic',traced=arm!='plain')
    common=dict(r09_clock_observer_protocol='r09-clock-observer/v1',r09_clock_observer_arm=arm,
        plain_prefix_protocol='r09-plain-module-sw1-16/v1',plain_prefix_abi=1,
        plain_prefix_preparation='trace-memset-device-sync-per-call',plain_prefix_clear_bytes=1638912,
        warmup_policy='common-sw1-16',prefix_a_ptr=1024,target_a_ptr=1024,prefix_b_ptr=2048,target_b_ptr=2048)
    s.update(common);c=dict(common,plain_prefix_same_buffers=True,plain_prefix_calls=16,warmup_calls=16,
        plain_prefix_inter_call_us=[1.]*16,warmup_us=[20.]*16,trace=[])
    if arm=='plain':return s,c
    work=scheduled_work('cfg_b',2816,1536,[1,132,1],1);t=[0]*(132*1552)
    for cta in range(132):
        base=cta*1552;t[base:base+4]=[10,10,7,20];t[base+13]=20
        for role in (1,2):
            t[base+10+role]=7;t[base+4+(role-1)*3:base+7+(role-1)*3]=[600,600,1]
            start=base+16+(role-1)*768
            t[start:start+12]=[100,300,400,500,*[v+1 for v in work[cta][role-1]],100,300,400,500,*([110,310] if arm=='guarded_bracket' else [0,0])]
    c['trace']=t;return s,c

class PacketTest(unittest.TestCase):
    def test_rejects_missing_or_reversed_header_endpoints(self):
        for index,value in [(0,0),(1,0),(3,0),(13,0),(0,21),(1,21),(3,101),(13,101)]:
            with self.subTest(index=index,value=value):
                s,c=packet();c['trace'][index]=value
                result=decode(s,c,'guarded_bracket','synthetic')
                self.assertFalse(result['record_valid'])
                self.assertEqual([(e['cta'],e['role']) for e in result['errors']],[(0,1),(0,2)])
                self.assertEqual(len(result['records']),262)
    def test_rejects_consistent_but_out_of_range_sm(self):
        s,c=packet()
        for index in (2,11,12):c['trace'][index]=133
        result=decode(s,c,'guarded_bracket','synthetic')
        self.assertFalse(result['record_valid']);self.assertEqual(len(result['errors']),2)
    def test_three_arms(self):
        for arm in VERSIONS:
            result=decode(*packet(arm),arm,'synthetic');self.assertTrue(result['record_valid']);self.assertFalse(result['observation_qualified'])
    def test_rejects_old_layout_and_wrong_preparation(self):
        for field,value in [('trace_tile_words',10),('plain_prefix_clear_bytes',1391808),('r09_clock_observer_arm','plain')]:
            s,c=packet();s[field]=value
            with self.assertRaises(ValueError):decode(s,c,'guarded_bracket','synthetic')
    def test_keeps_invalid_roles(self):
        s,c=packet();c['trace'][16+10]=0
        result=decode(s,c,'guarded_bracket','synthetic');self.assertFalse(result['record_valid']);self.assertEqual(len(result['records']),263);self.assertEqual(result['errors'][0]['cta'],0)
    def test_wrong_work_sm_and_extra_outputs(self):
        for index,value in [(16+4,1000),(11,8),(16+12,1)]:
            s,c=packet();c['trace'][index]=value
            self.assertFalse(decode(s,c,'guarded_bracket','synthetic')['record_valid'])
    def test_rejects_noninteger_and_incomplete_history(self):
        s,c=packet();c['trace'][16]=True
        with self.assertRaises(ValueError):decode(s,c,'guarded_bracket','synthetic')
        s,c=packet();c['plain_prefix_inter_call_us'].pop()
        with self.assertRaises(ValueError):decode(s,c,'guarded_bracket','synthetic')
    def test_setup_call_pointer_disagreement_is_rejected(self):
        s,c=packet();s['prefix_a_ptr']=s['target_a_ptr']=4096
        with self.assertRaises(ValueError):decode(s,c,'guarded_bracket','synthetic')
    def test_no_cross_process_pointer_requirement(self):
        s,c=packet();s['prefix_a_ptr']=s['target_a_ptr']=c['prefix_a_ptr']=c['target_a_ptr']=4096
        self.assertTrue(decode(s,c,'guarded_bracket','synthetic')['record_valid'])

if __name__=='__main__':unittest.main()

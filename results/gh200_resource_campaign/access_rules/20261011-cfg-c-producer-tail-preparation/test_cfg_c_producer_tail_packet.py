import unittest
from decode_cfg_c_producer_tail import decode,VERSION,WORDS
from v08_model import scheduled_work

def packet(flag=1):
    s=dict(config='cfg_c',m=2304,n=3072,k=4096,tile=[256,128,64],cluster=[1,2],grid=[66,2,1],
        stages=4,sm_count=132,swizzle=1,trace_words=WORDS,trace_tile_words=10,trace_version=VERSION,traced=True,
        scratch_bytes=132*WORDS*8,storage_m=2304,storage_n=3072,input_map_m=2304,input_map_n=3072,
        zero_m=-1,zero_n=-1,lda=4096,alloc_lda=4096,ldb=3072,alloc_ldb=3072,ldd=3072,
        input_mode='dyadic',seed=17,history_mode='off',gpu_uuid='synthetic')
    common=dict(producer_tail_observer=flag,producer_tail_prefix_flag=0,warmup_policy='common-sw1-16',
        prefix_a_ptr=1024,target_a_ptr=1024,prefix_b_ptr=2048,target_b_ptr=2048)
    s.update(common);c=dict(common,warmup_calls=16,warmup_us=[1.]*16,trace=[0]*(132*WORDS))
    for cta,work in enumerate(scheduled_work('cfg_c',2304,3072,[66,2,1],1)):
        base=cta*WORDS;t=c['trace'];t[base:base+4]=[1,1,1,2];t[base+11:base+16]=[1,1,2,40*flag,40*flag]
        for wg in (0,1):
            t[base+4+3*wg:base+7+3*wg]=[90,90,len(work)]
            for j,coord in enumerate(work):
                offset=base+16+wg*640+j*10;v=10+30*j
                t[offset:offset+10]=[v,v+10,v+11,v+20,*[x+1 for x in coord],v,v+10,v+11,v+20]
    return s,c

class PacketTests(unittest.TestCase):
    def test_both_arms_and_unqualified(self):
        for flag in (0,1):
            r=decode(*packet(flag),'synthetic');self.assertTrue(r['record_valid']);self.assertEqual(len(r['records']),132);self.assertFalse(r['observation_qualified'])
    def test_tail_may_precede_or_follow_consumer_final(self):
        for tail in (3,100):
            s,c=packet();c['trace'][14:16]=[tail,tail];self.assertTrue(decode(s,c,'synthetic')['record_valid'])
    def test_missing_or_off_tail_is_retained_as_error(self):
        for flag,tail in ((1,0),(0,40)):
            s,c=packet(flag);c['trace'][14:16]=[tail,tail];r=decode(s,c,'synthetic');self.assertFalse(r['record_valid']);self.assertEqual(r['errors'][0]['cta'],0);self.assertEqual(len(r['records']),131)
    def test_rejects_protocol_types_and_old_version(self):
        for key,value in (('trace_version','r18-dual-clock-events'),('producer_tail_observer',True),('stages',6),('grid',[132,1,1])):
            s,c=packet();s[key]=value
            with self.assertRaises(ValueError):decode(s,c,'synthetic')
    def test_rejects_prefix_history_and_pointer_mismatch(self):
        for key,value in (('warmup_calls',15),('warmup_calls',True),('producer_tail_prefix_flag',1),('target_a_ptr',4096),('warmup_us',[1.]*15)):
            s,c=packet();c[key]=value
            with self.assertRaises(ValueError):decode(s,c,'synthetic')
    def test_rejects_bad_cta_work_sm_or_header(self):
        for index,value in ((0,0),(3,0),(13,0),(2,133),(11,2),(6,0),(20,999),(16+20,1)):
            s,c=packet();c['trace'][index]=value
            self.assertFalse(decode(s,c,'synthetic')['record_valid'])
    def test_rejects_noninteger_or_short_trace(self):
        s,c=packet();c['trace'][14]=40.0
        with self.assertRaises(ValueError):decode(s,c,'synthetic')
        s,c=packet();c['trace'].pop()
        with self.assertRaises(ValueError):decode(s,c,'synthetic')

if __name__=='__main__':unittest.main()

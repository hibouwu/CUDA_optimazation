import copy
import unittest
from decode_r18_first_output import decode, VERSION, WORDS
from analyze_r18_first_output import SELECTED_CTAS
from v08_model import scheduled_work


def fixture(flag=1):
    setup=dict(config='cfg_a',m=960,n=2816,k=4096,grid=[132,1,1],tile=[128,128,64],
        cluster=[2,1],stages=6,trace_words=WORDS,trace_tile_words=10,trace_version=VERSION,
        first_output_observer=flag,first_output_warmup=0,warmup_policy='common-sw1-16',
        swizzle=1,storage_m=1024,storage_n=2816,zero_m=960,zero_n=-1,input_map_m=960,input_map_n=2816,
        lda=4096,ldb=2816,ldd=2816,alloc_lda=4096,alloc_ldb=2816,input_mode='dyadic',seed=17)
    call=dict(first_output_observer=flag,first_output_warmup=0,warmup_policy='common-sw1-16',
        warmup_calls=16,trace=[0]*(132*WORDS))
    work=scheduled_work('cfg_a',960,2816,[132,1,1],1)
    for cta in SELECTED_CTAS:
        offset=cta*WORDS;trace=call['trace'];trace[offset+2]=7
        for role in (1,2):
            trace[offset+10+role]=7;head=offset+16+(role-1)*640
            trace[head:head+2]=[2**60,2**60+60]
            trace[head+6:head+8]=[2**60,2**60+60]
            trace[head+4:head+6]=[v+1 for v in work[cta][0]]
            extra=offset+1296+(role-1)*11;trace[extra]=0x52313810+flag
            if flag:
                trace[extra+1:extra+11]=[2**60+10*i for i in range(1,6) for _ in (0,1)]
    return setup,call


class ProbeDecodeTest(unittest.TestCase):
    def test_complete_integer_coverage(self):
        result=decode(*fixture(),'synthetic')
        self.assertEqual(len(result['records']),24)
        self.assertFalse(result['observation_qualified'])
        self.assertTrue(all(row['partition']['L']==dict(cycles=60,ns=60) for row in result['records']))

    def test_off_has_metadata_but_no_cuts(self):
        result=decode(*fixture(0),'synthetic')
        self.assertEqual(result['records'],[])
        setup,call=fixture(0);call['trace'][1297]=1
        with self.assertRaisesRegex(ValueError,'disabled'):decode(setup,call,'synthetic')

    def test_rejects_missing_weak_cta_or_wrong_sm(self):
        for index in (6*WORDS+1296,7*WORDS+1296,11,12):
            setup,call=fixture();call['trace'][index]=0
            with self.assertRaises(ValueError):decode(setup,call,'synthetic')

    def test_rejects_nonselected_and_out_of_order_writes(self):
        for index,value in ((2*WORDS+1296,1),(1299,1),(1297,float(2**60))):
            setup,call=fixture();call['trace'][index]=value
            with self.assertRaises(ValueError):decode(setup,call,'synthetic')

    def test_rejects_identity_layout_history_changes(self):
        for field,value in (('trace_words',1296),('trace_version','r18-dual-clock-events'),
                            ('stages',4),('first_output_observer',True),('input_map_m',961),('zero_m',1024)):
            setup,call=fixture();setup[field]=value
            with self.assertRaises(ValueError):decode(setup,call,'synthetic')
        setup,call=fixture();call['warmup_calls']=15
        with self.assertRaises(ValueError):decode(setup,call,'synthetic')


if __name__=='__main__':unittest.main()

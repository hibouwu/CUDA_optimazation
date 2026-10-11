import copy
from pathlib import Path
import subprocess
import tempfile
import unittest

from decode_r13_wait_sum import decode,decode_trace,MASK,VERSION
from analyze_r18 import reference


class WaitSumTests(unittest.TestCase):
    def fixture(self,flag=1):
        setup=dict(config='cfg_b',m=2816,n=1536,k=4096,grid=[1,132,1],tile=[128,128,64],
            cluster=[1,1,1],stages=4,trace_words=784,trace_tile_words=6,trace_version=VERSION,
            input_mode='dyadic',seed=17,swizzle=1,lda=4096,ldb=1536,ldd=1536,alloc_lda=4104,alloc_ldb=1536,
            storage_m=2816,storage_n=1536,input_map_m=2816,input_map_n=1536,zero_m=-1,zero_n=-1,evict=0,
            raster_order='heuristic',resolved_raster_order='along_n',mode='warm',gpu_uuid='test-gpu')
        history=dict(wait_sum_protocol=VERSION,wait_sum_observer=flag,wait_sum_prefix_flag=0,
            wait_sum_first_group=2,wait_sum_last_group=16,warmup_policy='common-sw1-16',
            warmup_fixed_calls=16,warmup_swizzle=1)
        setup.update(history)
        words=[0]*(132*784)
        for index,value in {1:7,23:7,13:12,14:1,21:90,20:100,15:190,17:200,18:205,
                            16:210,19:220,24:600,32:610,22:650,31:0x52313309,40:flag}.items():
            words[index]=value
        if flag:
            for index,value in {34:50,35:15,36:MASK,37:110,38:195}.items():words[index]=value
        call=dict(history,label='warm',target_swizzle=1,warmup_calls=16,warmup_us=[50.]*16,
                  elapsed_us=50.,trace=words)
        return setup,call

    def test_partition_and_disabled_is_missing(self):
        setup,call=self.fixture()
        result=decode_trace(setup,call)
        self.assertEqual((result['early_cycles'],result['full_wait_cycles'],result['outside_wait_cycles']),(100,50,50))
        self.assertIsNone(result['full_wait_ns'])
        self.assertFalse(result['observation_qualified'])
        setup,call=self.fixture(0)
        self.assertIsNone(decode_trace(setup,call)['full_wait_cycles'])
        call['trace'][34]=1
        with self.assertRaisesRegex(ValueError,'disabled'):decode_trace(setup,call)

    def test_missing_duplicate_and_escaping_intervals_rejected(self):
        for index,value in [(35,14),(36,MASK^4),(39,2),(37,99),(38,201),(34,86)]:
            setup,call=self.fixture();call['trace'][index]=value
            with self.subTest(index=index),self.assertRaises(ValueError):decode_trace(setup,call)

    def test_history_identity_other_cta_and_native_integer_rejected(self):
        for change in ('prefix','sm','work','other','float'):
            setup,call=self.fixture()
            if change=='prefix':call['wait_sum_prefix_flag']=1
            elif change=='sm':call['trace'][23]=8
            elif change=='work':call['trace'][13]=1
            elif change=='other':call['trace'][784]=1
            else:call['trace'][34]=50.0
            with self.subTest(change=change),self.assertRaises(ValueError):decode_trace(setup,call)

    def test_numerical_witness_checked_independently(self):
        setup,call=self.fixture();setup['event']='setup';call['event']='call'
        indices=list(range(4096))
        check=dict(event='check',status='ok',samples=4096,nonfinite=0,padding_errors=0,
            max_storage_reference_error=0,checked_indices=indices,
            checked_values=[reference((i//1536)%17,(i%1536)%17,4096,17) for i in indices])
        result=decode([setup,call,check],'test-gpu')
        self.assertEqual(result['independently_checked_values'],4096)
        with self.assertRaisesRegex(ValueError,'wrong GPU'):decode([setup,call,check],'other-gpu')
        check['checked_values'][200]+=1
        with self.assertRaisesRegex(ValueError,'independent GEMM'):decode([setup,call,check],'test-gpu')

    def test_actual_cpp_collector_partition_and_bad_sequences(self):
        # Execute the same accumulator body used by CUDA on the CPU. This checks
        # arithmetic and rejection bits, not compiler scheduling on the GPU.
        code=r'''
#include "r13_wait_sum.cuh"
#include <cassert>
int main() {
  R13WaitSum s;
  uint64_t previous_commit=100, outside=0;
  for (int i=2;i<=16;++i) {
    uint64_t begin=previous_commit+3, end=begin+i, commit=end+7;
    s.record(i,begin,end);
    outside+=(begin-previous_commit)+(commit-end);
    previous_commit=commit;
  }
  assert(s.count==15 && s.mask==0x1fffc && !s.invalid);
  assert(s.cycles==135 && outside==150 && previous_commit-100==s.cycles+outside);
  s.record(16,s.last_end,s.last_end+1); assert(s.invalid&2);
  R13WaitSum bad; bad.record(1,100,200); assert(bad.invalid&1);
  R13WaitSum reversed; reversed.record(2,200,100); assert(reversed.invalid&4);
  R13WaitSum overlap; overlap.record(2,100,200); overlap.record(3,199,201); assert(overlap.invalid&4);
  R13WaitSum wrap; wrap.cycles=UINT64_MAX; wrap.record(2,1,2); assert(wrap.invalid&8);
}
'''
        with tempfile.TemporaryDirectory(prefix='r13-wait-sum-') as directory:
            source=Path(directory)/'test.cpp';source.write_text(code)
            binary=Path(directory)/'test'
            subprocess.run(['g++','-std=c++17','-I'+str(Path(__file__).parent),str(source),'-o',str(binary)],check=True,capture_output=True)
            subprocess.run([str(binary)],check=True,capture_output=True)


if __name__=='__main__':unittest.main()

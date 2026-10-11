from pathlib import Path
import subprocess
import tempfile
import unittest
import test_r13_wait_sum as old_fixture
from decode_r13_wait_feedback import decode_trace, decode, VERSION, MODE_MASKS
from analyze_r18 import reference


class FeedbackTests(unittest.TestCase):
    def fixture(self,mode=1,stage=4,offset=0):
        setup,call=old_fixture.WaitSumTests().fixture(0)
        setup.update(trace_version=VERSION,stages=stage)
        for packet in (setup,call):
            packet.update(wait_sum_protocol=VERSION,wait_sum_observer=mode,wait_sum_selection_mask=MODE_MASKS[mode])
        b=call['trace'];b[31]=0x5231330a;b[40]=MODE_MASKS[mode]
        if mode==1:
            values={34:50,35:15,36:0x1fffc,37:110,38:195,41:4,42:3}
        elif mode==2:values={34:4,35:1,36:4,37:110,38:114,41:4}
        elif mode==3:values={34:3,35:1,36:65536,37:192,38:195,42:3}
        else:values={}
        for index,value in values.items():b[index]=value
        for index in (15,16,17,18,19,20,21,22,24,32,37,38):
            if b[index]:b[index]+=offset
        return setup,call

    def test_all_modes_and_large_native_clocks(self):
        for mode in range(4):
            for stage in (4,6):
                for offset in (0,2**63+17):
                    setup,call=self.fixture(mode,stage,offset)
                    result=decode_trace(setup,call)
                    self.assertEqual(result['early_cycles'],100)
                    self.assertEqual(result['group2_wait_cycles'],4 if mode in (1,2) else None)
                    self.assertEqual(result['group16_wait_cycles'],3 if mode in (1,3) else None)
                    self.assertEqual(result['full_wait_cycles'],50 if mode==1 else None)
                    self.assertEqual(result['outside_wait_cycles'],50 if mode==1 else None)
                    self.assertIsNone(result['full_wait_ns'])
                    self.assertFalse(result['observation_qualified'])

    def test_invalid_mode_mask_counts_endpoints_and_missingness(self):
        for mode,index,value in [(0,41,1),(1,35,14),(1,36,4),(1,39,2),(1,41,81),
                                 (1,42,6),(1,34,6),(2,42,1),(2,34,3),(2,38,115),
                                 (3,41,1),(3,37,191),(3,40,4)]:
            setup,call=self.fixture(mode);call['trace'][index]=value
            with self.subTest(mode=mode,index=index),self.assertRaises(ValueError):decode_trace(setup,call)
        setup,call=self.fixture();setup['wait_sum_selection_mask']=4
        with self.assertRaises(ValueError):decode_trace(setup,call)
        setup,call=self.fixture();setup['wait_sum_observer']=True
        with self.assertRaises(ValueError):decode_trace(setup,call)

    def test_old_protocol_rejected(self):
        setup,call=old_fixture.WaitSumTests().fixture()
        with self.assertRaises(ValueError):decode_trace(setup,call)

    def test_actual_cpp_mask_selection_and_recorded_points(self):
        code=r'''
#include "r13_wait_feedback.cuh"
#include <cassert>
int main() {
  for (unsigned mode=0;mode<4;++mode) {
    R13WaitSum s;
    unsigned mask=r13_wait_feedback_mask(mode);
    uint64_t expected=0;
    for (int group=-1;group<=64;++group) {
      bool selected=r13_wait_feedback_selected(mask,group);
      bool wanted=(mode==1 && group>=2 && group<=16) || (mode==2 && group==2) || (mode==3 && group==16);
      assert(selected==wanted);
      if(selected) {s.record(group,100*group,100*group+group);expected+=group;}
    }
    assert(s.mask==mask && s.cycles==expected && s.invalid==0);
    assert(s.count==(mode==0?0:mode==1?15:1));
    assert(s.group2_cycles==((mode==1 || mode==2)?2:0));
    assert(s.group16_cycles==((mode==1 || mode==3)?16:0));
  }
  R13WaitSum bad;bad.record(16,100,90);assert(bad.invalid&4);
  R13WaitSum duplicate;duplicate.record(2,100,101);duplicate.record(2,102,103);assert(duplicate.invalid&2);
  R13WaitSum wrap;wrap.cycles=UINT64_MAX;wrap.record(2,1,2);assert(wrap.invalid&8);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'test.cpp';source.write_text(code)
            binary=Path(directory)/'test'
            subprocess.run(['g++','-std=c++17','-I'+str(Path(__file__).parent),str(source),'-o',str(binary)],check=True,capture_output=True)
            subprocess.run([str(binary)],check=True,capture_output=True)

    def test_four_packets_and_numeric_error(self):
        setup,call=self.fixture(3);setup['event']='setup';call['event']='call'
        indices=list(range(4096))
        check=dict(event='check',status='ok',samples=4096,nonfinite=0,padding_errors=0,
            max_storage_reference_error=0,checked_indices=indices,
            checked_values=[reference((i//1536)%17,(i%1536)%17,4096,17) for i in indices])
        witness=dict(event='input_zero_storage',checked=0,nonzero=0)
        result=decode([setup,call,witness,check],'test-gpu')
        self.assertIsNone(result['full_wait_cycles'])
        self.assertEqual(result['group16_wait_cycles'],3)
        check['checked_values'][0]+=1
        with self.assertRaises(ValueError):decode([setup,call,witness,check],'test-gpu')


if __name__=='__main__':unittest.main()

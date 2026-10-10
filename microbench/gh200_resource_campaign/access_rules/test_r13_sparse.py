import copy
import unittest
from analyze_r18 import decode_stage_sparse

class StageSparseTests(unittest.TestCase):
    def trace(self,stages):
        width=16+2*64*6
        b=[0]*(132*width)
        # Global Ktile = 1 output * 64 Ktiles + local Ktile16.
        global_k=80
        slot=global_k%stages
        cphase=(global_k//stages)%2
        b[1]=b[23]=7
        b[2]=b[13]=12
        b[3]=b[14]=1
        b[4]=b[24]=b[25]=b[27]=slot+1
        b[26]=cphase+1
        b[5]=(1-cphase)+1
        b[28]=cphase+1
        b[29]=1;b[30]=16;b[31]=0x52313301
        for i,value in {0:1,6:20,7:30,8:40,9:125,11:171,12:180,
                        21:10,15:50,16:80,17:100,18:120,19:170,20:172,
                        32:200,33:205,22:250}.items():
            b[i]=value
        return dict(k=4096,stages=stages,grid=[1,132,1],trace_words=width),dict(trace=b)

    def test_real_stage4_and_stage6_slot_phase_sequences(self):
        for stages,slot,consumer_phase in [(4,0,0),(6,2,1)]:
            setup,call=self.trace(stages)
            result=decode_stage_sparse(setup,call)
            self.assertEqual(result['slot'],slot)
            self.assertEqual(result['phase'],1-consumer_phase)
            self.assertEqual(result['intervals']['consumer_full_wait'],30)
            self.assertEqual(result['intervals']['L'],240)
            self.assertEqual(result['intervals']['mainloop_before_tail'],190)
            self.assertEqual(result['intervals']['mma_tail'],45)
            # Acquire can return between release's before/after stamps.
            self.assertLess(result['clocks']['refill_end'],result['clocks']['release_end'])

    def test_same_producer_and_consumer_phase_is_rejected(self):
        setup,call=self.trace(6)
        call['trace'][5]=call['trace'][26]
        with self.assertRaisesRegex(ValueError,'phase mismatch'):
            decode_stage_sparse(setup,call)

    def test_wrong_slot_or_other_CTA_write_is_rejected(self):
        setup,call=self.trace(4)
        wrong=copy.deepcopy(call)
        for idx in (4,24,25,27):wrong['trace'][idx]=2
        with self.assertRaisesRegex(ValueError,'sequence mismatch'):
            decode_stage_sparse(setup,wrong)
        call['trace'][setup['trace_words']]=1
        with self.assertRaisesRegex(ValueError,'non-target CTA'):
            decode_stage_sparse(setup,call)

    def test_progress_allows_sub512_commit_spacing_and_closes(self):
        width=16+2*64*6
        b=[0]*(132*width);b[1]=b[23]=3;b[13]=12;b[14]=1;b[31]=0x52313302
        for i,value in {21:100,20:500,15:8000,17:8300,18:8306,16:8400,
                        19:8700,24:32000,32:32100,22:32500}.items():
            b[i]=value
        setup=dict(grid=[1,132,1],trace_words=width,trace_version='r13-stage-progress-v1')
        r=decode_stage_sparse(setup,dict(trace=b))
        self.assertEqual(r['intervals']['commit16_to17'],400)
        self.assertEqual(r['intervals']['loop_advance_16_to17'],94)
        self.assertEqual(r['intervals']['last_commit_to_main_end'],500)
        self.assertEqual(r['intervals']['first_to_commit63']+
                         r['intervals']['last_commit_to_main_end'],r['intervals']['L'])
        b[19]=8200
        with self.assertRaisesRegex(ValueError,'event order'):
            decode_stage_sparse(setup,dict(trace=b))

    def test_handoff_accepts_early_or_late_receiver_but_not_missing_sender(self):
        width=16+2*64*6
        b=[0]*(132*width);b[1]=b[23]=b[38]=3
        b[13]=12;b[14]=b[39]=b[40]=1;b[31]=0x52313303
        for i,value in {21:300,20:900,15:8000,17:8300,18:8306,16:8400,
                        19:8700,24:32000,32:32100,22:32500,
                        34:100,35:50,36:200,37:230}.items():b[i]=value
        setup=dict(grid=[1,132,1],trace_words=width,trace_version='r13-stage-handoff-v1')
        r=decode_stage_sparse(setup,dict(trace=b));v=r['intervals']
        self.assertEqual(v['sender_to_first'],200)
        self.assertEqual(v['sender_to_math_return']+v['math_return_to_full_begin']+v['first_full_wait'],200)
        self.assertEqual(v['receiver_late_entry'],0)
        self.assertEqual(r['sender_work'],[0,0])
        b[35]=150
        v=decode_stage_sparse(setup,dict(trace=b))['intervals']
        self.assertEqual(v['receiver_late_entry'],50)
        self.assertEqual(v['after_sender_and_wait_entry'],50)
        b[34]=201
        with self.assertRaisesRegex(ValueError,'handoff event order'):
            decode_stage_sparse(setup,dict(trace=b))
        b[34]=100;b[38]=4
        with self.assertRaisesRegex(ValueError,'sender SM'):
            decode_stage_sparse(setup,dict(trace=b))


if __name__=='__main__':unittest.main()

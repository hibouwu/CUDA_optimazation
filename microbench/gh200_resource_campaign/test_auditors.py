import copy
import unittest
from audit_campaign import validate, expected_matrix, check_memory_loops
from audit_legacy import check_trial, operation_work

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.case={"id":"smem_read_stride1","mode":"smem_read","bytes":32768,"iterations":8192,"stride":1}
        total=256*8*4*8192
        self.row={"mode":"smem_read","trial":0,"iterations":8192,"stride":1,"errors":0,
                  "cache_residency_proven":False,"blocks":1,"threads":256,
                  "requested_working_set_bytes":32768,"seed":3,"allocation_per_array_bytes":32768,
                  "read_payload_bytes":total,"write_payload_bytes":0,"observed_sms":1,
                  "start_ns":1000,"stop_ns":2000,"max_cta_cycles":2000,"event_ms":0.002,
                  "payload_gbytes_per_second":total/1000,
                  "blocks_detail":[{"smid":137,"start_ns":1000,"stop_ns":2000,"start_cycle":4000,"stop_cycle":6000}]}
    def test_non_contiguous_sm_ids_valid(self):
        self.assertGreater(validate(self.row,self.case,{"sms":132},0),0)
    def test_work_and_timer_tamper_rejected(self):
        for field in ("read_payload_bytes","stop_ns","max_cta_cycles","payload_gbytes_per_second"):
            row=copy.deepcopy(self.row);row[field]+=1
            with self.assertRaises(ValueError,msg=field):
                validate(row,self.case,{"sms":132},0)
    def test_cache_claim_and_correctness_rejected(self):
        for field,val in (("cache_residency_proven",True),("errors",1)):
            row=copy.deepcopy(self.row);row[field]=val
            with self.assertRaises(ValueError):validate(row,self.case,{"sms":132},0)
    def test_hoisted_load_rejected(self):
        function="""
        /*0010*/ CS2R R0, SR_GLOBALTIMERLO ; /* dummy */
        /*0020*/ LDS R1, [R2] ; /* dummy */
        /*0030*/ IADD3 R4, R1, R4, RZ ; /* dummy */
        /*0040*/ @P0 BRA 0x30 ; /* dummy */
        /*0050*/ CS2R R0, SR_GLOBALTIMERLO ; /* dummy */
        """
        with self.assertRaises(ValueError):check_memory_loops(function,("LDS",))
        check_memory_loops(function.replace("BRA 0x30","BRA 0x20"),("LDS",),1)

    def test_matrix_has_independent_12_cases(self):
        self.assertEqual(len(expected_matrix()),12)
    def test_matrix_work_includes_all_warpgroups(self):
        c={"ptx":"wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16","kind":"wgmma_f16","threads":256}
        self.assertEqual(operation_work(c),2*2*64*64*16)
    def test_initial_work_mismatch_rejected(self):
        c={"ptx":"fma.rn.f32","kind":"f32","threads":32,"chains":8,"batch":16,"work_per_collective":64}
        r={"iterations":128,"max_abs_error":0,"work_flop":128*8*16*64,"cycles":100000,
           "ptx_collectives":128*8*16,"smid":137,"event_ms":1}
        self.assertGreater(check_trial(r,c,"initial",132),0)
        r["work_flop"]//=2
        with self.assertRaises(ValueError):check_trial(r,c,"initial",132)

if __name__=="__main__":unittest.main()

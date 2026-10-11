import copy,unittest
from cfg_c_native_compute import compute_reference

class NativeCompute(unittest.TestCase):
    def setUp(self):
        self.s=dict(config='cfg_c',tile=[256,128,64],cluster=[1,2],stages=4,k=4096,gpu_uuid='ref')
        self.r=dict(protocol='cfg-c-mma-reference-v1',iterations=1024,mma_shape=[64,128,16],warpgroups=2,chains_per_wg=2,per_chain_batch=4,b_major='MN',wait_group=1,cycle_samples=[1049379]*3,gpu_uuid='ref')
    def test_work_ratio_without_target_or_intercept(self):
        result=compute_reference(self.s,self.r)
        self.assertEqual(result['work_flop'],268435456)
        self.assertEqual(result['reference_cycles_fraction'],[1049379,16])
        self.assertFalse(result['prediction_qualified'])
    def test_device_and_organization_rejected(self):
        for key,value in [('gpu_uuid','other'),('config','cfg_b'),('stages',6),('k',True),('k',65),('cluster',[1,1])]:
            s=dict(self.s);s[key]=value
            with self.assertRaises(ValueError):compute_reference(s,self.r)
    def test_reference_must_match_instruction_contract(self):
        for key,value in [('b_major','K'),('per_chain_batch',16),('warpgroups',1),('cycle_samples',[1049379,True,1049379])]:
            r=dict(self.r);r[key]=value
            with self.assertRaises(ValueError):compute_reference(self.s,r)

if __name__=='__main__':unittest.main()

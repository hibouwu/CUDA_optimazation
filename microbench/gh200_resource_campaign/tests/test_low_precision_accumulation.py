import copy
import json
from pathlib import Path
import struct
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auditors.low_precision_accumulation import audit_values,compare_target_sass


class AccumulationValueTests(unittest.TestCase):
    def setUp(self):
        self.contract=json.loads((ROOT/'contracts/low_precision_accumulation_diagnostic_v1.json').read_text())
        self.device={'uuid':'test-device','sms':132}
        self.resource={'kernel_symbol':'lp_wgmma_e4m3_g1','registers_per_thread':148,'static_smem_bytes':5136,
            'dynamic_smem_bytes':0,'local_size_bytes':0,'occupancy_limit_ctas_per_sm':3,'extensions':{}}
        self.row={'schema_version':1,'type':'long_accumulation_diagnostic','profile_id':self.contract['profile_id'],
            'case_id':self.contract['case_id'],'target_symbol':self.contract['target_symbol'],'iterations':8192,'seed':3,
            'threads':128,'blocks':1,'target_launches':1,'explicit_auxiliary_launches':0,'mathematical_expected':16384,
            'resource_identity':copy.deepcopy(self.resource),'stamp':{'begin_ns':10,'end_ns':10,'begin_cycle':100,'end_cycle':100,'smid':143},
            'event_ms':0.1,'outputs':[{'index':i,'value':16384.0,'f32_bits':0x46800000} for i in range(8192)]}
        for field in ['warmup_executed','pilot_executed','performance_eligible','family_B3_eligible','numerical_kernel_qualification']:
            self.row[field]=False

    def audit(self,row):
        return audit_values(self.device,row,self.contract,self.device,self.resource)

    def test_exact_and_finite_difference_are_observations_not_qualification(self):
        self.row['event_ms']=0.0
        result=self.audit(self.row)
        self.assertEqual(result['status'],'collected_exact_mathematical_match')
        value=8192.0;bits=struct.unpack('<I',struct.pack('<f',value))[0]
        for item in self.row['outputs']:item.update(value=value,f32_bits=bits)
        result=self.audit(self.row)
        self.assertEqual(result['difference_count'],8192)
        self.assertEqual(result['status'],'collected_finite_difference')
        self.assertFalse(result['numerical_kernel_qualification'])
        self.assertFalse(result['performance_eligible'])
        self.assertFalse(result['historical_failure_phase_proven'])

    def test_output_mutations_rejected(self):
        mutations=[lambda r:r['outputs'].pop(),
                   lambda r:r['outputs'][1].update(index=0),
                   lambda r:r['outputs'][1].update(index=True),
                   lambda r:r['outputs'][0].update(value=float('nan')),
                   lambda r:r['outputs'][0].update(value=True),
                   lambda r:r['outputs'][0].update(f32_bits=0x7f800000),
                   lambda r:r['outputs'][0].update(value=8192.0),
                   lambda r:r['outputs'][0].update(extra=1),
                   lambda r:r['outputs'][0].update(value=0,f32_bits=0x80000000)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                row=copy.deepcopy(self.row);mutate(row)
                with self.assertRaises(ValueError):self.audit(row)

    def test_identity_completion_and_scope_mutations_rejected(self):
        mutations=[lambda r:r.update(target_launches=2),lambda r:r.update(iterations=2),
                   lambda r:r.update(warmup_executed=True),lambda r:r.update(family_B3_eligible=True),
                   lambda r:r.update(event_ms=-1),lambda r:r.update(extra=1),
                   lambda r:r['stamp'].update(smid=0xffffffff),
                   lambda r:r['stamp'].update(begin_ns=(1<<64)-1),
                   lambda r:r['stamp'].update(end_cycle=99),
                   lambda r:r['resource_identity'].update(local_size_bytes=False)]
        for mutate in mutations:
            row=copy.deepcopy(self.row);mutate(row)
            with self.assertRaises(ValueError):self.audit(row)

    def test_sass_complete_stream_not_token_presence(self):
        baseline='Function : _Z16lp_wgmma_e4m3_g1ijbPN2gh5StampEPd\n /*0000*/ QGMMA.F32 R0, R2;\n /*0010*/ EXIT;\n'
        self.assertEqual(compare_target_sass(baseline,baseline,'lp_wgmma_e4m3_g1')['instruction_count'],2)
        for altered in [baseline.replace('R2','R3'),baseline.replace('0010','0020'),baseline+baseline,baseline.replace('EXIT','NOP')]:
            with self.assertRaises(ValueError):compare_target_sass(altered,baseline,'lp_wgmma_e4m3_g1')


if __name__=='__main__':unittest.main()

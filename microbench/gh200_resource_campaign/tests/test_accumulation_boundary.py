"""Boundary diagnostics must preserve disagreement, not convert it into a pass."""
import copy
import json
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from auditors.low_precision_accumulation_boundary import audit_values, aggregate_profiles

ROOT=Path(__file__).resolve().parents[1]


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=json.loads((ROOT/'contracts/low_precision_accumulation_boundary_diagnostic_v1.draft.json').read_text())
        cls.device={'uuid':'fixture-not-GPU','runtime_version':12090}
        cls.resource={'registers_per_thread':148,'static_smem_bytes':5136,'local_size_bytes':0}

    def row(self,n,value):
        bits=struct.unpack('<I',struct.pack('<f',value))[0]
        return {'schema_version':1,'type':'accumulation_boundary_diagnostic',
          'profile_id':'original_uniform_'+str(n),'case_id':'wgmma_e4m3_g1_one_cta',
          'target_symbol':'lp_wgmma_e4m3_g1','iterations':n,'seed':3,'threads':128,'blocks':1,
          'target_launches':1,'explicit_auxiliary_launches':0,'mathematical_expected':2*n,
          **{k:False for k in ('warmup_executed','pilot_executed','performance_eligible',
                             'family_B3_eligible','numerical_kernel_qualification')},
          'resource_identity':self.resource,'stamp':{'begin_ns':0,'end_ns':0,'begin_cycle':0,'end_cycle':0,'smid':137},
          'event_ms':0,'outputs':[{'index':i,'value':value,'f32_bits':bits} for i in range(8192)]}

    def audit(self,row):
        return audit_values(self.device,row,self.contract,self.device,self.resource,row['profile_id'])

    def test_all_four_lengths_and_differences_retained(self):
        for n in (31,32,33,64):
            exact=self.audit(self.row(n,2.0*n));self.assertEqual(exact['difference_count'],0)
            observed=self.audit(self.row(n,64.0))
            self.assertEqual(observed['difference_count'],0 if n==32 else 8192)
            self.assertFalse(observed['numerical_kernel_qualification'])
            self.assertFalse(observed['performance_eligible'])

    def test_output_identity_and_qualification_rejected(self):
        for mutation in ('missing','duplicate','bits','nan','poison_stamp','extra_launch',
                         'warmup','performance','wrong_reference','length'):
            row=self.row(33,64.0)
            if mutation=='missing':row['outputs'].pop()
            elif mutation=='duplicate':row['outputs'][-1]['index']=0
            elif mutation=='bits':row['outputs'][-1]['f32_bits']=0
            elif mutation=='nan':row['outputs'][-1]['value']=float('nan')
            elif mutation=='poison_stamp':row['stamp']['smid']=2**32-1
            elif mutation=='extra_launch':row['target_launches']=2
            elif mutation=='warmup':row['warmup_executed']=True
            elif mutation=='performance':row['performance_eligible']=True
            elif mutation=='wrong_reference':row['mathematical_expected']=64
            else:row['iterations']=8192
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.audit(row)

    def test_adjacent_lengths_and_physical_chain_positions(self):
        rows={'original_uniform_'+str(n):self.row(n,62.0 if n==31 else 64.0) for n in (31,32,33,64)}
        result=aggregate_profiles(rows)
        self.assertEqual(result['position_count'],8192)
        self.assertEqual(result['positions'][0]['adjacent_differences'],[
            {'numerator':2,'denominator':1},{'numerator':0,'denominator':1},{'numerator':0,'denominator':1}])
        self.assertTrue(all(x['bitwise_mismatch_count']==0 for x in result['chain_comparisons'].values()))
        index=(7*2+1)*32+3
        rows['original_uniform_33']['outputs'][index].update(value=65.0,f32_bits=struct.unpack('<I',struct.pack('<f',65.0))[0])
        result=aggregate_profiles(rows)
        position=result['positions'][index]
        self.assertEqual([position[k] for k in ('thread','warp','lane','chain','fragment')],[7,0,7,1,3])
        self.assertEqual(position['adjacent_differences'][1:],
                         [{'numerator':1,'denominator':1},{'numerator':-1,'denominator':1}])
        self.assertEqual(result['chain_comparisons']['original_uniform_33']['bitwise_mismatch_count'],1)
        rows.pop('original_uniform_64')
        with self.assertRaises(ValueError):aggregate_profiles(rows)

    def test_no_arbitrary_coordinate_or_profile_substitution(self):
        row=self.row(8192,64.0)
        with self.assertRaises(ValueError):self.audit(row)
        contract=copy.deepcopy(self.contract);contract['profiles'][-1]['iterations']=128
        with self.assertRaises(ValueError):
            audit_values(self.device,self.row(33,64.0),contract,self.device,self.resource,'original_uniform_33')


if __name__=='__main__':unittest.main()

"""Reject double-counted transport and invalid formal launch/completion evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auditors.async_copy_formal import validate_contract, validate_trial, work

ROOT = Path(__file__).resolve().parents[1]


class AsyncCopyFormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((ROOT/'contracts/async_copy.json').read_text())
        cls.protocol = json.loads((ROOT/'contracts/protocol.json').read_text())
        cls.device = {'sms':132,'registers_per_sm':65536,'smem_per_sm_bytes':233472,
                      'smem_per_cta_optin_bytes':232448}

    def row(self, case):
        p = case['parameters']
        blocks = 1 if case['scope'] == 'one_cta' else 528
        # Independent explicit byte totals, rather than using the auditor work().
        payload = blocks * 128 * 65536 * p['request_bytes_per_thread']
        return {'schema_version':2,'type':'trial','case_id':case['id'],'iterations':8192,
          'seed':3,'threads':128,'blocks':blocks,'scope':case['scope'],'errors':0,
          'correctness':{'method':'modular_checksum_and_all_final_slots',
            'checked_elements':blocks*128*(1+p['stages']*(p['request_bytes_per_thread']//4)),
            'input_conditions':'uint32(17*word_index+seed); neighbor=(thread+1)%128; poison reset each launch'},
          'work_unit':'byte','work_count':payload,'read_payload_bytes':payload,
          'write_payload_bytes':payload,'consumer_read_bytes':payload,
          'start_ns':1000,'stop_ns':1001000,'event_ms':1.1,
          'blocks_detail':[{'block_id':b,'smid':b%132,'start_ns':1000,'stop_ns':1001000,
                            'start_cycle':10000+b*1000000,'stop_cycle':2010000+b*1000000} for b in range(blocks)],
          'warmup_samples_ns':[1000000]*8,'warmup_converged':True,
          'cache_residency_proven':False,'physical_hbm_bytes_proven':False,
          'registers_per_thread':32,'static_smem_bytes':0,'dynamic_smem_bytes':p['shared_payload_bytes'],
          'local_size_bytes':0,'occupancy_limit_ctas_per_sm':4,
          'request_bytes_per_thread':p['request_bytes_per_thread'],'stages':p['stages'],
          'global_array_bytes':8388608}

    def test_all_coordinates_and_64bit_totals(self):
        validate_contract(self.contract)
        for case in self.contract['cases']:
            with self.subTest(case=case['id']):
                row = self.row(case)
                result = validate_trial(row, case, self.device, 3, self.protocol)
                denominator = 2000000 if case['scope']=='one_cta' else 1000000
                self.assertEqual(result['value'], row['work_count']/denominator)
        self.assertGreater(work(self.contract['cases'][-1],528)['work'],2**32)

    def test_matrix_and_units_reject(self):
        for mutation in ('duplicate','missing','unit','coverage','numerator','clock'):
            contract = copy.deepcopy(self.contract)
            c = contract['cases'][0]
            if mutation=='duplicate':contract['cases'][-1]=copy.deepcopy(c)
            elif mutation=='missing':contract['cases'].pop()
            elif mutation=='unit':c['work_unit']='FLOP'
            elif mutation=='coverage':c['coverage_policy']='observed_subset'
            elif mutation=='numerator':c['metric']['numerator']='payload_bytes'
            else:c['metric']['denominator']='elapsed_ns'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):validate_contract(contract)

    def test_raw_tampering_reject(self):
        for case in (self.contract['cases'][0], self.contract['cases'][-1]):
            for field in ('work_count','read_payload_bytes','write_payload_bytes','consumer_read_bytes',
                          'blocks','dynamic_smem_bytes','request_bytes_per_thread','stages','global_array_bytes'):
                row=self.row(case);row[field]*=2
                with self.subTest(case=case['id'],field=field),self.assertRaises(ValueError):
                    validate_trial(row,case,self.device,3,self.protocol)
            for mutation in ('missing_CTA','error','missing_output','early_stop','bad_registers','local_memory','static_memory','resident_threads','cache_claim'):
                row=self.row(case)
                if mutation=='missing_CTA':row['blocks_detail'].pop()
                elif mutation=='error':row['errors']=1
                elif mutation=='missing_output':row['correctness']['checked_elements']-=1
                elif mutation=='early_stop':row['blocks_detail'][0]['stop_cycle']=row['blocks_detail'][0]['start_cycle']
                elif mutation=='bad_registers':row['registers_per_thread']=256
                elif mutation=='local_memory':row['local_size_bytes']=4096
                elif mutation=='static_memory':row['static_smem_bytes']=48
                elif mutation=='resident_threads':
                    row['occupancy_limit_ctas_per_sm']=32;row['registers_per_thread']=8
                else:row['cache_residency_proven']=True
                with self.subTest(case=case['id'],mutation=mutation),self.assertRaises(ValueError):
                    validate_trial(row,case,self.device,3,self.protocol)


if __name__=='__main__':unittest.main()

"""Offline regression checks for the fixed Dense compiler evidence, not GPU tests."""
import hashlib
import json
from pathlib import Path
import unittest

ATLAS = Path(__file__).resolve().parents[1]
MODULE = ATLAS/'data/modules/dense_fp16'

class DenseGemmDispatchEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((MODULE/'gemm_dispatch.json').read_text())
        cls.index=json.loads((MODULE/'gemm_dispatch_debug_index.json').read_text())
        cls.compile=cls.data['compile_evidence'][0]
        cls.metadata=json.loads((ATLAS/cls.compile['path']).read_text())
        cls.edges={e['id']:e for e in cls.data['edges']}

    def test_compiler_artifacts_are_immutable_and_successful(self):
        self.assertEqual(self.metadata['status'],'static_dispatch_probe_pass')
        self.assertEqual(self.metadata['all_snapshot_file_hashes_verified'],824)
        self.assertFalse(self.metadata['gpu_executed'])
        self.assertEqual(len(self.metadata['commands']),5)
        self.assertTrue(all(c['returncode']==0 for c in self.metadata['commands']))
        run=(ATLAS/self.compile['path']).parent
        for name,record in self.metadata['artifacts'].items():
            self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(),record['sha256'],name)

    def test_dwarf_proves_forwarding_overload_not_direct_rank4_to_rank1(self):
        actual={(e['source']['line'],e['callsite']['line'],e['target']['line']) for e in self.index['entries']}
        self.assertEqual(actual,{(83,88,275),(275,298,142),(142,148,190),(190,197,94),(94,104,2072)})
        calls={(e['source'],e['target']) for e in self.edges.values() if e['relation']=='calls'}
        self.assertNotIn(('dispatch.api.rank4','contract.api.gemm_rank1'),calls)
        self.assertIn(('dispatch.api.rank4','dispatch.api.five_rvalue'),calls)

    def test_evidence_quotes_match_physical_lines(self):
        for edge in self.edges.values():
            for ev in edge['evidence']:
                path=ATLAS/'snapshot'/ev['path'] if ev['path'].startswith('include/') else ATLAS/ev['path']
                actual='\n'.join(path.read_text().splitlines()[ev['start_line']-1:ev['end_line']])
                self.assertEqual(ev['quote'],actual,(edge['id'],ev['path'],ev['start_line']))

    def test_branch_condition_uses_descriptor_bytes_and_correct_row_line(self):
        e=self.edges['dispatch.edge.rank4_to_rvalue']
        self.assertEqual(e['callsite']['line'],298)
        self.assertTrue(e['compile_time_selection']['value'])
        self.assertEqual(e['compile_time_selection']['preprocessor']['line'],291)
        values=dict(x.split('=',1) for x in (ATLAS/self.compile['type_log']).read_text().splitlines())
        for tensor in ['ASlice','BSlice']:
            self.assertEqual(values[tensor+'.rank'],'2')
            self.assertEqual(values[tensor+'.V_elements'],'1')
            self.assertEqual(values[tensor+'.value_bytes'],'8')
        self.assertEqual(values['Accumulator.rank'],'3')
        self.assertEqual(values['Accumulator.is_tmem'],'1')
        self.assertEqual(values['Accumulator.is_rmem'],'1')
        self.assertEqual(values['Dispatch4.M'],'1')
        self.assertEqual(values['Dispatch4.N'],'1')

    def test_each_new_api_is_one_exact_original_overload(self):
        api=[n for n in self.data['nodes'] if n['kind']=='api']
        self.assertEqual({(n['path'],n['line']) for n in api},{('include/cute/algorithm/gemm.hpp',275),('include/cute/algorithm/gemm.hpp',142)})
        self.assertEqual(len(api),2)
        self.assertTrue(all(n['signature'].startswith('template ') for n in api))

    def test_no_source_binding_edge_pretends_to_be_a_runtime_call(self):
        for key,e in self.edges.items():
            if '.bind_' in key or '.slice_' in key:
                self.assertIn(e['relation'],['template_binds','type_uses'])
        self.assertEqual(len([e for e in self.edges.values() if e['relation']=='calls']),3)

if __name__=='__main__':
    unittest.main()

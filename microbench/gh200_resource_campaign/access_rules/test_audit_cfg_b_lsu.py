"""CPU regressions for the fixed cfg_b LSU build audit."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_cfg_b_lsu as audit
import cfg_b_composition as aggregate
import cfg_b_program


ROOT = aggregate.DEFAULT_ARCHIVE


class CfgBLsuBuildAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = audit.audit(ROOT)

    def test_actual_counts_and_widths(self):
        for name, expected in audit.EXPECTED_COUNTS.items():
            build = self.facts['builds'][name]
            self.assertEqual({op: build['counts'][op] for op in expected}, expected)
            self.assertEqual(build['counts']['LDG.E'], 2)
            for op in ('STG', 'LDL', 'STL', 'LDS', 'LDSM', 'STSM'):
                self.assertEqual(build['counts'][op], 0)
            self.assertEqual(sum(v['width_bits'] == 32 for v in build['constant_instructions']), 57)
            self.assertEqual(sum(v['width_bits'] == 64 for v in build['constant_instructions']), 22)
            self.assertEqual(build['static_resources']['registers_per_thread'], 168)
            program_demand = cfg_b_program.ordinary_lsu_demands({'stages': build['stages']})
            self.assertEqual(program_demand['constants']['static_opcodes'],
                             {op: build['counts'][op] for op in expected})
            global_demand = program_demand['ordinary_global']
            dynamic = self.facts['dynamic_ordinary_global_demand']
            self.assertEqual(global_demand['static_scalar_ldg_sites'], build['counts']['LDG.E'])
            self.assertEqual(global_demand['active_loads'], dynamic['read_instructions'])
            self.assertEqual(global_demand['active_stores'], dynamic['write_instructions'])
            self.assertEqual(global_demand['active_payload_bytes'], dynamic['read_bytes'] + dynamic['write_bytes'])
            self.assertEqual(program_demand['local']['static_ldl'], build['counts']['LDL'])
            self.assertEqual(program_demand['local']['static_stl'], build['counts']['STL'])
            self.assertEqual(program_demand['local']['stack_bytes'], build['static_resources']['stack_bytes'])
            self.assertEqual(program_demand['local']['spill_load_bytes'], build['static_resources']['spill_load_bytes'])
            self.assertEqual(program_demand['local']['spill_store_bytes'], build['static_resources']['spill_store_bytes'])

    def test_zero_demand_requires_source_and_both_branch_fragments(self):
        source = self.facts['source_facts']
        self.assertEqual((source['alpha'], source['beta'], source['alpha_ptr'],
                          source['alpha_batch_stride']), (1.0, 0.0, None, 0))
        demand = self.facts['dynamic_ordinary_global_demand']
        self.assertEqual(demand['read_bytes'], 0)
        self.assertEqual(demand['qualification'], 'source_and_sass_branch_derivation_not_counters')
        expected = {'cfg_b_plain': [('0x0840', '0x860'), ('0x1e40', '0x1e70')],
                    'cfg_b_s4_plain': [('0x0800', '0x820'), ('0x1e30', '0x1e60')]}
        for name, pairs in expected.items():
            branches = self.facts['builds'][name]['alpha_load_branches']
            self.assertEqual([(v['load']['pc'], v['branch_target']) for v in branches], pairs)
            self.assertTrue(all(not v['active'] and v['checked_fragment'] for v in branches))

    def test_target_function_excludes_following_helper_global_stores(self):
        # Add a helper instruction after the actual main function. It must not
        # alter the target count, even when it uses a previously unseen opcode.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'build').mkdir()
            source = (ROOT / 'build/cfg_b_plain.sass').read_text()
            first = source.split('Function :', 2)
            text = first[0] + 'Function :' + first[1]
            text += '\nFunction : fake_helper\n/*0000*/ STG.E.128 [R2], R4;\n'
            (root / 'build/cfg_b_plain.sass').write_text(text)
            _, instructions = audit.read_kernel(root, 'cfg_b_plain')
            self.assertFalse(any(v['opcode'].startswith('STG') for v in instructions))

    def test_changed_skip_branch_is_rejected(self):
        _, instructions = audit.read_kernel(ROOT, 'cfg_b_plain')
        changed = [dict(v, instruction='@P0 BRA 0x860') if v['pc'] == '0x0820' else v
                   for v in instructions]
        with self.assertRaisesRegex(ValueError, 'changed branch fragment at 0x820'):
            audit.alpha_branches('cfg_b_plain', changed)

    def test_changed_sass_identity_is_rejected(self):
        original_sha = audit.sha

        def changed_sha(path):
            return '0' * 64 if path.name == 'cfg_b_s4_plain.sass' else original_sha(path)

        with patch.object(audit, 'sha', side_effect=changed_sha):
            with self.assertRaisesRegex(ValueError, 'changed frozen artifact: build/cfg_b_s4_plain.sass'):
                audit.identity(ROOT)


if __name__ == '__main__':
    unittest.main()

"""Fixed S4/S6 capacity regressions; CPU only, no placement/timing model."""
import copy
import json
import unittest

import audit_cfg_b_lsu as build_audit
import cfg_b_composition as aggregate
import cfg_b_program
import sm_residency as residency


class CfgBCapacityTests(unittest.TestCase):
    def program(self, stages, **setup):
        return dict(stages=stages, tile=[128, 128, 64], setup=setup)

    def test_two_independent_capacity_limits(self):
        for stages, smem in ((4, 150528), (6, 216064)):
            program = self.program(stages)
            original = copy.deepcopy(program)
            device = copy.deepcopy(residency.GH200_DEVICE_CONDITIONS)
            report = residency.cfg_b_capacity(program, device=device)
            self.assertEqual(report['allocations']['smem_per_cta_bytes'], smem)
            self.assertEqual(report['allocations']['registers_per_warp'], 5376)
            self.assertEqual(report['allocations']['registers_per_cta'], 64512)
            self.assertEqual(report['allocations']['launch_check_registers_per_cta'], 64512)
            self.assertEqual(report['allocations']['warps_per_subpartition'], 3)
            self.assertEqual(report['capacity_limits'], dict(registers=1, smem=1, threads=5, blocks=32))
            self.assertEqual(report['limiting_resources'], ['registers', 'smem'])
            self.assertEqual(report['capacity_ctas_per_sm'], 1)
            self.assertIsNone(report['historical_api']['ctas_per_sm'])
            self.assertIn('frozen_build_requirement', report['demands']['smem_origin'])
            self.assertEqual(program, original)
            self.assertEqual(device, residency.GH200_DEVICE_CONDITIONS)

    def test_setup_complete_smem_has_priority_and_epilogue_is_not_added(self):
        report = residency.cfg_b_capacity(self.program(4, smem=149504, smem_epilogue=18432,
            threads=384, max_active_ctas_per_sm=1), device=residency.GH200_DEVICE_CONDITIONS)
        self.assertEqual(report['demands']['dynamic_smem_bytes'], 149504)
        self.assertEqual(report['allocations']['smem_per_cta_bytes'], 149504 + 1024)
        self.assertFalse(report['demands']['smem_epilogue_counted_again'])
        self.assertTrue(report['historical_api']['matches_calculation'])
        changed = residency.cfg_b_capacity(self.program(4, smem=150528, smem_epilogue=999999),
            device=residency.GH200_DEVICE_CONDITIONS)
        self.assertEqual(changed['demands']['dynamic_smem_bytes'], 150528)
        self.assertEqual(changed['allocations']['smem_per_cta_bytes'], 151552)

    def test_actual_build_setup_and_register_demand_match(self):
        root = aggregate.DEFAULT_ARCHIVE
        facts = build_audit.audit(root)
        setups = {value['case']: value['setup'] for value in json.loads((root / 'static_setup.json').read_text())}
        for case, build in (('v09d_cfg_b_G3_base', 'cfg_b_plain'), ('v09d_cfg_b_G3_stage4', 'cfg_b_s4_plain')):
            setup = setups[case]
            program = dict(stages=setup['stages'], tile=[128, 128, 64], setup=setup)
            report = residency.cfg_b_capacity(program, device=residency.GH200_DEVICE_CONDITIONS)
            registers = cfg_b_program.register_demands(program)
            self.assertEqual(report['demands']['threads'], setup['threads'])
            self.assertEqual(report['demands']['dynamic_smem_bytes'], setup['smem'])
            self.assertEqual(report['demands']['registers_per_thread'], facts['builds'][build]['static_resources']['registers_per_thread'])
            self.assertEqual(report['demands']['cta_initial_pool_registers'], registers['cta_reserved_registers'])
            self.assertEqual(report['capacity_ctas_per_sm'], setup['max_active_ctas_per_sm'])
            self.assertTrue(report['historical_api']['matches_calculation'])


if __name__ == '__main__':
    unittest.main()

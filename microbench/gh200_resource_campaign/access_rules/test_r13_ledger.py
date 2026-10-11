"""Source ownership, multicast conservation and persistent generation checks."""
import unittest

from supply_model import input_slot_state, supply_work_ledger


class SupplyLedgerTests(unittest.TestCase):
    def ledger(self, config, m, n, k=128, stages=6):
        cluster = dict(cfg_a=[2, 1], cfg_b=[1, 1], cfg_c=[1, 2])[config]
        tile = dict(cfg_a=[128, 128, 64], cfg_b=[128, 128, 64], cfg_c=[256, 128, 64])[config]
        row = dict(config=config, m=m, n=n, k=k, swizzle=1)
        setup = dict(row, grid=cluster+[1], cluster=cluster, tile=tile, stages=stages)
        return supply_work_ledger(row, setup)

    def test_cooperative_slices_conserve_each_destination(self):
        # cfg_a: each A is local; both B K-halves reach each rank.
        a = self.ledger('cfg_a', 256, 128)
        self.assertEqual([len(o['operands']['B']['source_atoms']) for o in a['outputs']], [4, 4])
        self.assertEqual(a['totals']['source_payload_KiB'], 96)
        self.assertEqual(a['totals']['destination_valid_KiB'], 128)
        # cfg_c: two different A M-halves reach both destinations.
        c = self.ledger('cfg_c', 192, 256)
        self.assertEqual([x['valid_KiB'] for x in c['source_atoms'] if x['operand']=='A'], [16, 8])
        self.assertEqual([o['operands']['A']['valid_KiB'] for o in c['outputs']], [24, 24])
        self.assertEqual([o['operands']['A']['fill_KiB'] for o in c['outputs']], [8, 8])
        self.assertEqual([len(o['operands']['A']['source_atoms']) for o in c['outputs']], [2, 2])
        self.assertEqual(c['totals']['source_valid_KiB'], 112)
        self.assertEqual(c['totals']['destination_valid_KiB'], 160)
        self.assertEqual(c['totals']['destination_fill_KiB'], 32)

    def test_pingpong_counts_reads_deliveries_and_D_separately(self):
        b = self.ledger('cfg_b', 128, 256)
        self.assertEqual(b['totals']['source_requests'], 12)
        self.assertEqual(b['totals']['destination_atoms'], 12)
        self.assertEqual(b['totals']['source_valid_KiB'], 128)
        self.assertEqual(b['totals']['distinct_valid_source_KiB'], 96)
        self.assertEqual(b['totals']['output_valid_KiB'], 128)
        self.assertEqual(b['totals']['output_nominal_subtiles'], 16)

    def test_stage6_second_output_does_not_reset_ring_or_phase(self):
        b = self.ledger('cfg_b', 128, 256, k=4096, stages=6)
        state = b['outputs'][1]['first_input_state']
        self.assertEqual(state, dict(group=64, slot=4, generation=10,
            full_wait_phase=0, empty_wait_phase=1, previous_group=58))
        self.assertEqual(input_slot_state(6, 6)['full_wait_phase'], 1)
        self.assertEqual(input_slot_state(6, 6)['empty_wait_phase'], 0)

    def test_descriptor_mismatch_and_rotation_are_rejected(self):
        row = dict(config='cfg_b', m=128, n=128, k=128, swizzle=1, ldb=128)
        with self.assertRaisesRegex(ValueError, 'ldb'):
            supply_work_ledger(row, dict(ldb=136))
        with self.assertRaisesRegex(ValueError, 'A rotation'):
            supply_work_ledger(dict(a_row_rotation=1), {})

    def test_observed_timing_cannot_change_static_work(self):
        row = dict(config='cfg_b', m=128, n=128, k=128, swizzle=1)
        setup = dict(row, grid=[1, 1, 1], stages=6)
        baseline = supply_work_ledger(row, setup)
        poisoned = dict(row, L_cycle=-1e30, L_ns=1e30, measured_frequency=0.001,
            elapsed_us=0., first_mma=99999, MAIN_END=-99999)
        self.assertEqual(supply_work_ledger(poisoned, setup), baseline)
        self.assertEqual(baseline['schema'], 'gh200-supply-work-ledger/1')
        with self.assertRaisesRegex(ValueError, 'allocation'):
            supply_work_ledger(dict(row, source_copies=2), setup)
        with self.assertRaisesRegex(ValueError, 'exchange'):
            supply_work_ledger(dict(row, work_swap55_91=1), setup)


if __name__ == '__main__':
    unittest.main()

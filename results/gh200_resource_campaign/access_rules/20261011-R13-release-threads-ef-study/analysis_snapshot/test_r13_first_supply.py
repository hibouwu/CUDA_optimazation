"""First-supply event controls preserve the producer-delay experiment."""
import json
from pathlib import Path
import unittest

import v08_model
from analyze_r13_supply import _first_supply_intervals


class FirstSupplyConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configs = Path(__file__).parent / "configs"
        cls.original = json.loads((configs / "r10-producer-delay.json").read_text())
        cls.rows = json.loads((configs / "r13-first-supply.json").read_text())

    def test_plain_references_unchanged_and_first(self):
        self.assertEqual(self.rows[:2], self.original[:2])
        self.assertTrue(all(row["set"] == "reference" and row["variants"] == ["plain"]
                            for row in self.rows[:2]))
        self.assertTrue(all(row["set"] == "ctrl" for row in self.rows[2:]))

    def test_eight_factorial_conditions_and_140_processes(self):
        controls = self.rows[2:]
        self.assertEqual(len(controls), 8)
        self.assertEqual(len({row["id"] for row in self.rows}), 10)
        self.assertEqual(
            {(row["first_supply_events"], row["producer_delay_ns"], row["ldb"])
             for row in controls},
            {(events, delay, pitch) for events in (0, 1)
             for delay in (0, 512) for pitch in (3072, 3088)},
        )
        for row in controls:
            events = row["first_supply_events"]
            delay = row["producer_delay_ns"]
            pitch = 2 * (row["ldb"] - 3072)
            self.assertEqual(row["id"], f"r13_first_e{events}_d{delay}_p{pitch}")
            self.assertEqual(row["group"], f"r13_first_e{events}_d{delay}")
            self.assertEqual(row["original_experiment"], "R13")
            self.assertEqual(row["variants"], ["plain", "dual"] if events == 0 else ["dual"])
        self.assertEqual(sum(len(row["variants"]) * 10 for row in self.rows), 140)

    def test_old_controls_change_only_event_identity_and_variants(self):
        original = {(row["producer_delay_ns"], row["ldb"]): row
                    for row in self.original[2:]}
        ignored = {"id", "group", "original_experiment", "first_supply_events", "variants"}
        for row in self.rows[2:]:
            baseline = original[row["producer_delay_ns"], row["ldb"]]
            self.assertEqual({key: value for key, value in row.items() if key not in ignored},
                             {key: value for key, value in baseline.items() if key not in ignored})

    def test_capacity_and_all_132_CTA_worksets_are_unchanged(self):
        expected = v08_model.scheduled_work("cfg_b", 2304, 3072, [132, 1, 1], 1, "along_m")
        self.assertEqual(len(expected), 132)
        self.assertEqual({coord for cta in expected for coord in cta},
                         {(mi, ni) for mi in range(18) for ni in range(24)})
        self.assertEqual(sorted(map(len, expected)), [3] * 96 + [4] * 36)
        self.assertEqual(expected[55][0], (1, 3))
        self.assertEqual(expected[91][0], (1, 5))
        for row in self.rows[2:]:
            self.assertEqual((row["m"], row["n"], row["k"]), (2304, 3072, 1024))
            self.assertEqual((row["storage_m"], row["storage_n"]), (2304, 3072))
            self.assertEqual((row["input_map_m"], row["input_map_n"]), (2304, 3072))
            self.assertEqual((row["alloc_lda"], row["alloc_ldb"]), (1024, 3136))
            self.assertEqual((row["b_offset_bytes"], row["b_allocation_extra_bytes"]), (0, 256))
            self.assertEqual(row["work_swap55_91"], 0)
            actual = v08_model.scheduled_work(row["config"], row["m"], row["n"],
                [row["sm_count"], 1, 1], row["swizzle"], row["raster_order"])
            self.assertEqual(actual, expected)


class FirstSupplyEndpointTests(unittest.TestCase):
    def sample(self):
        cta=dict(entry_c=100,entry_ns=1000,prod_c=300,prod_ns=1100,
            tiles=[[1300,3300,3320,3500]],tiles_ns=[[1600,2600,2610,2700]])
        events=dict(cta=55,enabled=True,protocol='r13-first-supply-v1',
            issue_end_c=600,issue_end_ns=1250,wait_begin_c=500,wait_begin_ns=1200)
        return cta,events

    def test_role_partitions_close_without_requiring_wait_after_issue(self):
        cta,events=self.sample();result=_first_supply_intervals(cta,events)
        self.assertEqual(result['wait_minus_issue_ns'],-50)
        self.assertEqual(result['wait_to_FIRST_ns'],400)
        self.assertEqual(result['prod_to_issue_ns']+result['issue_to_FIRST_ns'],result['S_ns'])
        for unit in ('cycles','ns'):
            self.assertEqual(result['wait_minus_issue_'+unit]+result['wait_to_FIRST_'+unit],result['issue_to_FIRST_'+unit])
            self.assertEqual(result['P0_'+unit]+result['S_plus_L_'+unit],result['entry_to_MAIN_END_'+unit])

    def test_late_producer_marker_is_retained_as_a_signed_observation(self):
        cta,events=self.sample();events.update(issue_end_c=1400,issue_end_ns=1650)
        result=_first_supply_intervals(cta,events)
        self.assertEqual(result['issue_to_FIRST_cycles'],-100)
        self.assertEqual(result['issue_to_FIRST_ns'],-50)
        self.assertEqual(result['prod_to_issue_ns']+result['issue_to_FIRST_ns'],500)

    def test_disabled_events_retain_old_boundaries_without_synthetic_local_times(self):
        cta,_=self.sample();result=_first_supply_intervals(cta,dict(enabled=False))
        self.assertEqual(result['S_ns'],500)
        self.assertEqual(result['L_cycles'],2000)
        self.assertNotIn('issue_to_FIRST_ns',result)
        self.assertNotIn('wait_to_FIRST_cycles',result)

    def test_role_local_inversions_are_rejected(self):
        cta,events=self.sample()
        for changes in (dict(issue_end_c=299),dict(wait_begin_ns=1601),dict(wait_begin_c=99)):
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'role-local'):
                _first_supply_intervals(cta,dict(events,**changes))

    def test_public_observer_identity_and_equal_scratch_are_required(self):
        from test_r09_history_series import synthetic
        from analyze_r18 import validate_warmup
        for enabled in (0,1):
            row,setup,call,_=synthetic();row['first_supply_events']=enabled
            for event in (setup,call):
                event.update(first_supply_events=enabled,first_supply_protocol='r13-first-supply-v1',first_supply_scratch_bytes=64)
            validate_warmup(setup,call,row)
            for changes in (dict(first_supply_events=1-enabled),dict(first_supply_scratch_bytes=32),dict(first_supply_protocol='other')):
                with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'first supply'):
                    validate_warmup(setup,dict(call,**changes),row)


if __name__ == "__main__":
    unittest.main()

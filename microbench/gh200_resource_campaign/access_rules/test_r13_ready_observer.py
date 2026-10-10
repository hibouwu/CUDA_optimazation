"""Ready-observer controls preserve the preceding first-supply conditions."""
import json
from pathlib import Path
import unittest

import v08_model
from analyze_r13_supply import _ready_window_bounds, _ready_bound_difference, _ready_bound_summary, _ready_call_bounds, _startup_shift_check


class ReadyObserverConfigTests(unittest.TestCase):
    def test_current_ready_config_binds_only_controls_to_first_release(self):
        self.assertTrue(all(row["set"] == "reference" and "ready_generation_bound" not in row
                            for row in self.rows[:2]))
        self.assertEqual(len(self.rows[2:]), 8)
        self.assertTrue(all(row["set"] == "ctrl" and row["ready_generation_bound"] == "first_release"
                            for row in self.rows[2:]))

    @classmethod
    def setUpClass(cls):
        configs = Path(__file__).parent / "configs"
        cls.previous = json.loads((configs / "r13-first-supply.json").read_text())
        cls.rows = json.loads((configs / "r13-ready-observer.json").read_text())

    def test_original_plain_references_are_first(self):
        references = [row for row in self.previous if row["set"] == "reference"]
        self.assertEqual(len(references), 2)
        self.assertEqual(self.rows[:2], references)
        self.assertTrue(all(row["variants"] == ["plain"] for row in self.rows[:2]))
        self.assertTrue(all(row["set"] == "ctrl" for row in self.rows[2:]))

    def test_complete_factorial_and_140_processes(self):
        controls = self.rows[2:]
        self.assertEqual(len(controls), 8)
        self.assertEqual(len({row["id"] for row in self.rows}), 10)
        self.assertEqual(
            {(row["ready_observer"], row["producer_delay_ns"], row["ldb"])
             for row in controls},
            {(observer, delay, pitch) for observer in (0, 1)
             for delay in (0, 512) for pitch in (3072, 3088)},
        )
        for row in controls:
            observer = row["ready_observer"]
            delay = row["producer_delay_ns"]
            pitch = 2 * (row["ldb"] - 3072)
            self.assertEqual(row["id"], f"r13_ready_o{observer}_d{delay}_p{pitch}")
            self.assertEqual(row["group"], f"r13_ready_o{observer}_d{delay}")
            self.assertEqual(row["original_experiment"], "R13")
            self.assertEqual(row["variants"], ["plain", "dual"] if observer == 0 else ["dual"])
        self.assertEqual(sum(len(row["variants"]) * 10 for row in self.rows), 140)

    def test_previous_events1_controls_change_only_observer_and_identity(self):
        previous = {(row["producer_delay_ns"], row["ldb"]): row
                    for row in self.previous
                    if row["set"] == "ctrl" and row["first_supply_events"] == 1}
        self.assertEqual(len(previous), 4)
        ignored = {"id", "group", "variants", "ready_observer", "ready_generation_bound"}
        for row in self.rows[2:]:
            baseline = previous[row["producer_delay_ns"], row["ldb"]]
            self.assertEqual({key: value for key, value in row.items() if key not in ignored},
                             {key: value for key, value in baseline.items() if key not in ignored})

    def test_first_supply_capacity_and_132_CTA_worksets_stay_fixed(self):
        expected = v08_model.scheduled_work("cfg_b", 2304, 3072, [132, 1, 1], 1, "along_m")
        self.assertEqual(len(expected), 132)
        self.assertEqual({coord for cta in expected for coord in cta},
                         {(mi, ni) for mi in range(18) for ni in range(24)})
        self.assertEqual(sorted(map(len, expected)), [3] * 96 + [4] * 36)
        self.assertEqual(expected[55][0], (1, 3))
        self.assertEqual(expected[91][0], (1, 5))
        for row in self.rows[2:]:
            self.assertEqual(row["first_supply_events"], 1)
            self.assertEqual(row["work_swap55_91"], 0)
            self.assertEqual((row["m"], row["n"], row["k"]), (2304, 3072, 1024))
            self.assertEqual((row["storage_m"], row["storage_n"]), (2304, 3072))
            self.assertEqual((row["input_map_m"], row["input_map_n"]), (2304, 3072))
            self.assertEqual((row["alloc_lda"], row["alloc_ldb"]), (1024, 3136))
            self.assertEqual((row["b_offset_bytes"], row["b_allocation_extra_bytes"]), (0, 256))
            actual = v08_model.scheduled_work(row["config"], row["m"], row["n"],
                [row["sm_count"], 1, 1], row["swizzle"], row["raster_order"])
            self.assertEqual(actual, expected)


class ReadyIntervalTests(unittest.TestCase):
    def test_startup_shift_is_a_necessary_bound_with_explicit_violations(self):
        for delta_l in (-100,-40,0):
            self.assertTrue(_startup_shift_check(100,delta_l)['satisfies'])
        low=_startup_shift_check(100,-101)
        self.assertFalse(low['satisfies']);self.assertEqual(low['lower_violation'],1)
        high=_startup_shift_check(100,12)
        self.assertFalse(high['satisfies']);self.assertEqual(high['upper_violation'],12)
        self.assertFalse(_startup_shift_check(-1,0)['premise_delta_S_nonnegative'])

    def test_ready_partitions_keep_complementary_bounds(self):
        result=_ready_window_bounds(150,170,100,220)
        self.assertEqual(result['issue_to_ready'],dict(lower=50,upper=70))
        self.assertEqual(result['ready_to_FIRST'],dict(lower=50,upper=70))
        self.assertEqual(result['width'],20)
        self.assertEqual(result['issue_to_ready']['lower']+result['ready_to_FIRST']['upper'],120)
        self.assertEqual(result['issue_to_ready']['upper']+result['ready_to_FIRST']['lower'],120)

    def test_first_poll_success_has_only_upper_not_a_synthetic_zero_lower(self):
        result=_ready_window_bounds(None,90,100,220)
        self.assertEqual(result['issue_to_ready'],dict(lower=None,upper=-10))
        self.assertEqual(result['ready_to_FIRST'],dict(lower=130,upper=None))
        self.assertIsNone(result['width'])

    def test_condition_difference_propagates_bounds_not_midpoints(self):
        delta=_ready_bound_difference(dict(lower=10,upper=30),dict(lower=40,upper=80))
        self.assertEqual(delta,dict(lower=10,upper=70))
        unknown=_ready_bound_difference(dict(lower=None,upper=30),dict(lower=40,upper=80))
        self.assertEqual(unknown,dict(lower=10,upper=None))

    def test_censored_repetitions_remain_in_median_bounds(self):
        finite=[dict(lower=10,upper=20) for _ in range(5)]
        censored=[dict(lower=None,upper=None) for _ in range(5)]
        result=_ready_bound_summary(finite+censored)
        self.assertEqual(result['processes'],10)
        self.assertEqual(result['both_finite'],5)
        self.assertIsNone(result['median_lower'])
        self.assertIsNone(result['median_upper'])
        result=_ready_bound_summary(finite+finite[:4]+censored[:1])
        self.assertEqual((result['median_lower'],result['median_upper']),(10,20))
        self.assertIsNone(result['process_lower'])
        self.assertIsNone(result['process_upper'])

    def test_first_generation_uses_cycles_and_preserves_ns_quantization_sign(self):
        cta=dict(sm=55,tiles=[[300,600]],tiles_ns=[[1000,1200]])
        events=dict(issue_end_c=100,issue_end_ns=900,wait_begin_c=50,wait_begin_ns=800)
        observer=dict(success=True,exit_c=290,exit_ns=1024,last_false_pre_c=270,last_false_pre_ns=992,
            tests=2,sm=55,first_generation=True,status='interval')
        result=_ready_call_bounds(cta,events,observer)
        self.assertEqual(result['bounds']['ready_to_FIRST_cycles'],dict(lower=10,upper=30))
        self.assertEqual(result['bounds']['ready_to_FIRST_ns'],dict(lower=-24,upper=8))
        self.assertEqual(result['width'],dict(cycles=20,ns=32))

    def test_after_first_and_poll_limit_are_not_assigned_ready_upper_bounds(self):
        cta=dict(sm=55,tiles=[[300,600]],tiles_ns=[[1000,1200]])
        events=dict(issue_end_c=100,issue_end_ns=900,wait_begin_c=50,wait_begin_ns=800)
        for success,status,exit_c in ((True,'after_first',301),(False,'poll_limit',290)):
            observer=dict(success=success,exit_c=exit_c,exit_ns=1024,last_false_pre_c=270,last_false_pre_ns=992,
                tests=4096,sm=55,first_generation=False,status=status)
            result=_ready_call_bounds(cta,events,observer)
            self.assertTrue(all(b==dict(lower=None,upper=None) for b in result['bounds'].values()))
            self.assertEqual(result['status'],status)

    def test_new_release_boundary_keeps_signed_FIRST_interval_without_upgrading_old_protocol(self):
        cta=dict(sm=55,tiles=[[300,900]],tiles_ns=[[1000,1500]])
        events=dict(issue_end_c=100,issue_end_ns=900,wait_begin_c=50,wait_begin_ns=800)
        old=dict(success=True,exit_c=340,exit_ns=1056,last_false_pre_c=270,last_false_pre_ns=992,
            tests=12,sm=55,first_generation=False,status='after_first')
        self.assertEqual(_ready_call_bounds(cta,events,old)['bounds']['ready_to_FIRST_cycles'],dict(lower=None,upper=None))
        new=dict(old,generation_bound='first_release',release_pre_c=700,release_pre_ns=1280,
            first_generation=True,status='interval')
        bound=_ready_call_bounds(cta,events,new)
        self.assertEqual(bound['bounds']['ready_to_FIRST_cycles'],dict(lower=-40,upper=30))
        self.assertEqual(bound['bounds']['issue_to_ready_cycles'],dict(lower=170,upper=240))
        self.assertFalse(_ready_call_bounds(cta,events,old)['first_generation'])

    def test_success_after_release_stays_censored(self):
        cta=dict(sm=55,tiles=[[300,900]],tiles_ns=[[1000,1500]])
        events=dict(issue_end_c=100,issue_end_ns=900,wait_begin_c=50,wait_begin_ns=800)
        observer=dict(success=True,exit_c=701,exit_ns=1312,last_false_pre_c=600,last_false_pre_ns=1248,
            tests=12,sm=55,first_generation=False,status='after_release',generation_bound='first_release',
            release_pre_c=700,release_pre_ns=1280)
        result=_ready_call_bounds(cta,events,observer)
        self.assertEqual(result['status'],'after_release')
        self.assertTrue(all(v==dict(lower=None,upper=None) for v in result['bounds'].values()))


class ReadyDecoderTests(unittest.TestCase):
    def setUp(self):
        from analyze_r18 import decode_ready_observer
        self.decode=decode_ready_observer
        self.cta=dict(sm=55,entry_c=100,entry_ns=1000,tiles=[[300,600]],tiles_ns=[[1100,1300]])
        self.success_flag=((55+1)<<1)|1

    def test_first_test_success_and_regular_interval(self):
        first=self.decode([110,1024,0,0,200,1088,1,self.success_flag],True,self.cta)
        self.assertEqual(first['status'],'upper_only')
        self.assertTrue(first['first_generation'])
        normal=self.decode([110,1024,160,1056,200,1088,3,self.success_flag],True,self.cta)
        self.assertEqual(normal['status'],'interval')
        self.assertEqual(normal['last_false_pre_c'],160)

    def test_censored_late_success_and_limit_are_returned(self):
        late=self.decode([110,1024,280,1088,301,1120,12,self.success_flag],True,self.cta)
        self.assertEqual(late['status'],'after_first')
        limited=self.decode([110,1024,280,1088,290,1120,4096,self.success_flag-1],True,self.cta)
        self.assertEqual(limited['status'],'poll_limit')
        self.assertFalse(limited['first_generation'])

    def test_same_cycle_generation_can_have_ns_exit_after_FIRST(self):
        result=self.decode([110,1024,280,1088,290,1120,12,self.success_flag],True,self.cta)
        self.assertTrue(result['first_generation'])
        self.assertGreater(result['exit_ns'],self.cta['tiles_ns'][0][0])

    def test_old_invocation_and_wrong_SM_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'invocation'):
            self.decode([99,1024,160,1056,200,1088,3,self.success_flag],True,self.cta)
        with self.assertRaisesRegex(ValueError,'SM'):
            self.decode([110,1024,160,1056,200,1088,3,self.success_flag+2],True,self.cta)
        with self.assertRaisesRegex(ValueError,'first-test'):
            self.decode([110,1024,160,1056,200,1088,1,self.success_flag],True,self.cta)
        with self.assertRaisesRegex(ValueError,'limit'):
            self.decode([110,1024,160,1056,200,1088,3,self.success_flag-1],True,self.cta)

    def test_off_has_no_synthetic_bounds_or_stale_record(self):
        result=self.decode([0]*8,False,self.cta)
        self.assertEqual(result,dict(cta=55,enabled=False,protocol='r13-ready-observer-v1'))
        with self.assertRaisesRegex(ValueError,'disabled'):
            self.decode([1]+[0]*7,False,self.cta)

    def test_release_decoder_does_not_change_the_legacy_generation_rule(self):
        values=[110,1024,280,1088,340,1152,12,self.success_flag]
        self.assertEqual(self.decode(values,True,self.cta)['status'],'after_first')
        new=self.decode(values,True,self.cta,release_values=[500,1248])
        self.assertEqual(new['protocol'],'r13-ready-release-v1')
        self.assertEqual(new['generation_bound'],'first_release')
        self.assertEqual(new['status'],'interval')
        self.assertTrue(new['first_generation'])
        self.assertEqual(self.decode(values,True,self.cta)['status'],'after_first')

    def test_release_decoder_censors_late_success_and_checks_marker_identity(self):
        values=[110,1024,480,1216,501,1280,12,self.success_flag]
        late=self.decode(values,True,self.cta,release_values=[500,1248])
        self.assertEqual(late['status'],'after_release')
        self.assertFalse(late['first_generation'])
        off=self.decode([0]*8,False,self.cta,release_values=[0,0])
        self.assertEqual(off,dict(cta=55,enabled=False,protocol='r13-ready-release-v1'))
        with self.assertRaisesRegex(ValueError,'disabled'):
            self.decode([0]*8,False,self.cta,release_values=[500,1248])
        with self.assertRaisesRegex(ValueError,'outside'):
            self.decode(values,True,self.cta,release_values=[299,1248])

    def test_public_observer_identity_and_128B_allocation(self):
        from test_r09_history_series import synthetic
        from analyze_r18 import validate_warmup
        for enabled in (0,1):
            row,setup,call,_=synthetic();row.update(first_supply_events=1,ready_observer=enabled)
            for event in (setup,call):
                event.update(first_supply_events=1,first_supply_protocol='r13-first-supply-v1',
                    first_supply_scratch_bytes=128,ready_observer=enabled,ready_observer_protocol='r13-ready-observer-v1')
            validate_warmup(setup,call,row)
            for changes in (dict(ready_observer=1-enabled),dict(ready_observer_protocol='other'),dict(first_supply_scratch_bytes=64)):
                with self.subTest(changes=changes),self.assertRaises(ValueError):
                    validate_warmup(setup,dict(call,**changes),row)

    def test_release_protocol_requires_its_144B_identity(self):
        from test_r09_history_series import synthetic
        from analyze_r18 import validate_warmup
        for enabled in (0,1):
            row,setup,call,_=synthetic()
            row.update(first_supply_events=1,ready_observer=enabled,ready_generation_bound='first_release')
            for event in (setup,call):
                event.update(first_supply_events=1,first_supply_protocol='r13-first-supply-v1',
                    first_supply_scratch_bytes=144,ready_observer=enabled,ready_observer_protocol='r13-ready-release-v1')
            validate_warmup(setup,call,row)
            for changes in (dict(first_supply_scratch_bytes=128),dict(ready_observer_protocol='r13-ready-observer-v1')):
                with self.subTest(changes=changes),self.assertRaises(ValueError):
                    validate_warmup(setup,dict(call,**changes),row)


if __name__ == "__main__":
    unittest.main()

"""Caller-specified cycle phases and the existing release/slot feedback path."""
import hashlib
import json
from pathlib import Path
import unittest

from supply_model import admission_stage_events


class AdmissionObserverTests(unittest.TestCase):
    def test_omitted_and_none_inputs_preserve_complete_legacy_json(self):
        fixture_path=Path(__file__).parent/'testdata'/'r13_admission_legacy.json'
        fixture=json.loads(fixture_path.read_text())
        for case in fixture['cases']:
            for explicit_none in (False,True):
                with self.subTest(case=case['name'],explicit_none=explicit_none):
                    arguments=dict(case['arguments'])
                    if explicit_none:
                        arguments.update(pre_full_wait_cycles=None,post_full_wait_cycles=None)
                    result=admission_stage_events(**arguments)
                    serialized=json.dumps(result,separators=(',',':')).encode()
                    self.assertEqual(hashlib.sha256(serialized).hexdigest(),
                        case['legacy_complete_json_sha256'])
                    self.assertNotIn('full_wait_begin',result)
                    self.assertNotIn('mma_begin',result)

    def test_explicit_zero_phases_preserve_all_legacy_fields(self):
        fixture_path=Path(__file__).parent/'testdata'/'r13_admission_legacy.json'
        fixture=json.loads(fixture_path.read_text())
        for case in fixture['cases']:
            arguments=case['arguments']
            if 'observed_second_entry' in arguments:
                continue  # Observed anchors and caller phase costs are separate cuts.
            with self.subTest(case=case['name']):
                legacy=admission_stage_events(**arguments)
                phased=admission_stage_events(**arguments,
                    pre_full_wait_cycles=[0.]*(2*arguments['kt']),
                    post_full_wait_cycles=[0.]*(2*arguments['kt']))
                old_fields={name:phased[name] for name in legacy}
                self.assertEqual(json.dumps(old_fields,separators=(',',':')),
                    json.dumps(legacy,separators=(',',':')))
                self.assertEqual(phased['full_wait_begin'],legacy['consumer_available'])
                self.assertEqual(phased['full_wait_return'],legacy['first_mma_or_commit'])
                self.assertEqual(phased['mma_begin'],legacy['first_mma_or_commit'])

    def test_pre_delay_is_absorbed_by_existing_ready_wait(self):
        arguments=dict(kt=4,stages=2,producer_interval=0,ready_latency=20,
            commit_lead=0,completion_lag=0,service_cycles=1)
        legacy=admission_stage_events(**arguments)
        phased=admission_stage_events(**arguments,pre_full_wait_cycles=[5,0,0,0,0,0,0,0])
        self.assertEqual(phased['consumer_phase_entry'][0],0)
        self.assertEqual(phased['full_wait_begin'][0],5)
        self.assertEqual(phased['full_wait_return'][0],20)
        self.assertEqual(phased['mma_begin'][0],20)
        for field in legacy:
            if field=='consumer_available':
                self.assertEqual(phased[field][1:],legacy[field][1:])
            else:
                self.assertEqual(phased[field],legacy[field],field)

    def test_pre_delay_reaches_current_commit_and_previous_group_release(self):
        legacy=admission_stage_events(4,2,0,0,0,0,service_cycles=1)
        phased=admission_stage_events(4,2,0,0,0,0,service_cycles=1,
            pre_full_wait_cycles=[0,0,10,0,0,0,0,0])
        self.assertEqual(phased['consumer_phase_entry'][2],2)
        self.assertEqual(phased['full_wait_begin'][2],12)
        self.assertEqual(phased['full_wait_return'][2],12)
        self.assertEqual(phased['mma_begin'][2],12)
        self.assertEqual(phased['commits'][2],13)
        self.assertEqual(phased['released'][1],13)
        self.assertEqual(phased['issued'][3],13)
        self.assertEqual(phased['commits'][:2],legacy['commits'][:2])
        self.assertEqual(phased['released'][0],legacy['released'][0])
        self.assertEqual(phased['issued'][:3],legacy['issued'][:3])

    def test_post_delay_keeps_full_wait_return_and_delays_same_group_commit(self):
        legacy=admission_stage_events(4,2,0,0,0,0,service_cycles=1)
        phased=admission_stage_events(4,2,0,0,0,0,service_cycles=1,
            post_full_wait_cycles=[0,0,10,0,0,0,0,0])
        self.assertEqual(phased['full_wait_begin'][2],2)
        self.assertEqual(phased['full_wait_return'][2],2)
        self.assertEqual(phased['mma_begin'][2],12)
        self.assertEqual(phased['commits'][2],13)
        self.assertEqual(phased['wait1_return'][2],13)
        self.assertEqual(phased['released'][1],13)
        self.assertEqual(phased['issued'][3],13)
        self.assertEqual(phased['commits'][:2],legacy['commits'][:2])
        self.assertEqual(phased['released'][0],legacy['released'][0])
        self.assertEqual(phased['issued'][:3],legacy['issued'][:3])

    def test_post_delay_can_be_absorbed_by_prior_service_progress(self):
        legacy=admission_stage_events(4,2,0,0,9,0,service_cycles=10)
        phased=admission_stage_events(4,2,0,0,9,0,service_cycles=10,
            post_full_wait_cycles=[0,3,0,0,0,0,0,0])
        self.assertEqual(phased['full_wait_return'][1],1)
        self.assertEqual(phased['mma_begin'][1],4)
        self.assertEqual(phased['commits'][1],11)
        for field in ('service_frontier','commits','released','issued','main_ends'):
            self.assertEqual(phased[field],legacy[field],field)

    def test_second_prologue_does_not_release_previous_output_group(self):
        legacy=admission_stage_events(4,2,0,0,0,0,service_cycles=1)
        phased=admission_stage_events(4,2,0,0,0,0,service_cycles=1,
            post_full_wait_cycles=[0,0,0,0,10,0,0,0])
        self.assertEqual(phased['released'][:4],legacy['released'][:4])
        self.assertEqual(phased['commits'][4],15)
        self.assertEqual(phased['wait1_return'][4],15)
        self.assertEqual(phased['released'][4],16)
        self.assertEqual(phased['wait1_return'][5],16)
        self.assertEqual(phased['issued'][:6],legacy['issued'][:6])
        self.assertEqual(phased['issued'][6],16)

    def test_stage4_and_stage6_feedback_reaches_global_group80(self):
        for stages in (4,6):
            with self.subTest(stages=stages):
                legacy=admission_stage_events(64,stages,0,0,0,0,service_cycles=1)
                intervention_group=80-stages+1
                post=[0]*128
                post[intervention_group]=10
                phased=admission_stage_events(64,stages,0,0,0,0,service_cycles=1,
                    post_full_wait_cycles=post)
                release_group=80-stages
                self.assertEqual(phased['released'][release_group],
                    legacy['released'][release_group]+10)
                self.assertEqual(phased['issued'][80],legacy['issued'][80]+10)
                self.assertEqual(phased['issued'][:80],legacy['issued'][:80])
                self.assertEqual(phased['slot'],legacy['slot'])
                self.assertEqual(phased['slot'][80],phased['slot'][release_group])

    def test_pre_and_post_phases_remain_distinct_after_second_prefill(self):
        phased=admission_stage_events(4,2,0,20,0,0,service_cycles=1,second_prefill=True,
            pre_full_wait_cycles=[0,0,0,0,3,0,0,0],
            post_full_wait_cycles=[0,0,0,0,5,0,0,0])
        entry=phased['second_prefill']['end']
        self.assertEqual(phased['consumer_phase_entry'][4],entry)
        self.assertEqual(phased['full_wait_begin'][4],entry+3)
        self.assertEqual(phased['full_wait_return'][4],entry+3)
        self.assertEqual(phased['mma_begin'][4],entry+8)
        self.assertEqual(phased['commits'][4],entry+9)

    def test_phase_vectors_require_every_group_to_have_an_explicit_price(self):
        invalid_vectors=([],[0]*7,[0]*9,3,'00000000',{i:0 for i in range(8)})
        for name in ('pre_full_wait_cycles','post_full_wait_cycles'):
            for vector in invalid_vectors:
                with self.subTest(name=name,vector=vector):
                    with self.assertRaisesRegex(ValueError,'sequence|exactly'):
                        admission_stage_events(4,2,0,0,0,0,service_cycles=1,**{name:vector})
            for invalid in (-1,float('nan'),float('inf'),True,'0',complex(0)):
                vector=[0]*8
                vector[2]=invalid
                with self.subTest(name=name,invalid=invalid):
                    with self.assertRaisesRegex(ValueError,'finite nonnegative real'):
                        admission_stage_events(4,2,0,0,0,0,service_cycles=1,**{name:vector})
            vector=[0]*8
            vector[2]=None
            with self.subTest(name=name,invalid=None):
                with self.assertRaisesRegex(ValueError,'unpriced'):
                    admission_stage_events(4,2,0,0,0,0,service_cycles=1,**{name:vector})

    def test_observed_entry_cannot_be_reused_as_a_phase_price(self):
        for name in ('pre_full_wait_cycles','post_full_wait_cycles'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError,'observed entry cut'):
                    admission_stage_events(4,2,0,0,0,0,service_cycles=1,
                        observed_second_entry=(0,1),**{name:[0]*8})


if __name__=='__main__':
    unittest.main()

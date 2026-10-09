"""CPU checks for freeze boundaries and retention of failed heldout conditions."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import run_v09 as runner
import v08_model


class FreezeChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        for folder in ('inputs','build','source'):(self.root/folder).mkdir()
        files={'cases.json':[dict(id='h',set='heldout')],
               'static_setup.json':[dict(case='h',setup={})],
               'environment.json':dict(gpu='test fixture'),
               'run_config.json':dict(source_commit='fixture'),
               'source_hashes.json':{},'build/binary_hashes.json':{},
               'build/sass_hashes.json':{},'inputs/policy.json':{}}
        for name,value in files.items():runner.write(self.root/name,value)
        for name in ('freeze.sh','heldout.sh'):(self.root/name).write_text('fixture\n')

    def tearDown(self):self.tmp.cleanup()

    def freeze(self):
        with patch.object(runner.v09_model,'predict_components',return_value=dict(all_supply_supported=True,plain_us=3)):
            return runner.freeze_predictions(self.root,{}, {}, {})

    def test_bound_files_and_approved_hashes(self):
        hashes=self.freeze()
        args=(hashes['prediction_sha256'],hashes['manifest_sha256'])
        runner.verify_approved(self.root,*args)
        with self.assertRaisesRegex(ValueError,'SHA256'):runner.verify_approved(self.root,'wrong',args[1])
        runner.write(self.root/'inputs/policy.json',dict(changed=True))
        with self.assertRaisesRegex(ValueError,'input changed'):runner.verify_approved(self.root,*args)

    def test_readonly_required(self):
        hashes=self.freeze();(self.root/'frozen/calibration.json').chmod(0o644)
        with self.assertRaisesRegex(ValueError,'read-only'):
            runner.verify_approved(self.root,hashes['prediction_sha256'],hashes['manifest_sha256'])

    def test_target_samples_block_freeze(self):
        (self.root/'samples/h').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'predate freeze'):self.freeze()
        self.assertFalse((self.root/'frozen').exists())

    def test_failed_condition_is_scored(self):
        runner.write(self.root/'prediction_binding.json',dict(first_sample_unix_ns=1))
        runner.score(self.root,{'h':{'plain_us':3}},dict(median_percent=5,maximum_percent=10),
                     [dict(case='h',variant='plain',error='numeric failure')])
        result=runner.read(self.root/'score.json')
        self.assertEqual(result['total_conditions'],1)
        self.assertEqual(result['valid_plain_conditions'],0)
        self.assertFalse(result['complete_time_passed'])
        self.assertEqual(len(result['cases'][0]['errors']),1)


class ModelChecks(unittest.TestCase):
    def test_trace_does_not_change_recurrence(self):
        p=dict(P0=200,S=500,l0=100,l1=50,dL0=60,w=20,h=10,E0=70,Elast=50,
               E=80,Etail=30,x0=0,x1=0,xk=0,gm=15,we=5,r=1.2)
        for schedule in ('cooperative','pingpong'):
            for n in (0,1,2,3,24):
                for kt in (4,16,64,512):
                    with self.subTest(schedule=schedule,n=n,kt=kt):
                        old=v08_model.cta_cycles(p,schedule,n,kt,detail=True)
                        new=v08_model.cta_cycles(p,schedule,n,kt,detail=True,trace_events=True)
                        self.assertEqual(old[0],new[0])
                        if n:
                            self.assertEqual(len(new[1]['events']),n)
                            for key,value in old[1].items():self.assertEqual(value,new[1][key])
                            self.assertEqual(new[0],max(t[3] for t in new[1]['events'])+p['Etail'])

    def test_reference_drift(self):
        expected={'ref':dict(plain_us=10,effective_ghz=1.8,event_extra_us=3)}
        policy=dict(reference_relative_limits=dict(plain_us=.05,effective_ghz=.03),
                    reference_absolute_limits=dict(event_extra_us=.5))
        self.assertTrue(runner.compare_references(expected,expected,policy)['compatible'])
        for key,value in [('plain_us',10.6),('effective_ghz',1.86),('event_extra_us',3.6)]:
            actual={'ref':dict(expected['ref'],**{key:value})}
            self.assertFalse(runner.compare_references(actual,expected,policy)['compatible'])


if __name__=='__main__':unittest.main()

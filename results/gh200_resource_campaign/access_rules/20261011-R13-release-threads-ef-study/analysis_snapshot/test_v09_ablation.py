"""Check the ablation's cycle/ns boundary and shared clock work definition."""
import json
from pathlib import Path
import unittest

import analyze_v09_ablation as ablation


class AblationUnits(unittest.TestCase):
    def test_all_event_fields_are_converted_from_cycles(self):
        row=dict(config='cfg_b',m=128,n=128,k=64,kind='ordinary',swizzle=1)
        setup=dict(grid=[1,1,1],stages=6)
        p=dict(P0=200,S=500,l0=100,l1=50,dL0=60,w=20,h=10,E0=70,Elast=50,
               E=80,Etail=30,x0=3,x1=2,xk=.1,gm=15,we=5,r=1.2,
               S_form='const',E_form='const',rho={},boundary_form='none')
        cal=dict(clock=dict(coefficients=[2,0,0,0,0,0,0],tau_us=1),
                 plain_transfer=dict(cfg_b=dict(F_us=3,kappa=1)))
        predicted=ablation.aggregate_prediction(row,setup,{'cfg_b':p},cal,observed_f=2)
        cycles,parts=ablation.events.cta_cycles(p,'pingpong',1,1,detail=True,trace_events=True)
        for key,value in parts.items():
            if key=='events':self.assertEqual(predicted['cta_events'][0][key],[[x/2 for x in t] for t in value])
            else:self.assertEqual(predicted['cta_events'][0][key],value/2,key)
        self.assertAlmostEqual(predicted['window_us'],cycles/2000)
        self.assertAlmostEqual(predicted['plain_us'],3+cycles/2000)

    def test_shared_clock_features_match_frozen_v09(self):
        root=Path(__file__).resolve().parents[3]/'results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2'
        if not root.exists():self.skipTest('local V09 archive required')
        cal=ablation.read(root/'frozen/calibration.json');frozen=ablation.read(root/'frozen/predictions.json')['predictions']
        setups={s['case']:s['setup'] for s in ablation.read(root/'static_setup.json')}
        for row in ablation.read(root/'cases.json'):
            if row['set']!='heldout' or frozen[row['id']]['status']!='predicted':continue
            with self.subTest(case=row['id']):
                work=ablation.events.scheduled_work(row['config'],row['m'],row['n'],setups[row['id']]['grid'],row['swizzle'])
                point,_=ablation.clock_inputs(row,work,cal)
                self.assertEqual(point,frozen[row['id']]['static_clock_work'])


if __name__=='__main__':unittest.main()

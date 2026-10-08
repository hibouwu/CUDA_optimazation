import json
from pathlib import Path
import tempfile
import unittest
import v06_model as base
import v07_model as model

class V07RecurrenceTests(unittest.TestCase):
    def setUp(self):
        self.params=dict(P0=20,S=40,l0=50,l1=512,dL0=100,w=8,E0=100,E=80,h=10,Etail=30,x0=12,x1=2,xk=.01)
        self.clock=dict(a=2,b=.01,c=.01,d=.01)
        self.row=dict(config='cfg_a',kind='aligned',m=1792,n=1920,k=1536,zero_m=-1,zero_n=-1)

    def test_no_boundary_matches_existing_recurrence(self):
        grid=base.emulate_grid('cfg_a',self.row['m'],self.row['n'])
        original=base.predict_case(self.row,grid,self.params,self.clock,3)
        actual=model.predict(self.row,grid,self.params,self.clock,3,{})
        self.assertAlmostEqual(actual['critical_cycles'],original['critical_cycles'])
        self.assertAlmostEqual(actual['predicted_us'],original['predicted_us'])

    def test_extra_window_is_added_once(self):
        row=dict(self.row,kind='oob',m=1664);grid=base.emulate_grid('cfg_a',row['m'],row['n'])
        rules={f'cfg_a|oob|{j}|{cls}':dict(a=10,b=0,kt_min=1,kt_max=256) for j in [0,1] for cls in ['other','same_column','oob','partner']}
        original=base.predict_case(row,grid,self.params,self.clock,3)
        actual=model.predict(row,grid,self.params,self.clock,3,rules)
        self.assertAlmostEqual(actual['critical_cycles']-original['critical_cycles'],20.2)
        self.assertFalse(actual['boundary_unknown_cells'])

    def test_unqualified_K_point_cannot_define_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'rules.json').write_text(json.dumps(dict(conditions=[dict(id='lo',kind='oob',config='cfg_a'),dict(id='hi',kind='oob',config='cfg_a')])))
            points=[dict(case='lo',round=0,cls='oob',k=1024,delta_cycles=100,qualified=False),dict(case='hi',round=0,cls='oob',k=8192,delta_cycles=200,qualified=True)]
            (p/'paired-tile-increments.json').write_text(json.dumps(points))
            result=model.boundary_models(p)['cfg_a|oob|0|oob']
            self.assertNotIn('a',result)
            self.assertEqual(result['status'],'insufficient_qualified_K_support')

if __name__=='__main__':unittest.main()

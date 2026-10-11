"""Persistent boundary work coverage, WG alternation and cross-output ring state."""
import unittest
from collections import Counter

from supply_model import cfg_b_logical_boundary_ledger,cfg_b_persistent_boundary_ledger


class BoundaryBindingTests(unittest.TestCase):
    def row(self,m=136,n=264,k=72):
        return dict(config='cfg_b',m=m,n=n,k=k,lda=k,ldb=n,ldd=n,swizzle=1,raster_order='heuristic')

    def setup(self,grid=(2,1,1),stages=3):
        return dict(grid=list(grid),stages=stages,tile=[128,128,64],cluster=[1,1,1])

    def test_tail_bounds_and_source_identity_survive_binding(self):
        row=self.row();logical=cfg_b_logical_boundary_ledger(row)
        bound=cfg_b_persistent_boundary_ledger(row,self.setup())
        self.assertEqual(bound['totals'],logical['totals'])
        originals={o['coord']:o for o in logical['outputs']}
        for o in bound['outputs']:
            original=originals[o['coord']]
            self.assertEqual(o['D_boxes'],original['D_boxes'])
            for g,old in zip(o['groups'],original['groups']):
                self.assertEqual(g['atoms'],old['atoms'])
                self.assertEqual(g['transaction_expected_bytes'],32768)
        self.assertTrue(all(o['cta'] is None for o in logical['outputs']))

    def test_persistent_ordinals_cross_output_stage_boundaries(self):
        ledger=cfg_b_persistent_boundary_ledger(self.row(),self.setup())
        outputs=[o for o in ledger['outputs'] if o['cta']==0]
        self.assertEqual([o['coord'] for o in outputs],[(0,0),(0,1),(0,2)])
        self.assertEqual([o['consumer_warp_group'] for o in outputs],[0,1,0])
        states=[g['slot_state'] for o in outputs for g in o['groups']]
        self.assertEqual([s['group'] for s in states],list(range(6)))
        self.assertEqual(states[2],dict(group=2,slot=2,generation=0,full_wait_phase=0,empty_wait_phase=1,previous_group=None))
        self.assertEqual(states[3],dict(group=3,slot=0,generation=1,full_wait_phase=1,empty_wait_phase=0,previous_group=0))

    def test_heuristic_and_explicit_rasters_keep_all_coordinates(self):
        for m,n,raster,grid in ((264,136,'heuristic',(1,2,1)),(136,264,'along_n',(1,2,1)),
                                (264,136,'along_m',(2,1,1))):
            row=dict(self.row(m,n),raster_order=raster)
            ledger=cfg_b_persistent_boundary_ledger(row,self.setup(grid))
            self.assertEqual({o['coord'] for o in ledger['outputs']},
                {(mi,ni) for mi in range((m+127)//128) for ni in range((n+127)//128)})
            self.assertEqual(len(ledger['outputs']),6)

    def test_original_r10_boundary_and_recorded_window_coordinates(self):
        row=self.row(2600,3000,2000)
        ledger=cfg_b_persistent_boundary_ledger(row,self.setup((132,1,1),6))
        self.assertEqual(Counter(ledger['scheduler']['output_counts']),{4:108,3:24})
        self.assertEqual(len({o['coord'] for o in ledger['outputs']}),504)
        for o in ledger['outputs']:
            linear=o['cta']+132*o['output_ordinal']
            self.assertEqual(o['coord'],(linear%21,linear//21))
        windows=[o for o in ledger['outputs'] if o['cta']<4 and o['output_ordinal']==2]
        self.assertEqual([o['coord'] for o in windows],[(12,12),(13,12),(14,12),(15,12)])
        self.assertEqual([o['groups'][0]['slot_state']['slot'] for o in windows],[4]*4)
        self.assertEqual([o['groups'][-1]['slot_state']['slot'] for o in windows],[5]*4)

    def test_rejects_unestablished_scheduler_and_incompatible_shapes(self):
        for row,setup in ((dict(self.row(),swizzle=2),self.setup()),
                          (dict(self.row(),raster_order=None),self.setup()),
                          (self.row(),self.setup((1,2,1))),
                          (self.row(),self.setup(stages=1)),
                          (self.row(),dict(self.setup(),cluster=[1,2,1]))):
            with self.subTest(row=row,setup=setup):
                with self.assertRaises(ValueError):cfg_b_persistent_boundary_ledger(row,setup)


if __name__=='__main__':unittest.main()

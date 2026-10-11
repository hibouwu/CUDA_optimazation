"""Boundary source/fill conservation and compatibility with full-tile inputs."""
import unittest

from supply_model import cfg_b_logical_boundary_ledger,cfg_b_source_program


class BoundaryLedgerTests(unittest.TestCase):
    def row(self,m=136,n=136,k=72):
        return dict(config='cfg_b',m=m,n=n,k=k,lda=k,ldb=n,ldd=n,swizzle=1)

    def test_partial_k_and_mn_keep_zero_source_atom_and_full_transaction(self):
        ledger=cfg_b_logical_boundary_ledger(self.row())
        corner=next(o for o in ledger['outputs'] if o['coord']==(1,1))
        last=corner['groups'][-1]
        self.assertEqual(last['valid_k'],8)
        self.assertEqual([a['source_valid_bytes'] for a in last['atoms']],[128,128,0])
        self.assertEqual([a['destination_fill_bytes'] for a in last['atoms']],[16256,8064,8192])
        self.assertTrue(all(a['destination_required'] for a in last['atoms']))
        self.assertEqual(last['transaction_expected_bytes'],32768)
        self.assertEqual(sum(d['global_valid_bytes'] for d in corner['D_boxes']),256)
        self.assertEqual(sum(d['global_oob_bytes'] for d in corner['D_boxes']),65536-256)
        self.assertIsNone(corner['cta']);self.assertIsNone(corner['output_ordinal'])
        self.assertIsNone(corner['global_group_binding'])

    def test_small_problem_uses_one_full_destination_group(self):
        ledger=cfg_b_logical_boundary_ledger(self.row(8,8,8));total=ledger['totals']
        self.assertEqual(total['logical_input_atoms'],3)
        self.assertEqual(total['source_valid_bytes'],256)
        self.assertEqual(total['destination_fill_bytes'],32768-256)
        self.assertEqual(total['transaction_expected_bytes'],32768)
        self.assertEqual(total['zero_source_atoms'],1)
        self.assertEqual(total['D_global_valid_bytes'],256)
        self.assertEqual(total['D_zero_valid_boxes'],7)

    def test_all_valid_source_elements_are_counted_with_logical_reuse(self):
        row=self.row();total=cfg_b_logical_boundary_ledger(row)['totals']
        self.assertEqual(total['source_valid_bytes'],2*row['m']*row['k']*2+2*row['n']*row['k']*2)
        self.assertEqual(total['source_valid_bytes']+total['destination_fill_bytes'],4*2*32768)
        self.assertEqual(total['D_global_valid_bytes'],136*136*4)

    def test_full_tile_keys_match_existing_source_program(self):
        row=self.row(128,128,128);setup=dict(grid=[1,1,1],stages=6,tile=[128,128,64],cluster=[1,1,1])
        previous=cfg_b_source_program(row,setup);new=cfg_b_logical_boundary_ledger(row)
        keys=[tuple(a['source_key'] for a in group['atoms']) for group in new['outputs'][0]['groups']]
        self.assertEqual(keys,previous['source_keys'][0])
        self.assertEqual(new['totals']['source_valid_bytes']/1024,previous['totals']['source_valid_KiB'])
        self.assertEqual(new['totals']['destination_fill_bytes'],0)
        self.assertEqual(new['totals']['D_global_valid_bytes']/1024,previous['totals']['output_valid_KiB'])

    def test_rejects_unsupported_source_maps_and_illegal_pitches(self):
        for extra in ({'config':'cfg_c'},{'input_map_n':256},{'source_copies':2},
                      {'b_offset_bytes':16},{'lda':73},{'ldd':135},{'k':True}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):cfg_b_logical_boundary_ledger(dict(self.row(),**extra))


if __name__=='__main__':unittest.main()

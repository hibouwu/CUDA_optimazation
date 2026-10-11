"""Static controls for the prepared A-only rotation experiment; no GPU claims."""
import json
from pathlib import Path
import tempfile
import unittest

import run_r18
from supply_model import shared_stage_events, static_tma_requests


class ARotationTests(unittest.TestCase):
    def test_unrotated_source_parser_cannot_silently_accept_rotation(self):
        with self.assertRaisesRegex(ValueError,'A rotation'):
            static_tma_requests({'a_row_rotation':1},{})

    def test_all_rows_round_trip_and_each_round_keeps_its_addresses(self):
        for shift in (0, 1):
            physical_rows=[]
            for logical in range(2816):
                tile,within=divmod(logical,128)
                physical_tile=(tile//11)*11+(tile%11+shift)%11
                physical=physical_tile*128+within
                inverse=(physical_tile//11)*11+(physical_tile%11+11-shift)%11
                self.assertEqual(inverse*128+within,logical)
                self.assertEqual(physical//1408,logical//1408)
                physical_rows.append(physical)
            self.assertEqual(sorted(physical_rows),list(range(2816)))

    def test_pair_changes_only_source_rotation(self):
        rows=json.loads((Path(__file__).parent/'configs/r13-a-row-rotation.json').read_text())
        self.assertEqual([r['a_row_rotation'] for r in rows],[0,1])
        self.assertEqual({k:v for k,v in rows[0].items() if k not in ('id','a_row_rotation')},
                         {k:v for k,v in rows[1].items() if k not in ('id','a_row_rotation')})
        for row in rows:
            self.assertEqual(row['variants'],['plain','dual'])
            self.assertEqual(row['epilogue_defer'],2)
            self.assertIn('--a-row-rotation',run_r18.case_args(row))

    def test_preparation_changes_A_partition_but_keeps_B_and_work_coordinate(self):
        original=('      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;\n'
            '      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);\n'
            '      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);\n')
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp);path=source/'cutlass/include'/run_r18.common.MAINLOOP
            path.parent.mkdir(parents=True);path.write_text(original)
            run_r18.make_a_row_rotation(source)
            text=path.read_text()
            self.assertIn('auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;',text)
            self.assertIn('Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);',text)
            self.assertIn('Tensor gA = gA_mkl(_,_,r13_source_m,_,l_coord);',text)
            self.assertNotIn('m_coord =',text)

    def test_A_only_rename_keeps_the_equality_model_prediction(self):
        # All A rows rotate within the same round; B keys and destinations stay put.
        kt=4
        def keys(shift):
            result=[]
            for cta in range(132):
                a,b=divmod(cta,12);actor=[]
                for i in range(2*kt):
                    j,k=divmod(i,kt)
                    actor.append((('A',11*j+(a+shift)%11,k),('B0',b,k),('B1',b,k)))
                result.append(actor)
            return result
        params=dict(service_ns=2.,read_latency_ns=100.,compute_ns=64.,
                    destination_service_ns=3.,producer_gap_ns=5.,operand_gap_ns=2.,
                    share_inflight_sources=True,epilogue_defer=2)
        self.assertEqual(shared_stage_events(kt,4,132,source_keys=keys(0),**params),
                         shared_stage_events(kt,4,132,source_keys=keys(1),**params))


if __name__=='__main__':
    unittest.main()

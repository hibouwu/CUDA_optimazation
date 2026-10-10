"""Explicit scheduler order preserves work while changing source-panel grouping."""
import json
from pathlib import Path
import unittest
import v08_model


class RasterOrderTests(unittest.TestCase):
    def test_explicit_orders_cover_same_tiles_and_keep_counts(self):
        orders={}
        for order,grid in [('along_m',[132,1,1]),('along_n',[1,132,1])]:
            orders[order]=v08_model.scheduled_work('cfg_b',2304,3072,grid,1,order)
            self.assertEqual({p for c in orders[order] for p in c},
                             {(m,n) for m in range(18) for n in range(24)})
            self.assertEqual(sorted(map(len,orders[order])),[3]*96+[4]*36)
        def panels(work):return len({c[0][0] for c in work}),len({c[0][1] for c in work})
        self.assertEqual(panels(orders['along_m']),(18,8))
        self.assertEqual(panels(orders['along_n']),(6,24))
        self.assertEqual(orders['along_m'],v08_model.scheduled_work('cfg_b',2304,3072,[132,1,1]))
        self.assertEqual(v08_model.scheduled_work('cfg_b',2304,2304,[1,132,1]),
                         v08_model.scheduled_work('cfg_b',2304,2304,[1,132,1],1,'along_n'))

    def test_declared_pair_only_changes_pitch_and_order(self):
        rows=json.loads((Path(__file__).parent/'configs/r10-raster-pitch.json').read_text())
        rows=[r for r in rows if r['set']=='ctrl']
        self.assertEqual(len(rows),4)
        ignored={'id','group','raster_order','ldb'}
        normalized=[{k:v for k,v in r.items() if k not in ignored} for r in rows]
        self.assertTrue(all(r==normalized[0] for r in normalized))
        self.assertEqual({(r['raster_order'],r['ldb']) for r in rows},
                         {(order,ldb) for order in ('along_m','along_n') for ldb in (3072,3088)})


if __name__=='__main__':unittest.main()

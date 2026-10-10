"""Explicit scheduler order preserves work while changing source-panel grouping."""
import json
from pathlib import Path
import unittest
import v08_model
from analyze_r10 import mainloop_overlap_features, source_panel_features


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

    def test_peer_CTA_overlapping_outputs_are_counted_once(self):
        windows=[dict(cta=0,j=0,mi=0,ni=0,begin=0,end=10),
            dict(cta=1,j=0,mi=0,ni=0,begin=0,end=8),
            dict(cta=1,j=1,mi=0,ni=0,begin=4,end=10)]
        target=mainloop_overlap_features(windows)[0]
        self.assertEqual(target['other_cta'],1)
        self.assertEqual(target['same_B'],1)
        self.assertEqual(target['other_ordinal'],.6)

    def test_overlap_excludes_all_outputs_of_own_CTA(self):
        windows=[dict(cta=0,j=j,mi=0,ni=0,begin=0,end=10) for j in (0,1)]
        self.assertTrue(all(all(value==0 for value in row.values()) for row in mainloop_overlap_features(windows)))

    def test_overlap_counts_different_ordinals_with_their_actual_coordinates(self):
        windows=[dict(cta=0,j=0,mi=0,ni=0,begin=0,end=10),
            dict(cta=1,j=2,mi=1,ni=0,begin=0,end=5),
            dict(cta=2,j=0,mi=0,ni=1,begin=5,end=10)]
        target=mainloop_overlap_features(windows)[0]
        self.assertEqual(target,dict(other_cta=1,same_A=.5,same_B=.5,other_ordinal=.5))

    def test_source_pressure_controls_preserve_selected_panels_and_remove_next_output(self):
        rows=json.loads((Path(__file__).parent/'configs/r10-source-pressure.json').read_text())
        controls=[r for r in rows if r['set']=='ctrl'];self.assertEqual(len(controls),6)
        fixed=('m','k','lda','ldd','storage_m','storage_n','input_map_m','input_map_n',
            'alloc_lda','alloc_ldb','swizzle','input_mode','seed','raster_order')
        self.assertTrue(all(all(r[k]==controls[0][k] for k in fixed) for r in controls))
        selected=[]
        for row in controls:
            work=v08_model.scheduled_work('cfg_b',row['m'],row['n'],[row['sm_count'],1,1],
                row['swizzle'],row['raster_order'])
            selected.append([c[0] for c in work[:36]])
            self.assertEqual(sum(c[0][1]==0 for c in work),18)
            self.assertEqual(sum(c[0][1]==1 for c in work),18)
            if 'single' in row['id']:self.assertTrue(all(len(c)==1 for c in work))
            self.assertEqual(row['input_map_n'],3072)
        self.assertTrue(all(coords==selected[0] for coords in selected))
        for shape in ('single36','single126','persistent132'):
            a,b=[r for r in controls if shape in r['id']]
            self.assertEqual({r['ldb'] for r in (a,b)},{3072,3088})
            self.assertEqual({k:v for k,v in a.items() if k not in ('id','ldb')},
                {k:v for k,v in b.items() if k not in ('id','ldb')})

    def test_static_counts_keep_full_distributions_and_logical_history(self):
        work=v08_model.scheduled_work('cfg_b',2304,3072,[132,1,1],1,'along_m')
        data=source_panel_features(work)
        first=next(x for x in data['groups'] if x['j']==0 and x['T']==4)
        last=next(x for x in data['groups'] if x['j']==3)
        self.assertEqual(first['B_peer']['median'],17)
        self.assertEqual(last['B_peer']['median'],17)
        self.assertEqual(last['active_times_B_peer']['median'],612)
        self.assertEqual(last['B_seen_earlier_wave_fraction'],0)


if __name__=='__main__':unittest.main()

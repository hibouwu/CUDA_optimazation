"""Explicit scheduler order preserves work while changing source-panel grouping."""
import json
from pathlib import Path
import unittest
from fractions import Fraction
from supply_model import source_sharing_delta_cycles
import v08_model
from analyze_r10 import (mainloop_overlap_features, source_panel_features,
    _source_service_clip, _source_service_interval, _source_polygon_vertices, _source_service_value,
    _source_service_window_errors, _source_interval_error_floor)


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

    def test_service_response_parameters_remain_nonunique(self):
        F=Fraction
        regions=[[(-F(1),F(0),F(0)),(F(0),-F(1),F(0))]]
        regions=_source_service_clip(regions,36,[204],1,0,0)
        regions=_source_service_clip(regions,126,[204],1,148,148)
        vertices={v for r in regions for v in _source_polygon_vertices(r)}
        self.assertGreater(len(vertices),1)
        for a,b in vertices:
            self.assertEqual(_source_service_value(a,b,36,204),0)
            self.assertEqual(_source_service_value(a,b,126,204),148)
        self.assertGreater(max(a for a,b in vertices),10)
        bounded=[r+[(F(1),F(0),F('4.2'))] for r in regions]
        bounded=[r for r in bounded if _source_polygon_vertices(r)]
        self.assertEqual(max(a for r in bounded for a,b in _source_polygon_vertices(r)),F('4.2'))

    def test_transfer_interval_includes_new_max_branch_crossing(self):
        F=Fraction
        regions=[[(-F(1),F(0),F(0)),(F(0),-F(1),F(0))]]
        regions=_source_service_clip(regions,36,[204],1,0,0)
        regions=_source_service_clip(regions,126,[204],1,148,148)
        bounds=_source_service_interval(regions,132,[204],1)
        self.assertEqual(bounds[0],148)
        self.assertAlmostEqual(bounds[1],148+6*512/132)
        training_corner_max=max(float(_source_service_value(a,b,132,204))
            for r in regions for a,b in _source_polygon_vertices(r))
        self.assertLess(training_corner_max,bounds[1])

    def test_even_window_median_averages_predictions_not_features(self):
        F=Fraction
        fixed=[[(F(1),F(0),F(3)),(-F(1),F(0),-F(3)),
                (F(0),F(1),F(3)),(F(0),-F(1),-F(3))]]
        # Two middle windows straddle the max branch. f(mean X) is not mean f(X).
        self.assertEqual(_source_service_interval(fixed,132,[24,60],1),[32,32])
        self.assertEqual(_source_service_value(F(3),F(3),132,42),10)

    def test_pure_service_candidate_shares_parameters_and_preserves_exact_boundary(self):
        a,b=Fraction(2),Fraction(2)
        self.assertEqual(source_sharing_delta_cycles(a,b,36,204,16),0)
        self.assertEqual(source_sharing_delta_cycles(a,b,126,204,16),2368)
        self.assertEqual(source_sharing_delta_cycles(a,b,128,128),0)
        self.assertEqual(source_sharing_delta_cycles(a,b,128,Fraction(257,2)),1)
        self.assertEqual(source_sharing_delta_cycles(a,b,200,0,16),0)

    def test_window_response_uses_each_CTA_sharing_count(self):
        windows=[dict(cta=0,j=0,T=1,mi=0,ni=0,L_cycles_base=10000,L_cycles_variant=12560),
            dict(cta=1,j=0,T=1,mi=1,ni=1,L_cycles_base=10000,L_cycles_variant=10000)]
        features={(0,0):dict(mi=0,ni=0,active_ctas=132,B_peer=17),
                  (1,0):dict(mi=1,ni=1,active_ctas=132,B_peer=5)}
        scores=_source_service_window_errors(2,2,windows,features,16)
        self.assertEqual([r['predicted_delta'] for r in scores],[2560,0])
        self.assertEqual([r['relative_error'] for r in scores],[0,0])
        features[1,0]['ni']=2
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            _source_service_window_errors(2,2,windows,features,16)

    def test_domain_error_floor_handles_both_sides_and_contained_response(self):
        self.assertEqual(_source_interval_error_floor((20,40),30,100),0)
        self.assertEqual(_source_interval_error_floor((20,40),10,100),.1)
        self.assertEqual(_source_interval_error_floor((20,40),50,100),.1)


if __name__=='__main__':unittest.main()

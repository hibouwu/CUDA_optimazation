"""Explicit scheduler order preserves work while changing source-panel grouping."""
import json
from copy import deepcopy
from pathlib import Path
import unittest
from fractions import Fraction
from supply_model import source_sharing_delta_cycles, supply_request_rows
import v08_model
from analyze_r10 import (mainloop_overlap_features, source_panel_features,
    _source_service_clip, _source_service_interval, _source_polygon_vertices, _source_service_value,
    _source_service_window_errors, _source_interval_error_floor,
    _base_shift_response_pairs, _base_shift_pattern_metrics,
    _work_swap_expected, _work_swap_pitch_windows, _work_swap_correspondence)


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

    def test_base_shift_factorial_keeps_capacity_and_all_other_control_fields(self):
        rows=json.loads((Path(__file__).parent/'configs/r10-base-shift.json').read_text())
        references=[r for r in rows if r['set']=='reference']
        controls=[r for r in rows if r['set']=='ctrl']
        self.assertEqual(len(references),2)
        self.assertTrue(all(r['variants']==['plain'] for r in references))
        self.assertEqual(rows[:2],references)
        self.assertEqual({(r['b_offset_bytes'],r['ldb']) for r in controls},
            {(offset,ldb) for offset in (0,256) for ldb in (3072,3088)})
        ignored={'id','group','b_offset_bytes','ldb'}
        fixed=[{k:v for k,v in r.items() if k not in ignored} for r in controls]
        self.assertTrue(all(r==fixed[0] for r in fixed))
        self.assertTrue(all(r['b_allocation_extra_bytes']==256 and r['alloc_ldb']==3136 for r in controls))

    def test_static_service_features_preserve_256B_shift_and_track_low_bits(self):
        row=dict(id='static',config='cfg_b',m=2304,n=3072,k=1024,lda=1024,ldb=3072,
            storage_m=2304,storage_n=3072,swizzle=1,raster_order='along_m')
        setup=dict(grid=[132,1,1],stages=6,tile=[128,128,64],cluster=[1,1,1])
        original=supply_request_rows(row,setup)
        self.assertEqual(original,supply_request_rows(dict(row,b_offset_bytes=0),setup))
        self.assertEqual(original,supply_request_rows(dict(row,b_offset_bytes=256),setup))
        shifted=supply_request_rows(dict(row,b_offset_bytes=16),setup)
        self.assertNotEqual(original[0]['X'],shifted[0]['X'])
        self.assertEqual(original[0]['X'][0],shifted[0]['X'][0])

    def test_base_offset_public_validation_rejects_pointer_and_capacity_changes(self):
        from test_r09_history_series import synthetic
        from analyze_r18 import validate_warmup
        for offset in (0,256):
            row,setup,call,_=synthetic();row.update(alloc_ldb=128,b_offset_bytes=offset,b_allocation_extra_bytes=256)
            payload=row['k']*row['alloc_ldb']
            for event in (setup,call):
                event.update(b_offset_bytes=offset,b_allocation_extra_bytes=256,b_payload_elements=payload,
                    allocated_b_elements=payload+128,b_buffer_ptr=1048576,target_b_ptr=1048576+offset)
            validate_warmup(setup,call,row)
            bad=deepcopy(call);bad['target_b_ptr']+=16
            with self.assertRaisesRegex(ValueError,'declared offset'):validate_warmup(setup,bad,row)
            bad=deepcopy(setup);bad['allocated_b_elements']-=128
            with self.assertRaisesRegex(ValueError,'common extra capacity'):validate_warmup(bad,call,row)
            bad=deepcopy(call);bad['b_buffer_ptr']+=256;bad['target_b_ptr']+=256
            with self.assertRaisesRegex(ValueError,'within process'):validate_warmup(setup,bad,row)

    def test_base_shift_neighbor_mapping_never_wraps_M_or_hides_context_change(self):
        def row(cta,mi,ni,j=0):
            return dict(cta=cta,mi=mi,ni=ni,j=j,T=1,active_ctas=4,A_peer=1,B_peer=1,
                base_sm=cta,variant_sm=cta,L_cycles_delta=cta*10)
        reference=[row(0,0,0),row(1,0,1),row(2,1,0),row(3,1,1,1)]
        shifted=deepcopy(reference)
        groups=_base_shift_response_pairs(reference,shifted)
        self.assertEqual([(a['cta'],b['cta']) for a,b in groups['next_B_panel']],[(1,0),(3,2)])
        self.assertEqual([(a['cta'],b['cta']) for a,b in groups['next_B_panel_matched_context']],[(1,0)])
        self.assertEqual([b['cta'] for a,b in groups['same_work_on_neighbor_subset']],[0])
        self.assertEqual(_base_shift_pattern_metrics(groups['same_work'],'L_cycles_delta')['maximum_absolute_residual'],0)
        for r in reference+shifted:r['base_sm']=r['variant_sm']=0
        self.assertEqual(_base_shift_response_pairs(reference,shifted)['same_SM_and_ordinal'],[(reference[3],shifted[3])])
        self.assertIsNone(_base_shift_pattern_metrics([],'L_cycles_delta')['correlation'])

    def test_work_swap_preserves_wave_sets_and_only_changes_two_sequences(self):
        work=v08_model.scheduled_work('cfg_b',2304,3072,[132,1,1],1,'along_m')
        swapped=_work_swap_expected(work,True)
        self.assertEqual(swapped[55],work[91]);self.assertEqual(swapped[91],work[55])
        self.assertEqual([p[0] for p in work[55]],[p[0] for p in work[91]])
        self.assertEqual(len(work[55]),3);self.assertEqual(len(work[91]),3)
        self.assertEqual(sum(a==b for a,b in zip(work,swapped)),130)
        self.assertEqual(_work_swap_expected(swapped,True),work)
        for j in range(4):
            self.assertEqual(sorted(c[j] for c in work if len(c)>j),sorted(c[j] for c in swapped if len(c)>j))

    def test_work_swap_config_and_public_flag_validation(self):
        from test_r09_history_series import synthetic
        from analyze_r18 import validate_warmup
        rows=json.loads((Path(__file__).parent/'configs/r10-work-swap.json').read_text())
        controls=[r for r in rows if r['set']=='ctrl']
        self.assertEqual(len(rows),6);self.assertTrue(all(r['variants']==['plain'] for r in rows[:2]))
        self.assertEqual({(r['work_swap55_91'],r['ldb']) for r in controls},
            {(s,p) for s in (0,1) for p in (3072,3088)})
        ignored={'id','group','work_swap55_91','ldb'}
        fixed=[{k:v for k,v in r.items() if k not in ignored} for r in controls]
        self.assertTrue(all(v==fixed[0] for v in fixed))
        self.assertTrue(all(r['b_offset_bytes']==0 and r['b_allocation_extra_bytes']==256 for r in controls))
        for flag in (0,1):
            row,setup,call,_=synthetic();row['work_swap55_91']=setup['work_swap55_91']=call['work_swap55_91']=flag
            validate_warmup(setup,call,row)
            with self.assertRaisesRegex(ValueError,'permutation identity'):
                validate_warmup(setup,dict(call,work_swap55_91=1-flag),row)

    def test_work_swap_physical_and_logical_correspondence_are_distinct(self):
        original=[dict(cta=55,j=0,T=3,mi=1,ni=3,L_cycles_delta=100),
            dict(cta=91,j=0,T=3,mi=1,ni=5,L_cycles_delta=10)]
        swapped=[dict(original[1],cta=55),dict(original[0],cta=91)]
        pairs=_work_swap_correspondence(original,swapped)
        self.assertEqual([(a['cta'],b['cta']) for a,b in pairs['physical_CTA']],[(55,55),(91,91)])
        self.assertEqual([(a['cta'],b['cta']) for a,b in pairs['logical_work']],[(91,55),(55,91)])
        self.assertEqual(_base_shift_pattern_metrics(pairs['logical_work'],'L_cycles_delta')['maximum_absolute_residual'],0)
        self.assertEqual(_base_shift_pattern_metrics(pairs['physical_CTA'],'L_cycles_delta')['maximum_absolute_residual'],90)

    def test_work_swap_endpoints_use_call_origins_and_close_with_L(self):
        def cta(index,shift=0,prefix=0,extra_L=0):
            entry=100+20*index+shift;first=entry+50+prefix;end=first+100+extra_L
            return dict(cta=index,sm=index,entry_ns=entry,entry_c=2*entry,prod_ns=entry+10,end_ns=end+30,
                work=[[index,0]],tiles_ns=[[first,end,end+5,end+20]],tiles=[[2*first,2*end,2*end+10,2*end+40]])
        base=dict(ctas=[cta(0),cta(1)])
        variant=dict(ctas=[cta(0,shift=1000,extra_L=40),cta(1,shift=1000,prefix=20)])
        windows=_work_swap_pitch_windows(base,variant)
        self.assertEqual([w['entry_relative_ns_delta'] for w in windows],[0,0])
        self.assertEqual([w['FIRST_relative_ns_delta'] for w in windows],[0,20])
        self.assertEqual([w['L_cycles_delta'] for w in windows],[80,0])
        self.assertEqual([w['entry_to_MAIN_END_ns_delta'] for w in windows],[40,20])
        one=_work_swap_pitch_windows(base,variant,exclude_ctas=(0,))[0]
        self.assertEqual(one['entry_relative_ns_base'],20)


if __name__=='__main__':unittest.main()

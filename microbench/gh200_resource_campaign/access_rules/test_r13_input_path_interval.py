"""Initial source expiry, exact queue envelopes and finite-stage bound failures."""
from fractions import Fraction as F
import unittest

from analyze_r10_input_path_interval import finite_stage_bounds,initial_destinations,initial_sources
from analyze_r10_shared_path_prefix import value
from supply_model import shared_stage_events


class InputPathIntervalTests(unittest.TestCase):
    def program(self):
        keys=[[[(atom,i) for atom in ('A','B0','B1')] for i in range(4)] for _ in range(3)]
        return dict(source_keys=keys,output_counts=[1]*3)

    def test_initial_source_expiry_matches_existing_graph(self):
        program=self.program()
        for latency,producer in ((F(1),F(0)),(F(17),F(0)),(F(17),F(3))):
            with self.subTest(latency=latency,producer=producer):
                # Distinct boxes share one unit. At latency=1 the next atom
                # arrives exactly at expiry and must pay for a new generation.
                units={key:('shared-unit',) for groups in program['source_keys'] for group in groups for key in group}
                rows,_=initial_sources(program,2,units,F(2),latency,F(3),producer)
                numeric=shared_stage_events(4,2,3,service_ns=2.,read_latency_ns=float(latency),
                    compute_ns=64.,operand_gap_ns=3.,producer_gap_ns=float(producer),source_keys=program['source_keys'],output_counts=[1]*3,
                    share_inflight_sources=True,source_address_units=units,source_unit_service_ns=2.,trace_requests=True)
                actual=[d for d in numeric['input_demands'] if d['index']<2]
                self.assertEqual(len(rows),len(actual))
                for row,demand in zip(rows,actual):
                    self.assertEqual(float(row['issue']),demand['arrival_ns'])
                    self.assertEqual(float(row['source_ready']),demand['source_ready_ns'])
                if latency==1:
                    self.assertEqual([rows[i]['source_ready'] for i in (0,3,6)],[F(3),F(6),F(9)])
                self.assertEqual(rows[9]['issue'],6+producer)

    def test_affine_envelopes_match_direct_rational_fifos(self):
        program=self.program()
        units={key:(key,) for groups in program['source_keys'] for group in groups for key in group}
        rows,_=initial_sources(program,2,units,F(2),F(17),F(3))
        full,_=initial_destinations(rows,3,F(17),F(4))
        for price in (F(0),F(1,8),F(2,3),F(1),F(7,5),F(8)):
            frontier=F(0);local=[F(0)]*3
            for row in rows:
                frontier=max(frontier,row['issue'])+price*row['work']
                arrival=max(row['source_ready'],frontier+17);cta=row['cta']
                local[cta]=max(local[cta],arrival)+4*row['work']
                if row['atom']==2:self.assertEqual(value(full[cta,row['group']],price),local[cta])

    def test_initial_ready_order_must_be_proved(self):
        rows=[dict(cta=0,group=0,atom=0,issue=F(0),source_ready=F(10),work=F(1)),
              dict(cta=0,group=0,atom=1,issue=F(1),source_ready=F(9),work=F(1,2))]
        with self.assertRaisesRegex(ValueError,'completion order'):initial_destinations(rows,1,F(2),F(1))

    def test_conservative_bound_retains_capacity_and_subscription_failures(self):
        args=dict(stages=6,kt=16,compute=F(288),lead=F(144),operand=F(10),latency=F(754),
            destination=F(72),first_min=F(970),first_max=F(1060),source_work=F(190),path_work=F(230),
            initial_source_end=F(1200),initial_path_end=F(1380),initial_last_ready=F(2300))
        passing=finite_stage_bounds(**args)
        self.assertGreater(passing['refill_TC_slack_ns'],0)
        self.assertTrue(all(v>0 for v in passing.values()))
        overloaded=finite_stage_bounds(**dict(args,path_work=F(350)))
        self.assertLess(overloaded['request_capacity_headroom_ns'],0)
        self.assertLess(overloaded['refill_TC_slack_ns'],0)
        expired=finite_stage_bounds(**dict(args,first_max=F(1800)))
        self.assertLess(expired['cohort_span_below_subscription_ns'],0)


if __name__=='__main__':unittest.main()

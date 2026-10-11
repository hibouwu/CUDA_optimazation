from copy import deepcopy
import unittest

from r09_shared_clock import clock_graph,minimum_shared_slack,solve_clock_graph,verify_certificate,verify_shared_witness


class SharedClockConsistency(unittest.TestCase):
    def test_independent_huge_counter_epochs_share_a_unit_rate(self):
        timer=10**18
        points={3:[(10**18,timer),(10**18+100,timer+100)],
                7:[(5*10**18,timer+30),(5*10**18+80,timer+110)]}
        result=minimum_shared_slack(points,1)
        self.assertEqual(result['minimum_integer_endpoint_slack_ns'],0)
        self.assertEqual(result['original_sample_verification']['points_verified'],4)
        self.assertEqual(result['original_sample_verification']['maximum_constructed_endpoint_error_ns'],0)

    def test_local_cap_has_analytic_endpoint_minimum(self):
        points={0:[(0,0),(200,100)]}
        result=minimum_shared_slack(points,1)
        self.assertEqual(result['minimum_integer_endpoint_slack_ns'],50)
        self.assertEqual(result['real_minimum_bracket_ns'],[49,50])
        self.assertFalse(result['infeasible_certificate']['feasible'])
        self.assertEqual(result['original_sample_verification']['maximum_constructed_endpoint_error_ns'],50)

    def test_nested_windows_have_a_proved_negative_cycle(self):
        points={0:[(0,0),(100,100)],1:[(0,25),(200,75)]}
        graph=clock_graph(points,0,10);proof=solve_clock_graph(graph)
        self.assertFalse(proof['feasible'])
        self.assertLess(proof['negative_cycle_bound_scaled_cycles'],0)
        self.assertTrue(verify_certificate(graph,proof))
        result=minimum_shared_slack(points,10)
        self.assertGreater(result['minimum_integer_endpoint_slack_ns'],12)
        self.assertTrue(verify_shared_witness(points,result['feasible_graph'],result['feasible_certificate']))

    def test_zero_duration_different_cycles_requires_positive_slack(self):
        result=minimum_shared_slack({0:[(0,100),(3,100)]},2)
        self.assertEqual(result['minimum_integer_endpoint_slack_ns'],1)
        self.assertTrue(result['real_lower_bound_open'])

    def test_common_rate_rejection_is_separate_from_declared_cap(self):
        self.assertEqual(minimum_shared_slack({0:[(0,0),(200,100)]},None)['minimum_integer_endpoint_slack_ns'],0)
        nested={0:[(0,0),(100,100)],1:[(0,25),(200,75)]}
        self.assertEqual(minimum_shared_slack(nested,None)['minimum_integer_endpoint_slack_ns'],13)

    def test_shifted_sm_epochs_and_global_time_leave_result_unchanged(self):
        points={0:[(0,0),(100,100)],1:[(0,25),(200,75)]}
        shifted={sm:[(c+10**18*(sm+1),n+10**18) for c,n in values] for sm,values in points.items()}
        a,b=minimum_shared_slack(points,10),minimum_shared_slack(shifted,10)
        self.assertEqual(a['minimum_integer_endpoint_slack_ns'],b['minimum_integer_endpoint_slack_ns'])
        self.assertEqual(a['feasible_certificate'],b['feasible_certificate'])

    def test_tampered_certificate_does_not_pass(self):
        points={0:[(0,0),(100,100)],1:[(0,25),(200,75)]}
        result=minimum_shared_slack(points,10)
        wrong=deepcopy(result['feasible_certificate']);wrong['potential_scaled_cycles'][0]+=1000000
        with self.assertRaises(ValueError):verify_certificate(result['feasible_graph'],wrong)
        wrong=deepcopy(result['infeasible_certificate']);wrong['negative_cycle_edge_ids']=[]
        with self.assertRaises(ValueError):verify_certificate(result['infeasible_graph'],wrong)

    def test_invalid_noninteger_or_empty_inputs_fail(self):
        for points,epsilon in [({},0),({0:[]},0),({0:[(0.,0)]},0),({0:[(0,0)]},.5),({True:[(0,0)]},0)]:
            with self.subTest(points=points,epsilon=epsilon),self.assertRaises(ValueError):clock_graph(points,epsilon)
        with self.assertRaises(ValueError):clock_graph({0:[(0,0)]},0,0)


if __name__=='__main__':
    unittest.main()

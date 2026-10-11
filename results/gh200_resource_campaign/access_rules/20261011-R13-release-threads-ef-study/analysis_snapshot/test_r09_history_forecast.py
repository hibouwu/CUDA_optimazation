import unittest

from r09_history_forecast import causal_event_forecasts, role_forecast


class HistoryForecast(unittest.TestCase):
    def test_target_and_future_changes_do_not_change_current_prediction(self):
        a=causal_event_forecasts([5,6],[7,8,9],'previous_call')
        b=causal_event_forecasts([5,6],[7,800,900],'previous_call')
        self.assertEqual(a,[6,7,8])
        self.assertEqual(a[:2],b[:2])
        self.assertNotEqual(a[2],b[2])

    def test_held_prefix_never_uses_target_observations(self):
        self.assertEqual(causal_event_forecasts([5,6],[7,8,9],'hold_prefix'),[6,6,6])
        self.assertEqual(causal_event_forecasts([5,6],[70,80,90],'hold_prefix'),[6,6,6])

    def test_opposite_component_errors_are_retained(self):
        result=role_forecast((100,100),(200,100))
        self.assertEqual(result['cycle_forecast_error'],-.5)
        self.assertEqual(result['conditional_conversion_error'],1)
        self.assertEqual(result['free_role_window_error'],0)
        self.assertEqual(result['closure'],0)

    def test_invalid_history_or_native_scope_fails(self):
        for args in [([], [1], 'previous_call'),([1],[0],'previous_call'),([1],[2],'unknown')]:
            with self.subTest(args=args),self.assertRaises(ValueError):causal_event_forecasts(*args)
        with self.assertRaises(ValueError):role_forecast((0,1),(1,1))


if __name__=='__main__':unittest.main()

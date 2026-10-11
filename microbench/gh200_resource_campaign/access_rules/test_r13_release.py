"""Conservative last-arrival intervals from all four warp brackets."""
import unittest

from analyze_r13_release import release_arrival_bounds


class ReleasePrerequisiteTests(unittest.TestCase):
    def test_late_warp_and_post_arrival_paths_are_distinct(self):
        late=release_arrival_bounds([100,110,120,270],[120,130,140,290],310)
        self.assertEqual(late['other_warp_skew'],dict(lower=150,upper=190))
        self.assertEqual(late['after_last_emit_to_acquire_end'],dict(lower=20,upper=40))
        post=release_arrival_bounds([100,103,106,110],[120,123,126,130],310)
        self.assertEqual(post['other_warp_skew'],dict(lower=0,upper=30))
        self.assertEqual(post['after_last_emit_to_acquire_end'],dict(lower=180,upper=200))

    def test_post_clock_can_follow_producer_return(self):
        result=release_arrival_bounds([100,110,120,140],[120,130,160,170],150)
        self.assertEqual(result['last_arrival_emit'],dict(lower=140,upper=150))
        self.assertEqual(result['after_last_emit_to_acquire_end'],dict(lower=0,upper=10))

    def test_interval_contains_all_feasible_emission_points(self):
        before=[1,4,7,10];after=[4,7,10,13];end=15
        bounds=release_arrival_bounds(before,after,end)
        for e0 in range(1,5):
            for e1 in range(4,8):
                for e2 in range(7,11):
                    for e3 in range(10,14):
                        last=max(e0,e1,e2,e3)
                        for name,value in [('last_arrival_emit',last),('other_warp_skew',last-e0),
                            ('after_last_emit_to_acquire_end',end-last)]:
                            self.assertLessEqual(bounds[name]['lower'],value)
                            self.assertGreaterEqual(bounds[name]['upper'],value)

    def test_incomplete_or_causally_invalid_records_are_refused(self):
        with self.assertRaisesRegex(ValueError,'four'):
            release_arrival_bounds([1],[2],3)
        with self.assertRaisesRegex(ValueError,'four'):
            release_arrival_bounds([1,2,3,4],[2,3,4,3],9)
        with self.assertRaisesRegex(ValueError,'returned'):
            release_arrival_bounds([1,2,3,8],[2,3,4,9],7)


if __name__=='__main__':
    unittest.main()

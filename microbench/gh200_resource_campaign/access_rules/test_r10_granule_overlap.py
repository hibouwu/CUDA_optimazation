"""Logical overlap and strict fixed-schedule subscriptions, not physical traffic."""
import unittest

from analyze_r10_granule_overlap import source_granules, replay_granule_joins


def box(op='B',n=0,pitch=320,offset=0):
    return (op,'fp16','SW128',(128,64),(2,pitch),offset,(n,0),(64,64))


class GranuleOverlapTests(unittest.TestCase):
    def test_adjacent_boxes_overlap_at128_but_not32(self):
        left,right = box(),box(n=64)
        self.assertEqual(len(set(source_granules(left)) & set(source_granules(right))),32)
        self.assertEqual(len(source_granules(left)),96)
        self.assertEqual(len(set(source_granules(left,32)) & set(source_granules(right,32))),0)
        self.assertEqual(len(source_granules(left,32)),256)

    def test_aligned_boxes_do_not_overlap(self):
        self.assertFalse(set(source_granules(box(pitch=256))) & set(source_granules(box(n=64,pitch=256))))
        self.assertEqual(len(source_granules(box(pitch=256))),64)

    def test_translation_changes_identity_not_coverage(self):
        units = source_granules(box())
        self.assertEqual(source_granules(box(offset=128)),tuple(i+1 for i in units))

    def test_valid_rectangle_and_OOB_do_not_create_source_addresses(self):
        key = list(box())
        key[3] = (32,16)
        self.assertLess(len(source_granules(tuple(key))),len(source_granules(box())))
        key[6] = (64,0)
        self.assertEqual(source_granules(tuple(key)),())

    def test_strict_subscription_closes_at_ready(self):
        first = dict(source_key=box(),index=0,arrival_ns=0.,ready_ns=10.)
        second = dict(source_key=box(n=64),index=0,arrival_ns=9.,ready_ns=20.)
        result = replay_granule_joins([first,second],16)[0]
        self.assertEqual(result['additional_join_units'],32)
        self.assertEqual(result['coverage_units'],192)
        self.assertEqual(result['granule_generations'],160)
        second['arrival_ns'] = 10.
        self.assertEqual(replay_granule_joins([first,second],16)[0]['additional_join_units'],0)

    def test_different_allocations_do_not_alias(self):
        a = ('A','fp16','SW128',(128,64),(128,2),0,(0,0),(128,64))
        b = ('B','fp16','SW128',(64,64),(2,128),0,(0,0),(64,64))
        rows = replay_granule_joins([dict(source_key=a,index=0,arrival_ns=0.,ready_ns=10.),
            dict(source_key=b,index=0,arrival_ns=1.,ready_ns=11.)],16)
        self.assertTrue(all(row['additional_join_units']==0 for row in rows))

    def test_invalid_scope_and_model_times_are_rejected(self):
        with self.assertRaises(ValueError):source_granules(box(),64)
        key = list(box());key[7] = (128,64)
        with self.assertRaises(ValueError):source_granules(tuple(key))
        for end in (-1.,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):
                replay_granule_joins([dict(source_key=box(),index=0,arrival_ns=0.,ready_ns=end)],16)


if __name__=='__main__':
    unittest.main()

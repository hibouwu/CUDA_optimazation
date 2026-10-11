"""Tail work stays full; releases and output boundaries retain their real owners."""
import unittest
from collections import Counter

from analyze_r10_boundary_lifecycle import mma_atom_work,stage_lifecycle


class BoundaryLifecycleTests(unittest.TestCase):
    def test_last_K16_has_full_atom_work_and_six_zero_K_atoms(self):
        atoms=mma_atom_work((2600,3000,2000),(12,12),31)
        self.assertEqual(len(atoms),8)
        self.assertEqual(sum(a['nominal_FLOP'] for a in atoms),2097152)
        self.assertEqual(sum(a['valid_math_FLOP'] for a in atoms),2*128*128*16)
        self.assertEqual(sum(a['valid_shape'][2]==0 for a in atoms),6)

    def test_MNK_corner_keeps_eight_calls_with_one_valid_math_atom(self):
        atoms=mma_atom_work((2600,3000,2000),(20,23),31)
        self.assertEqual(sum(a['valid_math_FLOP']>0 for a in atoms),1)
        self.assertEqual(sum(a['valid_math_FLOP'] for a in atoms),2*40*56*16)
        self.assertEqual(sum(a['nominal_FLOP'] for a in atoms),2097152)

    def test_valid_math_atom_partition_covers_the_logical_problem(self):
        shape=(136,136,72)
        valid=sum(a['valid_math_FLOP'] for mi in range(2) for ni in range(2) for q in range(2)
            for a in mma_atom_work(shape,(mi,ni),q))
        self.assertEqual(valid,2*136*136*72)

    def test_output_boundary_only_sixth_next_acquire_requires_tail(self):
        graph=stage_lifecycle(32,6,4)
        self.assertEqual(Counter(g['release_path'] for g in graph['groups']),{'normal_wait1':124,'mma_tail_wait0':4})
        for b in graph['output_boundaries']:
            self.assertEqual([g['release_path'] for g in b['first_next_output_acquires']],['normal_wait1']*5+['mma_tail_wait0'])
        self.assertEqual(graph['groups'][32]['consumer_wg'],1)
        self.assertEqual(graph['groups'][32]['previous_group'],26)
        self.assertEqual(graph['groups'][37]['previous_group'],31)
        self.assertEqual(graph['groups'][31]['empty_arrivals_required'],128)
        self.assertEqual(graph['outputs'][2]['mma_return_requires_normal_release_group'],94)
        self.assertEqual(graph['outputs'][2]['last_group'],95)

    def test_producer_tail_visits_last_generations_or_initial_unused_slots(self):
        self.assertEqual([x['release_group'] for x in stage_lifecycle(32,6,3)['producer_tail_waits']],list(range(90,96)))
        short=stage_lifecycle(1,6,1)
        self.assertEqual([x['release_group'] for x in short['producer_tail_waits']],[None]*5+[0])
        self.assertEqual(short['outputs'][0]['normal_wait1_calls_per_consumer_thread'],0)
        self.assertEqual(short['groups'][0]['release_path'],'mma_tail_wait0')
        for graph in (stage_lifecycle(32,6,3),short):
            certificate=graph['input_exit_certificate']
            covered=certificate['releases_required_by_later_acquire']+certificate['releases_required_by_tail']
            self.assertEqual(sorted(covered),list(range(len(graph['groups']))))
            self.assertEqual(len(set(covered)),len(covered))

    def test_each_group_has_one_release_owner_and_same_slot_reuse(self):
        graph=stage_lifecycle(16,4,3)
        for g in graph['groups']:
            self.assertEqual(g['consumer_wg'],g['output_ordinal']%2)
            if g['previous_group'] is not None:
                previous=graph['groups'][g['previous_group']]
                self.assertEqual(g['slot'],previous['slot'])
                self.assertEqual(g['generation'],previous['generation']+1)
                self.assertNotEqual(g['empty_wait_phase'],previous['empty_wait_phase'])


if __name__=='__main__':unittest.main()

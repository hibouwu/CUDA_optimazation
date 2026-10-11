"""Zero-source boxes keep completion obligations and equal sources stay separate."""
import copy
import unittest

from analyze_r10_boundary_completion import completion_fragment,work_identity_projection
from supply_model import cfg_b_persistent_boundary_ledger


class BoundaryCompletionTests(unittest.TestCase):
    def ledger(self,ldb=136):
        row=dict(config='cfg_b',m=136,n=136,k=72,lda=72,ldb=ldb,ldd=136,swizzle=1,raster_order='heuristic')
        setup=dict(tile=[128,128,64],cluster=[1,1,1],grid=[1,1,1],stages=3)
        return cfg_b_persistent_boundary_ledger(row,setup)

    def test_zero_source_B1_keeps_submission_fill_and_full_predecessor(self):
        out=next(o for o in self.ledger()['outputs'] if o['coord']==(1,1))
        graph=completion_fragment(out,out['groups'][1],3);b1=graph['atoms'][2]
        self.assertEqual((b1['source_valid_bytes'],b1['destination_fill_bytes']),(0,8192))
        self.assertIsNone(b1['valid_available']);self.assertIsNone(b1['valid_completion'])
        self.assertEqual(graph['nodes'][b1['fill_completion']],[b1['submit']])
        self.assertEqual(graph['nodes'][b1['destination_completion']],[b1['fill_completion']])
        self.assertIn(b1['destination_completion'],graph['nodes'][graph['full_output_port']])
        self.assertIsNone(graph['completion_values'])

    def test_partial_atom_requires_valid_delivery_and_fill_completion(self):
        out=next(o for o in self.ledger()['outputs'] if o['coord']==(0,0))
        graph=completion_fragment(out,out['groups'][1],3)
        for atom in graph['atoms']:
            self.assertEqual(graph['nodes'][atom['destination_completion']],[atom['valid_completion'],atom['fill_completion']])
        self.assertEqual(sum(a['transaction_bytes'] for a in graph['atoms']),32768)

    def test_full_atoms_have_no_fill_branch_and_B_order_is_unbound(self):
        out=next(o for o in self.ledger()['outputs'] if o['coord']==(0,0))
        graph=completion_fragment(out,out['groups'][0],3);a,b0,b1=graph['atoms']
        self.assertTrue(all(x['fill_completion'] is None for x in graph['atoms']))
        self.assertEqual(graph['nodes'][b0['submit']],[a['submit']])
        self.assertEqual(graph['nodes'][b1['submit']],[a['submit']])

    def test_equal_source_identity_does_not_merge_demand_completion_nodes(self):
        ledger=self.ledger();first=next(o for o in ledger['outputs'] if o['coord']==(0,0))
        second=next(o for o in ledger['outputs'] if o['coord']==(0,1))
        a=completion_fragment(first,first['groups'][0],3)
        b=completion_fragment(second,second['groups'][0],3)
        self.assertEqual(a['atoms'][0]['source_key'],b['atoms'][0]['source_key'])
        self.assertTrue(set(a['nodes']).isdisjoint(b['nodes']))

    def test_pitch_changes_addresses_but_not_work_and_source_equality_projection(self):
        projections=[];sources=[]
        for ldb in (136,144):
            labels={};view=[];keys=[]
            for out in self.ledger(ldb)['outputs']:
                for group in out['groups']:
                    fragment=completion_fragment(out,group,3)
                    view.append(work_identity_projection(fragment,labels))
                    keys.extend(a['source_key'] for a in fragment['atoms'])
            projections.append(view);sources.append(keys)
        self.assertEqual(projections[0],projections[1])
        self.assertNotEqual(sources[0],sources[1])

    def test_rejects_lost_destination_budget_and_wrong_ring_generation(self):
        out=next(o for o in self.ledger()['outputs'] if o['coord']==(1,1))
        for name in ('transaction_bytes','destination_required','slot'):
            group=copy.deepcopy(out['groups'][1])
            if name=='transaction_bytes':group['atoms'][2][name]=0
            elif name=='destination_required':group['atoms'][2][name]=False
            else:group['slot_state'][name]=(group['slot_state'][name]+1)%3
            with self.subTest(name=name):
                with self.assertRaises(ValueError):completion_fragment(out,group,3)


if __name__=='__main__':unittest.main()

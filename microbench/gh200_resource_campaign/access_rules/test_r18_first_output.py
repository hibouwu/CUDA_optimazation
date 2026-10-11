import copy
import unittest

from analyze_r18_first_output import EVENTS, first_output_windows


def packet():
    # Synthetic values exceed binary64's exact integer range on purpose.
    origin=2**60
    return dict(invocation_id='synthetic-only',config='cfg_a',cta=6,role=2,
        physical_thread=256,tile_seq=0,sm_start=37,sm_end=37,
        stamps={event:dict(clock64=origin+10*i,globaltimer_ns=origin+16*i)
                for i,event in enumerate(EVENTS)})


class FirstOutputWindowsTest(unittest.TestCase):
    def test_exact_native_integer_partition(self):
        result=first_output_windows(packet())
        self.assertEqual(result['L'],dict(cycles=60,ns=96))
        self.assertEqual([w['ns'] for w in result['windows']],[16]*6)
        self.assertEqual([w['cycles'] for w in result['windows']],[10]*6)

    def test_equal_quantized_ns_endpoints_are_allowed(self):
        record=packet()
        record['stamps']['COMMIT_G0']['globaltimer_ns']=record['stamps']['FIRST']['globaltimer_ns']
        self.assertEqual(first_output_windows(record)['windows'][0]['ns'],0)

    def test_rejects_wrong_control_flow_or_missing_point(self):
        for change in ('reverse','missing','extra'):
            record=packet()
            if change=='reverse':record['stamps']['MMA_RETURN']['clock64']=1
            elif change=='missing':record['stamps'].pop('MMA_RETURN')
            else:record['stamps']['COMMIT_G1']=copy.deepcopy(record['stamps']['COMMIT_G0'])
            with self.assertRaises(ValueError):first_output_windows(record)

    def test_rejects_unobserved_and_noninteger_stamps(self):
        for value in (0,-1,True,1.5,float('nan'),float('inf'),2**64):
            record=packet();record['stamps']['COMMIT_G16']['globaltimer_ns']=value
            with self.assertRaises(ValueError):first_output_windows(record)

    def test_rejects_unbound_identity(self):
        mutations=[('invocation_id',''),('config','cfg_b'),('cta',2),('role',0),
            ('physical_thread',296),('tile_seq',1),('sm_end',38),('sm_start',-1),('role',True)]
        for field,value in mutations:
            record=packet();record[field]=value
            with self.assertRaises(ValueError):first_output_windows(record)

    def test_signed_paired_deltas_telescope_without_cross_call_clocks(self):
        reference=packet();target=packet()
        # Different invocation origins and both positive and negative segment responses.
        target['invocation_id']='another-synthetic-invocation'
        offsets=(0,8,-8,24,8,40,32)
        for index,event in enumerate(EVENTS):
            target['stamps'][event]['globaltimer_ns']+=4096+offsets[index]
        left=first_output_windows(target);right=first_output_windows(reference)
        delta=[a['ns']-b['ns'] for a,b in zip(left['windows'],right['windows'])]
        self.assertTrue(any(value<0 for value in delta))
        self.assertEqual(sum(delta),left['L']['ns']-right['L']['ns'])
        self.assertEqual(sum(delta),32)


if __name__=='__main__':unittest.main()

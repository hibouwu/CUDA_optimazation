import copy
import unittest

from analyze_r18_first_output import (EVENTS, SELECTED_CTAS, first_output_windows,
                                      first_output_call_windows, paired_first_output_windows)


def packet():
    # Synthetic values exceed binary64's exact integer range on purpose.
    origin=2**60
    return dict(invocation_id='synthetic-only',config='cfg_a',cta=6,role=2,
        physical_thread=256,tile_seq=0,sm_start=37,sm_end=37,
        stamps={event:dict(clock64=origin+10*i,globaltimer_ns=origin+16*i)
                for i,event in enumerate(EVENTS)})


def packets(invocation):
    records=[]
    for cta in SELECTED_CTAS:
        for role in (1,2):
            record=packet()
            record.update(invocation_id=invocation,cta=cta,role=role,physical_thread=128*role,
                          sm_start=cta+5,sm_end=cta+5)
            for stamp in record['stamps'].values():
                stamp['clock64']+=100000*cta
            records.append(record)
    return records


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

    def test_complete_call_retains_all_static_clusters_and_both_roles(self):
        records=list(reversed(packets('one-call')))
        result=first_output_call_windows(records)
        self.assertEqual([(row['cta'],row['role']) for row in result['records']],
                         [(cta,role) for cta in SELECTED_CTAS for role in (1,2)])

    def test_call_rejects_missing_duplicate_mixed_invocation_or_role_sm(self):
        for change in ('missing','duplicate','mixed','sm'):
            records=packets('one-call')
            if change=='missing':records=[record for record in records if record['cta'] not in (6,7)]
            elif change=='duplicate':records.append(copy.deepcopy(records[0]))
            elif change=='mixed':records[-1]['invocation_id']='another-call'
            else:records[1].update(sm_start=99,sm_end=99)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):first_output_call_windows(records)

    def test_pair_keeps_signed_windows_and_distinct_native_origins(self):
        target=packets('map960-call');reference=packets('map1024-call')
        offsets=(0,16,0,32,16,48,32)
        for record in target:
            for index,event in enumerate(EVENTS):
                record['stamps'][event]['clock64']+=10**9
                record['stamps'][event]['globaltimer_ns']+=10**6+offsets[index]
        result=paired_first_output_windows(list(reversed(target)),reference)
        self.assertEqual(len(result['records']),24)
        for row in result['records']:
            self.assertEqual(row['delta_L'],dict(cycles=0,ns=32))
            self.assertEqual([window['delta']['ns'] for window in row['windows']],[16,-16,32,-16,32,-16])
        reversed_pair=paired_first_output_windows(reference,target)
        self.assertTrue(all(row['delta_L']['ns']==-32 for row in reversed_pair['records']))

    def test_pair_rejects_reusing_one_invocation_or_incomplete_reference(self):
        with self.assertRaisesRegex(ValueError,'distinct'):
            paired_first_output_windows(packets('same'),packets('same'))
        with self.assertRaisesRegex(ValueError,'coverage'):
            paired_first_output_windows(packets('target'),packets('reference')[:-1])

    def test_pair_retains_cross_call_sm_changes_without_comparing_native_origins(self):
        target=packets('target');reference=packets('reference')
        for record in target[:2]:record.update(sm_start=99,sm_end=99)
        result=paired_first_output_windows(target,reference)
        self.assertTrue(result['records'][0]['sm_changed'])
        self.assertEqual(result['records'][0]['map960_sm'],99)
        self.assertEqual(result['records'][0]['map1024_sm'],5)
        self.assertEqual(result['records'][0]['delta_L'],dict(cycles=0,ns=0))


if __name__=='__main__':unittest.main()

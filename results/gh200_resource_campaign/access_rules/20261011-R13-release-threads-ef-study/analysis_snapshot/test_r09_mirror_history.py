"""The mirror intervention holds work fixed and retains the whole target trajectory."""
import copy,json,tempfile
from pathlib import Path
import unittest

import run_r18
from r09_input_source_clock import mirror_pair_metrics,mirror_nvml_snapshots


def history():
    calls=[]
    for j in range(35):
        calls.append(dict(event_us=.2,window_us=.1,
            role_windows=[[0,0,12,0,200,100],[0,1,12,0,200,100]]))
    return dict(calls=calls,prefix_event_us=16*.2,prefix_kernel_sum_us=16*.2)


class MirrorHistory(unittest.TestCase):
    def test_manifest_has_fixed_slots_and_only_source_identity_changes(self):
        rows=json.loads((Path(__file__).parent/'configs/r09-mirror-history.json').read_text())
        self.assertEqual(len(rows),4)
        self.assertEqual({(r['history_source'],r['mirror_target_slot']) for r in rows},
            {(s,t) for s in ('same','mirror') for t in (0,1)})
        changed={'id','group','history_source','mirror_target_slot'}
        fixed=[{k:v for k,v in r.items() if k not in changed} for r in rows]
        self.assertTrue(all(r==fixed[0] for r in fixed))
        self.assertEqual(fixed[0]['swizzle'],1)
        self.assertEqual(fixed[0]['history_mode'],'series')
        self.assertEqual(fixed[0]['variants'],['ends'])
        for row in rows:
            args=run_r18.case_args(row)
            self.assertEqual(args[args.index('--history-source')+1],row['history_source'])
            self.assertEqual(args[args.index('--mirror-target-slot')+1],str(row['mirror_target_slot']))
        for slot in (0,1):
            pair=[r for r in rows if r['mirror_target_slot']==slot]
            self.assertEqual(len({r['group'] for r in pair}),1)
            self.assertEqual(sum(len(run_r18.trial_rows(pair,t)) for t in range(10)),20)

    def test_initial_cold_difference_is_separate_from_target_catchup(self):
        same=history();mirror=copy.deepcopy(same)
        for j in range(16,29):
            ns=120 if j<19 else 110
            mirror['calls'][j]['role_windows']=[[0,0,12,1,2*ns,ns],[0,1,12,1,2*ns,ns]]
        result=mirror_pair_metrics(same,mirror)
        self.assertEqual(len(result['calls']),35)
        self.assertEqual(result['segments']['prefix_all']['role_window_change'],0)
        self.assertAlmostEqual(result['segments']['target_first3']['role_window_change'],.2)
        self.assertEqual(result['segments']['target_last3']['role_window_change'],0)
        self.assertAlmostEqual(result['target_gap_reduction']['role_window_change'],.2)
        self.assertEqual(result['segments']['target_first3']['role_frequency_change'],0)
        # A lone first-call delay must not become a persistent source penalty.
        mirror=copy.deepcopy(same)
        mirror['calls'][16]['role_windows']=[[0,0,12,1,240,120],[0,1,12,1,240,120]]
        result=mirror_pair_metrics(same,mirror)
        self.assertAlmostEqual(result['segments']['target_first1']['role_window_change'],.2)
        self.assertEqual(result['segments']['target_first3']['role_window_change'],0)

    def test_role_clock_and_ns_are_paired_without_cross_sm_subtraction(self):
        same=history();mirror=copy.deepcopy(same)
        for call in mirror['calls']:
            call['role_windows']=[[0,0,12,80,200,125],[0,1,12,80,200,125]]
        result=mirror_pair_metrics(same,mirror)
        self.assertAlmostEqual(result['segments']['prefix_all']['role_window_change'],.25)
        self.assertEqual(result['segments']['prefix_all']['role_cycles_change'],0)
        self.assertAlmostEqual(result['segments']['prefix_all']['role_frequency_change'],-.2)
        mirror['calls'][4]['role_windows'][0][2]=11
        with self.assertRaisesRegex(ValueError,'work'):
            mirror_pair_metrics(same,mirror)


    def test_batch_snapshot_selects_run_gpu_and_closes_counter_section(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'environment.json').write_text(json.dumps({'gpu':'GPU-target, GH200'}))
            for name,counter in [('nvidia-smi-before.txt',100),('nvidia-smi-after.txt',140)]:
                text=('Timestamp : example\nGPU other\n    GPU UUID : GPU-other\n'
                    '    Clocks Event Reasons Counters\n        SW Power Capping : 9999 us\n'
                    'GPU selected\n    GPU UUID : GPU-target\n'
                    '    Clocks Event Reasons Counters\n        SW Power Capping : '+str(counter)+' us\n'
                    '        SW Thermal Slowdown : 0 us\n    Sparse Operation Mode : Disabled\n'
                    '    Utilization\n        GPU : 0 %\n    Clocks\n        Memory : 2619 MHz\n')
                (root/name).write_text(text)
            result=mirror_nvml_snapshots(root)
            self.assertEqual(result['counter_delta_us'],{'SW Power Capping':40,'SW Thermal Slowdown':0})
            self.assertEqual(result['snapshots']['nvidia-smi-before.txt']['gpu_header'],'selected')
            self.assertIn('numeric checks',result['scope'])


if __name__=='__main__':unittest.main()

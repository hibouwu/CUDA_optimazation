"""History stays a trajectory; its last target cannot masquerade as common16."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import analyze_r18
import run_r18
import run_v08
import v08_model


def synthetic():
    row=dict(id='series',config='cfg_b',kind='ordinary',m=128,n=128,k=64,
        lda=64,ldb=128,ldd=128,storage_m=128,storage_n=128,zero_m=-1,zero_n=-1,
        swizzle=8,input_mode='zero',warmup_policy='common-sw1-16',history_mode='series')
    setup=dict(row,grid=[1,1,1],trace_words=784,trace_version='v08-ends',
        warmup_swizzle=1,warmup_fixed_calls=16,history_protocol='prefix16-target19-ends-v1',
        history_capacity_bytes=35*16*8)
    durations=[.2,.4]*8+[.2]*19
    gaps=[.8]*34
    heads=[]
    for j in range(35):
        entry=1000+j*1000
        heads.extend([entry*2,entry,1,entry*2+10,entry*2+200,entry+100,1,0,0,0,0,1,0,0,0,0])
    event=dict(event='history_series',protocol='prefix16-target19-ends-v1',prefix_calls=16,
        target_calls=19,prefix_swizzle=1,target_swizzle=8,ctas=1,head_words=16,
        prefix_elapsed_us=sum(durations[:16])+sum(gaps[:15]),
        series_elapsed_us=sum(durations)+sum(gaps),call_elapsed_us=durations,
        inter_call_event_us=gaps,heads=heads)
    call=dict(event='call',elapsed_us=durations[-1],warmup_policy='common-sw1-16',
        warmup_swizzle=1,warmup_fixed_calls=16,target_swizzle=8,warmup_us=durations[:16],
        warmup_calls=16,history_protocol=event['protocol'],target_calls=19,
        preceding_target_calls=18,trace=heads[-16:]+[0]*768,
        history_recorded_calls=35,history_prefix_event_us=event['prefix_elapsed_us'],
        history_event_us=event['series_elapsed_us'],history_inter_call_event_us=gaps[:])
    return row,setup,call,event


class HistorySeries(unittest.TestCase):
    def test_trajectory_has_explicit_phases_and_local_clocks(self):
        row,setup,call,event=synthetic()
        self.assertGreater(analyze_r18.validate_warmup(setup,call,row),.02)
        result=analyze_r18.decode_history_series(setup,call,event,row)
        self.assertEqual(len(result['calls']),35)
        self.assertEqual([x['swizzle'] for x in result['calls']],[1]*16+[8]*19)
        self.assertEqual([x['phase'] for x in result['calls']],['prefix']*16+['target']*19)
        self.assertEqual([x['frequency_ghz'] for x in result['calls']],[2.]*35)
        self.assertEqual(result['inter_call_globaltimer_us'],[.9]*34)
        self.assertEqual(result['inter_call_event_us'],[.8]*34)
        self.assertIn('18 preceding target',result['scope'])

    def test_history_identity_and_prefix_cv_have_distinct_meanings(self):
        row,setup,call,event=synthetic()
        for bad in (dict(call,preceding_target_calls=0),dict(call,target_calls=1),
                    dict(call,history_protocol='single-common16-ends-v1'),
                    dict(call,history_recorded_calls=17)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                analyze_r18.validate_warmup(setup,bad,row)
        with self.assertRaisesRegex(ValueError,'capacity'):
            analyze_r18.validate_warmup(dict(setup,history_capacity_bytes=0),call,row)
        with self.assertRaisesRegex(ValueError,'converge'):
            analyze_r18.validate_warmup({},dict(warmup_us=call['warmup_us']),dict(swizzle=8))

    def test_reject_truncated_reordered_or_mismatched_series(self):
        row,setup,call,event=synthetic()
        bad_events=[]
        x=deepcopy(event);x['heads']=x['heads'][:-1];bad_events.append(x)
        x=deepcopy(event);x['target_calls']=18;bad_events.append(x)
        x=deepcopy(event);x['heads'][16:32]=x['heads'][:16];bad_events.append(x)
        x=deepcopy(event);x['call_elapsed_us'][0]+=2;bad_events.append(x)
        x=deepcopy(event);x['heads'][-16]+=1;bad_events.append(x)
        x=deepcopy(event);x['inter_call_event_us']=x['inter_call_event_us'][:-1];bad_events.append(x)
        for bad in bad_events:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                analyze_r18.decode_history_series(setup,call,bad,row)

    def test_single_has_same_capacity_but_only_one_target(self):
        row,setup,call,event=synthetic()
        row['history_mode']=setup['history_mode']='single'
        setup['history_protocol']=call['history_protocol']='single-common16-ends-v1'
        call.update(target_calls=1,preceding_target_calls=0,history_recorded_calls=17,
            history_inter_call_event_us=[.8]*16,history_event_us=sum(call['warmup_us'])+.2+16*.8)
        self.assertGreater(analyze_r18.validate_warmup(setup,call,row),.02)
        call['history_event_us']+=2
        with self.assertRaisesRegex(ValueError,'single event timeline'):
            analyze_r18.validate_warmup(setup,call,row)

    def test_replay_routes_series_and_single_observer_rejects_it(self):
        row,setup,call,event=synthetic()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'sample.gz'
            check=dict(event='check',status='ok',nonfinite=0,padding_errors=0,
                checked_indices=list(range(4096)),checked_values=[0.]*4096)
            with gzip.open(raw,'wt') as stream:
                for value in (dict(setup,event='setup'),event,call,check):
                    stream.write(json.dumps(value)+'\n')
            record=dict(raw=raw.name,raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                variant='ends',trial=0,elapsed_us=call['elapsed_us'])
            result=analyze_r18.replay(root,record,row)
            self.assertEqual(result['preceding_target_calls'],18)
            self.assertEqual(len(result['history_series']['calls']),35)
            with self.assertRaisesRegex(ValueError,'not single-call observe'):
                v08_model.observe(root,record,row,setup)

    def test_four_cases_and_five_before_after_pairs(self):
        rows=json.loads((Path(__file__).parent/'configs/r09-history-series.json').read_text())
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r['variants']==['ends'] and r['warmup_policy']=='common-sw1-16' for r in rows))
        for sw in (1,8):
            pair=[r for r in rows if r['swizzle']==sw]
            self.assertEqual(len({r['group'] for r in pair}),1)
            first=[run_r18.trial_rows(pair,t)[0]['history_mode'] for t in range(10)]
            self.assertEqual(first.count('single'),5)
            self.assertEqual(first.count('series'),5)
        for row in rows:
            for runner in (run_r18,run_v08):
                args=runner.case_args(row)
                self.assertEqual(args[args.index('--history-mode')+1],row['history_mode'])

    def test_mirror_identity_is_local_to_each_process(self):
        row,setup,call,event=synthetic()
        row.update(swizzle=1,history_source='mirror',mirror_target_slot=1)
        for item,offset in ((setup,0),(call,4096)):
            item.update(history_source='mirror',mirror_target_slot=1,source_copies=2,
                target_a_ptr=256+offset,target_b_ptr=512+offset,
                prefix_a_ptr=768+offset,prefix_b_ptr=1024+offset)
        call['target_swizzle']=1
        analyze_r18.validate_warmup(setup,call,row)
        with self.assertRaisesRegex(ValueError,'declared source'):
            analyze_r18.validate_warmup(setup,dict(call,prefix_a_ptr=call['target_a_ptr']),row)
        with self.assertRaisesRegex(ValueError,'capacity'):
            analyze_r18.validate_warmup(setup,dict(call,source_copies=1),row)
        row['history_source']='same'
        for item in (setup,call):
            item.update(history_source='same',prefix_a_ptr=item['target_a_ptr'],prefix_b_ptr=item['target_b_ptr'])
        analyze_r18.validate_warmup(setup,call,row)

    def test_lightweight_reader_import(self):
        script=('import sys; sys.path.insert(0,'+repr(str(Path(__file__).parent))+'); '
            'import analyze_r18,run_r18,run_v08; '
            "assert 'numpy' not in sys.modules; assert 'scipy' not in sys.modules")
        subprocess.run([sys.executable,'-I','-S','-c',script],check=True)


if __name__=='__main__':unittest.main()

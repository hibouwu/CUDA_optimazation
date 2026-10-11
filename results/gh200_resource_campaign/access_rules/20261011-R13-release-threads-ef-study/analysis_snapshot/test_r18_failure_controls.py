"""CPU checks for matched controls and sparse ends/dual observation routing."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_r18
import v08_model

ROWS=json.loads((Path(__file__).parent/'configs/v09-failure-controls.json').read_text())

class FailureControlsTests(unittest.TestCase):
    def test_matrix_preserves_pair_interventions_and_allocation(self):
        self.assertEqual(len(ROWS),12)
        self.assertEqual(sum(r['set']=='reference' for r in ROWS),4)
        for row in ROWS:
            self.assertGreaterEqual(row['lda'],row['k'])
            self.assertGreaterEqual(row['ldb'],row['storage_n'])
            self.assertGreaterEqual(row['ldd'],row['storage_n'])
            self.assertGreaterEqual(row['alloc_lda'],row['lda'])
            self.assertGreaterEqual(row['alloc_ldb'],row['ldb'])
            self.assertLessEqual((row['storage_m']-1)*row['lda']+row['k'],row['storage_m']*row['alloc_lda'])
            self.assertLessEqual((row['k']-1)*row['ldb']+row['storage_n'],row['k']*row['alloc_ldb'])
        maps=[r for r in ROWS if r['group']=='r18_partial_map']
        self.assertEqual({r['input_map_m'] for r in maps},{960,1024})
        self.assertEqual({k:v for k,v in maps[0].items() if k not in ('id','input_map_m')},
                         {k:v for k,v in maps[1].items() if k not in ('id','input_map_m')})
        self.assertEqual((maps[0]['m'],maps[0]['storage_m'],maps[0]['zero_m']),(960,1024,960))
        for prefix in ('r13_stage','r09_protocol'):
            group=[r for r in ROWS if r['id'].startswith(prefix)]
            sizes={(r['storage_m']*r['alloc_lda'],r['k']*r['alloc_ldb'],r['storage_m']*r['ldd']) for r in group}
            self.assertEqual(len(sizes),1)
        for k in (1024,4096):
            a,b=[r for r in ROWS if r['group']==f'r13_stage_k{k}']
            self.assertEqual({x:v for x,v in a.items() if x not in ('id','stages')},
                             {x:v for x,v in b.items() if x not in ('id','stages')})

    def test_prepare_only_builds_requested_variants_and_ends_layout(self):
        def overlay(_cutlass,out):
            for name in (run_r18.common.COOP,run_r18.common.PING):
                p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('        ++v06_tile_seq;\n')
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);cutlass=base/'cutlass'
            (cutlass/'include').mkdir(parents=True);(cutlass/'tools/util/include').mkdir(parents=True)
            root=base/'run'
            with patch.object(run_r18.common,'make_overlay',side_effect=overlay):
                run_r18.prepare(root,cutlass,'r18',ROWS,dual_clock=True)
            commands=json.loads((root/'build/commands.json').read_text())
            self.assertEqual(set(commands),{'cfg_a_plain','cfg_a_dual','cfg_b_plain','cfg_b_dual','cfg_b_ends',
                                            'cfg_b_s4_plain','cfg_b_s4_dual'})
            ends=commands['cfg_b_ends']
            self.assertIn('-DV08_ENDS',ends);self.assertIn('-DV06_TRACE',ends)
            self.assertNotIn('-DR18_DUAL_CLOCK',ends);self.assertNotIn('-DR18_MATCH_DUAL_LAYOUT',ends)
            self.assertIn('-DR18_DUAL_CLOCK',commands['cfg_b_dual'])
            self.assertIn('-DR18_STAGE_COUNT=4',commands['cfg_b_s4_plain'])
            # The transferred package must import its ends decoder without repo PYTHONPATH.
            code='import sys;sys.path.insert(0,sys.argv[1]);import v08_model;assert callable(v08_model.observe)'
            subprocess.run([sys.executable,'-I','-c',code,str(root/'source')],check=True)

    def test_set_filters_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.ExitStack() as opened:
            root=Path(directory);(root/'cases.json').write_text(json.dumps(ROWS))
            (root/'run_config.json').write_text(json.dumps(dict(family='r18',cases_sha256=run_r18.common.sha(root/'cases.json'))))
            env={'gpu':'test-device'};(root/'environment.json').write_text(json.dumps(env))
            seen=[]
            original_open=Path.open
            def track_lock(path,*args,**kwargs):
                handle=original_open(path,*args,**kwargs)
                return opened.enter_context(handle) if path.name=='.run.lock' else handle
            with patch.object(Path,'open',track_lock),patch.dict(run_r18.os.environ,{'SLURM_JOB_ID':'cpu-test'}),patch.object(run_r18.common,'identity',return_value=env),patch.object(run_r18,'open',create=True),patch.object(run_r18.fcntl,'flock'),patch.object(run_r18.common,'verify'),patch.object(run_r18,'run_one',side_effect=lambda root,row,v,t:seen.append((row['set'],v))):
                for case_set in ('reference','ctrl'):
                    with patch.object(sys,'argv',['run_r18.py','sample','--output',str(root),'--set',case_set,'--procs','1']),contextlib.redirect_stdout(io.StringIO()):
                        run_r18.main()
            self.assertEqual([s for s,v in seen[:8]],['reference']*8)
            self.assertEqual([s for s,v in seen[8:]],['ctrl']*18)
            self.assertEqual(sum(v=='ends' for s,v in seen),2)

    def test_second_prefill_uses_copy_and_preserves_tail_order_in_both_builds(self):
        # Minimal source sites include both prologues, so only mma is patched.
        mainloop=('mma body\n    // Prologue GMMAs\noriginal prologue\n'
            '  /// Perform a Consumer Epilogue\n    // Prologue GMMAs\noriginal tail\n')
        ping=('        collective_mainloop.mma_tail(\noriginal arguments\n        );\n'
            '        // Update starting mainloop pipeline state for the next tile\n')
        def overlay(cutlass,out):
            for name in (run_r18.common.COOP,run_r18.common.PING,run_r18.common.MAINLOOP):
                dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
                body=(cutlass/'include'/name).read_text()
                dest.write_text(body+('        ++v06_tile_seq;\n' if name!=run_r18.common.MAINLOOP else ''))
        row=dict(ROWS[0],id='prefill0',config='cfg_b',m=2816,n=1536,k=4096,
            stages=4,epilogue_defer=2,second_prefill=0,variants=['plain','progress'])
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);cutlass=base/'cutlass';(cutlass/'tools/util/include').mkdir(parents=True)
            for name,body in ((run_r18.common.MAINLOOP,mainloop),(run_r18.common.PING,ping),
                              (run_r18.common.COOP,'cooperative body\n')):
                path=cutlass/'include'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body)
            root=base/'run'
            with patch.object(run_r18.common,'make_overlay',side_effect=overlay),patch.object(run_r18,'make_stage_sparse_overlay'):
                run_r18.prepare(root,cutlass,'r18',[row,dict(row,id='prefill1',second_prefill=1)],stage_progress=True)
            commands=json.loads((root/'build/commands.json').read_text())
            self.assertEqual(set(commands),{'cfg_b_s4_plain','cfg_b_s4_progress'})
            for command in commands.values():
                self.assertIn('-DR13_SECOND_PREFILL',command)
                self.assertIn('-DR13_OUTPUT_BACKGROUND',command)
            self.assertNotIn('-DR13_STAGE_PROGRESS',commands['cfg_b_s4_plain'])
            self.assertIn('-DR13_STAGE_PROGRESS',commands['cfg_b_s4_progress'])
            self.assertEqual(json.loads((root/'run_config.json').read_text())['trace_version'],'r13-second-prefill-v1')
            for flag in (0,1):
                args=run_r18.case_args(dict(row,second_prefill=flag))
                self.assertEqual(args[args.index('--second-prefill')+1],str(flag))
            raw=(root/'source/cutlass/include'/run_r18.common.MAINLOOP).read_text()
            stamped=(root/'source/overlay'/run_r18.common.MAINLOOP).read_text()
            self.assertEqual(raw,stamped)
            added=raw[:raw.index('    // Prologue GMMAs')]
            self.assertIn('auto prefill_state=smem_pipe_read;',added)
            self.assertIn('pipeline.consumer_wait(prefill_state);',added)
            self.assertIn('++prefill_state;',added)
            self.assertNotIn('consumer_release',added)
            self.assertNotIn('++smem_pipe_read',added)
            self.assertIn('threadIdx.x>=256 && threadIdx.x<384',added)
            self.assertIn('smem_pipe_read.count()==64 && k_tile_count==64',added)
            self.assertLess(added.index('prefill_begin_c'),added.index('if (::r13_second_prefill)'))
            self.assertGreater(added.index('prefill_end_c'),added.index('++prefill_state'))
            self.assertEqual(raw[raw.index('  /// Perform a Consumer Epilogue'):],mainloop[mainloop.index('  /// Perform a Consumer Epilogue'):])
            kernel=(root/'source/cutlass/include'/run_r18.common.PING).read_text()
            self.assertLess(kernel.index('collective_mainloop.mma_tail'),kernel.index('r13_s.first_tail_c'))
            self.assertLess(kernel.index('r13_s.first_tail_ns'),kernel.index('math_wg_order.barrier_[0][0].wait(0)'))
            self.assertIn('threadIdx.x==128 && mainloop_pipe_consumer_state.count()==0',kernel)

    def test_ends_uses_ends_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            row=next(r for r in ROWS if r['id']=='r09_protocol_k4096')
            fake=subprocess.CompletedProcess([],0,json.dumps(dict(event='call',elapsed_us=1,warmup_us=[1]*8))+'\n','')
            with patch.object(run_r18.subprocess,'run',return_value=fake),patch.object(v08_model,'observe') as observe:
                run_r18.run_one(Path(directory),row,'ends',0)
                observe.assert_called_once()
                self.assertEqual(observe.call_args.args[1]['variant'],'ends')

if __name__=='__main__':unittest.main()

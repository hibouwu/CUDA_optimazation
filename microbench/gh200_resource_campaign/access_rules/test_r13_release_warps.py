"""Release observer protocol rejection tests; no claim about GPU coverage."""
import copy
import json
from pathlib import Path
import unittest
import tempfile
import run_r18
from analyze_r18 import decode_stage_sparse, validate_warmup
import test_r13_sparse
from run_r18 import validate_release_warp_rows


class ReleaseWarpTests(unittest.TestCase):
    def trace(self,stage,flag):
        setup,call=test_r13_sparse.StageSparseTests().trace(stage)
        setup.update(trace_version='r13-release-warps-v1',release_warp_observer=flag)
        call['release_warp_observer']=flag
        b=call['trace'];b[31]=0x52313307
        b[40:48]=[170,172,169,171,168,171,170,173] if flag else [170,172,0,0,0,0,0,0]
        return setup,call

    def test_on_off_and_stage_phase(self):
        for stage in (4,6):
            for flag in (0,1):
                s,c=self.trace(stage,flag);r=decode_stage_sparse(s,c)['release_warps']
                self.assertEqual(r['consumer_phase'],(80//stage)%2)
                self.assertEqual(r['enabled'],bool(flag))
                self.assertTrue(r['conditional_on_sass_coverage'])

    def test_missing_wrong_type_or_impossible_trace(self):
        for index,value in [(40,169),(42,0),(43,168),(42,172),(42,True),(42,169.),(42,2**64),(48,1),(31,0x52313301)]:
            with self.subTest(index=index,value=value):
                s,c=self.trace(6,1);c['trace'][index]=value
                with self.assertRaises(ValueError):decode_stage_sparse(s,c)
        s,c=self.trace(4,0);c['trace'][42]=169
        with self.assertRaises(ValueError):decode_stage_sparse(s,c)

    def test_identity_and_warmup(self):
        row=dict(swizzle=1,release_warp_observer=1)
        setup=dict(release_warp_observer=1,release_warp_warmup=0)
        call=dict(setup,warmup_us=[100.]*8)
        self.assertEqual(validate_warmup(setup,call,row),0)
        for key,value in [('release_warp_observer',0),('release_warp_observer',True),('release_warp_warmup',1)]:
            bad=dict(call);bad[key]=value
            with self.assertRaises(ValueError):validate_warmup(setup,bad,row)
        with self.assertRaises(ValueError):validate_warmup(setup,call,dict(swizzle=1))

    def test_overlay_keeps_release_and_flushes_after_original_exit_record(self):
        mainloop=('    const bool r13_target = r13_sparse_cta() && threadIdx.x==256 && v06_tile_seq==0;\n'
            'if (r13_target && r13_k==17) { r13_s->release_slot=smem_pipe_release.index(); }\n'
            'pipeline.consumer_release(smem_pipe_release);\n'
            'if (r13_target && r13_k==17) r13_s->release_end=v06_clock();\n')
        flush='      if (r13_sparse_cta() && threadIdx.x==256) r13_sparse_consumer_flush(r13_s);\n'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,text in [(run_r18.common.MAINLOOP,mainloop),(run_r18.common.PING,flush)]:
                path=root/'overlay'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
            run_r18.make_release_warp_overlay(root)
            text=(root/'overlay'/run_r18.common.MAINLOOP).read_text()
            self.assertEqual(text.count('pipeline.consumer_release(smem_pipe_release);'),1)
            self.assertNotIn('r13_release_warp_flush',text)
            for operation in ('__syncthreads','warpgroup_wait','consumer_wait','expect_tx'):
                self.assertNotIn(operation,text)
            text=(root/'overlay'/run_r18.common.PING).read_text()
            self.assertLess(text.index('r13_sparse_consumer_flush'),text.index('r13_release_warp_flush'))
            with self.assertRaises(ValueError):run_r18.make_release_warp_overlay(root)

    def test_configuration_rejects_other_treatments(self):
        rows=json.loads((Path(__file__).parent/'configs/r13-release-warps.json').read_text())
        validate_release_warp_rows(rows,True,False,False)
        for key,value in [('alloc_lda',4096),('epilogue_defer',2),('a_row_rotation',1),('release_warp_observer',True),('input_map_m',2800)]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):validate_release_warp_rows(bad,True,False,False)
        with self.assertRaises(ValueError):validate_release_warp_rows(rows,True,False,True)


if __name__=='__main__':unittest.main()

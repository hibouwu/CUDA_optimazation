#!/usr/bin/env python3
"""CPU fixtures for the proposed R15 profile; no measured GPU data is generated."""
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

from analyze_r15_output_ns import PROFILE, WIDTH, analyze, direct_windows, summarize_call
from analyze_r18 import reference
from analyze_r15 import sha
from analyze_r15 import analyze_dual_roles, output_parts, paired_J_decomposition, role_windows
from run_v08 import row as make_row


def fixture():
    row = dict(id='synthetic', config='cfg_c', m=256, n=256, k=64,
        lda=64, ldb=256, ldd=256, storage_m=256, storage_n=256,
        zero_m=-1, zero_n=-1, swizzle=1, kind='ordinary')
    setup = dict(row, event='setup', trace_version=PROFILE, trace_words=WIDTH, grid=[1, 2, 1])
    words = [0] * (2 * WIDTH)
    origin = 2**56  # Subtract integer anchors before floating-point overlap calculations.
    for cta in range(2):
        i = cta * WIDTH
        words[i:i + 16] = [1000, origin, cta + 1, 1200,
            4900, origin + 1000, 1, 5000 + cta * 5000, origin + 1000, 1,
            0, cta + 1, cta + 1, origin + 100 + 100 * cta, origin + 300 + 100 * cta, 0]
        words[i + 16:i + 22] = [1300, 1400, 2500, 3504, 1, cta + 1]
        j = i + 16 + 64 * 6
        words[j:j + 6] = [1300, 1400, 2000, 3500, 1, cta + 1]
    call = dict(event='call', elapsed_us=2.0, warmup_us=[2.0] * 8, trace=words)
    return row, setup, call


class OutputNsTests(unittest.TestCase):
    def test_two_role_event_algebra(self):
        parts = output_parts([50,100,160,500], [40,80,120,496])
        self.assertEqual([parts[k] for k in ('w','arrival_gap','issuer','post_permit','done_join','E','after_main')],
                         [60,40,376,336,4,340,400])
        ns = output_parts([10,100,130,310], [10,90,110,310])
        self.assertEqual(ns['done_join'],0)
        self.assertEqual(ns['post_permit'],180)

    def test_dual_layout_decodes_both_roles_without_interpolation(self):
        row, _, _ = fixture()
        setup = dict(row,event='setup',trace_version='r18-dual-clock-events',
                     trace_tile_words=10,trace_words=1296,grid=[1,2,1])
        words = [0] * (2*1296)
        origin = 2**56
        for cta in range(2):
            i=cta*1296;ns=origin+cta*32
            words[i:i+16]=[1000,ns,cta+1,1100,2400,ns+800,1,2500,ns+896,1,0,cta+1,cta+1,ns+32,0,0]
            words[i+16:i+26]=[1200,1400,1600,2004,1,cta+1,ns+100,ns+200,ns+320,ns+640]
            j=i+16+64*10
            words[j:j+10]=[1200,1380,1500,2000,1,cta+1,ns+96,ns+192,ns+288,ns+640]
        call=dict(event='call',elapsed_us=2.0,warmup_us=[2.0]*8,trace=words)
        indices=list(range(4096))
        check=dict(event='check',status='ok',padding_errors=0,checked_indices=indices,
                   checked_values=[reference(i//row['n'],i%row['n'],row['k']) for i in indices])
        with tempfile.TemporaryDirectory(prefix='r15-dual-roles-cpu-') as tmp:
            root=Path(tmp);folder=root/'samples'/row['id'];folder.mkdir(parents=True)
            raw=folder/'dual-00.txt.gz'
            with gzip.open(raw,'wt') as stream:
                for event in (setup,call,check):stream.write(json.dumps(event)+'\n')
            (folder/'dual-00.json').write_text(json.dumps(dict(variant='dual',trial=0,returncode=0,
                raw=str(raw.relative_to(root)),raw_sha256=sha(raw),elapsed_us=2.0)))
            process=role_windows(root,row,'dual')[0]
            m=process['phases']['first']['median']
            self.assertEqual(m['post_permit'],400)
            self.assertEqual(m['post_permit_ns'],320)
            self.assertEqual(m['arrival_gap_ns'],32)
            self.assertEqual(m['done_join_ns'],0)
            self.assertEqual(m['post_permit_cycles_per_ns'],1.25)
            self.assertEqual(m['tail_after_merged_done_ns'],256)
            self.assertEqual(len(process['tiles']),2)

    def test_complete_dual_analysis_separates_ns_and_local_rate(self):
        # Full-size logical work and 80 synthetic processes; never saved as GPU evidence.
        rows = [dict(make_row('cfg_c_'+label,'cfg_c','ctrl',1536,2816,k),
                     input_mode='dyadic',seed=17,sm_count=0)
                for label,k in (('c2_k4096',4096),('c6_longk',16384))]
        with tempfile.TemporaryDirectory(prefix='r15-dual-analysis-cpu-') as tmp:
            root=Path(tmp);(root/'build').mkdir()
            for manifest in ('source_hashes.json','build/binary_hashes.json'):
                (root/manifest).write_text('{}')
            (root/'cases.json').write_text(json.dumps(rows))
            (root/'environment.json').write_text(json.dumps(dict(gpu='GPU-synthetic, GH200, CPU fixture')))
            for row in rows:
                scale=1 if row['k']==4096 else 2
                indices=list(range(4096))
                check=dict(event='check',status='ok',padding_errors=0,checked_indices=indices,
                    checked_values=[reference(i//row['n'],i%row['n'],row['k']) for i in indices])
                folder=root/'samples'/row['id'];folder.mkdir(parents=True)
                for variant in ('plain','wide','stamped','dual'):
                    width=784 if variant=='plain' else 1296
                    setup=dict(row,event='setup',grid=[66,2,1],trace_tile_words=6 if variant=='plain' else 10,
                        trace_words=width,trace_version='r18-dual-clock-events' if variant=='dual' else 'r18-wide-clock-events',
                        gpu_uuid='GPU-synthetic',input_map_m=1536,input_map_n=2816,
                        initialized_input_rows=1536,initialized_input_columns=2816,scratch_bytes=132*width*8)
                    words=[0]*(132*width)
                    if variant in ('stamped','dual'):
                        for cta in range(132):
                            i=cta*width;ns=2**56+cta*32;mi,ni=divmod(cta,22)
                            words[i:i+16]=[1000*scale,ns,cta+1,1100*scale,2400*scale,ns+800,
                                1,2500*scale,ns+896,1,0,cta+1,cta+1,ns+32,0,0]
                            a=[1200*scale,1400*scale,1600*scale,2004*scale,mi+1,ni+1]
                            b=[1200*scale,1380*scale,1500*scale,2000*scale,mi+1,ni+1]
                            a+= [ns+100,ns+200,ns+320,ns+640] if variant=='dual' else [0]*4
                            b+= [ns+96,ns+192,ns+288,ns+640] if variant=='dual' else [0]*4
                            words[i+16:i+26]=a;j=i+16+640;words[j:j+10]=b
                    call=dict(event='call',elapsed_us=2.0,warmup_us=[2.0]*8,trace=words)
                    for trial in range(10):
                        raw=folder/f'{variant}-{trial:02d}.txt.gz'
                        with gzip.open(raw,'wt') as stream:
                            for event in (setup,call,check):stream.write(json.dumps(event)+'\n')
                        raw.with_name(f'{variant}-{trial:02d}.json').write_text(json.dumps(dict(
                            variant=variant,trial=trial,returncode=0,raw=str(raw.relative_to(root)),
                            raw_sha256=sha(raw),elapsed_us=2.0)))
            analyze_dual_roles(root,root/'analysis')
            result=json.loads((root/'analysis/dual-roles.json').read_text())
            decomposition=paired_J_decomposition(result['dual_processes'][rows[0]['id']][0],
                                                 result['dual_processes'][rows[1]['id']][0])
            self.assertAlmostEqual(decomposition['mean_log_ratio']['cycles'],math.log(2))
            self.assertEqual(decomposition['mean_log_ratio']['ns'],0)
            self.assertAlmostEqual(decomposition['mean_log_ratio']['local_rate'],math.log(2))
            self.assertLess(decomposition['max_identity_residual'],1e-12)
            contrast=result['contrast_median']
            self.assertEqual(contrast['post_permit_ns'],0)
            self.assertEqual(contrast['post_permit_cycles'],400)
            self.assertEqual(contrast['J_ratio_ns'],1)
            self.assertEqual(contrast['J_ratio_local_rate'],2)
            for case in result['cases']:
                recursion=case['development_recursion']['ns']
                self.assertEqual(recursion['E0_from_J_R'],320)
                self.assertEqual(recursion['M_to_final_from_parts'],696)

    def test_direct_overlap_ignores_clock_conversion(self):
        _, setup, call = fixture()
        result = summarize_call(direct_windows(setup, call))
        self.assertEqual(result['issuer_store_ns'], 200)
        self.assertEqual(result['overlap_direct'], 1.5)
        self.assertEqual(result['peak_overlap'], 2)
        self.assertEqual(result['issuer_store_cycles'], 1500)
        self.assertEqual(result['merged_store_cycles'], 1004)
        self.assertNotEqual(result['overlap_affine'], result['overlap_direct'])

    def test_old_profile_has_no_direct_fallback(self):
        _, setup, call = fixture()
        setup['trace_version'] = 'r18-work-coordinates'
        with self.assertRaisesRegex(ValueError, 'no affine fallback'):
            direct_windows(setup, call)

    def test_rates_are_paired_before_aggregation(self):
        _, setup, call = fixture()
        call['trace'][WIDTH + 14] -= 100
        result = summarize_call(direct_windows(setup, call))
        self.assertEqual(result['issuer_cycles_per_ns'], (7.5 + 15) / 2)
        self.assertNotEqual(result['issuer_cycles_per_ns'],
                            result['issuer_store_cycles'] / result['issuer_store_ns'])
        self.assertAlmostEqual(result['issuer_to_cta_rate_ratio'], (7.5 / 4 + 15 / 9) / 2)

    def test_second_tile_is_not_silently_omitted(self):
        _, setup, call = fixture()
        call['trace'][6] = call['trace'][9] = 2
        with self.assertRaisesRegex(ValueError, 'exactly one tile'):
            direct_windows(setup, call)

    def test_same_time_end_and_start_do_not_overlap(self):
        _, setup, call = fixture()
        call['trace'][WIDTH + 13] += 100
        result = summarize_call(direct_windows(setup, call))
        self.assertEqual(result['overlap_direct'], 1)
        self.assertEqual(result['peak_overlap'], 1)

    def test_archive_entry_uses_the_raw_global_setup(self):
        row, setup, call = fixture()
        indices = list(range(4096))
        check = dict(event='check', status='ok', padding_errors=0,
            checked_indices=indices, checked_values=[reference(i // row['n'], i % row['n'], row['k']) for i in indices])
        with tempfile.TemporaryDirectory(prefix='r15-output-ns-cpu-') as tmp:
            root = Path(tmp)
            folder = root / 'samples' / row['id']
            folder.mkdir(parents=True)
            (root / 'cases.json').write_text(json.dumps([row]))
            (root / 'static_setup.json').write_text(json.dumps([dict(case=row['id'], setup=dict(setup, trace_version='r18-work-coordinates'))]))
            for variant in ('plain', 'stamped', 'ends', 'global'):
                raw = folder / (variant + '-00.txt.gz')
                with gzip.open(raw, 'wt') as stream:
                    actual_setup = setup if variant == 'global' else dict(setup, trace_version='r18-work-coordinates')
                    for event in (actual_setup, call, check):
                        stream.write(json.dumps(event) + '\n')
                record = dict(variant=variant, trial=0, returncode=0, elapsed_us=2.0,
                    raw=str(raw.relative_to(root)), raw_sha256=sha(raw))
                (folder / (variant + '-00.json')).write_text(json.dumps(record))
            analyze(root, root / 'analysis')
            result = json.loads((root / 'analysis/output-ns.json').read_text())
            self.assertEqual(result['cases'][0]['median']['overlap_direct'], 1.5)
            self.assertEqual(result['cases'][0]['global_relative'], dict(plain=0, stamped=0, ends=0))


if __name__ == '__main__':
    unittest.main()

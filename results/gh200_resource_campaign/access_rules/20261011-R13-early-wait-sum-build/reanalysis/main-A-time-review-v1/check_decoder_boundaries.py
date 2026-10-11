"""Replay fixed CPU-only history/clock boundary cases without GPU acquisition."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    fixed = root / 'fixed-snapshot'
    identity = json.loads((root / 'fixed-source-identity.json').read_text())
    for name, digest in identity.items():
        assert sha(fixed / name) == digest
    assert sha(root / 'source_hashes.json') == 'b57bbb9852ce7da56f87ad5eac6be5412589e0d32c369e0499b41414a0b8021d'
    assert sha(root / 'contract.json') == '258e11dfd79c98991ec71a6b192292fbea4009f9289397855b859341a1264357'
    manifest = json.loads((root / 'source_hashes.json').read_text())
    for path in (root / 'prepared-snapshot/source').rglob('*'):
        if path.is_file():
            relative = str(path.relative_to(root / 'prepared-snapshot'))
            assert sha(path) == manifest[relative]
    sys.path.insert(0, str(fixed))
    from test_r13_wait_sum import WaitSumTests
    from decode_r13_wait_sum import decode, decode_trace
    from analyze_r18 import reference
    old_path = root / 'snapshot/decode_r13_wait_sum.py'
    assert sha(old_path) == 'add8b409031814910c048e32768916d4cc37fef604420264b6f053646d722259'
    spec = importlib.util.spec_from_file_location('old_wait_sum_decoder', old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    fixture = WaitSumTests()
    positives = []
    clock_fields = (15,16,17,18,19,20,21,22,24,32,37,38)
    for stage in (4,6):
        for flag in (0,1):
            for offset in (0,2**63+17):
                setup, call = fixture.fixture(flag)
                setup['stages'] = stage
                for index in clock_fields:
                    if call['trace'][index]:
                        call['trace'][index] += offset
                result = decode_trace(setup, call)
                assert result['early_cycles'] == 100
                assert result['full_wait_cycles'] == (50 if flag else None)
                assert result['outside_wait_cycles'] == (50 if flag else None)
                assert result['full_wait_ns'] is None and result['outside_wait_ns'] is None
                assert len(result['completed_prefix_event_us']) == 16
                assert not result['observation_qualified'] and not result['prediction_qualified']
                positives.append(dict(stages=stage,observer=flag,absolute_clock_offset=offset,
                    early_cycles=result['early_cycles'],full_wait_cycles=result['full_wait_cycles'],
                    outside_wait_cycles=result['outside_wait_cycles'],component_ns=None))
    zeros = []
    negatives = []
    checks = dict(event='check',status='ok',samples=4096,nonfinite=0,padding_errors=0,
        max_storage_reference_error=0,checked_indices=list(range(4096)),
        checked_values=[reference((i//1536)%17,(i%1536)%17,4096,17) for i in range(4096)])
    for stage in (4,6):
        setup, call = fixture.fixture()
        setup['stages'] = stage
        call['trace'][34] = 0
        zero = decode_trace(setup,call)
        assert zero['full_wait_cycles'] == 0 and zero['outside_wait_cycles'] == 100
        zeros.append(dict(stages=stage,enabled_measured_wait=0,disabled_wait=None))
        setup, call = fixture.fixture()
        setup.update(stages=stage,event='setup')
        call['event'] = 'call'
        call['trace'][37] = 193
        call['trace'][34] = 2
        old_result = old.decode([setup,call,checks],'test-gpu')
        assert old_result['full_wait_cycles'] == 2 and old_result['outside_wait_cycles'] == 98
        assert old_result['independently_checked_values'] == 4096
        try:
            decode([setup,call,checks],'test-gpu')
        except ValueError as error:
            assert str(error) == 'cumulative waits escape the early window'
            negatives.append(dict(stages=stage,first_H2=193,old_H16=190,last_B16=195,
                original_full_decoder_accepted=True,fixed_full_decoder_rejected=True,error=str(error)))
        else:
            raise AssertionError('first_H2 after old H16 was accepted')
    shortened = []
    for stage in (4,6):
        setup, call = fixture.fixture()
        setup['stages'] = stage
        call['warmup_us'] = call['warmup_us'][:15]
        try:
            decode_trace(setup,call)
        except ValueError as error:
            assert str(error) == 'incomplete common prefix'
            shortened.append(dict(stages=stage,completed_prefix_events=15,rejected=True))
        else:
            raise AssertionError('missing completed prefix event was accepted')
    source = root / 'prepared-snapshot/source'
    host = (source / 'probes/r18.cu').read_text()
    off = host.index('cudaMemcpyToSymbol(r13_wait_sum_observer,&wait_sum_off')
    warm = host.index('for (int i = 0; i < 16; ++i) warm.push_back(timed(warmup_launch_params));')
    target_flag = host.index('cudaMemcpyToSymbol(r13_wait_sum_observer,&o.wait_sum_observer')
    target = host.index('float ms = timed(launch_params);',target_flag)
    assert off < warm < target_flag < target
    assert 'warmup_launch_params.mainloop = launch_params.mainloop;' in host
    collective = (source / 'overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp').read_text()
    begin = collective.index('if (r13_sum_selected) r13_wait_begin = v06_clock();')
    test = collective.index('auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);',begin)
    wait = collective.index('pipeline.consumer_wait(smem_pipe_read, barrier_token);',test)
    end = collective.index('uint64_t r13_wait_end = v06_clock();',wait)
    record = collective.index('r13_s->wait_sum.record(r13_k, r13_wait_begin, r13_wait_end);',end)
    assert begin < test < wait < end < record
    assert 'r13_sparse_cta() && threadIdx.x==256 && v06_tile_seq==0;' in collective
    assert 'r13_k>=2 && r13_k<=16;' in collective
    input_paths = [root/'input-identity.json',root/'source_hashes.json',root/'contract.json',
        root/'fixed-source-identity.json',root/'stock-tests.json',root/'fixed-stock-tests.json',
        root/'first-H2-after-H16-negative.json']
    input_paths += [p for folder in (root/'snapshot',fixed,root/'prepared-snapshot')
                    for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    result = dict(schema='r09-r13-cycle-time-input-review/1',
        original_decoder_sha256=sha(old_path),fixed_decoder_sha256=sha(fixed/'decode_r13_wait_sum.py'),
        source_manifest_sha256=sha(root/'source_hashes.json'),
        selected_window=dict(cta=0,physical_thread=256,cta_output_ordinal=1,WG_local_output_ordinal=0,
            groups=list(range(2,17)),native_cycle_intervals=15,prefix_calls=16,prefix_observer=0),
        source_order_checks=dict(same_thread_target_and15groups=True,H_before_try_and_wait_B_after_wait=True,
            off_before16_completed_prefix_before_target_flag_and_target=True,same_mainloop_buffers_within_process=True),
        positive_integer_and_switch_cases=positives,enabled_zero_vs_disabled_missing_cases=zeros,
        repaired_order_negative_cases=negatives,shortened_prefix_negative_cases=shortened,
        stock_test_counts=dict(original=5,fixed=6),
        conclusion='Fixed decoder rejects first_H2 after old H16; legal native-cycle partition,16off prefix and large-integer subtraction checks pass. Component ns remains unsupported and off remains unobserved.',
        limits=['CPU synthetic records and source review only; no actual GPU traces or performance were acquired by A.',
                'Same-SM nonwrapping recorded cycles; outside_wait includes observer arithmetic and all nonselected work inside C1->C16.',
                'Stage4 andstage6 prefix kernels differ; same within-stage declared history does not establish equal cross-stage states.',
                'Source ordering and numerical fixture checks do not establish independent timestamp accuracy, no perturbation, hardware readiness, TMA pricing or prediction qualification.'],
        observation_qualified=False,prediction_qualified=False,
        input_sha256={str(p):sha(p) for p in input_paths},review_sha256=sha(Path(__file__)))
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()

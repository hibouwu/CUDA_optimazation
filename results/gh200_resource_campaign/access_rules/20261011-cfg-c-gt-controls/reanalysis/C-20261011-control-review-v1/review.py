#!/usr/bin/env python3
"""C-only independent raw-slot replay of the frozen off/on and off/off controls."""
import argparse
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tarfile
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
FREEZE_COMMIT = '79a86b8d68de1e6fd8a13e3e0ab018f67d01df35'


def review(run, ordering_review, output):
    if output.exists():
        raise ValueError('output must be a new C-only directory')
    inputs = {}
    repo = run.parents[3]
    data = run/'returned'

    def read(path):
        content = path.read_bytes()
        inputs[str(path.resolve())] = hashlib.sha256(content).hexdigest()
        return content

    def sha(path):
        return hashlib.sha256(read(path)).hexdigest()

    def load(path):
        return json.loads(read(path))

    public_analysis_sha = sha(run/'analysis.json')
    public_summary_sha = sha(run/'summary.json')
    manifest = load(run/'manifest.json')
    freeze = load(run/'freeze.json')
    assert manifest['status'] == 'frozen' and manifest['comparisons'] == ['off_on','off_off']
    assert manifest['gpu_uuid'] == 'GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'
    assert manifest['gates'] == dict(per_common_series_median_absolute_ns=64,
        per_common_series_maximum_absolute_ns=256,event_median_absolute_ratio=.02,
        event_maximum_absolute_ratio=.05,off_event_population_cv=.05)
    for name,digest in freeze['files'].items():
        assert sha(run/name) == sha(data/name) == digest, name
    assert read(run/'freeze.json') == read(data/'freeze.json')
    for name in ('freeze.json','input.tar.gz'):
        path = run/name
        blob = subprocess.check_output(['git','show',FREEZE_COMMIT+':'+str(path.relative_to(repo))],cwd=repo)
        assert hashlib.sha256(blob).hexdigest() == sha(path), name
    commit_time = datetime.fromisoformat(subprocess.check_output(
        ['git','show','-s','--format=%cI',FREEZE_COMMIT],cwd=repo).decode().strip())

    def check_archive(path, directory, expected):
        read(path)
        members = {}
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = member.name.removeprefix('./')
                assert name not in members and not name.startswith('/') and '..' not in Path(name).parts
                content = archive.extractfile(member).read()
                assert content == read(directory/name), name
                members[name] = hashlib.sha256(content).hexdigest()
        assert set(expected) <= set(members)
        return len(members)

    input_files = check_archive(run/'input.tar.gz',run,[*freeze['files'],'freeze.json'])
    receipt = load(run/'receipt.json')
    assert receipt['returncode'] == 0 and sha(run/'output.tar.gz') == receipt['sha256']
    output_files = check_archive(run/'output.tar.gz',data,[*freeze['files'],'freeze.json','complete.json'])
    assert read(data/'exit_code.txt').decode().strip() == '0'
    build = Path(manifest['build_run'])
    assert sha(build/'build-audit.json') == manifest['build_audit_sha256']
    assert sha(build/'returned/build/cfg_c_s4_dual') == manifest['binary_sha256']
    ordering = load(ordering_review)
    assert ordering['binary_sha256'] == manifest['binary_sha256']
    assert ordering['source_and_sass_dependency_review_passed'] is True
    assert not any(ordering['qualification'].values())
    correctness_path = Path(manifest['correctness_record'])
    assert sha(correctness_path) == manifest['correctness_sha256']
    correct = load(correctness_path)
    assert correct['full_D_reference_ok'] and correct['results'][0]['off_on_bit_equal']

    output.mkdir(parents=True,exist_ok=False)
    snapshot = output/'analysis_snapshot'
    snapshot.mkdir()
    for path in sorted((run/'analysis_snapshot').glob('*.py')):
        content = read(path)
        assert hashlib.sha256(content).hexdigest() == manifest['analysis_sha256'][path.name]
        (snapshot/path.name).write_bytes(content)
    (snapshot/'validate_allocation.py').write_bytes(read(run/'validate_allocation.py'))
    test_names = ['test_cfg_c_producer_tail_packet','test_cfg_c_tail_control',
                  'test_cfg_c_gt_bracket_packet','test_cfg_c_gt_control']
    for name in test_names:
        path = repo/'microbench/gh200_resource_campaign/access_rules'/(name+'.py')
        content = read(path)
        frozen = subprocess.check_output(['git','show',FREEZE_COMMIT+':'+str(path.relative_to(repo))],cwd=repo)
        assert content == frozen, name
        (snapshot/path.name).write_bytes(content)
    tests = subprocess.run([sys.executable,'-B','-m','unittest','-v',*test_names],cwd=snapshot,
        capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    (output/'tests.txt').write_text(tests.stdout+tests.stderr)
    assert tests.returncode == 0 and 'Ran 26 tests' in tests.stderr
    sys.path.insert(0,str(snapshot))
    from analyze_cfg_c_gt_control import process,evaluate
    from validate_allocation import validate

    allocation = read(data/'allocation.txt').decode()
    validate(allocation)
    fields = dict(re.findall(r'(\w+)=(\S+)',allocation))
    assert fields['JobId'] == '740161' and fields['Restarts'] == '0'
    submit_time = datetime.fromisoformat(fields['SubmitTime']).replace(tzinfo=ZoneInfo('Europe/Paris'))
    assert commit_time <= submit_time
    context = load(data/'assigned-context.json')
    assert context['SLURM_JOB_ID'] == '740161'
    assert context['CUDA_VISIBLE_DEVICES'] == context['SLURM_STEP_GPUS'] == '0,1,2,3'
    setup = json.loads(read(data/'setup.stdout'))
    assert setup['gpu_uuid'] == manifest['gpu_uuid']
    records = load(data/'records.json')
    sample_fields = ('id','comparison','pair','arm','flag','command')
    assert len(records) == len(manifest['samples']) == 40
    assert [{name:record[name] for name in sample_fields} for record in records] == manifest['samples']
    assert load(data/'complete.json') == dict(manifest_sha256=sha(run/'manifest.json'),records=40)
    correctness_manifest = load(correctness_path.parent/'manifest.json')
    command_by_flag = {sample['flag']:sample['command'] for sample in correctness_manifest['samples']
                       if sample['id'] in ('flag0','flag1')}
    for index,sample in enumerate(manifest['samples']):
        comparison = 'off_on' if index < 20 else 'off_off'
        pair = (index%20)//2
        arm = (index%2) if pair%2 == 0 else 1-(index%2)
        assert sample['comparison'] == comparison
        assert type(sample['pair']) is int and sample['pair'] == pair
        assert type(sample['arm']) is int and sample['arm'] == arm
        assert type(sample['flag']) is int and sample['flag'] == (arm if comparison == 'off_on' else 0)
        assert sample['command'] == command_by_flag[sample['flag']]

    # Independent integer-product reference for all 4096 values in every record.
    reference = []
    for row in range(17):
        line = []
        for col in range(17):
            products = [((7*row+13*k)%17-8)*((5*k+11*col)%17-8) for k in range(17)]
            line.append((sum(products)*240+sum(products[:16]))/1024)
        reference.append(line)
    expected_counts = dict(producer=132,GLOBAL_ENTRY=132,GLOBAL_PRODUCER=132,
        FINAL=264,GLOBAL_FINAL=264,TAIL=264,
        **{name:432 for name in ('FIRST','MAIN','PERMIT','DONE','L','W','E','GAP',
                                'GLOBAL_FIRST','GLOBAL_MAIN','GLOBAL_PERMIT','GLOBAL_DONE')})
    cohorts = {'off_on':[],'off_off':[]}
    numeric_values = telescoping = global_identities = zero_tail_ctas = 0
    width_histogram = Counter()
    for sample in records:
        raw = data/sample['raw']
        assert sample['returncode'] == 0 and sha(raw) == sample['sha256']
        events = [json.loads(line) for line in gzip.decompress(read(raw)).decode().splitlines()]
        assert [event['event'] for event in events] == ['setup','call','check']
        setup,call,check = events
        assert check['status'] == 'ok' and check['max_storage_reference_error'] == 0
        packet = process(events,sample,manifest['gpu_uuid'])
        trace,shadow = call['trace'],call['gt_shadow']
        assert len(trace) == 132*1296 and len(shadow) == 132*9
        assert setup['producer_tail_observer'] == call['producer_tail_observer'] == 0
        assert setup['gt_bracket_observer'] == call['gt_bracket_observer'] == sample['flag']
        origin = min(trace[1296*cta+1] for cta in range(132))
        independent = {}
        brackets = {}
        for index,value in zip(check['checked_indices'],check['checked_values']):
            assert value == reference[(index//3072)%17][(index%3072)%17]
            numeric_values += 1
        for cta in range(132):
            head = trace[1296*cta:1296*(cta+1)]
            assert head[14] == head[15] == 0
            zero_tail_ctas += 1
            assert shadow[cta*9+8] == 0
            if not sample['flag']:
                assert shadow[cta*9:cta*9+9] == [0]*9
            entry,producer = [head[1],head[0]],[head[13],head[3]]
            independent[f'{cta}/GLOBAL_ENTRY'] = [entry[0]-origin,None]
            independent[f'{cta}/GLOBAL_PRODUCER'] = [producer[0]-origin,None]
            independent[f'{cta}/producer'] = [producer[d]-entry[d] for d in (0,1)]
            for wg in (0,1):
                final = [head[5+3*wg],head[4+3*wg]]
                independent[f'{cta}/{wg}/GLOBAL_FINAL'] = [final[0]-origin,None]
                independent[f'{cta}/{wg}/FINAL'] = [final[d]-entry[d] for d in (0,1)]
                previous = producer
                accumulated = independent[f'{cta}/producer'][:]
                count = head[6+3*wg]
                for ordinal in range(count):
                    tile = head[16+640*wg+10*ordinal:26+640*wg+10*ordinal]
                    prefix = f'{cta}/{wg}/{ordinal}/{tile[4]-1},{tile[5]-1}/'
                    stamps = [[tile[6+event],tile[event]] for event in range(4)]
                    for name,stamp in zip(('FIRST','MAIN','PERMIT','DONE'),stamps):
                        independent[prefix+name] = [stamp[d]-entry[d] for d in (0,1)]
                        independent[prefix+'GLOBAL_'+name] = [stamp[0]-origin,None]
                        assert independent[prefix+'GLOBAL_'+name][0] == independent[prefix+name][0]+entry[0]-origin
                        global_identities += 1
                    independent[prefix+'GAP'] = [stamps[0][d]-previous[d] for d in (0,1)]
                    for name,a,b in (('L',0,1),('W',1,2),('E',2,3)):
                        independent[prefix+name] = [stamps[b][d]-stamps[a][d] for d in (0,1)]
                    for d in (0,1):
                        accumulated[d] += independent[prefix+'GAP'][d]+sum(independent[prefix+name][d] for name in ('L','W','E'))
                    previous = stamps[3]
                    if sample['flag']:
                        before = shadow[cta*9+wg*4+ordinal*2:cta*9+wg*4+ordinal*2+2]
                        after = tile[6:8]
                        assert 0 < before[0] <= after[0] < before[1] <= after[1]
                        bracket_stamps = [dict(gt_before=before[e],cycles=tile[e],gt_after=after[e]) for e in (0,1)]
                        widths = [after[e]-before[e] for e in (0,1)]
                        width_histogram.update(widths)
                        brackets[prefix+'L'] = dict(stamps=bracket_stamps,
                            conditional_elapsed_gt=[before[1]-after[0],after[1]-before[0]],post_gt_error_bound=max(widths))
                tail = [final[d]-previous[d] for d in (0,1)]
                independent[f'{cta}/{wg}/TAIL'] = tail
                for d in (0,1):
                    assert accumulated[d]+tail[d] == final[d]-entry[d]
                    telescoping += 1
                assert all(x == 0 for x in shadow[cta*9+wg*4+count*2:cta*9+(wg+1)*4])
        assert independent == packet['series'] and brackets == packet['brackets']
        assert len(independent) == 6372
        assert dict(Counter(key.split('/')[-1] for key in independent)) == expected_counts
        cohorts[sample['comparison']].append(packet)

    main = load(run/'analysis.json')
    summary = load(run/'summary.json')
    assert main['record_errors'] == summary['record_errors'] == []
    assert main['manifest_sha256'] == summary['manifest_sha256'] == sha(run/'manifest.json')
    assert main['observation_qualified'] is summary['observation_qualified'] is False
    results = {}
    for comparison,processes in cohorts.items():
        pairs = []
        for pair in range(10):
            rows = sorted(processes[2*pair:2*pair+2],key=lambda row:row['arm'])
            assert [row['arm'] for row in rows] == [0,1]
            pairs.append(rows)
        local = []
        for key in sorted(processes[0]['series']):
            signed = [b['series'][key][0]-a['series'][key][0] for a,b in pairs]
            absolute = sorted(abs(value) for value in signed)
            median,maximum = (absolute[4]+absolute[5])/2,absolute[-1]
            local.append(dict(series=key,signed=signed,median_absolute=median,maximum_absolute=maximum,
                passed=median<=64 and maximum<=256,
                native_cycle_deltas=[None if a['series'][key][1] is None else b['series'][key][1]-a['series'][key][1] for a,b in pairs],
                sm_changed=[a['sms'][key.split('/')[0]] != b['sms'][key.split('/')[0]] for a,b in pairs]))
        signed = [b['elapsed_us']/a['elapsed_us']-1 for a,b in pairs]
        absolute = sorted(abs(value) for value in signed)
        event = dict(signed=signed,median_absolute=(absolute[4]+absolute[5])/2,maximum_absolute=absolute[-1],
                     passed=(absolute[4]+absolute[5])/2<=.02 and absolute[-1]<=.05)
        off_times = [a['elapsed_us'] for a,b in pairs]
        cv = statistics.pstdev(off_times)/statistics.fmean(off_times)
        result = main['cohorts'][comparison]
        assert result['local'] == local and result['event'] == event and result['off_event_population_cv'] == cv
        sub = dict(manifest,comparison=comparison,samples=[s for s in manifest['samples'] if s['comparison']==comparison])
        assert evaluate(processes,sub) == result
        assert result['paired_interval_differences_available'] is False
        assert result['gt_bracket_precision_qualified'] is result['hardware_sampling_enclosure'] is False
        assert result['observation_qualified'] is result['baseline_dual_qualified'] is False
        control_key = 'incremental_control_passed' if comparison == 'off_on' else 'baseline_consistency_passed'
        assert result[control_key] is False
        if comparison == 'off_off':
            assert 'incremental_control_passed' not in result and result['enabled_read_diagnostics'] == []
        diagnostic = []
        if comparison == 'off_on':
            enabled = [row for row in processes if row['flag']==1]
            for key in sorted(enabled[0]['brackets']):
                bounds = [row['brackets'][key]['post_gt_error_bound'] for row in enabled]
                widths = [[stamp['gt_after']-stamp['gt_before'] for stamp in row['brackets'][key]['stamps']] for row in enabled]
                diagnostic.append(dict(series=key,point_error_bounds=bounds,read_widths=widths,
                    median_point_error_bound=statistics.median(bounds),maximum_point_error_bound=max(bounds),diagnostic_only=True))
        assert diagnostic == result['enabled_read_diagnostics']
        groups = {}
        for name in expected_counts:
            rows = [row for row in local if row['series'].split('/')[-1]==name]
            groups[name] = dict(series=len(rows),failed=sum(not row['passed'] for row in rows),
                median_absolute_range=[min(row['median_absolute'] for row in rows),max(row['median_absolute'] for row in rows)],
                maximum_absolute_range=[min(row['maximum_absolute'] for row in rows),max(row['maximum_absolute'] for row in rows)])
        expected_failures = 4912 if comparison == 'off_on' else 4729
        assert sum(not row['passed'] for row in local) == expected_failures
        assert groups['L']['failed'] == 432 and groups['FINAL']['failed'] == 264
        assert not any(any(row['sm_changed']) for row in local)
        assert event['passed'] and cv<=.05
        assert summary['cohorts'][comparison]['by_metric'] == groups
        assert summary['cohorts'][comparison]['failed_point_series'] == expected_failures
        assert summary['cohorts'][comparison]['event'] == event
        assert summary['cohorts'][comparison]['off_event_population_cv'] == cv
        results[comparison] = dict(processes=20,series_per_process=6372,failed=expected_failures,
            groups=groups,event=event,off_event_population_cv=cv,necessary_control_passed=False,
            observation_qualified=False,all_local_and_evaluator_fields_equal_main=True)
    assert numeric_values == 163840 and telescoping == 21120 and global_identities == 69120
    assert zero_tail_ctas == 5280 and width_histogram == Counter({0:670,32:7240,64:718,96:12})
    assert sum(width_histogram.values()) == 8640
    assert summary['cohorts']['off_on']['read_width_histogram'] == {str(k):v for k,v in sorted(width_histogram.items())}
    assert summary['cohorts']['off_off']['read_width_histogram'] == {}
    # Ensure public reports have not been mutated by this replay.
    assert sha(run/'analysis.json') == public_analysis_sha
    assert sha(run/'summary.json') == public_summary_sha
    read(Path(__file__))
    report = dict(schema='C-GT-control-review/1',input_sha256=inputs,freeze_commit=FREEZE_COMMIT,
        freeze_commit_time_Paris=commit_time.astimezone(ZoneInfo('Europe/Paris')).isoformat(),
        job_submit_time_Paris=submit_time.isoformat(),job=740161,input_archive_files=input_files,
        output_archive_files=output_files,prescribed_target_processes=40,no_record_or_pair_dropped=True,
        public_analysis_and_summary_unchanged=True,
        phase_order=['off_on_20_targets','off_off_20_targets'],numeric_values_independently_verified=numeric_values,
        integer_telescoping_identities=telescoping,global_origin_identities=global_identities,
        zero_producer_tail_CTAs=zero_tail_ctas,CPU_tests_passed=26,cohorts=results,
        enabled_read_stamps=8640,enabled_read_width_counts=dict(sorted(width_histogram.items())),
        disabled_GT_pre_available=False,paired_interval_differences_available=False,
        on_on_created=False,cohort_centers_subtracted=False,V09_candidates_changed=False,
        interpretation='Enabling GT-pre/helper is not necessary for same-binary local paired differences. Neither observer harmlessness nor pure timer error is identified by these two phased cohorts.',
        qualification=dict(hardware_sampling_enclosure=False,physical_ns_accuracy=False,observation=False,
            gt_bracket_precision=False,baseline_dual=False,local_prediction=False,critical_CTA=False,whole_gemm=False),
        C_new_gpu_processes=0,public_probe_reader_runner_or_model_changed_by_C=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(inputs=len(inputs),targets=40,numeric_values=numeric_values,CPU_tests=26,
        failed={name:value['failed'] for name,value in results.items()},widths=dict(sorted(width_histogram.items())))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--ordering-review',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    review(args.run.resolve(),args.ordering_review.resolve(),args.output.resolve())

#!/usr/bin/env python3
"""Unqualified record arithmetic for the existing two-map correctness run.

No causal map estimate, performance gate, service fitting or GPU execution.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile


EVENTS = ('FIRST','COMMIT_G0','COMMIT_G16','COMMIT_G32','COMMIT_G63','MMA_RETURN','MAIN')
SELECTED = (0,1,6,7,14,15,22,23,30,31,38,39)
WORDS = 1318


def analyze(run, output):
    if output.exists():
        raise ValueError('output must be a new C-only directory')
    root = run.parent
    build = root/'20261011-R18-first-output-build'
    inputs = {}

    def read(path):
        data = path.read_bytes()
        inputs[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data

    def load(path):
        return json.loads(read(path))

    manifest = load(run/'manifest.json')
    assert read(run/'manifest.json') == read(run/'returned/manifest.json')
    assert hashlib.sha256(read(run/'returned/run.sh')).hexdigest() == manifest['run_sha256']
    assert read(run/'run.sh') == read(run/'returned/run.sh')
    binary_sha = hashlib.sha256(read(run/'returned/build/cfg_a_s6_dual')).hexdigest()
    assert binary_sha == manifest['binary_sha256'] == hashlib.sha256(read(build/'returned/build/cfg_a_s6_dual')).hexdigest()
    assert read(run/'returned/exit_code.txt').decode().strip() == '0'
    receipt = load(run/'receipt.json')
    assert receipt['returncode'] == 0
    assert hashlib.sha256(read(run/'output.tar.gz')).hexdigest() == receipt['sha256']
    archive_files = 0
    with tarfile.open(run/'output.tar.gz') as archive:
        for member in archive:
            if member.isfile():
                name = member.name.removeprefix('./')
                assert not name.startswith('/') and '..' not in Path(name).parts
                assert archive.extractfile(member).read() == read(run/'returned'/name), name
                archive_files += 1
    allocation = read(run/'returned/allocation.txt').decode()
    fields = dict(re.findall(r'(\w+)=(\S+)',allocation))
    assert fields['JobId'] == '740067' and fields['NodeList'] == 'romeo-a057'
    assert fields['NumCPUs'] == '4' and fields['OverSubscribe'] == 'OK'
    assert 'gres/gpu=1' in fields['AllocTRES']

    contract = load(root/'20261010-R18-cluster-fill-discrimination/reanalysis/C-20261011-first-output-source-ports-v1/first-output-source-ports.json')
    assert contract['fixed_scope']['ctas'] == list(SELECTED)
    assert contract['existing_endpoints'] == ['FIRST','MAIN']
    assert [point['event'] for point in contract['source_ports']] == list(EVENTS[1:-1])
    static = {extent:{row['cta']:row['work'] for row in selection['selected']}
              for extent,selection in zip((960,1024),contract['static_selection'])}
    assert static[960] == static[1024]
    source_review = load(root/'20261011-R18-first-output-control/reanalysis/C-20261011-output-clock-boundaries-v1/source-review.json')
    assert source_review['binary_sha256'] == binary_sha and source_review['qualified'] is False
    for projection in source_review['projection']:
        assert projection['projection_exact']
        assert hashlib.sha256(read(Path(projection['source']))).hexdigest() == projection['source_sha256']
    assert hashlib.sha256(read(Path(source_review['sass_path']))).hexdigest() == source_review['sass_sha256']
    host_path = build/'source/probes/r18.cu'
    host = read(host_path).decode()
    needles = ['unsigned first_output_off=0;',
        'CUDA_CHECK(cudaMemcpyToSymbol(r18_first_output_observer,&first_output_off,sizeof(unsigned)));',
        'warmup_launch_params.mainloop = launch_params.mainloop;',
        'warm.push_back(timed(warmup_launch_params));', 'set_target_epilogue();',
        'float ms = timed(launch_params);']
    for needle in needles:
        assert needle in host, needle
    source_lines = {needle:host.count('\n',0,host.index(needle))+1 for needle in needles}
    assert source_lines[needles[1]] < source_lines[needles[2]] < source_lines[needles[3]]

    # Reuse previously verified full-D evidence only after matching its exact bytes.
    prior_path = root/'20261011-R18-first-output-control/reanalysis/C-20261011-output-clock-boundaries-v1/data/diagnosis.json.gz'
    prior = json.loads(gzip.decompress(read(prior_path)))
    for item in prior['numerical']:
        assert item['exact'] is True and item['values'] == 960*2816
        path = run/f'returned/full-d/map{item["map_m"]}.bin'
        assert hashlib.sha256(read(path)).hexdigest() == item['sha256']
    assert len(prior['numerical']) == 2
    verification = load(run/'verification.json')
    saved = {(row['map'],row['flag']):row for row in verification['records']}
    assert verification['errors'] == [] and verification['observation_qualified'] is False
    controls = load(root/'20261011-R18-first-output-control/analysis.json')
    control_manifest = load(root/'20261011-R18-first-output-control/manifest.json')
    assert controls['common_L_control_passed'] is controls['internal_endpoints_qualified'] is controls['observation_qualified'] is False

    table = []
    for row in range(17):
        values = []
        for col in range(17):
            products = [((7*row+13*k)%17-8)*((5*k+11*col)%17-8) for k in range(17)]
            values.append((sum(products)*240+sum(products[:16]))/1024)
        table.append(values)
    records = []
    partitions = []
    identities = []
    sampled_values = 0
    for extent in (960,1024):
        for flag in (0,1):
            invocation = f'map{extent}_flag{flag}'
            path = run/'returned/raw'/(invocation+'.jsonl.gz')
            raw_bytes = read(path)
            raw_sha = hashlib.sha256(raw_bytes).hexdigest()
            assert raw_sha == saved[extent,flag]['raw_sha256']
            packets = [json.loads(line) for line in gzip.decompress(raw_bytes).decode().splitlines()]
            assert [packet['event'] for packet in packets] == ['setup','call','input_zero_storage','check']
            setup,call,zero_check,check = packets
            assert zero_check['checked'] == 64*4096 and zero_check['nonzero'] == 0
            fixed = dict(config='cfg_a',m=960,n=2816,k=4096,grid=[132,1,1],tile=[128,128,64],
                cluster=[2,1],stages=6,trace_words=WORDS,trace_tile_words=10,trace_version='r18-first-output-v1',
                swizzle=1,storage_m=1024,storage_n=2816,zero_m=960,zero_n=-1,input_map_m=extent,
                input_map_n=2816,lda=4096,ldb=2816,ldd=2816,input_mode='dyadic',seed=17)
            for name,value in fixed.items():
                assert setup[name] == value, name
            history_fields = dict(warmup_policy='common-sw1-16',warmup_swizzle=1,warmup_fixed_calls=16,
                history_protocol='none',history_source='original',mirror_target_slot=0,source_copies=1,
                first_output_observer=flag,first_output_warmup=0,release_warp_observer=0,release_warp_warmup=0,
                epilogue_defer=0,epilogue_warmup_defer=0,second_prefill=0,second_prefill_warmup=0,
                a_row_rotation=0,first_supply_events=0,producer_delay_ns=0,work_swap55_91=0,b_offset_bytes=0)
            for packet in (setup,call):
                for name,value in history_fields.items():
                    assert packet[name] == value, name
                assert packet['target_a_ptr'] == packet['prefix_a_ptr']
                assert packet['target_b_ptr'] == packet['prefix_b_ptr'] == packet['b_buffer_ptr']
            assert call['warmup_calls'] == len(call['warmup_us']) == 16
            assert setup['gpu_uuid'] == saved[extent,flag]['gpu_uuid'] == 'GPU-43269fbc-449d-3e0f-908a-9c81229546d3'
            trace = call['trace']
            assert len(trace) == 132*WORDS and all(type(value) is int and 0<=value<2**64 for value in trace)
            assert check['status'] == 'ok' and check['samples'] == len(check['checked_indices']) == len(check['checked_values']) == 4096
            for index,value in zip(check['checked_indices'],check['checked_values']):
                assert value == table[(index//2816)%17][(index%2816)%17]
                sampled_values += 1
            identities.append(dict(invocation=invocation,raw_sha256=raw_sha,map_m=extent,flag=flag,
                gpu_uuid=setup['gpu_uuid'],binary_sha256=binary_sha,history_fields=history_fields,
                target_a_ptr=setup['target_a_ptr'],target_b_ptr=setup['target_b_ptr'],
                prefix_a_ptr=setup['prefix_a_ptr'],prefix_b_ptr=setup['prefix_b_ptr'],
                warmup_us=call['warmup_us'],warmup_last5_cv=call['warmup_last5_cv'],
                elapsed_us_record_only=call['elapsed_us'],prefix_tensor_map_follows_target=True))
            for cta in range(132):
                block = trace[cta*WORDS:(cta+1)*WORDS]
                if cta not in SELECTED:
                    assert not any(block[1296:])
                    continue
                assert block[10] == 0 and 0 < block[2] <= 132
                for role in (1,2):
                    assert block[10+role] == block[2]
                    head = 16+(role-1)*640
                    assert block[head+4:head+6] == [value+1 for value in static[extent][cta][0]]
                    extra = block[1296+(role-1)*11:1296+role*11]
                    assert extra[0] == 0x52313810+flag
                    if not flag:
                        assert extra[1:] == [0]*10
                        continue
                    stamps = {'FIRST':dict(clock64=block[head],globaltimer_ns=block[head+6]),
                              'MAIN':dict(clock64=block[head+1],globaltimer_ns=block[head+7])}
                    for index,event in enumerate(EVENTS[1:-1]):
                        stamps[event] = dict(clock64=extra[1+2*index],globaltimer_ns=extra[2+2*index])
                    record = dict(invocation_id=invocation,config='cfg_a',cta=cta,role=role,
                        physical_thread=role*128,tile_seq=0,sm_start=block[2]-1,sm_end=block[10+role]-1,stamps=stamps)
                    expected = next(row for row in saved[extent,flag]['decoded']['records']
                                    if row['record']['cta']==cta and row['record']['role']==role)
                    assert record == expected['record']
                    windows = []
                    for start,end in zip(EVENTS,EVENTS[1:]):
                        cycles = stamps[end]['clock64']-stamps[start]['clock64']
                        ns = stamps[end]['globaltimer_ns']-stamps[start]['globaltimer_ns']
                        assert cycles>=0 and ns>=0
                        windows.append(dict(start=start,end=end,cycles=cycles,ns=ns))
                        partitions.append(dict(map_m=extent,cta=cta,role=role,sm=block[2]-1,
                            physical_thread=role*128,coord_m=static[extent][cta][0][0],
                            coord_n=static[extent][cta][0][1],start=start,end=end,cycles=cycles,ns=ns))
                    duration = dict(cycles=stamps['MAIN']['clock64']-stamps['FIRST']['clock64'],
                                    ns=stamps['MAIN']['globaltimer_ns']-stamps['FIRST']['globaltimer_ns'])
                    assert all(sum(window[unit] for window in windows)==duration[unit] for unit in ('cycles','ns'))
                    assert windows == expected['partition']['windows'] and duration == expected['partition']['L']
                    records.append(dict(map_m=extent,record=record,windows=windows,L=duration))
    assert sampled_values == 16384 and len(records) == 48 and len(partitions) == 288
    indexed = {(row['map_m'],row['record']['cta'],row['record']['role']):row for row in records}
    differences = []
    role_differences = []
    for cta in SELECTED:
        for role in (1,2):
            a,b = indexed[960,cta,role],indexed[1024,cta,role]
            signed = dict(cycles=a['L']['cycles']-b['L']['cycles'],ns=a['L']['ns']-b['L']['ns'])
            sm960,sm1024 = a['record']['sm_start'],b['record']['sm_start']
            role_differences.append(dict(cta=cta,role=role,map960_sm=sm960,map1024_sm=sm1024,
                sm_changed=sm960!=sm1024,map960_L=a['L'],map1024_L=b['L'],signed_record_difference=signed))
            rows = []
            for left,right in zip(a['windows'],b['windows']):
                row = dict(cta=cta,role=role,start=left['start'],end=left['end'],map960_sm=sm960,map1024_sm=sm1024,
                    map960_cycles=left['cycles'],map1024_cycles=right['cycles'],signed_cycles=left['cycles']-right['cycles'],
                    map960_ns=left['ns'],map1024_ns=right['ns'],signed_ns=left['ns']-right['ns'])
                rows.append(row)
                differences.append(row)
            assert sum(row['signed_cycles'] for row in rows)==signed['cycles']
            assert sum(row['signed_ns'] for row in rows)==signed['ns']
    assert len(differences)==144 and not any(row['sm_changed'] for row in role_differences)
    read(Path(__file__))
    report = dict(schema='C-first-output-correctness-record-partitions/1',input_sha256=inputs,
        job=740067,node='romeo-a057',GPU_uuid=identities[0]['gpu_uuid'],binary_sha256=binary_sha,
        archive_files_verified=archive_files,allocation=dict(GPUs=1,CPUs=4,exclusive=False),
        actual_command_order=['map960_full_check','map960_flag0','map960_flag1',
                              'map1024_full_check','map1024_flag0','map1024_flag1'],
        selected_CTAs=list(SELECTED),roles=[1,2],on_records=48,partition_rows=288,
        signed_record_difference_rows=144,exact_window_telescopes=96,exact_signed_telescopes=48,
        saved_verification_records_and_windows_equal=True,all_role_SM_identities_unchanged_across_calls=True,
        independently_checked_warm_numeric_values=16384,reused_full_D_evidence=prior['numerical'],
        source_history_lines=source_lines,prefix_input_map_same_as_own_target=True,
        identical_history_state_between_maps_proven=False,
        control_gpu_uuid=control_manifest['gpu_uuid'],same_card_qualified_observer_control_available=False,
        prior_common_L_control_passed=False,old_control_result_changed=False,
        role_record_differences=role_differences,
        comparison_status='Two ordered correctness invocations; arithmetic difference only, not a randomized or qualified map effect.',
        unresolved=['No repeated or exclusive two-map observations in this run.',
            'Different target maps also supply different maps to each invocation\'s sixteen prefix calls.',
            'Full waits, issue/commit, group waits and releases are combined inside each K-loop window.',
            'These independent SR reads do not provide GT-before/clock/GT-after sampling brackets.',
            'The records cannot identify bare service prices, current-map effects separated from history, or observer effects.'],
        qualification=dict(observer=False,hardware_sampling_enclosure=False,physical_ns_accuracy=False,
            map_causal_effect=False,performance_estimate=False,service_price=False,
            local_holdout=False,critical_CTA=False,whole_gemm=False),
        GPU_processes_added_by_C=0,public_code_changed_by_C=False,V09_candidates_changed=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'call-identities.json').write_text(json.dumps(identities,indent=2,allow_nan=False)+'\n')
    (output/'records.json.gz').write_bytes(gzip.compress(json.dumps(records,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    for name,rows in [('record-partitions.csv',partitions),('signed-record-differences.csv',differences)]:
        with (output/name).open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(inputs=len(inputs),on_records=48,partitions=288,signed_rows=144,
        numeric_values=16384,qualified=False,gpu_uuid=report['GPU_uuid'])))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    analyze(args.run.resolve(),args.output.resolve())

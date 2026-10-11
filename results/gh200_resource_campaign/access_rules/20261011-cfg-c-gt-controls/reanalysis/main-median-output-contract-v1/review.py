#!/usr/bin/env python3
"""C median contract audit: same-call composition before cross-call statistics."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def zipped_json(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    with Path(path).open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, filename='', mode='wb', mtime=0) as output:
            output.write(raw)


def gap_summary(rows):
    gaps = [r['median_composed_minus_composed_medians'] for r in rows]
    return {'series': len(gaps), 'nonzero': sum(g != 0 for g in gaps),
            'minimum': min(gaps), 'maximum': max(gaps),
            'median_absolute_gap': statistics.median(abs(g) for g in gaps)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = Path(__file__).resolve().parent
    inputs = {}
    def load(path):
        inputs[str(path.resolve())] = sha(path)
        return json.loads(path.read_text())
    manifest = load(args.run / 'manifest.json')
    records = load(args.run / 'returned/records.json')
    protocol = load(base / 'source/protocol-proposal.json')
    inputs[str((base/'source/evaluate.py').resolve())] = sha(base/'source/evaluate.py')
    assert protocol['original_manifest_sha256'] == sha(args.run/'manifest.json')
    assert protocol['repeats_per_arm'] == 64 and len(protocol['series']) == 6372
    for name, digest in manifest['analysis_sha256'].items():
        path = args.run / 'analysis_snapshot' / name
        assert sha(path) == digest
        inputs[str(path.resolve())] = digest
    sys.path.insert(0, str(args.run/'analysis_snapshot'))
    from analyze_cfg_c_gt_control import process
    spec = importlib.util.spec_from_file_location('candidate_evaluate', base/'source/evaluate.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    prescribed = {row['id']: row for row in manifest['samples']}
    calls = defaultdict(list)
    seen = set()
    for record in sorted(records, key=lambda r:(r['comparison'],r['arm'],r['pair'])):
        assert record['id'] not in seen and record['returncode'] == 0
        seen.add(record['id'])
        for key, value in prescribed[record['id']].items():
            assert record[key] == value
        path = args.run/'returned'/record['raw']
        assert sha(path) == record['sha256']
        inputs[str(path.resolve())] = record['sha256']
        with gzip.open(path, 'rt') as stream:
            events = [json.loads(line) for line in stream]
        decoded = process(events,record,manifest['gpu_uuid'])
        assert set(decoded['series']) == set(protocol['series'])
        calls[record['comparison'],record['arm']].append(decoded)
    assert seen == set(prescribed)
    per_role_rows, merged_rows, envelopes = [], [], []
    exact_checks = {'role_window_telescopes':0, 'merged_window_telescopes':0}
    for (cohort,arm), batch in sorted(calls.items()):
        assert len(batch) == 10
        names = sorted(batch[0]['series'])
        role_prefixes = [name[:-1] for name in names if name.endswith('/L')]
        final_keys = [name for name in names if name.endswith('/GLOBAL_FINAL')]
        groups = defaultdict(list)
        for prefix in role_prefixes:
            cta,wg,j,coord,empty = prefix.split('/')
            groups[cta,j,coord].append(prefix)
            components, totals = [], []
            for call in batch:
                s = call['series']
                parts = [s[prefix+metric][0] for metric in ['L','W','E']]
                total = s[prefix+'DONE'][0]-s[prefix+'FIRST'][0]
                assert sum(parts) == total
                exact_checks['role_window_telescopes'] += 1
                components.append(parts)
                totals.append(total)
            sum_medians = sum(statistics.median(values[i] for values in components) for i in range(3))
            median_composed = statistics.median(totals)
            per_role_rows.append({'cohort':cohort,'arm':arm,'series':prefix,
                                  'composed_medians':sum_medians,'median_composed':median_composed,
                                  'median_composed_minus_composed_medians':median_composed-sum_medians})
        for key,prefixes in sorted(groups.items()):
            assert len(prefixes) == 2
            windows, totals = [], []
            for call in batch:
                s = call['series']
                main,permit,done = [max(s[prefix+'GLOBAL_'+name][0] for prefix in prefixes)
                                    for name in ['MAIN','PERMIT','DONE']]
                W,E = permit-main,done-permit
                assert W+E == done-main
                exact_checks['merged_window_telescopes'] += 1
                windows.append((W,E));totals.append(done-main)
            sum_medians = sum(statistics.median(pair[i] for pair in windows) for i in range(2))
            median_composed = statistics.median(totals)
            merged_rows.append({'cohort':cohort,'arm':arm,'series':'/'.join(key),
                                'composed_medians':sum_medians,'median_composed':median_composed,
                                'median_composed_minus_composed_medians':median_composed-sum_medians})
        per_call_max = [max(call['series'][key][0] for key in final_keys) for call in batch]
        max_medians = max(statistics.median(call['series'][key][0] for call in batch) for key in final_keys)
        envelopes.append({'cohort':cohort,'arm':arm,'per_call_recorded_consumer_FINAL_envelope':per_call_max,
                          'median_of_per_call_envelope':statistics.median(per_call_max),
                          'max_of_coordinate_medians':max_medians,
                          'median_envelope_minus_max_medians':statistics.median(per_call_max)-max_medians,
                          'scope':'Recorded common-origin GT coordinates only. Not physical ns, source_safe_exit or CUDA completion.'})

    # Synthetic record arithmetic: preserve all window fields from one validated
    # record and change only terminal waiting envelopes and matching FINAL/TAIL.
    template = calls['off_on',0][0]['series']
    role_keys = sorted((name[:-len('GLOBAL_FINAL')] for name in template if name.endswith('/GLOBAL_FINAL')),
                      key=lambda p:tuple(map(int,p.strip('/').split('/'))))
    targets = [prefix for prefix in role_keys if prefix.split('/')[1]=='1'][:8]
    assert len(targets) == 8
    floor = max(template[name][0] for name in template if name.endswith('/GLOBAL_FINAL')) + 1000
    extra = 1000  # A symbolic witness amount, never a calibrated price.
    n = 64
    synth = {cohort:{name:{'arm0':[template[name][0]]*n,'arm1':[template[name][0]]*n}
                    for name in protocol['series']} for cohort in ['off_on','off_off']}
    tail_origin = {}
    for prefix in role_keys:
        cta,wg = prefix.strip('/').split('/')
        role_outputs = [p for p in role_prefixes if p.startswith(prefix)]
        last = max(role_outputs,key=lambda p:int(p.split('/')[2]))
        tail_origin[prefix] = template[last+'GLOBAL_DONE'][0]
        for cohort in synth:
            for arm in ['arm0','arm1']:
                synth[cohort][prefix+'GLOBAL_FINAL'][arm] = [floor]*n
                synth[cohort][prefix+'FINAL'][arm] = [floor-template[cta+'/GLOBAL_ENTRY'][0]]*n
                synth[cohort][prefix+'TAIL'][arm] = [floor-tail_origin[prefix]]*n
    for index,prefix in enumerate(targets):
        for cohort in synth:
            for arm in ['arm0','arm1']:
                high = range(index*8,(index+1)*8) if cohort=='off_on' and arm=='arm1' else range(8)
                for repeat in high:
                    for metric in ['GLOBAL_FINAL','FINAL','TAIL']:
                        synth[cohort][prefix+metric][arm][repeat] += extra
    marginal_equal = all(sorted(row['arm0'])==sorted(row['arm1']) for cohort in synth.values() for row in cohort.values())
    assert marginal_equal
    checked = evaluator.evaluate(protocol,synth)
    assert checked['counts']['conditional equivalence'] == 12744
    for field in ['observation_qualified','prediction_qualified','physical_timestamp_accuracy_verified','critical_cta_ordering_verified']:
        assert checked[field] is False
    synthetic_envelopes = {arm:[max(synth['off_on'][prefix+'GLOBAL_FINAL'][arm][i] for prefix in role_keys)
                               for i in range(n)] for arm in ['arm0','arm1']}
    envelope_difference = statistics.median(synthetic_envelopes['arm1'])-statistics.median(synthetic_envelopes['arm0'])
    assert envelope_difference == extra
    zipped_json(args.output/'synthetic-samples.json.gz',synth)
    zipped_json(args.output/'synthetic-evaluator-result.json.gz',checked)
    synthetic_witness = {'data_kind':'Synthetic necessary record-arithmetic witness, not64 new measurements or physical execution.',
                         'target_roles_static':targets,'all_series_entire_sorted_arm_marginals_equal':True,
                         'statistical_subgate_counts':checked['counts'],
                         'original_point_failures':checked['original_point_failures'],
                         'per_call_recorded_FINAL_envelopes':synthetic_envelopes,
                         'envelope_arm_median_difference':envelope_difference,
                         'symbolic_extra_units':extra,
                         'qualification_false_preserved':True}
    # A unique coordinate-median winner can fail every per-call last-finisher.
    finals = {'A':[100,100,100],'B':[110,90,90],'C':[90,110,90],'D':[90,90,110]}
    marginal_medians = {key:statistics.median(values) for key,values in finals.items()}
    winner = max(marginal_medians,key=marginal_medians.get)
    call_winners = [max(finals,key=lambda key:finals[key][i]) for i in range(3)]
    assert winner=='A' and call_winners==['B','C','D']
    rank_witness = {'data_kind':'Synthetic mathematical witness','finals':finals,'coordinate_medians':marginal_medians,
                    'unique_maximum_median_role':winner,'per_call_last_roles':call_winners,'median_winner_per_call_hits':0}
    for filename,rows in [('role-composition.csv',per_role_rows),('merged-composition.csv',merged_rows)]:
        with (args.output/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    report = {'schema':'C-median-composition-contract-review/1','input_sha256':inputs,
              'scope':'Median estimand and same-call composition boundaries. Existing data is arithmetic evidence only; no confidence, physical timing, causal observer, or prediction qualification is assigned.',
              'real_data_exact_identities':exact_checks,
              'real_role_sum_median_gaps':gap_summary(per_role_rows),
              'real_merged_sum_median_gaps':gap_summary(merged_rows),
              'real_consumer_FINAL_envelopes':envelopes,
              'synthetic_full_family_witness':synthetic_witness,
              'synthetic_unique_median_winner_witness':rank_witness,
              'evaluator_implementation_findings':[],
              'contract_requirements':[
                  'The statistical target is difference of recorded arm population medians, not median paired difference or per-invocation accuracy.',
                  'Compose W/E, rank joins, CTA/source-exit/CUDA endpoints and envelopes inside one matched invocation before summarizing across invocations.',
                  'Local median equivalence cannot qualify max/sum composition or frozen per-call last-CTA coverage. Derived endpoints require their own prospective population target/family, or an independently validated joint model.',
                  'Same-call recorded consumer FINAL envelope is distinct from source_safe_exit and CUDA completion; unavailable bounds remain unknown.',
                  'Old point failures and physical sampling/observer perturbation gates remain independent. Synthetic inputs do not satisfy acquisition identity or IID evidence.'],
              'qualification':{'statistical_subgate_scope_reviewed':True,'observations_qualified':False,
                               'physical_joint_execution_verified':False,'median_DAG_price_transfer_qualified':False,
                               'critical_CTA_qualified':False,'independent_local_holdout_passed':False}}
    write_json(args.output/'report.json',report)
    print(json.dumps({key:report[key] for key in ['real_data_exact_identities','real_role_sum_median_gaps','real_merged_sum_median_gaps','real_consumer_FINAL_envelopes','qualification']},indent=2))
    print(json.dumps({'synthetic_statistical_counts':checked['counts'],'synthetic_envelope_median_difference':envelope_difference,
                      'synthetic_point_failures':checked['original_point_failures']},indent=2))


if __name__ == '__main__':
    main()

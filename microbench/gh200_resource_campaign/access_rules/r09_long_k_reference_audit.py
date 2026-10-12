#!/usr/bin/env python3
"""Offline audit of job741159 and its failed frozen-reference gate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(base, root):
    manifest=read(root/'export-manifest.json')
    for name,identity in manifest['files'].items():
        path=root/name
        if path.stat().st_size!=identity['bytes'] or sha(path)!=identity['sha256']:
            raise ValueError('export identity differs: '+name)
    config=read(root/'run_config.json')
    for name in ('cases','schedule'):
        if sha(root/f'{name}.json')!=config[f'{name}_sha256']:
            raise ValueError('prepared matrix identity changed')
    if sha(root/'source/r09_long_k_measurement.py')!=config['runner_sha256']:
        raise ValueError('runner identity changed')
    for name,digest in read(root/'native-binary-hashes.json').items():
        if sha(root/name)!=digest:
            raise ValueError('native ARM binary changed')
    for protocol,binding in read(root/'protocol-bindings.json').items():
        for name,digest in binding['source_files'].items():
            if sha(root/'protocols'/protocol/'source'/name)!=digest:
                raise ValueError('original protocol source changed')
    gate=read(root/'cpu-precheck/gate.json')
    assert gate['passed'] and len(gate['results'])==9
    commands=read(root/'build-commands.json')
    for name,result in gate['results'].items():
        folder=root/'cpu-precheck'/commands[name]['protocol']/'build'
        assert sha(folder/name)==result['cpu_binary_sha256']
        assert sha(folder/f'{name}.sass')==result['cpu_sass_sha256']
        assert result['native_gemm_instruction_sequence_equal'] and result['resource_identity_equal']
    frozen=base/'20261009-V09-freeze-job738972-r2/frozen'
    assert sha(frozen/'predictions.json')==config['baseline_frozen_prediction_sha256']
    assert sha(frozen/'manifest.json')==config['baseline_frozen_manifest_sha256']
    ledger=[json.loads(line) for line in (root/'process-ledger.jsonl').read_text().splitlines()]
    starts=[e for e in ledger if e['event']=='start']
    returns=[e for e in ledger if e['event']=='return']
    assert len(starts)==len(returns)==147
    assert all(e['returncode']==0 for e in returns)
    counts={category:sum(e['category']==category for e in starts) for category in ('setup','sampling')}
    assert counts==dict(setup=27,sampling=120)
    assert not (root/'recovery.json').exists()
    assert read(root/'progress.json')['completed_slots']==120
    assert read(root/'submissions.json')['gpu_submission_count']==1
    assert read(root/'ends-decoder-precheck.json')['passed']
    sys.path.insert(0,str(root/'protocols/v09/source'))
    import analyze_r18
    rows={r['id']:r for r in read(root/'cases.json')}
    expected=read(root/'reference-expected.json')
    policy=read(root/'reference-policy.json')
    actual={cid:dict(plain=[],envelope=[],frequency=[],extra=[]) for cid in expected}
    all_records=list((root/'samples').glob('*/*.json'))
    assert len(all_records)==120
    for path in all_records:
        record=read(path)
        assert record['returncode']==0 and record['attempt']==0
        row=rows[record['case']]
        values=actual[row['id']]
        observation=analyze_r18.replay(root,record,row)
        if record['variant']=='plain':
            values['plain'].append(record['elapsed_us'])
        else:
            assert record['variant']=='dual'
            ctas=observation['ctas'];maximum=max(len(c['tiles']) for c in ctas)
            envelope=(max(c['end_ns'] for c in ctas)-min(c['entry_ns'] for c in ctas))/1000
            values['envelope'].append(envelope)
            values['frequency'].append(st.median((c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns'])
                                                 for c in ctas if len(c['tiles'])==maximum))
            values['extra'].append(record['elapsed_us']-envelope)
    failures=[];comparison={}
    for cid,values in actual.items():
        assert all(len(v)==10 for v in values.values())
        metrics=dict(plain_us=st.median(values['plain']),envelope_us=st.median(values['envelope']),
                     effective_ghz=st.median(values['frequency']),event_extra_us=st.median(values['extra']))
        relative={key:metrics[key]/expected[cid][key]-1 for key in policy['reference_relative_limits']}
        absolute={key:metrics[key]-expected[cid][key] for key in policy['reference_absolute_limits']}
        comparison[cid]=dict(actual=metrics,expected=expected[cid],relative_changes=relative,absolute_changes=absolute)
        assert metrics==read(root/'reference-check.json')['changes'][cid]['actual']
        for key,change in relative.items():
            if abs(change)>policy['reference_relative_limits'][key]:
                failures.append(dict(case=cid,metric=key,relative_change=change))
        for key,change in absolute.items():
            if abs(change)>policy['reference_absolute_limits'][key]:
                failures.append(dict(case=cid,metric=key,absolute_change=change,limit=policy['reference_absolute_limits'][key]))
    assert len(failures)==4 and all(f['metric']=='event_extra_us' for f in failures)
    assert not read(root/'reference-check.json')['passed']
    result=dict(status='failed_frozen_reference_gate',job='741159',reference_gpu=config['reference_gpu'],
        new_gpu_submissions=1,process_counts=counts,failed_sampling_processes=0,recovery_processes=0,
        completed_reference_slots=120,unexecuted_development_slots=540,new_conditions_sampled=0,
        ends_bridge_groups_sampled=0,G6_development_conditions_sampled=0,
        transfer_files_verified=len(manifest['files']),transfer_manifest_sha256=sha(root/'export-manifest.json'),
        cpu_gate_sha256=sha(root/'cpu-precheck/gate.json'),reference_comparison=comparison,reference_failures=failures,
        independent_parameter_pricing='not_executed_no_new_development_data',
        fixed_29_candidate_adoption='not_executed_no_new_candidate',new_correction_refuted=False,
        candidate_adopted=False,V10=False,frozen_v09_identities_unchanged=True,
        audit_source_sha256=sha(Path(__file__)),decoder_source_sha256=sha(Path(analyze_r18.__file__)))
    (root/'terminal-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('status','process_counts','reference_failures','transfer_files_verified')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    audit(args.base.resolve(),args.run.resolve())

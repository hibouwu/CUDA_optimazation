#!/usr/bin/env python3
"""Check the packaged rule against past forecasts and reject target leakage."""
import copy
from fractions import Fraction as F
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from r09_prefix_clock_rule import predict_from_prefix,source_hashes
from r09_prefix_ordinal_clock import SETUP_FIELDS


HERE=Path(__file__).resolve().parent
BASE=Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules')
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rule=read(HERE/'rule.json');identity=(HERE/'rule.sha256').read_text().strip()
    assert identity==sha(HERE/'rule.json') and rule['source_sha256']==source_hashes()
    for name,digest in rule['origin_sha256'].items():assert sha(Path(name))==digest
    original=BASE/'20261011-cfg-c-producer-tail-off-off/reanalysis/A-prefix-ordinal-clock-v1'
    calibration=read(original/'calibration.json')
    assert rule['calibration']=={k:calibration[k] for k in ('schema','setup','kappa_cycles','common_kappa_cycles')}
    ref=BASE/'20261011-cfg-c-mma-reference/returned';samples=[]
    for record in read(ref/'records.json'):
        if record['iterations']==1024:
            samples.append(read(ref/'samples'/record['id']/'stdout.json')['elapsed'])
    assert samples==rule['native_reference']['cycle_samples']==[1049379]*3
    old_forecasts={(r['pair'],r['arm']):r for r in read(original/'report.json')['forecasts']}
    root=BASE/'20261011-cfg-c-producer-tail-off-off';manifest=read(root/'manifest.json')
    prefix_example=None;expected_example=None;raw_identity={};checked=0
    for record in read(root/'returned/records.json'):
        path=root/'returned'/record['raw'];assert sha(path)==record['sha256'];raw_identity[str(path)]=sha(path)
        with gzip.open(path,'rt') as stream:setup,call,check=map(json.loads,stream)
        supplied=dict(setup,binary_sha256=manifest['binary_sha256'],prefix_calls=call['warmup_calls'],
            prefix_same_a=call['prefix_a_ptr']==call['target_a_ptr'],
            prefix_same_b=call['prefix_b_ptr']==call['target_b_ptr'],target_calls=call['target_calls'],
            preceding_target_calls=call['preceding_target_calls'])
        prefix=dict(setup={k:supplied[k] for k in SETUP_FIELDS},prefix_event_us=call['warmup_us'])
        prediction=predict_from_prefix(rule,prefix)
        old=old_forecasts[record['pair'],record['arm']]
        assert prediction['cycle_per_ns']==old['common_rates']
        assert prediction['prediction_ns']==old['common_prediction_ns']
        assert prediction['native_cycles']=='1049379/16'
        assert all(prediction[k] is False for k in ('observation_qualified','prediction_qualified',
                                                  'prospective_holdout_qualified','cross_card_qualified'))
        checked+=1
        if prefix_example is None:prefix_example=prefix;expected_example=prediction
    assert checked==20
    rejected=[]
    def reject(label,bad_prefix,bad_rule=None):
        try:predict_from_prefix(rule if bad_rule is None else bad_rule,bad_prefix)
        except (ValueError,KeyError):rejected.append(label)
        else:raise AssertionError('inadmissible forecast accepted: '+label)
    for name in ('trace','target_cycles','target_ns','elapsed_us','residual','paired_arm'):
        prefix=copy.deepcopy(prefix_example);prefix[name]=123;reject('top:'+name,prefix)
        prefix=copy.deepcopy(prefix_example);prefix['setup'][name]=123;reject('setup:'+name,prefix)
    histories=(('empty',[]),('fifteen',prefix_example['prefix_event_us'][:15]),
        ('seventeen',prefix_example['prefix_event_us']+[100]),
        ('zero',[0]+prefix_example['prefix_event_us'][1:]),
        ('negative',[-1]+prefix_example['prefix_event_us'][1:]),
        ('nan',['nan']+prefix_example['prefix_event_us'][1:]),
        ('boolean',[True]+prefix_example['prefix_event_us'][1:]))
    for label,events in histories:
        prefix=copy.deepcopy(prefix_example);prefix['prefix_event_us']=events;reject('history:'+label,prefix)
    prefix=copy.deepcopy(prefix_example);prefix['setup']['prefix_calls']=15;reject('declared-prefix-count',prefix)
    prefix=copy.deepcopy(prefix_example);prefix['setup']['gpu_uuid']='another-card';reject('target-card',prefix)
    bad_rule=copy.deepcopy(rule);bad_rule['native_reference']['gpu_uuid']='another-card'
    reject('reference-card',prefix_example,bad_rule)
    bad_rule=copy.deepcopy(rule);bad_rule['source_sha256']['r09_prefix_clock_rule.py']='0'*64
    reject('prediction-source',prefix_example,bad_rule)
    # All sixteen events are bound, although only the final one enters the rule.
    changed=copy.deepcopy(prefix_example);changed['prefix_event_us'][0]*=2
    assert predict_from_prefix(rule,changed)==expected_example
    doubled=copy.deepcopy(prefix_example);doubled['prefix_event_us'][-1]*=2
    prediction=predict_from_prefix(rule,doubled)
    assert list(map(F,prediction['prediction_ns']))==[2*F(v) for v in expected_example['prediction_ns']]
    with tempfile.TemporaryDirectory(prefix='r09-frozen-prefix-') as directory:
        isolated=Path(directory)
        for name in ('r09_prefix_clock_rule.py','r09_prefix_ordinal_clock.py','prepare.py','rule.json','rule.sha256'):
            shutil.copy2(HERE/name,isolated/name)
        input_path=isolated/'prefix.json';input_path.write_text(json.dumps(prefix_example,indent=2)+'\n')
        command=[sys.executable,str(isolated/'r09_prefix_clock_rule.py'),'--rule',str(isolated/'rule.json'),
            '--rule-sha256',identity,'--prefix-input',str(input_path),'--output',str(isolated/'prediction.json')]
        subprocess.run(command,check=True,cwd=isolated,stdout=subprocess.DEVNULL)
        receipt=read(isolated/'prediction.json')
        assert receipt==dict(expected_example,rule_sha256=identity,prefix_input_sha256=sha(input_path))
        # Overwrite and caller-pinned SHA mismatch must fail, not silently rebind.
        assert subprocess.run(command,cwd=isolated,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        command[command.index('--rule-sha256')+1]='0'*64
        command[command.index('--output')+1]=str(isolated/'bad.json')
        assert subprocess.run(command,cwd=isolated,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        assert not (isolated/'bad.json').exists()
        (HERE/'example-prefix.json').write_bytes(input_path.read_bytes())
        (HERE/'example-prediction.json').write_text(json.dumps(receipt,indent=2)+'\n')
        (isolated/'rule.json').unlink();(isolated/'rule.sha256').unlink()
        subprocess.run([sys.executable,str(isolated/'prepare.py')],cwd=isolated,check=True,stdout=subprocess.DEVNULL)
        assert (isolated/'rule.json').read_bytes()==(HERE/'rule.json').read_bytes()
    verification=dict(passed=True,rule_sha256=identity,rule_origin_hashes=len(rule['origin_sha256']),
        exact_past_forecast_matches=checked,known_development_raw_sha256=raw_identity,
        negative_cases=rejected,cli_sha_mismatch_rejected=True,cli_output_overwrite_rejected=True,
        self_contained_cli_prediction=True,preparation_replay_byte_identical=True,
        acquisition_status='No new acquisition; all twenty replay prefixes are known development records.',
        qualifications=dict(observation=False,prediction=False,prospective_holdout=False,cross_card=False))
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verification.items() if k!='known_development_raw_sha256'},indent=2))


if __name__=='__main__':main()

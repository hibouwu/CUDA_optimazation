#!/usr/bin/env python3
"""Independently separate static protocol contradictions from target results."""
import argparse
import copy
from fractions import Fraction as F
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE=Path(__file__).resolve().parent
BASE=Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules')
CODE=BASE/'20261011-cfg-c-producer-tail-off-off/reanalysis/A-prefix-clock-local-rule-v1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(CODE))
from r09_prefix_clock_rule import predict_from_prefix
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


CASES=(
    ('warmup_policy','warmup_policy','adaptive'),('warmup_swizzle','warmup_swizzle',8),
    ('warmup_fixed_calls','warmup_fixed_calls',15),('target_swizzle','swizzle',8),
    ('producer_tail_observer','producer_tail_observer',1),
    ('producer_tail_prefix_flag','producer_tail_prefix_flag',1),
    ('history_protocol','history_protocol','other'),('history_source','history_source','mirror'),
    ('source_copies','source_copies',2),('mirror_target_slot','mirror_target_slot',1),
    ('warmup_swizzle','warmup_swizzle',True),('source_copies','source_copies',True),
    ('producer_tail_observer','producer_tail_observer',False),
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extractor',type=Path,default=HERE/'extractor-snapshot.py')
    parser.add_argument('--output',type=Path,default=HERE/'report.json')
    parser.add_argument('--expect-rejection',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('new review output path required')
    spec=importlib.util.spec_from_file_location('reviewed_extractor',args.extractor)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=BASE/'20261011-cfg-c-producer-tail-off-off';manifest=read(root/'manifest.json')
    for name,digest in read(root/'freeze.json')['files'].items():
        if name=='manifest.json':assert sha(root/name)==sha(root/'returned'/name)==digest
    rule_root=root/'reanalysis/A-prefix-clock-local-rule-v1';rule=read(rule_root/'rule.json')
    assert sha(rule_root/'rule.json')==(rule_root/'rule.sha256').read_text().strip()
    inputs={str(root/'manifest.json'):sha(root/'manifest.json'),str(root/'returned/records.json'):sha(root/'returned/records.json'),
            str(rule_root/'rule.json'):sha(rule_root/'rule.json')}
    outcomes=[];valid=0;poisoned=0;first=None
    records=read(root/'returned/records.json');assert len(records)==20
    for sample,record in zip(manifest['samples'],records):
        assert {key:record[key] for key in sample}==sample and record['returncode']==0
        path=root/'returned'/record['raw'];assert sha(path)==record['sha256'];inputs[str(path)]=sha(path)
        with gzip.open(path,'rt') as stream:setup,call,check=map(json.loads,stream)
        projection=module.extract_prefix_input(setup,call,manifest['binary_sha256'])
        prediction=predict_from_prefix(rule,projection)
        kappa=F(rule['calibration']['common_kappa_cycles'])
        expected=F(1049379,16)*F(str(call['warmup_us'][-1]))*1000/kappa
        assert list(map(F,prediction['prediction_ns']))==[expected,expected]
        valid+=1
        invalid_results=copy.deepcopy(call)
        invalid_results['trace']=object();invalid_results['elapsed_us']=object()
        assert module.extract_prefix_input(setup,invalid_results,manifest['binary_sha256'])==projection
        poisoned+=1
        for index,(field,setup_field,value) in enumerate(CASES):
            assert type(call[field]) is type(setup[setup_field]) and call[field]==setup[setup_field]
            changed=copy.deepcopy(call);changed[field]=value
            try:
                candidate=module.extract_prefix_input(setup,changed,manifest['binary_sha256'])
                actual=predict_from_prefix(rule,candidate)
            except (ValueError,KeyError):accepted=False;unchanged=None
            else:accepted=True;unchanged=candidate==projection and actual==prediction
            outcomes.append(dict(pair=record['pair'],arm=record['arm'],case=index,call_field=field,
                setup_field=setup_field,setup_value=setup[setup_field],changed_value=value,
                changed_type=type(value).__name__,accepted=accepted,unchanged_projection_and_prediction=unchanged))
        if first is None:first=(setup,call,check,projection,prediction)
    # An actual failed numerical check and invalid target measurements must not
    # prevent prefix projection. The later scorer retains these failures.
    with tempfile.TemporaryDirectory(prefix='r09-projection-review-') as directory:
        temp=Path(directory);shutil.copy2(args.extractor,temp/'extractor.py')
        shutil.copy2(CODE/'r09_prefix_ordinal_clock.py',temp/'r09_prefix_ordinal_clock.py')
        setup,call,check,projection,prediction=first
        changed=copy.deepcopy(call);changed['trace']=['invalid target trace'];changed['elapsed_us']='invalid target event'
        failed_check=copy.deepcopy(check);failed_check['status']='failed';failed_check['samples']=0
        raw=temp/'synthetic.jsonl.gz'
        with gzip.open(raw,'wt') as stream:
            for packet in (setup,changed,failed_check):stream.write(json.dumps(packet)+'\n')
        command=[sys.executable,str(temp/'extractor.py'),'--raw',str(raw),'--raw-sha256',sha(raw),
            '--binary-sha256',manifest['binary_sha256'],'--output',str(temp/'prefix.json')]
        receipt=json.loads(subprocess.check_output(command,cwd=temp,text=True))
        assert read(temp/'prefix.json')==projection
        assert receipt['target_validity_checked'] is False
        assert predict_from_prefix(rule,read(temp/'prefix.json'))==prediction
    accepted=sum(row['accepted'] for row in outcomes)
    if args.expect_rejection:assert accepted==0
    result=dict(schema='r09-prefix-projection-review/1',extractor_sha256=sha(args.extractor),
        rule_sha256=sha(rule_root/'rule.json'),input_sha256=inputs,
        valid_original_records=valid,target_measurement_poison_noninterference=poisoned,
        failed_check_cli_noninterference=True,static_contradiction_cases=len(outcomes),
        accepted_static_contradictions=accepted,rejected_static_contradictions=len(outcomes)-accepted,
        cases=outcomes,
        finding='Projection must preserve or validate repeated pre-target protocol declarations; raw SHA alone does not establish their consistency.',
        scope='Synthetic software negatives only. All twenty original records are consistent. No numeric rescoring, GPU work or new qualification.',
        observation_qualified=False,prediction_qualified=False,prospective_holdout=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('extractor_sha256','valid_original_records',
        'target_measurement_poison_noninterference','failed_check_cli_noninterference',
        'static_contradiction_cases','accepted_static_contradictions','rejected_static_contradictions')},indent=2))


if __name__=='__main__':main()

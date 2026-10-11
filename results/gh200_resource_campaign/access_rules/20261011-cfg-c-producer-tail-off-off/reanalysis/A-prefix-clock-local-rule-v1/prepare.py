#!/usr/bin/env python3
"""Package the simple development rule; do not acquire or score a new target."""
import hashlib
import json
from pathlib import Path

from r09_prefix_clock_rule import source_hashes


HERE=Path(__file__).resolve().parent
BASE=Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules')
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE/'rule.json').exists():raise ValueError('new rule output required')
    previous=BASE/'20261011-cfg-c-producer-tail-off-off/reanalysis/A-prefix-ordinal-clock-v1'
    calibration=read(previous/'calibration.json');prior_freeze=read(previous/'freeze.json')
    for name,digest in prior_freeze['files'].items():assert sha(previous/name)==digest
    assert calibration['calibration_processes']==10
    assert not any('/20261011-cfg-c-producer-tail-off-off/returned/raw/' in path
                   for path in calibration['input_sha256'])
    origin={str(previous/'calibration.json'):sha(previous/'calibration.json'),
            str(previous/'freeze.json'):sha(previous/'freeze.json')}
    ref=BASE/'20261011-cfg-c-mma-reference';manifest=read(ref/'manifest.json')
    returned=ref/'returned';samples=[]
    assert manifest['status']=='frozen' and manifest['gpu_uuid']==calibration['setup']['gpu_uuid']
    for name,digest in read(ref/'freeze.json')['files'].items():
        for path in (ref/name,returned/name):assert sha(path)==digest
    for path in (ref/'manifest.json',ref/'freeze.json',returned/'records.json'):
        origin[str(path)]=sha(path)
    for record in read(returned/'records.json'):
        if record['iterations']!=1024:continue
        path=returned/'samples'/record['id']/'stdout.json';value=read(path)
        assert record['returncode']==0 and sha(path)==record['files'][str(path.relative_to(returned))]
        assert value['gpu_uuid']==manifest['gpu_uuid'] and value['work_flop']==4294967296
        assert value['unit']=='clock64_cycle/CTA' and value['iterations']==1024
        assert (value['dtype'],value['source'],value['n'],value['chains'],value['groups'],
                value['per_chain_batch'],value['b_major'],value['wait'])==('fp16','SS',128,2,2,4,'MN',1)
        origin[str(path)]=sha(path);samples.append(value['elapsed'])
    assert len(samples)==3
    payload={key:calibration[key] for key in ('schema','setup','kappa_cycles','common_kappa_cycles')}
    rule=dict(schema='r09-frozen-prefix-clock-rule/1',method='common',
        stage='prepared_development_rule_no_new_acquisition',source_sha256=source_hashes(),
        calibration=payload,native_reference=dict(protocol='cfg-c-mma-reference-v1',
            gpu_uuid=manifest['gpu_uuid'],iterations=1024,work_flop=4294967296,cycle_samples=samples),
        origin_sha256=origin,
        support='Fixed cfg_c/K4096/compact dyadic17/sw1/S4/cluster1x2/same card and binary/common16 observer-off first target.',
        input_contract='Exact static setup and all sixteen completed prefix events only; last event used. No current target observations.',
        sampling_contract='The experiment owner must pin this SHA and allowed history inputs before a new acquisition; current preparation proves no such acquisition.',
        limits=['Evaluate the frozen rule after acquisition using only prior-prefix inputs; this is not proof of an online runner hook.',
            'Future raw identities must be disjoint from known development records and match the frozen protocol and card.',
            'Valid local observation, prospective independent acquisition, full-model composition and cross-card evaluation remain required.',
            'No new binary, observer protocol or host-gap equivalence is authorized by this rule. Changed protocol requires new binding and validation.'],
        observation_qualified=False,prediction_qualified=False,prospective_holdout_qualified=False,cross_card_qualified=False)
    (HERE/'rule.json').write_text(json.dumps(rule,indent=2)+'\n')
    (HERE/'rule.sha256').write_text(sha(HERE/'rule.json')+'\n')
    print(json.dumps(dict(rule_sha256=sha(HERE/'rule.json'),origin_hashes=len(origin),source_hashes=len(rule['source_sha256']))))


if __name__=='__main__':main()

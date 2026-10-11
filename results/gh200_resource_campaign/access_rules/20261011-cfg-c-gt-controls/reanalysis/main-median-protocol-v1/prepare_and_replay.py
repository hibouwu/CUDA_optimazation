"""Prepare a reviewable proposal and replay old controls without launching work."""
import argparse
import copy
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

from evaluate import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve();args.output.mkdir(parents=True,exist_ok=False)
    hashes={}
    def load(relative):
        path=run/relative;hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    manifest=load('manifest.json');old=load('analysis.json');records=load('returned/records.json')
    assert old['manifest_sha256']==hashes['manifest.json']
    assert old['observation_qualified'] is False
    for filename,digest in manifest['analysis_sha256'].items():
        path=run/'analysis_snapshot'/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        hashes['analysis_snapshot/'+filename]=digest
    sys.path.insert(0,str(run/'analysis_snapshot'))
    from analyze_cfg_c_gt_control import process
    names=sorted(row['series'] for row in old['cohorts']['off_on']['local'])
    assert len(names)==6372 and len(set(names))==6372
    assert set(names)=={row['series'] for row in old['cohorts']['off_off']['local']}
    proposal=dict(schema='gh200-median-equivalence-proposal/1',status='proposal_not_frozen_or_authorized_for_execution',
        target='Difference of arm population medians of recorded ns, conditional on valid sampling assumptions.',
        repeats_per_arm=64,family_failure_probability='1/20',median_difference_margin_ns=64,
        series=names,gpu_uuid=manifest['gpu_uuid'],binary_sha256=manifest['binary_sha256'],
        original_manifest_sha256=hashes['manifest.json'],
        original_point_gates=manifest['gates'],
        sampling='64 blocks; one fresh process for each of off_on.arm0/off_on.arm1/off_off.arm0/off_off.arm1; original16 observer-off prefix per process; randomized within block with16 enabled targets in each of four positions.',
        stopping='Fixed64 blocks, no early success or extra sampling after inspecting results. Any process/numeric/identity failure stops acquisition and leaves an incomplete invalid dataset; no replacement or filtering.',
        unresolved=['IID/stationary population justification is not supplied by randomization alone.',
                    'Independent timestamp accuracy and critical CTA ordering evidence remain required.',
                    '64 repeats give finite intervals, not a power or precision guarantee.',
                    'The64ns population-median margin is a candidate carried from the old magnitude budget; it is not the old median-absolute-pair criterion.',
                    'No V10 freeze, multi-config qualification or cross-card applicability is granted.'])
    (args.output/'protocol-proposal.json').write_text(json.dumps(proposal,indent=2)+'\n')
    rng=random.Random(20261011);positions=list(range(4))*16;rng.shuffle(positions)
    schedule=[]
    templates={(row['comparison'],row['arm']):row for row in manifest['samples'] if row['pair']==0}
    for block,on_position in enumerate(positions):
        off=[('off_on',0),('off_off',0),('off_off',1)];rng.shuffle(off)
        order=off[:];order.insert(on_position,('off_on',1))
        for position,(cohort,arm) in enumerate(order):
            template=templates[cohort,arm]
            schedule.append(dict(block=block,position=position,comparison=cohort,arm=arm,
                flag=template['flag'],command=template['command']))
    assert len(schedule)==256
    assert all(sum(s['flag']==1 and s['position']==p for s in schedule)==16 for p in range(4))
    (args.output/'schedule-proposal.json').write_text(json.dumps(schedule,indent=2)+'\n')
    samples={cohort:{name:dict(arm0=[],arm1=[]) for name in names} for cohort in ('off_on','off_off')}
    original_samples={s['id']:s for s in manifest['samples']}
    seen=set()
    for record in sorted(records,key=lambda s:(s['comparison'],s['pair'],s['arm'])):
        assert record['id'] not in seen;seen.add(record['id'])
        assert record['returncode']==0
        for key,value in original_samples[record['id']].items():assert record[key]==value
        path=run/'returned'/record['raw'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==record['sha256'];hashes['returned/'+record['raw']]=digest
        with gzip.open(path,'rt') as stream:events=[json.loads(line) for line in stream]
        decoded=process(events,record,manifest['gpu_uuid'])
        assert set(decoded['series'])==set(names)
        for name in names:samples[record['comparison']][name]['arm'+str(record['arm'])].append(decoded['series'][name][0])
    assert seen==set(original_samples)
    retrospective=dict(proposal,repeats_per_arm=10,status='retrospective_capacity_replay_only')
    result=evaluate(retrospective,samples)
    assert result['counts']['insufficient simultaneous coverage']==12744
    for cohort in ('off_on','off_off'):
        expected={row['series']:row for row in old['cohorts'][cohort]['local']}
        for row in result['series_results']:
            if row['cohort']!=cohort:continue
            prior=expected[row['series']];current=row['original_point_control']
            assert current['median_absolute_ns']==prior['median_absolute']
            assert current['maximum_absolute_ns']==prior['maximum_absolute']
            assert current['passed']==prior['passed']
    (args.output/'old-control-replay.json.gz').write_bytes(gzip.compress(json.dumps(result,separators=(',',':')).encode(),mtime=0))
    from evaluate import __file__ as evaluator_path
    result_summary=dict(input_sha256=hashes,raw_processes=len(records),series_per_cohort=6372,
        median_results=result['counts'],original_point_failures=result['original_point_failures'],
        schedule_calls=256,on_target_positions=[sum(s['flag']==1 and s['position']==p for s in schedule) for p in range(4)],
        evaluator_sha256=hashlib.sha256(Path(evaluator_path).read_bytes()).hexdigest(),
        prepare_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((args.output/'protocol-proposal.json').read_bytes()).hexdigest(),
        schedule_sha256=hashlib.sha256((args.output/'schedule-proposal.json').read_bytes()).hexdigest(),
        observation_qualified=False,prediction_qualified=False)
    (args.output/'replay-summary.json').write_text(json.dumps(result_summary,indent=2)+'\n')
    print(json.dumps({k:result_summary[k] for k in ('raw_processes','median_results','original_point_failures','schedule_calls')},indent=2))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Same-card V07 calibration, qualified boundary interpolation, immutable prediction freeze."""
import argparse,datetime,json,statistics,time
from pathlib import Path
import v06_fit
import v06_model as base
import v06_run as common
import v07_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--r18',required=True,type=Path);p.add_argument('--r18-analysis',required=True,type=Path)
    p.add_argument('--r18-lite-analysis',type=Path);p.add_argument('--novelty',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();root=a.run.resolve()
    common.verify(root);env=json.loads((root/'environment.json').read_text())
    if json.loads((a.r18/'environment.json').read_text())['gpu']!=env['gpu']:raise ValueError('boundary measurements are on a different GPU')
    novelty=json.loads(a.novelty.read_text())
    if novelty['heldout_clashes']:raise ValueError('target shapes are not new')
    if a.output.exists():raise ValueError('never overwrite a frozen prediction')
    derived=root/'reanalysis/v07-calibration';eligibility=v07_model.convert(root,'calib',derived,True)
    cal=v06_fit.fit(derived)
    for cfg in ['cfg_a','cfg_b','cfg_c']:
        cases=[c for c in cal['cases'].values() if c['config']==cfg]
        if len(cases)<4 or len({c['ktiles'] for c in cases})<2 or len({c['tiles_max'] for c in cases})<2:
            raise ValueError('insufficient qualified K/work calibration coverage: '+cfg)
    boundary=v07_model.boundary_models(a.r18_analysis)
    if a.r18_lite_analysis:boundary.update(v07_model.boundary_models(a.r18_lite_analysis))
    setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
    predictions={}
    for row in json.loads((root/'cases.json').read_text()):
        if row['set']!='heldout':continue
        predictions[row['id']]=v07_model.predict(row,setups[row['id']]['grid'],cal['params'][row['config']],cal['clock_rule'],cal['fixed_us'][row['config']],boundary)
    if len(predictions)!=24:raise ValueError('expected 24 held-out cases')
    if any((root/'samples'/key).exists() for key in predictions):raise ValueError('target timing already exists')
    now=time.time_ns();document=dict(status='frozen',frozen_unix_ns=now,
        frozen_utc=datetime.datetime.fromtimestamp(now/1e9,datetime.timezone.utc).isoformat(),gpu=env['gpu'],environment=env,
        source_hashes_sha256=common.sha(root/'source_hashes.json'),binary_hashes_sha256=common.sha(root/'build/binary_hashes.json'),
        cases_sha256=common.sha(root/'cases.json'),static_setup_sha256=common.sha(root/'static_setup.json'),
        model=v07_model.__doc__,model_sha256=common.sha(Path(v07_model.__file__)),fitter_sha256=common.sha(Path(__file__)),
        calibration=cal,calibration_eligibility=eligibility,boundary_hypotheses=boundary,
        boundary_evidence_sha256=common.sha(a.r18_analysis/'paired-tile-increments.json'),
        boundary_lite_evidence_sha256=common.sha(a.r18_lite_analysis/'paired-tile-increments.json') if a.r18_lite_analysis else None,
        novelty=novelty,predictions=predictions)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(document,stream,indent=2);stream.write('\n')
    a.output.chmod(0o444)
    common.write_json(derived/'calibration.json',cal)
    print('V07 frozen',common.sha(a.output),'qualified calibration cases',sum(x['qualified'] for x in eligibility))
    for key,r in predictions.items():print(key,round(r['predicted_us'],3),'us','unknown cells',len(r['boundary_unknown_cells']))
if __name__=='__main__':main()

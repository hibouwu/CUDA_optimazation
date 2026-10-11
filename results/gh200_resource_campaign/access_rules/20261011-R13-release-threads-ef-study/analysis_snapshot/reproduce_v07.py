#!/usr/bin/env python3
"""Recompute the frozen V07 parameters and predictions from calibration only, offline."""
import argparse,json,math
from pathlib import Path
import v06_fit
import v07_model
import v06_run as common


def equal(a,b,path='root'):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('different keys: '+path)
        for k in a:equal(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('different length: '+path)
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f'[{i}]')
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-8):raise ValueError('different numeric result: '+path)
    elif a!=b:raise ValueError('different result: '+path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--r18-analysis',type=Path,required=True);p.add_argument('--r18-lite-analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.run.resolve();frozen=json.loads((root/'v07-predictions.json').read_text());binding=json.loads((root/'calibration_binding.json').read_text())
    common.verify(root)
    for n,h in binding['files'].items():
        if common.sha(root/n)!=h:raise ValueError('calibration identity changed: '+n)
    a.output.mkdir(parents=True,exist_ok=False)
    v07_model.convert(root,'calib',a.output/'derived',True);cal=v06_fit.fit(a.output/'derived')
    for key in ['params','clock_rule','fixed_us']:equal(cal[key],frozen['calibration'][key],key)
    boundary=v07_model.boundary_models(a.r18_analysis);boundary.update(v07_model.boundary_models(a.r18_lite_analysis));equal(boundary,frozen['boundary_hypotheses'],'boundary')
    for cid,pred in frozen['predictions'].items():
        row=pred['row'];actual=v07_model.predict(row,pred['grid'],cal['params'][row['config']],cal['clock_rule'],cal['fixed_us'][row['config']],boundary)
        equal(json.loads(json.dumps(actual)),pred,cid)
    common.write_json(a.output/'receipt.json',dict(status='passed',predictions_recomputed=24,calibration_only=True,
       prediction_sha256=common.sha(root/'v07-predictions.json'),absolute_tolerance=1e-8,relative_tolerance=1e-11))
    print('Frozen calibration parameters, boundary increments and all 24 predictions reproduced')
if __name__=='__main__':main()

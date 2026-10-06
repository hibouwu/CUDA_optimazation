#!/usr/bin/env python3
"""Isolated, read-only adapter to the frozen v2 auditor's public evidence functions."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
sys.dont_write_bytecode = True


def require(ok, why):
    if not ok: raise ValueError(why)


def read(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result, 'duplicate JSON key')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def safe(root,name):
    path=Path(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name,'unsafe relative path')
    target=root
    for part in path.parts:
        target=target/part;require(not target.is_symlink(),'symlink input forbidden')
    require(target.is_file(),'missing artifact: '+name)
    return target


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def guard(event,args):
    if event=='open':
        _,mode,flags=args
        require(not (isinstance(mode,str) and any(c in mode for c in 'wax+')),'read-only worker rejected write')
        require(not (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)),'read-only worker rejected open flags')
    if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn','os.spawn','os.fork','os.forkpty',
                 'os.remove','os.rmdir','os.rename','os.mkdir','os.chmod','os.chown','os.truncate','os.symlink','os.link'):
        raise ValueError('read-only worker rejected '+event)


def accepted(root,case_id,scope):
    directory=root/scope/case_id/'batch_00/trial_00'
    receipts=list(directory.glob('attempt_*/receipt.json'))
    statuses=[(p,read(p)) for p in receipts]
    require(all(r.get('status') in ('valid','warmup_unconverged','interrupted') for _,r in statuses),'failed attempt hidden by successful sample')
    paths=[p for p,r in statuses if r.get('status') in ('valid','warmup_unconverged')]
    require(len(paths)==1,'one accepted process required per preflight/pilot')
    return paths[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('strict','recompute'));ap.add_argument('root',type=Path);args=ap.parse_args()
    root=args.root.resolve();sys.addaudithook(guard)
    spec=read(safe(root,'run_spec.json'));require(spec.get('schema_version')==2,'unsupported archive schema')
    manifest=safe(root,'snapshot/manifest.json');require(sha(manifest)==spec['snapshot_manifest_sha256'],'snapshot manifest hash')
    for name,expected in read(manifest).items():require(sha(safe(root/'snapshot',name))==expected,'snapshot file hash: '+name)
    frozen=safe(root,'snapshot/repo/microbench/gh200_resource_campaign/run_suite.py').parent
    sys.path.insert(0,str(frozen))
    if args.mode=='strict':
        sys.argv=[str(frozen/'run_suite.py'),'audit',str(root)]
        if (root/'COMPLETE').exists():sys.argv.append('--require-complete')
        runpy.run_path(str(frozen/'run_suite.py'),run_name='__main__')
        return
    from auditors import suite
    spec,contract,protocol,device,family=suite.load_run(root)
    records=[]
    if spec['kind']=='preflight':
        from common.calibration import calibrated_cases,pilot_case
        record=read(safe(root,'preflight_summary.json'))
        require({p.name for p in (root/'preflight').iterdir() if p.is_dir()}=={c['id'] for c in contract['cases']},'preflight case coverage')
        expected={'schema_version':2,'status':'preflight_pending_review','hardware_qualification':False,'binary_sha256':spec['binary_sha256'],'cases':{}}
        for case in contract['cases']:
            receipt=accepted(root,case['id'],'preflight')
            result=suite.trial_evidence(root,spec,case,device,protocol,0,0,receipt.parent)
            _,row=suite.parse_raw(receipt.parent/'raw.jsonl')
            base=family.validate_trial(row,case,device,read(receipt)['seed'],protocol)
            shape=set(record.get('cases',{}).get(case['id'],{}))
            require(shape in (set(base),set(result)),'unsupported/omitted preflight result fields')
            expected['cases'][case['id']]={key:result[key] for key in shape}
            records.append({'raw':str((receipt.parent/'raw.jsonl').relative_to(root)),'case_id':case['id'],'scope':'preflight','metric':case['metric']})
        targets=calibrated_cases(contract)
        if targets:
            require('resolved_cases' in spec,'resolved calibration missing')
            require({p.name for p in (root/'calibration').iterdir() if p.is_dir()}=={c['id'] for c in targets},'pilot case coverage')
            for case in targets:
                pc=pilot_case(case);receipt=accepted(root,case['id'],'calibration')
                suite.trial_evidence(root,spec,pc,device,protocol,0,0,receipt.parent)
                records.append({'raw':str((receipt.parent/'raw.jsonl').relative_to(root)),'case_id':case['id'],'scope':'pilot','metric':case['metric']})
        require(not (root/'COMPLETE').exists(),'preflight cannot have COMPLETE')
    else:
        record=read(safe(root,'summary.json'));expected=suite.recompute(root,require_terminal=True)
        # Reuse every original raw/work/receipt/order/counter check, separately from final float comparison.
        manifest=read(safe(root,'measurement_manifest.json'));suite.verify_files(root,manifest)
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
                and p.relative_to(root).as_posix() not in ('measurement_manifest.json','campaign_status.json','COMPLETE')
                and not p.relative_to(root).as_posix().startswith('reviews/C/')}
        require(actual==set(manifest),'measurement manifest coverage')
        if (root/'COMPLETE').exists():
            finish=read(root/'COMPLETE')
            require(finish=={'schema_version':2,'summary_sha256':sha(root/'summary.json'),'measurement_manifest_sha256':sha(root/'measurement_manifest.json'),'C_review_sha256':sha(safe(root,'reviews/C/review.json'))},'COMPLETE binding')
            review=suite.validate_gate(root/'reviews/C/review.json',root,contract['stage'],'C')
            require({'summary.json','measurement_manifest.json'}<=set(suite.gate_mapping(review)),'C gate binding')
            require(read(safe(root,'campaign_status.json'))['status']=='complete','completion state')
        by_id={c['id']:c for c in contract['cases']}
        for case in expected['cases']:
            for batch in case['batches']:
                for sample in batch['samples']:
                    path=safe(root,sample['receipt']).parent/'raw.jsonl'
                    records.append({'raw':str(path.relative_to(root)),'case_id':case['case_id'],'scope':'formal','batch':batch['batch'],'trial':sample['trial'],'metric':by_id[case['case_id']]['metric']})
    suite.validate_telemetry(root,device['uuid'])
    print(json.dumps({'kind':spec['kind'],'recorded':record,'recomputed':expected,'raw_records':records,'protocol':protocol,'device':device},allow_nan=False,sort_keys=True))


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError) as exc:
        print('REPLAY_WORKER_REJECTED: '+str(exc),file=sys.stderr);raise SystemExit(2)

#!/usr/bin/env python3
"""Compare complete default results against the actual pre-change module."""
from pathlib import Path
import hashlib,json,subprocess,sys,types
out=Path(__file__).resolve().parent
repo=next(p for p in out.parents if (p/'microbench').is_dir());sys.path.insert(0,str(repo/'microbench/gh200_resource_campaign/access_rules'))
import supply_model as current
revision='76c9f619^'
source=subprocess.check_output(['git','show',revision+':microbench/gh200_resource_campaign/access_rules/supply_model.py'],cwd=repo,text=True)
old=types.ModuleType('pre_destination_latency');exec(source,old.__dict__)
root=repo/'results/gh200_resource_campaign/access_rules';run=root/'20261010-R13-second-prefill'
params=json.loads((out/'B-latency-relocation.json').read_text())['parameters']
rows=json.loads((run/'cases.json').read_text());row=next(r for r in rows if r.get('set')!='reference')
setup=next(s['setup'] for s in json.loads((run/'static_setup.json').read_text()) if s['case']==row['id'])
program=current.cfg_b_source_program(row,setup);assert program==old.cfg_b_source_program(row,setup)
checks=[]
for stage in (4,6):
    for name,extra in (('ordinary',{}),('root512',dict(diagnostic_producer_offsets_ns=[512. if c==55 else 0. for c in range(132)])),
        ('all_E0_deferred',dict(epilogue_defer=2)),('prefill_after_defer',dict(epilogue_defer=2,second_prefill=True))):
        kwargs=dict(params,source_keys=program['source_keys'],output_counts=program['output_counts'],share_inflight_sources=True,trace_requests=True,**extra)
        a=old.shared_stage_events(64,stage,132,**kwargs);b=current.shared_stage_events(64,stage,132,**kwargs)
        assert a==b,(stage,name)
        checks.append(dict(stages=stage,treatment=name,complete_dictionary_equal=True,
            result_sha256=hashlib.sha256(json.dumps(b,separators=(',',':')).encode()).hexdigest()))
report=dict(previous_commit=subprocess.check_output(['git','rev-parse',revision],cwd=repo,text=True).strip(),
    previous_source_sha256=hashlib.sha256(source.encode()).hexdigest(),current_source_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
    checks=checks,scope='Complete default-output regression, no calibrated mechanism or observer qualification.')
(out/'default-regression.json').write_text(json.dumps(report,indent=2)+'\n');print('8 complete dictionaries identical to pre-change implementation')

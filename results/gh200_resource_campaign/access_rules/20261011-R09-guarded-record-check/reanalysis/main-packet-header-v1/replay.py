#!/usr/bin/env python3
"""Replay immutable smoke records through the stricter current packet reader."""
from pathlib import Path
import copy,gzip,hashlib,json,subprocess,sys
out=Path(__file__).resolve().parent
run=out.parent.parent
repo=next(p for p in out.parents if (p/'microbench').is_dir())
source=repo/'microbench/gh200_resource_campaign/access_rules'
subprocess.run([sys.executable,str(run/'verify.py')],check=True,capture_output=True)
sys.path.insert(0,str(source))
from decode_r09_clock_observer import decode
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads((run/'returned/records.json').read_text())
report={'parser_sha256':sha(source/'decode_r09_clock_observer.py'),'arms':[],
        'observation_qualified':False,'limits':'Current-reader replay; original frozen reader and raw bytes unchanged. Fixed full GH200 132-SM contract, not a general CUDA SM-ID rule.'}
for row in rows:
    raw=run/'returned'/row['raw']
    assert sha(raw)==row['sha256']
    with gzip.open(raw,'rt') as f:events={v['event']:v for line in f for v in [json.loads(line)]}
    setup,call=events['setup'],events['call']
    result=decode(setup,call,row['arm'],setup['gpu_uuid'])
    assert result['record_valid'],result.get('errors')
    mutations=[]
    if row['arm']!='plain':
        for index in (0,1,3,13):
            bad=copy.deepcopy(call);bad['trace'][index]=0
            checked=decode(setup,bad,row['arm'],setup['gpu_uuid'])
            assert not checked['record_valid'] and len(checked['records'])==262
            mutations.append({'mutation':f'CTA0 header[{index}]=0','errors':checked['errors']})
        bad=copy.deepcopy(call)
        for index in (2,11,12):bad['trace'][index]=133
        checked=decode(setup,bad,row['arm'],setup['gpu_uuid'])
        assert not checked['record_valid'] and len(checked['records'])==262
        mutations.append({'mutation':'CTA0 all SM fields=133','errors':checked['errors']})
    report['arms'].append({'arm':row['arm'],'raw_sha256':sha(raw),'records':len(result['records']),
        'record_valid':result['record_valid'],'rejected_mutations':mutations})
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({a['arm']:{'records':a['records'],'rejected_mutations':len(a['rejected_mutations'])} for a in report['arms']}))

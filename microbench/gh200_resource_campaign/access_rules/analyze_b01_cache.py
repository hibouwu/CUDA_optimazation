#!/usr/bin/env python3
"""B01 matched CUTLASS cache preparation: exact witnesses, protocol and complete time."""
import argparse,gzip,json,statistics
from pathlib import Path
import v06_run as common
from analyze_r18 import replay


def analyze(root,micro,output):
    common.verify(root);env=json.loads((root/'environment.json').read_text());other=json.loads((micro/'environment.json').read_text())
    if env['gpu'].split(',')[0]!=other['gpu'].split(',')[0]:raise ValueError('micro/cache GPU identity differs')
    reference=json.loads((micro/'samples/tma_input_one_cta_above_l2/formal-00/result.json').read_text());l2=reference['l2_bytes']
    rows=[];checked=0
    for row in json.loads((root/'cases.json').read_text()):
        values=[]
        for path in sorted((root/'samples'/row['id']).glob('plain-*.json')):
            record=json.loads(path.read_text());r=replay(root,record,row);checked+=r['checked_values'];values.append(r['elapsed_us'])
            with gzip.open(root/record['raw'],'rt') as stream:
                setup=next(json.loads(l) for l in stream if '"event":"setup"' in l)
            if setup['evict']!=row['evict'] or setup['eviction_bytes']!=2*l2:raise ValueError('cache preparation changed')
            if setup['stages']!=6 or setup['tile']!=[128,128,64] or setup['cluster']!=[2,1]:raise ValueError('CUTLASS configuration changed')
        if len(values)!=10:raise ValueError('ten cache processes required')
        variation=statistics.pstdev(values)/statistics.mean(values)
        rows.append(dict(case=row['id'],swizzle=row['swizzle'],evict=row['evict'],processes=len(values),
         microseconds=statistics.median(values),stddev=statistics.pstdev(values),cv=variation,qualified=variation<=.05,
         flop=2*row['m']*row['n']*row['k'],m=row['m'],n=row['n'],k=row['k'],eviction_bytes=2*l2,
         scope='single plain CUTLASS call; allocation/fill/eviction/synchronization/check outside CUDA events'))
    result=dict(conditions=rows,checked_values=checked,environment=env,working_set_input_bytes=2*(4096*2048+2048*4096),
                output_bytes=4096*4096*4,physical_cold_HBM_proven=False)
    output.mkdir(parents=True,exist_ok=True);common.write_json(output/'rules.json',result)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--micro',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.input.resolve(),a.micro.resolve(),a.output)

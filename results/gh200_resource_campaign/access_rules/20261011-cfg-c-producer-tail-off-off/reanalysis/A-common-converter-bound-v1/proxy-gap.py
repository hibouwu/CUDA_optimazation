#!/usr/bin/env python3
"""Keep the predeclared prefix proxy separate from target-selected oracle scales."""
from fractions import Fraction as Q
import gzip,hashlib,json,statistics as st
from pathlib import Path

root=Path(__file__).resolve().parent
report=json.loads((root/'report.json').read_text())
with gzip.open(root/'windows.json.gz','rt') as stream:rows=json.load(stream)
main_root=next(Path(p).parent for p in report['input_sha256'] if p.endswith('/20261011-cfg-c-producer-tail-off-off/manifest.json'))
source=main_root/'reanalysis/main-prefix-scalar-diagnostic-v1/windows.json.gz'
with gzip.open(source,'rt') as stream:
    reference={(r['pair'],r['cta'],r['wg'],r['ordinal']):Q(r['relative_residual']) for r in json.load(stream)}
results=[];matched=0
for bound in report['bounds']:
    if bound['scope']!='WG':continue
    values=rows[bound['row_offset']:bound['row_offset']+bound['n']]
    alpha=Q(bound['alpha_last_prefix'])
    errors=[]
    for row in values:
        error=alpha*Q(row['n0']*row['c1'],row['c0']*row['n1'])-1
        errors.append(error)
        if bound['run'].endswith('off-off'):
            assert error==reference[row['pair'],row['cta'],row['wg'],row['ordinal']];matched+=1
    results.append(dict(run=bound['run'],pair=bound['pair'],window_pairs=len(values),
        prefix_maximum_relative_residual=str(max(map(abs,errors))),
        optimal_maximum_relative_residual=bound['relative']['floor']))
groups=[]
for run in sorted({r['run'] for r in results}):
    values=[r for r in results if r['run']==run]
    prefix=[float(Q(r['prefix_maximum_relative_residual'])) for r in values]
    oracle=[float(Q(r['optimal_maximum_relative_residual'])) for r in values]
    groups.append(dict(run=run,prefix_pair_maximum_relative=dict(minimum=min(prefix),median=st.median(prefix),maximum=max(prefix)),
                       oracle_pair_maximum_relative=dict(minimum=min(oracle),median=st.median(oracle),maximum=max(oracle))))
assert len(results)==20 and matched==4320
result=dict(schema='gh200-r09-prefix-oracle-gap/1',pairs=results,groups=groups,
    main_prefix_exact_matches=matched,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'report.json',root/'windows.json.gz',source)},
    limits=['Prefix alpha itself is prior-event based, but the local test uses both target C and arm0 N, hence remains conditional.',
        'Oracle uses all target windows; its lower error is not an achievable prospective result.',
        'No new state function, hardware frequency estimate, acceptance gate, or corrected qualification is introduced.'])
(root/'proxy-gap.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(groups=groups,main_prefix_exact_matches=matched),indent=2))

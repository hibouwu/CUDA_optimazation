#!/usr/bin/env python3
"""Check all conditional cross-K transfers, without median-only reporting."""
from fractions import Fraction
from pathlib import Path
import gzip,hashlib,json,statistics
out=Path(__file__).resolve().parent
with gzip.open(out/'windows.json.gz','rt') as f:rows=json.load(f)
base=json.loads((out/'report.json').read_text())
assert len(rows)==base['paired_windows']==12960
for path,h in base['input_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
keys=[(r['pitch'],r['trial'],r['cta'],r['role'],r['output_ordinal']) for r in rows]
assert len(keys)==len(set(keys))
results=[]
for pitch in ('aligned','a16','b16'):
    selected=[r for r in rows if r['pitch']==pitch];assert len(selected)==4320
    for metric in ('cycles','ns'):
        proportional=[Fraction(r[f'{metric}_4096'],4*r[f'{metric}_1024'])-1 for r in selected]
        transfer=[Fraction(4*r[f'{metric}_1024']-r[f'{metric}_4096'],r[f'{metric}_4096']) for r in selected]
        assert all(b==-a/(1+a) for a,b in zip(proportional,transfer))
        absolute=list(map(abs,transfer));worst=max(range(len(selected)),key=lambda i:absolute[i])
        record=selected[worst]
        results.append(dict(pitch=pitch,metric=metric,windows=len(selected),
            proportional_absolute_max=float(max(map(abs,proportional))),
            target_denominator_absolute_median=float(statistics.median(absolute)),
            target_denominator_absolute_max=float(absolute[worst]),
            above_5_percent=sum(v>Fraction(1,20) for v in absolute),above_10_percent=sum(v>Fraction(1,10) for v in absolute),
            worst=dict(window=record,predicted=4*record[f'{metric}_1024'],observed=record[f'{metric}_4096'],
                signed_error_fraction=[transfer[worst].numerator,transfer[worst].denominator])))
report=dict(input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'windows.json.gz',out/'report.json',Path(__file__))},
    complete_windows=12960,results=results,observation_qualified=False,
    limits=['Conditional transfer using measured K1024 for the corresponding window, not static-input autonomous prediction.',
        'All windows retained including changed SMs; trial index is descriptive, not causal randomization.',
        '5/10 percent counts are diagnostics only; frozen local, critical CTA, total and cross-card acceptance are not assessed.',
        'Native-cycle agreement cannot establish ns conversion or local observation validity.'])
(out/'all-window-check.json').write_text(json.dumps(report,indent=2)+'\n')
for r in results:print(r['pitch'],r['metric'],'median/max',100*r['target_denominator_absolute_median'],100*r['target_denominator_absolute_max'],'>5/>10',r['above_5_percent'],r['above_10_percent'])

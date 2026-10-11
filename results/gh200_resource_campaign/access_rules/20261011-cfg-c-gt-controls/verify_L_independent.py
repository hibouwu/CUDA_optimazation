"""Independent raw slot arithmetic for all paired L windows and event gates."""
from pathlib import Path
import gzip,json,statistics
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text())
analysis=json.loads((ROOT/'analysis.json').read_text())
records=json.loads((ROOT/'returned/records.json').read_text())
parsed={}
for rec in records:
 with gzip.open(ROOT/'returned'/rec['raw'],'rt') as f:packets={v['event']:v for v in map(json.loads,f)}
 call=packets['call'];trace=call['trace'];windows={}
 for cta in range(132):
  base=cta*1296
  for wg in range(2):
   for ordinal in range(trace[base+6+3*wg]):
    pos=base+16+wg*640+ordinal*10
    m,n=trace[pos+4]-1,trace[pos+5]-1
    windows[f'{cta}/{wg}/{ordinal}/{m},{n}/L']=trace[pos+7]-trace[pos+6]
 assert len(windows)==432
 parsed[rec['id']]=(windows,call['elapsed_us'])
report={}
for comparison in manifest['comparisons']:
 rows=[row for row in records if row['comparison']==comparison]
 paired=[];ratios=[];reference=[]
 for pair in range(10):
  arms={r['arm']:parsed[r['id']] for r in rows if r['pair']==pair}
  assert set(arms)=={0,1}
  paired.append((arms[0][0],arms[1][0]));reference.append(arms[0][1]);ratios.append(arms[1][1]/arms[0][1]-1)
 saved=analysis['cohorts'][comparison]
 saved_L={row['series']:row for row in saved['local'] if row['series'].endswith('/L')}
 assert len(saved_L)==432
 failed=0
 for key,row in saved_L.items():
  delta=[b[key]-a[key] for a,b in paired];absolute=list(map(abs,delta))
  median=statistics.median(absolute);maximum=max(absolute)
  passed=median<=64 and maximum<=256
  assert delta==row['signed'] and median==row['median_absolute'] and maximum==row['maximum_absolute'] and passed==row['passed']
  failed+=not passed
 assert ratios==saved['event']['signed']
 assert statistics.median(map(abs,ratios))==saved['event']['median_absolute']
 assert max(map(abs,ratios))==saved['event']['maximum_absolute']
 assert statistics.pstdev(reference)/statistics.fmean(reference)==saved['off_event_population_cv']
 report[comparison]=dict(L_windows=432,paired_L_differences=4320,all_raw_L_results_equal=True,L_failed=failed,event_and_CV_equal=True)
(ROOT/'independent-L-verification.json').write_text(json.dumps(dict(cohorts=report,qualification=False),indent=2)+'\n')
print(json.dumps(report,indent=2))

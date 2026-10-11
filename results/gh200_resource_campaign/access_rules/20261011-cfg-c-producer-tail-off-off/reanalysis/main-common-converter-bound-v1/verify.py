from pathlib import Path
from fractions import Fraction as F
import json,gzip,hashlib
out=Path(__file__).resolve().parent;a=out.parent/'A-common-converter-bound-v1'
r=json.loads((a/'report.json').read_text())
with gzip.open(a/'windows.json.gz','rt') as f:rows=json.load(f)
checks=[];count=0
for bound in r['bounds']:
 values=rows[bound['row_offset']:bound['row_offset']+bound['n']]
 assert all((x['run'],x['pair'],x['scope'])==(bound['run'],bound['pair'],bound['scope']) for x in values)
 q=[F(x['n0']*x['c1'],x['c0']) for x in values];y=[F(x['n1']) for x in values]
 absolute=bound['absolute'];radius=F(absolute['radius_ns']);alpha=F(absolute['alpha'])
 lo=max((v-radius)/w for v,w in zip(y,q));hi=min((v+radius)/w for v,w in zip(y,q))
 assert 0<alpha and lo<=alpha<=hi
 assert max(abs(alpha*w-v) for w,v in zip(q,y))==radius
 i,j=absolute['witness_indices']
 lower=abs(y[i]*q[j]-y[j]*q[i])/(q[i]+q[j]);assert lower==radius
 relative=bound['relative'];z=[w/v for w,v in zip(q,y)];low,high=min(z),max(z)
 rf=(high-low)/(high+low);ra=2/(high+low)
 assert rf==F(relative['floor']) and ra==F(relative['alpha'])
 assert max(abs(ra*w-1) for w in z)==rf
 count+=len(values)
 checks.append(dict(run=bound['run'],pair=bound['pair'],scope=bound['scope'],absolute_optimum_ns=str(radius),relative_optimum=str(rf),absolute_and_relative_alpha_equal=alpha==ra))
assert count==12960 and len(checks)==40
result=dict(input_report_sha256=hashlib.sha256((a/'report.json').read_bytes()).hexdigest(),input_windows_sha256=hashlib.sha256((a/'windows.json.gz').read_bytes()).hexdigest(),window_checks=count,certificates=len(checks),checks=checks,observation_qualified=False,prediction_qualified=False)
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print('40 exact lower-bound and attainment certificates;12960 window checks;no optimizer or model imported')

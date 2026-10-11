"""Count certificate restrictions without discarding or changing any comparison."""
from collections import Counter
from fractions import Fraction as F
import hashlib,inspect,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[5]
code=repo/'microbench/gh200_resource_campaign/access_rules'
sys.path.insert(0,str(code));sys.dont_write_bytecode=True
import interval_snapshot as interval
import supply_model
from analyze_r10_granule_overlap import source_granules

counts=Counter();narrowing=Counter()
original=interval.PriceCell.sign
def counted(self,slope,intercept):
    frame=inspect.currentframe()
    while frame and frame.f_code.co_filename!='<exact existing first-output graph>':
        frame=frame.f_back
    line=frame.f_lineno if frame else 0
    before=(self.lower,self.upper)
    result=original(self,slope,intercept)
    counts[(line,bool(slope))]+=1
    if before!=(self.lower,self.upper):narrowing[line]+=1
    return result
interval.PriceCell.sign=counted
reference=json.loads((root/'reference-cell.json').read_text())
fixed=json.loads((root.parent/'B-20261011-input-path-v1/B-input-path.json').read_text())
contexts=json.loads((root.parent/'B-20261011-source-event-transfer-v3/B-r10-event-transfer.json').read_text())['contexts']
context=contexts['r10_source_single126_p32'];row,setup=context['row'],context['setup']
program=supply_model.cfg_b_source_program(row,setup)
keys={key for groups in program['source_keys'] for group in groups for key in group}
units={key:tuple((key[0],i) for i in source_granules(key,128)) for key in keys}
parameters=fixed['parameters']
args=dict(source_keys=program['source_keys'],output_counts=program['output_counts'],
    share_inflight_sources=True,source_address_units=units,
    source_unit_service_ns=parameters['service_ns']/128,
    input_min_latency_ns=parameters['read_latency_ns'],**parameters)
lower=F(fixed['selection']['lower_exact']);upper=F(fixed['selection']['upper_exact'])
graph,_=interval.exact_first_graph()
cell,_=interval.first_certificate(graph,(lower+upper)/2,lower,upper,row['k']//64,6,126,args)
assert cell==reference['cells'][0]
source=inspect.getsource(supply_model.shared_stage_events).splitlines()
rows=[dict(line=line,source=source[line-1] if line else 'outside graph',
           nonconstant=nonconstant,comparisons=count,narrowings=narrowing[line])
      for (line,nonconstant),count in sorted(counts.items())]
width=F(cell['upper'])-F(cell['lower'])
whole=upper-lower
result=dict(snapshot_sha256=hashlib.sha256((root/'interval_snapshot.py').read_bytes()).hexdigest(),
    reference_cell_identical=True,cell_width_exact=str(width),
    fraction_of_prescribed_interval_exact=str(width/whole),
    fraction_of_prescribed_interval=float(width/whole),counts=rows,
    nonconstant_comparisons=sum(v for (line,nonconstant),v in counts.items() if nonconstant),
    whole_interval_proven=False,
    limits='No comparison removed. This cell fraction is not an estimate of total cell count or runtime. Heap comparisons may affect resource ordering and cannot be discarded without a commutativity or dependency argument.')
assert result['nonconstant_comparisons']==cell['comparisons']
(root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

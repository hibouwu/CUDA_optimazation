from pathlib import Path
from collections import deque
import json,gzip,hashlib
out=Path(__file__).resolve().parent;run=out.parents[1];base=run.parent
inputs={}
def sha(path):inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return inputs[str(path)]
def path_to(nodes,start,target):
 children={n:[] for n in nodes}
 for n,parents in nodes.items():
  for p in parents:children[p].append(n)
 queue=deque([start]);previous={start:None}
 while queue:
  u=queue.popleft()
  if u==target:
   result=[]
   while u is not None:result.append(u);u=previous[u]
   return result[::-1]
  for v in children[u]:
   if v not in previous:previous[v]=u;queue.append(v)
 return None
queries=[]
for tiles in (1,2):
 p=base/'20261009-R13-R15-R18-composition-job738496/reanalysis/main-cfg-c-rank-dependencies-v2'/f'cfg_c-S4-K64-T{tiles}-rank-dependencies.json.gz';sha(p)
 with gzip.open(p,'rt') as f:graph=json.load(f)
 for rank in (0,1):
  for wg in (0,1):
   for tail_rank in (0,1):
    start=f'r{rank}.w{wg}.t0.MAIN';end=f'r{tail_rank}.producer_tail'
    path=path_to(graph['nodes'],start,end);reverse=path_to(graph['nodes'],end,start)
    assert bool(path)==(tiles==2) and reverse is None
    queries.append(dict(output_tiles=tiles,source=start,target=end,path=path,reverse_path=reverse))
records=json.loads((run/'returned/records.json').read_text());observations=[]
for row in records:
 if row['flag']!=1:continue
 raw=run/'returned'/row['raw'];assert sha(raw)==row['sha256']
 with gzip.open(raw,'rt') as f:events={p['event']:p for p in map(json.loads,f)}
 t=events['call']['trace'];tails=[t[c*1296+15] for c in range(132)];earliest=min(tails);earliest_cta=tails.index(earliest)
 before=[];after=[];same_cluster=[]
 for cta in range(132):
  h=cta*1296;tiles=t[h+6];assert tiles==t[h+9]
  partner=(cta+66)%132;assert tiles==t[partner*1296+6]
  cluster_min=min(tails[cta],tails[partner])
  for wg in (0,1):
   main=t[h+16+wg*640+7]
   item=dict(cta=cta,wg=wg,output_tiles=tiles,first_MAIN_ns=main)
   (before if main<earliest else after).append(item)
   if tiles==2:same_cluster.append(cluster_min-main)
 assert len(same_cluster)==168 and min(same_cluster)>0
 observations.append(dict(pair=row['pair'],earliest_tail_cta=earliest_cta,earliest_tail_output_tiles=t[earliest_cta*1296+6],earliest_tail_ns=earliest,first_MAIN_strictly_before_all_tail_stamps=before,first_MAIN_at_or_after_earliest_tail_stamp=after,two_output_MAIN_to_cluster_tail_margin_ns=[min(same_cluster),max(same_cluster)]))
result=dict(input_sha256=inputs,static_queries=queries,processes=observations,observation_qualified=False,limits=['Source reachability is within each physical cluster, not a global scheduling or shared-resource order.','One-output MAIN and producer_tail are both downstream of releases; neither orders the other.','Cross-CTA raw globaltimer order is reported as observed, not as qualified physical-time enclosure.','Two-output first MAIN precedes own cluster tail, but other clusters can execute tail observation earlier.','No counterfactual proof of tail overhead, no gate change, no new GPU.'])
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([dict(pair=r['pair'],earliest_tail_tiles=r['earliest_tail_output_tiles'],before=len(r['first_MAIN_strictly_before_all_tail_stamps']),at_or_after=len(r['first_MAIN_at_or_after_earliest_tail_stamp']),local_margin=r['two_output_MAIN_to_cluster_tail_margin_ns']) for r in observations]))

"""Scope and work normalization of the existing last-WG0 R10 windows only."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json,statistics,re
ROOT=Path(__file__).resolve().parent
RUN=ROOT.parents[1]
inputs={}
def read(path):
 data=path.read_bytes();inputs[str(path.resolve())]=hashlib.sha256(data).hexdigest();return data
def load(path):return json.loads(read(path))
def summarize(values):return dict(minimum=str(min(values)),median=str(statistics.median(values)),maximum=str(max(values)))
binding=load(RUN/'reanalysis/main-boundary-binding-v1/B-boundary-binding.json')
prior=load(RUN/'reviews/ROOT-independent.json')
summary=load(RUN/'analysis-formal/summary.json')
source_manifest=load(RUN/'source_hashes.json')
source_names=['probes/r10.cu','overlay/r10_trace.hpp',
 'overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
 'overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp']
source={}
for name in source_names:
 path=RUN/'source'/name;source[name]=read(path).decode()
 assert inputs[str(path.resolve())]==source_manifest[name]
collective=source[source_names[2]];kernel=source[source_names[3]]
assert 'static constexpr int K_PIPE_MMAS = 1;' in collective
assert collective.index('pipeline.consumer_wait(smem_pipe_read, barrier_token);',collective.index('int prologue_mma_count'))<collective.index('r10_stamp(2 *')
assert kernel.index('collective_mainloop.mma_tail(')<kernel.index('r10_tail(int(')
assert 'warpgroup_wait<0>();' in collective[collective.index('void\n  mma_tail'):]
binaries=load(RUN/'build/binary_hashes.json')
assert hashlib.sha256(read(RUN/'build/cfg_b_trace')).hexdigest()==binaries['cfg_b_trace']
sass=read(RUN/'build/cfg_b_trace.sass').decode()
for name in ['build/cfg_b_trace.sass','build/binary_hashes.json']:
 assert inputs[str((RUN/name).resolve())]==prior['evidence'][str((RUN/name).resolve())]
hgmma=[line.strip() for line in sass.splitlines() if 'HGMMA.' in line]
assert len(hgmma)==16 and all('HGMMA.64x128x16.F32' in line for line in hgmma)
valid_work=2*128*128*2000
dense_work=2*128*128*(32*64)
assert Q(dense_work,valid_work)==Q(128,125)
cases=[];all_rows=[]
for case in binding['cases']:
 name=case['case'];old=next(c for c in summary['cases'] if c['id']==name)
 observed=[]
 for trial in range(10):
  path=RUN/f'samples/formal/{name}/trace/{trial:02d}/result.json'
  record=load(path);assert inputs[str(path.resolve())]==prior['evidence'][str(path.resolve())]
  assert record['binary_sha256']==binaries['cfg_b_trace']
  assert (record['m'],record['n'],record['k'],record['stages'],record['grid_ctas'])==(2600,3000,2000,6,132)
  assert record['trace_schema']==4 and record['trace_tile']=='last' and record['trace_consumer_groups']==[0]
  for window in case['trace_windows']:
   cta=window['cta'];assert window['derived_output_ordinal']==2 and window['derived_coord']==[12+cta,12]
   assert window['source_valid_bytes']==1024000 and window['destination_fill_bytes']==24576
   assert window['first_ring_state']['group']==64 and window['last_ring_state']['group']==95
   words=record['trace'][cta*8:cta*8+8]
   assert words[4]==2 and words[2:4]==[0,0] and words[5]==0 and words[7]==4
   cycles=words[1]-words[0];assert cycles>0
   prior_window=next(v for v in old['cta_observations'] if v['trial']==trial and v['cta']==cta and v['group']==0)
   assert prior_window['last_tile_mainloop_cycles']==cycles
   assert Q(cycles,32)==Q(prior_window['cycles_per_ktile_including_fill_tail'])
   item=dict(case=name,trial=trial,cta=cta,sm=words[6],WG=0,output_ordinal=2,coord=window['derived_coord'],
    observed_window_cycles=cycles,cycles_per_complete_K64_group=str(Q(cycles,32)),
    cycles_per_effective_K64_equivalent=str(Q(cycles*64,2000)),
    valid_flop_per_WG_window_cycle=str(Q(valid_work,cycles)),
    dense_tile_flop_equivalent_per_WG_window_cycle=str(Q(dense_work,cycles)))
   assert Q(item['cycles_per_effective_K64_equivalent'])/Q(item['cycles_per_complete_K64_group'])==Q(128,125)
   assert Q(item['dense_tile_flop_equivalent_per_WG_window_cycle'])/Q(item['valid_flop_per_WG_window_cycle'])==Q(128,125)
   observed.append(item)
 assert len(observed)==40
 cycle_values=[Q(v['observed_window_cycles']) for v in observed]
 per_group=[Q(v['cycles_per_complete_K64_group']) for v in observed]
 assert statistics.median(per_group)==Q(old['mainloop_cycles_per_ktile_median'])
 assert [min(per_group),max(per_group)]==[Q(v) for v in old['mainloop_cycles_per_ktile_range']]
 cases.append(dict(case=name,windows=40,window_cycles=summarize(cycle_values),cycles_per_complete_K64_group=summarize(per_group),
  cycles_per_effective_K64_equivalent=summarize([Q(v['cycles_per_effective_K64_equivalent']) for v in observed]),
  valid_flop_per_WG_window_cycle=summarize([Q(v['valid_flop_per_WG_window_cycle']) for v in observed]),
  dense_tile_flop_equivalent_per_WG_window_cycle=summarize([Q(v['dense_tile_flop_equivalent_per_WG_window_cycle']) for v in observed])))
 all_rows+=observed
responses=[]
for a,b in [(0,1),(0,2),(1,2)]:
 ratio=Q(cases[a]['window_cycles']['median'])/Q(cases[b]['window_cycles']['median'])-1
 for field in ['cycles_per_complete_K64_group','cycles_per_effective_K64_equivalent']:
  assert Q(cases[a][field]['median'])/Q(cases[b][field]['median'])-1==ratio
 responses.append(dict(numerator=cases[a]['case'],denominator=cases[b]['case'],median_cycle_response=str(ratio),normalization_invariant=True))
report=dict(scope='Archived four interior-MN, last-WG0 j2 windows; 30 processes, 120 observations only.',
 input_sha256=inputs,valid_window_flop=valid_work,dense_tile_window_flop_equivalent=dense_work,
 dense_to_valid_work_ratio='128/125',K_effective=2000,K_dense_tile=2048,K_groups=32,
 static_SASS_HGMMA_shape='64x128x16',static_SASS_HGMMA_sites=16,
 cases=cases,observations=all_rows,median_cycle_responses=responses,
 original_cycles_per_K_group_exactly_reproduced=True,
 boundary='First full-barrier wait returned -> mma_tail returned. Initial data wait excluded; later input waits, loop/control, math completion and final releases remain inside.',
 limits=['Dense tile work is an instruction-shape equivalent, not a count of physical internal arithmetic or achieved Tensor Core throughput.',
 'WG-window cycles are not exclusive SM compute cycles; no naked-MMA, whole-SM or whole-GEMM throughput claim.',
 'The final partial K group remains one complete tile group. No startup/tail subtraction or service price fit.',
 'Old GPU-54896349, geometry and prefix protocol remain distinct from newer ef-card records; no cross-card price transfer.',
 'MN edge outputs and first-output service are not observed by these four j2 windows.'],
 numerical_evidence='Prior SHA-bound exact sample review reused; no new numeric matrix replay.',
 observation_qualified=False,prediction_qualified=False,new_GPU=False)
(ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(windows=len(all_rows),input_SHA=len(inputs),work_ratio='128/125',cases=cases,responses=responses),indent=2))

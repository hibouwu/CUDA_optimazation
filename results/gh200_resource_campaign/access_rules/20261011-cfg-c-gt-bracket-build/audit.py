"""Check the frozen SM90a dependency witnesses; no hardware qualification."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.with_name('20261011-cfg-c-producer-tail-check-build')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=ROOT/'returned/build'
sass=build/'cfg_c_s4_dual.sass'
text=sass.read_text()
# The CUTLASS kernel is the first function; PC addresses are function-local.
functions=text.split('Function : ')[1:]
assert 'cutlass' in functions[0]
rows={int(m.group(1),16):' '.join(m.group(2).split()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s+;',functions[0])}
witnesses=[]
for event,start in [('FIRST_MMA',0x1790),('MAIN_END',0x2450)]:
 invalid=start+0x90;done=start+0xc0
 expected=[
  'CS2R R12, SR_GLOBALTIMERLO',
  'ISETP.NE.U32.AND P3, PT, R12, RZ, PT',
  'ISETP.NE.U32.AND.EX P3, PT, R13, RZ, PT, P3',
  f'@!P3 BRA 0x{invalid:x}',
  'CS2R R14, SR_CLOCKLO',
  'ISETP.NE.U32.AND P3, PT, R14, RZ, PT',
  'ISETP.NE.U32.AND.EX P3, PT, R15, RZ, PT, P3',
  '@P3 CS2R R16, SR_GLOBALTIMERLO',
  f'@P3 BRA 0x{done:x}',
  'CS2R R12, SRZ','CS2R R14, SRZ','CS2R R16, SRZ']
 for j,instruction in enumerate(expected):assert rows[start+16*j]==instruction,(event,j,rows[start+16*j])
 # Only the preceding flag branch enters off path. No other branch targets the
 # middle of the chain, bypassing either prerequisite. Internal exit is allowed.
 for pc,instruction in rows.items():
  target=re.search(r'\bBRA 0x([0-9a-f]+)',instruction)
  if target and start<int(target.group(1),16)<invalid:
   raise AssertionError(('chain bypass',pc,instruction))
 witnesses.append(dict(event=event,pc_gt_before=hex(start),pc_cycle=hex(start+0x40),
  pc_gt_after=hex(start+0x70),pc_invalid=hex(invalid),instructions=expected,
  gt_result_consumed_by_control=True,cycle_result_consumed_by_predicate=True))
resources=(build/'cfg_c_s4_dual.resources').read_text()
oldresources=(BASE/'returned/build/cfg_c_s4_dual.resources').read_text()
resource_rows=[s.strip() for s in resources.splitlines() if 'REG:' in s]
old_resource_rows=[s.strip() for s in oldresources.splitlines() if 'REG:' in s]
assert resource_rows==old_resource_rows
assert 'REG:168 STACK:0 SHARED:1024 LOCAL:0' in resource_rows[0]
log=(build/'cfg_c_s4_dual.log').read_text()
assert 'Used 168 registers, used 2 barriers' in log
assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in log
oldtext=(BASE/'returned/build/cfg_c_s4_dual.sass').read_text()
counts={x:dict(base=len(re.findall(x,oldtext)),candidate=len(re.findall(x,text))) for x in ['HGMMA','WARPGROUP','SYNCS.ARRIVE','SR_GLOBALTIMERLO','SR_CLOCKLO']}
for x in ['HGMMA','WARPGROUP','SYNCS.ARRIVE']:assert counts[x]['base']==counts[x]['candidate']
# Changed/added source identities and archive/returned linkage are checked too.
freeze=json.loads((ROOT/'freeze.json').read_text())
for name,digest in freeze.items():assert sha(ROOT/name)==digest,name
hashes=json.loads((ROOT/'source_hashes.json').read_text())
for name,digest in hashes.items():assert sha(ROOT/name)==digest,name
for name in ['contract.json','source_hashes.json','run.sh']:
 assert (ROOT/name).read_bytes()==(ROOT/'returned'/name).read_bytes(),name
assert json.loads((ROOT/'receipt.json').read_text())['returncode']==0
allocation=(ROOT/'returned/allocation.txt').read_text()
assert 'JobId=740156' in allocation
assert 'gres/gpu' not in allocation
report=dict(job=740156,device_executed=False,binary_sha256=sha(build/'cfg_c_s4_dual'),
 sass_sha256=sha(sass),source_manifest_sha256=sha(ROOT/'source_hashes.json'),
 source_files_verified=len(hashes),witnesses=witnesses,resource_rows=resource_rows,
 resource_rows_equal_base=True,gemm_registers=168,barriers=2,stack_bytes=0,spill_bytes=0,
 instruction_counts=counts,shadow_bytes=9504,original_scratch_bytes=132*1296*8,
 compiler_dependency_witness=True,hardware_sampling_enclosure=False,
 physical_ns_accuracy=False,numerical_correctness=False,observer_qualified=False,
 null_control_passed=False,prediction_qualified=False,
 limitation='Exact frozen binary dependency witness; no claim about physical special-register sampling latency or nonperturbation.')
(ROOT/'build-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['job','binary_sha256','source_files_verified','compiler_dependency_witness','observer_qualified']},indent=2))

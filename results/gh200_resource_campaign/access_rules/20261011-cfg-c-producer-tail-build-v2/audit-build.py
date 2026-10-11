#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,re
r=Path(__file__).resolve().parent;d=r/'returned';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((r/'receipt.json').read_text());assert receipt['returncode']==0 and sha(r/'output.tar.gz')==receipt['sha256']
for name,h in json.loads((r/'freeze.json').read_text())['files'].items():assert sha(r/name)==sha(d/name)==h
assert (d/'exit_code.txt').read_text().strip()=='0'
assert 'release 12.9' in (d/'compiler.txt').read_text()
sass=d/'build/cfg_c_s4_dual.sass';binary=d/'build/cfg_c_s4_dual'
part=next(p for p in sass.read_text().split('Function : ') if p.startswith('_ZN7cutlass13device_kernel'))
symbol=part.splitlines()[0].strip();ins={int(pc,16):op.strip() for pc,op in re.findall(r'/\*([0-9a-f]+)\*/\s+([^;]+);',part)}
log=(d/'build/cfg_c_s4_dual.log').read_text().split('Function properties for '+symbol)[1][:250]
assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in log and 'Used 168 registers, used 2 barriers' in log
checks={0x5ab0:'ELECT P0, URZ, PT',0x5ac0:'@!P0 EXIT',0x6030:'BRA 0x5ac0',0x5bb0:'@!P0 BRA 0x6040',0x6090:'BRA 0x5bc0',
0x5c40:'@!P0 BRA 0x60a0',0x60f0:'BRA 0x5c50',0x5cd0:'@!P0 BRA 0x6100',0x6150:'BRA 0x5ce0',0x5cf0:'@!P0 BRA 0x6160',0x61b0:'BRA 0x5d00',
0x5d20:'@!P0 EXIT',0x5d30:'CS2R R6, SR_CLOCKLO',0x5d40:'CS2R R4, SR_GLOBALTIMERLO',0x5da0:'IMAD.WIDE.U32 R2, R9, 0x2880, R2',
0x5db0:'STG.E.64 desc[UR22][R2.64+0x70], R6',0x5dc0:'STG.E.64 desc[UR22][R2.64+0x78], R4',0x5dd0:'EXIT'}
for pc,op in checks.items():assert ins[pc]==op,(hex(pc),ins[pc])
for fast,slow,retry,back in ((0x5b40,0x6070,0x6050,0x6080),(0x5bc0,0x60d0,0x60b0,0x60e0),(0x5c50,0x6130,0x6110,0x6140),(0x5ce0,0x6190,0x6170,0x61a0)):
 assert 'SYNCS.PHASECHK.TRANS64.TRYWAIT P0' in ins[fast]
 assert ins[slow].startswith('@!P0 SYNCS.PHASECHK.TRANS64 P0')
 assert ins[back]=='@!P0 BRA '+hex(retry)
report=dict(job=740082,binary_sha256=sha(binary),sass_sha256=sha(sass),registers=168,barriers=2,stack_bytes=0,spill_bytes=0,
    static_shared_bytes=1024,trace_stride_bytes=0x2880,writer='original elected mainloop producer lane',
    instructions={hex(pc):op for pc,op in ins.items() if 0x5a90<=pc<=0x5dd0 or 0x5fb0<=pc<=0x61b0},
    target_compile_passed=True,elected_tail_control_checked=True,device_numerical_verified=False,observation_qualified=False,
    limits='Specific PCs and binary only; static control/compiled resources, not hardware clock enclosure, dynamic shared size, numeric correctness or observer perturbation.')
(r/'build-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='instructions'}))

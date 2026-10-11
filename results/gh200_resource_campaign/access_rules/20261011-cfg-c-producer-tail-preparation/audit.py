#!/usr/bin/env python3
"""Check only prepared source identity and the exact cooperative hook insertion."""
from pathlib import Path
import hashlib,json,sys,tempfile
r=Path(__file__).resolve().parent;repo=next(p for p in r.parents if (p/'microbench').is_dir());code=repo/'microbench/gh200_resource_campaign/access_rules';sys.path.insert(0,str(code))
import run_r18,v06_run as common
from prepare_cfg_c_producer_tail import HELPER,HOOK
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in json.loads((r/'source_hashes.json').read_text()).items():assert sha(r/name)==h
with tempfile.TemporaryDirectory() as tmp:
    plain=Path(tmp)/'baseline';run_r18.prepare(plain,r/'source/cutlass','r18',rows=json.loads((r/'cases.json').read_text()),dual_clock=True)
    a=plain/'source';b=r/'source'
    assert (b/'overlay'/common.COOP).read_text().replace(HOOK+'\n        r18_producer_tail_stamp();',HOOK)==(a/'overlay'/common.COOP).read_text()
    assert (b/'probes/v06_trace.hpp').read_text()==(a/'overlay/v06_trace.hpp').read_text()+HELPER
    for name in (common.MAINLOOP,common.PING):assert (b/'overlay'/name).read_bytes()==(a/'overlay'/name).read_bytes()
    assert (b/'overlay/v06_trace.hpp').read_text()=='#pragma once\n#include "../probes/v06_trace.hpp"\n'
text=(r/'source/probes/r18.cu').read_text()
assert text.count('<< ",\\\"producer_tail_observer\\\":"')==2
assert text.index('unsigned producer_tail_off=0;')<text.index('auto set_target_epilogue')
assert '#if defined(R18_PRODUCER_TAIL) || defined(R10_WORK_SWAP)' in text
assert 'o.mode!="setup" && o.mode!="warm"' in text
assert 'threadIdx.x != 0 || !r18_producer_tail_observer' in HELPER
commands=json.loads((r/'build/commands.json').read_text());assert len(commands)==1
command=next(iter(commands.values()))
assert all(flag in command for flag in ('-DR18_PRODUCER_TAIL','-DR18_DUAL_CLOCK','-DV06_CFG=2','-DR18_STAGE_COUNT=4','-DV06_TRACE'))
report=dict(source_files=len(json.loads((r/'source_hashes.json').read_text())),kernel_hook_exact=True,original_collectives_unchanged=True,
    canonical_header=True,trace_words_per_CTA=1296,trace_allocation_bytes=132*1296*8,
    new_timestamp_payload_bytes=132*16,device_compile_verified=False,numerical_verified=False,observation_qualified=False)
(r/'source-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

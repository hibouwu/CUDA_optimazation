#!/usr/bin/env python3
"""Join separately verified compile, runtime and current-reader evidence."""
from pathlib import Path
import hashlib,json,subprocess,sys
out=Path(__file__).resolve().parent
run=out.parents[1]
frame=run.parent/'20261011-R09-guarded-frame-build/reanalysis/A-frame-compile-audit-v1'
smoke=run/'reanalysis/A-smoke-boundary-audit-v1'
reader=run/'reanalysis/main-packet-header-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for directory,script in ((frame,'audit.py'),(smoke,'audit.py'),(reader,'replay.py')):
    path=directory/script
    result=subprocess.run([sys.executable,str(path)],capture_output=True,text=True,check=True)
    checks.append(dict(script=str(path),sha256=sha(path),stdout=result.stdout,returncode=result.returncode))
f=json.loads((frame/'audit.json').read_text());s=json.loads((smoke/'audit.json').read_text());r=json.loads((reader/'report.json').read_text())
assert f['compile_interface_checks_passed'] and f['source_files_verified']==860
assert s['actual_raw_complete_heads_valid'] and s['actual_packets_valid']
assert not s['parser_complete_head_refusal'] # Historical snapshot counterexamples must remain reproducible.
assert all(a['record_valid'] for a in r['arms'])
assert [len(a['rejected_mutations']) for a in r['arms']]==[0,5,5]
assert all(not a['observation_qualified'] for a in (f,s,r))
report=dict(checks=checks,input_sha256={str(p):sha(p) for p in (frame/'audit.json',smoke/'audit.json',reader/'report.json')},
    compile=dict(source_files=860,all_four_kernels_registers=168,barriers=2,stack_bytes=0,spill_bytes=0,
        consumer_register_pool=232,producer_register_pool=40,static_shared_bytes=1024,
        original_overlay_bytes_identical=True,guarded_sites=f['sites'],launch_threads=384),
    runtime=dict(resources=s['resources'],sampled_values=s['numerical_values_verified'],full_D_verified=False,
        entry_producer_and_SM_valid=True,original_snapshot_rejection_gap_preserved=True,current_reader_rejects_all_ten_mutations=True),
    conditional_arithmetic=dict(feasible_CTA_pairs_at_128=s['feasible_at_128'],minimum_total_interval_slack_ns=s['minimum_total_interval_slack_ns']),
    observation_qualified=False,remaining=['Hardware sampling enclosure is not established by SASS ordering.',
        'One call per arm cannot qualify local or spatial perturbation.',
        'API maximum CTA count is not measured executed occupancy.',
        'Static operation counts and unchanged overlays do not prove dynamic timing equivalence.',
        'No new local/critical/total heldout validation or cross-card evaluation follows.'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(verified_scripts=len(checks),source_files=860,sampled_values=s['numerical_values_verified'],observation_qualified=False)))

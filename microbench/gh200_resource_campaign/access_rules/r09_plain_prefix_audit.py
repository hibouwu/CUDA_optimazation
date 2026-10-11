#!/usr/bin/env python3
"""Read-only history/endpoint audit of the two-process plain-module smoke.

No timing score, fitting, GPU work, or observer-validity qualification. Runtime
prefix identity is checked separately from the old common-sw1-16 label.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import tarfile

from analyze_r18 import replay
from r09_time_contract import constant_rate_slack


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe_prefix(setup,call):
    if (call.get('plain_prefix_abi')!=1 or call.get('plain_prefix_calls')!=16
            or call.get('warmup_calls')!=16 or call.get('plain_prefix_same_buffers') is not True):
        raise ValueError('explicit sixteen-call plain-module ABI identity required')
    if setup['grid']!=[1,132,1] or setup['config']!='cfg_b' or setup['stages']!=6:
        raise ValueError('plain-module grid/config/stage scope changed')
    if [setup[k] for k in ('m','n','k','swizzle','lda','ldb','ldd')]!=[2816,1536,4096,1,4096,1536,1536]:
        raise ValueError('plain-module target changed')
    if any(call[k]!=setup[k] for k in ('target_a_ptr','target_b_ptr','prefix_a_ptr','prefix_b_ptr')):
        raise ValueError('within-process prefix pointer identity changed')
    if call['prefix_a_ptr']!=call['target_a_ptr'] or call['prefix_b_ptr']!=call['target_b_ptr']:
        raise ValueError('prefix input buffers differ from target')
    warm,gaps=call['warmup_us'],call['plain_prefix_inter_call_us']
    if len(warm)!=16 or len(gaps)!=16 or any(not math.isfinite(v) or v<=0 for v in warm) or any(not math.isfinite(v) or v<0 for v in gaps):
        raise ValueError('complete finite prefix/event gaps required')
    kernel=sum(warm);between=sum(gaps[:15]);prefix_envelope=kernel+between
    return dict(protocol='plain-module-16-sw1/abi1',legacy_policy=call['warmup_policy'],
        grid=setup['grid'],scratch_clear_bytes_per_invocation=setup['scratch_bytes'],
        prefix_calls=16,prefix_kernel_event_sum_us=kernel,
        prefix_first_to_last_event_us=prefix_envelope,
        prefix_between_gap_sum_us=between,prefix_between_gap_fraction=between/prefix_envelope,
        last_prefix_to_target_event_gap_us=gaps[15],all_gap_median_us=st.median(gaps),
        observed_kernel_event_us=warm,observed_event_gaps_us=gaps,
        scope='Observed prefix diagnostics only; gaps include preparation/host/device effects, not pure idle. Physical initial states are not identified.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.run.resolve();returned=root/'returned'
    runtime=json.loads((root/'runtime-summary.json').read_text())
    provenance={str(root/'runtime-summary.json'):sha(root/'runtime-summary.json'),
                str(root/'input.tar.gz'):sha(root/'input.tar.gz'),
                str(returned/'run.sh'):sha(returned/'run.sh')}
    reports={}
    for variant in ('plain','dual'):
        raw=returned/(variant+'.jsonl.gz')
        if sha(raw)!=runtime[variant]['raw_sha256']:
            raise ValueError('smoke raw identity mismatch')
        with gzip.open(raw,'rt') as stream:
            events={e['event']:e for line in stream if line.strip() for e in [json.loads(line)]}
        setup,call=events['setup'],events['call']
        history=describe_prefix(setup,call)
        row={k:setup[k] for k in ('config','m','n','k','lda','ldb','ldd','zero_m','zero_n',
            'storage_m','storage_n','swizzle','alloc_lda','alloc_ldb','input_mode','seed')}
        row.update(id='plain-prefix-smoke-'+variant,kind='ordinary',warmup_policy='common-sw1-16')
        record=dict(raw=raw.name,raw_sha256=sha(raw),variant=variant,trial=0)
        observed=replay(returned,record,row,retain_roles=True)
        item=dict(history=history,gpu_uuid=setup['gpu_uuid'],numerical_values_replayed=observed['checked_values'],
                  ctas_replayed=len(observed['ctas']),formal_timing=False)
        if variant=='dual':
            phases={'first':[],'later':[]};slacks=[]
            for cta in observed['ctas']:
                windows=[]
                for role,phase in enumerate(('first','later')):
                    r=cta['roles'][role]
                    if len(r['tiles'])!=1:
                        raise ValueError('two-output original-role mapping changed')
                    c,n=r['tiles'][0],r['tiles_ns'][0]
                    dc,dn=c[1]-c[0],n[1]-n[0]
                    phases[phase].append(dict(cta=cta['cta'],role=role,sm=cta['sm'],cycles=dc,ns=dn,rate_ghz=dc/dn))
                    windows.append((dc,dn))
                slacks.append(dict(cta=cta['cta'],**constant_rate_slack(windows)))
            item['endpoint_arithmetic']=dict(phases=phases,per_cta_slack=slacks,
                phase_median_rate_ghz={phase:st.median(x['rate_ghz'] for x in xs) for phase,xs in phases.items()},
                necessary_endpoint_slack_median_ns=st.median(x['necessary_endpoint_slack_ns'] for x in slacks),
                scope='Single smoke call, original CLOCK/GT endpoints. Arithmetic only; no read-pair error bound or mechanism/perturbation qualification.')
        reports[variant]=item
        provenance[str(raw)]=sha(raw)
    archive_sources={}
    with tarfile.open(root/'input.tar.gz','r:gz') as archive:
        # The package's last overlay member is a stub; the effective trace
        # implementation is source/probes/v06_trace.hpp. Keep all duplicate
        # member hashes instead of mistaking the earlier member for compiled IO.
        trace_members=[m for m in archive.getmembers() if m.name=='source/overlay/v06_trace.hpp']
        if not trace_members:
            raise ValueError('packaged overlay include path absent')
        duplicate_trace_members=[hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in trace_members]
        for name in ('source/probes/r18.cu','source/probes/r09_plain_prefix.cu','source/probes/r09_plain_prefix_host.hpp',
                     'source/overlay/v06_trace.hpp','source/probes/v06_trace.hpp'):
            payload=archive.extractfile(name).read()
            archive_sources[name]=dict(sha256=hashlib.sha256(payload).hexdigest(),text=payload.decode())
    if '#include "../probes/v06_trace.hpp"' not in archive_sources['source/overlay/v06_trace.hpp']['text']:
        raise ValueError('effective trace include path needs a new source audit')
    host=archive_sources['source/probes/r18.cu']['text']
    if 'cudaMemset(trace.pointer, 0, trace.count * 8)' not in host or 'if (use_plain_prefix) plain_prefix.run();' not in host:
        raise ValueError('packaged preparation path needs a new source audit')
    result=dict(scope='Development audit of two correctness-smoke processes, not a formal clock experiment.',
        run=str(root),processes=reports,matched_prefix_kernel_from_integration=runtime['plain']['frozen_kernel_sass_identical'] and runtime['dual']['frozen_kernel_sass_identical'],
        matched_preparation_bytes=reports['plain']['history']['scratch_clear_bytes_per_invocation']==reports['dual']['history']['scratch_clear_bytes_per_invocation'],
        input_sha256=provenance,archive_source_sha256={k:v['sha256'] for k,v in archive_sources.items()},
        overlay_duplicate_members_in_archive_order=duplicate_trace_members,
        analysis_sha256=sha(Path(__file__)),
        implications=['The same common-sw1-16 label does not identify the same prefix executor or preparation byte sequence.',
            'The two smoke processes clear different scratch sizes before each prefix/target. Their event gaps are explicit history observations.',
            'Matching a plain kernel and A/B pointers does not prove matched preparation or physical initial state.',
            'These observed prefix gaps can diagnose history; a free predictor needs a declared preparation/gap rule, not target W/f.',
            'The single dual endpoint arithmetic is retained only to guide observation validity; it is neither a local holdout nor a clock candidate score.'])
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'prefix-history-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,source in archive_sources.items():
        destination=out/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(source['text'])
    print(json.dumps({v:dict(prefix=reports[v]['history']['protocol'],clear_bytes=reports[v]['history']['scratch_clear_bytes_per_invocation'],
        gap_median_us=reports[v]['history']['all_gap_median_us'],gap_fraction=reports[v]['history']['prefix_between_gap_fraction']) for v in reports},indent=2))
    print('endpoint diagnostic',reports['dual']['endpoint_arithmetic']['phase_median_rate_ghz'],reports['dual']['endpoint_arithmetic']['necessary_endpoint_slack_median_ns'])


if __name__=='__main__':
    main()

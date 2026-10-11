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

from analyze_r18 import reference
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


LEGACY_SMOKE_SHA256 = {
    'plain':'8772e4bfd6192ece9875714b3ad866625c3c032d7ada3c277c05d84a4a2d51eb',
    'dual':'4119552e23356f00aba11f2df791e142a2bdb84ba5c47c4b05e2a2622ebfe690',
}


def historical_smoke_events(events,variant):
    """Diagnostic decoder for 740048 r5, never current prefix qualification.

    The IO entry below restricts this path to the two original raw hashes.
    Keep missing protocol/preparation metadata missing; do not call or weaken
    the public replay validator. Only numeric checks and original endpoints
    are needed here, so no whole-kernel event model is reconstructed.
    """
    if variant not in LEGACY_SMOKE_SHA256:
        raise ValueError('only the two historical smoke variants are supported')
    setup,call,check=(events[k] for k in ('setup','call','check'))
    if any(key.startswith(('plain_prefix_protocol','plain_prefix_preparation','plain_prefix_clear_bytes'))
           for event in (setup,call) for key in event):
        raise ValueError('new prefix metadata cannot use the historical smoke path')
    describe_prefix(setup,call)
    if setup.get('input_mode')!='dyadic' or setup.get('seed')!=17 or setup.get('gpu_uuid')!='GPU-43269fbc-449d-3e0f-908a-9c81229546d3':
        raise ValueError('historical smoke input/card identity changed')
    if check['status']!='ok' or check['padding_errors'] or check['nonfinite']:
        raise ValueError('historical smoke correctness check failed')
    indices,values=check['checked_indices'],check['checked_values']
    if len(indices)!=4096 or len(values)!=4096 or len(set(indices))!=4096:
        raise ValueError('all 4096 distinct historical check values required')
    for index,value in zip(indices,values):
        i,j=divmod(index,setup['n'])
        if not 0<=i<setup['m'] or value is None or not math.isfinite(value) or value!=reference(i%17,j%17,setup['k'],17):
            raise ValueError('historical smoke numeric replay failed')
    result=dict(checked_values=4096,ctas=[],historical_reader='740048-r5-only',
                current_prefix_protocol_qualified=False,
                qualification_reason='Original r5 lacks current protocol and preparation identity; arithmetic/correctness diagnostics only.')
    if variant=='plain':
        return result
    width=16+2*64*10
    if setup.get('trace_version')!='r18-dual-clock-events' or setup.get('trace_words')!=width or len(call['trace'])!=132*width:
        raise ValueError('historical dual trace layout changed')
    seen=[]
    for cta in range(132):
        head=call['trace'][cta*width:(cta+1)*width]
        if not head[0] or not head[2] or head[10]:
            raise ValueError('historical CTA trace absent or overflowed')
        roles=[]
        for role in (0,1):
            end,ns,count=head[4+3*role:7+3*role]
            if count!=1 or head[11+role]!=head[2]:
                raise ValueError('historical original-role work/SM identity changed')
            start=16+role*64*10
            cycles=head[start:start+4];mi,ni=head[start+4:start+6];times=head[start+6:start+10]
            if not all(cycles) or min(mi,ni)<=0 or not cycles[0]<=cycles[1]<=cycles[2]<=cycles[3]<=end:
                raise ValueError('historical cycle endpoints invalid')
            if not head[1]<=head[13]<=times[0]<=times[1]<=times[2]<=times[3]<=ns:
                raise ValueError('historical timer endpoints invalid')
            roles.append(dict(final_c=end,final_ns=ns,tiles=[cycles],tiles_ns=[times]))
            seen.append((mi-1,ni-1))
        result['ctas'].append(dict(cta=cta,sm=head[2]-1,entry_c=head[0],entry_ns=head[1],
                                  prod_c=head[3],prod_ns=head[13],roles=roles))
    required={(i,j) for i in range(22) for j in range(12)}
    if len(seen)!=len(set(seen)) or set(seen)!=required or len({c['sm'] for c in result['ctas']})!=132:
        raise ValueError('historical work coverage/SM mapping changed')
    return result


def replay_historical_smoke(returned,variant):
    raw=returned/(variant+'.jsonl.gz')
    if variant not in LEGACY_SMOKE_SHA256 or sha(raw)!=LEGACY_SMOKE_SHA256[variant]:
        raise ValueError('historical smoke raw SHA mismatch; no fallback for other runs')
    with gzip.open(raw,'rt') as stream:
        events={e['event']:e for line in stream if line.strip() for e in [json.loads(line)]}
    result=historical_smoke_events(events,variant)
    result['raw_sha256']=LEGACY_SMOKE_SHA256[variant]
    return result


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
        observed=replay_historical_smoke(returned,variant)
        item=dict(history=history,gpu_uuid=setup['gpu_uuid'],numerical_values_replayed=observed['checked_values'],
                  ctas_replayed=len(observed['ctas']),formal_timing=False,
                  current_prefix_protocol_qualified=False,historical_reader=observed['historical_reader'])
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

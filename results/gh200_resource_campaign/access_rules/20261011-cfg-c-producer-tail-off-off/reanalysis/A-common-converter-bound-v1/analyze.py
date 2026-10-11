#!/usr/bin/env python3
"""Exact conditional limits of one conversion multiplier per recorded call pair."""
import argparse
from fractions import Fraction as Q
import gzip,hashlib,json
from pathlib import Path
import statistics as st


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def summary(values):
    return dict(n=len(values),minimum=min(values),median=st.median(values),maximum=max(values))


def oracle(rows):
    # q_i=N0_i*C1_i/C0_i, N1_i is the target. q_i and N1_i are
    # explicitly target observations; alpha is an oracle, not a state input.
    inputs=[(r['n0']*r['c1'],r['c0'],r['n1']) for r in rows]
    numerator,denominator=0,1;witness=(0,0)
    for i,(a,b,n) in enumerate(inputs):
        for j,(c,d,m) in enumerate(inputs[:i]):
            nn=abs(n*c*b-m*a*d);dd=a*d+c*b
            if nn*denominator>numerator*dd:
                numerator,denominator,witness=nn,dd,(i,j)
    radius=Q(numerator,denominator)
    q=[Q(a,b) for a,b,n in inputs];targets=[n for a,b,n in inputs]
    lower=max((n-radius)/x for x,n in zip(q,targets))
    upper=min((n+radius)/x for x,n in zip(q,targets))
    assert 0<lower<=upper
    alpha=(lower+upper)/2
    residual=[alpha*x-n for x,n in zip(q,targets)]
    assert max(map(abs,residual))==radius
    z=[x/n for x,n in zip(q,targets)];lo,hi=min(z),max(z)
    relative_alpha=2/(lo+hi);relative_floor=(hi-lo)/(hi+lo)
    assert max(abs(relative_alpha*x-1) for x in z)==relative_floor
    return dict(n=len(rows),absolute=dict(radius_ns=str(radius),alpha=str(alpha),
        alpha_lower=str(lower),alpha_upper=str(upper),witness_indices=list(witness)),
        relative=dict(floor=str(relative_floor),alpha=str(relative_alpha),
            minimum_z_index=z.index(lo),maximum_z_index=z.index(hi)),
        absolute_floor_ns=float(radius),relative_floor=float(relative_floor))


def decode(trace,cta):
    h=cta*1296;head=trace[h:h+16]
    assert 1<=head[2]<=132 and head[11]==head[12]==head[2] and head[10]==0
    assert head[6]==head[9] and head[6] in (1,2)
    assert head[0]>0 and head[1]>0 and head[3]>0
    result=[]
    for ordinal in range(head[6]):
        slots=[trace[h+16+wg*640+ordinal*10:h+16+wg*640+ordinal*10+10] for wg in (0,1)]
        assert slots[0][4:6]==slots[1][4:6]
        for slot in slots:
            assert all(x>0 for x in slot) and slot[1]>slot[0] and slot[7]>slot[6]
        # The clock-selected interval uses the same labeled endpoints in ns.
        first=min(range(2),key=lambda wg:slots[wg][0])
        last=max(range(2),key=lambda wg:slots[wg][1])
        c=slots[last][1]-slots[first][0]
        n=slots[last][7]-slots[first][6]
        ns_span=max(s[7] for s in slots)-min(s[6] for s in slots)
        assert c>0 and n>0 and ns_span>=n
        result.append(dict(ordinal=ordinal,work=[v-1 for v in slots[0][4:6]],sm=head[2]-1,
            slots=slots,c=c,n=n,clock_first_wg=first,clock_last_wg=last,
            ns_envelope_extra_ns=ns_span-n,
            ns_first_tie=slots[0][6]==slots[1][6],ns_last_tie=slots[0][7]==slots[1][7]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve();out=args.output.resolve();out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise ValueError('new output directory required')
    provenance={};windows=[];results=[];endpoint_checks=[];events=[];crosscheck=0
    for run_name in ('20261011-cfg-c-producer-tail-control','20261011-cfg-c-producer-tail-off-off'):
        root=base/run_name;data=root/'returned';manifest=read(root/'manifest.json')
        assert manifest['status']=='frozen' and manifest['gpu_uuid']=='GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'
        for name,digest in read(root/'freeze.json')['files'].items():
            assert sha(root/name)==sha(data/name)==digest
            provenance[str(root/name)]=digest;provenance[str(data/name)]=digest
        receipt=read(root/'receipt.json')
        assert receipt['returncode']==0 and sha(root/'output.tar.gz')==receipt['sha256']
        for path in (root/'freeze.json',root/'receipt.json',root/'output.tar.gz',data/'records.json',root/'summary.json'):
            provenance[str(path)]=sha(path)
        assert sha(data/'build/cfg_c_s4_dual')==manifest['binary_sha256']
        provenance[str(data/'build/cfg_c_s4_dual')]=sha(data/'build/cfg_c_s4_dual')
        records=read(data/'records.json');assert len(records)==20
        packets={};prefix={};ordered=[]
        for sample,record in zip(manifest['samples'],records):
            assert {key:record[key] for key in sample}==sample and record['returncode']==0
            raw=data/record['raw'];assert sha(raw)==record['sha256'];provenance[str(raw)]=sha(raw)
            with gzip.open(raw,'rt') as stream:items=[json.loads(line) for line in stream if line.strip()]
            assert [i['event'] for i in items]==['setup','call','check']
            setup,call,check=items
            assert (setup['config'],setup['m'],setup['n'],setup['k'],setup['grid'],setup['tile'],setup['cluster'],setup['stages'])==(
                'cfg_c',2304,3072,4096,[66,2,1],[256,128,64],[1,2],4)
            assert setup['gpu_uuid']==manifest['gpu_uuid'] and setup['trace_version']=='cfg-c-producer-tail-v2'
            assert setup['trace_words']==1296 and setup['trace_tile_words']==10
            assert setup['producer_tail_observer']==call['producer_tail_observer']==record['flag']
            assert call['producer_tail_prefix_flag']==0 and call['warmup_calls']==16 and len(call['warmup_us'])==16
            assert call['warmup_swizzle']==call['target_swizzle']==1 and call['source_copies']==1
            assert call['target_a_ptr']==call['prefix_a_ptr'] and call['target_b_ptr']==call['prefix_b_ptr']
            assert check['status']=='ok' and check['samples']==4096 and check['nonfinite']==check['padding_errors']==0
            arm=record.get('arm',record['flag']);key=record['pair'],arm
            assert key not in packets
            trace=call['trace'];assert len(trace)==132*1296
            packets[key]=[decode(trace,cta) for cta in range(132)]
            prefix[key]=Q(str(call['warmup_us'][-1]))
            events.append(dict(run=run_name,pair=record['pair'],arm=arm,flag=record['flag'],
                last_prefix_event_us=str(prefix[key]),event_us=call['elapsed_us']))
            ordered.append(key)
        for pair in range(10):
            assert [a for p,a in ordered if p==pair]==([0,1] if pair%2==0 else [1,0])
            scopes={'WG':[],'clock_selected_common':[]}
            for cta,(left,right) in enumerate(zip(packets[pair,0],packets[pair,1])):
                assert len(left)==len(right)
                for a,b in zip(left,right):
                    assert a['work']==b['work'] and a['sm']==b['sm'] and a['ordinal']==b['ordinal']
                    metadata=dict(run=run_name,pair=pair,cta=cta,ordinal=a['ordinal'],work=a['work'],sm=a['sm'])
                    for wg in (0,1):
                        x,y=a['slots'][wg],b['slots'][wg]
                        scopes['WG'].append(dict(metadata,scope='WG',wg=wg,
                            c0=x[1]-x[0],c1=y[1]-y[0],n0=x[7]-x[6],n1=y[7]-y[6]))
                    scopes['clock_selected_common'].append(dict(metadata,scope='clock_selected_common',wg=None,
                        c0=a['c'],c1=b['c'],n0=a['n'],n1=b['n'],
                        clock_labels0=[a['clock_first_wg'],a['clock_last_wg']],clock_labels1=[b['clock_first_wg'],b['clock_last_wg']]))
                    for arm,window in ((0,a),(1,b)):
                        endpoint_checks.append(dict(metadata,arm=arm,span_cycles=window['c'],
                            ns_envelope_extra_ns=window['ns_envelope_extra_ns'],
                            ns_first_tie=window['ns_first_tie'],ns_last_tie=window['ns_last_tie']))
            for scope,rows in scopes.items():
                assert len(rows)==(432 if scope=='WG' else 216)
                bound=oracle(rows);offset=len(windows);windows.extend(rows)
                bound.update(run=run_name,pair=pair,scope=scope,row_offset=offset,
                    alpha_last_prefix=str(prefix[pair,1]/prefix[pair,0]))
                results.append(bound)
        if run_name.endswith('off-off'):
            old=data.parent/'reanalysis/main-cta-compute-envelope-v1/windows.json.gz'
            provenance[str(old)]=sha(old)
            with gzip.open(old,'rt') as stream:reference={(r['pair'],r['arm'],r['cta'],r['ordinal']):r['span_cycles'] for r in json.load(stream)}
            for r in endpoint_checks:
                if r['run']==run_name:
                    assert r['span_cycles']==reference[r['pair'],r['arm'],r['cta'],r['ordinal']];crosscheck+=1
    assert len(windows)==12960 and len(endpoint_checks)==8640 and len(results)==40 and len(events)==40
    grouped=[]
    for run_name in sorted({r['run'] for r in results}):
        for scope in ('WG','clock_selected_common'):
            values=[r for r in results if r['run']==run_name and r['scope']==scope]
            grouped.append(dict(run=run_name,scope=scope,absolute_floor_ns=summary([r['absolute_floor_ns'] for r in values]),
                relative_floor=summary([r['relative_floor'] for r in values]),
                absolute_floor_above_256ns=sum(Q(r['absolute']['radius_ns'])>256 for r in values)))
    result=dict(schema='gh200-r09-common-converter-bound/1',input_sha256=provenance,analysis_sha256=sha(Path(__file__)),
        predictor='N1_i = alpha_pair * (C1_i/C0_i) * N0_i, one arbitrary positive alpha per pair',
        verification=dict(raw_calls=40,matched_WG_windows=8640,matched_common_windows=4320,
            endpoint_audits=8640,main_native_common_span_exact_matches=crosscheck,oracle_problems=40),
        summaries=grouped,bounds=results,events=events,
        endpoint_selection=dict(nonzero_extra_ns=sum(r['ns_envelope_extra_ns']>0 for r in endpoint_checks),
            extra_ns=summary([r['ns_envelope_extra_ns'] for r in endpoint_checks])),
        observation_qualified=False,prediction_qualified=False,
        limits=['Both target cycle windows and arm0 target ns are read. Alpha is optimized with all target windows: conditional oracle only.',
            'Absolute and relative minimax choose different alphas; their summaries must not be combined as one predictor.',
            '256ns is only a diagnostic scale here, not a corrected observer qualification or a prediction error gate.',
            'Small relative oracle floors do not prove a prior-state predictor achieves 5%/10%, nor rule out such an approximation.',
            'Clock-selected common interval retains its own start/end WG labels in ns; independent clock-domain extrema are not silently divided.',
            'C/N ratios include endpoint effects. No physical frequency, arbitrary common F(t), or endpoint error bound is identified.',
            'On/off and off/off remain separate; no background subtraction, re-pairing or previous failure replacement.',
            'Original sample check markers are verified, numerical references are not recomputed in this analysis.',
            'No B source/price, public model/runner, GPU allocation, or independent holdout changed.'])
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with gzip.open(out/'windows.json.gz','wt') as stream:json.dump(windows,stream,allow_nan=False)
    with gzip.open(out/'endpoint-selections.json.gz','wt') as stream:json.dump(endpoint_checks,stream,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('verification','summaries','endpoint_selection')},indent=2))


if __name__=='__main__':main()

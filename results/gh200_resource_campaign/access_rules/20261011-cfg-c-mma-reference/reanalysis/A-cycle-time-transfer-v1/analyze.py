#!/usr/bin/env python3
"""Separate native-work and time conversion transfer from the new same-card reference."""
import argparse
from fractions import Fraction as Q
import gzip,hashlib,json,re
from pathlib import Path
import statistics as st


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def stats(values):
    return dict(n=len(values),median=float(st.median(values)),median_absolute=float(st.median(abs(v) for v in values)),
        maximum_absolute=float(max(abs(v) for v in values)),minimum=float(min(values)),maximum=float(max(values)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve();out=args.output.resolve();out.mkdir(exist_ok=True)
    if (out/'transfer.json').exists():raise ValueError('new output required')
    root=base/'20261011-cfg-c-mma-reference';data=root/'returned'
    manifest=read(root/'manifest.json');build=Path(manifest['build_run']);provenance={}
    for name,digest in read(root/'freeze.json')['files'].items():
        assert sha(root/name)==sha(data/name)==digest
        provenance[str(root/name)]=digest;provenance[str(data/name)]=digest
    receipt=read(root/'receipt.json')
    assert receipt['returncode']==0 and sha(root/'output.tar.gz')==receipt['sha256']
    for path in (root/'freeze.json',root/'receipt.json',root/'output.tar.gz',root/'verification.json',data/'records.json',data/'complete.json'):
        provenance[str(path)]=sha(path)
    assert read(data/'complete.json')==dict(records=6,manifest_sha256=sha(root/'manifest.json'))
    verified=read(root/'verification.json');assert verified['formal_output_values_checked']==196608
    source=build/'source/cfg_c_mma_reference.cu';sass=build/'returned/build/reference.sass'
    assert sha(source)==read(root/'origin-source_hashes.json')['source/cfg_c_mma_reference.cu']
    assert sha(sass)==read(root/'origin-build-audit.json')['sass_sha256']
    provenance[str(source)]=sha(source);provenance[str(sass)]=sha(sass)
    sections=re.split(r'Function\s*:\s*',sass.read_text())
    formal=[s for s in sections[1:] if 'ELi4ELb1ELb0EE' in s.splitlines()[0]]
    assert len(formal)==1
    reads=re.findall(r'/\*([0-9a-f]+)\*/[^\n]*CS2R[^\n]*SR_(GLOBALTIMERLO|CLOCKLO)',formal[0])
    assert reads==[('14d0','GLOBALTIMERLO'),('1510','CLOCKLO'),('1fd0','CLOCKLO'),('1fe0','GLOBALTIMERLO')]
    records=read(data/'records.json');assert len(records)==6
    references=[]
    for sample,record in zip(manifest['samples'],records):
        assert {k:record[k] for k in sample}==sample and record['returncode']==0
        for name,digest in record['files'].items():
            assert sha(data/name)==digest;provenance[str(data/name)]=digest
        rec=read(data/'samples'/record['id']/'stdout.json')
        assert rec['gpu_uuid']==manifest['gpu_uuid']=='GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'
        assert (rec['groups'],rec['chains'],rec['per_chain_batch'],rec['wait'],rec['n'],rec['k'],rec['source'],rec['b_major'])==(2,2,4,1,128,16,'SS','MN')
        assert rec['warmup_calls']==16 and rec['warmup_policy']=='fixed16_same_probe'
        ns0,ns1,c0,c1,sm=rec['stamps'][0]
        assert len(rec['stamps'])==1 and ns1>ns0>0 and c1>c0>0 and sm==124
        assert rec['elapsed']==c1-c0 and rec['work_flop']==2*64*128*16*16*record['iterations']
        references.append(dict(id=record['id'],iterations=record['iterations'],cycles=c1-c0,ns=ns1-ns0,
            sm=sm,work_flop=rec['work_flop'],observed_cycle_per_ns=str(Q(c1-c0,ns1-ns0))))
    sustained=[r for r in references if r['iterations']==1024];assert len(sustained)==3
    cycle=Q(st.median(r['cycles'] for r in sustained));duration=Q(st.median(r['ns'] for r in sustained))
    assert len({r['cycles'] for r in sustained})==1
    assert st.median(Q(r['cycles'],r['ns']) for r in sustained)==cycle/duration
    target_work=2*256*128*4096;reference_work=sustained[0]['work_flop'];ratio=Q(target_work,reference_work)
    assert ratio==Q(1,16)
    predicted_cycle=cycle*ratio;reference_rate=cycle/duration
    predicted_ns=predicted_cycle/reference_rate
    # The candidate is formed entirely before reading GEMM target windows.
    paired_file=base/'20261011-cfg-c-producer-tail-off-off/reanalysis/A-common-converter-bound-v1/windows.json.gz'
    paired_report=paired_file.parent/'report.json'
    for path in (paired_file,paired_report):provenance[str(path)]=sha(path)
    previous=read(paired_report)
    for name,digest in previous['input_sha256'].items():
        assert sha(Path(name))==digest;provenance[name]=digest
    with gzip.open(paired_file,'rt') as stream:paired=json.load(stream)
    rows=[]
    for window in paired:
        if window['scope']!='clock_selected_common':continue
        for arm in (0,1):
            c,n=window['c'+str(arm)],window['n'+str(arm)]
            cycle_error=predicted_cycle/c-1
            conversion_error=Q(c,n)/reference_rate-1
            time_error=predicted_ns/n-1
            assert (1+cycle_error)*(1+conversion_error)==1+time_error
            row={k:window[k] for k in ('run','pair','cta','ordinal','sm','work')}
            row.update(arm=arm,cycle=c,ns=n,cycle_error=str(cycle_error),
                conditional_conversion_error=str(conversion_error),composed_window_error=str(time_error))
            rows.append(row)
    assert len(rows)==8640 and sum(r['sm']==124 for r in rows)==80
    groups=[];processes=[]
    for run in sorted({r['run'] for r in rows}):
        for selection in ('all_SM','SM124'):
            for ordinal in (0,1):
                values=[r for r in rows if r['run']==run and r['ordinal']==ordinal and (selection=='all_SM' or r['sm']==124)]
                metrics={field:stats([Q(r[field]) for r in values]) for field in ('cycle_error','conditional_conversion_error','composed_window_error')}
                groups.append(dict(run=run,selection=selection,ordinal=ordinal,metrics=metrics))
                for pair in range(10):
                    for arm in (0,1):
                        selected=[r for r in values if r['pair']==pair and r['arm']==arm]
                        processes.append(dict(run=run,selection=selection,ordinal=ordinal,pair=pair,arm=arm,
                            metrics={field:stats([Q(r[field]) for r in selected]) for field in metrics}))
    result=dict(schema='gh200-r09-same-card-cycle-time-transfer/1',input_sha256=provenance,
        analysis_sha256=sha(Path(__file__)),references=references,
        candidate=dict(selection='Prescribed sustained 1024-group reference; median of its three fixed repeats, no target-based fitting.',
            target_work_flop=target_work,reference_work_flop=reference_work,work_ratio=str(ratio),
            reference_cycles=str(cycle),reference_ns=str(duration),reference_cycle_per_ns=str(reference_rate),
            predicted_common_cycle=str(predicted_cycle),predicted_common_ns=str(predicted_ns),target_observations_used=False),
        timer_sites=dict(formal_kernel=formal[0].splitlines()[0],sass_reads=[dict(pc=pc,register=reg) for pc,reg in reads],
            interpretation='Reference start is GT->CLOCK, stop CLOCK->GT. GEMM FIRST/MAIN uses CLOCK->GT. These are distinct nominal endpoints; hardware enclosure and read-cost correction are unproved.'),
        verification=dict(reference_calls=6,target_windows=8640,same_reference_SM_target_windows=80,
            exact_multiplicative_closures=8640,original_formal_reference_values_already_verified=196608),
        groups=groups,processes=processes,observation_qualified=False,prediction_qualified=False,
        limits=['Reference was acquired after the known GEMM data; this is retrospective development transfer, not a new frozen holdout or online availability proof.',
            'Native cycle price includes ARRIVE, checksum/drain and sync; 803-cycle excess is not fitted or subtracted.',
            'Resident repeated SMEM/uniform one-CTA reference differs from real TMA/stage/cluster/dyadic 132-CTA GEMM and its history.',
            'SM124 is reported separately. Other SM observations are descriptive, not validation of universal SM price transfer.',
            'Conditional conversion holds target native cycles fixed; observed C/N includes endpoint effects and is not a physical frequency measurement.',
            'Per-window multiplicative identity is exact. Separately summarized medians do not multiply, and common windows do not close the full CUDA event.',
            'No B source work/price, public model/runner, GPU allocation, observation gate, accepted rule, or cross-card boundary changed.'])
    (out/'transfer.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with gzip.open(out/'window-errors.json.gz','wt') as stream:json.dump(rows,stream,allow_nan=False)
    print(json.dumps(dict(candidate=result['candidate'],verification=result['verification'],groups=groups),indent=2))


if __name__=='__main__':main()

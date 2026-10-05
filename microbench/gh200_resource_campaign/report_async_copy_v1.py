#!/usr/bin/env python3
"""Read-only S12 raw-work report; original bounded replay must already pass."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def stats(values):
    if not values:return {'samples':0,'mean':None,'median':None,'minimum':None,'maximum':None,'cv':None,'cv_squared_exact':None}
    exact=[Fraction(value) for value in values];mean=sum(exact)/len(exact)
    variance=sum((value-mean)**2 for value in exact)/(len(exact)-1) if len(exact)>1 else None
    cv2=variance/(mean*mean) if variance is not None else None
    return {'samples':len(values),'mean':float(mean),'median':float(statistics.median(values)),
            'minimum':float(min(values)),'maximum':float(max(values)),'cv':float(cv2)**.5 if cv2 is not None else None,
            'cv_squared_exact':str(cv2) if cv2 is not None else None}


def table(path,rows):
    require(rows,'empty report table')
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def render(run,replay_path,out):
    run,replay_path,out=run.resolve(),replay_path.resolve(),out.resolve()
    require(not out.exists() and not out.is_relative_to(run) and not run.is_relative_to(out),'new analysis directory outside run required')
    replay=read(replay_path)
    require(replay['status']=='pass_under_bounded_float_replay_v1' and not replay['failed_checks'],'passing bounded replay required')
    inventory={p.relative_to(run).as_posix():sha(p) for p in run.rglob('*') if p.is_file()}
    require(inventory==replay['input_artifacts_sha256'],'complete replay input identity changed')
    spec=read(run/'run_spec.json');summary=read(run/'summary.json')
    contract_path=(run/spec['contract_path']).resolve();require(contract_path.is_relative_to(run),'contract path escapes run')
    contract=read(contract_path)
    require(contract['adapter_id']=='async_copy_formal_v1' and summary['terminal'] and len(contract['cases'])==24,'terminal S12 formal archive required')
    cases={case['id']:case for case in contract['cases']}
    require({r['case_id'] for r in summary['cases']}==set(cases),'case coverage')
    trials=[];case_rows=[];batch_rows=[];examples=[]
    sample_paths=set();receipt_paths=set();pid_intervals=[]
    resource_names=['registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm']
    for result in summary['cases']:
        case=cases[result['case_id']];p=case['parameters'];all_rows=[];accepted=[];identity=None
        cache=p['ptx'].split('.')[2];width=p['request_bytes_per_thread'];stages=p['stages']
        for batch in result['batches']:
            values=[]
            for sample in batch['samples']:
                receipt_path=(run/sample['receipt']).resolve();raw_path=receipt_path.parent/'raw.jsonl'
                require(receipt_path.is_relative_to(run) and raw_path.is_relative_to(run),'sample path escapes run')
                require(str(raw_path) not in sample_paths and str(receipt_path) not in receipt_paths,'duplicate sample path')
                sample_paths.add(str(raw_path));receipt_paths.add(str(receipt_path))
                require(sha(raw_path)==sample['raw_sha256'],'raw hash changed')
                receipt=read(receipt_path);raws=[json.loads(line) for line in raw_path.read_text().splitlines()]
                require(len(raws)==2 and raws[1]['type']=='trial','raw device/trial shape');raw=raws[1]
                require(raw['case_id']==case['id'] and raw['iterations']==8192 and raw['threads']==128,'fixed case dimensions')
                require(raw['request_bytes_per_thread']==width and raw['stages']==stages,'request coordinates')
                require(raw['seed']==3+19*sample['trial']+1009*sample['batch'],'seed schedule')
                q=raw['blocks']*128*raw['iterations']*8*width
                require(all(raw[k]==q for k in ['work_count','read_payload_bytes','write_payload_bytes','consumer_read_bytes']),'Q transport/read/write/consumer mismatch')
                checked=raw['blocks']*128*(1+stages*(width//4))
                require(raw['errors']==0 and raw['correctness']['checked_elements']==checked,'long full check count')
                blocks=raw['blocks_detail'];require(len(blocks)==raw['blocks'],'CTA detail count')
                if case['scope']=='one_cta':
                    require(raw['blocks']==1,'CTA metric requires oneCTA')
                    start=blocks[0]['start_cycle'];stop=blocks[0]['stop_cycle'];denom=stop-start
                    unit='B_transport/clock64_cycle/CTA';domain='CTA_clock64'
                else:
                    start=min(b['start_ns'] for b in blocks);stop=max(b['stop_ns'] for b in blocks);denom=stop-start
                    unit='GB/s_transport/GPU';domain='GPU_globaltimer_ns'
                    require(len({b['smid'] for b in blocks})==raws[0]['sms'],'all SM coverage')
                require(denom>0 and unit==sample['unit'],'metric denominator/unit')
                rate=Fraction(q,denom);require(float(rate)==sample['value'],'raw rate and sample differ')
                resource=tuple(raw[k] for k in resource_names)+(raw['blocks'],raw['global_array_bytes'])
                if identity is None:identity=resource
                require(resource==identity,'within-case resource drift')
                row={'case_id':case['id'],'cache_modifier':cache,'request_bytes_per_thread':width,'stages':stages,
                     'scope':case['scope'],'process_batch':sample['batch'],'trial':sample['trial'],'seed':raw['seed'],
                     'iterations':raw['iterations'],'threads':128,'blocks':raw['blocks'],
                     'Q_transport_bytes':q,'GMEM_read_payload_bytes':q,'SMEM_write_payload_bytes':q,'consumer_SMEM_read_bytes':q,
                     'clock_domain':domain,'start':start,'stop':stop,'denominator':denom,'value':float(rate),'exact_rate_fraction':str(rate),'unit':unit,
                     'event_ms':raw['event_ms'],'observed_sms':sample['observed_sms'],
                     'warmup_converged':sample['warmup_converged'],'warmup_windows':len(raw['warmup_samples_ns']),
                     'warmup_cv':sample['warmup_cv'],'checked_elements':checked,
                     'pid':receipt['pid'],'host_start_ns':receipt['host_start_ns'],'host_stop_ns':receipt['host_stop_ns'],
                     **dict(zip(resource_names,resource[:5])),'global_array_bytes':raw['global_array_bytes'],
                     'case_status':result['status'],'receipt':receipt_path.relative_to(run).as_posix(),
                     'raw':raw_path.relative_to(run).as_posix(),'raw_sha256':sha(raw_path)}
                trials.append(row);all_rows.append(row);pid_intervals.append((receipt['host_start_ns'],receipt['host_stop_ns'],receipt['pid']))
                if sample['warmup_converged']:values.append(rate);accepted.append(rate)
            bs=stats(values);merged=stats(accepted)
            batch_rows.append({'case_id':case['id'],'scope':case['scope'],'batch':batch['batch'],
                'recorded_processes':len(batch['samples']),'complete':batch['complete'],'warmup_failed':batch['warmup_failed'],
                **bs,'cumulative_samples':merged['samples'],'cumulative_cv':merged['cv'],'cumulative_cv_squared_exact':merged['cv_squared_exact']})
        merged=stats(accepted)
        require(result['merged'] is None if len(accepted)<2 else result['merged']['samples']==len(accepted),'merged count')
        if result['status']=='stable':
            require(batch_rows[-1]['complete'] and Fraction(batch_rows[-1]['cv_squared_exact'])<=Fraction(1,400) and Fraction(merged['cv_squared_exact'])<=Fraction(1,400),'stable threshold exact recomputation')
        require(identity is not None,'missing recorded case process')
        case_rows.append({'case_id':case['id'],'cache_modifier':cache,'request_bytes_per_thread':width,'stages':stages,
            'scope':case['scope'],'status':result['status'],'exportable_after_C':result['exportable'],
            'processes':len(all_rows),'batches':len(result['batches']),'warmup_failed_processes':sum(not row['warmup_converged'] for row in all_rows),
            **merged,'unit':case['metric']['unit'],**dict(zip(resource_names,identity[:5])),
            'blocks':identity[5],'global_array_bytes':identity[6]})
        if case['id'] in ['ca_w4_stages1_one_cta','ca_w16_stages4_one_cta','ca_w16_stages4_all_gpu','cg_w16_stages4_all_gpu']:
            examples.append(next((r for r in all_rows if r['warmup_converged']),all_rows[0]))
    pid_intervals.sort()
    require(all(a[1]<=b[0] for a,b in zip(pid_intervals,pid_intervals[1:])),'sample processes overlap')
    known_receipts={str(p.resolve()) for p in (run/'batches').glob('*/batch_*/trial_*/attempt_*/receipt.json')}
    interrupted=[]
    for path in sorted(known_receipts-receipt_paths):
        row=read(Path(path));require(row['status']=='interrupted','unreported accepted or failed process')
        interrupted.append({'receipt':Path(path).relative_to(run).as_posix(),'status':row['status']})
    comparisons=[];by={(r['scope'],r['cache_modifier'],r['request_bytes_per_thread'],r['stages']):r for r in case_rows}
    for scope in ['one_cta','all_gpu']:
        for cache,width in [('ca',4),('ca',8),('ca',16),('cg',16)]:
            for a,b in [(1,2),(1,4),(2,4)]:
                left,right=by[(scope,cache,width,a)],by[(scope,cache,width,b)]
                valid=left['status']==right['status']=='stable'
                comparisons.append({'scope':scope,'comparison':'stage','baseline':left['case_id'],'variant':right['case_id'],
                    'both_stable':valid,'median_rate_ratio':float(right['median']/left['median']) if valid else None,'C_required':True})
        for stage in [1,2,4]:
            left,right=by[(scope,'ca',16,stage)],by[(scope,'cg',16,stage)];valid=left['status']==right['status']=='stable'
            comparisons.append({'scope':scope,'comparison':'cache_modifier_same16B','baseline':left['case_id'],'variant':right['case_id'],
                'both_stable':valid,'median_rate_ratio':float(right['median']/left['median']) if valid else None,'C_required':True})
    out.mkdir(parents=True)
    table(out/'trials.csv',trials);table(out/'cases.csv',case_rows);table(out/'batches.csv',batch_rows);table(out/'comparisons.csv',comparisons)
    (out/'manual-examples.json').write_text(json.dumps(examples,indent=2,sort_keys=True)+'\n')
    text=['# S12 原始字段手算','运输Q只计一次；下列点按固定case选择首个预热通过的记录，未挑选最快样本。']
    for row in examples:
        text+=['',f"## {row['case_id']}",f"原始路径：`{row['raw']}`，SHA256 `{row['raw_sha256']}`。",
               f"Q={row['blocks']}×128×8192×8×{row['request_bytes_per_thread']}={row['Q_transport_bytes']} B。",
               f"{row['clock_domain']}：{row['stop']}−{row['start']}={row['denominator']}。",
               f"Q/分母={row['exact_rate_fraction']}={row['value']:.12g} {row['unit']}。",
               f"GMEM读、SMEM写、consumer SMEM读各{row['Q_transport_bytes']} B，不能相加作为运输量；checked_elements={row['checked_elements']}。"]
    (out/'manual-examples.md').write_text('\n\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='gh200-async-copy-v1'
    import matplotlib.pyplot as plt
    figures=[]
    for scope in ['one_cta','all_gpu']:
        fig,axis=plt.subplots(figsize=(9,5),layout='constrained')
        for color,(cache,width) in zip(['#0072B2','#D55E00','#009E73','#CC79A7'],[('ca',4),('ca',8),('ca',16),('cg',16)]):
            rows=[by[(scope,cache,width,stage)] for stage in [1,2,4]]
            good=[r for r in rows if r['samples']]
            axis.plot([r['stages'] for r in good],[float(r['median']) for r in good],'o-',color=color,label=f'{cache}, {width} B/thread')
            for r in good:
                data=[v for v in trials if v['case_id']==r['case_id'] and v['warmup_converged']]
                axis.scatter([r['stages']+(i-(len(data)-1)/2)*.004 for i in range(len(data))],[v['value'] for v in data],s=12,color=color,alpha=.35)
                axis.errorbar(r['stages'],float(r['median']),yerr=[[float(r['median']-r['minimum'])],[float(r['maximum']-r['median'])]],color=color,capsize=4)
                if r['status']!='stable':axis.annotate('unstable',(r['stages'],float(r['median'])),fontsize=8)
        axis.set(xticks=[1,2,4],xlabel='Pipeline stages',ylabel=case_rows[0]['unit'] if scope=='one_cta' else 'GB/s_transport/GPU',title=f'GH200 S12: {scope}\nAll accepted processes; median and min/max, not confidence intervals')
        if scope=='one_cta':axis.set_ylabel('B_transport/clock64_cycle/CTA')
        axis.grid(alpha=.2);axis.legend();name='stages-'+scope
        fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'),metadata={'Date':None});plt.close(fig);figures.append(name+'.png')
    fig,axis=plt.subplots(figsize=(10,7),layout='constrained')
    rows=sorted(case_rows,key=lambda r:(r['scope'],r['cache_modifier'],r['request_bytes_per_thread'],r['stages']))
    axis.barh(range(len(rows)),[100*r['cv'] if r['cv'] is not None else 0 for r in rows],color=['#0072B2' if r['status']=='stable' else '#D55E00' for r in rows])
    axis.set_yticks(range(len(rows)),[r['case_id']+f" [n={r['samples']}, b={r['batches']}]" for r in rows]);axis.invert_yaxis();axis.axvline(5,color='black',linestyle='--',label='5% limit')
    axis.set(xlabel='CV of all accepted process rates (%)',title='S12 bounded batch outcomes; rejected warmup processes excluded from rate CV');axis.legend()
    fig.savefig(out/'process-cv.png',dpi=170);fig.savefig(out/'process-cv.svg',metadata={'Date':None});plt.close(fig);figures.append('process-cv.png')
    shutil.copyfile(__file__,out/'report_async_copy_v1.py')
    counter=read(run/'environment/ncu_status.json')
    totals={'cases':len(case_rows),'recorded_processes':len(trials),'accepted_processes':sum(r['warmup_converged'] for r in trials),
        'stable_cases':sum(r['status']=='stable' for r in case_rows),'batches':len(batch_rows),'interrupted_attempts':interrupted,'interrupted_attempt_count':len(interrupted),'all_recorded_attempts':len(known_receipts),
        'counter_state':counter['state'],'counter_family_profile':counter['family_profile'],'physical_hbm_bytes_proven':False,
        'C_review_required':True,'sample_process_intervals_nonoverlapping':True,'matplotlib_version':matplotlib.__version__}
    (out/'totals.json').write_text(json.dumps(totals,indent=2,sort_keys=True)+'\n')
    (out/'index.md').write_text('# S12 异步复制分析\n\n待独立C。运输Q只计一次，读/写/consumer分别列出；各scope独立单位。所有有效进程与有界批次都在CSV，散点/范围不是置信区间。8MiB共享只读source的缓存状态未知，不能称物理HBM带宽。\n\n'+'\n\n'.join(f'![{n}]({n})' for n in figures)+'\n')
    after={p.relative_to(run).as_posix():sha(p) for p in run.rglob('*') if p.is_file()};require(after==inventory,'run changed during reporting')
    outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    manifest={'schema_version':1,'kind':'S12_raw_transport_report','input_run_inventory':inventory,'replay_sha256':sha(replay_path),'output_files':outputs,'matplotlib_version':matplotlib.__version__,'C_review_required':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(totals,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_dir',type=Path);parser.add_argument('--replay',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.run_dir,args.replay,args.output)

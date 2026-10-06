"""Candidate S20 cycle rates and K-tile fits from actual native replay; C required."""
from pathlib import Path
import argparse,csv,hashlib,json,math,statistics,sys,tarfile
if sys.flags.optimize:raise RuntimeError('S20 reports require active assertions')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def fit_sequence(points,holdout):
    assert set(points)=={1,2,4,8,16,32}
    assert all(type(y) in (float,int) and math.isfinite(y) and y>0 for y in points.values())
    xs=sorted(points);mx=statistics.mean(xs);my=statistics.mean(points.values())
    slope=sum((x-mx)*(points[x]-my) for x in xs)/sum((x-mx)**2 for x in xs);intercept=my-slope*mx
    residual={str(x):{'observed':points[x],'predicted':intercept+slope*x,'relative_error':(intercept+slope*x-points[x])/points[x]} for x in xs}
    result={'model':'clock64_cycles_per_sequence = intercept + slope * Ktiles','fit_Ktiles':xs,
        'intercept_cycles':intercept,'slope_cycles_per_Ktile':slope,'fit_points':residual,
        'max_fit_relative_error':max(abs(v['relative_error']) for v in residual.values()),'physical_startup_interpretation_proven':False,'qualified':False}
    if holdout is not None:
        assert type(holdout) in (float,int) and math.isfinite(holdout) and holdout>0
        predicted=intercept+64*slope;result['holdout64']={'observed':holdout,'predicted':predicted,'relative_error':(predicted-holdout)/holdout,'used_in_fit':False}
    else:result['holdout64']={'status':'missing_stable_statistics','used_in_fit':False}
    result['service_parameter_export_allowed']=False
    return result


def render(run,replay_file,output,tables_only=False):
    run,output=Path(run),Path(output);replay=json.loads(Path(replay_file).read_text())
    assert replay['CPU_fixture_only'] is False and replay['all_saved_values_recomputed'] is True and replay['qualified'] is False and replay['coordinate_count']==105
    summary=json.loads((run/'summary.json').read_text());assert summary['case_results']==replay['cases']
    cases=json.loads((run/'repo/cases.json').read_text())['cases'];by_id={c['id']:c for c in cases};device=json.loads((run/'device.json').read_text());resource=json.loads((run/'resources.json').read_text())['measured-capability']
    samples=[];examples={};rows={}
    for path in sorted(run.glob('point*-state.json')):
        state=json.loads(path.read_text());assert state['state']=='verified_node_pack_pending_offhost'
        if state['kind']!='formal':continue
        case=by_id[state['case_id']];archive=run/'packs'/('point%04d.tar.xz'%state['index']);assert sha(archive)==state['pack_identity']['archive_sha256']
        with tarfile.open(archive) as packet:body=packet.extractfile('raw.jsonl').read()
        assert hashlib.sha256(body).hexdigest()==state['pack_identity']['members']['raw.jsonl']['sha256']
        actual,row=[json.loads(line) for line in body.decode().splitlines()];assert actual==device
        p=case['parameters'];mode=p['mode'];K=p['k_tiles'];N=row['iterations'];d=row['blocks_detail'][0];cycles=d['stop_cycle']-d['start_cycle'];assert cycles>0
        fma=0 if mode=='transport' else 524288*K*N;epi=8192*N if mode=='output' else 0;read=0 if mode=='compute' else 16384*K*N;write=16384*N if mode=='output' else 0;work=read if mode=='transport' else fma+epi
        assert row['MMA_FLOP']==fma and row['epilogue_FLOP']==epi and row['read_payload_bytes']==read and row['write_payload_bytes']==write and row['work_count']==work
        sample={'case_id':case['id'],'batch':state['batch'],'trial':state['trial'],'seed':state['seed'],'mode':mode,'stages':p['stages'],'Ktiles':K,'iterations':N,'holdout':K==64,'cta_cycles':cycles,'cycles_per_sequence':cycles/N,'MMA_FLOP':fma,'epilogue_FLOP':epi,'input_bytes':read,'output_bytes':write,'work_count':work,'value':work/cycles,'unit':case['metric']['unit'],'warmup_converged':row['warmup_converged'],'raw_sha256':hashlib.sha256(body).hexdigest()}
        samples.append(sample);rows.setdefault(case['id'],row)
        if state['batch']==state['trial']==0 and p['stages']==2 and K==8:examples[mode]=sample
    assert len(samples)==replay['formal_process_count'];table=[];parameters=[];excluded=[]
    for case in cases:
        key=case['id'];p=case['parameters'];result=replay['cases'][key];merged=result['merged'];selected=[v for v in samples if v['case_id']==key and v['warmup_converged']];values=[v['value'] for v in selected];seq=[v['cycles_per_sequence'] for v in selected]
        if merged:assert len(values)==merged['samples'] and statistics.median(values)==merged['median']
        else:assert len(values)<2
        row={'case_id':key,'mode':p['mode'],'stages':p['stages'],'Ktiles':p['k_tiles'],'holdout':p['holdout'],'status':result['status'],'median':merged['median'] if merged else None,'min':merged['min'] if merged else None,'max':merged['max'] if merged else None,'cv':merged['cv'] if merged else None,'samples':len(values),'unit':case['metric']['unit'],'cycles_per_sequence':statistics.median(seq) if seq else None,'sequence_min':min(seq) if seq else None,'sequence_max':max(seq) if seq else None,'qualified':False};table.append(row)
        param={'case_id':key,'status':row['status'],'value':row['median'],'unit':row['unit'],'samples':row['samples'],'cv':row['cv'],'statistic':'median','cycles_per_sequence':row['cycles_per_sequence'],'conditions':{**p,'iterations':rows[key]['iterations'],'threads':128,'blocks':1,'trace_enabled':False,'resources':resource},'device_uuid':device['uuid'],'qualified':False,'qualification':'candidate_requires_independent_C','physical_HBM_bytes_proven':False}
        (parameters if row['status']=='stable' else excluded).append(param)
    fits=[]
    for mode in ('compute','transport','serial','overlap','output'):
        for S in (1,2,4):
            group=[v for v in table if v['mode']==mode and v['stages']==S];points={v['Ktiles']:v['cycles_per_sequence'] for v in group if not v['holdout'] and v['status']=='stable'};hold=next(v for v in group if v['holdout']);record={'mode':mode,'stages':S,'qualified':False}
            if len(points)==6:record.update(status='candidate_fit',fit=fit_sequence(points,hold['cycles_per_sequence'] if hold['status']=='stable' else None))
            else:record.update(status='insufficient_stable_fit_points',missing_Ktiles=sorted({1,2,4,8,16,32}-set(points)))
            fits.append(record)
    output.mkdir(exist_ok=False)
    for name,data in [('samples.csv',samples),('results.csv',table)]:
        with (output/name).open('w',newline='') as stream:writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    for name,data in [('parameter-candidates.json',{'parameters':parameters,'excluded':excluded,'qualified':False}),('sequence-fits.json',fits),('worked-examples.json',examples)]: (output/name).write_text(json.dumps(data,indent=2)+'\n')
    text=['# BF16 WGMMA 固定tile受控组合','','主值为本CTA FLOP/cycle，transport单列B/cycle。cycles/sequence使用完整K-tile序列的重复次数N。拟合只用K=1/2/4/8/16/32，K=64只作检查；系数不自动解释为物理启动成本，也不自动导出服务参数。','', '| 模式 | S | K | 状态 | 中位数 | 单位 | cycles/sequence | CV | 样本 |','|---|---:|---:|---|---:|---|---:|---:|---:|']
    for r in table:text.append(f"| {r['mode']} | {r['stages']} | {r['Ktiles']} | {r['status']} | {r['median']} | {r['unit']} | {r['cycles_per_sequence']} | {r['cv']} | {r['samples']} |")
    for mode,v in examples.items():text+=['',f"`{mode}_s2_k8`：MMA={v['MMA_FLOP']} FLOP，epilogue={v['epilogue_FLOP']} FLOP，输入={v['input_bytes']} B，输出={v['output_bytes']} B；主量 {v['work_count']} 除以 {v['cta_cycles']} 本CTA周期，得 {v['value']:.9g} {v['unit']}。周期除以N={v['iterations']}得 {v['cycles_per_sequence']:.9g} cycles/sequence。raw SHA `{v['raw_sha256']}`。"]
    (output/'RESULTS.zh.md').write_text('\n'.join(text)+'\n');(output/'sources.json').write_text(json.dumps({'renderer_sha256':sha(__file__),'native_replay_sha256':sha(replay_file),'files':{n:sha(run/n) for n in ('summary.json','case-results.json','device.json','resources.json','resolved-cases.json','source-manifest.json')},'formal_process_count':len(samples),'qualified':False},indent=2)+'\n')
    if not tables_only:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for mode in ('compute','transport','serial','overlap','output'):
            fig,ax=plt.subplots(figsize=(7,4.5))
            for S in (1,2,4):
                group=[v for v in table if v['mode']==mode and v['stages']==S and v['cycles_per_sequence'] is not None]
                ax.errorbar([v['Ktiles'] for v in group],[v['cycles_per_sequence'] for v in group],yerr=[[v['cycles_per_sequence']-v['sequence_min'] for v in group],[v['sequence_max']-v['cycles_per_sequence'] for v in group]],marker='o',capsize=3,label='stage '+str(S))
            ax.set_xlabel('K tiles (64 held out)');ax.set_ylabel('clock64 cycles / sequence / CTA');ax.set_title(mode+'; median, retained sample min/max; pending C');ax.legend();fig.tight_layout();fig.savefig(output/(mode+'.png'),dpi=160);plt.close(fig)
    (output/'report-manifest.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in output.iterdir() if p.is_file()},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--replay',required=True);parser.add_argument('--output',required=True);parser.add_argument('--tables-only',action='store_true');args=parser.parse_args();render(args.run,args.replay,args.output,args.tables_only)

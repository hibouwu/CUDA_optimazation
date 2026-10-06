"""S17 native replay to candidate tables, examples and plots; independent C required."""
from pathlib import Path
import argparse,csv,hashlib,json,statistics,sys,tarfile
if sys.flags.optimize:raise RuntimeError('S17 reports require active assertions')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def render(run,replay_file,output,tables_only=False):
    run,output=Path(run),Path(output);replay=json.loads(Path(replay_file).read_text())
    assert replay['CPU_fixture_only'] is False and replay['all_saved_values_recomputed'] is True
    assert replay['qualified'] is False and replay['coordinate_count']==42
    summary=json.loads((run/'summary.json').read_text());assert summary['case_results']==replay['cases']
    cases=json.loads((run/'repo/cases.json').read_text())['cases'];by_id={c['id']:c for c in cases}
    resources=json.loads((run/'resources.json').read_text());device=json.loads((run/'device.json').read_text())
    samples=[];rows={};examples={}
    for path in sorted(run.glob('point*-state.json')):
        state=json.loads(path.read_text());assert state['state']=='verified_node_pack_pending_offhost'
        if state['kind']!='formal':continue
        case=by_id[state['case_id']];archive=run/'packs'/('point%04d.tar.xz'%state['index'])
        assert sha(archive)==state['pack_identity']['archive_sha256']
        with tarfile.open(archive) as packet:body=packet.extractfile('raw.jsonl').read()
        assert hashlib.sha256(body).hexdigest()==state['pack_identity']['members']['raw.jsonl']['sha256']
        actual,row=[json.loads(line) for line in body.decode().splitlines()];assert actual==device
        p=case['parameters'];mode=p['mode'];C=p['cluster_size'];G=row['clusters'];B=row['blocks'];N=row['iterations']
        assert B==C*G
        amount=G*N if mode==4 else G*N*16384 if mode>=5 else B*N*4096
        unit='cluster_phase' if mode==4 else 'byte'
        assert row['work_count']==amount and row['work_unit']==unit
        denominator=row['stop_ns']-row['start_ns'];assert denominator>0
        source=G*N*16384 if mode>=5 else 0;receiver=source*(C if mode==6 else 1)
        readback=B*N*4096 if mode in (2,3) else 0
        assert row['source_request_bytes']==source and row['total_receiver_bytes']==receiver and row['required_write_readback_bytes']==readback
        value=amount/denominator
        sample={'case_id':case['id'],'batch':state['batch'],'trial':state['trial'],'seed':state['seed'],'form':p['form'],
            'cluster_size':C,'clusters':G,'blocks':B,'iterations':N,'work_count':amount,'work_unit':unit,'elapsed_ns':denominator,
            'value':value,'unit':case['metric']['unit'],'source_request_bytes':source,'total_receiver_bytes':receiver,
            'required_write_readback_bytes':readback,'warmup_converged':row['warmup_converged'],
            'post_timing_export_bytes':row['post_timing_export_bytes'],'raw_sha256':hashlib.sha256(body).hexdigest()}
        samples.append(sample);rows.setdefault(case['id'],row)
        if state['batch']==state['trial']==0 and C==2:examples.setdefault(p['form']+'_'+case['scope'],sample)
    assert len(samples)==replay['formal_process_count']
    table=[];parameters=[];excluded=[]
    for case in cases:
        key=case['id'];p=case['parameters'];res=resources[key];entry={'case_id':key,'scope':case['scope'],'form':p['form'],
            'cluster_size':p['cluster_size'],'status':'capability_reject','median':None,'min':None,'max':None,'cv':None,'samples':0,
            'unit':case['metric']['unit'],'iterations':None,'clusters':None,'blocks':None,'qualified':False}
        if not res['supported']:
            excluded.append({'case_id':key,'reason':res['unsupported_reason'],'target_launches':0,'capability':res});table.append(entry);continue
        result=replay['cases'][key];merged=result['merged'];row=rows[key]
        values=[v['value'] for v in samples if v['case_id']==key and v['warmup_converged']]
        if merged:assert len(values)==merged['samples'] and statistics.median(values)==merged['median']
        else:assert len(values)<2
        entry.update(status=result['status'],median=merged['median'] if merged else None,min=merged['min'] if merged else None,
            max=merged['max'] if merged else None,cv=merged['cv'] if merged else None,samples=len(values),iterations=row['iterations'],clusters=row['clusters'],blocks=row['blocks'])
        table.append(entry)
        param={'case_id':key,'status':result['status'],'value':entry['median'],'unit':entry['unit'],'samples':len(values),'cv':entry['cv'],
            'statistic':'median','conditions':{**p,'scope':case['scope'],'threads':128,'iterations':row['iterations'],
                'clusters':row['clusters'],'blocks':row['blocks'],'capture':False,'source_slots_per_cluster':32 if p['mode']>=5 else 0,
                'required_write_readback_in_primary_window':p['mode'] in (2,3),'resources':res},
            'device_uuid':device['uuid'],'physical_HBM_bytes_proven':False,'actual_residency_proven':False,
            'qualified':False,'qualification':'candidate_requires_independent_C'}
        (parameters if result['status']=='stable' else excluded).append(param)
    assert len(table)==42 and len(parameters)+len(excluded)==42
    output.mkdir(exist_ok=False)
    for name,data in [('samples.csv',samples),('results.csv',table)]:
        with (output/name).open('w',newline='') as stream:
            fields=list(data[0]) if data else ['case_id','batch','trial','value','unit']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data)
    (output/'parameter-candidates.json').write_text(json.dumps({'parameters':parameters,'excluded':excluded,'qualified':False},indent=2)+'\n')
    (output/'worked-examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    text=['# Cluster 与 DSM 完成服务','','42 个坐标全部保留；独立 C 前均为候选结果。','',
        'DSM 主工作量为 B×N×4096 B；写实验含必要读回，应结合读回量解释。TMA 主量为 G×N×16384 B 源请求，接收总量另列。cluster 同步主量为 G×N 个阶段。两种范围均使用 globaltimer 完成包络；cluster_grid 表示查询所得 grid，不能据此声称全部 SM 覆盖或实际驻留。','']
    for scope in ('one_cluster','cluster_grid'):
        text += [f'## {scope}','','| 形式 | C | G | 状态 | 中位数 | 单位 | CV | 样本 |','|---|---:|---:|---|---:|---|---:|---:|']
        for row in table:
            if row['scope']!=scope:continue
            value='无有效统计' if row['median'] is None else format(row['median'],'.9g');cv='—' if row['cv'] is None else format(row['cv'],'.3%')
            text.append(f"| {row['form']} | {row['cluster_size']} | {row['clusters']} | {row['status']} | {value} | {row['unit']} | {cv} | {row['samples']} |")
    text+=['','## 原始字段算例','']
    for key,v in examples.items():
        formula=f"{v['clusters']}×{v['iterations']}" if v['work_unit']=='cluster_phase' else f"{v['clusters']}×{v['iterations']}×16384" if v['form'].startswith('bulk_') else f"{v['blocks']}×{v['iterations']}×4096"
        text += [f"`{key}`：{formula}={v['work_count']} {v['work_unit']}，除以 {v['elapsed_ns']} ns，得 {v['value']:.9g} {v['unit']}。接收总量 {v['total_receiver_bytes']} B；必要读回 {v['required_write_readback_bytes']} B；主计时后导出 {v['post_timing_export_bytes']} B。raw SHA `{v['raw_sha256']}`。",'']
    (output/'RESULTS.zh.md').write_text('\n'.join(text)+'\n')
    (output/'sources.json').write_text(json.dumps({'renderer_sha256':sha(__file__),'native_replay_sha256':sha(replay_file),
        'files':{n:sha(run/n) for n in ('summary.json','case-results.json','device.json','resources.json','resolved-cases.json','source-manifest.json')},
        'formal_process_count':len(samples),'qualified':False},indent=2)+'\n')
    if not tables_only:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for scope in ('one_cluster','cluster_grid'):
            fig,axes=plt.subplots(2,1,figsize=(12,8))
            for ax,sync in zip(axes,(False,True)):
                selected=[v for v in table if v['scope']==scope and (v['form']=='cluster_sync')==sync and v['median'] is not None]
                positions=list(range(len(selected)));med=[v['median'] for v in selected]
                if not selected:
                    ax.text(.5,.5,'No valid statistics',transform=ax.transAxes,ha='center')
                    continue
                ax.bar(positions,med,yerr=[[v['median']-v['min'] for v in selected],[v['max']-v['median'] for v in selected]],capsize=3)
                ax.set_xticks(positions,[v['form']+' C'+str(v['cluster_size']) for v in selected],rotation=55,ha='right')
                ax.set_ylabel('cluster phase / ns' if sync else 'GB transport / s');ax.set_title(scope+'; median with retained sample min/max')
            fig.tight_layout();fig.savefig(output/(scope+'.png'),dpi=160);plt.close(fig)
    (output/'report-manifest.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in output.iterdir() if p.is_file()},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--replay',required=True);parser.add_argument('--output',required=True);parser.add_argument('--tables-only',action='store_true')
    args=parser.parse_args();render(args.run,args.replay,args.output,args.tables_only)

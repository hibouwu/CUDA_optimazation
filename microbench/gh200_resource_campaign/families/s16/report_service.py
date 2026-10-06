"""Render completed S16 native replay; candidate status remains until C."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics
import tarfile
import sys

if sys.flags.optimize:
    raise RuntimeError("S16 reports require active assertions")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def render(run,replay_file,output,tables_only=False):
    run,output=Path(run),Path(output);replay=json.loads(Path(replay_file).read_text())
    assert replay['CPU_fixture_only'] is False and replay['all_saved_values_recomputed'] is True
    assert replay['qualified'] is False and replay['coordinate_count']==36
    summary=json.loads((run/'summary.json').read_text());assert summary['case_results']==replay['cases']
    cases=json.loads((run/'repo/cases.json').read_text())['cases'];by_id={c['id']:c for c in cases}
    resources=json.loads((run/'resources.json').read_text());device=json.loads((run/'device.json').read_text())
    samples=[];examples={};rows={}
    for path in sorted(run.glob('point*-state.json')):
        state=json.loads(path.read_text());assert state['state']=='verified_node_pack_pending_offhost'
        if state['kind']!='formal':continue
        case=by_id[state['case_id']];archive=run/'packs'/('point%04d.tar.xz'%state['index'])
        assert sha(archive)==state['pack_identity']['archive_sha256']
        with tarfile.open(archive) as packet:body=packet.extractfile('raw.jsonl').read()
        assert hashlib.sha256(body).hexdigest()==state['pack_identity']['members']['raw.jsonl']['sha256']
        actual,row=[json.loads(line) for line in body.decode().splitlines()];assert actual==device
        p=case['parameters'];amount=row['blocks']*row['iterations']*p['requests_per_item']*16384
        assert amount==row['work_count'] and row['software_stages']==p['software_stages']
        denominator=(row['blocks_detail'][0]['stop_cycle']-row['blocks_detail'][0]['start_cycle']
            if case['scope']=='one_cta' else row['stop_ns']-row['start_ns'])
        assert denominator>0
        sample={'case_id':case['id'],'batch':state['batch'],'trial':state['trial'],'seed':state['seed'],
            'stages':p['software_stages'],'requests_per_item':p['requests_per_item'],'blocks':row['blocks'],
            'iterations':row['iterations'],'payload_bytes':16384,'work_count_bytes':amount,'denominator':denominator,
            'value':amount/denominator,'unit':case['metric']['unit'],'warmup_converged':row['warmup_converged'],
            'post_timing_export_bytes':row['post_timing_final_export_bytes'],'raw_sha256':hashlib.sha256(body).hexdigest()}
        samples.append(sample);rows.setdefault(case['id'],row)
        if state['batch']==state['trial']==0 and p['software_stages']==p['requests_per_item']==1:
            examples.setdefault(case['scope'],sample)
    assert len(samples)==replay['formal_process_count']
    table=[];parameters=[];excluded=[]
    for case in cases:
        key=case['id'];p=case['parameters'];res=resources[key]
        if not res['legal']:
            excluded.append({'case_id':key,'reason':res['reason'],'target_launches':0,'resources':res})
            table.append({'case_id':key,'scope':case['scope'],'direction':p['direction'],'stages':p['software_stages'],
                'requests_per_item':p['requests_per_item'],'status':'capacity_reject','median':None,'min':None,'max':None,
                'cv':None,'samples':0,'unit':case['metric']['unit'],'iterations':None,'blocks':None,'qualified':False})
            continue
        result=replay['cases'][key];merged=result['merged'];row=rows[key]
        values=[v['value'] for v in samples if v['case_id']==key and v['warmup_converged']]
        if merged:assert len(values)==merged['samples'] and statistics.median(values)==merged['median']
        else:assert len(values)<2
        entry={'case_id':key,'scope':case['scope'],'direction':p['direction'],'stages':p['software_stages'],
            'requests_per_item':p['requests_per_item'],'status':result['status'],'median':merged['median'] if merged else None,
            'min':merged['min'] if merged else None,'max':merged['max'] if merged else None,'cv':merged['cv'] if merged else None,
            'samples':len(values),'unit':case['metric']['unit'],'iterations':row['iterations'],'blocks':row['blocks'],'qualified':False}
        table.append(entry)
        param={'case_id':key,'status':result['status'],'value':entry['median'],'unit':entry['unit'],'samples':len(values),
            'cv':entry['cv'],'statistic':'median','conditions':{**p,'threads':128,'iterations':row['iterations'],
                'blocks':row['blocks'],'capture':False,'global_slots_per_cta':32,'resources':res},
            'device_uuid':device['uuid'],'physical_HBM_bytes_proven':False,'physical_queue_depth_proven':False,
            'qualified':False,'qualification':'candidate_requires_independent_C'}
        (parameters if result['status']=='stable' else excluded).append(param)
    assert len(table)==36 and len(parameters)+len(excluded)==36
    output.mkdir(exist_ok=False)
    for name,data in [('samples.csv',samples),('results.csv',table)]:
        with (output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (output/'parameter-candidates.json').write_text(json.dumps({'parameters':parameters,'excluded':excluded,'qualified':False},indent=2)+'\n')
    (output/'worked-examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    text=['# TMA 软件 stage 与请求数的完成服务','','36 名义坐标全部保留；合法点的候选统计来自全部有效独立进程，超限项显示零 target launch。独立 C 前不授参数资格。','',
        '工作量为 B×N×R×16384 B；S 改变并发和缓冲资源，不直接乘入字节数。单 CTA 用自己的 clock64 差，全 GPU 用 globaltimer 完成包络。','']
    for scope in ('one_cta','all_gpu'):
        text += [f'## {scope}','','| 方向 | S | R | 状态 | 中位数 | 单位 | CV | 样本 |','|---|---:|---:|---|---:|---|---:|---:|']
        for row in table:
            if row['scope']!=scope:continue
            value='无有效统计' if row['median'] is None else format(row['median'],'.9g')
            cv='—' if row['cv'] is None else format(row['cv'],'.3%')
            text.append(f"| {row['direction']} | {row['stages']} | {row['requests_per_item']} | {row['status']} | {value} | {row['unit']} | {cv} | {row['samples']} |")
        if scope in examples:
            v=examples[scope];text += ['',f"真实算例 `{v['case_id']}`：{v['blocks']}×{v['iterations']}×{v['requests_per_item']}×16384={v['work_count_bytes']} B；除以 {v['denominator']} 得到 {v['value']:.9g} {v['unit']}。计时后导出的 {v['post_timing_export_bytes']} B 不计入主量。raw SHA `{v['raw_sha256']}`。",'']
    text += ['软件 stage、每 item 的请求数和硬件队列深度是不同对象；本实验只报告给定循环和资源的完成服务。不同 S/R 的比较还需匹配 CTA 数、占用上限和 N。','']
    (output/'RESULTS.zh.md').write_text('\n'.join(text)+'\n')
    (output/'sources.json').write_text(json.dumps({'renderer_sha256':sha(__file__),'native_replay_sha256':sha(replay_file),
        'files':{n:sha(run/n) for n in ['summary.json','case-results.json','device.json','resources.json','resolved-cases.json','source-manifest.json']},
        'formal_process_count':len(samples),'qualified':False},indent=2)+'\n')
    if not tables_only:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        for scope in ('one_cta','all_gpu'):
            fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
            for ax,direction in zip(axes,('gmem_to_smem','smem_to_gmem')):
                grid=np.full((3,3),np.nan)
                for i,S in enumerate((1,2,4)):
                    for j,R in enumerate((1,2,4)):
                        row=next(v for v in table if v['scope']==scope and v['direction']==direction and v['stages']==S and v['requests_per_item']==R)
                        if row['median'] is not None:grid[i,j]=row['median']
                        ax.text(j,i,'reject' if row['status']=='capacity_reject' else 'unstable' if row['status']!='stable' else format(row['median'],'.5g'),ha='center',va='center')
                image=ax.imshow(grid,cmap='Blues');ax.set_xticks(range(3),[1,2,4]);ax.set_yticks(range(3),[1,2,4])
                ax.set_xlabel('Requests per item R');ax.set_ylabel('Software stages S');ax.set_title(direction)
                fig.colorbar(image,ax=ax,label='B / clock64 cycle / CTA' if scope=='one_cta' else 'GB transport / s / GPU')
            fig.suptitle('S16 conditioned completion service; median of retained converged samples; pending C')
            fig.savefig(output/(scope+'.png'),dpi=160);plt.close(fig)
    (output/'report-manifest.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in output.iterdir() if p.is_file()},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--replay',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--tables-only',action='store_true')
    args=parser.parse_args();render(args.run,args.replay,args.output,args.tables_only)

#!/usr/bin/env python3
"""Render S09 raw-derived results; qualification is supplied by independent C."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def render(run, output, document):
    run=run.resolve();output=output.resolve();document=document.resolve()
    assert not output.exists() and not document.exists()
    summary=json.loads((run/'summary.json').read_text())
    spec=json.loads((run/'run_spec.json').read_text())
    contract=json.loads((run/spec['contract_path']).read_text())
    assert summary['family']=='shared_memory' and summary['terminal'] and spec['kind']=='formal' and not spec['fixture']
    conditions={c['id']:c for c in contract['cases']}
    records=[];parameters=[];examples=[]
    for case in summary['cases']:
        config=conditions[case['case_id']];p=config['parameters']
        directions=2 if p['mode']=='duplex' else 1
        work=256*8*p['access_bytes']*8192*directions
        words=set()
        for tid in range(256):
            for q in range(8):
                base=(tid//32)*8+q if p['broadcast'] else ((tid+q*256)*p['stride']*(p['access_bytes']//4))&8191
                for lane in range(p['access_bytes']//4):words.add((base+lane)&8191)
        for batch in case['batches']:
            for sample in batch['samples']:
                raw=run/sample['receipt'];raw=raw.parent/'raw.jsonl'
                assert sha(raw)==sample['raw_sha256']
                row=next(json.loads(l) for l in raw.read_text().splitlines() if json.loads(l)['type']=='trial')
                detail=row['blocks_detail'];assert len(detail)==1
                block=detail[0];cycles=block['stop_cycle']-block['start_cycle']
                assert row['iterations']==8192 and row['errors']==0 and row['work_count']==work and cycles>0
                value=work/cycles;assert value==sample['value']
                records.append({'case_id':case['case_id'],'batch':sample['batch'],'trial':sample['trial'],
                    'work_count':work,'clock64_cycles':cycles,'value':value,'unit':case['unit'],
                    'warmup_converged':sample['warmup_converged'],'raw':str(raw),'raw_sha256':sha(raw)})
                if sample['batch']==0 and sample['trial']==0:
                    examples.append({'case_id':case['case_id'],'raw':str(raw),'raw_sha256':sha(raw),'work_count':work,'cycles':cycles,'value':value})
        accepted=[x['value'] for x in records if x['case_id']==case['case_id'] and x['warmup_converged']]
        assert len(accepted)==case['merged']['samples'] and statistics.median(accepted)==case['merged']['median']
        parameters.append({'case_id':case['case_id'],'status':case['status'],'unit':case['unit'],'statistic':'median',
            'value':case['merged']['median'],'minimum':case['merged']['min'],'maximum':case['merged']['max'],
            'cv':case['merged']['cv'],'samples':len(accepted),'batches':len(case['batches']),
            'conditions':{'iterations':8192,'threads':256,'blocks':1,'dynamic_smem_bytes':65536,**p},
            'requested_bytes_per_iteration':work//8192,'distinct_address_bytes_per_direction_iteration':len(words)*4,
            'scope':'single_CTA_complete_access_checksum_and_barrier_loop',
            'source_summary_sha256':sha(run/'summary.json')})
    output.mkdir(parents=True)
    with (output/'samples.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (output/'worked_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    (output/'parameters.json').write_text(json.dumps({'schema_version':1,'kind':'conditioned_S09_services',
        'qualification':'requires_independent_C_and_COMPLETE','run':str(run),'parameter_count':len(parameters),
        'parameters':parameters,'physical_port_peak_claimed':False,'bare_latency_claimed':False},indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gh200-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6))
    for i,p in enumerate(parameters):
        vals=[x['value'] for x in records if x['case_id']==p['case_id'] and x['warmup_converged']]
        ax.scatter(vals,[i]*len(vals),s=12,alpha=.45,color='#26718a')
        ax.errorbar(p['value'],i,xerr=[[p['value']-min(vals)],[max(vals)-p['value']]],fmt='D',color='#183e52',capsize=3)
    ax.set_yticks(range(len(parameters)),[p['case_id'] for p in parameters]);ax.invert_yaxis()
    ax.set_xlabel('Requested B / clock64 cycle / CTA');ax.set_title('S09: median, min/max and all warmup-converged samples')
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(output/'services.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for batch in range(3):
        vals=[x for x in records if x['case_id']=='read_w4_broadcast' and x['batch']==batch]
        ax.scatter([batch*10+x['trial'] for x in vals],[x['value'] for x in vals],label='batch '+str(batch))
    ax.set_xlabel('Process index');ax.set_ylabel('Requested B / clock64 cycle / CTA')
    ax.set_title('Broadcast: all three batches retained');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(output/'broadcast.png',dpi=150);plt.close(fig)
    table='\n'.join('| '+p['case_id']+' | '+str(p['samples'])+' | '+f"{p['value']:.6f} | {p['minimum']:.6f}–{p['maximum']:.6f} | {100*p['cv']:.4f}% |" for p in parameters)
    example=next(x for x in examples if x['case_id']=='duplex_w16_stride1')
    text=f'''# EXP-09：GH200 SMEM 访问实测

本次固定单 CTA、256 线程、两个 32 KiB SMEM 数组，比较连续/跨步、向量、广播及独立读写。15 配置共 {len(records)} 个正式进程样本；结果单位为 **B/clock64 cycle/CTA**。窗口包含访问循环、加载 checksum 和 CTA 同步，初始化、回读与输出核验在窗口之外。

## 结果

| 配置 | 合并样本数 | 中位数 | 最小值–最大值 | CV |
|---|---:|---:|---:|---:|
{table}

![15配置的全部样本、中位数及最小最大范围]({output/'services.png'})

图中点为预热收敛的独立进程，菱形为中位数，横线为最小值到最大值，不是置信区间。广播配置触发三批，全部30个样本参与合并，CV约4.8%，接近5%的接受界限；应同时参考范围和原始分布。

![广播三批样本]({output/'broadcast.png'})

## 从原始字段算一条结果

`duplex_w16_stride1` 每线程每轮8次16 B读取和8次16 B写入，8192轮，逻辑请求为 `256×8×16×8192×2 = {example['work_count']:,} B`，读写各268,435,456 B。第一条原始记录的同 SM clock64差为 **{example['cycles']:,} cycle**，所以 `{example['work_count']}/{example['cycles']} = {example['value']:.9f} B/cycle/CTA`。见[原始记录]({example['raw']})。

广播每轮仍有 `256×8×4 = 8192 B` 逻辑请求；一个warp共享地址，整个CTA每轮只有 `8个warp×8个地址×4 B = 256 B` 不同地址。请求量与地址量含义不同，不能用请求速率推断物理端口流量。

## 如何用于模型

标量stride 2/4/8/16/32的请求速率约64/32/16/8/4 B/cycle，与地址映射相符；这组实验没有用硬件计数器证明bank事务量。连续8 B和16 B向量访问、向量写入及向量独立读写约128 B/cycle，但这是本循环的服务观测。4 B读取的checksum依赖与循环控制也占用计时窗口，不能把不同访问宽度之间的差异全部归因于SMEM裸带宽。

匹配线程数、访问宽度、地址、读写组织、64 KiB分配和完成边界后，可引用[条件参数]({output/'parameters.json'})。每条使用中位数，同时保留样本数、CV和范围。这里只测单CTA；ldmatrix、Tensor Core取数、TMA写SMEM、多CTA竞争及完整GEMM需要各自实验。

## 正确性、来源与复现

GPU job733542在romeo-a057正常结束。原b9f81编译/SASS产物保持不变；新GPU UUID采用独立审查的设备重验证分支，重新执行自身15点检查，同一分配再正式采样。原job733522在UUID准入检查时退出，无性能样本，继续保留。

每个加载lane参与host checksum参考，写入在计时后逐word核对；所有正式raw的errors=0、N=8192。未持久化全输出数组，数值证据限于原host检查。广播三批及全部失败/预热记录保留。ARM CPU原解释器的冻结审计、独立结果审查与COMPLETE决定最终资格；本文件及参数格式本身不授资格。当前未采集本组物理流量计数器。

原始归档：[run]({run})；逐样本数据：[samples.csv]({output/'samples.csv'})；手算字段：[worked_examples.json]({output/'worked_examples.json'})。

离线重算入口使用归档自己的冻结版本：

```sh
python3 -B {run/'snapshot/repo/microbench/gh200_resource_campaign/run_suite.py'} audit {run}
```

统计严格相等依赖原Python运行环境；本机与原ARM解释器的浮点末位差异单独记录，不能修改raw以消除差异。图表与表格可用 `report_shared_memory_v1.py --run <run> --output <新目录> --document <新文档>` 重生。
'''
    document.parent.mkdir(parents=True,exist_ok=True);document.write_text(text)
    manifest={'run':str(run),'summary_sha256':sha(run/'summary.json'),'measurement_manifest_sha256':sha(run/'measurement_manifest.json'),
        'script_sha256':sha(Path(__file__)),'sample_count':len(records),'parameter_count':len(parameters),
        'source_raw_files':{x['raw']:x['raw_sha256'] for x in records},'document':str(document),
        'outputs':{str(p):sha(p) for p in [*output.iterdir(),document]},'C_qualified_by_renderer':False}
    (output/'analysis-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'samples':len(records),'parameters':len(parameters),'document':str(document),'qualification':'pending_independent_C'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--document',type=Path,required=True)
    args=parser.parse_args();render(args.run,args.output,args.document)

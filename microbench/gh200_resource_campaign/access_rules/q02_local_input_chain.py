#!/usr/bin/env python3
"""D三请求到full就绪的局部接线与既有输入边界；CPU，无硬件拟合。"""
import argparse
from pathlib import Path
import statistics

from cfg_b_tma_chain import expected_descriptor, requests, timed_lifecycle
from l2_saturation import digest, read_json, require, save_json


def example_source(latencies, first_return=0.):
    def serve(t, request):
        atom = {'A':0, 'B0':1, 'B1':2}[request['operand']]
        return dict(source_id=3*request['tile']+atom,
            source_key=(request['operand'],request['source_k_tile']), merged=False,
            source_ready=t+latencies[atom], source_busy_ns=0., service_start=t,
            service_end=t, issue_return_ns=t+(first_return if atom == 0 else 0.))
    return serve


def report(root):
    score = read_json(root/'evidence/D-score.json')
    native = read_json(root/'evidence/native-B-plain.json')
    d_source = (root/'program/D.cu').read_text().replace(' ','')
    require('slot=tile%stages,gen=tile/stages' in d_source, 'D索引源码变化')
    require(digest(root/'program/D.cu') == score['source_sha256']['cfg_b_tma_chain.cu'], 'D源码身份不符')
    require(native['index_mode'] == 'incremental' and native['conditions']['prefilled_tiles'] == 6,
            '原生B索引或预填变化')
    contracts = []
    for stages, length in ((6,4),(6,13),(4,13)):
        case_id = f's{stages}_k{length}_plain'
        rows = [read_json(root/'evidence'/f'{case_id}-seed{seed}.json') for seed in (3,22,41)]
        require(all(row['stages'] == stages and row['ktiles'] == length and
            row['source_tiles'] == stages and row['source_k'] == 64*stages and
            row['conditions']['prefilled_tiles'] == 0 and not row['trace'] and
            row['conditions']['tma_request_bytes'] == [16384,8192,8192] for row in rows),
            '旧plain程序／请求条件变化')
        metadata = score['request_metadata'][case_id]
        require(metadata['encoding_inputs'] == expected_descriptor(stages), '描述符字段不符')
        require(metadata['requests'] == [request for tile in range(length)
            for request in requests(tile,stages,stages)], '请求地址／代次不符')
        cycles = statistics.median(row['timing']['elapsed_cycles'] for row in rows)
        require(cycles == score['aggregates'][case_id]['median_cycles'], '旧plain窗口与评分不符')
        contracts.append(dict(case_id=case_id,processes=3,plain_median_cycles=cycles,
            initial_tma_in_window=True,index_mode='runtime division/remainder',
            source_tiles=stages,A_row_stride_bytes=metadata['encoding_inputs']['A']['byte_strides'][0],
            B_row_stride_bytes=metadata['encoding_inputs']['B']['byte_strides'][0],
            logical_tma_requests=3*length,logical_tma_bytes=32768*length))
    timelines, checks = {}, []
    for name, latencies, returned, expected in (
        ('B1_last',(100.,140.,180.),0.,180.),
        ('A_last',(220.,140.,180.),0.,220.),
        ('A_return_delays_B',(100.,140.,180.),50.,230.)):
        value = timed_lifecycle(4,6,request_service=example_source(latencies,returned),
            compute_ns=512.,commit_lead_ns=256.)
        require(value['timeline'][0]['full_ready_ns'] == expected, '合成三请求连接失败')
        timelines[name] = value
        checks.append(dict(example=name,first_full_ready_ns=expected,passed=True))
    graphs = {}
    for stages in (4,6):
        for length in (4,13,32):
            value = timed_lifecycle(length,stages,
                request_service=example_source((100.,140.,180.)),
                compute_ns=512.,commit_lead_ns=256.)
            for row in value['timeline']:
                group = [request for request in value['requests'] if request['tile'] == row['tile']]
                require(row['full_ready_ns'] == max(request['source_ready_ns'] for request in group),
                        'full早于三笔完成')
                require(row['compute_begin_ns'] == row['full_return_ns'], '计算未绑定full返回')
                if row['tile'] >= stages:
                    require(row['request_A_issue_ns'] >= row['reuse_predecessor_ns'], '过早复用输入槽')
                if row['tile']:
                    require(value['timeline'][row['tile']-1]['release_ns'] == row['wait1_return_ns'],
                            'wait1释放组不符')
            require(value['timeline'][-1]['release_ns'] == value['tail_wait0_and_release_ns'],
                    'tail未释放最后一组')
            graphs[f's{stages}_k{length}'] = value
    files = [root/'plan.json',*(root/'evidence').iterdir(),*(root/'program').iterdir()]
    return dict(schema='gh200-q02-local-input-connection/1',qualification='CPU接线；合成ns不作为GH200服务参数',
        old_plain_contracts=contracts,native_B=dict(index_mode=native['index_mode'],
            prefilled_tiles=native['conditions']['prefilled_tiles'],full_transaction_bytes=0),
        synthetic_checks=checks,synthetic_timelines=timelines,graph_shapes=graphs,engine_calls=9,
        parameter_fits=0,gpu_jobs=0,independent_C2=False,independent_C3=False,
        missing_service_inputs=['匹配程序的三请求受理／就绪服务','runtime软件推进与条件返回',
            'tail释放后的累加器消费与CTA会合'],
        next_action='先固定匹配程序及源分配／描述符，再登记有限局部实测',
        input_sha256={str(path.relative_to(root)):digest(path) for path in sorted(files)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    value = report(args.root.resolve())
    save_json(args.output,value)
    print('D local connection: 9 engine calls, 9 old plain records, 3 synthetic cases, 6 graph shapes')


if __name__ == '__main__':
    main()

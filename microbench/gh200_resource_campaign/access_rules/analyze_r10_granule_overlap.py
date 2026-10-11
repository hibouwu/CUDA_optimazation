#!/usr/bin/env python3
"""R10 source-box versus address-granule overlap, without a new time model.

Static unions and a conditional replay of model-generated whole-box generations.
The old failed graph supplies the replay times; reduced work does not feed back
into it. Coverage/granule generations are hypotheses, not physical traffic.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from analyze_r10_supply_events import GPU, source_costs
from supply_model import cfg_b_source_program, shared_stage_events


@lru_cache(maxsize=16384)
def source_granules(key, width=128):
    """Valid global-address coverage within one A/B allocation, base aligned128.

    SW128 is a destination layout and does not alter these global addresses.
    The whole descriptor remains a distinct TMA request. Empty/OOB bytes do not
    generate address keys, but this does not remove destination fill or arrivals.
    """
    op,dtype,layout,extents,strides,offset,origin,shape = key
    if (op not in ('A','B') or dtype!='fp16' or layout!='SW128' or width not in (32,128)
            or shape!=((128,64) if op=='A' else (64,64))):
        raise ValueError('requires cfg_b FP16 source box and 32/128B coverage')
    if (any(type(v) is not int or v<0 for v in (*extents,*origin,offset))
            or any(type(v) is not int or v<=0 for v in strides)):
        raise ValueError('nonnegative integer descriptor coordinates and positive strides required')
    inner = 1 if op=='A' else 0
    outer = 1-inner
    if strides[inner]!=2:
        raise ValueError('source box must have contiguous FP16 inner coordinates')
    valid = [max(0,min(shape[d],extents[d]-origin[d])) for d in range(2)]
    row_bytes = valid[inner]*2
    if not row_bytes:
        return ()
    base = offset+sum(origin[d]*strides[d] for d in range(2))
    covered = set()
    for i in range(valid[outer]):
        start = base+i*strides[outer]
        covered.update(range(start//width,(start+row_bytes-1)//width+1))
    return tuple(sorted(covered))


def static_wave(keys, output, kt, width):
    """Software output ordinal, not a measured simultaneous request wave."""
    rows = []
    for op in ('A','B'):
        requests = [key for cta in keys if len(cta)>output*kt
            for key in cta[output*kt] if key[0]==op]
        distinct = set(requests)
        unions = set()
        multiplicity = Counter()
        distinct_coverage = 0
        for key in distinct:
            units = source_granules(key,width)
            unions.update(units)
            multiplicity.update(units)
            distinct_coverage += len(units)
        valid_payload = sum(key[7][0]*key[7][1]*2 for key in distinct)
        rows.append(dict(operand=op,width_bytes=width,requests=len(requests),distinct_boxes=len(distinct),
            request_coverage_KiB=sum(len(source_granules(k,width))*width/1024 for k in requests),
            distinct_box_valid_KiB=valid_payload/1024,
            distinct_box_coverage_KiB=distinct_coverage*width/1024,
            union_coverage_KiB=len(unions)*width/1024,
            overlap_between_distinct_boxes_KiB=(distinct_coverage-len(unions))*width/1024,
            distinct_box_extra_KiB=(distinct_coverage*width-valid_payload)/1024,
            union_extra_KiB=(len(unions)*width-valid_payload)/1024,
            units_covered_by_multiple_boxes=sum(n>1 for n in multiplicity.values())))
    return rows


def replay_granule_joins(sources, kt, width=128):
    """Additional overlap at the old graph's fixed arrivals/ready timestamps.

    Whole-box generations are already coalesced in `sources`. Different boxes
    can subscribe to a still-open address unit. Equality closes subscription.
    A unit inherits its old owner's ready time. No cache residency or new time
    prediction follows: feedback may shorten windows and change the generations.
    """
    pending = {}
    counts = {}
    last = -float('inf')
    for source in sources:
        t,end = source['arrival_ns'],source['ready_ns']
        if (type(t) not in (int,float) or type(end) not in (int,float)
                or not math.isfinite(t) or not math.isfinite(end) or t<last or end<t):
            raise ValueError('ordered model source arrivals and ready>=arrival required')
        last = t
        key = source['source_key']
        group = key[0],source['index']//kt
        row = counts.setdefault(group,dict(whole_box_generations=0,coverage_units=0,
            additional_join_units=0,granule_generations=0))
        row['whole_box_generations'] += 1
        for unit in source_granules(key,width):
            address = key[0],unit
            row['coverage_units'] += 1
            if address in pending and t<pending[address]:
                row['additional_join_units'] += 1
            else:
                row['granule_generations'] += 1
                pending[address] = end
    return [dict(operand=op,output=j,width_bytes=width,**value,
        coverage_KiB=value['coverage_units']*width/1024,
        additional_join_KiB=value['additional_join_units']*width/1024,
        generation_coverage_KiB=value['granule_generations']*width/1024)
        for (op,j),value in sorted(counts.items())]


def analyze(root, output):
    output.mkdir(parents=True,exist_ok=False)
    inputs = {}
    def read(path):
        inputs[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    price_report = read(root/'20261010-R10-raster-pitch/reanalysis/B-20261011-source-event-transfer-v3/B-r10-event-transfer.json')
    parameters = price_report['parameters']
    price = price_report['coverage_price_ns_per_KiB']
    cases = []
    runs = ('20261010-R10-raster-pitch','20261010-R10-source-pressure','20261009-V09-heldout-job739011')
    for run_name in runs:
        run = root/run_name
        rows = read(run/'cases.json')
        setups = {v['case']:v['setup'] for v in read(run/'static_setup.json')}
        for row in rows:
            if row.get('set')=='reference':
                continue
            if run_name==runs[-1] and row['id'] not in ('v09d_cfg_b_G5_base','v09d_cfg_b_G5_B32'):
                continue
            setup = setups[row['id']]
            program = cfg_b_source_program(row,setup)
            kt = row['k']//64
            waves = [dict(output=j,active_ctas=sum(n>j for n in program['output_counts']),
                coverage=[v for width in (32,128) for v in static_wave(program['source_keys'],j,kt,width)])
                for j in range(max(program['output_counts']))]
            result = dict(run=run_name,case=row['id'],row=row,gpu_uuid=setup['gpu_uuid'],
                stages=setup['stages'],k_tiles=kt,output_counts=program['output_counts'],
                ledger_totals=program['totals'],software_waves=waves)
            if run_name!=runs[-1]:
                if setup['gpu_uuid']!=GPU or setup['stages']!=6:
                    raise ValueError('conditional old-graph replay is confined to declared R10 card/stage')
                model = shared_stage_events(kt,6,len(program['output_counts']),
                    source_keys=program['source_keys'],output_counts=program['output_counts'],
                    share_inflight_sources=True,trace_requests=True,
                    source_service_costs_ns=source_costs(row,setup,parameters['service_ns'],
                        0. if row['ldb']==3072 else price),**parameters)
                totals = program['totals']
                if (model['input_demand_work']*16!=totals['source_payload_KiB']
                        or model['input_delivery_work']*16!=totals['destination_valid_KiB']+totals['destination_fill_KiB']
                        or sum(program['output_counts'])*8!=totals['output_nominal_subtiles']):
                    raise ValueError('old request/destination/D ledger changed')
                result['conditional_model_replay'] = dict(
                    original_whole_box_source_generations=model['input_source_count'],
                    original_source_payload_units16KiB=model['input_source_work'],
                    input_delivery_work_units16KiB=model['input_delivery_work'],
                    output_atoms=sum(program['output_counts'])*8,
                    address_join_counts=[v for width in (32,128)
                        for v in replay_granule_joins(model['sources'],kt,width)])
            else:
                result['conditional_model_replay'] = None
                result['timing_support'] = 'static geometry only; no GPU-ef prices transferred to GPU432'
            cases.append(result)
            source_granules.cache_clear()
    for name in ('analyze_r10_granule_overlap.py','analyze_r10_supply_events.py','supply_model.py','v08_model.py'):
        path = Path(__file__).with_name(name)
        inputs[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(schema='gh200-source-granule-overlap-screen/1',cases=cases,
        old_failed_parameters=parameters,old_approximate_B_coverage_price=price,
        status='Feature and conditional fixed-schedule screen only; no new service model, fit or holdout.',
        limits=['32/128B units are address coverage, not measured sectors/lines/physical traffic.',
            'Source allocation bases are declared aligned128; B offset remains explicit in source identities.',
            'Software output ordinal does not assert simultaneity; distinct boxes and their union are separate.',
            'Conditional replay uses only old model-generated arrivals/ready; no observed timestamps or clocks.',
            'Reduced work does not feed back into service/readiness/subscriptions; counts are not free predictions.',
            'No source/destination/D work is removed from the old ledger; a new granule service hypothesis needs its own graph.',
            'The old R10 parameter point remains rejected, including all first/later, spatial and intervention failures.',
            'Revealed V09 G5 is development geometry; cross-card timing remains unsupported.'],
        input_sha256=inputs)
    (output/'B-source-granule-overlap.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# R10：整个box与地址覆盖单元的重叠', '',
        '本报告只检查结构和旧失败图固定时间表下的附加订阅，不是新的时间预测或物理流量。', '',
        '| case / 软件轮 | B整box有效 KiB | B去重box覆盖 KiB | B覆盖并集 KiB | box间重叠 KiB |',
        '|---|---:|---:|---:|---:|']
    for case in cases:
        for wave in case['software_waves']:
            b = next(v for v in wave['coverage'] if v['operand']=='B' and v['width_bytes']==128)
            lines.append(f'| {case["case"]} / {wave["output"]} | {b["distinct_box_valid_KiB"]} | {b["distinct_box_coverage_KiB"]} | {b["union_coverage_KiB"]} | {b["overlap_between_distinct_boxes_KiB"]} |')
    lines += ['', '32B对照、A侧、每轮实际CTA数和旧模型生成时间下的逐轮附加订阅全部保存在JSON。',
        '单个源box、box去重和地址并集是不同账目；全并集只是静态极限，不能假定所有重叠都在途。',
        '旧时间表中可共享的地址单元若缩短服务，会反过来改变ready与订阅窗口，必须重新求解后才能预测。',
        '本报告不沿用失败图的价格作新模型结论，不迁GPU-ef价格到V09 GPU432。', '']
    (output/'B-source-granule-overlap.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),cases=len(cases),conditional_cases=sum(c['conditional_model_replay'] is not None for c in cases))))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args = parser.parse_args()
    analyze(args.data_root.resolve(),args.output.resolve())

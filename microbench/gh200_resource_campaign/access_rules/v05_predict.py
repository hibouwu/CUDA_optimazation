#!/usr/bin/env python3
"""V05 held-out case selection. Predictions must be frozen before GPU sampling.

The list command only prepares cases; it does not create a prediction or run a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import time
import warnings
import json
from pathlib import Path

CONFIGS = {
    "cfg_a": dict(tile=[128, 128, 64], cluster=[2, 1], schedule="cooperative"),
    "cfg_b": dict(tile=[128, 128, 64], cluster=[1, 1], schedule="pingpong"),
    "cfg_c": dict(tile=[256, 128, 64], cluster=[1, 2], schedule="cooperative"),
}
SHAPES = [
    ("partial_wave", 1280, 1536, 1536),
    ("short_k", 2304, 4096, 256),
    ("medium_k", 4096, 5120, 1024),
    ("large_output", 5120, 7168, 3072),
    ("long_k", 1024, 3584, 12288),
    ("edges_padded", 2696, 3336, 1544),
]


def ceil_div(a, b):
    return (a + b - 1) // b


def prior_shapes(history):
    """Conservatively exclude any previously logged shape, even on another config."""
    found = set()

    def visit(value):
        if isinstance(value, dict):
            shape = tuple(value.get(k) for k in ("m", "n", "k"))
            if all(isinstance(x, int) and x > 0 for x in shape):
                found.add(shape)
            args = value.get("args", [])
            if isinstance(args, list):
                options = dict(zip(args[::2], args[1::2])) if len(args) % 2 == 0 else {}
                if all(k in options for k in ("--m", "--n", "--k")):
                    found.add(tuple(int(options[k]) for k in ("--m", "--n", "--k")))
            for key, child in value.items():
                if key not in ("trace", "checked_indices", "checked_values", "warmup_us"):
                    if isinstance(child, (dict, list)):
                        visit(child)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    visit(item)

    for path in sorted(Path(history).rglob("samples.jsonl")):
        for line in path.open():
            if line.strip():
                visit(json.loads(line))
    # New families save one JSON object per independent process.
    for path in sorted(Path(history).rglob("result.json")):
        content=path.read_text()
        try:
            record=json.loads(content)
        except json.JSONDecodeError:
            # A quota-interrupted legacy record can still contain the complete
            # command identity before its large arrays. Exclude that attempted
            # shape conservatively; this does not qualify the incomplete sample.
            prefix=content.split('"calls":',1)[0].rstrip().rstrip(',')+'}'
            record=json.loads(prefix)
            if not record.get('args'): raise ValueError('unidentified partial record: '+str(path))
            warnings.warn('Novelty only: complete command prefix from partial '+str(path))
        visit(record)
    return found


def cases(used=()):
    used = set(used)
    selected = []
    for config, spec in CONFIGS.items():
        for label, original_m, n, k in SHAPES:
            m = original_m
            while (m, n, k) in used:
                m += 128
            padded = label == "edges_padded"
            tm, tn, _ = spec["tile"]
            cm, cn = spec["cluster"]
            units = ceil_div(ceil_div(m, tm), cm) * cm * ceil_div(ceil_div(n, tn), cn) * cn
            selected.append(dict(
                id=f"{config}_{label}", config=config, label=label, m=m, n=n, k=k,
                original_m=original_m,
                lda=ceil_div(k, 64) * 64 if padded else k,
                ldb=ceil_div(n, 64) * 64 if padded else n,
                ldd=ceil_div(n, 32) * 32 if padded else n,
                **spec, scheduled_units=units, ktiles=ceil_div(k, 64),
                grid_ctas=None, scheduler_status="requires_cutlass_static_setup"))
    return selected


def scheduled_work(setup):
    """CUTLASS 3.9.2 static scheduler, max_swizzle_size=1, heuristic raster."""
    tm,tn,_ = setup['tile']; cm,cn = setup['cluster']; gx,gy,gz = setup['grid']
    if gz != 1: raise ValueError('only one GEMM batch is in scope')
    nm = ceil_div(ceil_div(setup['m'],tm),cm)*cm
    nn = ceil_div(ceil_div(setup['n'],tn),cn)*cn
    along_n = nn <= nm
    minor,major,major_blocks = (cm,cn,nn//cn) if along_n else (cn,cm,nm//cm)
    result = []
    for by in range(gy):
        for bx in range(gx):
            first = bx + by*gx if along_n else bx*gy+by
            offset = bx%cm if along_n else by%cn
            work = []
            for linear in range(first,nm*nn,gx*gy):
                cluster,major_offset = divmod(linear//minor,major)
                cluster_minor,cluster_major = divmod(cluster,major_blocks)
                mi = cluster_minor*minor+offset
                ma = cluster_major*major+major_offset
                work.append((mi,ma,0) if along_n else (ma,mi,0))
            result.append(work)
    all_tiles = [tile for work in result for tile in work]
    expected = {(m,n,0) for m in range(nm) for n in range(nn)}
    if len(all_tiles)!=len(expected) or set(all_tiles)!=expected:
        raise ValueError('static scheduler mapping does not cover padded tiles')
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interpolate(points, ktiles):
    points = sorted(points, key=lambda p: p['ktiles'])
    if len(points) < 2 or len({p['ktiles'] for p in points}) != len(points):
        raise ValueError('each curve needs distinct calibration K lengths')
    pair = points[:2] if ktiles <= points[0]['ktiles'] else points[-2:]
    for left, right in zip(points, points[1:]):
        if left['ktiles'] <= ktiles <= right['ktiles']:
            pair = [left, right]
            break
    left, right = pair
    fraction = (ktiles-left['ktiles'])/(right['ktiles']-left['ktiles'])
    value = left['cycles'] + fraction*(right['cycles']-left['cycles'])
    return value, ktiles < points[0]['ktiles'] or ktiles > points[-1]['ktiles']


def freeze(run, parameters, output):
    """Freeze explicit calibrated curves; never infer missing parameters as zero."""
    if output.exists(): raise ValueError('prediction file already exists')
    rows = json.loads((run/'cases.json').read_text())
    setups = {r['case_id']: r for r in json.loads((run/'static_setup.json').read_text())}
    params = json.loads(parameters.read_text())
    # 'qualified_calibration' is the status used by the archived 2026-10-07 run.
    if params.get('status') not in ('calibrated', 'qualified_calibration'):
        raise ValueError('calibration has missing parameters')
    if not params.get('sources'):
        raise ValueError('no calibration evidence')
    evidence = {source['path'] for source in params['sources']}
    for source in params['sources']:
        if digest(source['path']) != source['sha256']:
            raise ValueError('parameter evidence changed: '+source['path'])
    predictions, allocation, microseconds = {}, {}, {}
    for row in rows:
        config = params['configs'][row['config']]
        if not config.get('critical_assumptions'):
            raise ValueError('critical model requires explicit homogeneity/overlap assumptions')
        setup = setups[row['id']]
        grid = math.prod(setup['grid'])
        work = scheduled_work(setup)
        rounds = max(map(len,work))
        if rounds > 32: raise ValueError('prediction exceeds trace capacity')
        allocation[row['id']] = dict(grid=setup['grid'], max_tiles_per_cta=rounds,
            scheduled_units=row['scheduled_units'], cta_work=work,
            rule='static persistent scheduler; each CTA advances by actual grid size',
            stages=setup['stages'], cluster=setup['cluster'])
        phases = {}
        for phase in ('supply', 'mainloop', 'output', 'handoff', 'epi_permission_wait'):
            curve = config['curves'][phase]
            if not isinstance(curve.get('source'), list) or not curve['source']:
                raise ValueError('curve source must list evidence paths')
            if not set(curve['source']).issubset(evidence):
                raise ValueError('curve source is not bound to checked evidence')
            if curve.get('kind') == 'constant_service':
                if phase not in ('output','epi_permission_wait') or len(curve.get('points',[]))!=1:
                    raise ValueError('single-point constant hypothesis is limited to E/W')
                point=curve['points'][0]
                if type(point.get('ktiles')) is not int or point['ktiles']<=0:
                    raise ValueError('constant anchor requires a positive K-tile count')
                if not curve.get('transfer_assumptions') or not curve.get('points'):
                    raise ValueError('constant service needs evidence and an explicit K-invariance assumption')
                value=curve['points'][0]['cycles']
                extrapolated=row['ktiles'] not in [point['ktiles'] for point in curve['points']]
            else:
                value, extrapolated = interpolate(curve['points'], row['ktiles'])
            if not math.isfinite(value) or (phase!='handoff' and value<0):
                raise ValueError('invalid predicted '+phase)
            if not curve.get('source') or not curve.get('events'):
                raise ValueError('curve lacks source/event conditions')
            phases[phase] = dict(cycles=value, source=curve['source'], events=curve['events'],
                extrapolation=extrapolated, transfer_assumptions=curve.get('transfer_assumptions', []))
        phases['handoff']['applicable'] = len(work[0]) > 1
        phases['handoff']['observation_rule'] = 'CTA0 needs at least two output tiles'
        # Repeated tile computation and its observed handoff gap already contain
        # pipeline waits. Do not add the full R13 or R15 window a second time.
        value = (phases['supply']['cycles'] + rounds*phases['mainloop']['cycles']
                 + (rounds-1)*phases['handoff']['cycles']
                 + phases['epi_permission_wait']['cycles'] + phases['output']['cycles'])
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid critical CTA prediction')
        phases['critical_cta'] = dict(cycles=value,
            source=[phases[k]['source'] for k in phases],
            events=['first_producer_work', 'final_source_release'],
            equation='S + tiles*L + (tiles-1)*G + W + E',
            transfer_assumptions=config.get('critical_assumptions', []))
        predictions[row['id']] = phases
        timing = config['microseconds']
        if timing['frequency_mhz'] <= 0 or not math.isfinite(timing['fixed_us']):
            raise ValueError('invalid pre-measurement time parameters')
        if not timing.get('source') or not set(timing['source']).issubset(evidence):
            raise ValueError('time prediction needs bound calibration evidence')
        microseconds[row['id']] = dict(
            predicted_us=value/timing['frequency_mhz']+timing['fixed_us'],
            frequency_mhz=timing['frequency_mhz'], fixed_us=timing['fixed_us'],
            source=timing['source'], assumptions=timing['assumptions'],
            role='secondary prediction; measured frequency must not refit it')
    value = dict(status='frozen', frozen_unix_ns=time.time_ns(), cases=rows,
        static_setup_sha256=digest(run/'static_setup.json'),
        source_manifest_sha256=digest(run/'source_hashes.json'),
        binary_manifest_sha256=digest(run/'build/binary_hashes.json'),
        environment_sha256=digest(run/'environment.json'),
        parameters_sha256=digest(parameters), parameters=params,
        work_assignment=allocation, predictions=predictions, microseconds=microseconds,
        validation='phase median absolute relative error <=10%, max <=20%; no refit on V05')
    with output.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    output.chmod(0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true')
    action.add_argument('--freeze', action='store_true')
    parser.add_argument('--history', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.freeze:
        if not all((args.run,args.parameters,args.output)):
            parser.error('--freeze requires --run, --parameters and --output')
        freeze(args.run,args.parameters,args.output)
        return
    if not args.history: parser.error('--list requires --history')
    selected = cases(prior_shapes(args.history))
    assert len(selected) == 18 and len({x['id'] for x in selected}) == 18
    print(json.dumps(dict(status="cases_only_not_frozen", cases=selected), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""cfg_b CPU 聚合组合、冻结回归与历史 trace 对照；不提交 GPU 作业。"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import sys

import cfg_b_program
import clock_model as clock
import output_model
import supply_model as supply
import v06_model
import v08_model

DEFAULT_ARCHIVE = Path(__file__).resolve().parents[3] / 'results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011'


class V09Mainloop:
    """整段 L 的经验替代：输入整个程序，输出各 CTA/tile 的 ns 与支持标记。"""
    version = 'V09-r2-full-L'

    def __init__(self, program, calibration):
        requests = supply.supply_request_rows(program['case'], program['setup'])
        later = calibration['supply']['cfg_b']
        first = calibration.get('first_supply', {}).get('cfg_b', later)
        self.phases = [(first, [r for r in requests if r['j'] == 0]),
                       (later, [r for r in requests if r['j'] > 0])]
        self.ctas = len(program['work'])

    def intervals(self, frequency_ghz):
        loops = [[] for _ in range(self.ctas)]
        supported = []
        for model, requests in self.phases:
            values, valid = supply.supply_predict(requests, model, frequency_ghz=frequency_ghz)
            for request, value, ok in zip(requests, values, valid):
                loops[request['cta']].append((request['j'], float(value)))
                supported.append((request['cta'], request['j'], bool(ok)))
        return [[v for _, v in sorted(cta)] for cta in loops], supported


class V09Events:
    """聚合前缀、交接与输出价格；E/r/w/we/Etail 及额外尾差沿用冻结版本。"""
    version = 'V09-r2-pingpong-events'

    def __init__(self, program, calibration):
        self.params = calibration['events_ns']['cfg_b']
        self.rule = calibration.get('output_rules', {}).get('cfg_b')
        self.fractions = [[o['output']['valid_KiB'] / o['output']['nominal_KiB']
                           for o in cta] for cta in program['outputs']]
        self.first_bytes = sum(f[0] * 128 * 128 * 4 for f in self.fractions if f)
        self.single = max(map(len, program['work'])) == 1

    def parameters(self, cta):
        params, _ = output_model.output_intervals(self.params, 'pingpong',
            self.fractions[cta], self.first_bytes, self.single, self.rule)
        return params

    def output_end(self, params, start_ns, next_mainloop):
        return v06_model.epilogue_end(params, start_ns, next_mainloop)


class V09Frequency:
    """整卡工作与包络联立，返回 GHz；程序实例中不保存时钟价格。"""
    version = 'V09-r2-source-clock'

    def __init__(self, program, calibration):
        row, work, kt = program['case'], program['work'], program['ktiles']
        tiles = program['totals']['output_tiles']
        self.point = dict(input_mode=row['input_mode'], tensor_mean_cycles=tiles * kt * 512.0 / 132,
            source_mean_kib=program['totals']['source_payload_KiB'] / 132)
        m, n = row['input_map_m'], row['input_map_n']
        if row['zero_m'] >= 0:
            m = min(m, row['zero_m'])
        if row['zero_n'] >= 0:
            n = min(n, row['zero_n'])
        useful = sum(max(0, min(128, m-mi*128)) * max(0, min(128, n-ni*128)) * row['k']
                     for cta in work for mi, ni in cta)
        nominal = tiles * 128 * 128 * kt * 64
        self.zero_fraction = 1.0 if row['input_mode'] == 'zero' else 1-useful/nominal
        self.mixing = bool(calibration.get('zero_activity_mixing', False))
        self.model = calibration['clock']
        if self.mixing:
            self.model = dict(self.model, coefficients=self.model['coefficients'][:])
            mode, zero = clock.MODES.index(self.point['input_mode']), clock.MODES.index('zero')
            z = self.zero_fraction
            for offset in (1, 1 + len(clock.MODES)):
                coefficients = self.model['coefficients']
                coefficients[offset+mode] = (1-z)*coefficients[offset+mode] + z*coefficients[offset+zero]

    def solve(self, envelope_us):
        return clock.solve_source_envelope(self.point, self.model, envelope_us)


class V09Observation:
    """包络 ns/us 到 dual/plain 协议的经验映射。"""
    version = 'V09-r2-plain-dual'

    def __init__(self, calibration):
        self.extra_ns = calibration['dual_event_extra_ns']['cfg_b']
        self.transfer = calibration.get('plain_transfer', {}).get('cfg_b')

    def project(self, solved):
        return dict(dual_event_us=solved['window_us'] + self.extra_ns/1000,
            plain_us=(self.transfer['F_us'] + self.transfer['kappa']*solved['window_us']) if self.transfer else None)


def default_services(program, calibration):
    return dict(mainloop=V09Mainloop(program, calibration), events=V09Events(program, calibration),
        frequency=V09Frequency(program, calibration), observation=V09Observation(calibration))


def cta_timeline(mainloops, params, output_end, details):
    """V09 pingpong 聚合递推；保持原运算顺序以复现冻结浮点输出。"""
    count = len(mainloops)
    if count == 0:
        return params['P0'], None
    fm, me, ep, ed = [], [], [], []
    for j, duration in enumerate(mainloops):
        if j == 0:
            start = params['P0'] + params['S']
        else:
            start = me[j-1] + params['gm']
            if j >= 2:
                start = max(start, ed[j-2] + params['h'])
        fm.append(start)
        me.append(start + duration)
        epi = me[j] + params['w']
        if j:
            epi = max(epi, ed[j-1] + params['we'])
        ep.append(epi)
        if j+1 < count:
            next_start = me[j] + params['gm']
            if j:
                next_start = max(next_start, ed[j-1] + params['h'])
            overlap = (next_start, next_start + mainloops[j+1])
        else:
            overlap = None
        ed.append(output_end(params, epi, overlap))
    base = max(ed) + params['Etail']
    excess = params['x0'] + params['x1']*count + params['xk']*base
    parts = None
    if details:
        parts = dict(supply=params['P0']+params['S'], mainloop=sum(mainloops),
            last_epilogue=ed[-1]-ep[-1], tail=params['Etail'], max_cta_excess=excess,
            events=[list(t) for t in zip(fm, me, ep, ed)], producer=params['P0'], prefill=params['S'])
    return base + excess, parts


def predict(program, calibration, services=None):
    """消费共同程序实例；services 可替换单个已准备好的模块，保留其余默认服务。"""
    selected = default_services(program, calibration)
    selected.update(services or {})
    mainloop, events, frequency, observation = (selected[k] for k in ('mainloop', 'events', 'frequency', 'observation'))

    def evaluate(ghz, details=False):
        loops, supported = mainloop.intervals(ghz)
        durations, timelines = [], []
        for cta, work in enumerate(program['work']):
            if len(loops[cta]) != len(work):
                raise ValueError('mainloop service must return one interval per program output')
            duration, timeline = cta_timeline(loops[cta], events.parameters(cta), events.output_end, details)
            durations.append(duration)
            if details:
                timelines.append(timeline)
        return durations, supported, timelines

    solved = frequency.solve(lambda f: max(evaluate(f)[0])/1000)
    durations, supported, timelines = evaluate(solved['frequency_ghz'], True)
    critical = max(range(len(durations)), key=durations.__getitem__)
    return dict(**solved, **observation.project(solved),
        zero_product_fraction=frequency.zero_fraction, zero_activity_mixing=frequency.mixing,
        all_supply_supported=all(ok for _, _, ok in supported),
        unsupported_windows=[dict(cta=c, j=j) for c, j, ok in supported if not ok],
        critical_cta=critical,
        critical_candidates=[i for i, v in enumerate(durations) if abs(v-max(durations)) < 1e-7],
        critical_work=program['work'][critical], cta_duration_ns=durations,
        first_output_bytes=events.first_bytes, static_clock_work=frequency.point, cta_events=timelines)


def load_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_differences(predicted, expected):
    """逐字段比较规范 JSON，保留浮点表示与正负零；忽略字典键顺序。"""
    return [key for key in sorted(expected.keys() | predicted.keys())
            if key not in expected or key not in predicted or
            json.dumps(predicted[key], sort_keys=True, allow_nan=False) !=
            json.dumps(expected[key], sort_keys=True, allow_nan=False)]


def archive_identity(root):
    """核对冻结清单、关键源码和 SASS；二进制身份保留构建时登记值。"""
    binding = load_json(root/'prediction_binding.json')
    for filename, key in (('frozen/predictions.json', 'prediction_sha256'), ('frozen/manifest.json', 'manifest_sha256')):
        if sha256(root/filename) != binding[key]:
            raise ValueError('frozen binding mismatch: ' + filename)
    manifest = load_json(root/'frozen/manifest.json')
    verified = {}
    for name in ('cases.json', 'source_hashes.json', 'frozen/calibration.json',
                 'build/binary_hashes.json', 'build/sass_hashes.json', 'build/commands.json', 'build/nvcc-version.txt'):
        actual = sha256(root/name)
        if actual != manifest['files'][name]:
            raise ValueError('manifest mismatch: ' + name)
        verified[name] = actual
    sources = load_json(root/'source_hashes.json')
    for name, expected in sources.items():
        if name.endswith(('sm90_gemm_tma_warpspecialized_pingpong.hpp',
                          'sm90_mma_tma_gmma_ss_warpspecialized.hpp', 'sm90_pipeline.hpp',
                          'copy_sm90_tma.hpp', 'probes/r18.cu', 'probes/v06_trace.hpp')):
            if sha256(root/name) != expected:
                raise ValueError('source mismatch: ' + name)
            verified[name] = expected
    for name, expected in load_json(root/'build/sass_hashes.json').items():
        if name.startswith('cfg_b'):
            if sha256(root/'build'/name) != expected:
                raise ValueError('SASS mismatch: ' + name)
            verified['build/'+name] = expected
    return dict(archive=str(root), source_commit=manifest['source_commit'], verified_sha256=verified,
        recorded_binary_hashes=load_json(root/'build/binary_hashes.json'),
        static_setup_sha256=sha256(root/'static_setup.json'), prediction_binding=binding)


def compare_trace(root, row, program, prediction, pattern='dual-00.json'):
    """回放已有记录，逐坐标核对；残差保留为开发对照，不作为回归通过条件。"""
    from analyze_r18 import replay
    paths = sorted((root/'samples'/row['id']).glob(pattern))
    if not paths:
        raise ValueError('no trace records for ' + row['id'])
    reports = []
    for path in paths:
        record = load_json(path)
        if record['returncode']:
            if sha256(root/record['raw']) != record['raw_sha256']:
                raise ValueError('failed process raw identity changed')
            reports.append(dict(record=str(path.relative_to(root)), record_sha256=sha256(path),
                raw_sha256=record['raw_sha256'], status='failed_process', returncode=record['returncode']))
            continue
        actual = replay(root, record, row, retain_roles=True)
        residuals = [[] for _ in range(4)]
        relative = {key: [] for key in ('prefix', 'prefill', 'first_L', 'later_L', 'last_E', 'tail')}
        for cta, observed in enumerate(actual['ctas']):
            if observed['work'] != program['work'][cta]:
                raise ValueError(f'trace work differs: {row["id"]} CTA {cta}')
            for role in range(2):
                if observed['role_work'][role] != program['work'][cta][role::2]:
                    raise ValueError('trace pingpong owner differs')
            predicted = prediction['cta_events'][cta]
            pe = predicted['events'] if predicted else []
            measured = observed['tiles_ns']
            for p, o in zip(pe, measured):
                for index in range(4):
                    residuals[index].append(p[index] - (o[index]-observed['entry_ns']))
            if measured:
                for key, actual_value, predicted_value in (
                    ('prefix', measured[0][0]-observed['entry_ns'], predicted['supply']),
                    ('prefill', measured[0][0]-observed['prod_ns'], predicted['prefill']),
                    ('first_L', measured[0][1]-measured[0][0], pe[0][1]-pe[0][0]),
                    ('last_E', measured[-1][3]-measured[-1][2], pe[-1][3]-pe[-1][2]),
                    ('tail', observed['end_ns']-max(t[3] for t in measured), predicted['tail'])):
                    if actual_value > 0:
                        relative[key].append(predicted_value/actual_value-1)
                relative['later_L'] += [(p[1]-p[0])/(o[1]-o[0])-1
                    for o, p in zip(measured[1:], pe[1:]) if o[1] > o[0]]
        last = max(range(len(actual['ctas'])), key=lambda c: actual['ctas'][c]['end_ns'])
        reports.append(dict(record=str(path.relative_to(root)), record_sha256=sha256(path), status='matched',
            raw_sha256=record['raw_sha256'], ctas=len(actual['ctas']),
            tiles=sum(len(c['work']) for c in actual['ctas']), coordinates_match=True,
            owners_match=True, observed_last_cta=last,
            last_in_predicted_ties=last in prediction['critical_candidates'],
            component_median_errors={key: statistics.median(values) if values else None
                for key, values in relative.items()},
            endpoint_error_ns={name: dict(median=statistics.median(values),
                max_abs=max(map(abs, values)), count=len(values))
                for name, values in zip(cfg_b_program.TRACE_ENDPOINTS, residuals)}))
    if not any(r['status'] == 'matched' for r in reports):
        raise ValueError('no successful trace process for ' + row['id'])
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--case', help='省略时检查全部 12 个 cfg_b 历史冻结条件')
    parser.add_argument('--trace', action='store_true', help='回放每个条件的 dual-00 历史记录')
    parser.add_argument('--all-traces', action='store_true', help='回放所选条件的全部 dual 进程')
    parser.add_argument('--cta', type=int, help='导出一个 CTA 的细事件依赖；须指定 --case')
    parser.add_argument('--output', type=Path, help='保存 JSON；省略时输出到 stdout')
    args = parser.parse_args()
    if args.cta is not None and not args.case:
        parser.error('--cta requires --case')
    root = args.archive.resolve()
    if args.output:
        output = args.output.resolve()
        results = Path(__file__).resolve().parents[3]/'results'
        if output.is_relative_to(results) or output.is_relative_to(root):
            parser.error('output must be outside historical results/archive')
    identity = archive_identity(root)
    calibration = load_json(root/'frozen/calibration.json')
    frozen = load_json(root/'frozen/predictions.json')['predictions']
    setups = {r['case']: r['setup'] for r in load_json(root/'static_setup.json')}
    rows = [r for r in load_json(root/'cases.json') if r['config'] == 'cfg_b'
            and r['set'] == 'heldout' and frozen[r['id']]['status'] == 'predicted']
    if args.case:
        rows = [r for r in rows if r['id'] == args.case]
    if not rows:
        parser.error('no matching frozen cfg_b case')
    reports = []
    for row in rows:
        program = cfg_b_program.build_program(row, setups[row['id']], identity)
        prediction = predict(program, calibration)
        expected = {k: v for k, v in frozen[row['id']].items() if k != 'status'}
        changed = frozen_differences(prediction, expected)
        if changed:
            raise ValueError(f'frozen prediction mismatch: {row["id"]}: {changed}')
        report = dict(case=row['id'], stages=program['stages'], frozen_fields=len(expected),
            frozen_exact=True, totals=program['totals'], prediction=prediction)
        if args.trace or args.all_traces:
            report['trace'] = compare_trace(root, row, program, prediction,
                'dual-*.json' if args.all_traces else 'dual-00.json')
        if args.cta is not None:
            if not 0 <= args.cta < len(program['work']):
                parser.error('--cta outside grid')
            report['program'] = program
            report['graph'] = cfg_b_program.expand_cta(program, args.cta)
        reports.append(report)
    result = dict(schema='gh200-cfg-b-composition-report/1', mode='aggregate',
        qualification='历史冻结回归与程序对照，未接入请求级时间服务', source_identity=identity,
        implementation_sha256={Path(module.__file__).name: sha256(Path(module.__file__))
            for module in (cfg_b_program, cfg_b_program.case_inputs, supply, clock, output_model, v06_model, v08_model)},
        composition_sha256=sha256(Path(__file__)),
        runtime=dict(python=sys.version, numpy=version('numpy'), scipy=version('scipy')),
        modules={name: dict(version=service.version, qualification='经验替代')
            for name, service in default_services(program, calibration).items()},
        cases=reports)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(encoded)
        print(f'PASS {len(reports)} cfg_b cases: all frozen fields exactly reproduced')
        if args.trace or args.all_traces:
            traces = [t for report in reports for t in report['trace']]
            print(f'TRACE {sum(t["status"] == "matched" for t in traces)} processes: coordinates and owners match; '
                  f'{sum(t["status"] == "failed_process" for t in traces)} historical failed attempts retained')
        print(args.output)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()

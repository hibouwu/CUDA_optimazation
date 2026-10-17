"""冻结 12 个条件、价格组、预测、判据和已核对的原生二进制。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(development, output):
    analysis = development/'candidate-screening.json'
    report = json.loads(analysis.read_text())
    checks = json.loads((development/'check.json').read_text())
    assert checks['memcheck'] == '0 errors' and checks['samples'] == 28
    assert len(report['predictions']) == 12
    assert all(report['retained'].values()), 'both candidate families must survive calibration'
    assert report['phase_contrasts']['shared_source_fifo']['response_min'] > .15
    assert report['phase_contrasts']['resident_source_fifo']['response_max'] == 0
    for name, expected in report['source_summaries'].items():
        assert digest(Path(name)) == expected
    source = development/'analysis-source'
    assert digest(source/'tma_tensor_multicta.py') == report['implementation_sha256']
    assert digest(source/'analyze_tma_tensor_candidates.py') == report['analysis_sha256']
    output.mkdir()
    (output/'build').mkdir()
    (output/'evidence').mkdir()
    shutil.copytree(source,output/'source')
    shutil.copy2(Path(__file__).with_name('run_tma_tensor_probe.py'),output/'source/run_tma_tensor_probe.py')
    shutil.copy2(Path(__file__).with_name('score_tma_tensor_discrimination.py'),output/'source/score_tma_tensor_discrimination.py')
    shutil.copy2(Path(__file__),output/'source'/Path(__file__).name)
    shutil.copy2(analysis,output/'predictions.json')
    shutil.copy2(development/'returned/build/probe',output/'build/probe')
    shutil.copy2(development/'input.tar.gz',output/'evidence/native-input.tar.gz')
    shutil.copy2(development/'check.json',output/'evidence/native-check.json')
    for name in ('probe.sass','resources.txt','compile.log','nvcc-version.txt'):
        shutil.copy2(development/'returned/build'/name,output/'evidence'/name)
    shutil.copy2(development/'calibration-endpoints.json',output/'evidence/calibration-endpoints.json')
    conditions = []
    for row in report['predictions']:
        p = row['program']
        for trace in (0,1):
            args = dict(ctas=p['ctas'],active=p['active_groups'],groups=p['groups'],ring=p['source_ring'],
                independent=int(p['source_protocol']=='independent'),verify=0,trace=trace,warmup=10)
            args['source-phase-stride'] = p['source_phase_stride']
            conditions.append(dict(id=row['id']+f'-t{trace}',arguments=args))
    plan = dict(schema='tensor-probe-run/1',purpose='frozen C2 discrimination: Q1/Q4 transfer and equal-footprint phase intervention',
        gpu_uuid='GPU-43269fbc-449d-3e0f-908a-9c81229546d3',probe_sha256=digest(output/'build/probe'),
        processes=7,order_seed=2026101404,process_timeout_seconds=60,conditions=conditions,
        primary_endpoint='trace=1 CTA begin to final group release, median over CTAs then processes, ns',
        scoring=dict(joint_parameter_requirement='one pre-frozen price set must explain all 12 conditions',
            max_absolute_relative_error=.05,phase_response_tolerance=.05,
            phase_response='shifted median CTA window / aligned median CTA window - 1',
            process_uncertainty='report full seven-process ranges and paired responses; do not enlarge thresholds afterward',
            trace_bridge='untraced phase response must be >=15% for shared_source_fifo support, or within +/-5% for resident_source_fifo support',
            frequency='score ns without measured-frequency substitution; report cycle/ns contrast separately',
            conclusion='support only if joint timing and phase/bridge tests pass and competing family is excluded; otherwise refuted price family or insufficient information'))
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    binding = dict(frozen_at=datetime.now(timezone.utc).isoformat(),
        files={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    with tarfile.open(output/'input.tar.gz','w:gz') as archive:
        for name in ['source','build','evidence','predictions.json','plan.json','binding.json']:
            archive.add(output/name,arcname=name)
    print('Frozen:',binding['frozen_at'],digest(output/'binding.json'))
    print('12 conditions x 2 observer modes x 7 processes =',len(conditions)*7)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('development',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    prepare(args.development,args.output)

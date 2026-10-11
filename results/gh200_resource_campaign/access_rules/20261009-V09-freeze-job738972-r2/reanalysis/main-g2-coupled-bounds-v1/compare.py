"""Post-hoc comparison kept separate from the sample-free bound calculation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heldout', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    binding = json.loads((args.heldout/'prediction_binding.json').read_text())
    assert binding['prediction_sha256'] == report['input_sha256']['frozen/predictions.json']
    assert binding['manifest_sha256'] == report['input_sha256']['frozen/manifest.json']
    case = report['case']
    paths = sorted((args.heldout/'samples'/case['id']).glob('plain-*.json'))
    assert len(paths) == 10
    values, hashes = [], {}
    for trial,path in enumerate(paths):
        record = json.loads(path.read_text())
        assert record['trial'] == trial and record['case'] == case['id']
        assert record['variant'] == 'plain' and record['returncode'] == 0
        raw = args.heldout/record['raw']
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == record['raw_sha256']
        with gzip.open(raw,'rt') as stream:
            setup,call,check = [json.loads(line) for line in stream]
        for name in ('m','n','k','lda','ldb','ldd','swizzle','input_mode','seed'):
            assert setup[name] == case[name]
        assert setup['gpu_uuid'] in binding['environment']['gpu']
        assert call['elapsed_us'] == record['elapsed_us'] and check['status'] == 'ok'
        values.append(call['elapsed_us'])
        for file in (path,raw):
            hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    observed = statistics.median(values)
    low, high = [report['coupled'][key]['plain_us'] for key in ('minimum','maximum')]
    result = dict(interpretation='Retrospective development only; no V09 score change or point selection.',
        case=case['id'],observed_plain_us=values,observed_plain_median_us=observed,
        conditional_plain_interval_us=[low,high],
        signed_relative_error_interval=[low/observed-1,high/observed-1],
        observed_median_inside_conditional_parameter_interval=low<=observed<=high,
        input_sha256=hashes,report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        binding_sha256=hashlib.sha256((args.heldout/'prediction_binding.json').read_bytes()).hexdigest(),
        comparison_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_frozen_status='unsupported',prediction_qualified=False,observation_qualified=False)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('observed_plain_median_us','signed_relative_error_interval','observed_median_inside_conditional_parameter_interval')},indent=2))


if __name__ == '__main__':
    main()

"""CPU regression checks for pure prediction modules and the frozen V09 r2 source."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT.parents[2] / 'results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2'


def run_python(source, program, *arguments):
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1')
    with tempfile.TemporaryDirectory() as directory:
        return subprocess.check_output([sys.executable, '-B', '-c', program, *map(str, arguments)],
                                       cwd=directory, env=env, text=True)


class ImportChecks(unittest.TestCase):
    def test_prediction_does_not_import_analysis(self):
        program = '''
import sys
import v09_model
assert not any(name.startswith('analyze_') or name.startswith('r09_') for name in sys.modules)
assert 'scipy.optimize' not in sys.modules
'''
        run_python(ROOT, program)

    def test_legacy_analysis_exports_the_same_functions(self):
        import analyze_r13_supply
        import clock_model
        import r09_input_source_clock
        import r09_v08_clock
        import supply_model
        self.assertIs(analyze_r13_supply.supply_predict, supply_model.supply_predict)
        self.assertIs(analyze_r13_supply.supply_predict_bounds, supply_model.supply_predict_bounds)
        self.assertIs(r09_input_source_clock.solve_source_envelope, clock_model.solve_source_envelope)
        self.assertIs(r09_v08_clock.duration_term, clock_model.duration_term)

    def test_prepare_copies_importable_model_dependencies(self):
        import run_v09

        def base_prepare(root, cutlass, family, rows, dual_clock):
            # CUTLASS overlay preparation is unrelated to these Python dependencies.
            (root/'source').mkdir(parents=True)
            (root/'build').mkdir()
            for name in ('run_r18.py', 'run_r18_lite.py', 'run_r19.py', 'run_b01_cache.py',
                         'analyze_r18.py', 'v06_run.py', 'v06_model.py'):
                shutil.copy2(ROOT/name, root/'source'/name)
            run_v09.write(root/'run_config.json', {})

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            inputs = directory/'inputs'
            inputs.mkdir()
            run_v09.write(inputs/'reference-cases.json', [])
            heldout = directory/'heldout.json'
            run_v09.write(heldout, [])
            root = directory/'prepared'
            with patch.object(run_v09.run_r18, 'prepare', side_effect=base_prepare):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_v09.prepare(root, heldout, inputs, directory/'unused-cutlass')
            hashes = run_v09.read(root/'source_hashes.json')
            for name in ('supply_model.py', 'output_model.py', 'clock_model.py'):
                self.assertIn('source/'+name, hashes)
                self.assertEqual(hashes['source/'+name], run_v09.common.sha(ROOT/name))
            run_python(root/'source', 'import run_v09, v09_model, supply_model, output_model, clock_model')


@unittest.skipUnless((FROZEN/'source/v09_model.py').exists(), 'frozen r2 archive not present')
class FrozenEquivalenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The old and current modules never share sys.modules. -B prevents writes
        # into the frozen source tree; all reports remain in process memory.
        program = '''
import json
from pathlib import Path
import sys
import v09_model

root = Path(sys.argv[1])
rows = [row for row in json.loads((root/'cases.json').read_text()) if row['set']=='heldout']
setups = {entry['case']:entry['setup'] for entry in json.loads((root/'freeze-setup.json').read_text())}
model = json.loads((root/'frozen/calibration.json').read_text())
predictions = {row['id']:v09_model.predict_components(row, setups[row['id']], model) for row in rows}
prediction_analysis_imports = sorted(name for name in sys.modules
                                    if name.startswith('analyze_') or name.startswith('r09_'))
import run_v09
identification = run_v09.supply_identification(root, model)
reports = {}
for row in rows:
    setup = setups[row['id']]
    prediction = predictions[row['id']]
    requests = run_v09.supply.supply_request_rows(row, setup)
    phases = {}
    for phase, key in [('first','first_supply'),('later','supply')]:
        selected = [request for request in requests if (request['j']==0)==(phase=='first')]
        values, supported = run_v09.supply.supply_predict(selected, model[key][row['config']],
                                                         prediction['frequency_ghz'])
        bounds = run_v09.supply.supply_predict_bounds(selected, model[key][row['config']],
                    prediction['frequency_ghz'], identification[key][row['config']])
        phases[phase] = dict(values=values.tolist(), supported=supported.tolist(), bounds=bounds)
    unsupported = run_v09.check_prediction_support(row, setup, model, prediction, identification)
    durations = prediction['cta_duration_ns']
    reports[row['id']] = dict(prediction=prediction, supply=phases,
        exact_critical_ctas=[i for i,value in enumerate(durations) if value==max(durations)],
        lp_unsupported_windows=unsupported,
        final_supported=prediction['all_supply_supported'] and not unsupported)
print(json.dumps(dict(identification=identification, cases=reports,
                      prediction_analysis_imports=prediction_analysis_imports), allow_nan=False))
'''
        cls.original = json.loads(run_python(FROZEN/'source', program, FROZEN))
        cls.extracted = json.loads(run_python(ROOT, program, FROZEN))

    def test_every_cta_event_duration_clock_and_tie_is_identical(self):
        self.assertEqual(self.extracted['prediction_analysis_imports'], [])
        self.assertEqual(len(self.original['cases']), 30)
        self.assertEqual(self.original['cases'].keys(), self.extracted['cases'].keys())
        for case, original in self.original['cases'].items():
            with self.subTest(case=case):
                # Full dictionaries cover every CTA event timestamp, all durations,
                # both total-time protocols, f, closure, work and tolerance ties.
                self.assertEqual(original['prediction'], self.extracted['cases'][case]['prediction'])
                self.assertEqual(original['exact_critical_ctas'], self.extracted['cases'][case]['exact_critical_ctas'])

    def test_feature_jacobian_and_lp_support_are_identical(self):
        self.assertEqual(self.original['identification'], self.extracted['identification'])
        frozen_predictions = json.loads((FROZEN/'frozen/predictions.json').read_text())['predictions']
        self.assertEqual(sum(row['final_supported'] for row in self.extracted['cases'].values()), 29)
        for case, original in self.original['cases'].items():
            with self.subTest(case=case):
                current = self.extracted['cases'][case]
                self.assertEqual(original['supply'], current['supply'])
                self.assertEqual(original['lp_unsupported_windows'], current['lp_unsupported_windows'])
                self.assertEqual(original['final_supported'], current['final_supported'])
                self.assertEqual(current['final_supported'], frozen_predictions[case]['status']=='predicted')
                if not current['final_supported']:
                    self.assertEqual(current['lp_unsupported_windows'], frozen_predictions[case]['unsupported_windows'])


if __name__ == '__main__':
    unittest.main()

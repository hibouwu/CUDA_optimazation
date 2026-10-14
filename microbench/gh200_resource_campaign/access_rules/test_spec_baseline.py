import json
import subprocess
import sys
import unittest
from pathlib import Path

from spec_baseline import PROFILES, calculate, profile_rates


class SpecBaselineTest(unittest.TestCase):
    def test_hand_calculated_small_gemm(self):
        result = calculate(2, 3, 4, 1, 1)
        self.assertEqual(result["flops"], 48)
        self.assertEqual(result["input_bytes"], 40)
        self.assertEqual(result["output_bytes"], 24)
        self.assertAlmostEqual(result["compute_us"], 0.000048)
        self.assertAlmostEqual(result["memory_us"], 0.064)
        self.assertEqual(result["ideal_us"], result["memory_us"])
        self.assertEqual(result["limiting_term"], "memory")

    def test_compute_bound_example(self):
        result = calculate(4096, 4096, 4096, 990, 4000)
        self.assertEqual(result["flops"], 137438953472)
        self.assertEqual(result["logical_bytes"], 134217728)
        self.assertAlmostEqual(result["ideal_us"], 138.82722572929292)
        self.assertEqual(result["limiting_term"], "compute")

    def test_units_and_scaling(self):
        a = calculate(64, 128, 256, 20, 100)
        b = calculate(64, 128, 256, 40, 200)
        self.assertEqual(a["ideal_us"], 2 * b["ideal_us"])
        self.assertEqual(a["flops"], b["flops"])

    def test_reject_invalid_inputs(self):
        for dim in (0, -1, True, 1.5):
            for dims in ((dim, 2, 3), (2, dim, 3), (2, 3, dim)):
                with self.subTest(dims=dims), self.assertRaises(ValueError):
                    calculate(*dims, 990, 4000)
        for rate in (0, -1, float("nan"), float("inf")):
            for rates in ((rate, 4000), (990, rate)):
                with self.subTest(rates=rates), self.assertRaises(ValueError):
                    calculate(1, 2, 3, *rates)

    def test_profiles_do_not_conflate_clock_and_product(self):
        profiles = json.loads(PROFILES.read_text())
        self.assertEqual(profile_rates(profiles["gh200_product"]), (990, 4000))
        self.assertAlmostEqual(profile_rates(profiles["c1_reference"])[0], 1070.53056)

    def test_cli_reports_assumptions_and_provenance(self):
        script = Path(__file__).with_name("spec_baseline.py")
        result = subprocess.run([sys.executable, str(script), "--m", "2", "--n", "3",
                                 "--k", "4", "--profile", "c1_reference"],
                                check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        self.assertEqual(data["profile"], "c1_reference")
        self.assertEqual(data["provenance"]["reference_sm_ghz"], 1.98)
        self.assertTrue(data["assumptions"])


if __name__ == "__main__":
    unittest.main()

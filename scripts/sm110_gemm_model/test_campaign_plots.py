from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.sm110_gemm_model.campaign_plots import generate_campaign_plots


class CampaignPlotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def generate(self, results, *, observations=None):
        source = self.root / "summary.json"
        data = {"schema_version": 1, "status": "complete", "results": results}
        if observations is not None:
            data = {"schema_version": 1, "observations": observations}
        source.write_text(json.dumps(data))
        manifest = generate_campaign_plots(source)
        self.assertEqual(manifest["source_json"], "../summary.json")
        self.assertEqual(
            manifest["generator_path"],
            "scripts/sm110_gemm_model/campaign_plots.py",
        )
        for chart in manifest["charts"]:
            ET.parse(self.root / "plots" / chart["path"])
        return manifest

    @staticmethod
    def rate_row(case_id, **extra):
        return {
            "case_id": case_id, "status": "ok", "trial_count": 10,
            "rate_per_second_median": 2.0e11,
            "rate_per_second_min": 1.9e11,
            "rate_per_second_max": 2.1e11,
            **extra,
        }

    def test_compute_generates_separate_launch_figures(self):
        rows = []
        for launch in ("single_warp_block", "full_sm_4warp_block"):
            for n in (64, 128, 256):
                rows.append(self.rate_row(
                    f"fp16_f32_m128n{n}k16_{launch}",
                    precision_id="fp16_f32", work_unit="flop"))
        manifest = self.generate(rows)
        self.assertEqual(manifest["campaign_kind"], "compute")
        self.assertEqual(manifest["chart_count"], 2)

    def test_payload_and_duplex_generate_service_curves(self):
        payload = [
            self.rate_row(f"tma_{residency}_{tile}", resource="tma.test",
                          residency=residency, tile_bytes=tile)
            for residency in ("hot_l2", "cold_hbm")
            for tile in (4096, 8192, 16384, 32768, 65536)
        ]
        self.assertEqual(self.generate(payload)["campaign_kind"], "tma_payload")
        duplex = [
            self.rate_row(f"{prefix}_duplex_r{read}_w{write}",
                          resource=f"{prefix}.duplex.r{read}_w{write}",
                          residency="cold_hbm" if prefix == "hbm" else "hot_l2")
            for prefix in ("hbm", "l2") for read, write in ((1, 4), (1, 1), (4, 1))
        ]
        self.assertEqual(self.generate(duplex)["campaign_kind"], "memory_duplex")

    def test_full_gemm_keeps_candidate_reference_and_ratio(self):
        rows = []
        for n in (1024, 2048, 4096):
            rows.append({
                "case_id": f"fp16_f32_n{n}_tc5a", "status": "ok",
                "precision_id": "fp16_f32", "backend_id": "tc5a", "n": n,
                "work_unit": "flop", "trial_count": 10,
                "custom_rate_per_second_median": 1.2e14,
                "custom_rate_per_second_min": 1.19e14,
                "custom_rate_per_second_max": 1.21e14,
                "reference_rate_per_second_median": 1.3e14,
                "ratio_of_paired_medians": 1.2 / 1.3,
            })
        manifest = self.generate(rows)
        self.assertEqual(manifest["campaign_kind"], "full_gemm")
        self.assertEqual(manifest["chart_count"], 2)

    def test_component_does_not_connect_incompatible_units(self):
        rows = [
            self.rate_row("l2_read", resource="l2.read", rate_unit="B/s"),
            self.rate_row("epilogue", resource="epilogue.nvfp4_requant",
                          rate_unit="element/s"),
        ]
        manifest = self.generate(rows)
        self.assertEqual(manifest["campaign_kind"], "component")
        svg = (self.root / "plots/component-throughput-by-contract.svg").read_text()
        self.assertIn("GB/s", svg)
        self.assertIn("Gelement/s", svg)

    def test_closure_plot_preserves_three_evidence_layers(self):
        observations = []
        for n in (1024, 2048, 4096):
            observations.append({
                "precision_id": "fp16_f32", "n": n,
                "performance_unit": "flop/s",
                "observed_median_per_second": 1.2e14,
                "reference_median_per_second": 1.3e14,
                "empirical_ideal_min_per_second": 1.25e14,
                "empirical_ideal_max_per_second": 1.28e14,
                "conditional_upper_min_per_second": 1.4e14,
                "conditional_upper_max_per_second": 2.5e14,
            })
        manifest = self.generate([], observations=observations)
        self.assertEqual(manifest["campaign_kind"], "closure")
        self.assertEqual(manifest["chart_count"], 2)
        svg = (self.root / "plots/closure-observed-envelope-upper.svg").read_text()
        self.assertIn("observed", svg)
        self.assertIn("empirical min", svg)
        self.assertIn("upper max", svg)
        ratio_svg = (self.root / "plots/closure-efficiency-ratios.svg").read_text()
        self.assertIn("observed/reference", ratio_svg)
        self.assertIn("observed/empirical max", ratio_svg)

    def test_static_summary_writes_manifest_without_fake_rate_plot(self):
        manifest = self.generate([{"case_id": "x", "status": "static_ok"}])
        self.assertEqual(manifest["chart_count"], 0)
        index = (self.root / "plots/index.md").read_text()
        self.assertIn("not runtime performance evidence", index)

    def test_incomplete_closure_draws_evidence_gaps_instead_of_inventing_values(self):
        observation = {
            "precision_id": "tf32_f32", "n": 1024,
            "performance_unit": "flop/s",
            "observed_median_per_second": 8.0e13,
            "reference_median_per_second": 9.0e13,
            "empirical_ideal_min_per_second": None,
            "empirical_ideal_max_per_second": None,
            "conditional_upper_min_per_second": None,
            "conditional_upper_max_per_second": None,
        }
        manifest = self.generate([], observations=[observation])
        self.assertEqual(manifest["chart_count"], 2)
        svg = (self.root / "plots/closure-observed-envelope-upper.svg").read_text()
        self.assertIn("observed", svg)
        self.assertNotIn("empirical max</text>", svg)

    def test_all_runtime_runners_generate_plots_before_completion(self):
        repo = Path(__file__).resolve().parents[2]
        runner_paths = (
            "microbench/sm110_gemm_campaign/run_compute_campaign.py",
            "microbench/sm110_gemm_component_campaign/run_component_campaign.py",
            "microbench/sm110_tma_payload_campaign/run_tma_payload_campaign.py",
            "microbench/sm110_memory_duplex_campaign/run_memory_duplex_campaign.py",
            "microbench/sm110_full_gemm_campaign/run_full_gemm_campaign.py",
        )
        for relative_path in runner_paths:
            text = (repo / relative_path).read_text()
            call = text.index("generate_campaign_plots(summary_path)")
            completion = text.index("complete_marker.write_text", call)
            self.assertLess(call, completion, relative_path)


if __name__ == "__main__":
    unittest.main()

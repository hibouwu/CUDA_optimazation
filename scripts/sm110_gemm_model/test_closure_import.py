from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from microbench.sm110_full_gemm_campaign.run_full_gemm_campaign import (
    CASES as FULL_CASES,
)
from microbench.sm110_gemm_campaign.run_compute_campaign import (
    make_manifest as make_compute_manifest,
)
from microbench.sm110_gemm_component_campaign.run_component_campaign import (
    CASES as COMPONENT_CASES,
)
from scripts.sm110_gemm_model.closure_import import (
    ClosurePaths,
    _audit_platform,
    _parse_counter_tsv,
    capacities_from_component,
    capacities_from_compute,
    observations_from_full,
    import_closure,
    reference_denominators_from_manifest,
)
from scripts.sm110_gemm_model.io import capacities_from_rows
from scripts.sm110_gemm_model.model import ModelError, precision_specs


COMMIT = "1" * 40
SUITE = "thor-test"
ROOT = Path(__file__).resolve().parents[2]
FULL_REFERENCES = {
    "fp16_f32": "cublas_tc",
    "bf16_f32": "cublas_bf16_gemmex",
    "tf32_f32": "cublas_tf32_gemmex",
    "e4m3_f32": "fp8_q8_cublaslt_matmul",
    "s8_s32": "int8_q19_cublas_gemmex",
}


class ClosureConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = ClosurePaths(self.root, SUITE)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_compute_selection_emits_all_twelve_precision_capacities(self) -> None:
        manifest = make_compute_manifest()
        selected = [
            row for row in manifest
            if row["launch"] == "full_sm_4warp_block"
            and row["m"] == 128 and row["n"] == 256
        ]
        results = []
        for index, row in enumerate(selected, 1):
            precision = row["precision"]
            results.append({
                "case_id": row["case_id"],
                "precision_id": precision["precision_id"],
                "work_unit": precision["work_unit"],
                "trial_count": 10,
                "rate_per_second_median": float(index),
                "ncu": {"selected": True},
            })
        capacities = capacities_from_compute(
            {"results": results},
            {"manifest": manifest},
            paths=self.paths,
            qualification="closure_qualified",
        )
        self.assertEqual(len(capacities), len(precision_specs()))
        self.assertEqual(
            {row.resource for row in capacities},
            {row.compute_resource for row in precision_specs().values()},
        )
        self.assertTrue(all(row.trial_count == 10 for row in capacities))
        self.assertTrue(all(any("profile.ncu-rep" in artifact
                                for artifact in row.artifact_paths)
                            for row in capacities))

    def test_component_mapping_preserves_byte_and_element_units(self) -> None:
        results = [{
            "case_id": row["id"],
            "resource": row["resource"],
            "rate_unit": ("element/s" if row["resource"].startswith("epilogue.")
                          else "B/s"),
            "source_path": "source.cu",
            "trial_count": 10,
            "rate_per_second_median": float(index),
        } for index, row in enumerate(COMPONENT_CASES, 1)]
        capacities = capacities_from_component(
            {"results": results},
            {"cases": COMPONENT_CASES},
            paths=self.paths,
            qualification="closure_qualified",
        )
        self.assertEqual(len(capacities), 9)
        resources = {row.resource for row in capacities}
        self.assertEqual(resources, {
            "tma.l2", "tma.hbm", "tmem.readback",
            "epilogue.nvfp4_requant",
        })
        epilogues = [row for row in capacities
                     if row.resource == "epilogue.nvfp4_requant"]
        self.assertTrue(all(row.work_unit == "element" for row in epilogues))
        self.assertTrue(all(row.work_unit == "byte" for row in capacities
                            if row not in epilogues))

    def test_full_gemm_conversion_emits_fifteen_same_precision_pairs(self) -> None:
        results = []
        for index, case in enumerate(FULL_CASES, 1):
            reference = float(index) * 100.0
            custom = reference * 0.9
            results.append({
                "case_id": case["id"],
                "precision_id": case["precision_id"],
                "backend_id": case["backend_id"],
                "n": case["n"],
                "work_unit": case["work_unit"],
                "trial_count": 10,
                "custom_rate_per_second_median": custom,
                "custom_rate_per_second_min": custom * 0.99,
                "custom_rate_per_second_max": custom * 1.01,
                "reference_rate_per_second_median": reference,
                "ratio_of_paired_medians": custom / reference,
                "sass_path": f"build/{case['binary']}.sass.txt",
            })
        observations = observations_from_full(
            {"results": results}, references=FULL_REFERENCES,
            paths=self.paths,
            qualification="closure_qualified")
        self.assertEqual(len(observations), 15)
        self.assertTrue(all(
            row.performance_reference_relation == "same_precision"
            for row in observations))
        s8_rows = [row for row in observations if row.precision_id == "s8_s32"]
        self.assertTrue(all(row.performance_unit == "operation/s"
                            for row in s8_rows))
        self.assertTrue(all(row.ratio_of_paired_medians == 0.9
                            for row in observations))
        s8_reference = {row.reference for row in s8_rows}
        self.assertEqual(s8_reference, {"int8_q19_cublas_gemmex"})

    def test_reference_names_come_from_same_precision_support_contract(self) -> None:
        manifest = json.loads(Path(
            "microbench/sm110_full_gemm_campaign/support_manifest.json"
        ).read_text())
        self.assertEqual(
            reference_denominators_from_manifest(manifest), FULL_REFERENCES)

    def test_capacity_json_round_trip_preserves_evidence_enum(self) -> None:
        row = capacities_from_compute(
            {"results": [{
                "case_id": entry["case_id"],
                "precision_id": entry["precision"]["precision_id"],
                "work_unit": entry["precision"]["work_unit"],
                "trial_count": 10,
                "rate_per_second_median": 1.0,
                "ncu": {"selected": True},
            } for entry in make_compute_manifest()
             if entry["launch"] == "full_sm_4warp_block"
             and entry["m"] == 128 and entry["n"] == 256]},
            {"manifest": make_compute_manifest()},
            paths=self.paths,
            qualification="closure_qualified",
        )[0]
        restored = capacities_from_rows([row.to_dict()])[0]
        self.assertEqual(restored, row)

    def write_json(self, path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    def create_common_run_files(
        self, run_dir: Path, summary: dict[str, object], spec: dict[str, object]
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        environment = {"git_head": {"returncode": 0, "output": COMMIT + "\n"}}
        self.write_json(run_dir / "environment.json", environment)
        (run_dir / "environment_snapshots.jsonl").write_text(
            json.dumps(environment) + "\n")
        self.write_json(run_dir / "run_spec.json", spec)
        self.write_json(run_dir / "summary.json", summary)
        (run_dir / "COMPLETE").write_text("complete\n")

    def test_complete_synthetic_tree_imports_end_to_end(self) -> None:
        self.paths.suite.mkdir(parents=True)
        self.write_json(self.paths.suite / "run_contract.json", {
            "schema_version": 1,
            "suite_id": SUITE,
            "expected_branch": "codex/thor-sm110-gemm-bounds-v2",
            "expected_commit": COMMIT,
            "ncu_required": True,
            "created_at_utc": "test",
        })
        (self.paths.suite / "preflight.txt").write_text(
            f"{COMMIT}\nNV Power Mode: MAXN\n"
            "min_freq=1575000000\nmax_freq=1575000000\n"
            "CurrentFreq=1575000000\ngovernor=performance\n")
        (self.paths.suite / "oc_before.tsv").write_text("oc3\t7\n")
        (self.paths.suite / "oc_after.tsv").write_text("oc3\t9\n")
        (self.paths.suite / "suite_launcher.log").write_text("SUITE_COMPLETE\n")
        self.write_json(self.paths.epilogue / "summary.json", {
            "schema_version": 3,
            "expected_commit": COMMIT,
            "pass": True,
            "profiles": [{
                "profile_id": profile,
                "returncode": 0,
                "timed_out": False,
                "termination_failed": False,
                "fields": {"value_mismatches": "0", "scale_mismatches": "0"},
            } for profile in (
                "single_cta_smoke", "full_gpu_smoke_bps1",
                "production_shape_bps1")],
        })

        manifest = make_compute_manifest()
        selected = [row for row in manifest
                    if row["launch"] == "full_sm_4warp_block"
                    and row["m"] == 128 and row["n"] == 256]
        compute_results = []
        for index, entry in enumerate(selected, 1):
            case_id = entry["case_id"]
            result = {
                "case_id": case_id,
                "precision_id": entry["precision"]["precision_id"],
                "work_unit": entry["precision"]["work_unit"],
                "trial_count": 10,
                "rate_per_second_median": float(index),
                "ncu": {"selected": True},
            }
            compute_results.append(result)
            case_dir = self.paths.compute / "cases" / case_id
            self.write_json(case_dir / "result.json", result)
            (case_dir / "trials.jsonl").write_text("{}\n")
            (case_dir / "sass.txt").write_text("SASS\n")
            (case_dir / "ncu").mkdir(parents=True)
            (case_dir / "ncu/profile.ncu-rep").write_text("NCU\n")
            (case_dir / "ncu/profile.log").write_text("NCU\n")
        self.create_common_run_files(
            self.paths.compute,
            {"results": compute_results},
            {"manifest": manifest},
        )

        (self.root / "source.cu").write_text("source\n")
        component_results = []
        for index, case in enumerate(COMPONENT_CASES, 1):
            result = {
                "case_id": case["id"],
                "resource": case["resource"],
                "rate_unit": ("element/s" if case["resource"].startswith("epilogue.")
                              else "B/s"),
                "source_path": "source.cu",
                "trial_count": 10,
                "rate_per_second_median": float(index),
            }
            component_results.append(result)
            case_dir = self.paths.component / "cases" / case["id"]
            self.write_json(case_dir / "result.json", result)
            (case_dir / "trials.jsonl").write_text("{}\n")
            sass = self.paths.component / f"build/{case['binary']}.sass.txt"
            sass.parent.mkdir(parents=True, exist_ok=True)
            sass.write_text("SASS\n")
        self.create_common_run_files(
            self.paths.component,
            {"results": component_results},
            {"cases": COMPONENT_CASES},
        )

        full_results = []
        for index, case in enumerate(FULL_CASES, 1):
            reference = float(index) * 100.0
            custom = reference * 0.9
            result = {
                "case_id": case["id"],
                "precision_id": case["precision_id"],
                "backend_id": case["backend_id"],
                "n": case["n"],
                "work_unit": case["work_unit"],
                "trial_count": 10,
                "custom_rate_per_second_median": custom,
                "custom_rate_per_second_min": custom * 0.99,
                "custom_rate_per_second_max": custom * 1.01,
                "reference_rate_per_second_median": reference,
                "ratio_of_paired_medians": custom / reference,
                "sass_path": f"build/{case['binary']}.sass.txt",
            }
            full_results.append(result)
            case_dir = self.paths.full / "cases" / case["id"]
            self.write_json(case_dir / "result.json", result)
            (case_dir / "trials.jsonl").write_text("{}\n")
            for trial in range(1, 11):
                trial_dir = case_dir / f"trial_{trial:02d}"
                trial_dir.mkdir(parents=True)
                (trial_dir / "stdout.log").write_text("PASS\n")
            sass = self.paths.full / str(result["sass_path"])
            sass.parent.mkdir(parents=True, exist_ok=True)
            sass.write_text("SASS\n")
        support_manifest = {
            "precisions": [{
                "precision_id": precision_id,
                "status": "ready_for_closure_campaign",
                "performance_denominator": {
                    "backend_id": backend_id,
                    "same_precision": True,
                    "status": "ready",
                },
            } for precision_id, backend_id in FULL_REFERENCES.items()]
        }
        self.write_json(self.root / "support_manifest.json", support_manifest)
        self.create_common_run_files(
            self.paths.full,
            {"results": full_results, "ncu_requested": True},
            {"cases": FULL_CASES, "support_manifest": "support_manifest.json"},
        )

        with mock.patch(
            "scripts.sm110_gemm_model.closure_import._run_auditor",
            return_value={"pass": True},
        ):
            imported = import_closure(
                repo_root=self.root, suite_id=SUITE, expected_commit=COMMIT)
        self.assertTrue(imported["closure_qualified"])
        self.assertEqual(len(imported["capacities"]), 21)
        self.assertEqual(len(imported["observed_best"]), 15)
        self.assertTrue(imported["model_input_audit"]["pass"])
        self.assertTrue(
            imported["platform_evidence"]["overcurrent_events_observed"])
        self.assertIn(
            "overcurrent_events_observed",
            {row["code"] for row in imported["model_input_audit"]["findings"]},
        )


class PlatformEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = ClosurePaths(self.root, SUITE)
        self.paths.suite.mkdir(parents=True)
        (self.paths.suite / "run_contract.json").write_text(json.dumps({
            "schema_version": 1,
            "suite_id": SUITE,
            "expected_branch": "codex/thor-sm110-gemm-bounds-v2",
            "expected_commit": COMMIT,
            "ncu_required": True,
            "created_at_utc": "test",
        }))
        (self.paths.suite / "preflight.txt").write_text(
            f"{COMMIT}\nNV Power Mode: MAXN\n"
            "min_freq=1575000000\nmax_freq=1575000000\n"
            "CurrentFreq=1575000000\ngovernor=performance\n")
        (self.paths.suite / "suite_launcher.log").write_text("SUITE_COMPLETE\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_counters(self, before: int, after: int) -> None:
        (self.paths.suite / "oc_before.tsv").write_text(f"oc3\t{before}\n")
        (self.paths.suite / "oc_after.tsv").write_text(f"oc3\t{after}\n")

    def test_stable_counters_allow_closure_qualification(self) -> None:
        self.write_counters(10, 10)
        deltas = _audit_platform(self.paths, COMMIT)
        self.assertEqual(deltas, {"oc3": 0})

    def test_incremented_counter_is_preserved_as_telemetry(self) -> None:
        self.write_counters(10, 11)
        deltas = _audit_platform(self.paths, COMMIT)
        self.assertEqual(deltas, {"oc3": 1})

    def test_counter_reset_is_rejected(self) -> None:
        self.write_counters(10, 9)
        with self.assertRaisesRegex(ModelError, "reset"):
            _audit_platform(self.paths, COMMIT)

    def test_counter_parser_is_fail_closed(self) -> None:
        bad = self.paths.suite / "bad.tsv"
        bad.write_text("oc3 10\n")
        with self.assertRaisesRegex(ModelError, "invalid counter row"):
            _parse_counter_tsv(bad)


class CommittedRunbookTest(unittest.TestCase):
    def test_wrapper_derives_commit_from_head_and_runbook_uses_wrapper(self) -> None:
        wrapper = (ROOT / "microbench/sm110_closure_campaign.sh").read_text()
        runbook = (ROOT / "Docs/blackwell_tensorcore/THOR_CLOSURE_RUNBOOK.md").read_text()
        self.assertIn("expected_commit=$(git rev-parse HEAD)", wrapper)
        self.assertNotRegex(
            wrapper,
            re.compile(r"(?<![A-Za-z0-9])[0-9a-f]{40}(?![A-Za-z0-9])"),
        )
        self.assertIn(
            'sm110_closure_campaign.sh start "$SUITE_ID"', runbook)
        self.assertIn(
            'sm110_closure_campaign.sh finish "$SUITE_ID"', runbook)


if __name__ == "__main__":
    unittest.main()

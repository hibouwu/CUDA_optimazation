from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from collections import Counter
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
    validate_fields as validate_component_fields,
)
from microbench.sm110_gemm_component_campaign.audit_campaign import (
    EXPECTED_CASE_RESOURCES,
)
from scripts.sm110_gemm_model.closure_import import (
    ClosurePaths,
    _audit_commit_environment,
    _audit_platform,
    _parse_counter_tsv,
    capacities_from_component,
    capacities_from_compute,
    observations_from_full,
    import_closure,
    import_composite_closure,
    reference_denominators_from_manifest,
)
from scripts.sm110_gemm_model.io import (
    capacities_from_rows,
    load_capacities,
    load_schedules,
    observations_from_rows,
)
from scripts.sm110_gemm_model.model import (
    Capacity,
    EvidenceKind,
    Hardware,
    LayerResult,
    ModelError,
    Schedule,
    WorkloadEnvelope,
    precision_specs,
)
from scripts.sm110_gemm_model.observations import ObservedBest
from scripts.sm110_gemm_model.closure_report import (
    build_closure_analysis,
    render_closure_markdown,
)
from scripts.sm110_gemm_model.coverage import campaign_measurement_coverage


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

    def test_compute_selection_emits_three_shapes_for_all_twelve_precisions(self) -> None:
        manifest = make_compute_manifest()
        selected = [
            row for row in manifest
            if row["launch"] == "full_sm_4warp_block"
            and row["m"] == 128 and row["n"] in {64, 128, 256}
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
                "ncu": {"selected": row["n"] == 256},
            })
        capacities = capacities_from_compute(
            {"results": results},
            {"manifest": manifest},
            paths=self.paths,
            qualification="closure_qualified",
        )
        self.assertEqual(len(capacities), 3 * len(precision_specs()))
        self.assertEqual(
            {row.resource for row in capacities},
            {
                f"{row.compute_resource}.m128n{n}"
                for row in precision_specs().values()
                for n in (64, 128, 256)
            },
        )
        self.assertEqual(
            len({row.capacity_id for row in capacities}), len(capacities))
        self.assertTrue(all(row.trial_count == 10 for row in capacities))
        self.assertTrue(all(
            any("profile.ncu-rep" in artifact for artifact in row.artifact_paths)
            == row.resource.endswith("n256")
            for row in capacities
        ))

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
            {"cases": COMPONENT_CASES, "expected_sm_count": 20},
            paths=self.paths,
            qualification="closure_qualified",
        )
        self.assertEqual(len(capacities), 18)
        resources = {row.resource for row in capacities}
        self.assertEqual(resources, {
            "tma.smem_ingress.per_sm", "tma.hbm", "hbm.read", "hbm.write",
            "tma.smem_ingress.diagnostic.serial32k.per_sm",
            "tma.smem_ingress.per_sm.inflight4",
            "tma.hbm.diagnostic.serial32k",
            "tma.hbm.inflight4",
            "l2.read", "l2.write", "tmem.scale_ingress", "tmem.readback",
            "tmem.readback.x8.warps1",
            "tmem.readback.x8.warps4",
            "tmem.readback.x16.warps1",
            "epilogue.nvfp4_requant",
        })
        epilogues = [row for row in capacities
                     if row.resource == "epilogue.nvfp4_requant"]
        self.assertTrue(all(row.work_unit == "element" for row in epilogues))
        self.assertTrue(all(row.work_unit == "byte" for row in capacities
                            if row not in epilogues))
        tma_per_sm = [
            row for row in capacities
            if row.resource == "tma.smem_ingress.per_sm"
        ]
        self.assertEqual(len(tma_per_sm), 1)
        for row in tma_per_sm:
            case_index = next(
                index for index, case in enumerate(COMPONENT_CASES, 1)
                if row.capacity_id.endswith(str(case["id"]))
            )
            self.assertEqual(row.rate_per_second, float(case_index))
            self.assertIsNone(row.original_value)

    def test_component_manifest_and_auditor_freeze_the_same_case_matrix(self) -> None:
        self.assertEqual(
            {str(row["id"]): str(row["resource"]) for row in COMPONENT_CASES},
            EXPECTED_CASE_RESOURCES,
        )

    def test_per_sm_tma_import_rejects_a_concurrent_full_grid_probe(self) -> None:
        cases = json.loads(json.dumps(COMPONENT_CASES))
        target = next(
            row for row in cases
            if row["id"] == "tma_l2_hit_tc5a_ab_inflight8")
        args = target["args"]
        args[args.index("--blocks") + 1] = "20"
        result = {
            "case_id": target["id"],
            "resource": target["resource"],
            "rate_unit": "B/s",
            "source_path": "source.cu",
            "trial_count": 10,
            "rate_per_second_median": 100.0,
        }
        with self.assertRaisesRegex(
                ModelError, "isolated one-CTA exact tc5a L2-hit case"):
            capacities_from_component(
                {"results": [result]},
                {"cases": cases, "expected_sm_count": 20},
                paths=self.paths,
                qualification="closure_qualified",
            )

    def test_tma_hbm_import_rejects_a_non_tc5a_pipeline(self) -> None:
        cases = json.loads(json.dumps(COMPONENT_CASES))
        target = next(
            row for row in cases
            if row["id"] == "tma_dram_stream_tc5a_ab_inflight8")
        args = target["args"]
        args[args.index("--pattern") + 1] = "uniform"
        result = {
            "case_id": target["id"],
            "resource": target["resource"],
            "rate_unit": "B/s",
            "source_path": "source.cu",
            "trial_count": 10,
            "rate_per_second_median": 100.0,
        }
        with self.assertRaisesRegex(
                ModelError, "full-grid exact tc5a DRAM-stream case"):
            capacities_from_component(
                {"results": [result]},
                {"cases": cases, "expected_sm_count": 20},
                paths=self.paths,
                qualification="closure_qualified",
            )

    def test_tma_runtime_fields_freeze_serial_and_inflight_contracts(self) -> None:
        by_id = {str(row["id"]): row for row in COMPONENT_CASES}
        for case_id, expected_tile, expected_inflight, expected_slots in (
            ("tma_l2_hit_32k", 32768, 1, 4),
            ("tma_dram_stream_32k", 32768, 1, 4),
            ("tma_l2_hit_32k_inflight4", 32768, 4, 4),
            ("tma_dram_stream_32k_inflight4", 32768, 4, 4),
            ("tma_l2_hit_tc5a_ab_inflight8", 16384, 8, 8),
            ("tma_dram_stream_tc5a_ab_inflight8", 16384, 8, 8),
        ):
            per_sm = case_id.startswith("tma_l2_hit")
            args = list(by_id[case_id]["args"])
            expected_warmup = args[args.index("--warmup-iters") + 1]
            expected_threads = args[args.index("--threads") + 1]
            expected_pattern = (
                args[args.index("--pattern") + 1]
                if "--pattern" in args else "uniform")
            expected_stage_count = 4 if expected_pattern == "tc5a-ab" else expected_slots
            exact = expected_pattern == "tc5a-ab"
            mode = args[args.index("--mode") + 1]
            backing_bytes = int(args[args.index("--bytes") + 1])
            expected_blocks = 1 if per_sm else 20
            if exact:
                total_tiles = max(1, backing_bytes // 49152)
                working_set = total_tiles * 49152
                allocation = total_tiles * 384 * 2048 * 2
            else:
                total_tiles = max(
                    backing_bytes // expected_tile, expected_blocks)
                if mode == "dram-stream":
                    total_tiles = max(
                        total_tiles,
                        expected_blocks * (
                            int(expected_warmup)
                            + int(args[args.index("--iters") + 1])),
                    )
                working_set = total_tiles * expected_tile
                allocation = working_set
            fields = {
                "sm_count": "20",
                "mode": str(mode),
                "unique_smid_count": "1" if per_sm else "20",
                "requested_bytes": "4096",
                "globaltimer_elapsed_ns": "8",
                "globaltimer_gbytes_per_second": "512.0",
                "slots": str(expected_slots),
                "inflight": str(expected_inflight),
                "tile_bytes": str(expected_tile),
                "pattern": str(expected_pattern),
                "stage_count": str(expected_stage_count),
                "requests_per_stage": (
                    "2" if expected_pattern == "tc5a-ab" else "1"),
                "barrier_count": str(expected_stage_count),
                "tensor_map": (
                    "2d-sw128" if expected_pattern == "tc5a-ab"
                    else "3d-none"),
                "row_stride_elements": "2048" if exact else "0",
                "smem_bytes": str(4 * 49152 if exact
                                  else expected_tile * expected_slots),
                "preferred_smem_carveout": "max" if exact else "default",
                "total_tiles": str(total_tiles),
                "total_tiles_b": str(total_tiles),
                "working_set_bytes": str(working_set),
                "allocation_bytes": str(allocation),
                "warmup_iters": str(expected_warmup),
                "requested_blocks": "1" if per_sm else "0",
                "blocks_per_sm": "1",
                "threads": str(expected_threads),
                "blocks": "1" if per_sm else "20",
            }
            with self.subTest(case_id=case_id):
                self.assertEqual(
                    validate_component_fields(by_id[case_id], fields),
                    512e9,
                )

    def test_memory_runtime_fields_match_all_four_frozen_case_ids(self) -> None:
        by_id = {str(row["id"]): row for row in COMPONENT_CASES}
        for residency in ("hbm", "l2"):
            for direction in ("read", "write"):
                case_id = f"{residency}_{direction}_aggregate"
                requested = 4096
                elapsed = 8
                rate = requested * 1e9 / elapsed
                fields = {
                    "case_id": case_id,
                    "sm_count": "20",
                    "unique_smid_count": "20",
                    "residency": residency,
                    "direction": direction,
                    "blocks_per_sm": "4",
                    "working_set_bytes": str(
                        (16 if residency == "l2" else 256) << 20),
                    "requested_bytes": str(requested),
                    "globaltimer_elapsed_ns": str(elapsed),
                    "bytes_per_second": str(rate),
                }
                with self.subTest(case_id=case_id):
                    self.assertEqual(
                        validate_component_fields(by_id[case_id], fields),
                        rate,
                    )

    def test_scale_runtime_fields_include_nonoverlapping_destination_contract(self) -> None:
        case = next(
            row for row in COMPONENT_CASES
            if row["id"] == "tmem_scale_ingress_32x128b_warpx4")
        issued = 4096
        elapsed = 8
        rate = issued * 1e9 / elapsed
        fields = {
            "case_id": str(case["id"]),
            "sm_count": "20",
            "unique_smid_count": "20",
            "source_bytes_per_instruction": "512",
            "multicast_partitions": "4",
            "destination_slots": "32",
            "destination_columns_per_copy": "4",
            "value_mismatches": "0",
            "issued_source_bytes": str(issued),
            "globaltimer_elapsed_ns": str(elapsed),
            "bytes_per_second": str(rate),
        }
        self.assertEqual(validate_component_fields(case, fields), rate)

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
             and entry["m"] == 128 and entry["n"] in {64, 128, 256}]},
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
        environment = {
            "git_head": {"returncode": 0, "output": COMMIT + "\n"},
            "git_status": {"returncode": 0, "output": ""},
        }
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
            "=== git ===\n"
            "codex/thor-sm110-gemm-bounds-v2\n"
            f"{COMMIT}\n"
            "=== nvpmodel ===\nNV Power Mode: MAXN\n"
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
                    and row["m"] == 128 and row["n"] in {64, 128, 256}]
        compute_results = []
        for index, entry in enumerate(selected, 1):
            case_id = entry["case_id"]
            result = {
                "case_id": case_id,
                "precision_id": entry["precision"]["precision_id"],
                "work_unit": entry["precision"]["work_unit"],
                "trial_count": 10,
                "rate_per_second_median": float(index),
                "ncu": {"selected": entry["n"] == 256},
            }
            compute_results.append(result)
            case_dir = self.paths.compute / "cases" / case_id
            self.write_json(case_dir / "result.json", result)
            (case_dir / "trials.jsonl").write_text("{}\n")
            (case_dir / "sass.txt").write_text("SASS\n")
            if entry["n"] == 256:
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
            {"cases": COMPONENT_CASES, "expected_sm_count": 20},
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
        self.assertEqual(len(imported["capacities"]), 54)
        resource_counts = Counter(
            row["resource"] for row in imported["capacities"])
        self.assertEqual(resource_counts["tmem.readback"], 1)
        self.assertEqual(resource_counts["tmem.readback.x8.warps1"], 1)
        self.assertEqual(resource_counts["tmem.readback.x8.warps4"], 1)
        self.assertEqual(resource_counts["tmem.readback.x16.warps1"], 1)
        self.assertEqual(resource_counts["tmem.scale_ingress"], 1)
        self.assertEqual(len(imported["observed_best"]), 15)
        self.assertTrue(imported["model_input_audit"]["pass"])
        self.assertEqual(
            imported["campaign_contract"]["full_gemm_observation_count"], 15)
        self.assertEqual(imported["campaign_contract"]["compute_case_count"], 36)
        self.assertTrue(
            imported["platform_evidence"]["overcurrent_events_observed"])
        self.assertIn(
            "overcurrent_events_observed",
            {row["code"] for row in imported["model_input_audit"]["findings"]},
        )
        analysis = build_closure_analysis(
            metadata=imported,
            base_capacities=load_capacities(
                ROOT / "scripts/sm110_gemm_model/profiles/capacities.json"),
            closure_capacities=capacities_from_rows(imported["capacities"]),
            observations=observations_from_rows(imported["observed_best"]),
            hardware=Hardware("thor", 20, 1.575e9),
            schedules=load_schedules(
                ROOT / "scripts/sm110_gemm_model/examples/schedules.json"),
        )
        self.assertFalse(any(
            finding["severity"] == "error"
            for finding in analysis["findings"]
        ))
        self.assertTrue(analysis["pass"])
        self.assertEqual(analysis["capacity_count"], 54)
        self.assertEqual(analysis["observation_count"], 15)
        self.assertTrue(
            analysis["campaign_measurement_coverage"]
                    ["all_campaign_measurements_closed"])
        self.assertTrue(analysis["all_common_resources_closed"])
        self.assertFalse(analysis["all_precisions_closed"])
        campaign = campaign_measurement_coverage(
            capacities_from_rows(imported["capacities"]),
            observations_from_rows(imported["observed_best"]),
        )
        self.assertTrue(campaign["all_campaign_measurements_closed"])
        self.assertTrue(all(
            all(shape_status.values())
            for shape_status in campaign["compute_shape_closed"].values()
        ))
        incomplete_capacities = [
            row for row in capacities_from_rows(imported["capacities"])
            if row.resource != "tensor.bf16.m128n64"
        ]
        incomplete = campaign_measurement_coverage(
            incomplete_capacities,
            observations_from_rows(imported["observed_best"]),
        )
        self.assertFalse(incomplete["precision_closed"]["bf16_f32"])
        self.assertFalse(incomplete["all_campaign_measurements_closed"])

        tampered_rows = [dict(row) for row in imported["capacities"]]
        epilogue_rows = [
            row for row in tampered_rows
            if row["resource"] == "epilogue.nvfp4_requant"
        ]
        epilogue_rows[0]["capacity_id"] = (
            f"{SUITE}.component.nvfp4_requant_4096x1024_duplicate")
        tampered_analysis = build_closure_analysis(
            metadata=imported,
            base_capacities=load_capacities(
                ROOT / "scripts/sm110_gemm_model/profiles/capacities.json"),
            closure_capacities=capacities_from_rows(tampered_rows),
            observations=observations_from_rows(imported["observed_best"]),
            hardware=Hardware("thor", 20, 1.575e9),
            schedules=load_schedules(
                ROOT / "scripts/sm110_gemm_model/examples/schedules.json"),
        )
        self.assertIn(
            "closure_capacity_matrix_incomplete",
            {row["code"] for row in tampered_analysis["findings"]},
        )

        supplement_id = "thor-component-supplement"
        supplement_paths = ClosurePaths(self.root, supplement_id)
        shutil.copytree(self.paths.component, supplement_paths.component)
        supplement_paths.suite.mkdir(parents=True)
        self.write_json(supplement_paths.suite / "run_contract.json", {
            "schema_version": 2,
            "kind": "component_supplement",
            "supplement_id": supplement_id,
            "component_run_id": f"{supplement_id}-components",
            "expected_branch": "codex/thor-sm110-gemm-bounds-v2",
            "expected_commit": COMMIT,
            "base_suite_id": SUITE,
            "base_expected_commit": COMMIT,
            "created_at_utc": "test",
        })
        (supplement_paths.suite / "preflight.txt").write_text(
            "=== git ===\n"
            "codex/thor-sm110-gemm-bounds-v2\n"
            f"{COMMIT}\n"
            "=== nvpmodel ===\nNV Power Mode: MAXN\n"
            "min_freq=1575000000\nmax_freq=1575000000\n"
            "CurrentFreq=1575000000\ngovernor=performance\n")
        (supplement_paths.suite / "oc_before.tsv").write_text("oc3\t9\n")
        (supplement_paths.suite / "oc_after.tsv").write_text("oc3\t9\n")
        (supplement_paths.suite / "supplement_launcher.log").write_text(
            "COMPONENT_SUPPLEMENT_COMPLETE\n")
        with mock.patch(
            "scripts.sm110_gemm_model.closure_import._run_auditor",
            return_value={"pass": True},
        ):
            composite = import_composite_closure(
                repo_root=self.root,
                composite_id=supplement_id,
                base_suite_id=SUITE,
                base_expected_commit=COMMIT,
                component_expected_commit=COMMIT,
            )
        self.assertTrue(composite["closure_qualified"])
        self.assertEqual(composite["composition"],
                         "base_compute_full_plus_component_supplement")
        self.assertEqual(len(composite["capacities"]), 54)
        self.assertEqual(len(composite["observed_best"]), 15)
        self.assertTrue(all(
            row["capacity_id"].startswith(f"{supplement_id}.")
            for row in composite["capacities"]))
        compute_sources = {
            row["source_id"] for row in composite["capacities"]
            if ".compute." in row["capacity_id"]
        }
        component_sources = {
            row["source_id"] for row in composite["capacities"]
            if ".component." in row["capacity_id"]
        }
        self.assertEqual(compute_sources, {SUITE})
        self.assertEqual(component_sources, {supplement_id})
        self.assertTrue(all(
            row["run_id"] == SUITE for row in composite["observed_best"]))
        self.assertEqual(
            set(composite["platform_evidence"]["overcurrent_deltas"]),
            {SUITE, supplement_id},
        )
        composite_analysis = build_closure_analysis(
            metadata=composite,
            base_capacities=load_capacities(
                ROOT / "scripts/sm110_gemm_model/profiles/capacities.json"),
            closure_capacities=capacities_from_rows(composite["capacities"]),
            observations=observations_from_rows(composite["observed_best"]),
            hardware=Hardware("thor", 20, 1.575e9),
            schedules=load_schedules(
                ROOT / "scripts/sm110_gemm_model/examples/schedules.json"),
        )
        self.assertTrue(composite_analysis["pass"])
        self.assertEqual(
            composite_analysis["campaign_sources"],
            composite["campaign_sources"],
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
            "=== git ===\n"
            "codex/thor-sm110-gemm-bounds-v2\n"
            f"{COMMIT}\n"
            "=== nvpmodel ===\nNV Power Mode: MAXN\n"
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

    def test_dirty_or_wrong_branch_preflight_is_rejected(self) -> None:
        self.write_counters(10, 10)
        (self.paths.suite / "preflight.txt").write_text(
            "=== git ===\nwrong-branch\n"
            f"{COMMIT}\n M tracked.py\n"
            "=== nvpmodel ===\nNV Power Mode: MAXN\n"
            "min_freq=1575000000\nmax_freq=1575000000\n"
            "CurrentFreq=1575000000\ngovernor=performance\n")
        with self.assertRaisesRegex(ModelError, "clean expected checkout"):
            _audit_platform(self.paths, COMMIT)

    def test_campaign_environment_requires_successful_git_probes(self) -> None:
        run_dir = self.root / "run"
        run_dir.mkdir()
        valid = {
            "git_head": {"returncode": 0, "output": f"{COMMIT}\n"},
            "git_status": {"returncode": 0, "output": "?? results/run\n"},
        }
        (run_dir / "environment.json").write_text(json.dumps(valid))
        (run_dir / "environment_snapshots.jsonl").write_text(
            json.dumps(valid) + "\n")
        _audit_commit_environment(run_dir, COMMIT)

        missing_status = dict(valid)
        missing_status.pop("git_status")
        (run_dir / "environment_snapshots.jsonl").write_text(
            json.dumps(missing_status) + "\n")
        with self.assertRaisesRegex(ModelError, "does not prove Git status"):
            _audit_commit_environment(run_dir, COMMIT)


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

    def test_component_supplement_instructions_ship_with_runner(self) -> None:
        wrapper = (
            ROOT / "microbench/sm110_component_supplement.sh").read_text()
        supervisor = (
            ROOT / "microbench/run_sm110_component_supplement.sh").read_text()
        runbook = (
            ROOT / "Docs/blackwell_tensorcore/THOR_CLOSURE_RUNBOOK.md").read_text()
        self.assertIn("expected_commit=$(git rev-parse HEAD)", wrapper)
        self.assertIn("COMPONENT_SUPPLEMENT_COMPLETE", supervisor)
        self.assertIn(
            'sm110_component_supplement.sh start \\\n  "$SUPPLEMENT_ID" "$BASE_SUITE_ID" "$BASE_EXPECTED_COMMIT"',
            runbook,
        )
        self.assertIn(
            'sm110_component_supplement.sh finish "$SUPPLEMENT_ID"',
            runbook,
        )
        self.assertIn("import-composite-closure", wrapper)


class ClosureReportTest(unittest.TestCase):
    @staticmethod
    def capacity(
        name: str, resource: str, rate: float, unit: str,
        evidence: EvidenceKind, *, qualification: str = "snapshot_only",
    ) -> Capacity:
        return Capacity(
            name, resource, rate, unit, evidence, "test", "source.json", name,
            source_url=("https://example.com/spec"
                        if evidence == EvidenceKind.SPECIFIED_UPPER else ""),
            qualification=qualification,
            trial_count=10 if qualification == "closure_qualified" else 1,
            artifact_paths=("source.json",) if qualification == "closure_qualified"
            else (),
        )

    def test_report_separates_data_split_from_residency_scenarios(self) -> None:
        base = [
            self.capacity("compute_upper", "tensor.bf16", 100e12, "flop",
                          EvidenceKind.DERIVED_UPPER),
            self.capacity("hbm_upper", "hbm.total", 200e9, "byte",
                          EvidenceKind.SPECIFIED_UPPER),
            self.capacity("l2_read_upper", "l2.read", 1e12, "byte",
                          EvidenceKind.PROFILER_MODEL_PEAK),
            self.capacity("l2_write_upper", "l2.write", 1e12, "byte",
                          EvidenceKind.PROFILER_MODEL_PEAK),
        ]
        closure = [
            self.capacity("compute", "tensor.bf16.m128n128", 80e12, "flop",
                          EvidenceKind.MEASURED_SUSTAINED,
                          qualification="closure_qualified"),
            self.capacity("hbm_read", "hbm.read", 90e9, "byte",
                          EvidenceKind.MEASURED_SUSTAINED),
            self.capacity("hbm_write", "hbm.write", 80e9, "byte",
                          EvidenceKind.MEASURED_SUSTAINED),
            self.capacity("l2_read", "l2.read", 900e9, "byte",
                          EvidenceKind.MEASURED_SUSTAINED),
            self.capacity("l2_write", "l2.write", 500e9, "byte",
                          EvidenceKind.MEASURED_SUSTAINED),
            self.capacity("tma_hbm", "tma.hbm.inflight4", 80e9, "byte",
                          EvidenceKind.MEASURED_SUSTAINED),
            self.capacity(
                "tma_per_sm", "tma.smem_ingress.per_sm.inflight4", 40e9, "byte",
                          EvidenceKind.MEASURED_SUSTAINED),
            self.capacity("tmem", "tmem.readback", 1e12, "byte",
                          EvidenceKind.MEASURED_SUSTAINED),
        ]
        observations = []
        for n in (1024, 2048, 4096):
            observations.append(ObservedBest(
                observation_id=f"bf16_n{n}",
                precision_id="bf16_f32",
                m=n, n=n, k=n,
                backend_id="candidate",
                reference="cublas_bf16_gemmex",
                performance_reference_relation="same_precision",
                trial_count=10,
                matched_count=10,
                median_per_second=1e12,
                maximum_per_second=1.01e12,
                minimum_per_second=0.99e12,
                performance_unit="flop/s",
                source_path="summary.json",
                source_locator="case",
                artifact_paths=("summary.json",),
                run_id="suite",
                reference_median_per_second=1.1e12,
                ratio_of_paired_medians=1e12 / 1.1e12,
                qualification="closure_qualified",
            ))
        analysis = build_closure_analysis(
            metadata={
                "suite_id": "suite", "expected_commit": COMMIT,
                "qualification": "closure_qualified"},
            base_capacities=base,
            closure_capacities=closure,
            observations=observations,
            hardware=Hardware("thor", 20, 1.575e9),
            schedules=[Schedule("s", 128, 128, 64, 2, tail_policy="pad")],
            require_complete_contract=False,
        )
        by_n = {row["n"]: row for row in analysis["observations"]}
        self.assertEqual(
            by_n[1024]["modeled_residencies"], ["hot_l2", "cold_hbm"])
        self.assertEqual(by_n[2048]["split"], "calibration")
        self.assertEqual(by_n[4096]["split"], "holdout")
        self.assertIn("hot_l2", by_n[4096]["residency_scenarios"])
        self.assertIn("cold_hbm", by_n[4096]["residency_scenarios"])
        self.assertIn(
            "empirical_resource_seconds",
            by_n[4096]["residency_scenarios"]["hot_l2"],
        )
        self.assertLessEqual(
            by_n[4096]["conditional_upper_min_per_second"],
            by_n[4096]["conditional_upper_max_per_second"],
        )
        self.assertFalse(any(row["severity"] == "error"
                             for row in analysis["findings"]))
        markdown = render_closure_markdown(analysis)
        self.assertIn("hot-L2", markdown)
        self.assertIn("cold-HBM", markdown)
        self.assertIn("cublas_bf16_gemmex", markdown)

    def test_report_flags_observation_above_conditional_upper(self) -> None:
        observation = ObservedBest(
            "impossible", "bf16_f32", 1024, 1024, 1024,
            "candidate", "reference", "same_precision", 10, 10,
            1e18, 1.01e18, 0.99e18, "flop/s", "summary.json")
        analysis = build_closure_analysis(
            metadata={},
            base_capacities=[self.capacity(
                "compute_upper", "tensor.bf16", 100e12, "flop",
                EvidenceKind.DERIVED_UPPER)],
            closure_capacities=[],
            observations=[observation],
            hardware=Hardware("thor", 20, 1.575e9),
            schedules=[Schedule("s", 128, 128, 64, 2, tail_policy="pad")],
            require_complete_contract=False,
        )
        self.assertIn(
            "observed_exceeds_conditional_upper",
            {row["code"] for row in analysis["findings"]},
        )

    def test_report_rejects_empirical_prediction_above_conditional_upper(self) -> None:
        observation = ObservedBest(
            "empirical-contradiction", "bf16_f32", 128, 128, 64,
            "candidate", "reference", "same_precision", 10, 10,
            50.0, 50.0, 50.0, "flop/s", "summary.json")
        contradiction = WorkloadEnvelope(
            workload_id="empirical-contradiction",
            valid_schedule_count=1,
            rejected_schedule_count=0,
            manifest_conditional_upper=LayerResult(
                "ok", 1.0, 100.0, "flop/s", ["tensor.bf16"]),
            empirical_ideal_envelope=LayerResult(
                "ok", 0.5, 200.0, "flop/s", ["synthetic-bug"]),
            conditional_schedule_id="s",
            empirical_schedule_id="s",
            rejected=[],
        )
        with mock.patch(
            "scripts.sm110_gemm_model.closure_report.evaluate_manifest",
            return_value=contradiction,
        ):
            analysis = build_closure_analysis(
                metadata={},
                base_capacities=[],
                closure_capacities=[],
                observations=[observation],
                hardware=Hardware("one-sm-test", 1, 1.0),
                schedules=[Schedule(
                    "s", 128, 128, 64, 2,
                    tail_policy="exact", fixed_seconds=0.0)],
                require_complete_contract=False,
            )
        self.assertIn(
            "empirical_exceeds_conditional_upper",
            {row["code"] for row in analysis["findings"]},
        )

    def test_report_uses_maximum_trial_for_upper_violation(self) -> None:
        observation = ObservedBest(
            "maximum-only", "bf16_f32", 128, 128, 64,
            "candidate", "reference", "same_precision", 10, 10,
            99.0, 103.0, 98.0, "flop/s", "summary.json")
        analysis = build_closure_analysis(
            metadata={},
            base_capacities=[self.capacity(
                "compute_upper", "tensor.bf16", 100.0, "flop",
                EvidenceKind.DERIVED_UPPER)],
            closure_capacities=[],
            observations=[observation],
            hardware=Hardware("thor", 20, 1.575e9),
            schedules=[Schedule(
                "s", 128, 128, 64, 2,
                tail_policy="exact", fixed_seconds=0.0)],
            require_complete_contract=False,
        )
        row = analysis["observations"][0]
        self.assertLess(row["observed_median_to_conditional_upper"], 1.0)
        self.assertGreater(row["observed_maximum_to_conditional_upper"], 1.02)
        self.assertIn(
            "observed_exceeds_conditional_upper",
            {finding["code"] for finding in analysis["findings"]},
        )

    def test_report_rejects_incomplete_closure_contract(self) -> None:
        analysis = build_closure_analysis(
            metadata={},
            base_capacities=[],
            closure_capacities=[],
            observations=[],
            hardware=Hardware("thor", 20, 1.575e9),
            schedules=[],
        )
        codes = {finding["code"] for finding in analysis["findings"]}
        self.assertIn("closure_metadata_not_qualified", codes)
        self.assertIn("closure_capacity_matrix_incomplete", codes)
        self.assertIn("full_gemm_observation_matrix_incomplete", codes)

    def test_report_materializes_observation_iterable_once(self) -> None:
        observation = ObservedBest(
            "generator", "bf16_f32", 128, 128, 64,
            "candidate", "reference", "same_precision", 10, 10,
            1.0, 1.0, 1.0, "flop/s", "summary.json")
        analysis = build_closure_analysis(
            metadata={},
            base_capacities=[self.capacity(
                "compute_upper", "tensor.bf16", 100.0, "flop",
                EvidenceKind.DERIVED_UPPER)],
            closure_capacities=[],
            observations=(row for row in [observation]),
            hardware=Hardware("thor", 20, 1.575e9),
            schedules=[Schedule(
                "s", 128, 128, 64, 2,
                tail_policy="exact", fixed_seconds=0.0)],
            require_complete_contract=False,
        )
        self.assertEqual(len(analysis["observations"]), 1)

    def test_report_deduplicates_identical_findings(self) -> None:
        finding = {
            "severity": "warning",
            "code": "same",
            "message": "same message",
        }
        analysis = build_closure_analysis(
            metadata={"model_input_audit": {"findings": [finding]}},
            base_capacities=[],
            closure_capacities=[],
            observations=[],
            hardware=Hardware("thor", 20, 1.575e9),
            schedules=[],
            require_complete_contract=False,
            input_findings=[finding, finding],
        )
        self.assertEqual(analysis["findings"], [finding])


if __name__ == "__main__":
    unittest.main()

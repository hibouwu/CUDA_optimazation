#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import statistics
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts.sm110_gemm_model import resource_import
from scripts.sm110_gemm_model.io import capacities_from_rows
from scripts.sm110_gemm_model.io import load_schedules
from scripts.sm110_gemm_model.model import Workload, account_work, precision_specs


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("sm110_resource_runner", HERE / "run_resource_campaign.py")
auditor = load_module("sm110_resource_auditor", HERE / "audit_campaign.py")
suite_auditor = load_module(
    "sm110_resource_suite_auditor", HERE / "audit_resource_suite.py"
)


class ResourceCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = runner.load_manifest()
        self.cases = runner.make_cases(self.manifest)

    def test_manifest_covers_exactly_twelve_precisions(self) -> None:
        precisions = {
            precision
            for family in self.manifest["families"]
            for precision in family["precision_ids"]
        }
        self.assertEqual(precisions, runner.EXPECTED_PRECISIONS)
        self.assertEqual(len(self.manifest["families"]), 9)
        self.assertIsNotNone(auditor.validate_manifest(self.manifest))

    def test_platform_preflight_requires_commit_and_all_clock_fields(self) -> None:
        commit = "b" * 40
        text = (
            "=== git ===\n"
            f"{suite_auditor.EXPECTED_BRANCH}\n{commit}\n"
            "=== nvpmodel ===\nNV Power Mode: MAXN\n"
            "min_freq=1575000000\nmax_freq=1575000000\n"
            "cur_freq=1575000000\ngovernor=performance\n"
        )
        self.assertEqual(suite_auditor.audit_preflight(
            text,
            expected_branch=suite_auditor.EXPECTED_BRANCH,
            expected_commit=commit,
        ), [])
        self.assertTrue(suite_auditor.audit_preflight(
            text.replace("cur_freq=1575000000\n", ""),
            expected_branch=suite_auditor.EXPECTED_BRANCH,
            expected_commit=commit,
        ))

    def test_oc_parser_rejects_duplicate_and_negative_counters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oc.tsv"
            path.write_text("/oc1\t1\n/oc2\t2\n")
            self.assertEqual(
                suite_auditor.parse_counter_tsv(path),
                {"/oc1": 1, "/oc2": 2},
            )
            path.write_text("/oc1\t1\n/oc1\t2\n")
            self.assertIsNone(suite_auditor.parse_counter_tsv(path))
            path.write_text("/oc1\t-1\n")
            self.assertIsNone(suite_auditor.parse_counter_tsv(path))

    def test_synthetic_platform_interval_preserves_oc_warning(self) -> None:
        commit = "c" * 40
        suite_id = "thor-resource-platform-synthetic"
        dependency_blobs = {
            path: f"blob:{path}".encode()
            for path in suite_auditor.EXPECTED_PLATFORM_DEPENDENCIES
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / suite_id
            root.mkdir()
            contract = {
                "schema_version": 1,
                "kind": "exact_resource_supplement",
                "suite_id": suite_id,
                "resource_run_id": f"{suite_id}-resources",
                "expected_branch": suite_auditor.EXPECTED_BRANCH,
                "expected_commit": commit,
                "ncu_required": True,
                "platform_dependencies": {
                    path: hashlib.sha256(blob).hexdigest()
                    for path, blob in dependency_blobs.items()
                },
            }
            (root / "run_contract.json").write_text(json.dumps(contract))
            (root / "preflight.txt").write_text(
                "=== git ===\n"
                f"{suite_auditor.EXPECTED_BRANCH}\n{commit}\n"
                "=== nvpmodel ===\nNV Power Mode: MAXN\n"
                "min_freq=1575000000\nmax_freq=1575000000\n"
                "cur_freq=1575000000\ngovernor=performance\n"
            )
            (root / "oc_before.tsv").write_text("/oc1\t10\n/oc2\t20\n")
            (root / "oc_after.tsv").write_text("/oc1\t10\n/oc2\t23\n")
            (root / "suite_launcher.log").write_text(
                "RESOURCE_CAMPAIGN_COMPLETE\nRESOURCE_SUPPLEMENT_COMPLETE\n"
            )
            fake_campaign = types.SimpleNamespace(
                audit=lambda *args, **kwargs: {"pass": True, "errors": []}
            )
            with (
                mock.patch.object(
                    suite_auditor, "git_blob",
                    side_effect=lambda actual_commit, relative: (
                        dependency_blobs.get(relative)
                        if actual_commit == commit else None
                    ),
                ),
                mock.patch.object(
                    suite_auditor, "load_campaign_auditor",
                    return_value=fake_campaign,
                ),
            ):
                result = suite_auditor.audit_suite(
                    root, expected_commit=commit
                )
            self.assertTrue(result["pass"], result["errors"])
            self.assertEqual(result["overcurrent_deltas"], {
                "/oc1": 0, "/oc2": 3,
            })
            self.assertEqual(
                result["warnings"], ["overcurrent_delta:/oc2:3"]
            )

    def test_case_matrix_is_exact_and_unique(self) -> None:
        self.assertEqual(len(self.cases), 54)
        self.assertEqual(
            len({case["case_id"] for case in self.cases}), 54
        )
        self.assertEqual(
            sum(bool(case["ncu_selected"]) for case in self.cases), 18
        )
        rebuilt = auditor.expected_cases(self.manifest)
        self.assertEqual(self.cases, rebuilt)

    def test_stage_payload_contracts(self) -> None:
        families = {
            family["family_id"]: family
            for family in self.manifest["families"]
        }
        expectations = {
            "generic_m128n64k64_s2_v8": (12 * 1024, 2, "64B"),
            "generic_m128n128k64_s2_v8": (16 * 1024, 2, "64B"),
            "generic_m128n256k64_s2_v8": (24 * 1024, 2, "64B"),
            "generic_m128n64k64_s2_v32": (48 * 1024, 4, "128B"),
            "generic_m128n128k64_s2_v32": (64 * 1024, 4, "128B"),
            "generic_m128n256k64_s2_v32": (96 * 1024, 4, "128B"),
            "block_m128n256k64_s2_v4_scale16": (13.5 * 1024, 4, "32B"),
            "block_m128n256k64_s2_v4_scale32": (13.5 * 1024, 4, "32B"),
            "tc5a_m128n256k64_s4_v16": (48 * 1024, 2, "128B"),
        }
        for family_id, (stage_bytes, requests, swizzle) in expectations.items():
            with self.subTest(family_id=family_id):
                contract = runner.family_contract(families[family_id])
                self.assertEqual(contract["stage_bytes"], stage_bytes)
                self.assertEqual(contract["requests_per_stage"], requests)
                self.assertEqual(contract["value_swizzle"], swizzle)

    def test_model_schedule_mapping_matches_every_campaign_family_contract(self) -> None:
        families = {
            row["family_id"]: row for row in self.manifest["families"]
        }
        precisions = precision_specs()
        schedules = load_schedules(
            REPO / "scripts/sm110_gemm_model/examples/schedules.json"
        )
        seen: set[tuple[str, str]] = set()
        for schedule in schedules:
            for precision_id, family_id in (
                schedule.tma_contract_family_by_precision.items()
            ):
                family = families[family_id]
                with self.subTest(
                    schedule=schedule.schedule_id,
                    precision=precision_id,
                    family=family_id,
                ):
                    self.assertIn(precision_id, family["precision_ids"])
                    self.assertIn(
                        schedule.input_transport_layout,
                        family["input_transport_layouts"],
                    )
                    self.assertEqual(
                        (schedule.bm, schedule.bn, schedule.bk, schedule.stages),
                        (
                            family["bm"], family["bn"], family["bk"],
                            family["stages"],
                        ),
                    )
                    self.assertEqual(schedule.threads, family["threads"])
                    work = account_work(
                        Workload(
                            "one-tile", schedule.bm, schedule.bn, schedule.bk,
                            precision_id,
                        ),
                        schedule,
                        precisions[precision_id],
                    )
                    self.assertEqual(
                        work.tma_input_bytes,
                        runner.family_contract(family)["stage_bytes"],
                    )
                    seen.add((precision_id, family_id))
        self.assertEqual(
            {precision for precision, _ in seen},
            runner.EXPECTED_PRECISIONS,
        )

    def test_block_scale_transport_is_32_byte_physical_rows(self) -> None:
        family = next(
            row for row in self.manifest["families"]
            if row["family_id"].endswith("scale16")
        )
        contract = runner.family_contract(family)
        self.assertEqual(contract["a_scale_bytes"], 512)
        self.assertEqual(contract["b_scale_bytes"], 1024)
        self.assertEqual(contract["a_scale_bytes"] % 32, 0)
        self.assertEqual(contract["b_scale_bytes"] % 32, 0)

    def test_hot_and_cold_scope_fields_are_not_interchangeable(self) -> None:
        family_id = "generic_m128n64k64_s2_v8"
        hot = next(
            case for case in self.cases
            if case["family_id"] == family_id
            and case["row_stride_elements"] == 1024
            and case["residency"] == "hot_l2"
        )
        cold = next(
            case for case in self.cases
            if case["family_id"] == family_id
            and case["row_stride_elements"] == 1024
            and case["residency"] == "cold_dram"
        )
        self.assertEqual(hot["expected"]["blocks"], 1)
        self.assertEqual(cold["expected"]["blocks"], 20)
        self.assertTrue(hot["resource"].endswith(".per_sm"))
        self.assertTrue(cold["resource"].startswith("tma.hbm.contract."))
        self.assertNotEqual(hot["args"], cold["args"])

    def test_row_stride_is_frozen_in_case_and_resource_identity(self) -> None:
        family_id = "generic_m128n256k64_s2_v32"
        rows = [
            case for case in self.cases
            if case["family_id"] == family_id
            and case["residency"] == "hot_l2"
        ]
        self.assertEqual(
            {case["row_stride_elements"] for case in rows},
            {1024, 2048, 4096},
        )
        for case in rows:
            self.assertIn(
                f"stride{case['row_stride_elements']}", case["resource"]
            )

    def test_field_audit_rejects_one_contract_mutation(self) -> None:
        case = self.cases[0]
        fields = {
            name: str(value) for name, value in case["expected"].items()
        }
        fields["contract_only"] = "1"
        self.assertEqual(
            auditor.field_errors(case, fields, runtime=False), []
        )
        fields["requests_per_stage"] = "99"
        self.assertIn(
            "field mismatch:requests_per_stage",
            auditor.field_errors(case, fields, runtime=False),
        )

    def test_manifest_rejects_missing_precision_and_extra_stride(self) -> None:
        missing = copy.deepcopy(self.manifest)
        for family in missing["families"]:
            family["precision_ids"] = [
                value for value in family["precision_ids"]
                if value != "u8_s32"
            ]
        self.assertIsNone(auditor.validate_manifest(missing))
        extra_stride = copy.deepcopy(self.manifest)
        extra_stride["row_stride_elements"].append(8192)
        self.assertIsNone(auditor.validate_manifest(extra_stride))

    def test_relocation_safe_binary_command(self) -> None:
        case = self.cases[0]
        run_id = "thor-resource-test"
        binary = (
            "/xplorer/checkout/results/sm110_gemm_resource_campaign/"
            f"{run_id}/build/tma_ab_contract_bandwidth"
        )
        self.assertTrue(auditor.valid_binary_command(
            [binary, *case["args"]], run_id, case["args"]
        ))
        self.assertFalse(auditor.valid_binary_command(
            ["relative/binary", *case["args"]], run_id, case["args"]
        ))

    def test_sass_attribution_is_function_scoped(self) -> None:
        sass = (
            "Function : unrelated\nUTMALDG.2D\n"
            "Function : tma_ab_contract_kernel\nBAR.SYNC\n"
        )
        block = auditor.function_sass(sass, "tma_ab_contract_kernel")
        self.assertIsNotNone(block)
        self.assertNotIn("UTMALDG.2D", block or "")

    def test_timeout_runner_is_bounded(self) -> None:
        result = runner.run_bounded(
            [sys.executable, "-c", "import time; time.sleep(60)"], 1
        )
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["termination_failed"])

    def test_resume_reuses_only_reaudited_trials(self) -> None:
        case = self.cases[0]
        fingerprint = "f" * 64
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            case_dir = run_dir / "cases" / case["case_id"]
            case_dir.mkdir(parents=True)
            binary = run_dir / "build/tma_ab_contract_bandwidth"
            rates = []
            rows = []
            for trial_index in range(1, 11):
                elapsed = 10_000 + trial_index
                requested = (
                    case["expected"]["blocks"]
                    * case["expected"]["iters"]
                    * case["expected"]["stage_bytes"]
                )
                rate = requested * 1.0e9 / elapsed
                fields = {
                    name: str(value)
                    for name, value in case["expected"].items()
                }
                fields.update({
                    "unique_smid_count": "1",
                    "globaltimer_start_min_ns": "1000",
                    "globaltimer_stop_max_ns": str(1000 + elapsed),
                    "globaltimer_elapsed_ns": str(elapsed),
                    "requested_bytes": str(requested),
                    "bytes_per_second": f"{rate:.9f}",
                    "occupancy_blocks_per_sm": "1",
                })
                raw = " ".join(
                    f"{name}={value}" for name, value in fields.items()
                ) + "\n"
                rates.append(rate)
                rows.append({
                    "trial": trial_index,
                    "command": [str(binary), *case["args"]],
                    "returncode": 0,
                    "timeout_seconds": 120,
                    "timed_out": False,
                    "termination_failed": False,
                    "raw_stdout": raw,
                    "fields": fields,
                    "audited_rate_per_second": rate,
                })
            result = {
                "status": "ok",
                "fingerprint": fingerprint,
                "trial_count": 10,
                "trial_timeout_seconds": 120,
                "rate_per_second_median": statistics.median(rates),
                "rate_per_second_min": min(rates),
                "rate_per_second_max": max(rates),
                "rate_per_second_mean": statistics.fmean(rates),
            }
            (case_dir / "result.json").write_text(json.dumps(result))
            trials_path = case_dir / "trials.jsonl"
            trials_path.write_text("".join(
                json.dumps(row) + "\n" for row in rows
            ))
            self.assertIsNotNone(runner.prior_result_is_reusable(
                case_dir, case, fingerprint, False
            ))
            rows[0]["fields"]["requested_bytes"] = "1"
            rows[0]["raw_stdout"] = " ".join(
                f"{name}={value}"
                for name, value in rows[0]["fields"].items()
            ) + "\n"
            trials_path.write_text("".join(
                json.dumps(row) + "\n" for row in rows
            ))
            self.assertIsNone(runner.prior_result_is_reusable(
                case_dir, case, fingerprint, False
            ))

    def test_resume_reuses_audited_frozen_compile_without_recompiling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "frozen"
            build = run_dir / "build"
            build.mkdir(parents=True)
            binary = build / "tma_ab_contract_bandwidth"
            sass = build / "tma_ab_contract_bandwidth.sass.txt"
            compile_path = build / "compile_command.json"
            binary.write_bytes(b"one retained nvcc output")
            sass.write_text(
                "Function : tma_ab_contract_kernel\n"
                " /*0000*/ UTMALDG.2D\n"
                "Function : unrelated\n /*0008*/ UTMASTG\n"
            )
            (build / "compile.log").write_text("compile ok\n")
            with mock.patch.object(
                runner, "tool", side_effect=lambda name: f"/tools/{name}"
            ):
                command = runner.compile_command(run_dir, None, False)
            compile_path.write_text(json.dumps(command, indent=2) + "\n")
            binary_sha = runner.sha256_path(binary)
            (build / "binary.sha256").write_text(
                f"{binary_sha}  tma_ab_contract_bandwidth\n"
            )
            artifact = {
                "binary": str(binary),
                "binary_sha256": binary_sha,
                "source_sha256": runner.sha256_path(runner.SOURCE_PATH),
                "sass_path": str(sass),
                "sass_sha256": runner.sha256_path(sass),
                "compile_command_sha256": runner.sha256_path(compile_path),
                "compile_command": command,
                "source_dependencies": {
                    path: runner.sha256_path(runner.REPO / path)
                    for path in runner.SOURCE_DEPENDENCIES
                },
                "sass_function_counts": {
                    "UTMALDG.2D": 1,
                    "UTMASTG": 0,
                },
            }
            (build / "artifact.json").write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n"
            )
            with mock.patch.object(
                runner, "tool", side_effect=lambda name: f"/tools/{name}"
            ):
                retained = runner.retained_artifact(run_dir, None, False)
            self.assertIsNotNone(retained)
            self.assertEqual(retained["binary_sha256"], binary_sha)
            binary.write_bytes(b"mutated retained output")
            with (
                mock.patch.object(
                    runner, "tool", side_effect=lambda name: f"/tools/{name}"
                ),
                self.assertRaisesRegex(RuntimeError, "binary_sha256"),
            ):
                runner.retained_artifact(run_dir, None, False)

    def test_source_exposes_contract_only_and_scale_swizzle(self) -> None:
        source = runner.SOURCE_PATH.read_text()
        self.assertIn("--contract-only", source)
        self.assertIn("CU_TENSOR_MAP_SWIZZLE_32B", source)
        self.assertIn("cudaMemset(data_a, 0, buffers.a_allocation_bytes)", source)
        self.assertIn("cp.async.bulk.tensor.2d", source)
        self.assertIn("mbarrier::complete_tx::bytes", source)

    def test_formal_wrapper_and_runbook_ship_together(self) -> None:
        wrapper = (REPO / "microbench/sm110_resource_supplement.sh").read_text()
        supervisor = (
            REPO / "microbench/run_sm110_resource_supplement.sh"
        ).read_text()
        runbook = (
            REPO / "Docs/blackwell_tensorcore/THOR_CLOSURE_RUNBOOK.md"
        ).read_text()
        self.assertIn("expected_commit=$(git rev-parse HEAD)", wrapper)
        self.assertIn(suite_auditor.EXPECTED_BRANCH, wrapper)
        for action in ("start", "resume", "status", "finish"):
            self.assertIn(
                f"sm110_resource_supplement.sh {action}", runbook
            )
        self.assertIn("--run-id \"$run_id\" --ncu", supervisor)
        self.assertIn("--require-ncu --expected-commit", supervisor)
        self.assertIn("import-resource-capacities", wrapper)
        self.assertIn("resource_capacities.json", wrapper)
        self.assertIn("sudo /usr/sbin/nvpmodel -m 1", runbook)
        self.assertIn(
            "results/sm110_gemm_resource_campaign/$SUITE_ID-resources",
            runbook,
        )

    def test_formal_compile_command_rejects_extra_definitions(self) -> None:
        run_id = "thor-resource-test"
        source = (
            "/x/repo/microbench/15_tma_ab_contract_bandwidth/"
            "tma_ab_contract_bandwidth.cu"
        )
        output = (
            "/x/repo/results/sm110_gemm_resource_campaign/"
            f"{run_id}/build/tma_ab_contract_bandwidth"
        )
        command = [
            "/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17", "-gencode",
            "arch=compute_110a,code=sm_110a", source, "-lcuda", "-o", output,
        ]
        self.assertTrue(auditor.audit_compile_command(command, run_id))
        command.insert(3, "-DEVIL_OVERRIDE=1")
        self.assertFalse(auditor.audit_compile_command(command, run_id))

    def test_ncu_hot_l2_warmup_covers_reduced_working_set(self) -> None:
        case = next(
            row for row in self.cases
            if row["residency"] == "hot_l2"
            and row["ncu_selected"]
            and row["family_id"] == "generic_m128n64k64_s2_v8"
        )
        args = runner.ncu_args(case)
        byte_target = int(args[args.index("--bytes") + 1])
        warmup = int(args[args.index("--warmup-iters") + 1])
        tiles = max(1, byte_target // case["expected"]["stage_bytes"])
        self.assertGreaterEqual(warmup, tiles + case["expected"]["stages"])
        self.assertEqual(args, auditor.ncu_expected_args(case))

    def test_ncu_cold_profile_requests_at_least_128_mib(self) -> None:
        case = next(
            row for row in self.cases
            if row["residency"] == "cold_dram"
            and row["ncu_selected"]
            and row["family_id"] == "generic_m128n64k64_s2_v8"
        )
        args = runner.ncu_args(case)
        iters = int(args[args.index("--iters") + 1])
        requested = (
            case["expected"]["blocks"]
            * iters
            * case["expected"]["stage_bytes"]
        )
        self.assertGreaterEqual(requested, 128 << 20)
        self.assertEqual(args, auditor.ncu_expected_args(case))

    def test_complete_synthetic_bundle_passes_independent_audit(self) -> None:
        run_id = "thor-resource-synthetic"
        commit = "a" * 40
        fake_repo = Path("/xplorer/checkout")
        binary_bytes = b"synthetic retained binary"
        binary_sha = hashlib.sha256(binary_bytes).hexdigest()
        sass_text = (
            "Function : tma_ab_contract_kernel\n"
            " /*0000*/ UTMALDG.2D\n"
            "Function : unrelated\n"
        )
        sass_sha = hashlib.sha256(sass_text.encode()).hexdigest()
        dependency_blobs = {
            relative: (REPO / relative).read_bytes()
            for relative in runner.SOURCE_DEPENDENCIES
        }
        dependencies = {
            relative: hashlib.sha256(payload).hexdigest()
            for relative, payload in dependency_blobs.items()
        }
        cases = self.cases

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / run_id
            build = root / "build"
            build.mkdir(parents=True)
            binary = build / "tma_ab_contract_bandwidth"
            binary.write_bytes(binary_bytes)
            (build / "binary.sha256").write_text(
                f"{binary_sha}  tma_ab_contract_bandwidth\n"
            )
            (build / "tma_ab_contract_bandwidth.sass.txt").write_text(
                sass_text
            )
            (build / "compile.log").write_text("")
            source = (
                fake_repo
                / "microbench/15_tma_ab_contract_bandwidth/"
                  "tma_ab_contract_bandwidth.cu"
            )
            recorded_binary = (
                fake_repo / "results/sm110_gemm_resource_campaign"
                / run_id / "build/tma_ab_contract_bandwidth"
            )
            compile_command = [
                "/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17",
                "-gencode", "arch=compute_110a,code=sm_110a",
                str(source), "-lcuda", "-o", str(recorded_binary),
            ]
            (build / "compile_command.json").write_text(
                json.dumps(compile_command) + "\n"
            )
            (build / "artifact.json").write_text(json.dumps({
                "binary": str(recorded_binary),
                "binary_sha256": binary_sha,
                "source_sha256": dependencies[
                    "microbench/15_tma_ab_contract_bandwidth/"
                    "tma_ab_contract_bandwidth.cu"
                ],
                "sass_path": str(recorded_binary.with_name(
                    "tma_ab_contract_bandwidth.sass.txt"
                )),
                "sass_sha256": sass_sha,
                "compile_command": compile_command,
                "compile_command_sha256": hashlib.sha256(
                    (build / "compile_command.json").read_bytes()
                ).hexdigest(),
                "source_dependencies": dependencies,
                "sass_function_counts": {
                    "UTMALDG.2D": 1,
                    "UTMASTG": 0,
                },
            }, indent=2, sort_keys=True) + "\n")

            manifest_relative = (
                "microbench/sm110_gemm_resource_campaign/"
                "contract_manifest.json"
            )
            generator_relative = (
                "microbench/sm110_gemm_resource_campaign/"
                "run_resource_campaign.py"
            )
            manifest_blob = dependency_blobs[manifest_relative]
            generator_blob = dependency_blobs[generator_relative]
            spec = {
                "schema_version": 1,
                "run_id": run_id,
                "campaign": "sm110_exact_tma_resource_contracts",
                "expected_sm_count": 20,
                "trials": 10,
                "case_count": 54,
                "family_count": 9,
                "static_only": False,
                "ncu_requested": True,
                "ncu_policy": (
                    "row_stride=2048 for every family and residency"
                ),
                "trial_timeout_seconds": 120,
                "ncu_timeout_seconds": 300,
                "termination_grace_seconds": 5,
                "generator": generator_relative,
                "generator_sha256": hashlib.sha256(
                    generator_blob
                ).hexdigest(),
                "contract_manifest": manifest_relative,
                "contract_manifest_sha256": hashlib.sha256(
                    manifest_blob
                ).hexdigest(),
                "source_dependencies": dependencies,
                "cases": cases,
            }
            (root / "run_spec.json").write_text(
                json.dumps(spec, indent=2, sort_keys=True) + "\n"
            )

            environment = {
                "gpu_identity": {"returncode": 0, "output": "Thor, 11.0"},
                "gpu_state": {"returncode": 0, "output": "P0"},
                "nvcc": {"returncode": 0, "output": "CUDA 13"},
                "ncu": {"returncode": 0, "output": "NCU 2026"},
                "git_head": {"returncode": 0, "output": commit + "\n"},
                "git_branch": {"returncode": 0, "output": "branch\n"},
                "git_status": {"returncode": 0, "output": ""},
                "power_mode": {"returncode": 0, "output": "MAXN"},
            }
            (root / "environment.json").write_text(
                json.dumps(environment) + "\n"
            )
            (root / "environment_snapshots.jsonl").write_text(
                json.dumps(environment, sort_keys=True) + "\n"
            )

            static_rows = []
            results = []
            for case in cases:
                case_id = case["case_id"]
                fields = {
                    name: str(value)
                    for name, value in case["expected"].items()
                }
                fields["contract_only"] = "1"
                static_rows.append({
                    "case_id": case_id,
                    "command": [
                        str(recorded_binary), "--contract-only", *case["args"]
                    ],
                    "fields": fields,
                })

                case_dir = root / "cases" / case_id
                case_dir.mkdir(parents=True)
                rates = []
                trials = []
                for trial_index in range(1, 11):
                    start = 1_000_000 + trial_index
                    elapsed = 10_000 + trial_index
                    stop = start + elapsed
                    requested = (
                        case["expected"]["blocks"]
                        * case["expected"]["iters"]
                        * case["expected"]["stage_bytes"]
                    )
                    rate = requested * 1.0e9 / elapsed
                    runtime_fields = {
                        name: str(value)
                        for name, value in case["expected"].items()
                    }
                    runtime_fields.update({
                        "unique_smid_count": str(
                            1 if case["residency"] == "hot_l2" else 20
                        ),
                        "globaltimer_start_min_ns": str(start),
                        "globaltimer_stop_max_ns": str(stop),
                        "globaltimer_elapsed_ns": str(elapsed),
                        "requested_bytes": str(requested),
                        "bytes_per_second": f"{rate:.9f}",
                        "occupancy_blocks_per_sm": "1",
                        "sink": "1",
                    })
                    raw = " ".join(
                        f"{name}={value}"
                        for name, value in runtime_fields.items()
                    ) + "\n"
                    rates.append(rate)
                    trials.append({
                        "trial": trial_index,
                        "captured_at_utc": "synthetic",
                        "command": [str(recorded_binary), *case["args"]],
                        "returncode": 0,
                        "timeout_seconds": 120,
                        "timed_out": False,
                        "termination_failed": False,
                        "raw_stdout": raw,
                        "fields": runtime_fields,
                        "audited_rate_per_second": rate,
                    })
                (case_dir / "trials.jsonl").write_text("".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in trials
                ))

                source_hash = dependencies[
                    "microbench/15_tma_ab_contract_bandwidth/"
                    "tma_ab_contract_bandwidth.cu"
                ]
                ncu_selected = bool(case["ncu_selected"])
                ncu = {
                    "selected": False,
                    "policy": (
                        "row_stride=2048 for every family and residency"
                    ),
                }
                if ncu_selected:
                    ncu_dir = case_dir / "ncu"
                    ncu_dir.mkdir()
                    report = ncu_dir / "profile.ncu-rep"
                    log = ncu_dir / "profile.log"
                    report.write_bytes(b"synthetic ncu report")
                    log.write_text("synthetic ncu log\n")
                    report_base = (
                        fake_repo / "results/sm110_gemm_resource_campaign"
                        / run_id / "cases" / case_id / "ncu/profile"
                    )
                    ncu = {
                        "selected": True,
                        "policy": (
                            "row_stride=2048 for every family and residency"
                        ),
                        "set": "basic",
                        "command": [
                            "/usr/local/cuda/bin/ncu", "--set", "basic",
                            "--target-processes", "all",
                            "--kernel-name-base", "demangled",
                            "--kernel-name", "regex:tma_ab_contract_kernel",
                            "--launch-count", "1", "--force-overwrite",
                            "--export", str(report_base), str(recorded_binary),
                            *runner.ncu_args(case),
                        ],
                        "returncode": 0,
                        "kernel_name_base": "demangled",
                        "kernel_name_regex": "tma_ab_contract_kernel",
                        "launch_count": 1,
                        "permission_denied": False,
                        "timed_out": False,
                        "termination_failed": False,
                        "timeout_seconds": 300,
                        "report_path": "ncu/profile.ncu-rep",
                        "report_sha256": hashlib.sha256(
                            report.read_bytes()
                        ).hexdigest(),
                        "log_sha256": hashlib.sha256(
                            log.read_bytes()
                        ).hexdigest(),
                        "pass": True,
                    }
                fingerprint = runner.sha256_json({
                    "case": case,
                    "binary_sha256": binary_sha,
                    "source_sha256": source_hash,
                    "trial_count": 10,
                    "ncu": ncu_selected,
                })
                results.append({
                    "schema_version": 1,
                    "case_id": case_id,
                    "family_id": case["family_id"],
                    "resource": case["resource"],
                    "residency": case["residency"],
                    "row_stride_elements": case["row_stride_elements"],
                    "precision_ids": case["precision_ids"],
                    "input_transport_layouts":
                        case["input_transport_layouts"],
                    "status": "ok",
                    "fingerprint": fingerprint,
                    "trial_count": 10,
                    "rate_unit": "B/s",
                    "rate_per_second_median": statistics.median(rates),
                    "rate_per_second_min": min(rates),
                    "rate_per_second_max": max(rates),
                    "rate_per_second_mean": statistics.fmean(rates),
                    "expected_contract": case["expected"],
                    "source_path": (
                        "microbench/15_tma_ab_contract_bandwidth/"
                        "tma_ab_contract_bandwidth.cu"
                    ),
                    "source_sha256": source_hash,
                    "binary_sha256": binary_sha,
                    "sass_path": "build/tma_ab_contract_bandwidth.sass.txt",
                    "sass_sha256": sass_sha,
                    "sass_tokens": ["UTMALDG.2D"],
                    "trial_timeout_seconds": 120,
                    "ncu": ncu,
                })

            (root / "static_contracts.json").write_text(
                json.dumps(static_rows) + "\n"
            )
            summary = {
                "schema_version": 1,
                "run_id": run_id,
                "status": "complete",
                "case_count": 54,
                "family_count": 9,
                "ncu_requested": True,
                "ncu_case_count": 18,
                "results": results,
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary) + "\n")
            (root / "campaign_status.json").write_text(json.dumps({
                "status": "complete", "completed_cases": 54,
                "total_cases": 54,
            }) + "\n")
            (root / "progress.jsonl").write_text(json.dumps({
                "status": "complete", "completed_cases": 54,
                "total_cases": 54,
            }) + "\n")
            (root / "COMPLETE").write_text(
                f"run_id={run_id}\n"
                f"summary_sha256={hashlib.sha256(summary_path.read_bytes()).hexdigest()}\n"
            )
            runner.write_artifact_manifest(root)

            with mock.patch.object(
                auditor, "git_blob",
                side_effect=lambda actual_commit, relative: (
                    dependency_blobs.get(relative)
                    if actual_commit == commit else None
                ),
            ):
                result = auditor.audit(
                    root, require_ncu=True, expected_commit=commit
                )
            self.assertTrue(result["pass"], result["errors"])

    def test_model_import_preserves_all_exact_resource_identities(self) -> None:
        suite_id = "thor-resource-import-synthetic"
        run_id = f"{suite_id}-resources"
        commit = "d" * 40
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            suite = repo / "results/sm110_resource_suite" / suite_id
            run = repo / "results/sm110_gemm_resource_campaign" / run_id
            build = run / "build"
            suite.mkdir(parents=True)
            build.mkdir(parents=True)
            (suite / "run_contract.json").write_text(json.dumps({
                "resource_run_id": run_id,
            }))
            for name in (
                "preflight.txt", "oc_before.tsv", "oc_after.tsv",
                "suite_launcher.log", "suite_audit.json",
            ):
                (suite / name).write_text("synthetic\n")
            for name in (
                "run_spec.json", "environment.json",
                "environment_snapshots.jsonl", "static_contracts.json",
                "COMPLETE", "artifact_sha256.txt",
            ):
                (run / name).write_text("synthetic\n")
            for name in (
                "compile_command.json", "compile.log", "artifact.json",
                "binary.sha256",
                "tma_ab_contract_bandwidth",
                "tma_ab_contract_bandwidth.sass.txt",
            ):
                (build / name).write_text("synthetic\n")
            source = (
                repo / "microbench/15_tma_ab_contract_bandwidth/"
                       "tma_ab_contract_bandwidth.cu"
            )
            source.parent.mkdir(parents=True)
            source.write_text("synthetic source\n")

            results = []
            for index, case in enumerate(self.cases, 1):
                case_dir = run / "cases" / case["case_id"]
                case_dir.mkdir(parents=True)
                (case_dir / "result.json").write_text("{}\n")
                (case_dir / "trials.jsonl").write_text("{}\n")
                ncu = {"selected": False}
                if case["ncu_selected"]:
                    ncu_dir = case_dir / "ncu"
                    ncu_dir.mkdir()
                    (ncu_dir / "profile.ncu-rep").write_text("report\n")
                    (ncu_dir / "profile.log").write_text("log\n")
                    ncu = {
                        "selected": True,
                        "report_path": "ncu/profile.ncu-rep",
                    }
                results.append({
                    "case_id": case["case_id"],
                    "family_id": case["family_id"],
                    "resource": case["resource"],
                    "residency": case["residency"],
                    "row_stride_elements": case["row_stride_elements"],
                    "rate_unit": "B/s",
                    "rate_per_second_median": float(index * 1_000_000),
                    "trial_count": 10,
                    "expected_contract": case["expected"],
                    "source_path": (
                        "microbench/15_tma_ab_contract_bandwidth/"
                        "tma_ab_contract_bandwidth.cu"
                    ),
                    "ncu": ncu,
                })
            summary = {
                "status": "complete",
                "case_count": 54,
                "results": results,
            }
            (run / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )

            fake_platform = types.SimpleNamespace(audit_suite=lambda *a, **k: {
                "pass": True,
                "errors": [],
                "warnings": ["overcurrent_delta:/oc3:1"],
                "overcurrent_deltas": {"/oc3": 1},
            })
            with mock.patch.object(
                resource_import,
                "_load_platform_auditor",
                return_value=fake_platform,
            ):
                imported = resource_import.import_resource_capacities(
                    repo_root=repo,
                    suite_id=suite_id,
                    expected_commit=commit,
                )
            capacities = capacities_from_rows(imported["capacities"])
            self.assertEqual(len(capacities), 54)
            self.assertEqual(len({row.resource for row in capacities}), 54)
            self.assertTrue(all(row.is_closure_qualified for row in capacities))
            self.assertEqual(
                imported["platform_evidence"]["overcurrent_deltas"],
                {"/oc3": 1},
            )
            self.assertIn(
                "tma.smem_ingress.contract."
                "tc5a_m128n256k64_s4_v16.stride2048.per_sm",
                {row.resource for row in capacities},
            )


if __name__ == "__main__":
    unittest.main()

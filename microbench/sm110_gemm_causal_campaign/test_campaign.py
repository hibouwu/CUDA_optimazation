from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from microbench.sm110_gemm_causal_campaign import audit_campaign as auditor
from microbench.sm110_gemm_causal_campaign import audit_causal_suite as suite_auditor
from microbench.sm110_gemm_causal_campaign import run_causal_campaign as runner
from scripts.sm110_gemm_model import causal_import
from scripts.sm110_gemm_model.io import pipeline_profiles_from_rows


COMMIT = "1" * 40


def synthetic_events(case: dict[str, object]) -> dict[str, int | float]:
    start = 1_000_000_000
    mode = str(case["mode"])
    k_tiles = int(case["k_tiles"])
    output_tasks = int(case["output_tasks"])
    operations = k_tiles * output_tasks
    first_tma = start + 100 if mode != "mma-only" else 0
    last_tma = first_tma + (operations - 1) * 30 if first_tma else 0
    first_mma = start + 200 if mode == "mma-only" else (
        start + 300 if mode in {"overlap", "full"} else 0
    )
    if mode == "mma-only":
        last_mma = first_mma + (operations - 1) * 40
        first_epi = last_store = 0
        exit_ns = last_mma
    elif mode == "overlap":
        last_mma = first_mma + (operations - 1) * 50
        first_epi = last_store = 0
        exit_ns = last_mma
    elif mode == "full":
        previous_mma = 0.0
        epi_done: list[float] = []
        first_epi = 0
        last_mma = 0
        for task in range(output_tasks):
            first = 300.0 if task == 0 else previous_mma + 50.0
            if task >= 2:
                first = max(first, epi_done[task - 2] + 50.0)
            last = first + (k_tiles - 1) * 50.0
            epi_start = max(last, epi_done[-1] if epi_done else 0.0)
            if task == 0:
                first_epi = start + int(epi_start)
            epi_done.append(epi_start + 100.0)
            previous_mma = last
            last_mma = start + int(last)
        last_store = start + int(epi_done[-1])
        exit_ns = last_store
    else:
        last_mma = first_epi = last_store = 0
        exit_ns = last_tma
    tma_span = last_tma - first_tma if first_tma else 0
    mma_span = last_mma - first_mma if first_mma else 0
    return {
        "start_ns": start,
        "first_tma_done_ns": first_tma,
        "last_tma_done_ns": last_tma,
        "first_mma_done_ns": first_mma,
        "last_mma_done_ns": last_mma,
        "first_epilogue_start_ns": first_epi,
        "last_store_done_ns": last_store,
        "kernel_exit_ns": exit_ns,
        "first_tma_latency_ns": first_tma - start if first_tma else 0,
        "tma_completion_span_ns": tma_span,
        "tma_interval_ns": tma_span / (operations - 1)
            if first_tma and operations > 1 else 0.0,
        "first_mma_latency_ns": first_mma - start if first_mma else 0,
        "mma_completion_span_ns": mma_span,
        "mma_interval_ns": mma_span / (operations - 1)
            if first_mma and operations > 1 else 0.0,
        "epilogue_to_store_ns": last_store - first_epi if first_epi else 0,
        "last_mma_to_store_ns": last_store - last_mma if last_store else 0,
        "total_measured_ns": exit_ns - start,
    }


def csv_fields(
    case: dict[str, object], manifest: dict[str, object], smid: int = 7,
) -> dict[str, str]:
    fields = auditor.expected_static_fields(case, manifest)
    fields["smid"] = str(smid)
    fields.update({name: str(value) for name, value in synthetic_events(case).items()})
    for name in ("tma_interval_ns", "mma_interval_ns"):
        fields[name] = f"{float(fields[name]):.9f}"
    return fields


def csv_text(fields: dict[str, str]) -> str:
    return ",".join(fields[name] for name in auditor.CSV_FIELDS) + "\n"


def write_hash_manifest(root: Path) -> None:
    excluded = {"artifact_sha256.txt", "launcher.log", "launcher.pid"}
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            rows.append(
                f"{auditor.sha256_path(path)}  {path.relative_to(root).as_posix()}\n"
            )
    (root / "artifact_sha256.txt").write_text("".join(rows))


class CausalCampaignTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = runner.load_manifest()
        self.cases = runner.make_cases(self.manifest)

    def test_frozen_matrix_covers_exact_shapes_and_holdouts(self) -> None:
        plan = runner.plan_payload(self.manifest, self.cases)
        self.assertEqual(plan["case_count"], 91)
        self.assertEqual(plan["family_count"], 13)
        self.assertEqual(plan["raw_trial_count"], 910)
        self.assertEqual(plan["ncu_case_count"], 4)
        self.assertEqual(plan["precision_ids"], ["fp16_f32"])
        self.assertEqual(plan["max_dynamic_smem_bytes"], 196608)
        self.assertEqual(plan["max_hot_input_bytes"], 3 * 1024 * 1024)
        self.assertEqual(plan["max_output_bytes"], 4 * 1024 * 1024)
        self.assertEqual(self.manifest["holdout_k_tiles"], [64])
        self.assertEqual(self.manifest["precision_ids"], ["fp16_f32"])
        self.assertEqual(
            self.manifest["fit_contract"]["holdout_output_tasks"], [32]
        )

    def test_source_contract_is_tc5a_specific(self) -> None:
        source = runner.SOURCE_PATH.read_text()
        self.assertIn("constexpr int kTileM = 128", source)
        self.assertIn("constexpr int kTileN = 256", source)
        self.assertIn("constexpr int kTileK = 64", source)
        self.assertIn("constexpr int kThreads = 192", source)
        self.assertIn("kAccumulatorBuffers = 2", source)
        self.assertIn("globaltimer", source)
        self.assertIn("ptx::tma_load_2d", source)
        self.assertIn("ptx::mma_f16", source)
        self.assertIn("fp16_f32", source)
        self.assertIn("tmem_load_32x32b_x8_no_wait", source)
        self.assertIn("const int offset_m = 0", source)

    def test_sass_attribution_is_function_scoped(self) -> None:
        good = "".join(
            f"Function : synthetic_tc5a_pipeline_dag_kernelILi{stages}EE\n"
            "UTMALDG.2D\nUTCHMMA\nUTCBAR\nLDTM.\n"
            for stages in (1, 2, 4)
        )
        counts, errors = auditor.sass_stage_function_counts(good)
        self.assertEqual(errors, [])
        self.assertEqual(set(counts), {"stage1", "stage2", "stage4"})
        bad = good.replace(
            "Function : synthetic_tc5a_pipeline_dag_kernelILi4EE\n"
            "UTMALDG.2D\nUTCHMMA\nUTCBAR\nLDTM.\n",
            "Function : synthetic_tc5a_pipeline_dag_kernelILi4EE\n"
            "UTMALDG.2D\nUTCBAR\nLDTM.\n",
        )
        _, errors = auditor.sass_stage_function_counts(bad)
        self.assertIn("stage-4 SASS token missing:UTCHMMA", errors)

    def test_resume_reuses_only_a_reaudited_frozen_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            build = run_dir / "build"
            build.mkdir(parents=True)
            binary = build / "tc5a_pipeline_dag"
            binary.write_bytes(b"non-reproducible nvcc binary snapshot")
            sass = build / "tc5a_pipeline_dag.sass.txt"
            sass.write_text("".join(
                f"Function : synthetic_tc5a_pipeline_dag_kernelILi{stages}EE\n"
                "UTMALDG.2D\nUTCHMMA\nUTCBAR\nLDTM.\n"
                for stages in (1, 2, 4)
            ))
            command = runner.compile_command(run_dir)
            compile_path = build / "compile_command.json"
            compile_path.write_text(json.dumps(command))
            (build / "compile.log").write_text("retained\n")
            (build / "csv_header.txt").write_text(
                ",".join(runner.CSV_FIELDS) + "\n"
            )
            binary_sha = runner.sha256_path(binary)
            (build / "binary.sha256").write_text(
                f"{binary_sha}  tc5a_pipeline_dag\n"
            )
            counts = runner.sass_stage_function_counts(sass.read_text())
            metadata = {
                "binary": str(binary),
                "binary_sha256": binary_sha,
                "source_sha256": runner.sha256_path(runner.SOURCE_PATH),
                "helper_sha256": runner.sha256_path(runner.HELPER_PATH),
                "manifest_sha256": runner.sha256_path(runner.MANIFEST_PATH),
                "sass_path": str(sass),
                "sass_sha256": runner.sha256_path(sass),
                "compile_command": command,
                "compile_command_sha256": runner.sha256_path(compile_path),
                "sass_tokens": sorted(runner.REQUIRED_SASS_TOKENS),
                "sass_function_counts": counts,
            }
            (build / "artifact.json").write_text(json.dumps(metadata))
            header_result = {
                "returncode": 0,
                "timed_out": False,
                "stdout": ",".join(runner.CSV_FIELDS) + "\n",
            }
            with mock.patch.object(
                runner, "run_bounded", return_value=header_result
            ):
                retained = runner.retained_artifact(run_dir)
            self.assertIsNotNone(retained)
            self.assertEqual(retained["binary_sha256"], binary_sha)
            binary.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                RuntimeError, "retained causal artifact hash changed"
            ):
                runner.retained_artifact(run_dir)

    def test_field_auditor_recomputes_raw_timer_arithmetic(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["case_id"]):
                fields = csv_fields(case, self.manifest)
                errors, metrics = auditor.field_errors(
                    case, fields, self.manifest, runtime=True
                )
                self.assertEqual(errors, [])
                self.assertIsNotNone(metrics)
        fields = csv_fields(self.cases[-1], self.manifest)
        fields["total_measured_ns"] = str(
            int(float(fields["total_measured_ns"])) + 1
        )
        errors, _ = auditor.field_errors(
            self.cases[-1], fields, self.manifest, runtime=True
        )
        self.assertIn("derived metric mismatch:total_measured_ns", errors)

    def test_dag_recurrence_models_double_buffer_and_epilogue(self) -> None:
        one = auditor.predict_worker_ns(4, 1, 300, 50, 100, 2)
        two = auditor.predict_worker_ns(4, 2, 300, 50, 100, 2)
        eight = auditor.predict_worker_ns(4, 8, 300, 50, 100, 2)
        self.assertEqual(one, 550)
        self.assertEqual(two, 750)
        self.assertGreater(eight, two)
        self.assertLess(eight, 8 * one)

    def _make_bundle(self, base: Path) -> tuple[Path, dict[str, bytes]]:
        run_id = "synthetic-causal"
        root = base / "results/sm110_gemm_causal_campaign" / run_id
        (root / "build").mkdir(parents=True)
        dependencies = {
            relative: (auditor.REPO / relative).read_bytes()
            for relative in auditor.EXPECTED_DEPENDENCIES
        }
        dependency_hashes = {
            relative: auditor.sha256_bytes(payload)
            for relative, payload in dependencies.items()
        }
        environment = {
            "captured_at_utc": "2026-08-15T00:00:00+00:00",
            "gpu_identity": {"returncode": 0, "output": "Thor, UUID, 11.0\n"},
            "gpu_state": {"returncode": 0, "output": "locked\n"},
            "git_head": {"returncode": 0, "output": f"{COMMIT}\n"},
            "git_branch": {"returncode": 0,
                           "output": "codex/sm110-all-precision-closure\n"},
            "git_status": {"returncode": 0, "output": ""},
            "nvcc": {"returncode": 0, "output": "CUDA 13.0\n"},
            "nvidia_smi": {"returncode": 0, "output": "Thor\n"},
            "ncu": {"returncode": 0, "output": "NCU\n"},
            "power_mode": {"returncode": 0, "output": "MAXN\n"},
        }
        (root / "environment.json").write_text(json.dumps(environment))
        (root / "environment_snapshots.jsonl").write_text(
            json.dumps(environment) + "\n" + json.dumps(environment) + "\n"
        )
        plan = runner.plan_payload(self.manifest, self.cases)
        (root / "plan.json").write_text(json.dumps(plan))
        (root / "manifest_snapshot.json").write_text(json.dumps(self.manifest))
        spec = {
            "schema_version": 1, "run_id": run_id,
            "campaign": "sm110_tc5a_causal_pipeline_dag",
            "expected_commit": COMMIT,
            "generator": "microbench/sm110_gemm_causal_campaign/run_causal_campaign.py",
            "generator_sha256": dependency_hashes[
                "microbench/sm110_gemm_causal_campaign/run_causal_campaign.py"
            ],
            "contract_manifest":
                "microbench/sm110_gemm_causal_campaign/contract_manifest.json",
            "contract_manifest_sha256": dependency_hashes[
                "microbench/sm110_gemm_causal_campaign/contract_manifest.json"
            ],
            "source_dependencies": dependency_hashes,
            "precision_ids": ["fp16_f32"],
            "case_count": 91, "family_count": 13, "trials": 10,
            "trial_timeout_seconds": 120, "ncu_timeout_seconds": 300,
            "termination_grace_seconds": 5, "ncu_requested": True,
            "ncu_case_count": 4,
            "ncu_policy": "four predeclared k16 attribution cases",
            "static_only": False, "cases": self.cases,
        }
        (root / "run_spec.json").write_text(json.dumps(spec))
        binary = root / "build/tc5a_pipeline_dag"
        binary.write_bytes(b"synthetic binary")
        binary_sha = auditor.sha256_path(binary)
        (root / "build/binary.sha256").write_text(
            f"{binary_sha}  tc5a_pipeline_dag\n"
        )
        sass = root / "build/tc5a_pipeline_dag.sass.txt"
        sass.write_text("".join(
            f"Function : synthetic_tc5a_pipeline_dag_kernelILi{stages}EE\n"
            "UTMALDG.2D\nUTCHMMA\nUTCBAR\nLDTM.\n"
            for stages in (1, 2, 4)
        ))
        sass_sha = auditor.sha256_path(sass)
        sass_function_counts, sass_errors = auditor.sass_stage_function_counts(
            sass.read_text()
        )
        self.assertEqual(sass_errors, [])
        (root / "build/compile.log").write_text("ok\n")
        (root / "build/csv_header.txt").write_text(
            ",".join(auditor.CSV_FIELDS) + "\n"
        )
        compile_command = [
            "/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17",
            "-gencode", "arch=compute_110a,code=sm_110a", "-I",
            "/thor/repo/GEMMsm110/include",
            "/thor/repo/microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu",
            "-lcuda", "-o",
            f"/thor/repo/results/sm110_gemm_causal_campaign/{run_id}/build/"
            "tc5a_pipeline_dag",
        ]
        (root / "build/compile_command.json").write_text(
            json.dumps(compile_command)
        )
        (root / "build/artifact.json").write_text("{}\n")
        static_rows = []
        results = []
        for case in self.cases:
            case_id = str(case["case_id"])
            binary_path = (
                f"/thor/repo/results/sm110_gemm_causal_campaign/{run_id}/build/"
                "tc5a_pipeline_dag"
            )
            static_fields = auditor.expected_static_fields(case, self.manifest)
            static_fields.update({
                name: "0.000000000" if name in {"tma_interval_ns", "mma_interval_ns"}
                else "0" for name in auditor.CSV_FIELDS
                if name not in static_fields
            })
            static_rows.append({
                "case_id": case_id,
                "command": [binary_path, *case["args"][:-1],
                            "--contract-only", "--csv"],
                "fields": static_fields, "stdout": csv_text(static_fields),
            })
            case_dir = root / "cases" / case_id
            case_dir.mkdir(parents=True)
            trials = []
            metric_values = {name: [] for name in auditor.METRICS}
            for trial_index in range(1, 11):
                fields = csv_fields(case, self.manifest)
                metrics = {name: float(fields[name]) for name in auditor.METRICS}
                for name, value in metrics.items():
                    metric_values[name].append(value)
                trials.append({
                    "trial": trial_index,
                    "command": [binary_path, *case["args"]], "returncode": 0,
                    "timeout_seconds": 120, "timed_out": False,
                    "termination_failed": False, "raw_stdout": csv_text(fields),
                    "fields": fields, "audited_metrics": metrics,
                })
            (case_dir / "trials.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in trials)
            )
            fingerprint = auditor.sha256_json({
                "case": case, "binary_sha256": binary_sha,
                "source_sha256": dependency_hashes[
                    "microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu"
                ],
                "helper_sha256": dependency_hashes[
                    "GEMMsm110/include/sm110_ptx_helpers.cuh"
                ],
                "trial_count": 10, "ncu": bool(case["ncu_selected"]),
            })
            result = {
                "schema_version": 1, "case_id": case_id,
                "family_id": case["family_id"], "mode": case["mode"],
                "stages": case["stages"], "k_tiles": case["k_tiles"],
                "output_tasks": case["output_tasks"], "status": "ok",
                "fingerprint": fingerprint, "trial_count": 10,
                "metric_unit": "ns",
                "metric_stats": {
                    name: auditor.stats(values)
                    for name, values in metric_values.items()
                },
                "source_path":
                    "microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu",
                "source_sha256": dependency_hashes[
                    "microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu"
                ],
                "helper_path": "GEMMsm110/include/sm110_ptx_helpers.cuh",
                "helper_sha256": dependency_hashes[
                    "GEMMsm110/include/sm110_ptx_helpers.cuh"
                ],
                "binary_sha256": binary_sha,
                "sass_path": "build/tc5a_pipeline_dag.sass.txt",
                "sass_sha256": sass_sha,
                "sass_tokens": ["LDTM.", "UTCBAR", "UTCHMMA", "UTMALDG.2D"],
                "sass_function_counts": sass_function_counts,
                "trial_timeout_seconds": 120,
            }
            if case["ncu_selected"]:
                ncu_dir = case_dir / "ncu"
                ncu_dir.mkdir()
                report = ncu_dir / "profile.ncu-rep"
                log = ncu_dir / "profile.log"
                report.write_bytes(b"ncu")
                log.write_text("ok\n")
                result["ncu"] = {
                    "pass": True, "report_path": "ncu/profile.ncu-rep",
                    "report_sha256": auditor.sha256_path(report),
                    "log_sha256": auditor.sha256_path(log),
                    "command": [
                        "ncu", "--launch-skip", "3", "--launch-count", "1",
                        binary_path, *case["args"],
                    ],
                }
            else:
                result["ncu"] = {"selected": False, "pass": None}
            (case_dir / "result.json").write_text(json.dumps(result))
            results.append(result)
        (root / "static_contracts.json").write_text(json.dumps(static_rows))
        profile = runner.build_profile(results, self.manifest, run_id, COMMIT)
        self.assertTrue(profile["closure_qualified"])
        model_profile = pipeline_profiles_from_rows([profile])[0]
        model_profile.validate()
        self.assertEqual(
            model_profile.resource,
            "pipeline.tc5a_m128n256k64_stage4",
        )
        self.assertEqual(model_profile.precision_ids, ("fp16_f32",))
        (root / "pipeline_profile.json").write_text(json.dumps(profile))
        summary = {
            "schema_version": 1, "run_id": run_id,
            "expected_commit": COMMIT, "status": "complete", "case_count": 91,
            "trial_count": 910, "ncu_case_count": 4,
            "profile_qualified": True, "results": results,
        }
        (root / "summary.json").write_text(json.dumps(summary))
        (root / "campaign_status.json").write_text(json.dumps({
            "status": "complete", "completed_cases": 91,
            "total_cases": 91, "profile_qualified": True,
        }))
        (root / "progress.jsonl").write_text(
            json.dumps({"status": "complete"}) + "\n"
        )
        (root / "COMPLETE").write_text(
            f"run_id={run_id}\ncommit={COMMIT}\nprofile_qualified=true\n"
        )
        write_hash_manifest(root)
        return root, dependencies

    def test_complete_synthetic_bundle_passes_independent_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root, blobs = self._make_bundle(Path(temp))

            def loader(_commit: str, relative: str) -> bytes | None:
                return blobs.get(relative)

            result = auditor.audit(
                root, require_ncu=True, expected_commit=COMMIT,
                blob_loader=loader,
            )
            self.assertEqual(result["errors"], [])
            self.assertTrue(result["pass"])
            self.assertTrue(result["profile_qualified"])
            self.assertEqual(result["trial_count"], 910)

    def test_importer_reaudits_and_preserves_the_joint_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo_root = Path(temp)
            root, blobs = self._make_bundle(repo_root)
            for relative, payload in blobs.items():
                path = repo_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)

            class BoundAuditor:
                @staticmethod
                def audit(
                    run_dir: Path, *, require_ncu: bool,
                    expected_commit: str,
                ) -> dict[str, object]:
                    return auditor.audit(
                        run_dir,
                        require_ncu=require_ncu,
                        expected_commit=expected_commit,
                        blob_loader=lambda _commit, path: blobs.get(path),
                    )

            with mock.patch.object(
                causal_import,
                "_load_independent_auditor",
                return_value=BoundAuditor(),
            ):
                imported = causal_import.import_causal_profile(
                    repo_root=repo_root,
                    run_id=root.name,
                    expected_commit=COMMIT,
                )
        self.assertTrue(imported["audit"]["pass"])
        self.assertTrue(imported["profile_qualified"])
        self.assertEqual(imported["qualification"], "closure_qualified")
        self.assertEqual(len(imported["pipeline_profiles"]), 1)

    def test_timer_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root, blobs = self._make_bundle(Path(temp))
            path = root / "cases/full_s4_o32_k64/trials.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["fields"]["kernel_exit_ns"] = str(
                int(rows[0]["fields"]["kernel_exit_ns"]) + 1
            )
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            write_hash_manifest(root)

            def loader(_commit: str, relative: str) -> bytes | None:
                return blobs.get(relative)

            result = auditor.audit(
                root, require_ncu=True, expected_commit=COMMIT,
                blob_loader=loader,
            )
            self.assertFalse(result["pass"])
            self.assertTrue(any("raw stdout" in error or "kernel exit" in error
                                or "stored fields" in error
                                for error in result["errors"]))


class CausalPlatformSuiteTest(unittest.TestCase):
    @staticmethod
    def _make_suite(root: Path, *, after: int = 3) -> tuple[Path, dict[str, bytes]]:
        suite = root / "suite-a"
        suite.mkdir()
        blobs = {
            path: f"blob:{path}".encode()
            for path in suite_auditor.EXPECTED_PLATFORM_DEPENDENCIES
        }
        contract = {
            "schema_version": 1,
            "kind": "exact_tc5a_causal_pipeline_suite",
            "suite_id": "suite-a",
            "causal_run_id": "suite-a-causal",
            "expected_branch": suite_auditor.EXPECTED_BRANCH,
            "expected_commit": COMMIT,
            "ncu_required": True,
            "platform_dependencies": {
                path: hashlib.sha256(payload).hexdigest()
                for path, payload in blobs.items()
            },
        }
        (suite / "run_contract.json").write_text(json.dumps(contract))
        (suite / "preflight.txt").write_text(
            "=== git ===\n"
            f"{suite_auditor.EXPECTED_BRANCH}\n{COMMIT}\n"
            "=== nvpmodel ===\nNV Power Mode: MAXN\n"
            "min_freq=1575000000\nmax_freq=1575000000\n"
            "cur_freq=1575000000\ngovernor=performance\n"
        )
        (suite / "oc_before.tsv").write_text("/sys/oc1\t1\n")
        (suite / "oc_after.tsv").write_text(f"/sys/oc1\t{after}\n")
        (suite / "suite_launcher.log").write_text(
            "CAUSAL_CAMPAIGN_COMPLETE\nCAUSAL_SUITE_COMPLETE\n"
        )
        return suite, blobs

    def test_platform_audit_preserves_oc_warning_without_quarantining_data(self) -> None:
        class FakeAuditor:
            @staticmethod
            def audit(*_args: object, **_kwargs: object) -> dict[str, object]:
                return {
                    "pass": True,
                    "errors": [],
                    "warnings": [],
                    "profile_qualified": True,
                }

        with tempfile.TemporaryDirectory() as temp:
            suite, blobs = self._make_suite(Path(temp))
            with mock.patch.object(
                suite_auditor,
                "git_blob",
                side_effect=lambda _commit, path: blobs.get(path),
            ), mock.patch.object(
                suite_auditor,
                "load_campaign_auditor",
                return_value=FakeAuditor(),
            ):
                result = suite_auditor.audit_suite(
                    suite, expected_commit=COMMIT
                )
        self.assertTrue(result["pass"])
        self.assertTrue(result["profile_qualified"])
        self.assertEqual(result["overcurrent_deltas"], {"/sys/oc1": 2})
        self.assertIn("overcurrent_delta:/sys/oc1:2", result["warnings"])

    def test_platform_audit_rejects_counter_reset(self) -> None:
        class FakeAuditor:
            @staticmethod
            def audit(*_args: object, **_kwargs: object) -> dict[str, object]:
                return {"pass": True, "errors": [], "warnings": []}

        with tempfile.TemporaryDirectory() as temp:
            suite, blobs = self._make_suite(Path(temp), after=0)
            with mock.patch.object(
                suite_auditor,
                "git_blob",
                side_effect=lambda _commit, path: blobs.get(path),
            ), mock.patch.object(
                suite_auditor,
                "load_campaign_auditor",
                return_value=FakeAuditor(),
            ):
                result = suite_auditor.audit_suite(
                    suite, expected_commit=COMMIT
                )
        self.assertFalse(result["pass"])
        self.assertIn(
            "OC counters reset during the evidence interval",
            result["errors"],
        )


if __name__ == "__main__":
    unittest.main()

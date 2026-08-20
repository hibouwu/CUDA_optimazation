from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.sm110_gemm_model.io import (
    load_capacities, load_hardware, load_schedules,
)
from scripts.sm110_gemm_model.model import (
    Capacity,
    EvidenceKind,
    Hardware,
    PipelineProfile,
    Schedule,
    Workload,
    account_work,
    audit_inputs,
    evaluate,
    evaluate_manifest,
    precision_specs,
    predict_pipeline_worker_seconds,
    _resource_demands,
    _resolve_tma_capacity_resources,
    _select_capacity,
)
from scripts.sm110_gemm_model.observations import (
    ObservedBest,
    audit_observed_against_upper,
    summarize_observed_csvs,
)
from scripts.sm110_gemm_model.coverage import (
    campaign_measurement_coverage,
    common_resource_coverage,
    precision_coverage,
)
from scripts.sm110_gemm_model.precision_report import (
    build_precision_evidence_analysis,
    render_precision_evidence_markdown,
)
from scripts.sm110_gemm_model.completion import joint_pipeline_profile_matches
from scripts.sm110_gemm_model.tcgen05_descriptors import (
    DescriptorError,
    decode_fields,
    encode_block_scaled_fp4,
    encode_unscaled,
)
from microbench.sm110_gemm_campaign.run_compute_campaign import (
    DEFAULT_NCU_TIMEOUT_SECONDS as COMPUTE_NCU_TIMEOUT,
    DEFAULT_TRIAL_TIMEOUT_SECONDS as COMPUTE_TRIAL_TIMEOUT,
    PRECISIONS as CAMPAIGN_PRECISIONS,
    SCHEMA_VERSION as COMPUTE_SCHEMA_VERSION,
    make_manifest as make_compute_campaign_manifest,
    run_bounded as run_compute_bounded,
    source_for as make_compute_campaign_source,
)
from microbench.sm110_gemm_campaign.audit_campaign import audit as audit_compute_bundle


ROOT = Path(__file__).resolve().parents[2]
CAPACITY_PATH = Path(__file__).resolve().parent / "profiles" / "capacities.json"
SCHEDULE_PATH = Path(__file__).resolve().parent / "examples" / "schedules.json"
SUPPORT_MANIFEST_PATH = (
    ROOT / "microbench/sm110_full_gemm_campaign/support_manifest.json"
)
DOCUMENT_PATH = (ROOT / "Docs/blackwell_tensorcore/"
                 "thor_sm110_gemm_performance_bounds.md")
TUTORIAL_PATH = (ROOT / "Docs/blackwell_tensorcore/"
                 "thor_sm110_gemm_performance_model_tutorial.md")
CURRENT_REPLAY_PATH = (ROOT / "Docs/blackwell_tensorcore/"
                       "thor_sm110_current_model_replay.md")
PRECISION_MATRIX_PATH = (ROOT / "Docs/blackwell_tensorcore/"
                         "thor_sm110_all_precision_evidence_matrix.json")


def proven_l2_hardware(
    hardware_id: str = "thor",
    sm_count: int = 20,
    clock_hz: float = 1.575e9,
    operating_mode: str = "test",
    l2_capacity_bytes: int = 1 << 50,
) -> Hardware:
    return Hardware(
        hardware_id,
        sm_count,
        clock_hz,
        operating_mode,
        l2_capacity_bytes=l2_capacity_bytes,
        l2_capacity_evidence_kind="device_record",
        l2_capacity_source_path="synthetic/l2_capacity.json",
        l2_capacity_source_locator="l2_capacity_bytes",
    )


class DocumentContractTest(unittest.TestCase):
    def test_generated_precision_matrix_separates_solver_from_profile_evidence(self) -> None:
        matrix = json.loads(PRECISION_MATRIX_PATH.read_text())
        self.assertEqual(matrix["schema_version"], 3)
        self.assertTrue(matrix["causal_pipeline_dag_implemented"])
        self.assertEqual(matrix["causal_pipeline_closed_count"], 0)
        self.assertEqual(matrix["empirical_ideal_closed_count"], 0)
        self.assertEqual(matrix["end_to_end_closed_count"], 0)

    def test_current_model_replay_preserves_fail_closed_findings(self) -> None:
        text = CURRENT_REPLAY_PATH.read_text()
        self.assertIn("- audit pass：`False`", text)
        self.assertIn("- campaign measurement closed：`True`", text)
        self.assertIn("- all common resources closed：`True`", text)
        self.assertIn("- all precisions closed：`False`", text)
        self.assertIn("- causal DAG solver implemented：`True`", text)
        self.assertIn("- loaded pipeline profiles：0 项", text)
        self.assertIn(
            "- resource/causal/integrated complete observations：2/0/0",
            text,
        )
        self.assertEqual(
            text.count(
                "`residency_empirical_resource_prediction_incomplete`"
            ),
            13,
        )
        self.assertEqual(
            text.count("`residency_causal_pipeline_prediction_incomplete`"),
            15,
        )
        self.assertEqual(
            text.count("`residency_empirical_prediction_incomplete`"),
            15,
        )
        self.assertEqual(text.count("`overcurrent_events_observed`"), 1)
        self.assertIn(
            "`ok/ok` | 128.436 TFLOP/s–128.436 TFLOP/s | "
            "`insufficient_evidence/insufficient_evidence`",
            text,
        )
        self.assertIn("| Per cycle |", text)
        self.assertIn("1,024.000 B/cycle/GPU", text)
        self.assertIn("512.000 B/cycle/GPU", text)
        self.assertIn("122.772 B/cycle/SM", text)
        self.assertIn("l2_capacity_staircase.svg", text)
        self.assertIn("generic `tcgen05.cp`", text)
        self.assertIn("DSMEM topology/contention", text)

    def test_core_symbols_are_defined_at_first_use(self) -> None:
        text = DOCUMENT_PATH.read_text()
        definitions = {
            r"\(P_{\mathrm{obs}}\)": r"定义 \(P_{\mathrm{obs}}\)",
            r"\(P^\star\)": r"定义 \(P^\star\)",
            r"\(P_{\mathrm{ub}}\)": r"定义 \(P_{\mathrm{ub}}\)",
            r"\(\widehat P_{\mathrm{env}}\)":
                r"定义 \(\widehat P_{\mathrm{env}}\)",
            r"\(M\)": r"定义 \(M\)",
            r"\(N\)": r"定义 \(N\)",
            r"\(K\)": r"定义 \(K\)",
            r"\(A\)": r"定义 \(A\)",
            r"\(B\)": "定义 \\(A\\) 和\n\\(B\\)",
            r"\(C\)": r"定义 \(C\)",
            r"\(D\)": r"定义 \(D\)",
            r"\(\operatorname{op}(\cdot)\)":
                r"定义 \(\operatorname{op}(\cdot)\)",
            r"\(\alpha\)": "定义\n\\(\\alpha\\)",
            r"\(\beta\)": r"定义 \(\beta\)",
            r"\(s_{\mathrm{in}}\)": r"定义 \(s_{\mathrm{in}}\)",
            r"\(s_{\mathrm{acc}}\)": r"定义 \(s_{\mathrm{acc}}\)",
            r"\(s_{\mathrm{out}}\)": r"定义 \(s_{\mathrm{out}}\)",
            r"\(K_{\mathrm{mma}}\)": r"定义 \(K_{\mathrm{mma}}\)",
            r"\(W_{\mathrm{use}}\)": r"定义 \(W_{\mathrm{use}}\)",
            r"\(B_M\)": r"定义 \(B_M\)",
            r"\(B_N\)": r"定义 \(B_N\)",
            r"\(B_K\)": r"定义 \(B_K\)",
            r"\(N_M=": "定义\n\\(N_M=",
            r"\(N_N=": "定义\n\\(N_N=",
            r"\(N_K=": "定义\n\\(N_K=",
            r"W_{\mathrm{reduce}}": r"定义 \(W_{\mathrm{reduce}}\)",
            r"W_{\mathrm{issue}}": r"定义 \(W_{\mathrm{issue}}\)",
            r"\(\eta_{\mathrm{shape}}\)":
                r"定义形状效率 \(\eta_{\mathrm{shape}}\)",
            r"Q_{\mathrm{in,val}}^{\mathrm{LB}}":
                r"定义 \(Q_{\mathrm{in,val}}^{\mathrm{LB}}\)",
            r"\(b_s\)": r"定义 \(b_s\)",
            r"\(s_s\)": r"定义 \(s_s\)",
            r"Q_{\mathrm{in,scale}}^{\mathrm{LB}}":
                r"定义为 \(Q_{\mathrm{in,scale}}^{\mathrm{LB}}\)",
            r"Q_C^{\mathrm{LB}}": r"定义 \(Q_C^{\mathrm{LB}}\)",
            r"Q_D^{\mathrm{LB}}": r"定义 \(Q_D^{\mathrm{LB}}\)",
            r"Q_{\mathrm{TMA}}(x,w)":
                r"定义 \(Q_{\mathrm{TMA}}(x,w)\)",
            r"\(a_s=128\)": "定义\n\\(a_s=128\\)",
            r"\(g_s=4\)": "定义\n\\(g_s=4\\)",
            r"\(S(X,B_K,b_s,s_s)\)":
                "继续定义\n\\(S(X,B_K,b_s,s_s)\\)",
            r"Q_{\mathrm{TMA,scale/tile}}":
                "定义\n\\(Q_{\\mathrm{TMA,scale/tile}}\\)",
            r"Q_{\mathrm{TMA,scale}}(x,w)":
                r"定义 \(Q_{\mathrm{TMA,scale}}(x,w)",
            r"Q_{\mathrm{tmem,scale}}(x,w)":
                r"定义 \(Q_{\mathrm{tmem,scale}}(x,w)\)",
            r"Q_{\mathrm{TMA,unique}}(x,w)":
                r"定义 \(Q_{\mathrm{TMA,unique}}(x,w)\)",
            r"Q_{\mathrm{tmem}}(x,w)":
                r"定义 \(Q_{\mathrm{tmem}}(x,w)\)",
            r"\(\mathcal R\)": r"定义资源集合 \(\mathcal R\)",
            r"\(r\in\mathcal R\)": "定义\n\\(r\\in\\mathcal R\\)",
            r"Q_r^{\mathrm{LB}}": r"定义 \(Q_r^{\mathrm{LB}}\)",
            r"\(U_r\)": "定义\n\\(U_r\\)",
            r"T_r^{\mathrm{LB}}": r"定义 \(T_r^{\mathrm{LB}}\)",
            r"T_{\mathrm{resource}}^{\mathrm{LB}}":
                r"定义 \(T_{\mathrm{resource}}^{\mathrm{LB}}\)",
            r"\(n_t\)": r"定义 \(n_t\)",
            r"\(i\)": r"定义 \(i\)",
            r"\(p_i\)": r"定义 \(p_i\)",
            r"\(U_t\)": r"定义 \(U_t\)",
            r"\(T_{\mathrm{parallel}}\)":
                r"定义 \(T_{\mathrm{parallel}}\)",
            r"T_{\mathrm{parallel}}^{\mathrm{LB}}":
                r"定义 \(T_{\mathrm{parallel}}^{\mathrm{LB}}\)",
            r"\(p\)": r"定义 \(p\)",
            r"T_{\mathrm{parallel,identical}}":
                "定义\n\\(T_{\\mathrm{parallel,identical}}\\)",
            r"\(G=(V,E)\)": r"定义执行依赖图 \(G=(V,E)\)",
            r"\(V\)": r"其中 \(V\)",
            r"\(E\)": r"\(E\) 是生产者到消费者的真实依赖",
            r"T_{\mathrm{span}}^{\mathrm{LB}}":
                "定义\n\\(T_{\\mathrm{span}}^{\\mathrm{LB}}\\)",
            r"\(\mathbf y\)": r"定义资源吞吐向量 \(\mathbf y\)",
            r"\(\mathbf H\)": r"定义矩阵 \(\mathbf H\)",
            r"\(\mathbf c\)": r"定义向量 \(\mathbf c\)",
            r"T_{\mathrm{joint}}^{\mathrm{LB}}":
                "定义\n\\(T_{\\mathrm{joint}}^{\\mathrm{LB}}\\)",
            r"T_{\mathrm{fixed}}^{\mathrm{LB}}":
                r"定义 \(T_{\mathrm{fixed}}^{\mathrm{LB}}\)",
            r"T_{\mathrm{ub}}^{\mathrm{LB}}":
                r"定义 \(T_{\mathrm{ub}}^{\mathrm{LB}}\)",
            r"\(\widehat C_r\)": r"定义 \(\widehat C_r\)",
            r"\(w\)": r"定义 workload 描述 \(w\)",
            r"\(x\)": r"定义 schedule 描述 \(x\)",
            r"\(Q_r(x,w)\)": r"定义 \(Q_r(x,w)\)",
            r"\widehat T_{\mathrm{resource}}(x,w)":
                "定义\n\\(\\widehat T_{\\mathrm{resource}}(x,w)\\)",
            r"\widehat T_{\mathrm{parallel}}(x,w)":
                r"定义 \(\widehat T_{\mathrm{parallel}}(x,w)\)",
            r"\widehat T_{\mathrm{span}}(x,w)":
                r"定义 \(\widehat T_{\mathrm{parallel}}(x,w)\)",
            r"\widehat T_{\mathrm{joint}}(x,w)":
                r"定义 \(\widehat T_{\mathrm{parallel}}(x,w)\)",
            r"\widehat T_{\mathrm{fixed}}(x,w)":
                r"定义 \(\widehat T_{\mathrm{parallel}}(x,w)\)",
            r"\(\widehat T(x,w)\)": r"定义 \(\widehat T(x,w)\)",
            r"\(\mathcal X_{\mathrm{manifest}}\)":
                r"定义 \(\mathcal X_{\mathrm{manifest}}\)",
            r"\widehat T_{\mathrm{env}}(w)":
                r"定义 \(\widehat T_{\mathrm{env}}(w)\)",
            r"\(b\)": r"定义 \(b\)",
            r"\(j\)": r"定义 \(j\)",
            r"\(P_{b,j}\)": r"定义 \(P_{b,j}\)",
            r"P_{\mathrm{obs,median}}":
                r"定义 \(P_{\mathrm{obs,median}}\)",
        }
        for symbol, definition in definitions.items():
            with self.subTest(symbol=symbol):
                first_use = text.find(symbol)
                definition_start = text.find(definition)
                self.assertGreaterEqual(first_use, 0, f"missing symbol {symbol}")
                self.assertGreaterEqual(
                    definition_start, 0, f"missing definition for {symbol}")
                self.assertLessEqual(
                    definition_start, first_use,
                    f"{symbol} appears before its definition")


class TutorialContractTest(unittest.TestCase):
    def assert_defined_at_first_use(
        self, text: str, definitions: dict[str, str]
    ) -> None:
        for symbol, definition in definitions.items():
            with self.subTest(symbol=symbol):
                first_use = text.find(symbol)
                definition_start = text.find(definition)
                self.assertGreaterEqual(first_use, 0, f"missing symbol {symbol}")
                self.assertGreaterEqual(
                    definition_start, 0, f"missing definition for {symbol}")
                self.assertLessEqual(
                    definition_start, first_use,
                    f"{symbol} appears before its definition")

    def test_lesson1_core_symbols_are_defined_at_first_use(self) -> None:
        document = TUTORIAL_PATH.read_text()
        lesson = document.split("# 第 1 课：", 1)[1].split("# 第 2 课：", 1)[0]
        definitions = {
            r"\(A\)": r"- \(A\)：左输入矩阵",
            r"\(B\)": r"- \(B\)：右输入矩阵",
            r"\(C\)": r"- \(C\)：可选的原始输出矩阵",
            r"\(D\)": r"- \(D\)：最终输出矩阵",
            r"\(\alpha\)": r"- \(\alpha\)：乘积",
            r"\(\beta\)": r"- \(\beta\)：原始输出",
            r"\(M\)": r"- \(M\)：输出矩阵的行数",
            r"\(N\)": r"- \(N\)：输出矩阵的列数",
            r"\(K\)": r"- \(K\)：点积归约维度",
            r"\(i\)": r"定义 \(i\) 为输出矩阵的行索引",
            r"\(j\)": r"\(j\) 为输出矩阵的列索引",
            r"\(k\)": r"\(k\) 为当前点积的",
            r"\(W_{\mathrm{use}}\)":
                r"\(W_{\mathrm{use}}\) 为用户可见的数学计算工作量",
            r"\(W_{\mathrm{issued}}\)":
                r"定义 \(W_{\mathrm{issued}}\) 为硬件实际发出的计算工作量",
            r"\(T\)": r"定义 \(T\) 为一次设备端 GEMM 的执行时间",
            r"\(P\)": r"定义 \(P\) 为 GEMM 性能",
            r"\(T^{\mathrm{LB}}\)":
                r"定义 \(T^{\mathrm{LB}}\) 为任何满足当前条件",
            r"\(P_{\mathrm{ub}}\)":
                r"定义 \(P_{\mathrm{ub}}\) 为条件性能上界",
            r"\(s_{\mathrm{in}}": r"定义 \(s_{\mathrm{in}}",
            r"\(Q_A\)": r"定义 \(Q_A\) 为矩阵 A 的最低输入字节数",
            r"\(Q_B\)": r"定义 \(Q_B\) 为矩阵 B 的最低输入字节数",
            r"\(Q_{\mathrm{read}}^{\mathrm{LB}}\)":
                r"定义 \(Q_{\mathrm{read}}^{\mathrm{LB}}\)",
            r"\(s_{\mathrm{out}}": r"定义 \(s_{\mathrm{out}}",
            r"\(Q_{\mathrm{write}}^{\mathrm{LB}}\)":
                r"定义 \(Q_{\mathrm{write}}^{\mathrm{LB}}\)",
            r"\(Q_D": r"定义 \(Q_D",
            r"\(f_{\mathrm{GPU}}": r"定义 \(f_{\mathrm{GPU}}",
            r"\(c_{\mathrm{L2,read}}^{\mathrm{UB}}":
                r"定义 \(c_{\mathrm{L2,read}}^{\mathrm{UB}}",
            r"\(C_{\mathrm{L2,read}}^{\mathrm{UB}}\)":
                r"定义 \(C_{\mathrm{L2,read}}^{\mathrm{UB}}\)",
            r"\(T_{\mathrm{L2,read}}^{\mathrm{LB}}\)":
                (
                    r"定义 \(T_{\mathrm{tensor}}^{\mathrm{LB}}\)、" "\n"
                    r"\(T_{\mathrm{HBM}}^{\mathrm{LB}}\)、" "\n"
                    r"\(T_{\mathrm{L2,read}}^{\mathrm{LB}}\) 和" "\n"
                    r"\(T_{\mathrm{L2,write}}^{\mathrm{LB}}\) 分别为"
                ),
            r"\(c_{\mathrm{L2,write}}^{\mathrm{UB}}":
                r"定义 \(c_{\mathrm{L2,write}}^{\mathrm{UB}}",
            r"\(C_{\mathrm{L2,write}}^{\mathrm{UB}}\)":
                r"定义 \(C_{\mathrm{L2,write}}^{\mathrm{UB}}\)",
            r"\(T_{\mathrm{L2,write}}^{\mathrm{LB}}\)":
                (
                    r"定义 \(T_{\mathrm{tensor}}^{\mathrm{LB}}\)、" "\n"
                    r"\(T_{\mathrm{HBM}}^{\mathrm{LB}}\)、" "\n"
                    r"\(T_{\mathrm{L2,read}}^{\mathrm{LB}}\) 和" "\n"
                    r"\(T_{\mathrm{L2,write}}^{\mathrm{LB}}\) 分别为"
                ),
            r"\(C_{\mathrm{tensor,FP16}}^{\mathrm{UB}}":
                r"定义 \(C_{\mathrm{tensor,FP16}}^{\mathrm{UB}}",
            r"\(Q_{\mathrm{HBM,total}}^{\mathrm{LB}}\)":
                r"定义 \(Q_{\mathrm{HBM,total}}^{\mathrm{LB}}\)",
            r"\(C_{\mathrm{HBM,total}}^{\mathrm{UB}}":
                r"定义 \(C_{\mathrm{HBM,total}}^{\mathrm{UB}}",
            r"\(T_{\mathrm{cold}}^{\mathrm{LB}}\)":
                r"定义 \(T_{\mathrm{cold}}^{\mathrm{LB}}\)",
            r"\(P_{\mathrm{cold}}^{\mathrm{ub}}\)":
                r"定义 \(P_{\mathrm{cold}}^{\mathrm{ub}}\)",
            r"\(T_{\mathrm{hot}}^{\mathrm{LB}}\)":
                r"\(T_{\mathrm{hot}}^{\mathrm{LB}}\) 为该场景的总时间下界",
            r"\(P_{\mathrm{hot}}^{\mathrm{ub}}\)":
                r"定义 \(P_{\mathrm{hot}}^{\mathrm{ub}}\)",
            r"\(N_{\mathrm{SM}}": r"定义 \(N_{\mathrm{SM}}",
            r"\(R\)": r"定义 \(R\) 为整卡 L2 read 服务率",
            r"\(W\)": r"\(W\) 为整卡 L2 write 服务率",
        }
        self.assert_defined_at_first_use(lesson, definitions)

    def test_lesson2_core_symbols_are_defined_at_first_use(self) -> None:
        document = TUTORIAL_PATH.read_text()
        lesson = document.split("# 第 2 课：", 1)[1].split("# 第 3 课：", 1)[0]
        definitions = {
            r"\(\widehat T_{\mathrm{L2,shared}}\)":
                r"- \(\widehat T_{\mathrm{L2,shared}}\)：",
            r"\(\widehat T_{\mathrm{ingress,makespan}}\)":
                r"- \(\widehat T_{\mathrm{ingress,makespan}}\)：",
            r"\(\widehat T_{\mathrm{input}}\)":
                "定义\n\\(\\widehat T_{\\mathrm{input}}\\)",
            r"\(B_M": r"- \(B_M",
            r"\(B_N": r"- \(B_N",
            r"\(B_K": r"- \(B_K",
            r"\(S=": r"- \(S=",
            r"\(G_{\mathrm{CTA}}": r"- \(G_{\mathrm{CTA}}",
            r"\(N_M\)": r"定义 \(N_M\) 为 M 方向 output tile 数",
            r"\(N_N\)": r"定义 \(N_N\) 为 N 方向 output tile 数",
            r"\(N_{\mathrm{task}}\)":
                r"定义 \(N_{\mathrm{task}}\) 为完整 GEMM",
            r"\(N_K\)": r"定义 \(N_K\) 为一个 output task",
            r"\(q_{A,\mathrm{stage}}\)":
                r"定义 \(q_{A,\mathrm{stage}}\)",
            r"\(q_{B,\mathrm{stage}}\)":
                r"定义 \(q_{B,\mathrm{stage}}\)",
            r"\(q_{\mathrm{stage}}\)":
                r"定义 \(q_{\mathrm{stage}}\)",
            r"\(q_{\mathrm{task}}\)": r"定义 \(q_{\mathrm{task}}\)",
            r"\(Q_{\mathrm{L2,issued}}\)":
                r"定义 \(Q_{\mathrm{L2,issued}}\)",
            r"\(\widehat C_{\mathrm{L2,read}}":
                "定义\n\\(\\widehat C_{\\mathrm{L2,read}}",
            r"\(\widehat C_{\mathrm{ingress,SM}}":
                "定义\n\\(\\widehat C_{\\mathrm{ingress,SM}}",
            r"\(\widehat t_{\mathrm{task,ingress}}\)":
                r"定义 \(\widehat t_{\mathrm{task,ingress}}\)",
            r"\(N_{\mathrm{service}}\)":
                r"定义 \(N_{\mathrm{service}}\) 为能同时服务",
            r"\(N_{\mathrm{wave}}\)":
                r"定义 \(N_{\mathrm{wave}}\) 为服务全部",
        }
        self.assert_defined_at_first_use(lesson, definitions)

    def test_lesson3_symbols_are_defined_at_first_use(self) -> None:
        text = TUTORIAL_PATH.read_text()
        lesson = text.split(
            "# 第 3 课：useful、minimum、unique 与 issued work", 1
        )[1]
        definitions = {
            r"\(w\)": r"定义 \(w\)",
            r"\(x\)": r"定义 \(x\)",
            r"\(W_{\mathrm{use}}(w)\)":
                r"定义 \(W_{\mathrm{use}}(w)\)",
            r"\(W_{\mathrm{issued}}(x,w)\)":
                r"定义 \(W_{\mathrm{issued}}(x,w)\)",
            r"\(M_x\)": r"定义 \(M_x\)、\(N_x\)、\(K_x\)",
            r"\(N_x\)": r"定义 \(M_x\)、\(N_x\)、\(K_x\)",
            r"\(K_x\)": r"定义 \(M_x\)、\(N_x\)、\(K_x\)",
            r"\(\eta_{\mathrm{shape}}(x,w)\)":
                r"定义 \(\eta_{\mathrm{shape}}(x,w)\)",
            r"\(Q_{\mathrm{input,min}}(w)\)":
                r"定义 \(Q_{\mathrm{input,min}}(w)\)",
            r"\(Q_{\mathrm{TMA,unique}}(x,w)\)":
                r"定义 \(Q_{\mathrm{TMA,unique}}(x,w)\)",
            r"\(Q_{\mathrm{TMA,issued}}(x,w)\)":
                r"定义 \(Q_{\mathrm{TMA,issued}}(x,w)\)",
            r"\(Q_{A,\mathrm{issued}}(x,w)\)":
                r"定义 \(Q_{A,\mathrm{issued}}(x,w)\)",
            r"\(Q_{B,\mathrm{issued}}(x,w)\)":
                r"定义 \(Q_{B,\mathrm{issued}}(x,w)\)",
            r"\(a_{\mathrm{request}}(x,w)\)":
                "定义为无量纲比值\n\\(a_{\\mathrm{request}}(x,w)\\)",
            r"\(Q_{\mathrm{HBM,read,emp}}(x,w)\)":
                r"定义 \(Q_{\mathrm{HBM,read,emp}}(x,w)\)",
            r"\(Q_{\mathrm{L2,read,emp}}(x,w)\)":
                r"定义 \(Q_{\mathrm{L2,read,emp}}(x,w)\)",
            r"\(Q_{\mathrm{transaction}}(x,w)\)":
                r"定义 \(Q_{\mathrm{transaction}}(x,w)\)",
        }
        self.assert_defined_at_first_use(lesson, definitions)

    def test_lesson4_symbols_are_defined_at_first_use(self) -> None:
        document = TUTORIAL_PATH.read_text()
        lesson = document.split("# 第 4 课：", 1)[1]
        definitions = {
            r"\(u\)": r"定义 \(u\) 为 M 方向 output tile",
            r"\(v\)": r"定义 \(v\) 为 N 方向",
            r"\(\mathcal T(x,w)\)":
                r"定义 \(\mathcal T(x,w)\) 为 schedule",
            r"\(t\)": r"定义 \(t\) 为其中一个 task",
            r"\(n_t(x,w)": r"定义\n\(n_t(x,w)",
            r"\(r\)": r"定义 \(r\) 为当前分析的本地资源",
            r"\(q_{t,r}(x,w)\)": r"定义\n\(q_{t,r}(x,w)\)",
            r"\(\widehat C_{r,\mathrm{unit}}\)":
                r"定义 \(\widehat C_{r,\mathrm{unit}}\)",
            r"\(p_{t,r}(x,w)\)": r"定义 \(p_{t,r}(x,w)\)",
            r"\(U_r\)": r"定义 \(U_r\) 为能并行服务",
            r"\(\widehat T_{r,\mathrm{fractional}}\)":
                r"定义 \(\widehat T_{r,\mathrm{fractional}}\)",
            r"\(\widehat T_{r,\mathrm{span}}\)":
                r"定义 \(\widehat T_{r,\mathrm{span}}\)",
            r"\(p_r\)": r"定义 \(p_r\) 为所有 task 都相同时",
            r"\(N_{r,\mathrm{wave}}\)":
                r"定义\n\(N_{r,\mathrm{wave}}\)",
            r"\(\widehat T_{r,\mathrm{wave}}\)":
                r"定义 \(\widehat T_{r,\mathrm{wave}}\)",
            r"\(U_{\mathrm{local}}\)":
                r"定义 \(U_{\mathrm{local}}\) 为可并行",
            r"\(p_{\mathrm{ingress}}\)":
                r"定义 \(p_{\mathrm{ingress}}\)",
            r"\(\widehat T_{\mathrm{ingress,fractional}}\)":
                r"定义 \(\widehat T_{\mathrm{ingress,fractional}}\)",
            r"\(N_{\mathrm{ingress,wave}}\)":
                r"定义 \(N_{\mathrm{ingress,wave}}\)",
            r"\(\widehat T_{\mathrm{ingress,wave}}\)":
                r"定义 \(\widehat T_{\mathrm{ingress,wave}}\)",
            r"\(q_{\mathrm{small,task}}\)":
                r"定义本例单 task\nissued payload \(q_{\mathrm{small,task}}\)",
            r"\(p_{\mathrm{small,ingress}}\)":
                r"定义 \(p_{\mathrm{small,ingress}}\)",
            r"\(\widehat T_{\mathrm{small,ingress,wave}}\)":
                r"定义\n\(\widehat T_{\mathrm{small,ingress,wave}}\)",
        }
        definitions = {
            symbol: definition.replace(r"\n", "\n")
            for symbol, definition in definitions.items()
        }
        self.assert_defined_at_first_use(lesson, definitions)

    def test_lesson3_hand_calculations_match_executable_model(self) -> None:
        precision = precision_specs()["fp16_f32"]
        schedule = Schedule(
            "tc5a",
            128,
            256,
            64,
            4,
            mma_n=256,
            tail_policy="pad",
            tmem_columns=256,
            threads=192,
        )
        exact = account_work(
            Workload("exact", 2048, 2048, 2048, "fp16_f32"),
            schedule,
            precision,
        )
        self.assertEqual(exact.task_count, 128)
        self.assertEqual(exact.k_tiles, 32)
        self.assertEqual(exact.useful_compute_work, exact.issued_compute_work)
        self.assertEqual(exact.input_value_bytes_min, 16 * 1024**2)
        self.assertEqual(exact.tma_unique_input_bytes, 16 * 1024**2)
        self.assertEqual(exact.tma_input_bytes, 192 * 1024**2)

        irregular = account_work(
            Workload("irregular", 130, 260, 70, "fp16_f32"),
            schedule,
            precision,
        )
        self.assertEqual(irregular.task_count, 4)
        self.assertEqual(irregular.k_tiles, 2)
        self.assertEqual(irregular.useful_compute_work, 4_732_000)
        self.assertEqual(irregular.issued_compute_work, 33_554_432)
        self.assertAlmostEqual(
            irregular.shape_efficiency, 0.14102458953857422
        )
        self.assertEqual(irregular.input_value_bytes_min, 54_600)
        self.assertEqual(irregular.tma_unique_input_bytes, 196_608)
        self.assertEqual(irregular.tma_input_bytes, 393_216)

    def test_lesson4_wave_calculations_match_executable_model(self) -> None:
        precision = precision_specs()["fp16_f32"]
        schedule = Schedule(
            "tc5a",
            128,
            256,
            64,
            4,
            mma_n=256,
            tail_policy="pad",
            tmem_columns=256,
            threads=192,
        )
        ingress_rate = 193_366_116_675.77954
        l2_rate = 1_505_111_656_194.0369
        exact = account_work(
            Workload("exact", 2048, 2048, 2048, "fp16_f32"),
            schedule,
            precision,
        )
        exact_task_bytes = exact.tma_input_bytes / exact.task_count
        exact_task_span = exact_task_bytes / ingress_rate
        exact_waves = (exact.task_count + 20 - 1) // 20
        self.assertEqual(exact_task_bytes, 1.5 * 1024**2)
        self.assertEqual(exact_waves, 7)
        self.assertAlmostEqual(exact_task_span * 1e6, 8.134124152874465)
        self.assertAlmostEqual(
            exact_waves * exact_task_span * 1e6,
            56.93886907012125,
        )

        small = account_work(
            Workload("small", 130, 260, 70, "fp16_f32"),
            schedule,
            precision,
        )
        small_task_bytes = small.tma_input_bytes / small.task_count
        small_task_span = small_task_bytes / ingress_rate
        small_waves = (small.task_count + 20 - 1) // 20
        self.assertEqual(small_task_bytes, 96 * 1024)
        self.assertEqual(small_waves, 1)
        self.assertAlmostEqual(small_task_span * 1e6, 0.5083827595546541)
        self.assertAlmostEqual(
            small.tma_input_bytes / l2_rate * 1e6,
            0.2612537072460936,
        )

    def test_lesson5_symbols_are_defined_at_first_use(self) -> None:
        document = TUTORIAL_PATH.read_text()
        lesson = document.split("# 第 5 课：", 1)[1]
        definitions = {
            r"\(R_{\mathrm{stage}}\)":
                r"定义 \(R_{\mathrm{stage}}\)",
            r"\(I_{\max}\)": "定义\n\\(I_{\\max}\\)",
            r"\(R_{\mathrm{task}}\)":
                r"定义 \(R_{\mathrm{task}}\)",
            r"\(\lambda_L\)": r"定义 \(\lambda_L\)",
            r"\(\iota_L\)": r"定义 \(\iota_L\)",
            r"\(\lambda_M\)": r"定义 \(\lambda_M\)",
            r"\(\iota_M\)": r"定义 \(\iota_M\)",
            r"\(\widehat C_{\mathrm{load}}\)":
                r"定义 \(\widehat C_{\mathrm{load}}\)",
            r"\(i\)": r"定义 \(i\) 为当前 task 内的 K-tile",
            r"\(\mathsf L_i\)": r"定义 \(\mathsf L_i\)",
            r"\(\mathsf M_i\)": r"定义 \(\mathsf M_i\)",
            r"\(\mathsf R_{\mathrm{TMEM}}\)":
                r"定义 \(\mathsf R_{\mathrm{TMEM}}\)",
            r"\(\mathsf E_{\mathrm{epi}}\)":
                r"定义 \(\mathsf E_{\mathrm{epi}}\)",
            r"\(\mathsf S_D\)": "定义\n\\(\\mathsf S_D\\)",
            r"\(G_{\mathrm{pipe}}":
                r"定义 \(G_{\mathrm{pipe}}",
            r"\(T_{\mathrm{span}}(x,w)\)":
                "定义\n\\(T_{\\mathrm{span}}(x,w)\\)",
            r"\(\ell\)": r"定义 \(\ell\)",
            r"\(c\)": "定义\n\\(c\\)",
            r"\(T_{\mathrm{toy},S=1}\)":
                r"定义 \(T_{\mathrm{toy},S=1}\)",
            r"\(T_{\mathrm{toy},S\ge2}\)":
                r"定义 \(T_{\mathrm{toy},S\ge2}\)",
            r"\(\rho_4\)": r"定义 \(\rho_4\)",
            r"\(\rho_8\)": r"定义 \(\rho_8\)",
            r"\(\widehat T_{\mathrm{resource}}(x,w)\)":
                r"定义 \(\widehat T_{\mathrm{resource}}(x,w)\)",
            r"\(\widehat T_{\mathrm{DAG}}(x,w)\)":
                r"定义 \(\widehat T_{\mathrm{DAG}}(x,w)\)",
            r"\(\widehat T_{\mathrm{schedule}}(x,w)\)":
                r"定义 \(\widehat T_{\mathrm{schedule}}(x,w)\)",
        }
        definitions = {
            symbol: definition.replace(r"\n", "\n")
            for symbol, definition in definitions.items()
        }
        self.assert_defined_at_first_use(lesson, definitions)

    def test_lesson5_toy_pipeline_and_request_counts(self) -> None:
        def toy_pipeline(
            k_tiles: int, load_us: float, compute_us: float, stages: int
        ) -> float:
            if stages == 1:
                return k_tiles * (load_us + compute_us)
            return (
                load_us
                + (k_tiles - 1) * max(load_us, compute_us)
                + compute_us
            )

        self.assertAlmostEqual(toy_pipeline(4, 0.5, 0.8, 1), 5.2)
        self.assertAlmostEqual(toy_pipeline(4, 0.5, 0.8, 2), 3.7)
        self.assertAlmostEqual(toy_pipeline(4, 0.5, 0.8, 4), 3.7)
        self.assertEqual(32 * 2, 64)
        self.assertEqual(4 * 2, 8)
        self.assertAlmostEqual(129.398 / 68.615, 1.8858558624207535)
        self.assertAlmostEqual(193.366 / 68.615, 2.8181301464694313)

    def test_lesson6_symbols_are_defined_at_first_use(self) -> None:
        document = TUTORIAL_PATH.read_text()
        lesson = document.split("# 第 6 课：", 1)[1].split(
            "# 第 7 课：", 1
        )[0]
        definitions = {
            r"\(\mathcal X(w)\)":
                r"定义 \(\mathcal X(w)\)",
            r"\(T(x,w)\)": r"定义 \(T(x,w)\)",
            r"\(T^\star(w)\)": r"定义 \(T^\star(w)\)",
            r"\(P^\star(w)\)": r"定义 \(P^\star(w)\)",
            r"\(\mathcal R_{\mathrm{strict}}(w)\)":
                r"定义 \(\mathcal R_{\mathrm{strict}}(w)\)",
            r"\(T_{\mathrm{ub}}^{\mathrm{LB}}(w)\)":
                r"定义 \(T_{\mathrm{ub}}^{\mathrm{LB}}(w)\)",
            r"\(P_{\mathrm{ub}}(w)\)":
                r"定义 \(P_{\mathrm{ub}}(w)\)",
            r"\(\mathcal S_{\mathrm{v1}}(w)\)":
                r"定义 \(\mathcal S_{\mathrm{v1}}(w)\)",
            r"\(\widehat P(x,w)\)":
                r"定义 \(\widehat P(x,w)\)",
            r"\(\widehat P_{\mathrm{env}}(w)\)":
                r"定义 \(\widehat P_{\mathrm{env}}(w)\)",
            r"\(\mathcal O(w)\)": r"定义 \(\mathcal O(w)\)",
            r"\(P_{o,\mathrm{median}}(w)\)":
                r"定义 \(P_{o,\mathrm{median}}(w)\)",
            r"\(P_{\mathrm{obs}}(w)\)":
                r"定义 \(P_{\mathrm{obs}}(w)\)",
            r"\(P_{o,\max}(w)\)":
                "定义\n\\(P_{o,\\max}(w)\\)",
            r"\(P_{\mathrm{tc5a}}": r"定义 \(P_{\mathrm{tc5a}}",
            r"\(P_{\mathrm{cuBLAS}}":
                r"定义 \(P_{\mathrm{cuBLAS}}",
            r"\(N_{\mathrm{impl}}\)":
                r"- \(N_{\mathrm{impl}}\)：",
            r"\(N_{\mathrm{numeric}}\)":
                r"- \(N_{\mathrm{numeric}}\)：",
            r"\(N_{\mathrm{env}}\)":
                r"- \(N_{\mathrm{env}}\)：",
            r"\(N_{\mathrm{DAG}}\)":
                r"- \(N_{\mathrm{DAG}}\)：",
            r"\(N_{\mathrm{e2e}}\)":
                r"- \(N_{\mathrm{e2e}}\)：",
        }
        definitions = {
            symbol: definition.replace(r"\n", "\n")
            for symbol, definition in definitions.items()
        }
        self.assert_defined_at_first_use(lesson, definitions)

    def test_lesson6_three_layer_numbers_match_report(self) -> None:
        useful = 2 * 2048**3
        empirical = 128.43619466189114e12
        tc5a = 120.0389157918936e12
        cublas = 130.6325516802194e12
        cublas_max = 131.16327385421102e12
        self.assertAlmostEqual(useful / empirical * 1e6, 133.76189811)
        self.assertAlmostEqual(tc5a / cublas, 0.9189050833649917)
        self.assertAlmostEqual(tc5a / empirical, 0.934619062079008)
        self.assertAlmostEqual(cublas / empirical, 1.0171007637224865)
        self.assertLess(cublas_max, 139.776e12)

    def test_lesson7_symbols_are_defined_at_first_use(self) -> None:
        document = TUTORIAL_PATH.read_text()
        lesson = document.split("# 第 7 课：", 1)[1].split(
            "# 第 8 课：", 1
        )[0]
        definitions = {
            r"\(d\)": r"定义 \(d\)",
            r"\(\mathbf q": "定义\n\\(\\mathbf q",
            r"\(\mathbf y": r"定义 \(\mathbf y",
            r"\(\mathcal F\)": r"定义 \(\mathcal F\)",
            r"\(\mathcal C^{\mathrm{UB}}\)":
                "定义\n\\(\\mathcal C^{\\mathrm{UB}}\\)",
            r"\(T_{\mathrm{joint}}^{\mathrm{LB}}":
                r"定义 \(T_{\mathrm{joint}}^{\mathrm{LB}}",
            r"\(J\)": r"定义 \(J\)",
            r"\(\mathbf a_j\)": r"定义 \(\mathbf a_j\)",
            r"\(b_j\)": r"定义 \(b_j\)",
            r"\(T_{\mathrm{linear}}^{\mathrm{LB}}\)":
                r"定义 \(T_{\mathrm{linear}}^{\mathrm{LB}}\)",
            r"\(R\)": r"定义 \(R\) 和 \(W\)",
            r"\(W\)": r"定义 \(R\) 和 \(W\)",
            r"\(C_R^{\mathrm{UB}}\)":
                "定义\n\\(C_R^{\\mathrm{UB}}\\)",
            r"\(C_W^{\mathrm{UB}}\)":
                r"\(C_W^{\mathrm{UB}}\) 为两个方向",
            r"\(Q_R\)": r"定义 \(Q_R\) 为一次 GEMM 的 read",
            r"\(Q_W\)": r"定义 \(Q_W\) 为同一次 GEMM",
            r"\(C_{R+W}^{\mathrm{UB}}\)":
                r"定义 \(C_{R+W}^{\mathrm{UB}}\)",
            r"\(\widehat C_{\mathrm{HBM,R}}":
                r"定义 \(\widehat C_{\mathrm{HBM,R}}",
            r"\(\widehat C_{\mathrm{HBM,W}}":
                "定义\n\\(\\widehat C_{\\mathrm{HBM,W}}",
            r"\(T_{\mathrm{L2}}\)":
                r"定义 \(T_{\mathrm{L2}}\)",
            r"\(T_{\mathrm{TMA}}\)":
                "定义\n\\(T_{\\mathrm{TMA}}\\)",
            r"\(\gamma": r"定义 \(\gamma",
        }
        definitions = {
            symbol: definition.replace(r"\n", "\n")
            for symbol, definition in definitions.items()
        }
        self.assert_defined_at_first_use(lesson, definitions)

    def test_lesson7_joint_capacity_numbers(self) -> None:
        q = 16 * 1024**2
        hbm_read_seconds = q / 253.588e9
        hbm_write_seconds = q / 201.158e9
        hbm_total_seconds = 2 * q / 273e9
        self.assertAlmostEqual(hbm_read_seconds * 1e6, 66.1593450794)
        self.assertAlmostEqual(hbm_write_seconds * 1e6, 83.4031756132)
        self.assertAlmostEqual(hbm_total_seconds * 1e6, 122.9100073260)
        l2_read_seconds = 192 * 1024**2 / 1_505_111_656_194.0369
        l2_write_seconds = 16 * 1024**2 / 545.416e9
        self.assertAlmostEqual(l2_read_seconds * 1e6, 133.76189811)
        self.assertAlmostEqual(
            (l2_read_seconds + l2_write_seconds) * 1e6,
            164.5223011785,
        )

    def test_lesson8_symbols_are_defined_at_first_use(self) -> None:
        document = TUTORIAL_PATH.read_text()
        lesson = document.split("# 第 8 课：", 1)[1].split(
            "# 第 9 课：", 1
        )[0]
        definitions = {
            r"\(b_v\)": r"定义 \(b_v\)",
            r"\(s_v": "定义\n\\(s_v",
            r"\(s_a\)": r"定义 \(s_a\)",
            r"\(s_o\)": r"定义 \(s_o\)",
            r"\(K_{\mathrm{mma}}\)":
                r"定义 \(K_{\mathrm{mma}}\)",
            r"\(Q_{\mathrm{value,min}}(w)\)":
                r"定义 \(Q_{\mathrm{value,min}}(w)\)",
            r"\(Q_{D,\min}\)": "定义\n\\(Q_{D,\\min}\\)",
            r"\(s_{\mathrm{transport}}(x)\)":
                r"定义 \(s_{\mathrm{transport}}(x)\)",
            r"\(g\)": r"定义 \(g\)",
            r"\(s_s\)": r"定义 \(s_s\)",
            r"\(Q_{\mathrm{scale,min}}(w)\)":
                r"定义 \(Q_{\mathrm{scale,min}}(w)\)",
            r"\(V\)": r"定义 \(V\) 为当前 scale tensor",
            r"\(V_{128}": "定义\n\\(V_{128}",
            r"\(G_4": "定义\n\\(G_4",
            r"\(Q_{\mathrm{TMEM,scale}}(x,w)\)":
                "定义\n\\(Q_{\\mathrm{TMEM,scale}}(x,w)\\)",
        }
        definitions = {
            symbol: definition.replace(r"\n", "\n")
            for symbol, definition in definitions.items()
        }
        self.assert_defined_at_first_use(lesson, definitions)

    def test_lesson8_precision_bytes_match_executable_model(self) -> None:
        specs = precision_specs()
        expected_mib = {
            "fp16_f32": (16.0, 0.0),
            "tf32_f32": (32.0, 0.0),
            "e3m2_f32": (6.0, 0.0),
            "e2m1_f32": (4.0, 0.0),
            "mxfp4_f32": (4.0, 0.25),
            "nvfp4_f32": (4.0, 0.5),
            "s8_s32": (8.0, 0.0),
        }
        m = n = k = 2048
        for precision_id, (expected_value, expected_scale) in expected_mib.items():
            spec = specs[precision_id]
            value = (m * k + k * n) * spec.input_bytes
            scale = 0
            if spec.input_scale_block is not None:
                groups = (
                    k + spec.input_scale_block - 1
                ) // spec.input_scale_block
                scale = (
                    m * groups * spec.input_scale_bytes
                    + n * groups * spec.input_scale_bytes
                )
            with self.subTest(precision_id=precision_id):
                self.assertAlmostEqual(value / 2**20, expected_value)
                self.assertAlmostEqual(scale / 2**20, expected_scale)

    def test_lesson9_symbols_are_defined_at_first_use(self) -> None:
        document = TUTORIAL_PATH.read_text()
        lesson = document.split("# 第 9 课：", 1)[1].split(
            "# 附录 A：", 1
        )[0]
        definitions = {
            r"\(c\)": r"定义 \(c\) 为一条模型 capacity",
            r"\(\kappa\)": r"定义 \(\kappa\)",
            r"\(\kappa_x\)": r"定义 \(\kappa_x\)",
            r"\(\kappa_c\)": "定义\n\\(\\kappa_c\\)",
            r"\(b\)": r"定义 \(b\) 为网格中 CTA",
            r"\(t_{b,\mathrm{start}}\)":
                r"定义 \(t_{b,\mathrm{start}}\)",
            r"\(t_{b,\mathrm{stop}}\)":
                r"\(t_{b,\mathrm{stop}}\) 为第",
            r"\(T_{\mathrm{grid}}\)":
                "定义\n\\(T_{\\mathrm{grid}}\\)",
            r"\(n\)": r"定义 \(n\) 为同一冻结合同",
            r"\(p_1": "定义\n\\(p_1",
            r"\(\widetilde p\)": r"定义 \(\widetilde p\)",
            r"\(p_{\max}": r"定义 \(p_{\max}",
            r"\(\mathcal A\)": r"定义 \(\mathcal A\)",
        }
        definitions = {
            symbol: definition.replace(r"\n", "\n")
            for symbol, definition in definitions.items()
        }
        self.assert_defined_at_first_use(lesson, definitions)

    def test_all_tutorial_local_links_resolve(self) -> None:
        text = TUTORIAL_PATH.read_text()
        targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
        local_targets = [
            target.split("#", 1)[0]
            for target in targets
            if not target.startswith(("https://", "http://", "mailto:"))
        ]
        for target in local_targets:
            with self.subTest(target=target):
                self.assertTrue(
                    (TUTORIAL_PATH.parent / target).resolve().exists(),
                    f"broken tutorial link: {target}",
                )


class WorkAccountingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = Schedule(
            "s",
            128,
            128,
            64,
            2,
            tail_policy="pad",
            tma_ingress_capacity_resource=
                "tma.smem_ingress.per_sm.inflight4",
            tma_hbm_capacity_resource="tma.hbm.inflight4",
        )
        self.precisions = precision_specs()

    def test_beta_zero_eliminates_c_read(self) -> None:
        workload = Workload("w", 128, 128, 64, "fp16_f32", beta=0.0)
        work = account_work(workload, self.schedule, self.precisions["fp16_f32"])
        self.assertEqual(work.c_read_bytes_min, 0)

    def test_beta_nonzero_requires_c_read(self) -> None:
        workload = Workload("w", 128, 128, 64, "fp16_f32", beta=1.0)
        work = account_work(workload, self.schedule, self.precisions["fp16_f32"])
        self.assertEqual(work.c_read_bytes_min, 128 * 128 * 4)

    def test_padded_issue_work_is_not_less_than_useful_work(self) -> None:
        workload = Workload("tail", 129, 130, 65, "fp16_f32")
        work = account_work(workload, self.schedule, self.precisions["fp16_f32"])
        self.assertGreater(work.issued_compute_work, work.useful_compute_work)
        self.assertLess(work.shape_efficiency, 1.0)

    def test_fp6_packed_storage_is_three_quarters_byte(self) -> None:
        workload = Workload("fp6", 128, 128, 64, "e3m2_f32")
        work = account_work(
            workload,
            Schedule("fp6-byte", 128, 128, 64, 2,
                     tail_policy="pad", input_transport_layout="byte_padded"),
            self.precisions["e3m2_f32"],
        )
        self.assertEqual(work.input_value_bytes_min, (128 * 64 + 64 * 128) * 0.75)

    def test_fp6_transport_padding_is_not_charged_as_logical_storage(self) -> None:
        workload = Workload("fp6-cp", 128, 128, 64, "e3m2_f32")
        byte_container = account_work(
            workload,
            Schedule(
                "fp6-byte",
                128,
                128,
                64,
                2,
                tail_policy="pad",
                input_transport_layout="byte_padded",
            ),
            self.precisions["e3m2_f32"],
        )
        decompressed = account_work(
            workload,
            Schedule(
                "fp6-cp",
                128,
                128,
                64,
                2,
                tail_policy="pad",
                input_transport_layout="b6x16_p32",
            ),
            self.precisions["e3m2_f32"],
        )
        self.assertEqual(
            decompressed.input_value_bytes_min,
            byte_container.input_value_bytes_min,
        )
        self.assertEqual(
            decompressed.tma_input_bytes,
            128 * 64 + 64 * 128,
        )

    def test_transport_layout_must_match_precision(self) -> None:
        with self.assertRaisesRegex(Exception, "only valid for six-bit"):
            account_work(
                Workload("bad-layout", 128, 128, 64, "fp16_f32"),
                Schedule(
                    "bad-layout",
                    128,
                    128,
                    64,
                    2,
                    input_transport_layout="b6x16_p32",
                ),
                self.precisions["fp16_f32"],
            )

    def test_unmodeled_schedule_families_fail_closed(self) -> None:
        workload = Workload("w", 128, 128, 64, "fp16_f32")
        for schedule, pattern in (
            (Schedule("g2", 128, 128, 64, 2, cta_group=2), "CTA-group-2"),
            (Schedule("split", 128, 128, 64, 2, split_k=2), "split-K"),
            (Schedule("persistent", 128, 128, 64, 2, persistent=True), "persistent"),
            (Schedule("non-tma", 128, 128, 64, 2, uses_tma=False), "non-TMA"),
        ):
            with self.subTest(schedule=schedule.schedule_id):
                with self.assertRaisesRegex(Exception, pattern):
                    account_work(workload, schedule, self.precisions["fp16_f32"])

    def test_fractional_direct_smem_requires_byte_container(self) -> None:
        workload = Workload("fp6", 128, 256, 64, "e3m2_f32")
        with self.assertRaisesRegex(Exception, "uses byte containers"):
            account_work(
                workload,
                Schedule("packed", 128, 256, 64, 2, mma_n=256,
                         tmem_columns=256),
                self.precisions["e3m2_f32"],
            )
        work = account_work(
            workload,
            Schedule("byte", 128, 256, 64, 2, mma_n=256,
                     tmem_columns=256, input_transport_layout="byte_padded"),
            self.precisions["e3m2_f32"],
        )
        self.assertEqual(work.input_value_bytes_min, (128 + 256) * 64 * 0.75)
        self.assertEqual(work.tma_unique_input_bytes, (128 + 256) * 64)

    def test_block_scaled_schedule_reserves_accumulator_and_scale_tmem(self) -> None:
        workload = Workload("nv", 128, 256, 64, "nvfp4_f32")
        with self.assertRaisesRegex(Exception, "512-column TMEM"):
            account_work(
                workload,
                Schedule("small-tmem", 128, 256, 64, 2, mma_n=256,
                         tmem_columns=256),
                self.precisions["nvfp4_f32"],
            )
        account_work(
            workload,
            Schedule("full-tmem", 128, 256, 64, 2, mma_n=256,
                     tmem_columns=512),
            self.precisions["nvfp4_f32"],
        )

    def test_int8_n_shape_uses_table39_nonuniform_rule(self) -> None:
        workload = Workload("i8", 128, 40, 32, "s8_s32")
        with self.assertRaisesRegex(Exception, "MMA N=40 is invalid"):
            account_work(
                workload,
                Schedule("bad-i8-n", 128, 40, 32, 2, mma_n=40, tmem_columns=64),
                self.precisions["s8_s32"],
            )

    def test_quantized_scale_bytes_are_separate_from_values(self) -> None:
        workload = Workload("nv", 128, 128, 64, "nvfp4_f32")
        schedule = Schedule(
            "nv", 128, 128, 64, 2, tail_policy="pad", tmem_columns=512)
        work = account_work(workload, schedule, self.precisions["nvfp4_f32"])
        self.assertGreater(work.input_scale_bytes_min, 0)
        self.assertEqual(work.accumulator_readback_bytes, 128 * 128 * 4)

    def test_block_scales_do_not_cross_k_vectors(self) -> None:
        workload = Workload("nv-irregular", 2, 3, 17, "nvfp4_f32")
        schedule = Schedule(
            "nv-pad",
            128,
            128,
            64,
            2,
            tail_policy="pad",
            tmem_columns=512,
        )
        work = account_work(workload, schedule, self.precisions["nvfp4_f32"])
        self.assertEqual(work.input_scale_bytes_min, (2 + 3) * 2)
        self.assertEqual(
            work.tma_input_bytes,
            128 * 64 * 0.5 + 64 * 128 * 0.5 + (128 + 128) * 4,
        )
        self.assertEqual(
            work.tma_unique_input_bytes,
            128 * 64 * 0.5 + 64 * 128 * 0.5 + (128 + 128) * 4,
        )
        self.assertEqual(work.tma_scale_input_bytes, (128 + 128) * 4)
        self.assertEqual(
            work.tmem_scale_ingress_bytes,
            work.tma_scale_input_bytes,
        )

    def test_nvfp4_scale_transport_pads_n64_to_128_vectors(self) -> None:
        workload = Workload("nv-n64", 128, 64, 64, "nvfp4_f32")
        schedule = Schedule(
            "nv-n64",
            128,
            64,
            64,
            2,
            mma_n=64,
            tail_policy="exact",
            tmem_columns=512,
        )
        work = account_work(workload, schedule, self.precisions["nvfp4_f32"])
        # Each SFA/SFB tile is physically 128 vectors x 4 scale groups.
        self.assertEqual(work.input_scale_bytes_min, (128 + 64) * 4)
        self.assertEqual(work.tma_scale_input_bytes, 512 + 512)
        self.assertEqual(work.tmem_scale_ingress_bytes, 1024)

    def test_mxfp4_scale_transport_pads_two_groups_to_four(self) -> None:
        workload = Workload("mx-n256", 128, 256, 64, "mxfp4_f32")
        schedule = Schedule(
            "mx-n256",
            128,
            256,
            64,
            2,
            mma_n=256,
            tail_policy="exact",
            tmem_columns=512,
        )
        work = account_work(workload, schedule, self.precisions["mxfp4_f32"])
        # MXFP4 has two logical K scale groups at BK=64, but the physical
        # Swizzle32x4x4 atom pads that mode to four groups.
        self.assertEqual(work.input_scale_bytes_min, (128 + 256) * 2)
        self.assertEqual(work.tma_scale_input_bytes, 512 + 1024)
        self.assertEqual(work.tma_a_value_bytes, 4096)
        self.assertEqual(work.tma_b_value_bytes, 8192)
        self.assertEqual(work.tma_a_scale_bytes, 512)
        self.assertEqual(work.tma_b_scale_bytes, 1024)
        self.assertEqual(work.tmem_scale_ingress_bytes, 1536)

    def test_padded_tail_charges_issued_tmem_readback_but_useful_store(self) -> None:
        workload = Workload("tail-readback", 129, 130, 64, "fp16_f32")
        work = account_work(workload, self.schedule, self.precisions["fp16_f32"])
        self.assertEqual(work.accumulator_readback_bytes, 256 * 256 * 4)
        self.assertEqual(work.output_value_bytes_min, 129 * 130 * 4)

    def test_exact_shape_tmem_readback_equals_useful_output_extent(self) -> None:
        workload = Workload("exact-readback", 128, 128, 64, "fp16_f32")
        schedule = Schedule("exact", 128, 128, 64, 2, tail_policy="exact")
        work = account_work(workload, schedule, self.precisions["fp16_f32"])
        self.assertEqual(work.accumulator_readback_bytes, 128 * 128 * 4)

    def test_cold_schedule_separates_unique_hbm_from_repeated_l2_tma(self) -> None:
        workload = Workload("reuse", 256, 256, 64, "fp16_f32")
        work = account_work(workload, self.schedule, self.precisions["fp16_f32"])
        demands = _resource_demands(
            workload,
            self.schedule,
            work,
            self.precisions["fp16_f32"],
            empirical=True,
        )
        self.assertEqual(work.tma_unique_input_bytes, 2 * 256 * 64 * 2)
        self.assertEqual(work.tma_input_bytes, 4 * 2 * 128 * 64 * 2)
        write_bytes = work.output_value_bytes_min + work.output_scale_bytes_min
        self.assertEqual(
            demands["hbm.duplex"][0],
            work.tma_unique_input_bytes + write_bytes,
        )
        self.assertEqual(
            demands["tma.hbm.inflight4"][0],
            work.tma_unique_input_bytes,
        )
        self.assertEqual(
            demands["l2.duplex"][0], work.tma_input_bytes + write_bytes
        )
        self.assertNotIn("tma.l2", demands)

    def test_cold_strict_schedule_keeps_shared_l2_minimum_demands(self) -> None:
        workload = Workload("cold-strict", 256, 256, 64, "fp16_f32")
        work = account_work(workload, self.schedule, self.precisions["fp16_f32"])
        demands = _resource_demands(
            workload,
            self.schedule,
            work,
            self.precisions["fp16_f32"],
            empirical=False,
        )
        read_min = (
            work.input_value_bytes_min
            + work.input_scale_bytes_min
            + work.c_read_bytes_min
        )
        write_min = work.output_value_bytes_min + work.output_scale_bytes_min
        self.assertEqual(demands["l2.read"], (read_min, "byte"))
        self.assertEqual(demands["l2.write"], (write_min, "byte"))
        self.assertEqual(
            demands["hbm.total"], (read_min + write_min, "byte"))

    def test_hot_schedule_has_no_hbm_demand(self) -> None:
        workload = Workload(
            "hot-reuse", 256, 256, 64, "fp16_f32", residency="hot_l2")
        work = account_work(workload, self.schedule, self.precisions["fp16_f32"])
        demands = _resource_demands(
            workload,
            self.schedule,
            work,
            self.precisions["fp16_f32"],
            empirical=True,
        )
        self.assertNotIn("hbm.duplex", demands)
        self.assertNotIn("tma.hbm", demands)
        write_bytes = work.output_value_bytes_min + work.output_scale_bytes_min
        self.assertEqual(
            demands["l2.duplex"][0], work.tma_input_bytes + write_bytes
        )
        self.assertNotIn("tma.l2", demands)

    def test_per_sm_tma_ingress_uses_slowest_wave_makespan(self) -> None:
        workload = Workload(
            "tc5a-like", 2048, 2048, 2048, "fp16_f32",
            residency="hot_l2")
        schedule = Schedule(
            "tc5a", 128, 256, 64, 4,
            mma_n=256,
            tmem_columns=256,
            tma_ingress_capacity_resource="tma.smem_ingress.per_sm",
        )
        per_sm_rate = 80e9
        capacities = [
            Capacity(
                "compute", "tensor.fp16.m128n256", 1e30, "flop",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "compute"),
            Capacity(
                "l2_duplex", "l2.duplex", 1e30, "byte",
                EvidenceKind.MEASURED_JOINT,
                "test", "source.json", "l2-duplex"),
            Capacity(
                "tma_per_sm", "tma.smem_ingress.per_sm", per_sm_rate,
                "byte", EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "tma-per-sm"),
            Capacity(
                "readback", "tmem.readback", 1e30, "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "readback"),
        ]
        result = evaluate(
            workload, schedule, proven_l2_hardware(), capacities)
        work = result.work
        self.assertEqual(work.task_count, 128)
        task_bytes = work.tma_input_bytes / work.task_count
        self.assertEqual(
            result.empirical_envelope.resource_seconds[
                "tma.per_sm_parallel_makespan"],
            7 * task_bytes / per_sm_rate,
        )

    def test_exact_tma_contract_resolves_precision_and_packed_stride(self) -> None:
        schedule = Schedule(
            "mapped", 128, 256, 64, 2,
            mma_n=256,
            tmem_columns=256,
            supported_precisions=("e4m3_f32",),
            tma_contract_family_by_precision={
                "e4m3_f32": "generic_m128n256k64_s2_v8",
            },
            tma_contract_row_stride_elements=(1024, 2048, 4096),
        )
        precision = self.precisions["e4m3_f32"]
        ingress, hbm, gap = _resolve_tma_capacity_resources(
            Workload("square", 2048, 2048, 2048, "e4m3_f32"),
            schedule,
            precision,
        )
        self.assertIsNone(gap)
        self.assertEqual(
            ingress,
            "tma.smem_ingress.contract."
            "generic_m128n256k64_s2_v8.stride2048.per_sm",
        )
        self.assertEqual(
            hbm,
            "tma.hbm.contract."
            "generic_m128n256k64_s2_v8.stride2048",
        )

        ingress, hbm, gap = _resolve_tma_capacity_resources(
            Workload("rectangular", 2048, 1024, 2048, "e4m3_f32"),
            schedule,
            precision,
        )
        self.assertIsNone(ingress)
        self.assertIsNone(hbm)
        self.assertIn("a_ld=2048:b_ld=1024", gap or "")

    def test_legacy_tc5a_point_is_only_an_exact_stride2048_alias(self) -> None:
        legacy = Capacity(
            "legacy", "tma.smem_ingress.per_sm", 10.0, "byte",
            EvidenceKind.MEASURED_SUSTAINED,
            "test", "source.json", "legacy",
        )
        exact_2048 = (
            "tma.smem_ingress.contract."
            "tc5a_m128n256k64_s4_v16.stride2048.per_sm"
        )
        exact_1024 = exact_2048.replace("stride2048", "stride1024")
        self.assertIs(_select_capacity([legacy], exact_2048, strict=False), legacy)
        self.assertIsNone(_select_capacity([legacy], exact_1024, strict=False))

    def test_example_schedules_map_every_precision_to_an_exact_tma_family(self) -> None:
        schedules = load_schedules(SCHEDULE_PATH)
        mapped = {
            precision_id
            for schedule in schedules
            for precision_id in schedule.tma_contract_family_by_precision
        }
        self.assertEqual(mapped, set(self.precisions))
        for schedule in schedules:
            if schedule.tma_contract_family_by_precision:
                self.assertEqual(
                    schedule.tma_contract_row_stride_elements,
                    (1024, 2048, 4096),
                )

    def test_tma_capacity_is_not_inferred_from_stage_count(self) -> None:
        workload = Workload(
            "contract-gate",
            128,
            128,
            64,
            "fp16_f32",
            residency="hot_l2",
        )
        capacities = [
            Capacity(
                "compute", "tensor.fp16.m128n128", 1e30, "flop",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "compute"),
            Capacity(
                "l2_duplex", "l2.duplex", 1e30, "byte",
                EvidenceKind.MEASURED_JOINT,
                "test", "source.json", "l2-duplex"),
            Capacity(
                "legacy_stage_guess", "tma.smem_ingress.per_sm", 1e30,
                "byte", EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "legacy"),
            Capacity(
                "readback", "tmem.readback", 1e30, "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "readback"),
        ]
        unbound = evaluate(
            workload,
            Schedule("unbound-stage4", 128, 128, 64, 4),
            proven_l2_hardware(),
            capacities,
        ).empirical_envelope
        self.assertEqual(unbound.status, "insufficient_evidence")
        self.assertIn(
            "tma_ingress_capacity_contract:unbound-stage4",
            unbound.missing_resources,
        )
        self.assertNotIn(
            "tma.per_sm_parallel_makespan",
            unbound.resource_seconds,
        )

        bound = evaluate(
            workload,
            Schedule(
                "explicit-stage4",
                128,
                128,
                64,
                4,
                tma_ingress_capacity_resource=
                    "tma.smem_ingress.per_sm",
            ),
            proven_l2_hardware(),
            capacities,
        ).empirical_envelope
        self.assertEqual(bound.status, "ok")
        self.assertIn(
            "tma.per_sm_parallel_makespan",
            bound.resource_seconds,
        )

    def test_tmem_capacity_key_matches_instruction_and_warp_contract(self) -> None:
        workload = Workload(
            "tmem-contract", 128, 128, 64, "fp16_f32",
            residency="compute_oracle")
        schedule = Schedule(
            "x8-one-warp", 128, 128, 64, 2,
            threads=32, tmem_load_registers=8)
        work = account_work(workload, schedule, self.precisions["fp16_f32"])
        demands = _resource_demands(
            workload,
            schedule,
            work,
            self.precisions["fp16_f32"],
            empirical=True,
        )
        self.assertIn("tmem.readback.x8.warps1", demands)
        self.assertNotIn("tmem.readback", demands)

    def test_block_scaled_empirical_layer_requires_scale_tmem_ingress(self) -> None:
        capacities = [
            Capacity(
                "compute", "tensor.nvfp4.m128n256", 1e15, "flop",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "compute"),
            Capacity(
                "readback", "tmem.readback", 1e12, "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "readback"),
        ]
        result = evaluate(
            Workload(
                "nv-scale", 128, 256, 64, "nvfp4_f32",
                residency="compute_oracle"),
            Schedule(
                "nv-scale", 128, 256, 64, 2,
                mma_n=256, tmem_columns=512),
            proven_l2_hardware(),
            capacities,
        )
        self.assertEqual(
            result.empirical_envelope.status,
            "insufficient_evidence",
        )
        self.assertIn(
            "tmem.scale_ingress",
            result.empirical_envelope.missing_resources,
        )

    def test_example_manifest_has_an_executable_path_for_every_precision(self) -> None:
        schedules = load_schedules(SCHEDULE_PATH)
        expected_valid_counts = {
            "fp16_f32": 4,
            "bf16_f32": 4,
            "tf32_f32": 3,
            "e4m3_f32": 3,
            "e5m2_f32": 3,
            "e3m2_f32": 1,
            "e2m3_f32": 1,
            "e2m1_f32": 1,
            "mxfp4_f32": 1,
            "nvfp4_f32": 1,
            "s8_s32": 3,
            "u8_s32": 3,
        }
        for precision_id, expected_count in expected_valid_counts.items():
            envelope = evaluate_manifest(
                Workload(precision_id, 1024, 1024, 1024, precision_id),
                schedules,
                Hardware("thor", 20, 1.575e9),
                [],
            )
            with self.subTest(precision_id=precision_id):
                self.assertEqual(envelope.valid_schedule_count, expected_count)


class Tcgen05DescriptorTest(unittest.TestCase):
    def test_table42_type_codes_cover_all_unscaled_v1_precisions(self) -> None:
        cases = {
            ("f16", "f16"): 0,
            ("f16", "bf16"): 1,
            ("tf32", "tf32"): 2,
            ("f8f6f4", "e4m3"): 0,
            ("f8f6f4", "e5m2"): 1,
            ("f8f6f4", "e2m3"): 3,
            ("f8f6f4", "e3m2"): 4,
            ("f8f6f4", "e2m1"): 5,
            ("i8", "u8"): 0,
            ("i8", "s8"): 1,
        }
        for (kind, element_type), expected in cases.items():
            with self.subTest(kind=kind, element_type=element_type):
                record = encode_unscaled(
                    kind, m=128, n=256, a_type=element_type
                )
                fields = decode_fields(record.value_u32, kind=kind)
                self.assertEqual(fields["atype"], expected)
                self.assertEqual(fields["btype"], expected)
                self.assertEqual(fields["m"], 128)
                self.assertEqual(fields["n"], 256)

    def test_table44_does_not_reuse_table42_e2m1_code(self) -> None:
        raw = encode_unscaled("f8f6f4", m=128, n=256, a_type="e2m1")
        nvfp4 = encode_block_scaled_fp4(
            "mxf4nvf4", m=128, n=256, scale_type="ue4m3"
        )
        self.assertEqual(decode_fields(raw.value_u32, kind="f8f6f4")["atype"], 5)
        nv_fields = decode_fields(nvfp4.value_u32, kind="mxf4nvf4")
        self.assertEqual(nv_fields["atype"], 1)
        self.assertEqual(nv_fields["scale_type"], 0)

    def test_mxfp4_uses_ue8m0_scale_bit(self) -> None:
        mxfp4 = encode_block_scaled_fp4(
            "mxf4", m=128, n=128, scale_type="ue8m0"
        )
        fields = decode_fields(mxfp4.value_u32, kind="mxf4")
        self.assertEqual(fields["scale_type"], 1)
        self.assertEqual(fields["k_is_96"], 0)

    def test_invalid_table44_scale_contract_is_rejected(self) -> None:
        with self.assertRaises(DescriptorError):
            encode_block_scaled_fp4(
                "mxf4", m=128, n=128, scale_type="ue4m3"
            )

    def test_sm110_rejects_sm103a_only_k96(self) -> None:
        with self.assertRaisesRegex(DescriptorError, "sm_103a-only"):
            encode_block_scaled_fp4(
                "mxf4", m=128, n=128, scale_type="ue8m0", k=96
            )

    def test_i8_descriptor_rejects_n40(self) -> None:
        with self.assertRaisesRegex(DescriptorError, "nonuniform Table-39"):
            encode_unscaled("i8", m=128, n=40, a_type="s8")


class ComputeCampaignTest(unittest.TestCase):
    def test_bounded_contract_and_process_escalation(self) -> None:
        self.assertEqual(COMPUTE_SCHEMA_VERSION, 2)
        self.assertEqual(COMPUTE_TRIAL_TIMEOUT, 120)
        self.assertEqual(COMPUTE_NCU_TIMEOUT, 300)
        normal = run_compute_bounded(
            [sys.executable, "-c", "print('ok')"],
            cwd=ROOT, timeout_seconds=2)
        self.assertEqual(normal["returncode"], 0)
        self.assertFalse(normal["timed_out"])
        self.assertFalse(normal["termination_failed"])
        timed_out = run_compute_bounded(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=ROOT, timeout_seconds=1)
        self.assertTrue(timed_out["timed_out"])
        self.assertFalse(timed_out["termination_failed"])
        self.assertIsNotNone(timed_out["returncode"])

    def test_auditor_rejects_unbounded_compute_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {
                "run_spec.json": {
                    "schema_version": 1, "manifest": [], "trials": 10,
                    "expected_sm_count": 20,
                    "trial_timeout_seconds": None,
                    "ncu_timeout_seconds": None,
                    "termination_grace_seconds": None,
                },
                "summary.json": {
                    "schema_version": 1, "status": "complete", "results": []},
                "campaign_status.json": {"status": "complete"},
                "environment.json": {},
            }
            for name, payload in payloads.items():
                (root / name).write_text(json.dumps(payload))
            (root / "environment_snapshots.jsonl").write_text("{}\n")
            (root / "progress.jsonl").write_text("{}\n")
            (root / "COMPLETE").write_text("forged\n")
            findings = audit_compute_bundle(root, require_ncu=False)
        codes = {finding["code"] for finding in findings}
        self.assertIn("invalid_schema_version", codes)
        self.assertIn("invalid_trial_timeout_contract", codes)
        self.assertIn("invalid_ncu_timeout_contract", codes)
        self.assertIn("invalid_termination_grace_contract", codes)

    def test_manifest_has_all_72_unique_cases(self) -> None:
        manifest = make_compute_campaign_manifest()
        ids = [row["case_id"] for row in manifest]
        self.assertEqual(len(ids), 72)
        self.assertEqual(len(set(ids)), 72)
        self.assertEqual(
            {row["precision"]["precision_id"] for row in manifest},
            {precision.precision_id for precision in CAMPAIGN_PRECISIONS},
        )

    def test_corrected_nvfp4_source_uses_table44_descriptor(self) -> None:
        precision = next(
            row for row in CAMPAIGN_PRECISIONS if row.precision_id == "nvfp4_f32"
        )
        source, descriptor = make_compute_campaign_source(
            precision, 128, 256, "full_sm_4warp_block", "nv"
        )
        fields = decode_fields(descriptor.value_u32, kind="mxf4nvf4")
        self.assertEqual(fields["atype"], 1)
        self.assertEqual(fields["scale_type"], 0)
        self.assertIn("kind::mxf4nvf4.block_scale.block16", source)
        self.assertNotIn("tcgen05.mma.sp", source)

    def test_fp6_direct_smem_uses_byte_container_descriptor(self) -> None:
        precision = next(
            row for row in CAMPAIGN_PRECISIONS if row.precision_id == "e3m2_f32"
        )
        source, _ = make_compute_campaign_source(
            precision, 128, 128, "single_warp_block", "fp6"
        )
        self.assertIn("kLogicalBits = 6", source)
        self.assertIn("kDescriptorBits = 8", source)
        self.assertIn("smem_desc(a,8,4)", source)


class EvidenceSemanticsTest(unittest.TestCase):
    def test_shared_l2_upper_is_not_multiplied_by_sm_count(self) -> None:
        capacities = [
            Capacity(
                "compute_upper", "tensor.bf16", 1e30, "flop",
                EvidenceKind.DERIVED_UPPER,
                "test", "source.json", "compute-upper"),
            Capacity(
                "l2_read_gpu_upper", "l2.read", 1024.0, "byte",
                EvidenceKind.PROFILER_MODEL_PEAK,
                "test", "source.json", "l2-read-upper",
                condition="1024 B/cycle is one aggregate GPU-wide L2 read bus"),
            Capacity(
                "l2_write_gpu_upper", "l2.write", 1e30, "byte",
                EvidenceKind.PROFILER_MODEL_PEAK,
                "test", "source.json", "l2-write-upper",
                condition="aggregate GPU-wide L2 write bus"),
        ]
        workload = Workload(
            "shared-l2", 128, 128, 64, "bf16_f32", residency="hot_l2")
        schedule = Schedule("s", 128, 128, 64, 2)
        one_sm = evaluate(
            workload, schedule,
            proven_l2_hardware("one-sm", 1, 1.0), capacities,
        ).conditional_upper
        twenty_sm = evaluate(
            workload, schedule,
            proven_l2_hardware("twenty-sm", 20, 1.0), capacities,
        ).conditional_upper
        self.assertEqual(
            one_sm.resource_seconds["l2.read"],
            twenty_sm.resource_seconds["l2.read"],
        )
        self.assertEqual(
            one_sm.performance_per_second,
            twenty_sm.performance_per_second,
        )

    def test_empirical_cold_hbm_intersects_shared_total_upper(self) -> None:
        def capacity(
            capacity_id: str,
            resource: str,
            rate: float,
            unit: str,
            evidence_kind: EvidenceKind,
        ) -> Capacity:
            return Capacity(
                capacity_id,
                resource,
                rate,
                unit,
                evidence_kind,
                "test",
                "source.json",
                capacity_id,
                qualification=(
                    "closure_qualified"
                    if evidence_kind.is_empirical_rate
                    else "snapshot_only"
                ),
                trial_count=(10 if evidence_kind.is_empirical_rate else 1),
                artifact_paths=(
                    ("source.json",)
                    if evidence_kind.is_empirical_rate
                    else ()
                ),
            )

        capacities = [
            capacity(
                "compute_upper", "tensor.bf16", 1e30, "flop",
                EvidenceKind.DERIVED_UPPER),
            capacity(
                "hbm_total_upper", "hbm.total", 100.0, "byte",
                EvidenceKind.DERIVED_UPPER),
            capacity(
                "l2_read_upper", "l2.read", 1e30, "byte",
                EvidenceKind.DERIVED_UPPER),
            capacity(
                "l2_write_upper", "l2.write", 1e30, "byte",
                EvidenceKind.DERIVED_UPPER),
            capacity(
                "compute", "tensor.bf16.m128n128", 1e30, "flop",
                EvidenceKind.MEASURED_SUSTAINED),
            capacity(
                "hbm_duplex", "hbm.duplex", 1000.0, "byte",
                EvidenceKind.MEASURED_JOINT),
            capacity(
                "l2_duplex", "l2.duplex", 1000.0, "byte",
                EvidenceKind.MEASURED_JOINT),
            capacity(
                "tma_per_sm", "tma.smem_ingress.per_sm.inflight4", 1e30, "byte",
                EvidenceKind.MEASURED_SUSTAINED),
            capacity(
                "tma_hbm", "tma.hbm.inflight4", 1e30, "byte",
                EvidenceKind.MEASURED_SUSTAINED),
            capacity(
                "readback", "tmem.readback", 1e30, "byte",
                EvidenceKind.MEASURED_SUSTAINED),
        ]
        result = evaluate(
            Workload(
                "shared-hbm", 128, 128, 64, "bf16_f32",
                residency="cold_hbm"),
            Schedule(
                "s",
                128,
                128,
                64,
                2,
                tma_ingress_capacity_resource=
                    "tma.smem_ingress.per_sm.inflight4",
                tma_hbm_capacity_resource="tma.hbm.inflight4",
            ),
            Hardware("h", 20, 1.0),
            capacities,
        )
        strict = result.conditional_upper
        empirical = result.empirical_envelope
        self.assertEqual(strict.status, "ok")
        self.assertEqual(empirical.status, "ok")
        self.assertEqual(empirical.bottlenecks, ["hard_upper:hbm.total"])
        self.assertEqual(
            empirical.resource_seconds["hard_upper:hbm.total"],
            strict.resource_seconds["hbm.total"],
        )
        self.assertLessEqual(
            empirical.performance_per_second,
            strict.performance_per_second,
        )

    def test_hot_l2_does_not_receive_hbm_total_ceiling(self) -> None:
        capacities = [
            Capacity(
                "compute_upper", "tensor.bf16", 1e30, "flop",
                EvidenceKind.DERIVED_UPPER,
                "test", "source.json", "compute-upper"),
            Capacity(
                "hbm_total_upper", "hbm.total", 1.0, "byte",
                EvidenceKind.DERIVED_UPPER,
                "test", "source.json", "hbm-upper"),
            Capacity(
                "compute", "tensor.bf16.m128n128", 1e30, "flop",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "compute"),
            Capacity(
                "l2_duplex", "l2.duplex", 1e12, "byte",
                EvidenceKind.MEASURED_JOINT,
                "test", "source.json", "l2-duplex"),
            Capacity(
                "tma_per_sm", "tma.smem_ingress.per_sm.inflight4", 1e30, "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "tma-l2"),
            Capacity(
                "readback", "tmem.readback", 1e30, "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "readback"),
        ]
        result = evaluate(
            Workload(
                "hot-l2", 128, 128, 64, "bf16_f32",
                residency="hot_l2"),
            Schedule(
                "s",
                128,
                128,
                64,
                2,
                tma_ingress_capacity_resource=
                    "tma.smem_ingress.per_sm.inflight4",
            ),
            proven_l2_hardware("h", 20, 1.0),
            capacities,
        )
        self.assertFalse(any(
            key.endswith("hbm.total")
            for key in result.empirical_envelope.resource_seconds
        ))

    def test_closure_capacity_supersedes_faster_legacy_snapshot(self) -> None:
        snapshot = Capacity(
            "snapshot", "tensor.bf16", 200.0, "flop",
            EvidenceKind.MEASURED_SUSTAINED, "old", "old.csv", "row=1",
            qualification="snapshot_only")
        closure = Capacity(
            "closure", "tensor.bf16", 100.0, "flop",
            EvidenceKind.MEASURED_SUSTAINED, "new", "new.json", "case",
            qualification="closure_qualified", trial_count=10,
            artifact_paths=("new.json",))
        selected = _select_capacity(
            [snapshot, closure], "tensor.bf16", strict=False)
        self.assertEqual(selected, closure)

    def test_unimplemented_epilogue_fails_closed(self) -> None:
        with self.assertRaisesRegex(Exception, "work and I/O contract is not implemented"):
            evaluate(
                Workload(
                    "relu",
                    128,
                    128,
                    64,
                    "bf16_f32",
                    epilogue="relu",
                    residency="compute_oracle",
                ),
                Schedule("s", 128, 128, 64, 2),
                Hardware("h", 20, 1.0),
                [],
            )

    def test_measured_capacity_is_not_used_as_strict_upper(self) -> None:
        capacities = [
            Capacity(
                "measured",
                "tensor.bf16.m128n128",
                100.0,
                "flop",
                EvidenceKind.MEASURED_SUSTAINED,
                "source",
                "source.csv",
                "row=1",
            ),
            Capacity(
                "readback",
                "tmem.readback",
                100.0,
                "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "source",
                "source.csv",
                "row=2",
            ),
        ]
        result = evaluate(
            Workload("w", 128, 128, 64, "bf16_f32", residency="compute_oracle"),
            Schedule("s", 128, 128, 64, 2),
            Hardware("h", 20, 1.0),
            capacities,
        )
        self.assertEqual(result.conditional_upper.status, "insufficient_evidence")
        self.assertIsNone(result.conditional_upper.performance_per_second)
        self.assertIsNotNone(result.empirical_envelope.performance_per_second)

    def test_empirical_compute_capacity_is_not_reused_across_mma_shapes(self) -> None:
        capacities = [
            Capacity(
                "compute_n256",
                "tensor.bf16.m128n256",
                100.0,
                "flop",
                EvidenceKind.MEASURED_SUSTAINED,
                "source",
                "source.csv",
                "row=1",
            ),
            Capacity(
                "readback",
                "tmem.readback",
                100.0,
                "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "source",
                "source.csv",
                "row=2",
            ),
        ]
        workload = Workload(
            "shape-contract", 128, 256, 64, "bf16_f32",
            residency="compute_oracle")
        wrong_shape = evaluate(
            workload,
            Schedule("n128", 128, 256, 64, 2, mma_n=128,
                     tmem_columns=256),
            Hardware("h", 20, 1.0),
            capacities,
        )
        matching_shape = evaluate(
            workload,
            Schedule("n256", 128, 256, 64, 2, mma_n=256,
                     tmem_columns=256),
            Hardware("h", 20, 1.0),
            capacities,
        )
        self.assertEqual(
            wrong_shape.empirical_envelope.status, "insufficient_evidence")
        self.assertIn(
            "tensor.bf16.m128n128",
            wrong_shape.empirical_envelope.missing_resources,
        )
        self.assertEqual(matching_shape.empirical_envelope.status, "ok")

    def test_generic_compute_upper_applies_to_each_mma_shape(self) -> None:
        capacity = Capacity(
            "compute_upper",
            "tensor.bf16",
            100.0,
            "flop",
            EvidenceKind.DERIVED_UPPER,
            "source",
            "source.csv",
            "row=1",
        )
        workload = Workload(
            "upper-shape-contract", 128, 256, 64, "bf16_f32",
            residency="compute_oracle")
        for mma_n in (64, 128, 256):
            result = evaluate(
                workload,
                Schedule(
                    f"n{mma_n}", 128, 256, 64, 2,
                    mma_n=mma_n, tmem_columns=256),
                Hardware("h", 20, 1.0),
                [capacity],
            )
            self.assertEqual(result.conditional_upper.status, "ok")
            self.assertEqual(
                result.conditional_upper.performance_per_second, 100.0)

    def test_larger_proven_rate_cannot_lower_performance_upper(self) -> None:
        workload = Workload("w", 128, 128, 64, "bf16_f32", residency="compute_oracle")
        schedule = Schedule("s", 128, 128, 64, 2)
        hardware = Hardware("h", 20, 1.0)

        def cap(rate: float) -> Capacity:
            return Capacity(
                f"upper_{rate}",
                "tensor.bf16",
                rate,
                "flop",
                EvidenceKind.DERIVED_UPPER,
                "source",
                "source.csv",
                "row=1",
            )

        slow = evaluate(workload, schedule, hardware, [cap(100.0)])
        fast = evaluate(workload, schedule, hardware, [cap(200.0)])
        self.assertGreaterEqual(
            fast.conditional_upper.performance_per_second,
            slow.conditional_upper.performance_per_second,
        )

    def test_multiple_proven_uppers_use_their_tightest_intersection(self) -> None:
        workload = Workload(
            "w", 128, 128, 64, "bf16_f32", residency="compute_oracle")
        schedule = Schedule("s", 128, 128, 64, 2)
        hardware = Hardware("h", 20, 1.0)

        def cap(name: str, rate: float) -> Capacity:
            return Capacity(
                name, "tensor.bf16", rate, "flop",
                EvidenceKind.DERIVED_UPPER, "source", "source.csv", "row=1")

        tight = evaluate(
            workload, schedule, hardware, [cap("tight", 100.0)])
        intersection = evaluate(
            workload, schedule, hardware,
            [cap("tight", 100.0), cap("loose", 200.0)])
        self.assertEqual(
            intersection.conditional_upper.performance_per_second,
            tight.conditional_upper.performance_per_second)

    def test_strict_layer_does_not_infer_per_group_rate_from_gpu_rate(self) -> None:
        capacity = Capacity(
            "gpu_upper",
            "tensor.bf16",
            100.0,
            "flop",
            EvidenceKind.DERIVED_UPPER,
            "source",
            "source.csv",
            "row=1",
        )
        measured = Capacity(
            "gpu_measured",
            "tensor.bf16.m128n128",
            90.0,
            "flop",
            EvidenceKind.MEASURED_SUSTAINED,
            "source",
            "source.csv",
            "row=2",
        )
        readback = Capacity(
            "readback",
            "tmem.readback",
            100.0,
            "byte",
            EvidenceKind.MEASURED_SUSTAINED,
            "source",
            "source.csv",
            "row=3",
        )
        result = evaluate(
            Workload("tiny", 128, 128, 64, "bf16_f32", residency="compute_oracle"),
            Schedule("s", 128, 128, 64, 2),
            Hardware("h", 20, 1.0),
            [capacity, measured, readback],
        )
        self.assertNotIn("parallel_span", result.conditional_upper.resource_seconds)
        self.assertIn("parallel_span", result.empirical_envelope.resource_seconds)

    def test_profiler_peak_requires_condition(self) -> None:
        capacity = Capacity(
            "peak",
            "l2.read",
            100.0,
            "byte",
            EvidenceKind.PROFILER_MODEL_PEAK,
            "source",
            "source.csv",
            "row=1",
        )
        findings = audit_inputs([capacity])
        self.assertIn("conditional_peak_without_condition", {row["code"] for row in findings})

    def test_specified_upper_requires_primary_source_url(self) -> None:
        capacity = Capacity(
            "vendor", "tensor.bf16", 100.0, "flop",
            EvidenceKind.SPECIFIED_UPPER, "source", "source.md", "exact text")
        findings = audit_inputs([capacity])
        self.assertIn("invalid_capacity", {row["code"] for row in findings})

    def test_markdown_locator_must_match_source_text(self) -> None:
        capacity = Capacity(
            "bad_markdown_locator",
            "l2.read",
            100.0,
            "byte",
            EvidenceKind.MEASURED_SUSTAINED,
            "source",
            "microbench/L2throughtput/README.md",
            "this exact locator does not exist",
        )
        findings = audit_inputs([capacity], repo_root=ROOT)
        self.assertIn("text_locator_no_match", {row["code"] for row in findings})

    def test_repo_capacity_sources_exist(self) -> None:
        findings = audit_inputs(load_capacities(CAPACITY_PATH), repo_root=ROOT)
        self.assertEqual(findings, [])

    def test_csv_locator_value_mismatch_is_rejected(self) -> None:
        capacity = Capacity(
            "bad_value",
            "tensor.bf16",
            100.0,
            "flop",
            EvidenceKind.MEASURED_SUSTAINED,
            "source",
            "microbench/mma_compute_only/plots/benchmark_results.csv",
            "precision=BF16,warp_num=4,shape=M128N256K16,tflops",
            original_value=1.0,
            original_unit="TFLOP/s",
        )
        findings = audit_inputs([capacity], repo_root=ROOT)
        self.assertIn("csv_original_value_mismatch", {row["code"] for row in findings})

    def test_capacity_evidence_cannot_escape_repository(self) -> None:
        capacity = Capacity(
            "outside", "tensor.bf16", 100.0, "flop",
            EvidenceKind.MEASURED_SUSTAINED,
            "source", "../outside.txt", "locator")
        findings = audit_inputs([capacity], repo_root=ROOT)
        self.assertIn("invalid_source_path", {row["code"] for row in findings})


class CausalPipelineModelTest(unittest.TestCase):
    @staticmethod
    def profile(
        *,
        schedule_id: str = "tc5a-test",
        precision_ids: tuple[str, ...] = ("fp16_f32",),
        maximum_k_tiles: int = 64,
        maximum_output_tasks: int = 32,
        qualification: str = "closure_qualified",
    ) -> PipelineProfile:
        qualified = qualification == "closure_qualified"
        calibration_k = [1]
        holdout_k = [maximum_k_tiles] if maximum_k_tiles != 1 else [2]
        calibration_output = [1]
        holdout_output = (
            [maximum_output_tasks] if maximum_output_tasks != 1 else [2]
        )

        def prediction_ns(k_tiles: int, output_tasks: int) -> float:
            previous_mma_done = 0.0
            epilogue_done: list[float] = []
            for task in range(output_tasks):
                first = 10.0 if task == 0 else previous_mma_done + 2.0
                if task >= 2:
                    first = max(first, epilogue_done[task - 2] + 2.0)
                last = first + (k_tiles - 1) * 2.0
                start = max(last, epilogue_done[-1] if epilogue_done else 0.0)
                epilogue_done.append(start + 5.0)
                previous_mma_done = last
            return epilogue_done[-1]

        validation = []
        for k_tiles in (*calibration_k, *holdout_k):
            for output_tasks in (*calibration_output, *holdout_output):
                split = (
                    "calibration"
                    if k_tiles in calibration_k
                    and output_tasks in calibration_output
                    else "holdout"
                )
                relative_error = 0.01 if split == "calibration" else 0.02
                predicted = prediction_ns(k_tiles, output_tasks)
                actual = predicted / (1.0 + relative_error)
                validation.append({
                    "case_id": f"k{k_tiles}.o{output_tasks}",
                    "split": split,
                    "k_tiles": k_tiles,
                    "output_tasks": output_tasks,
                    "actual_median_ns": actual,
                    "predicted_ns": predicted,
                    "relative_error": abs(predicted - actual) / actual,
                })
        return PipelineProfile(
            profile_id=f"profile.{schedule_id}",
            resource="pipeline.tc5a-test",
            schedule_id=schedule_id,
            precision_ids=precision_ids,
            evidence_kind=EvidenceKind.MEASURED_JOINT,
            qualification=qualification,
            trial_count_per_case=10,
            source_id="causal-suite",
            expected_commit="1" * 40,
            source_path="microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu",
            source_locator="synthetic predeclared joint fit",
            input_residency="hot_l2",
            stages=4,
            accumulator_buffers=2,
            resident_ctas_per_sm=1,
            maximum_k_tiles=maximum_k_tiles,
            maximum_output_tasks_per_worker=maximum_output_tasks,
            tma_first_completion_seconds=8e-9,
            tma_completion_interval_seconds=1e-9,
            mma_first_completion_seconds=9e-9,
            mma_completion_interval_seconds=2e-9,
            joint_first_mma_completion_seconds=10e-9,
            joint_completion_interval_seconds=2e-9,
            epilogue_latency_seconds=5e-9,
            component_r_squared={
                "tma": 0.999,
                "mma": 0.999,
                "joint": 0.999 if qualified else 0.5,
            },
            max_calibration_relative_error=0.01,
            max_holdout_relative_error=0.02,
            fit_contract={
                "calibration_points": calibration_k,
                "holdout_points": holdout_k,
                "calibration_output_tasks": calibration_output,
                "epilogue_calibration_output_tasks": [1],
                "holdout_output_tasks": holdout_output,
                "minimum_r_squared": 0.98,
                "maximum_holdout_relative_error": 0.1,
                "profile_stages": 4,
                "profile_accumulator_buffers": 2,
                "profile_resident_ctas_per_sm": 1,
            },
            validation=tuple(validation),
            closure_qualified=qualified,
            artifact_paths=("pipeline_profile.json",),
            applicable_sm_counts=(20,),
            applicable_hardware_ids=("thor",),
            applicable_operating_modes=("test",),
            applicable_clock_hz=(1.575e9,),
        )

    @staticmethod
    def schedule() -> Schedule:
        return Schedule(
            "tc5a-test",
            128,
            256,
            64,
            4,
            mma_n=256,
            tail_policy="pad",
            threads=192,
            tmem_columns=512,
            tmem_load_registers=8,
            tmem_consumer_warps=4,
            tma_ingress_capacity_resource="tma.tc5a-test",
            causal_pipeline_resource="pipeline.tc5a-test",
            persistent=True,
        )

    @staticmethod
    def capacities() -> list[Capacity]:
        resources = (
            ("compute", "tensor.fp16.m128n256", "flop"),
            ("l2duplex", "l2.duplex", "byte"),
            ("tmem", "tmem.readback.x8.warps4", "byte"),
            ("tma", "tma.tc5a-test", "byte"),
        )
        return [
            Capacity(
                capacity_id=name,
                resource=resource,
                rate_per_second=1e30,
                work_unit=unit,
                evidence_kind=EvidenceKind.MEASURED_SUSTAINED,
                source_id="synthetic",
                source_path="source.json",
                source_locator=name,
                qualification="closure_qualified",
                trial_count=10,
                artifact_paths=("source.json",),
            )
            for name, resource, unit in resources
        ]

    def test_persistent_double_buffer_recurrence_is_exact(self) -> None:
        profile = self.profile()
        predicted = predict_pipeline_worker_seconds(
            profile, k_tiles=3, output_tasks=3
        )
        self.assertAlmostEqual(predicted, 31e-9)

    def test_target_completion_accepts_only_an_exact_causal_profile(self) -> None:
        profile = self.profile()
        schedule = self.schedule()
        workload = Workload(
            "profile-contract", 128, 256, 64, "fp16_f32",
            residency="hot_l2",
        )
        self.assertTrue(joint_pipeline_profile_matches(
            profile,
            workload=workload,
            schedule=schedule,
            hardware=proven_l2_hardware(),
        ))
        self.assertFalse(joint_pipeline_profile_matches(
            profile,
            workload=replace(workload, residency="cold_hbm"),
            schedule=schedule,
            hardware=proven_l2_hardware(),
        ))
        self.assertFalse(joint_pipeline_profile_matches(
            profile,
            workload=workload,
            schedule=schedule,
            hardware=proven_l2_hardware(hardware_id="other"),
        ))

    def test_profile_recomputes_every_validation_prediction(self) -> None:
        profile = self.profile()
        rows = [dict(row) for row in profile.validation]
        rows[0]["predicted_ns"] = float(rows[0]["predicted_ns"]) + 1.0
        with self.assertRaises(ValueError):
            replace(profile, validation=tuple(rows)).validate()

    def test_negative_r_squared_is_retained_only_as_quarantined(self) -> None:
        profile = self.profile(qualification="quarantined")
        profile = replace(
            profile,
            component_r_squared={"tma": 0.99, "mma": 0.99, "joint": -0.2},
        )
        profile.validate()
        self.assertFalse(profile.is_closure_qualified)

    def test_integrated_ideal_is_max_of_resource_and_causal_time(self) -> None:
        result = evaluate(
            Workload(
                "one-task", 128, 256, 64, "fp16_f32", residency="hot_l2"
            ),
            self.schedule(),
            proven_l2_hardware(),
            self.capacities(),
            [self.profile()],
        )
        self.assertEqual(result.empirical_envelope.status, "ok")
        self.assertEqual(result.causal_pipeline.status, "ok")
        self.assertAlmostEqual(result.causal_pipeline.seconds, 15e-9)
        self.assertAlmostEqual(result.empirical_ideal_envelope.seconds, 15e-9)
        self.assertIn(
            "causal:causal.pipeline.persistent_worker",
            result.empirical_ideal_envelope.bottlenecks,
        )

    def test_solver_without_an_exact_profile_fails_closed(self) -> None:
        result = evaluate(
            Workload(
                "missing-profile", 128, 256, 64, "fp16_f32",
                residency="hot_l2",
            ),
            self.schedule(),
            proven_l2_hardware(),
            self.capacities(),
        )
        self.assertEqual(result.empirical_envelope.status, "ok")
        self.assertEqual(result.causal_pipeline.status, "insufficient_evidence")
        self.assertEqual(
            result.causal_pipeline.missing_resources,
            ["pipeline.tc5a-test:precision=fp16_f32"],
        )
        self.assertIsNone(result.empirical_ideal_envelope.seconds)

    def test_fp16_profile_is_not_reused_for_bf16(self) -> None:
        result = evaluate(
            Workload(
                "bf16-needs-own-profile", 128, 256, 64, "bf16_f32",
                residency="hot_l2",
            ),
            self.schedule(),
            proven_l2_hardware(),
            self.capacities(),
            [self.profile(precision_ids=("fp16_f32",))],
        )
        self.assertEqual(result.causal_pipeline.status, "insufficient_evidence")
        self.assertEqual(
            result.causal_pipeline.missing_resources,
            ["pipeline.tc5a-test:precision=bf16_f32"],
        )
        self.assertIn(
            "profile.tc5a-test: precision_ids=fp16_f32",
            result.causal_pipeline.conditions,
        )

    def test_profile_precision_contract_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "precision_ids"):
            replace(self.profile(), precision_ids=()).validate()
        with self.assertRaisesRegex(ValueError, "unknown causal profile"):
            replace(self.profile(), precision_ids=("not_a_precision",)).validate()

    def test_profile_range_is_not_extrapolated(self) -> None:
        result = evaluate(
            Workload(
                "range", 4096, 4096, 4096, "fp16_f32", residency="hot_l2"
            ),
            self.schedule(),
            proven_l2_hardware(),
            self.capacities(),
            [self.profile(maximum_output_tasks=20)],
        )
        self.assertEqual(result.causal_pipeline.status, "insufficient_evidence")
        self.assertEqual(
            result.causal_pipeline.missing_resources,
            ["pipeline.tc5a-test:profile_range"],
        )
        self.assertTrue(any(
            "output_tasks=26" in condition
            for condition in result.causal_pipeline.conditions
        ))

    def test_quarantined_profile_cannot_close_the_layer(self) -> None:
        result = evaluate(
            Workload(
                "quarantined", 128, 256, 64, "fp16_f32",
                residency="hot_l2",
            ),
            self.schedule(),
            proven_l2_hardware(),
            self.capacities(),
            [self.profile(qualification="quarantined")],
        )
        self.assertEqual(result.causal_pipeline.status, "insufficient_evidence")
        self.assertIn(
            "profile.tc5a-test: quarantined",
            result.causal_pipeline.conditions,
        )


class SnapshotEvaluationTest(unittest.TestCase):
    def test_bf16_snapshot_exposes_incomplete_empirical_layer(self) -> None:
        result = evaluate(
            Workload("bf16", 1024, 1024, 1024, "bf16_f32"),
            Schedule("s", 128, 128, 64, 2, tail_policy="pad"),
            load_hardware(
                Path(__file__).resolve().parent / "profiles" / "thor_sm110.json"
            ),
            load_capacities(CAPACITY_PATH),
        )
        self.assertIn(result.conditional_upper.status, {"ok", "partial"})
        self.assertEqual(result.empirical_envelope.status, "insufficient_evidence")
        self.assertIsNotNone(result.conditional_upper.performance_per_second)
        self.assertIsNone(result.empirical_envelope.performance_per_second)

    def test_int8_uses_integer_operation_units(self) -> None:
        capacity = Capacity(
            "int8",
            "tensor.s8",
            100.0,
            "operation",
            EvidenceKind.DERIVED_UPPER,
            "source",
            "source.csv",
            "row=1",
        )
        result = evaluate(
            Workload("s8", 128, 128, 32, "s8_s32", residency="compute_oracle"),
            Schedule("s", 128, 128, 32, 2),
            Hardware("thor", 20, 1.575e9),
            [capacity],
        )
        self.assertEqual(result.work.compute_work_unit, "operation")
        self.assertEqual(result.conditional_upper.performance_unit, "operation/s")

    def test_manifest_rejects_unproven_exact_tail_schedule(self) -> None:
        workload = Workload(
            "bf16", 129, 129, 65, "bf16_f32", residency="compute_oracle"
        )
        hardware = Hardware("thor", 20, 1.575e9)
        capacities = [
            Capacity(
                "compute",
                "tensor.bf16.m128n128",
                100.0,
                "flop",
                EvidenceKind.MEASURED_SUSTAINED,
                "source",
                "source.csv",
                "row=1",
            ),
            Capacity(
                "readback",
                "tmem.readback",
                100.0,
                "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "source",
                "source.csv",
                "row=2",
            ),
        ]
        envelope = evaluate_manifest(
            workload,
            [
                Schedule("padded", 128, 128, 64, 2, tail_policy="pad"),
                Schedule("exact", 128, 128, 64, 2, tail_policy="exact"),
                Schedule("illegal", 128, 128, 63, 2),
            ],
            hardware,
            capacities,
        )
        self.assertEqual(envelope.valid_schedule_count, 1)
        self.assertEqual(envelope.rejected_schedule_count, 2)
        self.assertEqual(envelope.empirical_resource_schedule_id, "padded")
        self.assertIsNone(envelope.empirical_schedule_id)
        self.assertTrue(
            any("exact tail requires" in row["reason"] for row in envelope.rejected)
        )


class ObservationTest(unittest.TestCase):
    def test_snapshot_observations_require_ten_matched_trials(self) -> None:
        rows = summarize_observed_csvs(
            [
                ROOT / "results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv",
                ROOT / "results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv",
            ],
            repo_root=ROOT,
        )
        keys = {(row.precision_id, row.n) for row in rows}
        self.assertIn(("fp16_f32", 1024), keys)
        self.assertIn(("e4m3_f32", 1024), keys)
        self.assertIn(("s8_s32", 1024), keys)
        self.assertIn(("nvfp4_f32", 1024), keys)
        nvfp4 = next(row for row in rows if row.precision_id == "nvfp4_f32")
        self.assertEqual(
            nvfp4.performance_reference_relation, "cross_precision_denominator"
        )

    def test_upper_violation_is_an_error(self) -> None:
        row = ObservedBest(
            "obs",
            "bf16_f32",
            128,
            128,
            128,
            "backend",
            "reference",
            "same_precision",
            10,
            10,
            100.0,
            110.0,
            90.0,
            "flop/s",
            "results.csv",
        )
        findings = audit_observed_against_upper(row, 100.0, relative_tolerance=0.01)
        self.assertEqual(findings[0]["code"], "observed_exceeds_conditional_upper")

    def test_observation_evidence_cannot_escape_repository(self) -> None:
        row = ObservedBest(
            "outside", "bf16_f32", 128, 128, 128,
            "backend", "reference", "same_precision", 10, 10,
            100.0, 101.0, 99.0, "flop/s", "../outside.txt")
        with self.assertRaisesRegex(Exception, "escapes repository root"):
            row.validate(repo_root=ROOT)

    def test_snapshot_coverage_exposes_missing_precisions(self) -> None:
        capacities = load_capacities(CAPACITY_PATH)
        observed = summarize_observed_csvs(
            [
                ROOT / "results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv",
                ROOT / "results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv",
            ],
            repo_root=ROOT,
        )
        hardware = load_hardware(
            Path(__file__).resolve().parent / "profiles" / "thor_sm110.json"
        )
        rows = {
            row.precision_id: row
            for row in precision_coverage(capacities, observed, hardware)
        }
        self.assertFalse(rows["tf32_f32"].numeric_closure)
        self.assertIn(
            "empirical_compute_rate", rows["tf32_f32"].evidence_missing
        )
        self.assertFalse(rows["nvfp4_f32"].same_precision_performance_denominator)
        self.assertIn(
            "full_gemm_observed", rows["nvfp4_f32"].evidence_missing
        )
        self.assertIn(
            "same_precision_performance_denominator",
            rows["e5m2_f32"].comparison_missing,
        )
        self.assertFalse(rows["e4m3_f32"].numeric_closure)
        self.assertIn(
            "empirical_compute_rate",
            rows["e4m3_f32"].evidence_missing,
        )
        common = common_resource_coverage(capacities, hardware)
        self.assertFalse(common["tmem.readback"])
        campaign = campaign_measurement_coverage(capacities, observed)
        self.assertFalse(campaign["all_campaign_measurements_closed"])

    def test_precision_report_keeps_implementation_and_numeric_gates_separate(self) -> None:
        capacities = load_capacities(CAPACITY_PATH)
        observed = summarize_observed_csvs(
            [
                ROOT / "results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv",
                ROOT / "results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv",
            ],
            repo_root=ROOT,
        )
        analysis = build_precision_evidence_analysis(
            capacities=capacities,
            observations=observed,
            support_manifest=json.loads(SUPPORT_MANIFEST_PATH.read_text()),
            repo_root=ROOT,
            hardware=load_hardware(
                Path(__file__).resolve().parent / "profiles" / "thor_sm110.json"
            ),
            metadata={"suite_id": "snapshot", "expected_commit": "none"},
        )
        self.assertEqual(analysis["precision_count"], 12)
        self.assertEqual(analysis["schema_version"], 3)
        self.assertEqual(analysis["implementation_ready_count"], 6)
        self.assertEqual(analysis["resource_envelope_closed_count"], 0)
        self.assertTrue(analysis["causal_pipeline_dag_implemented"])
        self.assertEqual(analysis["causal_pipeline_closed_count"], 0)
        self.assertEqual(analysis["end_to_end_closed_count"], 0)
        self.assertFalse(analysis["all_precision_numeric_evidence_closed"])
        self.assertFalse(analysis["all_precisions_end_to_end_closed"])
        rows = {row["precision_id"]: row for row in analysis["precisions"]}
        self.assertTrue(rows["fp16_f32"]["implementation_ready"])
        self.assertFalse(rows["fp16_f32"]["numeric_closure"])
        self.assertTrue(rows["e5m2_f32"]["implementation_ready"])
        self.assertEqual(rows["e5m2_f32"]["support_gaps"], [])
        self.assertIn(
            "closure_qualified_causal_pipeline_profile_matrix",
            rows["fp16_f32"]["model_gaps"],
        )
        document = render_precision_evidence_markdown(analysis)
        self.assertIn("all precisions end-to-end closed：`false`", document)
        self.assertIn("### `e5m2_f32`", document)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.sm110_gemm_model.io import load_capacities, load_schedules
from scripts.sm110_gemm_model.model import (
    Capacity,
    EvidenceKind,
    Hardware,
    Schedule,
    Workload,
    account_work,
    audit_inputs,
    evaluate,
    evaluate_manifest,
    precision_specs,
    _resource_demands,
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


class DocumentContractTest(unittest.TestCase):
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
                r"定义 \(T_{\mathrm{tensor}}^{\mathrm{LB}}\)",
            r"\(c_{\mathrm{L2,write}}^{\mathrm{UB}}":
                r"定义 \(c_{\mathrm{L2,write}}^{\mathrm{UB}}",
            r"\(C_{\mathrm{L2,write}}^{\mathrm{UB}}\)":
                r"定义 \(C_{\mathrm{L2,write}}^{\mathrm{UB}}\)",
            r"\(T_{\mathrm{L2,write}}^{\mathrm{LB}}\)":
                r"定义 \(T_{\mathrm{tensor}}^{\mathrm{LB}}\)",
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
        self.schedule = Schedule("s", 128, 128, 64, 2, tail_policy="pad")
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
        self.assertEqual(demands["hbm.read"][0], work.tma_unique_input_bytes)
        self.assertEqual(
            demands["tma.hbm.inflight4"][0],
            work.tma_unique_input_bytes,
        )
        self.assertEqual(demands["l2.read"][0], work.tma_input_bytes)
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
        self.assertNotIn("hbm.read", demands)
        self.assertNotIn("tma.hbm", demands)
        self.assertEqual(demands["l2.read"][0], work.tma_input_bytes)
        self.assertNotIn("tma.l2", demands)

    def test_per_sm_tma_ingress_uses_slowest_wave_makespan(self) -> None:
        workload = Workload(
            "tc5a-like", 2048, 2048, 2048, "fp16_f32",
            residency="hot_l2")
        schedule = Schedule(
            "tc5a", 128, 256, 64, 4,
            mma_n=256, tmem_columns=256)
        per_sm_rate = 80e9
        capacities = [
            Capacity(
                "compute", "tensor.fp16.m128n256", 1e30, "flop",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "compute"),
            Capacity(
                "l2_read", "l2.read", 1e30, "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "l2-read"),
            Capacity(
                "l2_write", "l2.write", 1e30, "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "l2-write"),
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
            workload, schedule, Hardware("thor", 20, 1.575e9), capacities)
        work = result.work
        self.assertEqual(work.task_count, 128)
        task_bytes = work.tma_input_bytes / work.task_count
        self.assertEqual(
            result.empirical_envelope.resource_seconds[
                "tma.per_sm_parallel_makespan"],
            7 * task_bytes / per_sm_rate,
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
            Hardware("thor", 20, 1.575e9),
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
            workload, schedule, Hardware("one-sm", 1, 1.0), capacities,
        ).conditional_upper
        twenty_sm = evaluate(
            workload, schedule, Hardware("twenty-sm", 20, 1.0), capacities,
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
                "hbm_read", "hbm.read", 1000.0, "byte",
                EvidenceKind.MEASURED_SUSTAINED),
            capacity(
                "hbm_write", "hbm.write", 1000.0, "byte",
                EvidenceKind.MEASURED_SUSTAINED),
            capacity(
                "l2_read", "l2.read", 1000.0, "byte",
                EvidenceKind.MEASURED_SUSTAINED),
            capacity(
                "l2_write", "l2.write", 1000.0, "byte",
                EvidenceKind.MEASURED_SUSTAINED),
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
            Schedule("s", 128, 128, 64, 2),
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
                "l2_read", "l2.read", 1e12, "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "l2-read"),
            Capacity(
                "l2_write", "l2.write", 1e12, "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "test", "source.json", "l2-write"),
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
            Schedule("s", 128, 128, 64, 2),
            Hardware("h", 20, 1.0),
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


class SnapshotEvaluationTest(unittest.TestCase):
    def test_bf16_snapshot_exposes_incomplete_empirical_layer(self) -> None:
        result = evaluate(
            Workload("bf16", 1024, 1024, 1024, "bf16_f32"),
            Schedule("s", 128, 128, 64, 2, tail_policy="pad"),
            Hardware("thor", 20, 1.575e9),
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
        self.assertEqual(envelope.empirical_schedule_id, "padded")
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
        rows = {row.precision_id: row for row in precision_coverage(capacities, observed)}
        self.assertFalse(rows["tf32_f32"].numeric_closure)
        self.assertIn("empirical_compute_rate", rows["tf32_f32"].missing)
        self.assertFalse(rows["nvfp4_f32"].same_precision_performance_denominator)
        self.assertEqual(
            rows["nvfp4_f32"].missing_full_gemm_shapes,
            (1024, 2048, 4096),
        )
        self.assertIn(
            "closure_qualified_full_gemm_shape_matrix",
            rows["nvfp4_f32"].missing,
        )
        self.assertIn(
            "full_gemm_numerical_validation",
            rows["nvfp4_f32"].missing,
        )
        self.assertIn(
            "same_precision_performance_denominator",
            rows["e5m2_f32"].missing,
        )
        self.assertFalse(rows["e4m3_f32"].numeric_closure)
        self.assertIn(
            "closure_qualified_compute_rate", rows["e4m3_f32"].missing
        )
        common = common_resource_coverage(capacities)
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
            metadata={"suite_id": "snapshot", "expected_commit": "none"},
        )
        self.assertEqual(analysis["precision_count"], 12)
        self.assertEqual(analysis["implementation_ready_count"], 5)
        self.assertFalse(analysis["all_precision_numeric_evidence_closed"])
        self.assertFalse(analysis["all_precisions_end_to_end_closed"])
        rows = {row["precision_id"]: row for row in analysis["precisions"]}
        self.assertTrue(rows["fp16_f32"]["implementation_ready"])
        self.assertFalse(rows["fp16_f32"]["numeric_closure"])
        self.assertFalse(rows["e5m2_f32"]["implementation_ready"])
        self.assertIn(
            "same_precision_performance_denominator_impl",
            rows["e5m2_f32"]["support_gaps"],
        )
        document = render_precision_evidence_markdown(analysis)
        self.assertIn("all precisions end-to-end closed：`false`", document)
        self.assertIn("### `e5m2_f32`", document)


if __name__ == "__main__":
    unittest.main()

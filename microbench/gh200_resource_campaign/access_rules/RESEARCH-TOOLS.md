# GH200 独立研究工具

这些入口保存已有开发和反证，不修改公共V09默认预测，也不表示已通过独立留出。

| 工具 | 原交付 | 用途 |
|---|---|---|
| r09_v10_clock_expansion.py、test_r09_v10_clock_expansion.py | clock-parallel / 6ee1c0bd | 任务A的协议分组、合并敏感性与失败拟合 |
| analyze_r18_v10_cluster_residual.py、cluster_first_L_extra.py | boundary-parallel / 8b80aa76 | 首轮cluster加性延迟的失败与参数来源 |
| analyze_r10_block_identity_sensitivity.py | supply-parallel / 1bcc62cb | 已完成的位置/工作身份敏感性复核 |
| analyze_long_k_attribution.py、analyze_long_k_candidate.py、verify_long_k_candidate.py | long-k-event / a4ca14c9 | 旧归因与失败候选独立复现 |
| v09_long_k_rejected.py | a4ca14c9 的v09_model.py原字节副本 | 保留被否决的later-loop接口；只由上述研究脚本显式导入，不从公共v09_model调用 |
| prepare_r03_lds_sts.py、run_r03_lds_sts.py、analyze_r03_lds_sts.py、r03_lds_sts_model.py、probes/r03_lds_sts.cu | smem-lds-sts / a3e5f7ce | 普通LDS/STS探针、原作业复现与宽广播反证；未冻结/未执行独立留出 |

长K三入口只将模型模块导入改名为隔离库，算法不改。复现使用主checkout历史results的显式绝对路径和新输出目录，不能覆盖原失败或frozen。[2026-10-12主线校准](../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/MAINLINE-LONG-K.md#measurement-20261012)已结束；未来测量按[用途声明检查](../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/RESEARCH-GUIDE.md#measurement-checks)，新作业预算另经用户确认。

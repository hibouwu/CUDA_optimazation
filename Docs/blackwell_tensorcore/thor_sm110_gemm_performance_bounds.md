# Thor/SM110 稠密 GEMM：条件性能上界、经验理想包络与实测最好值

> **Legacy / historical notice（2026-08-18）**：本文件保留旧链接、历史 closure 数值和模型收紧过程，不再作为 current 规范入口。当前模型从 [`gemm/README.md`](./gemm/README.md) 开始；现行公式见 [`gemm/model/`](./gemm/model/01_scope_and_claims.md)，实验合同见 [`gemm/experiments/`](./gemm/experiments/EXP-01-compute-surface.md)。本文件中独立 empirical `hbm.read/write`、`l2.read/write` 和旧 128.436 TFLOP/s envelope 只按历史 schema 解读。

> **研究目标**：回答“在 Thor/SM110 的物理约束下，一个没有可避免性能浪费的稠密 GEMM 最快可以到哪里”，而不是只预测仓库中的 `tc3`。
>
> **模型状态**：结构模型、证据分级、工作量计算、完整 GEMM 结果导入和缺口审计已经可执行；Thor composite closure 已由代码提交 `25d8cf71fa566150b64f2eb1dc7f814ce70fa354` 生成，并由结果提交 `ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c` 回传。当前还实现了合同绑定的 persistent-worker causal DAG 求解器，并冻结了 tc5a 的 FP16/BF16 双精度因果采集合同：每种精度 91 case、910 trial、4 份 NCU，总计 182 case、1,820 trial、8 份 NCU；Thor timing profile 尚未回传。按“schedule/precision/row-stride 精确 TMA capacity + closure-qualified causal profile”重新收紧后，当前 12 精度证据矩阵计数为 implementation 6、numeric 4、完整 resource-envelope matrix 0、causal 0、end-to-end 0；新增 E5M2 runner 尚未产生 Thor full-GEMM observation，历史 tc5a 也仅精确支持 N=K=2048 的 hot/cold 两个 resource 场景。不能把 runner 就绪、求解器存在或旧 4/12 numeric closure 称为完整三层模型闭环。
>
> **可信度纪律**：可证明上界、microbenchmark 经验包络和完整 GEMM 实测值分别报告，任何一层都不能冒充另一层。
>
> **范围**：单次、稠密、经典矩阵乘法；不包含稀疏、Strassen/近似算法、batched/grouped GEMM 和多 GPU 通信。

## 0. 可执行 schema 参数首次定义

本节是模型 JSON/Python schema 的规范字典。下表中的空 tuple 表示“不额外收窄该维度”，而不是“已经证明适用于所有硬件”；closure 结论还必须经过硬件、场景与证据等级门禁。单位没有另行写明时为无量纲标识或枚举。

| PrecisionSpec 字段 | 首次定义 |
| --- | --- |
| `precision_id` | 精度与数值合同的稳定 ID。 |
| `input_bytes`, `accumulator_bytes`, `output_bytes` | 每个输入、累加器和输出元素的物理字节数，单位 B/element。 |
| `mma_k` | 该精度一次 MMA atom 的 K 深度，单位 element。 |
| `compute_resource`, `compute_work_unit` | 前者是算术资源 ID；后者是其工作单位，只能为 `flop` 或 `operation`。 |
| `input_scale_block`, `input_scale_bytes` | 输入 block scale 覆盖的元素数和每个 scale 的字节数；非 block-scaled 精度分别为 null 与 0。 |
| `output_scale_block`, `output_scale_bytes` | 输出 scale block 大小和每个输出 scale 的字节数；当前 accumulator-output 合同分别为 null 与 0。 |

| Workload 字段 | 首次定义 |
| --- | --- |
| `workload_id` | 一个 GEMM 问题与验证场景的稳定 ID。 |
| `m`, `n`, `k` | GEMM 的 M、N、K 逻辑维度，单位 element。 |
| `transpose_a`, `transpose_b` | A、B 是否转置；v1 只接受 false/false。 |
| `alpha`, `beta` | (D=\alpha AB+\beta C) 的标量系数。 |
| `epilogue`, `output_mode` | epilogue 语义和输出表示；当前完整模型要求 `none` 与 `accumulator`。 |
| `residency` | 输入入口场景：`cold_hbm`、`hot_l2` 或 `compute_oracle`。 |
| `include_launch` | 是否允许计入已建模的 launch/fixed 时间。 |
| `validation_split` | workload 是 `exploratory`、`calibration` 还是冻结 `holdout`。 |
| `implementation_domain` | 上界覆盖 `tensor_core_classical` 还是包括非 Tensor Core 的 `all_classical`。 |
| `timed_scope` | 计时边界，当前为 `device_kernel` 或 `device_kernel_plus_launch`。 |

| Schedule 字段 | 首次定义 |
| --- | --- |
| `schedule_id` | 一个候选实现 schedule 的稳定 ID。 |
| `bm`, `bn`, `bk` | CTA tile 的 M、N、K 尺寸，单位 element。 |
| `mma_m`, `mma_n` | MMA atom 的 M、N 尺寸，单位 element；K 来自精度的 `mma_k`。 |
| `stages`, `cta_group`, `split_k` | pipeline stage 数、协作 CTA 数和 K 分片数。 |
| `tail_policy`, `supported_precisions` | 尾块处理策略以及该 schedule 明确支持的精度 ID 集。 |
| `smem_limit_bytes`, `tmem_columns`, `registers_per_thread` | 每 CTA SMEM 上限、TMEM 列分配和可选每线程寄存器数。 |
| `threads`, `resident_ctas_per_sm` | 每 CTA 线程数和模型假定每 SM 驻留 CTA 数。 |
| `tmem_load_registers`, `tmem_consumer_warps`, `readback_warps` | LDTM 宽度、消费 warp 数和 readback warp 数；后两者若同时给出必须相同。 |
| `uses_tma`, `tma_destination_slots` | 是否使用 TMA，以及已声明的并行 TMA destination slot 数。 |
| `tma_ingress_capacity_resource`, `tma_hbm_capacity_resource` | 旧冻结合同显式绑定的 per-SM ingress 与全 GPU HBM resource ID。 |
| `tma_contract_family_by_precision`, `tma_contract_row_stride_elements` | 精度到精确 TMA topology family 的映射，以及测过的 packed leading-dimension 集。 |
| `causal_pipeline_resource`, `persistent` | schedule 绑定的 joint causal profile resource，以及是否采用 persistent-worker 调度。 |
| `input_transport_layout`, `input_scale_transport`, `data_path_contract` | 值 transport layout、scale transport 状态和整个 data path 是否完整建模。 |
| `global_memory_access_pattern` | schedule 对全局内存请求的结构化访问模式。 |
| `fixed_seconds` | 已证明的固定时间项，单位 s；0 表示零浪费上界放松，不表示测得固定开销为零。 |

| Hardware 字段 | 首次定义 |
| --- | --- |
| `hardware_id`, `sm_count`, `clock_hz`, `operating_mode` | 硬件稳定 ID、GPU 的 SM 数、GPU 时钟（cycle/s）和功耗/运行模式。 |
| `l2_capacity_bytes` | 可用于 hot-L2 可行性门禁的已证明 L2 容量，单位 B/GPU。 |
| `l2_capacity_evidence_kind` | L2 容量证据类型：设备记录或官方规格。 |
| `l2_capacity_source_path`, `l2_capacity_source_locator` | L2 容量仓库内来源及可机械定位谓词。 |

| Capacity 字段 | 首次定义 |
| --- | --- |
| `capacity_id`, `resource`, `rate_per_second`, `work_unit` | 容量记录 ID、资源 ID、服务率和工作单位；组合单位为 work-unit/s。 |
| `evidence_kind`, `qualification`, `trial_count`, `uncertainty_fraction` | 证据逻辑类型、资格等级、外部 trial 数和相对不确定度。 |
| `source_id`, `source_path`, `source_locator`, `source_url` | 生产者 ID、仓库内来源、机械 locator 和可选一手 URL。 |
| `original_value`, `original_unit`, `condition`, `artifact_paths` | 未换算来源值/单位、成立条件和不可缺失的证据工件路径。 |
| `applicable_precision_ids`, `applicable_mma_shapes`, `applicable_cta_groups` | 可使用该容量的精度、MMA shape 和 CTA-group 精确作用域。 |
| `applicable_sm_counts`, `applicable_hardware_ids`, `applicable_operating_modes`, `applicable_clock_hz` | SM 数、产品 ID、运行模式和时钟作用域。 |
| `applicable_residencies`, `measurement_operand_residency`, `residency_evidence_qualification` | workload residency、probe 请求入口和 residency 是未证明、构造证明还是 NCU 证明。 |
| `applicable_tma_tile_bytes`, `applicable_tma_destination_slots` | 该 TMA 容量测过的 payload 字节数和 destination-slot 合同。 |
| `applicable_tmem_load_registers`, `applicable_readback_warps` | TMEM readback 的 LDTM 宽度和参与 warp 数。 |
| `applicable_threads_per_cta`, `applicable_resident_ctas_per_sm` | probe 的 CTA 线程数和每 SM 驻留 CTA 数。 |
| `applicable_read_write_ratios`, `applicable_access_patterns` | joint memory 容量适用的 issued-byte read:write 比和访问模式。 |
| `applicable_schedule_ids`, `applicable_workload_ids`, `timed_scope` | 精确 schedule/workload ID 与计时边界作用域。 |
| `upper_scope` | rate upper 是单 schedule family、全部 Tensor-Core classical GEMM 还是全部 classical GEMM。 |

| PipelineProfile 字段 | 首次定义 |
| --- | --- |
| `profile_id`, `resource`, `schedule_id`, `precision_ids` | joint profile ID、资源 ID、精确 schedule ID 和允许的 singleton/显式精度集合。 |
| `evidence_kind`, `qualification`, `closure_qualified`, `trial_count_per_case` | profile 的证据类型、资格字符串、布尔资格和每 case 外部 trial 数。 |
| `source_id`, `expected_commit`, `source_path`, `source_locator`, `artifact_paths` | profile 生产 run、冻结 Git commit、源码/locator 与证据工件。 |
| `input_residency`, `timed_scope`, `applicable_sm_counts`, `applicable_hardware_ids`, `applicable_operating_modes`, `applicable_clock_hz` | 输入 residency、计时边界和精确硬件作用域。 |
| `accumulator_buffers`, `resident_ctas_per_sm`, `maximum_k_tiles`, `maximum_output_tasks_per_worker` | accumulator buffer 数、驻留 CTA 数以及允许插值的 K-tile/worker-task 最大范围。 |
| `tma_first_completion_seconds`, `tma_completion_interval_seconds` | TMA 首次完成延迟与稳态完成间隔，单位 s。 |
| `mma_first_completion_seconds`, `mma_completion_interval_seconds` | MMA 首次完成延迟与稳态完成间隔，单位 s。 |
| `joint_first_mma_completion_seconds`, `joint_completion_interval_seconds`, `epilogue_latency_seconds` | joint pipeline 首个 MMA、joint 稳态间隔和单 task epilogue drain，单位 s。 |
| `component_r_squared`, `max_calibration_relative_error`, `max_holdout_relative_error` | 预声明拟合质量与 calibration/holdout 最大相对误差。 |
| `fit_contract`, `validation` | 冻结拟合门禁/坐标集合，以及逐坐标 actual/predicted 验证行。 |

| WorkAccounting 字段 | 首次定义 |
| --- | --- |
| `useful_compute_work`, `issued_compute_work`, `compute_work_unit` | 用户语义要求的算术工作、padding/tail 后实际发射工作及其单位。 |
| `input_value_bytes_min`, `input_scale_bytes_min`, `c_read_bytes_min` | 零外部复用条件下 value、scale 和 βC 的最小读取字节数。 |
| `output_value_bytes_min`, `output_scale_bytes_min` | 逻辑输出 value 与 scale 的最小写字节数。 |
| `tma_unique_input_bytes`, `tma_value_input_bytes`, `tma_scale_input_bytes`, `tma_input_bytes` | unique 输入、schedule value 请求、scale 请求和二者合计的 TMA 字节数。 |
| `tma_a_value_bytes`, `tma_b_value_bytes`, `tma_a_scale_bytes`, `tma_b_scale_bytes` | A/B value 与 A/B scale 各自的 schedule-level TMA 请求字节数；四项保留独立 request 语义。 |
| `tma_a_input_bytes`, `tma_b_input_bytes` | A 与 B 各自 value+scale 的汇总字节数，只用于矩阵侧总量，不得冒充单条 TMA payload。 |
| `tmem_scale_ingress_bytes`, `accumulator_readback_bytes`, `reduction_bytes` | scale-to-TMEM、accumulator readback 和 split-K reduction I/O 字节数。 |
| `task_count`, `output_tiles`, `k_tiles` | CTA task 数、输出 tile 数和每输出 tile 的 K tile 数。 |

| LayerResult 与 WorkloadEnvelope 字段 | 首次定义 |
| --- | --- |
| `status`, `seconds`, `performance_per_second`, `performance_unit` | 一层模型的状态、makespan（s）、性能和性能单位。 |
| `bottlenecks`, `resource_seconds`, `missing_resources`, `conditions` | 并列瓶颈、逐资源时间、缺失资源和成立条件。 |
| `selected_capacity_ids`, `selected_capacity_evidence_kinds`, `selected_capacity_qualifications` | 选中容量 ID、证据类型和资格的审计映射。 |
| `valid_schedule_count`, `rejected_schedule_count`, `rejected` | 合法 schedule 数、拒绝数和逐 schedule 拒绝原因。 |
| `domain_conditional_upper`, `manifest_conditional_upper`, `conditional_schedule_id` | 实现域全局条件上界、manifest 内条件上界和后者的 schedule ID。 |
| `manifest_empirical_resource_envelope`, `empirical_resource_schedule_id` | 独立资源经验层及其 schedule ID。 |
| `causal_pipeline_envelope`, `causal_pipeline_schedule_id` | joint causal profile 层及其 schedule ID。 |
| `empirical_ideal_envelope`, `empirical_schedule_id` | 资源层与因果层取时间最大值后的经验理想包络及其 schedule ID。 |

| ObservedBest 字段 | 首次定义 |
| --- | --- |
| `observation_id`, `backend_id`, `reference`, `selection_rule` | 完整 GEMM observation ID、候选 backend、性能 denominator 和选择规则。 |
| `median_per_second`, `minimum_per_second`, `maximum_per_second`, `performance_unit` | 候选 trial 的中位/最小/最大性能及单位。 |
| `matched_count`, `reference_median_per_second`, `reference_minimum_per_second`, `reference_maximum_per_second`, `ratio_of_paired_medians` | correctness 匹配数、reference 三个统计量和候选/参考 paired median 比。 |
| `performance_reference_relation`, `correctness_reference`, `correctness_reference_relation`, `numerical_contract` | 性能 denominator 与 correctness reference 是否同精度/同合同，以及执行数值合同 ID。 |
| `calibration_split`, `arithmetic_path` | observation 的 calibration/holdout 角色及经 SASS 审计的算术路径。 |
| `run_id`, `operating_mode` | 生产 run ID 与运行功耗模式；其余 workload、硬件和 provenance 字段沿用前述同名定义。 |

| Coverage 字段 | 首次定义 |
| --- | --- |
| `conditional_upper_numeric`, `conditional_upper_complete`, `conditional_upper_per_second` | 场景 domain 上界是否有数值、是否资源完整以及数值本身。 |
| `closure_qualified_empirical_envelope`, `empirical_envelope_per_second`, `empirical_schedule_id` | 场景经验包络是否由合格证据闭合、其数值与 schedule。 |
| `contract_matched_full_gemm`, `scenario_aligned_full_gemm`, `observed_median_per_second` | 是否有问题合同匹配、residency/timing 对齐的完整 GEMM 及其中位性能。 |
| `upper_consistent`, `empirical_envelope_consistent`, `empirical_to_observed_ratio` | observation 是否不违反上界/包络，以及包络除以 observation 的比值。 |
| `numeric_closure`, `absolute_three_layer_closure`, `same_precision_ratio_closure`, `missing` | 数值证据、domain upper+经验包络+observation 三层、同精度比值是否闭合，以及缺口列表。 |
| `calibration_workload_ids`, `holdout_workload_ids`, `complete` | manifest 中 calibration/holdout workload ID 集及二者是否齐全。 |
| `domain_compute_upper`, `empirical_compute_rate`, `closure_qualified_compute_rate` | 每精度/实现域的算术 domain upper、任意经验 rate 和合格经验 rate 状态。 |
| `strict_compute_upper`, `required_compute_shapes`, `closure_qualified_compute_shapes`, `compute_shape_matrix_complete`, `missing_compute_shapes` | 旧 campaign 的严格 compute upper 与三 shape compute matrix 审计字段。 |
| `full_gemm_observed`, `closure_qualified_full_gemm`, `required_full_gemm_shapes`, `observed_full_gemm_shapes`, `closure_qualified_full_gemm_shapes`, `full_gemm_shape_matrix_complete`, `full_gemm_numerical_validation_complete`, `missing_full_gemm_shapes` | 完整 GEMM 三 shape matrix 的存在、资格、correctness 与缺口。 |
| `same_precision_performance_denominator`, `calibration_scenario_closure`, `holdout_scenario_closure`, `evidence_missing`, `comparison_missing` | 同精度 denominator、两个 split 的三层 closure，以及绝对证据/相对比较缺口。 |

| Suite 与目标完成度字段 | 首次定义 |
| --- | --- |
| `suite_id`, `expected_commit`, `hostname`, `gpu_identity`, `ncu_required` | suite ID、冻结提交、主机/GPU 身份及是否强制 NCU。 |
| `compute_run_id`, `component_run_id`, `full_gemm_run_id` | 三批相互链接的 run ID。 |
| `source_paths`, `source_urls` | suite 声明的仓库内源文件集合与外部一手 URL 集合。 |
| `compute_campaign_case_ids`, `compute_campaign_full_gpu_case_ids`, `legal_schedule_ids`, `complete_data_path_schedule_ids` | 每精度 compute case、full-GPU case、合法 schedule 和完整 data-path schedule 集合。 |
| `candidate_tma_payload_bytes`, `required_tma_payload_bytes`, `closure_qualified_tma_payload_bytes`, `required_tmem_readback_contracts` | 候选/必须/已实测 TMA payload 与必须 TMEM readback 合同。 |
| `required_hbm_duplex_read_write_ratios`, `closure_qualified_hbm_duplex_proxy_ratios`, `closure_qualified_hbm_duplex_ratios` | 必须 HBM ratio、已测 cold proxy ratio 和具备物理外部读写字节证明的 ratio。 |
| `candidate_l2_duplex_read_write_ratios`, `required_l2_duplex_read_write_ratios`, `closure_qualified_l2_duplex_ratios` | 候选、必须和已合格实测的 L2 read:write ratio。 |
| `required_joint_pipeline_contracts`, `closure_qualified_joint_pipeline_contracts` | 必须的 pipeline 合同，以及已由 exact joint capacity 或硬件/精度/拓扑/range 全匹配 causal profile 闭合的合同 ID。 |
| `full_gemm_support_status`, `full_gemm_campaign_case_ids`, `full_gemm_calibration_case_ids`, `full_gemm_holdout_case_ids`, `full_gemm_scenario_qualified_case_ids` | 每精度完整 GEMM 支持状态、全部 case、两个 split 和 residency-qualified case 集。 |
| `precision_audits`, `global_missing` | 逐精度审计行与全局缺口列表。 |
| `all_precision_contracts_present`, `all_compute_campaigns_planned`, `all_complete_data_paths_modeled`, `all_required_tma_payloads_planned` | 精度合同、compute 计划、data path 和 payload case 是否完整声明。 |
| `all_required_tma_payloads_measured`, `all_required_hbm_duplex_proxies_measured`, `all_required_hbm_duplex_ratios_measured`, `all_required_l2_duplex_ratios_measured` | payload、cold proxy、物理 HBM duplex 和 L2 duplex 的必须矩阵是否实测闭合。 |
| `all_full_gemm_campaigns_planned`, `all_full_gemm_scenarios_planned`, `all_precisions_absolute_three_layer_closed` | 完整 GEMM case/场景是否规划，以及所有精度是否三层闭合。 |
| `duplex_campaign_frozen`, `epilogue_campaign_frozen`, `joint_pipeline_campaign_frozen` | 三类补充 campaign 的源码/case/runner/auditor hash freeze 是否有效。 |
| `dependency_span_model_complete`, `hardware_capacity_source_present`, `cache_residency_model_complete`, `joint_overlap_model_complete`, `final_source_appendix_generated` | 依赖跨度、硬件容量来源、cache residency、joint overlap 和最终来源附录门禁。 |

| 顶层输出字段 | 首次定义 |
| --- | --- |
| `precision_coverage`, `scenario_coverage`, `workload_manifest_coverage`, `common_resource_coverage`, `target_completion` | coverage 输出的逐精度、逐场景、manifest、公共资源与完整目标审计对象。 |
| `all_precisions_numerically_closed`, `all_precisions_absolute_three_layer_closed`, `all_precisions_same_precision_ratio_closed`, `all_common_resources_closed`, `all_precisions_workload_manifest_complete` | 五个互不替代的聚合布尔门禁。 |
| `suite_linkage`, `imported_capacities`, `closure_observations`, `capacity_findings`, `coverage` | suite 输出的链接证明、导入容量、完整 GEMM observation、审计 finding 和 coverage 对象。 |

这里的顶层 `complete` 只在所有逐精度与全局门禁均通过时为 true，绝不把 runner 已规划、proxy 已测或局部数值存在当作最终完成。

第一次学习本模型时，建议先阅读伴随式教程
[`thor_sm110_gemm_performance_model_tutorial.md`](./thor_sm110_gemm_performance_model_tutorial.md)；
逐精度的当前实现与证据缺口见机器生成的
[`thor_sm110_all_precision_evidence_matrix.md`](./thor_sm110_all_precision_evidence_matrix.md)；
本文继续作为严格定义、最终证据和审计合同。

本文使用的 microbenchmark 研究问题、参数首次定义、case matrix、计时/NCU
门禁、自动图表、当前缺口与源码索引集中记录在
[`thor_sm110_gemm_microbenchmark_experiment.md`](./thor_sm110_gemm_microbenchmark_experiment.md)；
操作命令和失败恢复流程见
[`microbench/README.md`](../../microbench/README.md)。

## 1. 先给出结论

“比 cuBLAS 还完美的 GEMM”有三种不同含义，必须先分开：

1. **条件性能上界**描述在一组明确的硬件容量上界和算法假设下，任何合法实现都不能超过的位置。
2. **经验理想包络**描述按照现有 microbenchmark 测得的组件能力和已枚举的合法 schedule，一个消除了已知实现损失的 GEMM 应该达到的位置。
3. **完整 GEMM 已观测最好值**描述 cuBLAS、cuBLASLt、CUTLASS 和仓库 kernel 中，已经通过 correctness 验证的最好实测性能。

定义 \(P_{\mathrm{obs}}\) 为完整 GEMM 已观测最好性能；对浮点模式其单位为
FLOP/s，对 S8/U8 整数模式其单位为 OP/s。定义 \(P^\star\) 为所有物理可实现
GEMM 中真实但未知的最好性能；定义 \(P_{\mathrm{ub}}\) 为条件性能上界，二者
使用与 workload 相同的性能单位。在上界假设全部成立且
workload 语义一致时，必须满足：

主变量 \(P\) 保持 FLOP/s 或 OP/s，因为完整 kernel 的 elapsed time、HBM/L2
服务率与库结果最终都在秒域比较。另定义归一化指标
\(\Pi=P/f_g\)，其中 \(f_g\) 是与该证据完全相同运行区间和 clock domain 的 GPU
频率，\(\Pi\) 的单位为 FLOP/cycle/GPU 或 OP/cycle/GPU；若确实需要 per-SM
展示，再定义 \(\Pi_{\mathrm{SM}}=P/(S f_g)\)，其中 \(S\) 是参与计时的 SM 数。
\(\Pi\) 适合在同硬件作用域内解释利用率，不替代 \(P\)：不同 boost 区间、不同
clock domain、GPU-wide shared bus 与 per-SM 出口都禁止仅凭“每 cycle”数值直接
相除或相加。

\[
P_{\mathrm{obs}}\le P^\star\le P_{\mathrm{ub}}.
\]

定义 \(\widehat P_{\mathrm{env}}\) 为 microbenchmark 驱动的经验理想包络，单位
同 workload。它不自动进入上面的不等式。原因是 microbenchmark 实测值只能证明
“硬件至少已经做到这么快”，不能证明“硬件绝不可能更快”。

因此本文不会把最高 microbenchmark TFLOP/s 直接称为硬件物理上限。如果完整
GEMM 超过 \(\widehat P_{\mathrm{env}}\)，说明经验模型需要重校准；如果完整 GEMM
在语义相同的前提下超过 \(P_{\mathrm{ub}}\)，则上界的容量、工作量或适用条件至少
有一项错误。

## 2. 冻结第一版 GEMM 语义

定义 \(M\) 为输出矩阵的行数，单位为 element；定义 \(N\) 为输出矩阵的列数，
单位为 element；定义 \(K\) 为 reduction 维度，单位为 element。定义 \(A\) 和
\(B\) 为两个输入矩阵，定义 \(C\) 为可选的输入累加矩阵，定义 \(D\) 为输出
矩阵；定义 \(\operatorname{op}(\cdot)\) 为保持原布局或取转置的矩阵操作；定义
\(\alpha\) 为矩阵乘积的无量纲标量系数，定义 \(\beta\) 为矩阵 C 的无量纲标量
系数。在这些参数定义下，第一版 workload 计算：

\[
D=\alpha\operatorname{op}(A)\operatorname{op}(B)+\beta C.
\]

当 \(\beta=0\) 时，理想实现不必读取 C。

第一版固定以下边界：

- \(M,N,K\) 都是正整数，且 \(\alpha\ne0\)；
- workload schema 能声明 NN、NT、TN 和 TT 四种 A/B 转置组合；当前 v1 的最低
  工作量对四者相同，但 schedule manifest 尚未把布局与 leading dimension 的
  合法 data-movement 路径枚举完整，因此不能把当前搜索结果称为四种布局都完备；
- 独立 leading dimension 尚未进入可执行 schema；当前工作量公式只统计逻辑
  元素，不把行间 padding 当作必然传输量；
- v1 可执行数值模型只接受 `epilogue=none`。bias、ReLU、GELU、residual 和
  requant 是后续合同；在其工作量和 I/O 未实现前会 fail closed；
- 计时默认只包括设备端 GEMM，不包括 host-device copy、内存分配和一次性
  prepack；如果业务要求包含，必须在 workload 中单独声明；
- `cold_hbm` 表示模型入口只保证输入位于设备内存，`hot_l2` 表示输入工作集已
  预热到 L2，`compute_oracle` 表示操作数已经位于 Tensor Core 可消费位置；
- 经典 GEMM 的用户可见数学工作量固定为 \(2MNK\) 次标量操作，不允许用稀疏或快速矩阵
  乘法减少这一定义。

batched/grouped GEMM、跨算子 persistent reuse、稀疏 GEMM、Strassen、近似
矩阵乘法、多 GPU、PCIe/NVLink 和 host-device copy 不在第一版内。对声明范围
的“完备”不等于声称覆盖这些排除项。

## 3. 第一版精度合同

定义 \(s_{\mathrm{in}}\) 为单个 A/B 输入元素的平均存储字节数，单位为
B/element；定义 \(s_{\mathrm{acc}}\) 为单个 accumulator 元素的存储字节数，
单位为 B/element；定义 \(s_{\mathrm{out}}\) 为单个输出元素的存储字节数，单位
为 B/element；定义 \(K_{\mathrm{mma}}\) 为该精度 MMA 原子在 K 方向一次消费的
元素数，单位为 element/instruction。

| 模式 ID | \(s_{\mathrm{in}}\) | \(s_{\mathrm{acc}}\) | \(s_{\mathrm{out}}\) | \(K_{\mathrm{mma}}\) | scale 语义 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fp16_f32` | 2 | 4 | 4 | 16 | 无 |
| `bf16_f32` | 2 | 4 | 4 | 16 | 无 |
| `tf32_f32` | 4 | 4 | 4 | 8 | 无 |
| `e4m3_f32` | 1 | 4 | 4 | 32 | 无 |
| `e5m2_f32` | 1 | 4 | 4 | 32 | 无 |
| `e3m2_f32` | 0.75 | 4 | 4 | 32 | 无 |
| `e2m3_f32` | 0.75 | 4 | 4 | 32 | 无 |
| `e2m1_f32` | 0.5 | 4 | 4 | 32 | 无 |
| `mxfp4_f32` | 0.5 | 4 | 4 | 64 | 每 32 个输入值 1 B UE8M0 scale |
| `nvfp4_f32` | 0.5 | 4 | 4 | 64 | 每 16 个输入值 1 B UE4M3 scale |
| `s8_s32` | 1 | 4 | 4 | 32 | signed INT8 → INT32 |
| `u8_s32` | 1 | 4 | 4 | 32 | unsigned INT8 → INT32 |

FP6/FP4 的分数字节只表示 workload 合同所声明的紧凑逻辑存储下界，不表示
SMEM/TMEM 或某条 copy 指令必然以 0.75 B/0.5 B 的物理 transaction 搬运。
例如 PTX ISA 9.0 的 `tcgen05.cp` FP6 解压格式是每 16 个 6-bit 元素（12 B）
再带 4 B padding；若 schedule 使用这条路径，其 issued traffic 必须按实际
16 B 统计，不能沿用 12 B 的逻辑下界。raw `f8f6f4` 直接从 SMEM 消费和先经
`tcgen05.cp` 解压到 TMEM 是两种不同 schedule，模型必须分别声明。

MXFP4/NVFP4 的 value bytes、input scale bytes、accumulator bytes、output
value bytes 和 output scale bytes 必须分开统计。特别是 FP32 accumulator 从
TMEM 回读时仍按 \(s_{\mathrm{acc}}=4\) B/element 统计，不能用 packed E2M1 的
0.5 B/element 代替。

## 4. 工作量：先证明必须做什么

### 4.1 用户工作与发射工作

定义 \(W_{\mathrm{use}}\) 为用户要求的经典 GEMM 标量计算工作量；浮点模式单位
为 FLOP，S8/U8 模式单位为 OP：

\[
W_{\mathrm{use}}=2MNK.
\]

这里一次 multiply-add 按一次乘法和一次加法计为 2 个标量操作；只有浮点
workload 才把它称为 2 FLOP，整数 workload 写成 2 OP。

定义 \(B_M\) 为 schedule 的 CTA tile 在 M 方向的元素数，定义 \(B_N\) 为其在
N 方向的元素数，定义 \(B_K\) 为其在 K 方向的元素数，单位均为 element；定义
\(N_M=\lceil M/B_M\rceil\) 为 M 方向 tile 数，定义
\(N_N=\lceil N/B_N\rceil\) 为 N 方向 tile 数，定义
\(N_K=\lceil K/B_K\rceil\) 为 K 方向 tile 数，三者单位均为 tile。

定义 \(W_{\mathrm{reduce}}\) 为 split-K 最终 reduction 增加的计算工作，单位
同 workload；当前 v1 因 `split_k=1` 而令其为 0。若 schedule 用完整 tile
padding 边界，定义 \(W_{\mathrm{issue}}\) 为实际发射的计算工作，单位与
\(W_{\mathrm{use}}\) 相同：

\[
W_{\mathrm{issue}}
=2(N_MB_M)(N_NB_N)(N_KB_K)+W_{\mathrm{reduce}}.
\]

若使用恰好覆盖边界的专用 tail kernel，则 padding 项可以消失，但 tail
kernel 的合法指令粒度和固定成本仍需单独建模。当前 v1 对非整除 shape 的 `exact`
schedule 直接拒绝；只有加入该 tail kernel 的显式 manifest 后才会放行，避免凭空
假设任意尺寸的无 padding 指令。

定义形状效率 \(\eta_{\mathrm{shape}}\) 为：

\[
\eta_{\mathrm{shape}}=\frac{W_{\mathrm{use}}}{W_{\mathrm{issue}}}.
\]

### 4.2 通用最小 I/O

定义 \(Q_{\mathrm{in,val}}^{\mathrm{LB}}\) 为至少读取一次 A/B value 的字节数下界，
单位为 B：

\[
Q_{\mathrm{in,val}}^{\mathrm{LB}}=(MK+KN)s_{\mathrm{in}}.
\]

对 block-scaled 精度，定义 \(b_s\) 为一个 scale 覆盖的 value 个数，单位为
element/scale；定义 \(s_s\) 为单个 scale 的字节数，单位为 B/scale。输入
scale 的字节数下界定义为 \(Q_{\mathrm{in,scale}}^{\mathrm{LB}}\)，单位为 B：

\[
Q_{\mathrm{in,scale}}^{\mathrm{LB}} =
\left(
M\left\lceil\frac{K}{b_s}\right\rceil
+
N\left\lceil\frac{K}{b_s}\right\rceil
\right)s_s.
\]

这里每个 A 行向量和每个 B 列向量都在自己的 K 方向上独立分块；一个 scale
block 不允许跨过两个 K 向量的边界。因此不能先把 \(MK\) 或 \(KN\) 展平再只做
一次向上取整。无 scale 的精度令这一项为 0。

定义 \(Q_C^{\mathrm{LB}}\) 为 C 的最小读取量，单位为 B：

\[
Q_C^{\mathrm{LB}}=
\begin{cases}
0,&\beta=0,\\
MN s_{\mathrm{acc}},&\beta\ne0.
\end{cases}
\]

定义 \(Q_D^{\mathrm{LB}}\) 为最终输出至少写一次的字节数，单位为 B。普通
accumulator 输出时 \(Q_D^{\mathrm{LB}}=MN s_{\mathrm{out}}\)；packed quantized
输出还必须加 value packing 和 output scale。

上述公式只是对任何经典实现都成立的最小 I/O。具体 schedule 的 TMA 逻辑流量
可能因 output tile 重用边界、CTA group、split-K 和 cache 行为显著增加。不能
把 `TMA payload bytes`、`L2 request bytes` 和 `DRAM physical bytes` 当作同一
个量。

定义 schedule 描述 \(x\) 为 tile、MMA atom、stage、CTA group、split/stream-K、
persistent、tail 和资源 footprint 的集合；定义 workload 描述 \(w\) 为尺寸、
精度、转置、\(\alpha/\beta\)、epilogue、residency 和计时边界的集合。对当前
NN、CTA-group-1、完整 tile 的 schedule，定义 \(Q_{\mathrm{TMA}}(x,w)\) 为
schedule \(x\) 执行 workload \(w\) 发出的 TMA 输入 payload，单位为 B。
每个 output tile 的每个 K tile 都搬运一份 A/B tile；value 部分按
`input_transport_layout` 所声明的物理布局计数。对 block-scaled 输入，定义
\(a_s=128\) 为 scale transport atom 在 M/N 方向覆盖的 vector 数，定义
\(g_s=4\) 为该 atom 在 K 方向容纳的 scale-group 数；二者来自 Blackwell
`128 x 4` scale-factor storage atom。继续定义
\(S(X,B_K,b_s,s_s)\) 为一个外维为 \(X\)、K tile 为 \(B_K\) 的 SFA 或
SFB 物理 transport payload，单位为 B：

\[
S(X,B_K,b_s,s_s) =
\left\lceil\frac{X}{a_s}\right\rceil a_s
\left\lceil
  \frac{\left\lceil B_K/b_s\right\rceil}{g_s}
\right\rceil g_s s_s.
\]

定义
\(Q_{\mathrm{TMA,scale/tile}}\) 为一个 output/K tile 的 scale
transport payload，单位为 B：

\[
Q_{\mathrm{TMA,scale/tile}}
=S(B_M,B_K,b_s,s_s)+S(B_N,B_K,b_s,s_s).
\]

因此不能把逻辑 scale 数直接当成 transport bytes。例如 NVFP4 的
\(B_M=128,B_N=64,B_K=64,b_s=16,s_s=1\) 虽然只有 768 B 逻辑 scale，
但 SFA 与 SFB 都各占一个 512 B atom，transport payload 是 1024 B；MXFP4
在同一 \(B_K=64\) 下只有两个逻辑 K scale group，也必须补齐为四个。

因此 \(Q_{\mathrm{TMA}}(x,w)=N_MN_NN_K\) 乘以单 tile 的 value bytes 与上式
scale bytes 之和。该量是请求给 TMA data path 的 payload，不自动等于 L2
request bytes 或 DRAM physical bytes。

定义 \(Q_{\mathrm{TMA,scale}}(x,w)=N_MN_NN_K
Q_{\mathrm{TMA,scale/tile}}\) 为所有 output/K tile 的 scale TMA payload，单位
为 B。对 block-scaled MMA，这些 scale 到达 SMEM 后仍必须按 PTX 规定的 SFA/SFB
layout 进入 TMEM；定义 \(Q_{\mathrm{tmem,scale}}(x,w)\) 为这条 scale ingress
按**唯一 SMEM source payload**归一化的 bytes。当前 CTA-local schedule 不跨
output tile 共享 TMEM scale，故
\(Q_{\mathrm{tmem,scale}}=Q_{\mathrm{TMA,scale}}\)。PTX ISA 9.0 对 block16
明确要求每行四个 scale 位于 4-byte-aligned TMEM sub-column；它证明这条 TMEM
operand 路径存在，但没有给出服务率。CUTLASS 的同构 S2T atom 使用
`tcgen05.cp.cta_group::1.32x128b.warpx4`：每条指令读取一个 512 B source atom，
并把它 multicast 到四个 32-lane TMEM partition。模型和 capacity 均以 512 B
source payload 计费，不能把 2048 B multicast destination footprint 再乘一次。
因此在同构 `tmem.scale_ingress` capacity
实测完成前，MXFP4/NVFP4 的经验包络必须显示 `insufficient_evidence`，不能用
TMA rate 或 accumulator readback rate 代替它。布局来源见
[NVIDIA PTX ISA 9.0 block-scaling SFA/SFB layout](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-mma-scale-factor-a-layout-4x)，
S2T atom 来源见
[NVIDIA CUTLASS tcgen05 programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html#block-scaled-mma)。

定义 \(Q_{\mathrm{TMA,unique}}(x,w)\) 为该 schedule 在完美跨 output-tile
cache reuse 下只计一次的 A/B 物理输入并集，单位为 B。它按
`input_transport_layout` 统计 value bytes，按“每个 K 向量独立分块”的规则统计
scale bytes，并在 `tail_policy=pad` 时使用补齐后的
\(N_MB_M\)、\(N_NB_N\) 和 \(N_KB_K\) 范围。于是当前经验层明确区分：

- `hot_l2`：完整的 \(Q_{\mathrm{TMA}}\) 约束整 GPU 共享的 `l2.read`
  request；没有 HBM read 约束；
- `cold_hbm`：\(Q_{\mathrm{TMA,unique}}\) 约束首次 `tma.hbm` ingress 和
  `hbm.read`，而完整的 \(Q_{\mathrm{TMA}}\) 仍约束共享的 `l2.read`；
- C read 另加到对应的 HBM/L2 read 工作量，用户可见 D store 另由
  \(Q_D^{\mathrm{LB}}\) 约束 write path。

L2 总线与 SM 本地出口不是同一个资源。定义
\(N_{\mathrm{task}}=N_MN_N\) 为 output-tile task 数，单位 task；定义
\(Q_{\mathrm{TMA/task}}=Q_{\mathrm{TMA}}/N_{\mathrm{task}}\) 为一个 task 在
整个 K loop 发出的 TMA payload，单位 B/task；定义
\(\widehat C_{\mathrm{TMA,SM}}\) 为一个 SM 的 TMA→SMEM sustained ingress，
单位 B/s/SM；定义 \(S\) 为可用 SM 数。CTA-group-1、每 SM 一个 persistent
worker 的本地出口 makespan 为：

\[
\widehat T_{\mathrm{TMA,local}} =
\left\lceil\frac{N_{\mathrm{task}}}{S}\right\rceil
\frac{Q_{\mathrm{TMA/task}}}
     {\widehat C_{\mathrm{TMA,SM}}}.
\]

同时，完整 \(Q_{\mathrm{TMA}}\) 仍受整 GPU 共享 `l2.read` rate 约束。Thor
上 NCU 给出的 L2 read model peak 是 1024 B/cycle/GPU，不是 1024
B/cycle/SM。component campaign 用单个 CTA、单个观测 SM 直接隔离
`tma.smem_ingress.per_sm`；该 rate 不除以设备 SM 数。若让 20 个 SM 同时发起
TMA，再把 aggregate rate 除以 20，测量本身可能已经被共享 L2 总线限速，因而
不能独立证明每 SM 出口。整卡 `l2.read` 继续由单独的全网格 memory-path case
测量。

这等价于允许理想 schedule 进行完美的跨 CTA L2 reuse，但没有把入口冷数据
凭空变成 L2 resident。不同接口的时间在资源层取最大值，表示理想流水重叠；若
后续联合 microbenchmark 证明这些路径不能同时达到各自 rate，再增加联合容量
约束。

`hbm.*` 和 `cold_hbm` 是模型中沿用的通用“外部 DRAM 边界”资源名，不是在断言
Jetson Thor 使用 HBM 器件；Thor T5000 的物理内存是 LPDDR5X。本文在 Thor 上
出现 HBM 字样时，均应读作 LPDDR5X/DRAM 冷入口场景。

定义 \(Q_{\mathrm{tmem}}(x,w)\) 为 accumulator 从 TMEM 回读到寄存器的 issued
payload，单位为 B。当前 `tail_policy=pad` 的 schedule 使用完整输出 tile 的
固定宽度 TMEM load，再只对用户可见范围执行有效 GMEM store；`tail_policy=exact`
只在 shape 被 tile 恰好整除时合法。因此：

\[
Q_{\mathrm{tmem}}(x,w)=
\begin{cases}
(N_MB_M)(N_NB_N)s_{\mathrm{acc}},&\texttt{tail\_policy=pad},\\
MN s_{\mathrm{acc}},&\texttt{tail\_policy=exact}.
\end{cases}
\]

这里 \(Q_D^{\mathrm{LB}}\) 仍只统计用户可见的有效输出 store；TMEM issued
payload 和 GMEM 最小写回量处在不同资源边界，不能相互替代。

### 4.3 可执行输入参数合同

下面列出代码中出现、但不一定进入上述闭式公式的字段。字段首次在本表出现时即
给出定义与单位，避免同名参数靠读者猜测。

| workload 字段 | 定义与单位 |
| --- | --- |
| `workload_id` | workload 的稳定字符串标识，无单位 |
| `m`, `n`, `k` | 分别对应 \(M,N,K\)，单位 element |
| `precision_id` | 第 3 节某一精度合同的稳定标识，无单位 |
| `transpose_a`, `transpose_b` | 是否对 A/B 取转置的布尔值；v1 只执行二者均为 false 的 NN 路径 |
| `alpha`, `beta` | 分别对应 \(\alpha,\beta\) 的无量纲标量 |
| `epilogue` | 输出后处理语义枚举；v1 仅实现 `none` |
| `residency` | 入口数据驻留合同：`cold_hbm`、`hot_l2` 或 `compute_oracle` |
| `output_mode` | `accumulator` 或 `packed_quantized` 输出存储合同；v1 只实现前者 |
| `include_launch` | 是否允许经验层计入 launch/fixed time 的布尔值 |

| schedule 字段 | 定义与单位 |
| --- | --- |
| `schedule_id` | schedule 的稳定字符串标识，无单位 |
| `bm`, `bn`, `bk` | 对应 \(B_M,B_N,B_K\) 的 CTA tile 尺寸，单位 element |
| `stages` | pipeline 中同时驻留的 stage 数，单位 stage |
| `mma_m`, `mma_n` | 一条 MMA atom 的 M/N 尺寸，单位 element/instruction |
| `cta_group` | 一条操作协作的 CTA 数，只能为 1 或 2，单位 CTA/group；v1 只执行 1 |
| `split_k` | K 方向独立 partial 数，单位 partition；v1 只执行 1 |
| `tail_policy` | `pad` 或 `exact`；`pad` 发射完整 compute tile 和完整宽度 TMEM readback，但可屏蔽越界 GMEM store；非整除 `exact` 在 v1 中 fail closed |
| `supported_precisions` | schedule 显式允许的 `precision_id` 集合 |
| `smem_limit_bytes` | 单 CTA 可用于该 schedule 的 SMEM 上限，单位 B/CTA |
| `tmem_columns` | 分配的 TMEM column 数，单位 column/CTA；当前 block-scaled schedule 为 accumulator 与 SFA/SFB 固定使用 512-column 合同 |
| `threads` | CTA 线程数，单位 thread/CTA |
| `tmem_load_registers` | 每个参与 warp 的 TMEM readback 指令写入寄存器数，只能为 8 或 16，单位 32-bit register/thread；分别对应 `LDTM.x8`/`LDTM.x16` |
| `tmem_consumer_warps` | 可选的 TMEM readback 消费 warp 数，单位 warp/CTA；省略时默认 `threads/32`。tc5a 的 CTA 有 6 个 warp，但只有 4 个 epilogue warp 消费 TMEM，因此必须显式设为 4 |
| `registers_per_thread` | 可选寄存器占用，单位 32-bit register/thread |
| `uses_tma` | 是否声明使用 TMA data path 的布尔值；v1 只实现 true，false 在缺少另一套 ingress 合同时 fail closed |
| `tma_ingress_capacity_resource` | 可选的、经过审计且与 payload/request/stage/thread/cache/SM-coverage 合同匹配的 per-SM TMA ingress resource ID；未声明时 memory-resident 经验层 fail closed，不再只按 stage 数猜测 |
| `tma_hbm_capacity_resource` | 可选的、与同一 schedule cold-entry 合同匹配的整卡 TMA/DRAM ingress resource ID；`cold_hbm` 未声明时经验层 fail closed |
| `tma_contract_family_by_precision` | 从 `precision_id` 到精确 TMA transport-family ID 的有限映射，无单位；family 冻结 tile、value/scale payload、request、stage、thread 与驻留合同 |
| `tma_contract_row_stride_elements` | 已采集的共同 A/B packed row stride 集合，单位 element/row；v1 中 A/B leading dimension 分别为 (K,N)，当前合同只在 (K=N) 且该值属于集合时可用 |
| `input_transport_layout` | 输入物理搬运布局；`logical_packed` 是精度合同允许的紧凑 payload，`byte_padded` 是 raw FP6/FP4 direct-SMEM 的 b8 container，`b6x16_p32`/`b4x16_p64` 是显式 `tcgen05.cp` 物理格式 |
| `causal_pipeline_resource` | persistent schedule 绑定的联合 TMA/MMA/epilogue profile resource ID；没有精确 profile 时因果层 fail closed |
| `persistent` | 是否声明 persistent worker 调度的布尔值；true 必须同时声明 `causal_pipeline_resource` |
| `fixed_seconds` | 经验层已测固定成本，单位 s；严格层不使用实测固定成本 |

| causal profile 字段 | 定义与单位 |
| --- | --- |
| `profile_id` | 一次联合时序 profile 的稳定字符串标识，无单位 |
| `resource` | profile 所刻画的联合流水线 resource ID，无单位 |
| `schedule_id` | profile 精确适用的 schedule ID，无单位 |
| `precision_ids` | 已由该 profile 的源码、算术指令和数据格式直接验证的 `precision_id` 集合；不是“输入字节数相同”的推断集合，无单位 |
| `input_residency` | profile 采集时的输入驻留合同；v1 因果 profile 固定为 `hot_l2` |
| `stages` | profile 实测的流水线 stage 数，单位 stage |
| `accumulator_buffers` | persistent worker 轮换使用的 accumulator 数，单位 buffer |
| `resident_ctas_per_sm` | profile 假设的每 SM 常驻 CTA 数，单位 CTA/SM |
| `maximum_k_tiles` | 校准或留出集覆盖的最大 K tile 数，单位 tile/output-task |
| `maximum_output_tasks_per_worker` | 校准或留出集覆盖的最慢 worker 最大输出 task 数，单位 task/worker |

| hardware 字段 | 定义与单位 |
| --- | --- |
| `hardware_id` | 硬件配置的稳定标识，无单位 |
| `sm_count` | 可用 SM 数，单位 SM/GPU |
| `clock_hz` | 被记录的 GPU 时钟，单位 cycle/s；它是环境字段，不替代实测 elapsed time |

| capacity 字段 | 定义与单位 |
| --- | --- |
| `capacity_id` | 容量记录的稳定标识，无单位 |
| `resource` | 被约束资源的稳定标识，无单位 |
| `rate_per_second`, `work_unit` | 前者定义为 SI 基础单位下的服务率；后者定义该服务率的工作单位，组合单位为 `work_unit`/s |
| `work_unit` | `flop`、`operation` 或 `byte` |
| `evidence_kind` | 第 5.1 节定义的逻辑证据等级 |
| `source_id` | 来源记录的稳定标识，无单位 |
| `source_path`, `source_locator` | 仓库内文件路径及文件内可机械定位条件 |
| `source_url` | 外部一手来源 URL；`specified_upper` 必填 HTTPS URL，其余证据可选 |
| `original_value`, `original_unit` | 来源中的未换算数值及单位 |
| `condition` | 容量成立所需的 workload、功耗、频率或工具条件 |
| `uncertainty_fraction` | 相对不确定度，范围 \([0,1)\)，无量纲；v1 保存但尚不传播到中心值 |
| `qualification` | `snapshot_only`、`closure_qualified` 或 `quarantined` |
| `trial_count` | 支持该记录的独立 trial 数，单位 trial |
| `artifact_paths` | closure 所依赖的源码、原始结果、SASS/NCU、环境或 hash 路径集合 |

同一资源若已有 `closure_qualified` capacity，经验包络只在这些同合同点中取最大
实测 rate；旧 `snapshot_only` 即使数值更高也不再混入选择。只有该资源尚无
closure 点时，快照才继续作为显式的暂定校准值。严格条件上界不使用任何实测
capacity，仍取所有同时成立 rate upper 的最小值。

`tma.smem_ingress.per_sm` 是明确的每 SM rate。当前 closure 的 L2-hit TMA case
只启动一个 CTA，并要求 `sm_count=20`、`blocks=1`、
`unique_smid_count=1`；其 `%globaltimer` rate 直接作为单 SM 出口证据。模型按
task waves 使用该值，而不是把 aggregate payload 直接除以一个伪造的全局
TMA-L2 资源。`l2.read` 仍独立表示共享 L2 read 总线。DRAM-stream TMA case
仍启动 20 个 CTA、覆盖 20 个 SM，用于测量冷入口的整卡 aggregate rate。

“同合同点”不是只看名称相似。通用 schedule 使用 `threads=128`、4 warp 和
`LDTM.x16`，因此选择 `tmem.readback`；tc5a schedule 使用 192 threads、6
CTA warp，但显式声明 4 个 epilogue consumer warp 和 `LDTM.x8`，因此选择
`tmem.readback.x8.warps4`。其余对照点保留为
`tmem.readback.x8.warps1` 与 `tmem.readback.x16.warps1`。模型根据
`tmem_load_registers` 和 `tmem_consumer_warps`（省略时才用 `threads/32`）
机械选择资源，不得因另一合同数值更快而跨合同替换容量。

Tensor Core compute capacity 同样按指令 shape 精确绑定。定义经验层 compute
资源键 `tensor.<format>.m<MM>n<NN>`：`<format>` 是本节 `resource` 中的输入格式
标识，`<MM>` 与 `<NN>` 分别是该 `tcgen05.mma` 合同一次发出的 M、N 维度，单位
element。比如 `tensor.bf16.m128n64` 的实测率只能服务 `mma_m=128,mma_n=64`
的 schedule；它不能替代 `tensor.bf16.m128n128` 或
`tensor.bf16.m128n256`。严格层的产品级 rate upper 仍使用通用
`tensor.<format>` 键，因为其条件声明覆盖整个对应格式，而不是某一个实测 shape。

## 5. 第一层：条件可证明性能上界

定义资源集合 \(\mathcal R\) 为模型采用的硬件资源约束集合。定义
\(r\in\mathcal R\) 为其中一个资源的索引，无单位；定义 \(Q_r^{\mathrm{LB}}\)
为任何合法实现至少需要在资源
\(r\) 上完成的工作，单位可能是 FLOP、B、instruction 或 transaction；定义
\(U_r\) 为资源 \(r\) 的服务率上界，单位与 \(Q_r^{\mathrm{LB}}\) 每秒对应。

定义 \(T_r^{\mathrm{LB}}\) 为资源 \(r\) 单独给出的执行时间下界，单位为 s。
只有当证据能支持“真实服务率不大于 \(U_r\)”时，才有：

\[
T_r^{\mathrm{LB}}=\frac{Q_r^{\mathrm{LB}}}{U_r}.
\]

定义 \(T_{\mathrm{resource}}^{\mathrm{LB}}\) 为全部独立资源工作下界：

\[
T_{\mathrm{resource}}^{\mathrm{LB}}
=\max_{r\in\mathcal R}T_r^{\mathrm{LB}}.
\]

若同一资源有多个同时成立的服务率上界，严格层取其中最小的 \(U_r\)，即这些
上界约束的交集；取最大的上界虽然仍安全，却不是当前证据能给出的最紧约束。
资源时间取最大值代表允许资源完美重叠，是一个乐观时间下界。若两个资源共享端口或不能
同时达到各自峰值，必须增加联合容量约束，而不是把两个时间任意相加。

`cold_hbm` 只描述输入初始驻留条件，不表示 DRAM traffic 绕过 L2。当前 v1 的
TMA/global-store 路径仍必须经过整 GPU 共享 L2 fabric。因此 cold-HBM 严格层同时
保留 `hbm.total`、`l2.read` 和 `l2.write` 最低工作约束；其中 L2 read/write
分别使用 1024/512 B/cycle/GPU 的条件容量，绝不乘 `sm_count`。hot-L2 不使用
`hbm.total`，但继续使用两条共享 L2 条件上界。当前没有足够外边界证据证明
read/write 满足归一化联合约束
\(R/1024+W/512\le1\)，所以 v1 分别约束两个方向并允许理想重叠。

### 5.1 证据等级

每个容量参数都带 `evidence_kind`：

| 等级 | 含义 | 能否进入条件上界 |
| --- | --- | --- |
| `specified_upper` | 官方规格或明确架构合同给出的服务率上限 | 可以，需记录条件 |
| `derived_upper` | 从可复核的 issue、频率或端口约束推导 | 可以，需保留推导 |
| `profiler_model_peak` | NCU `%peak` 等工具内部模型峰值 | 只能形成带工具假设的条件上界 |
| `measured_sustained` | 独立 microbenchmark 实测持续值 | 不可以，只进入经验层 |
| `measured_joint` | 联合 microbenchmark 的实测工作点 | 不可以作为容量外边界 |
| `observed_gemm` | 完整 GEMM 实测 | 只用于已观测层和反证 |
| `derived_work` | 由 workload/schedule 推导的工作量 | 可用于对应层 |
| `unknown` | 当前无有效证据 | 不进入数值约束 |

选择 `unknown` 或省略约束会让条件上界变松，但仍保持逻辑方向正确。用一个测得的
sustained 值冒充 \(U_r\) 会产生虚假的低上界，可能被真实 GEMM 轻易超过。

### 5.2 有限并行与尾部

定义 \(n_t\) 为不可再分割的任务数，单位为 task；定义 \(i\) 为任务索引，
无单位；定义 \(p_i\) 为第 \(i\) 个任务的最小服务时间，单位为 s；定义 \(U_t\)
为能同时服务这类任务的等价硬件
单元数，单位为 service unit；定义 \(T_{\mathrm{parallel}}\) 为该任务集合的实际
makespan，单位为 s。有限并行调度满足：

\[
T_{\mathrm{parallel}}
\ge
\max\left(
\frac{\sum_{i=1}^{n_t}p_i}{U_t},
\max_i p_i
\right).
\]

定义 \(T_{\mathrm{parallel}}^{\mathrm{LB}}\) 为上式右侧，即当前任务分解能够证明
的有限并行时间下界。

若全部任务同构，定义 \(p\) 为每个任务共同的最小服务时间，单位为 s；定义
\(T_{\mathrm{parallel,identical}}\) 为这一同构、理想调度条件下的 makespan，
单位为 s，则：

\[
T_{\mathrm{parallel,identical}}
=\left\lceil\frac{n_t}{U_t}\right\rceil p.
\]

本文不统一乘一个“最后一波效率”。persistent、stream-K、多 resident CTA 和
专用 tail kernel 都会改变任务分解；将一个 wave 系数同时乘到 compute、TMA、
HBM 和 critical path 上容易重复计算尾部。

当前可执行 v1 只在经验层使用上述 per-group span/makespan。把全 GPU 规格峰值
平均除以 SM 数并不能单独证明单 CTA 的最大服务率，所以严格层在没有 per-SM 或
per-CTA 证据时不加这项约束；它保留全 GPU 总工作/总容量约束。代价是小 shape 的
严格上界更松，收益是不会用未经证明的“均匀切分”制造虚假上界。

### 5.3 因果关键路径

定义执行依赖图 \(G=(V,E)\)，其中 \(V\) 是 load、MMA、wait、TMEM readback、
epilogue 等阶段，\(E\) 是生产者到消费者的真实依赖。定义
\(T_{\mathrm{span}}^{\mathrm{LB}}\) 为图中任一合法执行都无法缩短的最长依赖链，
单位为 s。

例如，下一块 TMA load 和当前 MMA 可以在双缓冲稳态重叠；但 accumulator 完成
前不能读取最终 TMEM 结果。具体 schedule 可以用
`first load + steady max(load, compute) + last compute`，通用上界则用
work/span 形式，避免把 `tc3` 的固定阶段顺序误当成所有 GEMM 的唯一顺序。

### 5.4 联合容量区域

定义资源吞吐向量 \(\mathbf y\) 为同一时刻 TMA、L2、SMEM、TMEM、Tensor Core
等资源的服务率组合；定义矩阵 \(\mathbf H\) 为联合资源线性约束的系数矩阵；
定义向量 \(\mathbf c\) 为每条联合约束的容量上限。这里改用
\(\mathbf H,\mathbf c\)，避免与 GEMM 输入矩阵 \(A,B\) 混淆。若能证明硬件
容量外边界满足：

\[
\mathbf H\mathbf y\le\mathbf c,
\]

则 \(\mathbf H\) 和 \(\mathbf c\) 的每一行都能生成一条联合时间下界。定义
\(T_{\mathrm{joint}}^{\mathrm{LB}}\) 为所有已证明联合容量约束给出的最大时间下界，
单位为 s；没有有效外边界时该项为 0。

联合 microbenchmark 只提供一个可实现的 \(\mathbf y\) 点，即容量区域内点。它
能校准经验模型，却不能单独证明外边界。仓库现有 `tcgen05.cp`/MMA overlap
结果因此标为 `measured_joint`，不用于无条件上界。

### 5.5 总时间下界

定义 \(T_{\mathrm{fixed}}^{\mathrm{LB}}\) 为已经证明不可消除的 launch、分配或
同步固定时间下界，单位为 s。实测 launch latency 默认不是“不可更短”的证明；
没有可信下界时严格层令这一项为 0。

定义 \(T_{\mathrm{ub}}^{\mathrm{LB}}\) 为所有当前可证明时间约束的联合下界，
单位为 s：

\[
T_{\mathrm{ub}}^{\mathrm{LB}}
=
\max\left(
T_{\mathrm{resource}}^{\mathrm{LB}},
T_{\mathrm{parallel}}^{\mathrm{LB}},
T_{\mathrm{span}}^{\mathrm{LB}},
T_{\mathrm{joint}}^{\mathrm{LB}},
T_{\mathrm{fixed}}^{\mathrm{LB}}
\right).
\]

条件性能上界为：

\[
P_{\mathrm{ub}}=\frac{W_{\mathrm{use}}}{T_{\mathrm{ub}}^{\mathrm{LB}}}.
\]

如果没有任何有效约束而使 \(T_{\mathrm{ub}}^{\mathrm{LB}}=0\)，模型不得执行除法
或输出有限数值；它必须返回 `insufficient_evidence`。若只有部分资源具有合法
上界，则可以输出状态为 `partial` 的较松条件上界，同时列出缺失约束。

## 6. 第二层：microbenchmark 驱动的经验理想包络

定义 \(\widehat C_r\) 为资源 \(r\) 在匹配精度、shape、CTA group、频率、cache
状态和 occupancy 条件下测得的经验服务率，单位与资源工作每秒对应。

沿用第 4.2 节定义的 workload 描述 \(w\) 和 schedule 描述 \(x\)。

对一个合法 schedule，定义 \(Q_r(x,w)\) 为 schedule \(x\) 执行 workload \(w\)
时向资源 \(r\) 发出的工作量，单位与 \(\widehat C_r\) 的分子一致。定义
\(\widehat T_{\mathrm{resource}}(x,w)\) 为经验资源时间，单位为 s：

\[
\widehat T_{\mathrm{resource}}(x,w)
=
\max_r\frac{Q_r(x,w)}{\widehat C_r},
\]

定义 \(\widehat T_{\mathrm{parallel}}(x,w)\)、
\(\widehat T_{\mathrm{span}}(x,w)\)、
\(\widehat T_{\mathrm{joint}}(x,w)\) 和
\(\widehat T_{\mathrm{fixed}}(x,w)\) 分别为有限并行、依赖链、联合资源和固定成本
给出的经验时间约束，单位均为 s。定义 \(\widehat T(x,w)\) 为 schedule \(x\)
执行 workload \(w\) 的经验理想时间，单位为 s：

\[
\widehat T(x,w)=
\max\left(
\widehat T_{\mathrm{resource}},
\widehat T_{\mathrm{parallel}},
\widehat T_{\mathrm{span}},
\widehat T_{\mathrm{joint}},
\widehat T_{\mathrm{fixed}}
\right).
\]

这里的最大值明确假设不同约束可以完美重叠；只有依赖图证明两个阶段必须串行时，
它们的时间才应先沿同一 critical path 相加，再作为
\(\widehat T_{\mathrm{span}}\) 进入最大值。当前可执行模型已实现逐资源时间、由
compute rate 推出的单任务 span/有限 wave makespan、由 per-SM TMA ingress
推出的最慢 SM wave makespan、`fixed_seconds` 独立约束，以及精确绑定 profile
的 persistent-worker DAG。定义 \(\widehat T_{\mathrm{DAG}}(x,w)\) 为该
persistent-worker 因果完成时间，单位 s；当前 schedule 的最终经验时间实际取：

\[
\widehat T(x,w)=\max\left(
\widehat T_{\mathrm{resource}}(x,w),
\widehat T_{\mathrm{DAG}}(x,w)
\right).
\]

若 profile 缺失、被 quarantine 或 workload 超出 K-tile/output-task 留出范围，
模型返回 `insufficient_evidence`，不会用 resource 数值顶替 DAG。当前仍未实现可跨
任意 schedule 复用的通用 DAG 或联合容量外边界，因此每个非 tc5a candidate 仍需
自己的因果合同。

经验测得的 sustained rate 不是物理 rate upper。对同一 schedule，经验层还会
把所有适用的 `specified_upper`、`derived_upper` 和 `profiler_model_peak` 作为
`hard_upper:*` 时间约束取交集。例如 cold-HBM 同时使用方向独立的
`hbm.read`/`hbm.write` 实测值和共享 `hbm.total` 上界：

\[
\widehat T_{\mathrm{HBM}}
=\max\left(
\frac{Q_{\mathrm{read}}}{\widehat C_{\mathrm{read}}},
\frac{Q_{\mathrm{write}}}{\widehat C_{\mathrm{write}}},
\frac{Q_{\mathrm{read}}+Q_{\mathrm{write}}}{U_{\mathrm{HBM,total}}}
\right).
\]

因此读写 probe 即使分别很快，也不能合成超过共享 LPDDR5X 总带宽的经验包络。

定义 \(\mathcal X_{\mathrm{manifest}}\) 为通过当前 v1 已实现的 descriptor、
MMA shape、单 CTA SMEM/TMEM、thread 和显式精度白名单检查的 schedule 集合。
register-derived occupancy、CTA-group 2、split/stream-K、**未绑定因果 profile 的**
persistent schedule 和专用 tail kernel 尚未形成完整可执行合法性证明；相应
schedule 不得被称为已覆盖。当前只有 `tc5a_m128n256k64_stage4` 显式声明
`persistent=true` 与唯一 `causal_pipeline_resource`；这不推广到其他 schedule。
当前示例 manifest 把数据通路拆成三类：普通 FP16/BF16/TF32/FP8/INT8 使用
`logical_packed`；raw E3M2/E2M3/E2M1 direct-SMEM 使用 `byte_padded`，与 closure
compute campaign 的 8-bit descriptor container 一致；MXFP4/NVFP4 使用紧凑
4-bit value、独立 scale bytes 和 512-column accumulator+SFA+SFB TMEM 合同。
因此不会把 6-bit 的逻辑 0.75 B/element 直接冒充可执行的 direct-SMEM TMA
payload，也不会把普通 accumulator 的 TMEM allocation 套到 block-scaled MMA。
定义 \(\widehat T_{\mathrm{env}}(w)\) 为 manifest 内合法 schedule 经验理想时间的
最小值，单位为 s；经验理想包络为：

\[
\widehat T_{\mathrm{env}}(w)
=
\min_{x\in\mathcal X_{\mathrm{manifest}}}\widehat T(x,w),
\qquad
\widehat P_{\mathrm{env}}(w)
=
\frac{W_{\mathrm{use}}}{\widehat T_{\mathrm{env}}(w)}.
\]

这只是已枚举 schedule 中的理想预测。若 manifest 没有包含一种新算法，搜索
结果不能称为所有 GEMM 的绝对上界。经验层缺少任一必需 resource capacity 时
直接返回 `insufficient_evidence`，不输出用剩余资源拼出的“半个包络”；严格层则
可以保留单条已证明约束形成的合法但较松上界，并把状态标为 `partial`。

## 7. 第三层：完整 GEMM 已观测最好值

定义一个 eligible backend series 为同一 workload 上至少 10 个 trial、全部
`Matched=1`、没有 missing/timeout/launch failure 且性能率为正的完整 GEMM
结果序列；浮点性能率单位为 FLOP/s，整数性能率单位为 OP/s。

定义 \(b\) 为 backend 的索引，定义 \(j\) 为同一 backend series 内 trial 的索引，
定义 \(P_{b,j}\) 为第 \(j\) 次 trial 的性能，单位与 workload 相同。为降低单个
噪声尖峰的影响，当前工具先计算每个 backend series 的 median，
再选择 median 最大的 backend 作为稳定最好实现；同时保留该 series 的 minimum
和 maximum。定义 \(P_{\mathrm{obs,median}}\) 为所有 eligible backend series
中最大的 trial median，单位与 workload 相同。选择规则为：

\[
P_{\mathrm{obs,median}}
=
\max_b\operatorname{median}_{j\in\mathrm{valid\ trials}}P_{b,j}.
\]

最大单 trial 值只用于检查上界违规，不作为稳定性能中心值。closure importer 从
`trials.jsonl` 分别重算候选和同精度 reference 的 minimum、median 与 maximum；
`P_{\mathrm{obs,median}}` 在二者中选择较大 median，条件上界反证则使用二者中更大
的 maximum trial。旧报告只检查候选而把 cuBLAS 留作 denominator，会漏掉
reference 对上界或经验包络的反证；当前 auditor 明确拒绝这种不完整语义。

“Reference”字段当前主要表示性能 denominator，不必然是 correctness reference。
例如 NVFP4/MXFP4 CUTLASS 路径可以通过自己的 host correctness reference，但
CSV 的性能 denominator 仍是 FP16 cuBLAS。工具将这种情况标记为
`cross_precision_denominator`，禁止把 ratio 解读为同精度库胜负。

## 8. 最终 closure 证据状态

`coverage` 对应字段定义为可执行模型生成的覆盖率与缺口对象，而不是人工状态摘要。
`all_common_resources_closed` 定义为当前版本所列全部公共经验资源均有同硬件作用域的 closure-qualified 容量；资源清单变化时，旧报告中的同名布尔值不能跨 schema 直接沿用。

下面状态来自结果提交
`ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c` 中的可执行 `coverage`、三批独立
auditor 和 `report-closure`，不是人工印象。该结果把已完成基础 suite
`thor-t5000-closure-maxn-20260814-d382b57-a` 的 compute/full-GEMM 与新
supplement `thor-t5000-tma-ingress-supplement-maxn-20260814-c` 的 component
证据分别绑定到原始提交，不把跨提交运行伪装成一次采集。

Thor T5000 的条件规格锚点来自 NVIDIA 的官方产品表：MAXN 下 dense FP4 为
1035 TFLOP/s、dense FP8 为 517 TFLOP/s、sparse INT8 为 1035 TOPS、sparse
FP16 为 517 TFLOP/s，内存带宽 273 GB/s、最大 GPU 频率 1.57 GHz。本文只把表中明确的 dense 数字当
`specified_upper`；但产品表没有把 FP8 拆成 E4M3/E5M2，也没有指明 FP4 的具体
encoding，因此映射到某个 PTX 精度合同仍是显式条件。BF16/FP16 的 258.5
TFLOP/s 与 S8/U8 的 517.5 TOPS 都是按 2:1 稀疏倍率推得的
`derived_upper`，不是官方直接列出的 dense 项；INT8 表也没有区分 signed/unsigned。规格来源见
[NVIDIA Jetson Thor 官方介绍](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)。

`numeric_closure` 定义为该精度具有硬件匹配、closure-qualified 且独立同合同 correctness 的完整 GEMM 数值证据；它不等于 domain 上界、场景对齐或因果流水线闭环。

| 精度 | compute 条件上界 | closure compute 实测 | 完整 GEMM | 同精度性能 denominator | `numeric_closure` |
| --- | --- | --- | --- | --- | --- |
| FP16 | 有（推导） | 有 | 有 | 有 | 是 |
| BF16 | 有（推导） | 有 | 有 | 有 | 是 |
| TF32 | **缺** | 有 | 有 | 有 | 否 |
| FP8 E4M3 | 有（条件映射） | 有 | 有 | 有 | 是 |
| FP8 E5M2 | 有（条件映射） | 有 | 缺 | 缺 | 否 |
| FP6 E3M2 | **缺** | 有 | 缺 | 缺 | 否 |
| FP6 E2M3 | **缺** | 有 | 缺 | 缺 | 否 |
| raw FP4 E2M1 | **缺** | 有 | 缺 | 缺 | 否 |
| MXFP4 | **缺** | 有 | 缺 | 缺 | 否 |
| NVFP4 | 有（条件映射） | 有 | 缺 | 缺 | 否 |
| signed INT8 | 有（推导） | 有 | 有 | 有 | 是 |
| unsigned INT8 | 有（推导） | 有 | 缺 | 缺 | 否 |

这里的 `numeric_closure` 同时要求独立 compute 条件上界、closure-qualified
compute rate、完整 GEMM observation 和同精度 denominator。12 种 compute
合同都已取得三个 M/N shape、每项 10 trial 的 closure-qualified 实测；但只有
FP16、BF16、E4M3 和 S8 同时满足其余条件。TF32 已完成 campaign 测量，仍因缺少
独立 strict compute upper 而保持 `false`。这一区分禁止把 measured compute rate
冒充任何实现都不能突破的物理 rate upper。

历史 NVFP4 的 1032.111 TFLOP/s 被明确隔离：旧生成器把 PTX ISA Table 42 中
raw E2M1 的 type code `5` 写进了 Table 44 的 `mxf4nvf4` descriptor；Table 44
对 block-scaled E2M1 的编码是 `1`。`ptxas` 接受并生成 `UTCOMMA.4X` 只能证明
静态 lowering，不能证明该数值对应声明的 NVFP4 语义。新 campaign 使用独立的
Table 42/Table 44 encoder 和反例测试，在重跑前模型把旧数字标为 `unknown`、
`quarantined`，不让它进入经验包络。

当前公共资源已有 closure-qualified 经验数据：

- HBM/LPDDR streaming read 和 write；
- L2 unique read 和 end-to-end store path；
- TMA L2-hit 和 DRAM-stream ingress。

旧 `max(clock64 per CTA)` 数字仍只保留为 `snapshot_only` 对照。新的 closure
campaign 对 L2-hit ingress 使用单 CTA 的 device `%globaltimer`，对 aggregate
路径使用整网格最早 start 到最晚 stop，并同时冻结 `32 KiB × inflight=1`、
`32 KiB × inflight=4` 和四 stage 的精确
`A=16 KiB + B=32 KiB, inflight=8`。新结果已经通过 10-trial、源码、binary、
SASS、运行环境、SM coverage 和独立 auditor 门禁，获得 `closure_qualified`。
其中串行 32 KiB case 只进入带 `diagnostic` 的资源 ID；uniform inflight=4
提供 `.inflight4` capacity，精确 tc5a A/B 混合 case 提供 legacy
`tma.smem_ingress.per_sm` 与 `tma.hbm`，其采集时共同 A/B row stride 为 2048。
保存这些 capacity 不表示任意两级/四级 schedule 都能使用它们。新 schedule 通过
`tma_contract_family_by_precision` 和 `tma_contract_row_stride_elements` 生成包含
family 与 stride 的精确 resource ID；历史 tc5a 点只有一条到 stride2048 的单向
兼容别名，不覆盖 1024、4096 或其他 family。精度、payload/request/stage/thread/
cache/SM-coverage/leading dimension 任一不匹配时 empirical memory layer 返回
`insufficient_evidence`，不再按 `stages` 猜测。
HBM/L2 四个旧快照也采用相同处理：新 unified component campaign 的
`hbm.read`、`hbm.write`、`l2.read`、`l2.write` 整卡 `%globaltimer` case 已回传
并通过独立审计；同资源经验层优先选择这些 closure-qualified 点，不再混入较弱
快照。

定义本节所有 GB/s 为十进制 \(10^9\) B/s。最终 component 中位数为：

| 资源 | 精确 case 合同 | 中位数 |
| --- | --- | ---: |
| `tma.smem_ingress.diagnostic.serial32k.per_sm` | 单 CTA，32 KiB，inflight=1 | 68.615 GB/s |
| `tma.smem_ingress.per_sm.inflight4` | 单 CTA，32 KiB，inflight=4 | 129.398 GB/s |
| `tma.smem_ingress.per_sm` | 单 CTA，tc5a A16 KiB+B32 KiB，四 stage/八请求 | 193.366 GB/s |
| `tma.hbm.diagnostic.serial32k` | 20 SM，32 KiB，inflight=1 | 261.556 GB/s |
| `tma.hbm.inflight4` | 20 SM，32 KiB，inflight=4 | 259.193 GB/s |
| `tma.hbm` | 20 SM，tc5a A16 KiB+B32 KiB，四 stage/八请求 | 185.509 GB/s |
| `hbm.read` | 256 MiB stream，全 GPU | 253.588 GB/s |
| `hbm.write` | 256 MiB stream，全 GPU | 201.158 GB/s |
| `l2.read` | 16 MiB hot working set，全 GPU | 1505.112 GB/s |
| `l2.write` | 16 MiB hot working set，全 GPU | 545.416 GB/s |

对 FP16 \(N=2048\) 的完整 GEMM，定义候选/参考比为候选中位性能除以同精度
cuBLAS 中位性能。tc5a 实测为 120.039 TFLOP/s，cuBLAS 为
130.633 TFLOP/s，候选/参考比为 91.89%。hot-L2 与 cold-HBM 两场景的经验理想
包络都为 128.436 TFLOP/s，候选实测达到 93.46%；本 shape 的稳定已观测最好
backend 是 cuBLAS，其 median/经验包络为 101.71%，仍处于预声明的 2% 经验
重校准容差内。cuBLAS 最大 trial 为 131.163 TFLOP/s，未超过任一适用条件上界。
精确 `tc5a_m128n256k64_stage4` 由共享 `l2.read` 而非 per-SM TMA ingress
限制；tc5a 的
`tma.per_sm_parallel_makespan` 为 56.939 us，小于 `l2.read` 的 133.762 us。
这正是新 microbenchmark 验证的边界：在该精确 schedule/shape 下共享 L2 总线
先成为经验瓶颈。generic schedule 当前没有精确 ingress/HBM capacity 绑定，不能
再与 tc5a 并列或借用其 193.366/185.509 GB/s 合同。

完整 15 个 observation、两种 residency 的逐资源时间、条件上界和经验包络位于
结果提交的
[`closure_analysis.json`](https://github.com/hibouwu/CUDA_optimazation/blob/ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c/results/sm110_model_closure/thor-t5000-tma-ingress-supplement-maxn-20260814-c/closure_analysis.json)
与
[`closure_summary.md`](https://github.com/hibouwu/CUDA_optimazation/blob/ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c/results/sm110_model_closure/thor-t5000-tma-ingress-supplement-maxn-20260814-c/closure_summary.md)。
结果提交中的历史报告为 `pass=true`；它使用当时的 stage-only TMA 选择规则。用当前
显式绑定模型和同一批 freshly re-imported raw evidence 重放后为 `pass=false`：
没有条件上界矛盾，但 15 个 observation 中只有 FP16/BF16 的 N=2048 两项具有
row-stride 精确匹配的 resource prediction，其余 13 项缺少对应场景。当前规则下的完整重放表见
[`thor_sm110_current_model_replay.md`](./thor_sm110_current_model_replay.md)。
另有一条基础 suite 区间
`oc3_event_cnt +179` warning；新 component supplement 区间三个 OC counter 增量
均为 0。这个差异来自模型门禁收紧，不修改历史 GPU 数据，也不把 auditor 已通过的
采集证据判为无效。

当前 campaign 声明的公共 component case 已全部闭合，但“某个 schedule 是否具有
精确匹配 capacity”是更高一层的逐 schedule 门禁，不能由公共资源布尔值替代。
若只要求较松的 conditional upper，下列项目不是方向正确性的前置条件；若要求本项目
定义的完整 empirical envelope 和 causal end-to-end closure，则前两项与全精度
合同属于必需缺口：

- tc5a 以及其余实际 candidate 的 launch/TMEM alloc/barrier startup/drain
  closure-qualified latency/interval profile；
- 与实际 candidate 精确匹配的 TMA ingress/HBM capacity，以及
  TMA+MMA、MMA+readback+store 联合实验；
- 非 NVFP4 输出语义各自的正式 epilogue capacity；
- 全部 12 种精度的完整 GEMM、correctness reference 和同精度 denominator。

因此历史 closure report 保留
`all_precisions_closed=false`、`all_common_resources_closed=true` 和
`campaign_measurement_coverage.all_campaign_measurements_closed=true`；当前更完整
的精度矩阵另报 `resource_envelope_closed_count=0`、
`causal_pipeline_closed_count=0`、`end_to_end_closed_count=0`。前一组描述历史
numeric/campaign 范围，后一组描述当前三层模型完备性，字段不得互相替代。

> **2026-08-15 对抗式复审边界**：上面的
> `all_common_resources_closed=true` 只对当时定义的十个**独立**资源 ID 成立，
> 不能解释为所有联合性能参数均已测量。复审发现 HBM/L2 同核 read/write duplex
> surface 与 TMA payload/residency surface 尚未进入该布尔值。新增 runner、精确
> 缺口矩阵和 Thor 重跑合同见
> [`sm110_gemm_runner_adversarial_audit.md`](sm110_gemm_runner_adversarial_audit.md)。
> parameter supplement 已回传并按 12.9 节导入；它关闭 L2 duplex 与 cold proxy
> surface，但物理 HBM duplex 和 exact joint-pipeline 仍未闭合，所以既有数值仍不得
> 升级为联合可达性已证明的包络。

## 9. 自动化接口和反证规则

可执行模型位于
[`scripts/sm110_gemm_model`](../../scripts/sm110_gemm_model/README.md)。

证据审计：

```bash
python3 -m scripts.sm110_gemm_model.cli audit \
  --capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --pipeline-profiles scripts/sm110_gemm_model/profiles/pipeline_profiles.json \
  --repo-root .
```

统一 closure 结果不能手工抄写成模型参数。完成的 evidence tree 必须经过
[`closure_import.py`](../../scripts/sm110_gemm_model/closure_import.py) 再次调用三批
独立 auditor，并联合检查 epilogue preflight、固定 commit、MAXN/锁频、suite
完成标志和运行前后 OC counter。导入后才生成 `model_inputs.json`。OC counter
增加作为 MAXN 运行条件 warning，不单独否定数据；若 counter 倒退、artifact
缺失、hash/NCU/SASS/数值检查失败或 commit 不一致，导入失败。完整的随提交运行指令见
[`THOR_CLOSURE_RUNBOOK.md`](THOR_CLOSURE_RUNBOOK.md)。

精度和公共资源覆盖：

```bash
python3 -m scripts.sm110_gemm_model.cli coverage \
  --repo-root . \
  --capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --hardware scripts/sm110_gemm_model/profiles/thor_sm110.json \
  --workloads scripts/sm110_gemm_model/examples/workloads.json \
  --schedules scripts/sm110_gemm_model/examples/schedules.json \
  --pipeline-profiles scripts/sm110_gemm_model/profiles/pipeline_profiles.json \
  --observed-input results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv \
  --observed-input results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv
```

完整目标还必须把上述 numeric coverage 与 full-GEMM implementation、同输入精度且
同输出类型的数值参考、同精度 performance denominator 合并审计。当前端到端定义
还要求 N=1024/2048/4096 × hot-L2/cold-HBM 六个场景都只选择精确合同且
closure-qualified 的 resource capacity，并要求 causal pipeline DAG 完成。定义
`all_precisions_end_to_end_closed` 为 12 个精度逐项同时通过这些门禁的布尔值；
当前值必须保持 `false`，当前五级计数是 `(6,4,0,0,0)`。生成 JSON/Markdown 证据
矩阵，并在任何精度未闭环时使最终门禁非零退出：

```bash
MODEL_DIR="results/sm110_model_closure/$SUITE_ID"
PIPELINE_PROFILES="${PIPELINE_PROFILES:-scripts/sm110_gemm_model/profiles/pipeline_profiles.json}"
RESOURCE_INPUT="results/sm110_model_closure/$RESOURCE_SUITE_ID/resource_capacities.json"
python3 -m scripts.sm110_gemm_model.cli report-precision-closure \
  --repo-root . \
  --capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --closure-import "$MODEL_DIR/model_inputs.json" \
  --resource-import "$RESOURCE_INPUT" \
  --hardware scripts/sm110_gemm_model/profiles/thor_sm110.json \
  --schedules scripts/sm110_gemm_model/examples/schedules.json \
  --pipeline-profiles "$PIPELINE_PROFILES" \
  --support-manifest microbench/sm110_full_gemm_campaign/support_manifest.json \
  --output-json Docs/blackwell_tensorcore/thor_sm110_all_precision_evidence_matrix.json \
  --output-markdown Docs/blackwell_tensorcore/thor_sm110_all_precision_evidence_matrix.md \
  --require-all-closed
```

不带 `--require-all-closed` 时命令仍生成诚实的中间矩阵，供设计下一轮补测；带该
选项才是目标完成门禁。结构正确但仍列出 blocker 的 support manifest 不能单独
证明任何缺失精度已经闭环。

统一 closure 完成后，数值表、两种 residency 场景、最大 trial 上界反证和三个互不
混淆的完成状态由报告器直接生成：

```bash
MODEL_DIR="results/sm110_model_closure/$SUITE_ID"
PIPELINE_PROFILES="${PIPELINE_PROFILES:-scripts/sm110_gemm_model/profiles/pipeline_profiles.json}"
RESOURCE_INPUT="results/sm110_model_closure/$RESOURCE_SUITE_ID/resource_capacities.json"
python3 -m scripts.sm110_gemm_model.cli report-closure \
  --closure-import "$MODEL_DIR/model_inputs.json" \
  --repo-root . \
  --capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --resource-import "$RESOURCE_INPUT" \
  --hardware scripts/sm110_gemm_model/profiles/thor_sm110.json \
  --schedules scripts/sm110_gemm_model/examples/schedules.json \
  --pipeline-profiles "$PIPELINE_PROFILES" \
  --output-json "$MODEL_DIR/closure_analysis.json" \
  --output-markdown "$MODEL_DIR/closure_summary.md"
```

Thor causal 结果必须先经独立 auditor 导入；不能手工复制 fit 数值：

```bash
python3 -m scripts.sm110_gemm_model.cli import-causal-profile \
  --repo-root . \
  --run-id "$CAUSAL_RUN_ID" \
  --expected-commit "$EXPECTED_COMMIT" \
  --output "$MODEL_DIR/pipeline_profiles.json"
```

Thor resource 结果同样必须重新审计导入，不能手工复制 54 个 rate：

```bash
python3 -m scripts.sm110_gemm_model.cli import-resource-capacities \
  --repo-root . \
  --suite-id "$RESOURCE_SUITE_ID" \
  --expected-commit "$EXPECTED_COMMIT" \
  --output "$RESOURCE_INPUT"
```

模型测试：

```bash
python3 -m unittest -v scripts.sm110_gemm_model.test_model
```

自动审计至少执行以下规则：

- `measured_sustained` 和 `measured_joint` 不能进入条件上界；
- NCU model peak 没有适用条件时审计失败；
- 参数无 source path、locator 或单位时审计失败；
- source path 不存在时审计失败；
- \(\beta=0\) 时最小工作量不得读取 C；
- padding 的 issued compute work 不得小于 useful compute work；
- block scale 不得跨 K 向量边界，scale bytes、accumulator bytes 和 output bytes
  必须分开；
- cold-entry 去重 DRAM bytes、重复 L2/TMA request bytes 和 TMEM readback bytes
  必须处在各自资源边界；
- raw FP6/raw E2M1 direct-SMEM 必须按 b8 container 搬运，block-scaled schedule
  必须满足 scale TMEM allocation 合同；
- 增大一个有效 rate upper 不能降低性能上界；
- 候选与同精度 reference 的较大 median 用于完整 GEMM 稳定已观测最好值；二者
  任一最大合法 trial 超过同语义条件上界时审计失败；
- 同一 residency 下经验理想包络超过条件性能上界时同样审计失败；这表示 capacity
  语义、工作量计数或上界适用条件至少有一项互相矛盾，不能靠 clamp 掩盖；
- 完整 GEMM 超过经验包络时只触发重校准，不写成物理违规；
- 跨精度 denominator 不能用于同精度效率结论；
- 浮点路径用 FLOP/s，S8/U8 路径用 OP/s，二者禁止隐式互换；
- `closure_qualified` 必须有至少 10 次 trial 和可定位的原始 artifact；
- evidence path 必须保持在仓库根目录内，不能借绝对路径、`..` 或 symlink 逃逸；
- `unknown` 证据必须显式 `quarantined`，且不能参与任一数值层；
- residency 或 timed scope 不相同的观测不得直接作上界违规判定；
- `numeric_closure` 要求 strict compute upper、closure-qualified compute、完整
  GEMM 和同精度 denominator 全部存在；campaign 测量闭环不把 measured compute
  冒充 strict upper，而使用独立的 `campaign_measurement_coverage` 字段。

## 10. 与 `tc3` 阶段模型的关系

[`thor_sm110_gemm_stage_model.md`](./thor_sm110_gemm_stage_model.md) 对固定 FP16
`tc3` kernel 的 load/compute/epilogue 顺序做了代码对应分析，它仍然是有价值的
schedule-specific case study。

该文档的 load stage 已同步采用“共享 L2 总线 + 每 SM 独立 TMA→SMEM 出口”
两条约束；若两份文档在容量资格或通用公式上出现差异，以本文的 fail-closed
closure 合同为准。

本文不是把 `tc3` 的参数换成变量名，而是改变建模层级：

- `tc3` 模型问“这个固定 kernel 为什么花这些时间”；
- 经验包络问“当前已知合法 schedule 中哪个最理想”；
- 条件上界问“在声明的硬件容量上界下任何实现都不能超过哪里”。

固定 kernel 的 TMA/MMA 双缓冲公式可以成为 \(\widehat T(x,w)\) 的一个 schedule
实例，但不能直接代表所有 GEMM。

## 11. 硬件闭环的三个不同完成状态

不能用一个布尔值同时表示模型正确、一次 campaign 完成和所有产品精度都有完整
证据。本文分别报告：

1. **campaign 测量闭环**：预声明的 FP16、BF16、TF32、E4M3、S8 五种
   full-GEMM 合同均有与候选 schedule 的 MMA M/N shape 精确匹配的
   closure-qualified compute rate、10-trial 全矩阵
   correctness、同精度 denominator、1024/2048 calibration 和 4096 holdout；
   TMA L2/HBM、HBM/L2 read/write、block-scale TMEM ingress、TMEM readback 与
   NVFP4 epilogue component case 也全部完成。
2. **严格上界证据完备**：每个声明精度都具有适用的 compute rate upper，公共
   资源具有外边界证据。缺一项时仍可由其余约束给出方向正确但较松的 `partial`
   上界，不能把 measured rate 补成 rate upper。
3. **全部 12 精度产品覆盖**：除 compute-only 外，还要求每种精度有独立完整
   GEMM、correctness reference 和同语义性能 denominator。当前只有上述五种进入
   本轮 full-GEMM campaign；其中 TF32 仍缺 strict compute upper，所以当前只有
   FP16、BF16、E4M3、S8 四种满足 numeric closure。其余项目和每个缺失 shape
   必须继续显示为 coverage gap。
4. **逐 schedule 资源包络闭环**：每个 shape/residency 的最优 schedule 必须显式
   绑定 precision/payload/request/stage/thread/cache/SM-coverage/row-stride 完全匹配
   的 capacity。历史 tc5a probe 只冻结共同 A/B stride=2048，所以 FP16/BF16
   各自只有 N=2048 的 hot/cold 两个场景可用，没有任何精度的六场景 matrix 完整；
   不能用旧 schema 下的公共资源通过状态代替。
5. **因果流水线闭环**：latency、initiation interval、TMA/MMA/TMEM 依赖、startup
   和 drain 进入可执行 DAG。合同绑定的 persistent-worker 求解器已经实现，但
   当前没有 Thor profile 被导入；tc5a suite 已分别冻结 FP16 和 BF16 的 tensor-map
   类型、MMA instruction descriptor、91-case 矩阵和 singleton profile，二者会独立
   拟合并独立过门禁，不能因为 transport payload 都是 2 B/element 就互相复用。
   其他 candidate 仍需要自己的 profile，或另行给出并审计等价性证明。因此 12 种
   精度仍均未达到完整三层端到端 closure。
   详见全精度证据矩阵。

一次 campaign 的逐精度测量合同依次要求：PTX/descriptor 合法、目标函数块 SASS、
compute-only 10 trial、兼容的公共 data-movement/readback capacity、完整 GEMM
10 trial、独立 correctness reference、同精度 denominator、预声明
calibration/holdout、没有条件上界反证，以及 run spec、环境、源码、binary、SASS、
NCU 和结果 hash 完整。公共 component capacity 可以由兼容精度共享，不要求为每种
精度复制同一带宽实验。

TMA+MMA 或 MMA+readback 的联合 microbenchmark 不是较松 conditional upper
方向正确性的前置门禁，却是把 throughput-resource envelope 升级为本项目所要求的
causal pipeline closure 的必要证据。完整 GEMM holdout 若系统性超过或偏离基础
包络，还要据此重校准联合模型；不能为了形式上的全绿先加入一个没有外边界意义的
measured joint 点。

本地环境不能生成新的 SM110 数值；本轮 Thor 数值已经由上述结果提交回传。未来若
修改 GPU-facing 源码、case 合同、审计器或模型工作量，仍必须使用新的稳定
campaign ID、逐 case `result.json`、run fingerprint、持久日志、PID/status 和
安全 resume；目录存在或任务已启动不算完成。

正式 closure 固定所有 GPU-facing compute/component/full-GEMM trial 的 host
timeout 为 120 s，NCU holdout timeout 为 300 s；超时后按完整进程组执行
`SIGTERM`→5 s→`SIGKILL`→5 s，并记录 `timeout.json`。任何 timeout 或
`termination_failed=true` 都不能进入成功证据。总协调器必须使用 detached
launcher，避免交互终端的 `Ctrl-C` 中断实际 campaign。

## 12. Microbenchmark 与完整 GEMM 来源

本节是本文参数和验证数据的来源附录。路径均相对仓库根目录。

### 12.0 全精度 compute campaign 与最终 composite closure

- runner：
  [`microbench/sm110_gemm_campaign/run_compute_campaign.py`](../../microbench/sm110_gemm_campaign/run_compute_campaign.py)
- detached/resume launcher：
  [`microbench/sm110_gemm_campaign/launch_compute_campaign.sh`](../../microbench/sm110_gemm_campaign/launch_compute_campaign.sh)
- fail-closed 回传审计：
  [`microbench/sm110_gemm_campaign/audit_campaign.py`](../../microbench/sm110_gemm_campaign/audit_campaign.py)
- Git 往返说明：
  [`microbench/sm110_gemm_campaign/README.md`](../../microbench/sm110_gemm_campaign/README.md)
- 最终 composite model inputs：
  [`model_inputs.json`](https://github.com/hibouwu/CUDA_optimazation/blob/ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c/results/sm110_model_closure/thor-t5000-tma-ingress-supplement-maxn-20260814-c/model_inputs.json)
- 最终 artifact SHA-256 清单：
  [`artifact_sha256.txt`](https://github.com/hibouwu/CUDA_optimazation/blob/ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c/results/sm110_model_closure/thor-t5000-tma-ingress-supplement-maxn-20260814-c/artifact_sha256.txt)
- descriptor 一致性实现：
  [`scripts/sm110_gemm_model/tcgen05_descriptors.py`](../../scripts/sm110_gemm_model/tcgen05_descriptors.py)
- descriptor 与 shape 的规范来源：
  [NVIDIA PTX ISA 9.0, tcgen05 instruction descriptor](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-instruction-descriptor)

manifest 固定为 12 个计算合同、3 个 N shape、2 种 launch，共 72 个 case：FP16、
BF16、TF32、E4M3、E5M2、E3M2、E2M3、raw E2M1、MXFP4、NVFP4、S8、U8。
当前 campaign 还把硬件合同固定为 20 SM；full-SM case 回读每个 block 的 `%smid`，
只有 20 个 block 覆盖 20 个不同 SM 时，才允许用 `blocks × warps × iterations`
计算全 GPU issued work。
FP6/raw FP4 的 direct-SMEM campaign 把逻辑值放在 8-bit container 中，matrix
descriptor 沿用该物理布局的 8/4 byte-offset 编码；逻辑 `input_bits` 与物理
`descriptor_storage_bits` 在原始输出中分别记录。它不同于紧凑 packed 数据先经
`tcgen05.cp` 的 `b6x16_p32`/`b4x16_p64` 解压路径，后者必须作为另一类 schedule
和 microbenchmark 单独测量。
每个 case 至少 10 次 trial；compute window 用 PTX `%globaltimer` 的 nanosecond
计时；full-SM aggregate elapsed time 取所有 CTA 的最早 start 到最晚 stop，包含
CTA 启动偏斜，不用单 CTA 最大 duration 冒充整网格时长。另存 CUDA-event 的
host-observed 整 kernel 时间作交叉检查，避免把单点频率采样直接折算成吞吐。若
counter 权限可用，每种
精度另选一个 full-SM M128N256 case 保存 NCU 报告。runner 保存实际源码、idesc 字段、精确编译
命令、SASS、原始 stdout、environment、binary/SASS/source hash 和不可变 run spec。
closure importer 从 72 个执行 case 中只生成 36 个经验 compute capacity：12 种
精度各取 full-SM、4-warp 的 M128N64、M128N128、M128N256 三个点，并把 shape
写进 `capacity_id`、`resource` 与 `condition`。其余 36 个 single-warp case 是
拓扑/启动方式对照证据，不能冒充全 GPU schedule capacity；M128N256 的 NCU
artifact 是每种精度的结构证据，也不把该 shape 的吞吐外推到另外两个 shape。
本地 CUDA 13.0 `sm_110a` 静态门禁已经 72/72 通过；Thor 基础 suite 的 72 个
执行 case、三 shape closure capacity、NCU 结构证据和 full-GEMM observation 也已
通过独立审计。静态门禁与 Thor 运行证据仍是两个不同层次，不能互相替代。

### 12.1 Tensor Core compute-only

- 入口与生成器：
  [`microbench/mma_compute_only/run_thor_tcgen05_report.py`](../../microbench/mma_compute_only/run_thor_tcgen05_report.py)
- 使用说明：
  [`microbench/mma_compute_only/README.md`](../../microbench/mma_compute_only/README.md)
- 原始整理报告：
  [`microbench/mma_compute_only/分析报告.txt`](../../microbench/mma_compute_only/分析报告.txt)
- 结构化结果：
  [`microbench/mma_compute_only/plots/benchmark_results.csv`](../../microbench/mma_compute_only/plots/benchmark_results.csv)
- NCU 结构化结果：
  [`microbench/mma_compute_only/plots/ncu_results.csv`](../../microbench/mma_compute_only/plots/ncu_results.csv)
- SASS/NCU 入口：同目录生成的 `benchmark_src/`、`build/`、
  `run_ncu_reports.sh` 和 `ncu_reports/`。
- 当前只作历史快照的参数：BF16 258.030 TFLOP/s、E4M3 516.059 TFLOP/s；
  对应报告时钟为 1.575 GHz，尚不满足新的 10-trial artifact closure 合同。
- 历史 NVFP4 1032.111 TFLOP/s 因 descriptor type code 错位被隔离，不能进入
  经验包络；只有新的 Table 44 descriptor 重跑通过后才能替换。
- 当前边界：timed window 排除 TMA、copy pipeline、TMEM readback、epilogue 和
  launch；现有 FP4 case 是 `mxf4nvf4.block_scale.block16`，不能代表 raw E2M1
  或 MXFP4。

基本命令：

```bash
cd microbench/mma_compute_only
./build_and_run.sh run --iters 10000
./build_and_run.sh ncu
./build_and_run.sh plot
```

### 12.2 L2

- 源码：
  [`microbench/L2throughtput/demo.cu`](../../microbench/L2throughtput/demo.cu)
- 说明：
  [`microbench/L2throughtput/README.md`](../../microbench/L2throughtput/README.md)
- 原始结果：
  [`microbench/L2throughtput/results/l2_throughput.csv`](../../microbench/L2throughtput/results/l2_throughput.csv)
- SASS：
  [`microbench/L2throughtput/results/sass_summary.txt`](../../microbench/L2throughtput/results/sass_summary.txt)
- NCU：
  [`microbench/L2throughtput/results/ncu/ncu_l2_validation_summary.csv`](../../microbench/L2throughtput/results/ncu/ncu_l2_validation_summary.csv)
- 对抗式审查：
  [`microbench/L2throughtput/results/adversarial_review.md`](../../microbench/L2throughtput/results/adversarial_review.md)
- 已引用实测：unique read 946.701239 B/cycle/GPU；end-to-end store path
  299.372706 B/cycle/GPU。
- 已引用条件峰值：NCU model peak read 1024 B/cycle/GPU、write
  512 B/cycle/GPU。二者不是 measured sustained。

基本命令：

```bash
cd microbench/L2throughtput
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

### 12.3 GMEM/DRAM streaming

- 源码：
  [`microbench/05_gmem_dram_bandwidth/gmem_dram_bandwidth.cu`](../../microbench/05_gmem_dram_bandwidth/gmem_dram_bandwidth.cu)
- 说明：
  [`microbench/05_gmem_dram_bandwidth/README.md`](../../microbench/05_gmem_dram_bandwidth/README.md)
- 原始结果：
  [`microbench/05_gmem_dram_bandwidth/results/gmem_dram_bandwidth.csv`](../../microbench/05_gmem_dram_bandwidth/results/gmem_dram_bandwidth.csv)
- SASS：
  [`microbench/05_gmem_dram_bandwidth/results/sass_summary.txt`](../../microbench/05_gmem_dram_bandwidth/results/sass_summary.txt)
- NCU：
  [`microbench/05_gmem_dram_bandwidth/results/ncu/ncu_gmem_summary.csv`](../../microbench/05_gmem_dram_bandwidth/results/ncu/ncu_gmem_summary.csv)
- 对抗式审查：
  [`microbench/05_gmem_dram_bandwidth/results/adversarial_review.md`](../../microbench/05_gmem_dram_bandwidth/results/adversarial_review.md)
- 已引用实测：read-stream 126.010672 B/cycle/GPU；write-stream
  70.429363 B/cycle/GPU。
- 边界：本机 NCU 缺直接 `dram__bytes*`；read 验证可使用 LTS read-miss-sector
  proxy，LTS write sectors 只能证明进入 write path，不能证明等量 external DRAM
  write bytes。对应 duplex 结果必须保持 proxy qualification。

基本命令：

```bash
cd microbench/05_gmem_dram_bandwidth
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

### 12.4 TMA GMEM→SMEM

- 源码：
  [`microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu`](../../microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu)
- tc5a 生产 mainloop 的 stage、A/B 双请求、barrier 与 192-thread 来源：
  [`GEMMsm110/include/backends/tc5_persistent.cuh`](../../GEMMsm110/include/backends/tc5_persistent.cuh)
- tc5a 2D SW128 tensor-map 编码来源：
  [`GEMMsm110/include/sm110_ptx_helpers.cuh`](../../GEMMsm110/include/sm110_ptx_helpers.cuh)
- 说明：
  [`microbench/07_tma_gmem_smem_bandwidth/README.md`](../../microbench/07_tma_gmem_smem_bandwidth/README.md)
- 原始结果：
  [`microbench/07_tma_gmem_smem_bandwidth/results/tma_gmem_smem_bandwidth.csv`](../../microbench/07_tma_gmem_smem_bandwidth/results/tma_gmem_smem_bandwidth.csv)
- SASS：
  [`microbench/07_tma_gmem_smem_bandwidth/results/sass_summary.txt`](../../microbench/07_tma_gmem_smem_bandwidth/results/sass_summary.txt)
- NCU：
  [`microbench/07_tma_gmem_smem_bandwidth/results/ncu/ncu_tma_summary.csv`](../../microbench/07_tma_gmem_smem_bandwidth/results/ncu/ncu_tma_summary.csv)
- 对抗式审查：
  [`microbench/07_tma_gmem_smem_bandwidth/results/adversarial_review.md`](../../microbench/07_tma_gmem_smem_bandwidth/results/adversarial_review.md)
- 历史快照实测：L2-hit 773.443437 B/cycle/GPU；DRAM-stream
  155.779224 B/cycle/GPU。二者不具有新 campaign 的隔离/全网格计时合同。
- closure-qualified 18-case component summary：
  [`summary.json`](https://github.com/hibouwu/CUDA_optimazation/blob/ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c/results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json)
- closure-qualified component 独立审计：
  [`component_audit.json`](https://github.com/hibouwu/CUDA_optimazation/blob/ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c/results/sm110_closure_suite/thor-t5000-tma-ingress-supplement-maxn-20260814-c/component_audit.json)
- 边界：这是包含 issue、completion、mbarrier 和 SMEM destination 的端到端
  TMA ingress，不是纯 DRAM 或纯 SMEM port peak。历史结果用各 CTA 最大
  `clock64()` span；新的 closure 同时保留 issue→wait 的 `inflight=1`、四个
  slot 的 `inflight=4` 和八个 slot 的 `inflight=8`。八请求点精确使用 tc5a
  四个 stage 的 A=16 KiB、B=32 KiB destination、2D SW128 descriptor、四个
  48 KiB completion barrier 和 192 KiB SMEM staging；每个 stage 的 A/B 两笔
  TMA 共用一个 barrier，因此总计八笔请求在途。descriptor 固定
  `row_stride_elements=2048`，对应 calibration 的 N=K=2048；报告区分逻辑
  `working_set_bytes` 与包含 stride padding 的 `allocation_bytes`，后者不计入
  TMA payload。CTA 使用与 tc5a 相同的 192 threads/6 warps。
  L2-hit case 只启动一个 CTA 并直接形成每 SM
  ingress；整 GPU 共享 L2 read 仍由单独的全网格 memory-path case 和 1024
  B/cycle/GPU 的独立约束建模。DRAM-stream rate 保留为端到端 aggregate 条件。

基本命令：

```bash
cd microbench/07_tma_gmem_smem_bandwidth
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

### 12.5 TMEM ingress、consume、readback 与联合 overlap

- `tcgen05.cp` 说明与结果：
  [`microbench/06_tmem_cp_bandwidth/README.md`](../../microbench/06_tmem_cp_bandwidth/README.md)、
  [`results/tmem_cp_only_summary.csv`](../../microbench/06_tmem_cp_bandwidth/results/tmem_cp_only_summary.csv)
- TMEM consume 说明与结果：
  [`microbench/08_tmem_consume_bandwidth/README.md`](../../microbench/08_tmem_consume_bandwidth/README.md)、
  [`results/tmem_consume_results.csv`](../../microbench/08_tmem_consume_bandwidth/results/tmem_consume_results.csv)
- CP/MMA overlap 说明与结果：
  [`microbench/11_pipeline_overlap/README.md`](../../microbench/11_pipeline_overlap/README.md)、
  [`results/pipeline_overlap_results.csv`](../../microbench/11_pipeline_overlap/results/pipeline_overlap_results.csv)
- tc5a GMEM/L2→TMA→SMEM→MMA→TMEM readback/store 因果 source：
  [`tc5a_pipeline_dag.cu`](../../microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu)
- 双精度 182-case（每种精度 91 case）因果 manifest、runner、独立
  campaign/platform auditor 与模型导入器：
  [`contract_manifest.json`](../../microbench/sm110_gemm_causal_campaign/contract_manifest.json)、
  [`run_causal_campaign.py`](../../microbench/sm110_gemm_causal_campaign/run_causal_campaign.py)、
  [`audit_campaign.py`](../../microbench/sm110_gemm_causal_campaign/audit_campaign.py)、
  [`audit_causal_suite.py`](../../microbench/sm110_gemm_causal_campaign/audit_causal_suite.py)、
  [`causal_import.py`](../../scripts/sm110_gemm_model/causal_import.py)
- accumulator readback 源码与说明：
  [`tmem_readback_bandwidth.cu`](../../microbench/12_tmem_readback_bandwidth/tmem_readback_bandwidth.cu)、
  [`README.md`](../../microbench/12_tmem_readback_bandwidth/README.md)
- block-scale SFA/SFB ingress 源码与说明：
  [`tmem_scale_ingress_bandwidth.cu`](../../microbench/13_tmem_scale_ingress_bandwidth/tmem_scale_ingress_bandwidth.cu)、
  [`README.md`](../../microbench/13_tmem_scale_ingress_bandwidth/README.md)
- closure-compatible HBM/L2 读写源码与说明：
  [`memory_path_bandwidth.cu`](../../microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu)、
  [`README.md`](../../microbench/14_memory_path_bandwidth/README.md)
- unified component campaign、运行合同和独立审计：
  [`sm110_gemm_component_campaign`](../../microbench/sm110_gemm_component_campaign/README.md)
- 复用不可变 compute/full 基础证据时的 bounded component supervisor 与组合导入：
  [`sm110_component_supplement.sh`](../../microbench/sm110_component_supplement.sh)、
  [`run_sm110_component_supplement.sh`](../../microbench/run_sm110_component_supplement.sh)、
  [`closure_import.py`](../../scripts/sm110_gemm_model/closure_import.py)
- bounded epilogue preflight runner：
  [`run_epilogue_probe.py`](../../microbench/sm110_gemm_component_campaign/run_epilogue_probe.py)
- NVFP4 requant benchmark：
  [`requant_epilogue_benchmark.cu`](../../GEMMsm110/tests/requant_epilogue_benchmark.cu)
- E2M1 RNE/signed-zero reference：
  [`e2m1_encode.cuh`](../../GEMMsm110/include/requant/e2m1_encode.cuh)
- E2M1 packing、scale policy 和 SM110 TMEM epilogue：
  [`pack_fp4.cuh`](../../GEMMsm110/include/requant/pack_fp4.cuh)、
  [`scale_policy.cuh`](../../GEMMsm110/include/requant/scale_policy.cuh)、
  [`sm110_tcgen05_epilogue.cuh`](../../GEMMsm110/include/requant/sm110_tcgen05_epilogue.cuh)
- 各目录的 `results/sass_summary*` 和 `results/ncu/*` 保存 SASS/NCU 证据。
- 当前 headline：`tcgen05.cp` ingress 859.024 B/cycle/GPU；TS MMA consume
  115.699 B/cycle/GPU；steady CP/MMA pipeline 约 89% component-overlap
  efficiency。
- 边界：这些是特定 TS/CP 数据路径的需求率或经验工作点，不是 raw TMEM bank
  read/write peak，也不是联合容量外边界。`08_tmem_consume_bandwidth` 测的是
  TS MMA 从 TMEM 消费 A operand，不是 GEMM 尾部的 accumulator readback；两者
  不能共享参数。新 readback microbenchmark 明确发出 `tcgen05.ld` 并以 SASS
  `LDTM.x8`/`LDTM.x16` 为静态锚点，但只有 Thor 的 10-trial/20-SM 审计通过后
  才能进入经验包络。scale ingress 使用 `UTCCP.T.S.4x32dp128bit` 静态锚点和
  `LDTM.x4` value-check；rate 按每条 cp 的 512 B 唯一 source scale payload
  归一化，不按四分区 multicast 后的 2048 B destination footprint 夸大；每个
  commit batch 使用 32 个互不重叠的四列 TMEM slot，避免重复写同一异步目标造成
  人工 hazard。新的 HBM/L2 read case 让每个 16 B load 的四个 32-bit lane 全部
  进入最终 checksum，并要求 SASS 中存在 `LDG.E.128`；write case 要求
  `STG.E.128`，stop timestamp 之前执行 device-scope fence。因而四个 rate 都按
  实际保活的 16 B request 计数，而不是按源代码类型名猜测 transaction 宽度。
  还必须明确：`microbench/11_pipeline_overlap` 从已在 SMEM 的数据开始，测
  `tcgen05.cp`→TMEM 与 TS MMA；它不发出 tc5a 的 GMEM/L2 TMA request，不能提供
  causal profile 的 joint TMA+MMA startup/interval。新的双精度 campaign 才分别
  冻结 FP16/BF16 的 A16 KiB+B32 KiB、四 stage、八请求、192 threads、双
  accumulator persistent-worker 合同；在 Thor raw timing 回传前不产生模型
  profile 数值。

基本命令：

```bash
cd microbench/06_tmem_cp_bandwidth && ./build_and_run.sh summarize
cd ../08_tmem_consume_bandwidth && ./build_and_run.sh run
cd ../11_pipeline_overlap && ./build_and_run.sh run
bash ../../microbench/sm110_gemm_component_campaign/launch_component_campaign.sh <run-id>
```

### 12.6 SMEM、L1、DSMEM 与拓扑补充

- L1：
  [`microbench/04_l1_bandwidth`](../../microbench/04_l1_bandwidth/README.md)
- DSMEM：
  [`microbench/03_dsmem_bandwidth`](../../microbench/03_dsmem_bandwidth/README.md)
- DSMEM topology：
  [`microbench/09_dsmem_topology_contention`](../../microbench/09_dsmem_topology_contention/README.md)
- SMEM bank/stride：
  [`microbench/10_smem_bank_stride_bandwidth`](../../microbench/10_smem_bank_stride_bandwidth/README.md)

这些结果用于 schedule 合法性、bank conflict 和 CTA group/cluster 经验修正。由于
部分路径缺 direct byte counter，必须保留其 app clock、SASS、wavefront 或
miss-sector proxy 的证据边界。

### 12.7 完整 GEMM 已观测值

- 全精度实现/正确性 reference/同精度 denominator 覆盖合同：
  [`support_manifest.json`](../../microbench/sm110_full_gemm_campaign/support_manifest.json)
- 覆盖合同审计：
  [`audit_support_manifest.py`](../../microbench/sm110_full_gemm_campaign/audit_support_manifest.py)
- 首批 closure runner：
  [`run_full_gemm_campaign.py`](../../microbench/sm110_full_gemm_campaign/run_full_gemm_campaign.py)
- BF16/TF32/E5M2 完整 GEMM runner、E5M2 同精度 reference 与 host 自检源码：
  [`extended_gemm_bench.cu`](../../GEMMquant_sm110/src/extended_gemm_bench.cu)
- detached/resume launcher：
  [`launch_full_gemm_campaign.sh`](../../microbench/sm110_full_gemm_campaign/launch_full_gemm_campaign.sh)
- 独立结果审计：
  [`audit_campaign.py`](../../microbench/sm110_full_gemm_campaign/audit_campaign.py)
- reference minimum/median/maximum：由每个 case 的 `trials.jsonl` 中
  `reference_rate_per_second` 重算，不依赖摘要中单独保存的 median；
- Git 往返运行合同：
  [`README.md`](../../microbench/sm110_full_gemm_campaign/README.md)

- FP16→FP32 10-trial sweep：
  [`results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv`](../../results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv)
- 量化 1024 sweep：
  [`results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv`](../../results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv)
- FP16 runner：
  [`scripts/run_sm110_gemm_core_sweep.sh`](../../scripts/run_sm110_gemm_core_sweep.sh)
- 量化 runner：
  [`GEMMquant_sm110/scripts/run_quant_gemm_1024.py`](../../GEMMquant_sm110/scripts/run_quant_gemm_1024.py)
- FP16 主报告：
  [`GEMMsm110/SM110_GEMM_OPTIMIZATION_REPORT.md`](../../GEMMsm110/SM110_GEMM_OPTIMIZATION_REPORT.md)
- 量化主报告：
  [`GEMMquant_sm110/SM110_QUANT_GEMM_OPTIMIZATION_REPORT.md`](../../GEMMquant_sm110/SM110_QUANT_GEMM_OPTIMIZATION_REPORT.md)

当前工具只接收至少 10 个 trial 且全部 matched 的 backend series，再按最高
median 选择稳定最好实现。NVFP4/MXFP4 的性能 denominator 是 FP16 cuBLAS，
因此只保留绝对 GFLOP/s 和 correctness 状态，不使用历史 ratio 证明同精度胜负。
覆盖合同当前有 FP16→FP32、BF16→FP32、TF32→FP32、E4M3→FP32、
E5M2→FP32 和 S8→S32 达到“可启动 closure campaign”的实现条件。
[`cublasLtMatmul` 官方 FP8 类型表](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul)
没有列出 E5M2×E5M2 A/B 组合，因此 E5M2 改用独立启动的 global-load E5M2 MMA
作为同精度性能 denominator；该 reference 由独立 host E5M2 decoder 抽样验证，
candidate 的完整 FP32 输出再与 reference kernel 比较。两种 FP6、raw E2M1 和
U8 仍无完整路径，MXFP4/NVFP4 因输出合同、外部生成源码留存和
跨精度 denominator 只能标为 `partial`。这里的 `ready_for_closure_campaign`
仍不表示硬件闭环完成。

U8 被排除不是因为 PTX/SASS 不存在 U8 Tensor Core；compute-only campaign
已经覆盖该指令合同。缺口在完整 GEMM reference：
[`cublasGemmEx` 官方支持表](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex)
把 `CUBLAS_COMPUTE_32I` 的 A/B 类型限定为 signed `CUDA_R_8I`，没有列出
`CUDA_R_8U`。因此 U8 指令的静态存在不能充当同语义 cuBLAS reference；在独立
U8 full-GEMM candidate、reference 和 denominator 完成前，
模型宁可保留缺口。

新版 campaign 把可闭环的六种合同冻结为 18 个 square `NN` case：每种精度
`N=1024,2048,4096`，其中前两点是 calibration，4096 是预先保留的 holdout。
FP16 使用 `tc5b`（1024）和 `tc5a`（2048/4096），E4M3 使用 `q7`，S8 使用
`q15`；BF16 使用原生 BF16 WMMA，TF32 使用原生 TF32 WMMA，E5M2 使用 shared-
memory candidate 与 global-load reference。前五种库可支持的合同与同精度
cuBLAS/cuBLASLt reference 成对；E5M2 则使用上述同精度独立 kernel denominator，
不是把跨精度库结果冒充 reference。每个外层 case 运行 10 个独立 trial；每个 trial 内
候选和 reference 使用同一输入，FP16 内层计时 100 次，其余内层计时 10 次。
runner 独立从 `2N^3/time` 重算吞吐，S8 明确使用 OP/s。

数值证据包含两级门禁：先用实际量化后的输入在 CPU 上抽样 64 个输出，检查
cuBLAS/cuBLASLt reference；再把候选的完整输出矩阵与该 reference 比较。S8→S32
要求 bit-exact，浮点累加按显式 `atol`/`rtol` 合同判断。TF32 输入在 candidate
和 reference 之前都显式按 round-to-nearest-even 截到 TF32 的 10 个 fraction
bits；host-only 自检覆盖 retained-LSB 为偶/奇的两个 halfway case，并确认 Inf 和
NaN payload 不被改写。静态证据也不是二进制级 mnemonic 搜索：审计在被测
kernel 的 SASS 函数块内检查 `UTCHMMA`、`HMMA.16816.F32.BF16`、
`HMMA.1684.F32.TF32`、FP8 `HMMA.16816.F32` 或 `IMMA.16816.S8.S8` 及 store。
当前 CUDA 13.0 本地静态门禁 18/18 通过；在 Thor 返回 180 个 trial、环境和
必需的 NCU artifact 之前，不把它们升级为新的已观测值。

compute-only 和 full-GEMM runner 的成功 trial 都保存 120 s timeout 合同；选中的
NCU 记录保存 300 s timeout 合同。两个独立 auditor 均拒绝旧 schema、缺失
timeout 字段、`timed_out=true` 或 `termination_failed=true` 的证据。component
runner 同样对每个外部 trial 使用 120 s timeout。对应总协调器会先运行 30 s
bounded epilogue preflight，再串行启动并审计 compute、component 和 full-GEMM，
任何一级失败都不会继续到下一级。

compute、component、full-GEMM 三批使用同一非阻塞 GPU 文件锁，因此 Thor 上
只能串行运行；这把“不要并发争抢 GPU”从操作约定升级为 runner 的机械约束。
推荐用固定提交检查、逐批等待和逐批审计的总协调器运行：
[`run_sm110_closure_suite.sh`](../../microbench/run_sm110_closure_suite.sh)。

### 12.8 硬件和软件环境来源

- compute-only 报告记录 Thor、20 SM 和 1.575 GHz：
  [`microbench/mma_compute_only/分析报告.txt`](../../microbench/mma_compute_only/分析报告.txt)
- 当前模型硬件快照：
  [`scripts/sm110_gemm_model/profiles/thor_sm110.json`](../../scripts/sm110_gemm_model/profiles/thor_sm110.json)
- 当前参数与逐项 locator：
  [`scripts/sm110_gemm_model/profiles/capacities.json`](../../scripts/sm110_gemm_model/profiles/capacities.json)
- bounded closure detached launcher：
  [`microbench/launch_sm110_closure_suite.sh`](../../microbench/launch_sm110_closure_suite.sh)
- 固定提交、串行等待和逐级独立审计协调器：
  [`microbench/run_sm110_closure_suite.sh`](../../microbench/run_sm110_closure_suite.sh)
- 从当前 `HEAD` 冻结合同、保存平台证据、detached 启动并完成模型导入的统一入口：
  [`microbench/sm110_closure_campaign.sh`](../../microbench/sm110_closure_campaign.sh)
- 与 runner 同提交维护的 Thor 操作手册：
  [`Docs/blackwell_tensorcore/THOR_CLOSURE_RUNBOOK.md`](THOR_CLOSURE_RUNBOOK.md)
- closure evidence 到模型 `Capacity`/`ObservedBest` 的 fail-closed 导入器：
  [`scripts/sm110_gemm_model/closure_import.py`](../../scripts/sm110_gemm_model/closure_import.py)
- 从已审计输入机械生成容量表、完整 GEMM 对比、上界反证和 holdout 分析：
  [`scripts/sm110_gemm_model/closure_report.py`](../../scripts/sm110_gemm_model/closure_report.py)
- 合并 implementation readiness、逐 shape numeric coverage 和最终 fail-closed 门禁：
  [`scripts/sm110_gemm_model/precision_report.py`](../../scripts/sm110_gemm_model/precision_report.py)
- 当前 12 精度机器生成证据矩阵：
  [`thor_sm110_all_precision_evidence_matrix.md`](./thor_sm110_all_precision_evidence_matrix.md)
- generic/byte-container/block-scale/tc5a 的 54-case 精确 TMA resource 合同：
  [`contract_manifest.json`](../../microbench/sm110_gemm_resource_campaign/contract_manifest.json)
- 对应 CUDA source、bounded/resumable runner 和双层独立 auditor：
  [`tma_ab_contract_bandwidth.cu`](../../microbench/15_tma_ab_contract_bandwidth/tma_ab_contract_bandwidth.cu)、
  [`run_resource_campaign.py`](../../microbench/sm110_gemm_resource_campaign/run_resource_campaign.py)、
  [`audit_campaign.py`](../../microbench/sm110_gemm_resource_campaign/audit_campaign.py)、
  [`audit_resource_suite.py`](../../microbench/sm110_gemm_resource_campaign/audit_resource_suite.py)
- 资源结果到 54 个精确 `Capacity` 的重新审计导入器：
  [`resource_import.py`](../../scripts/sm110_gemm_model/resource_import.py)
- tc5a persistent-worker 因果 source、双精度 182-case runner 与双层 auditor：
  [`tc5a_pipeline_dag.cu`](../../microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu)、
  [`run_causal_campaign.py`](../../microbench/sm110_gemm_causal_campaign/run_causal_campaign.py)、
  [`audit_campaign.py`](../../microbench/sm110_gemm_causal_campaign/audit_campaign.py)、
  [`audit_causal_suite.py`](../../microbench/sm110_gemm_causal_campaign/audit_causal_suite.py)

### 12.9 2026-08-17 parameter supplement 的导入边界

Thor 已返回 parameter supplement 结果：采集代码提交为
`0c42cbb`，结果分支为
`thor-results/thor-t5000-parameter-plots-maxn-20260817-i`，GPU 数据提交为
`aa845dd`，suite-log 后续提交为 `78e0948`。TMA payload 为 10/10 case、每 case
10 trial；memory duplex 为 21/21 case、每 case 10 trial。两批独立 auditor 均通过，
当前导入器对该真实结果树的离线重放得到 10 个 payload capacity 和 21 个 duplex
capacity。

这 21 个 duplex capacity 必须按证据语义拆分：14 个 hot-L2 case 导入为精确
read:write ratio 作用域的 `l2.duplex`；7 个 cold case 只能导入为
`hbm.duplex.proxy`。后者的 NCU 证明 external read 的 L2 miss sector 和 write 已进入
L2 write path，但明确记录 `external_write_bytes_proven=false`，所以绝不能满足经验层
要求的物理 `hbm.duplex`。因此“cold proxy ratio 全测完”与“HBM 外部读写 joint
capacity 闭合”是两个不同布尔门禁。

迁移后的经验层对 cold-HBM 同时要求 unique-input/output ratio 匹配的物理
`hbm.duplex`，以及 schedule-level repeated-TMA/output ratio 匹配的 `l2.duplex`；
hot-L2 只要求后者。条件上界层仍独立使用共享 `hbm.total`，以及整卡共享的
`l2.read`=1024 B/cycle/GPU 和 `l2.write`=512 B/cycle/GPU。每 SM 的 TMA ingress
仍是独立出口，不能乘进或除进这两个共享 L2 总线参数。这样既不会同时拼接两个
互不保证可同时达到的单向 peak，也不会把 per-SM ingress 与 GPU-wide fabric
重复计数。

对已回传目录重新审计并生成可合并 JSON 的命令为：

```bash
python3 -m scripts.sm110_gemm_model.cli import-tma-payload-campaign \
  --repo-root . \
  --run-dir results/sm110_tma_payload_campaign/\
thor-t5000-parameter-plots-maxn-20260817-i-tma-payload

python3 -m scripts.sm110_gemm_model.cli import-memory-duplex-campaign \
  --repo-root . \
  --run-dir results/sm110_memory_duplex_campaign/\
thor-t5000-parameter-plots-maxn-20260817-i-memory-duplex
```

把真实 `-i` 结果与当前 workload/schedule manifest 合并重放后，所有必须的 L2
duplex ratio 均已测量，所有必须的 cold HBM ratio 也都有 proxy，但物理 HBM duplex
ratio 仍为 0。五点 payload surface 覆盖除 block-scaled scale transport 外的必须 payload；
MXFP4/NVFP4 当前精确合同还需要 512 B 和 1024 B 的独立 scale payload，
因此 `all_required_tma_payloads_measured=false`。这比简单写“10/10 case 完成”更严格：
runner 自己的 case matrix 完成，不代表后续扩展的 schedule manifest 每个 payload 都在
其覆盖域内。

所以这轮数据关闭的是已声明五点 TMA surface、全部当前 L2 duplex ratio 和全部
cold proxy ratio，不会把 `exact_tma_topology_surface.complete`、物理 HBM duplex、
joint pipeline、全部精度完整 GEMM或最终 `complete` 改成 true。物理 HBM duplex 与逐
workload/schedule 的 exact joint-pipeline capacity 仍需独立 runner；在它们返回前，
模型必须给出 `insufficient_evidence`，而不是用 proxy 或独立 component peak 补数。

parameter supplement 的具体 microbenchmark 来源为：

- TMA payload CUDA source：
  [`tma_gmem_smem_bandwidth.cu`](../../microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu)
- TMA payload runner、独立 auditor 与合同说明：
  [`run_tma_payload_campaign.py`](../../microbench/sm110_tma_payload_campaign/run_tma_payload_campaign.py)、
  [`audit_campaign.py`](../../microbench/sm110_tma_payload_campaign/audit_campaign.py)、
  [`README.md`](../../microbench/sm110_tma_payload_campaign/README.md)
- memory duplex CUDA source：
  [`memory_path_bandwidth.cu`](../../microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu)
- memory duplex runner、独立 auditor 与 cold-proxy 合同：
  [`run_memory_duplex_campaign.py`](../../microbench/sm110_memory_duplex_campaign/run_memory_duplex_campaign.py)、
  [`audit_campaign.py`](../../microbench/sm110_memory_duplex_campaign/audit_campaign.py)、
  [`README.md`](../../microbench/sm110_memory_duplex_campaign/README.md)
- 两批结果到模型容量的 fail-closed 导入器：
  [`evidence_import.py`](../../scripts/sm110_gemm_model/evidence_import.py)

后续复测必须另外保存 GPU 名称、SM/compute capability、driver、CUDA、NVCC、
NCU、时钟、功耗模式、温度、Git commit、编译命令、binary hash、SASS hash 和
运行时间戳。本轮 bounded compute/component/full-GEMM bundle 已保存这些 campaign
级证据；更早的零散 snapshot 仍缺少统一 manifest，只能保留为 `snapshot_only`。
逐 schedule TMA 合同的采集程序已经通过 54/54 本地静态合同、`sm_110a`
交叉编译与函数块级 SASS attribution；本机 GPU 是 SM120，formal binary 明确拒绝
非 SM110，因此没有把本机 runtime 冒充 Thor 证据。这里所指 54-case 精确
schedule/precision/row-stride TMA capacity 尚未从 Thor 返回，仍是
证据缺口，不能先写入模型。结果返回后 `finish` 会生成
`results/sm110_model_closure/$SUITE_ID/resource_capacities.json`，报告必须用
`--resource-import` 显式加载；hot-L2 仍是 B/s/SM，cold-DRAM 仍是 B/s/GPU，二者
都保持 `measured_sustained`。FP16/BF16 tc5a 因果程序已
通过 182/182 静态合同、`sm_110a` 编译、六个 precision/stage 函数块级 SASS
attribution 和合成 1,820-trial auditor 回归；epilogue latency 对每种精度都只用
单 output-task case 校准，多 task case 专门验证双 accumulator 与 drain 递推。
manifest、CSV、tensor-map 类型、SASS instruction descriptor immediate 和模型
profile 五处相互绑定；导入结果必须是 `precision_ids=["fp16_f32"]` 与
`precision_ids=["bf16_f32"]` 两个 singleton，不能因 payload 相同而互相复用。
本机 SM120 不能执行该 SM110
tcgen05 合同，所以也不能写入 timing profile。
当前目标其余未完成原因是 Thor causal/resource profile、其余 schedule 的因果合同、
缺失精度完整 GEMM 与部分 strict compute upper，而不是把已有环境字段重复抄写一遍。

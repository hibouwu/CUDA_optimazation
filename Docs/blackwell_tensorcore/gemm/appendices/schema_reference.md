# 可执行 schema reference\n\n本附录冻结代码提交 `f06f2cd917a4cb23806b5e1be06120be9152ed7b` 的模型字段语义。规范公式见 [`model/`](../model/01_scope_and_claims.md)；本页只定义 serialization、applicability 与审计字段。空 tuple 表示该记录没有额外收窄该维度，不代表已经证明跨所有硬件适用。\n\n## 0. 可执行 schema 参数首次定义

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
| `alpha`, `beta` | \(D=\alpha AB+\beta C\) 的标量系数。 |
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

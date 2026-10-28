# 探针源码索引

本目录是全部 GPU 探针的真实源码。多个模块与结果族共用的四套探针有阅读目录，集中展示相关代码；其余多为 R01–R17、V01–V06 等早期实验各自的探针，各自有一两个 `run_*`／`analyze_*` 脚本。下表由源码开头注释与模块引用生成，“microbench 中引用它的模块”不含测试；显示“无”的探针由轮次目录中的脚本构建和运行。

| 探针文件 | 源码开头的说明 | microbench 中引用它的模块 | 阅读目录 |
|---|---|---|---|
| `b01_smem.cu` | B01 independent LDS.128 + STS.128, disjoint address regions and values. | `run_b01` | — |
| `b01_tma_large.cu` | B01 single-CTA tensor transport with a 128 MiB rotating address set. | `run_b01` | — |
| `barrier_return.cu` | CUTLASS barrier return paths: already complete versus waiting before arrival. | 无；入口在轮次目录中 | — |
| `cfg_b_compute_chain.cu` | cfg_b consumer body: one WG, two M64 accumulator fragments, eight WGMMA/Ktile. Six immutable SMEM tiles are prepared before timing. This is not a full | `cfg_b_compute_chain`、`cfg_b_handoff_chain`、`cfg_b_tma_chain`、`q04_index_control`、`q04_native_compute_wait`、`q05_compute_wait` | [read-resident-chain](../read-resident-chain/README.md) |
| `cfg_b_handoff_chain.cu` | cfg_b resident-operand handoff: one producer WG and one consumer WG. Six physical operand slots, four or six active full(1)/empty(128) barriers. Ordin | `cfg_b_handoff_chain`、`cfg_b_tma_chain`、`q02_native_input_growth`、`q02_native_input_validation`、`q02_native_tma_chain`、`q04_first_entry_control` 等 11 个 | [read-resident-chain](../read-resident-chain/README.md) |
| `cfg_b_handoff_sparse.cu` | cfg_b sparse full/empty handoff endpoints; arithmetic and protocol retain the original chain. Six physical operand slots, four or six active full(1)/e | 无；入口在轮次目录中 | [read-resident-chain](../read-resident-chain/README.md) |
| `cfg_b_pingpong_chain.cu` | E: the actual S6 CUTLASS cfg_b pingpong kernel, three persistent output tiles. Plain uses the unmodified factory. Trace uses the frozen V06 overlay pl | `cfg_b_pingpong_chain` | [read-resident-chain](../read-resident-chain/README.md) |
| `cfg_b_pingpong_trace.hpp` | The E trace uses the frozen V06 hook sites and four additional call boundaries. | `cfg_b_pingpong_chain` | [read-resident-chain](../read-resident-chain/README.md) |
| `cfg_b_tma_chain.cu` | cfg_b input supply: three native TMA requests and eight WGMMA per Ktile. One producer WG and one consumer WG; six physical slots, four or six active.  | `cfg_b_tma_chain`、`q02_local_input_chain`、`q02_native_input_growth`、`q02_native_tma_chain` | [read-resident-chain](../read-resident-chain/README.md) |
| `gaps_common.hpp` | Helpers for the R10/R13/R14/R15 follow-up only. Leading dimensions are in elements. | `r17_run`、`run_r10`、`run_r13`、`run_r14`、`run_r15`、`run_r18` 等 9 个 | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |
| `gpc_l2_contention.cu` | Small same/cross-GPC reference contrasts for unresolved SM pairs. | 无；入口在轮次目录中 | [read-multicta-tensor](../read-multicta-tensor/README.md) |
| `gpc_topology.cu` | Cluster co-residency identifies same-GPC edges; no latency price is fitted. | 无；入口在轮次目录中 | [read-multicta-tensor](../read-multicta-tensor/README.md) |
| `l1_lsu.cu` | One CTA, warp-contiguous 16 B pointer nodes, fixed total work across J. J counts independent software chains; SASS owns the actual issue schedule. | `l1_lsu_probe` | — |
| `l2_saturation.cu` | Narrow L2 requested-service probe. Build for sm_90a; GPU execution is explicit. J is the number of independent software streams, not a physical queue  | `cfg_b_compute_chain`、`cfg_b_handoff_chain`、`cfg_b_pingpong_chain`、`cfg_b_tma_chain`、`l1_lsu_probe`、`l2_fit` 等 49 个 | — |
| `numa_latency.cu` | Memory-affinity development probe: discard on dedicated scratch lines, then CG cold/hot loads. | 无；入口在轮次目录中 | [read-multicta-tensor](../read-multicta-tensor/README.md) |
| `q02_fixed_source_pair.cu` | Same allocation; four arms pair stage and source-to-SM binding. | 无；入口在轮次目录中 | [read-resident-chain](../read-resident-chain/README.md) |
| `q02_tma_compute_ctas.cu` | Fixed D chain with per-CTA records/output; optional runtime source batches. | 无；入口在轮次目录中 | [read-resident-chain](../read-resident-chain/README.md) |
| `q07_input_output_joint.cu` | Four organizations of the same input and real cfg_b output services. | `q07_input_output_joint` | — |
| `r00_common.hpp` | — | `analyze_tma_input_events`、`analyze_v01`、`cfg_b_compute_chain`、`cfg_b_handoff_chain`、`cfg_b_pingpong_chain`、`cfg_b_tma_chain` 等 48 个 | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md)、[read-multicta-tensor](../read-multicta-tensor/README.md)、[read-resident-chain](../read-resident-chain/README.md)、[read-single-cta-tma](../read-single-cta-tma/README.md) |
| `r00_cutlass.cu` | — | `analyze_cutlass_collective`、`analyze_v01`、`cutlass_collective_model`、`prepare_cutlass_phases`、`r07_run`、`r08_run` 等 10 个 | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |
| `r00_gemm.cu` | — | `r07_run` | — |
| `r00_wgmma.cu` | — | `prepare_cfg_c_mma_reference`、`run_layout_pair` | — |
| `r01.cu` | R01: dependency chains (FFMA, 32-bit add, SMEM/global address chains, ldmatrix, WGMMA). Every timed loop iteration holds kUnroll dependent steps per c | `replay_l1_lsu_evidence`、`replay_register_evidence`、`run_r01`、`v01_predict` | — |
| `r01_support.hpp` | Shared host/device helpers for the R01 and R06 probes. | `run_b01`、`run_r01`、`run_r02`、`run_r11`、`run_r12`、`run_r16` | — |
| `r02.cu` | R02: legal source/pool/mixed-return probes. Timers distinguish producer progress from a consumer-dependent endpoint; neither claims completion of over | `plot_r02`、`q00_a044_legacy_order`、`replay_register_evidence`、`run_r02`、`run_r02_source_frame` | — |
| `r03.cu` | — | `analyze_r03`、`analyze_r03_lds_sts`、`prepare_r03_lds_sts`、`q00_a044_legacy_order`、`replay_smem_protocols`、`run_r03` 等 9 个 | — |
| `r03_lds_sts.cu` | — | `prepare_r03_lds_sts`、`replay_smem_evidence`、`run_r03_lds_sts` | — |
| `r04.cu` | R04 joint service on GH200 (sm_90a), single CTA, clock64 window.  Families: 0 WGMMA+FFMA in the same warpgroup, 1 WGMMA in warpgroup 0 + FFMA in warpg | `analyze_r04`、`q00_a044_legacy_order`、`run_r04` | — |
| `r05.cu` | — | `analyze_r05`、`analyze_tma_input_events`、`run_r05`、`tma_input_events`、`v01_predict` | — |
| `r05_wg_inline.cuh` | — | `run_r05` | — |
| `r06.cu` | R06: fixed-work FFMA service under three register-allocation tiers, one CTA or a 4 x SM grid. 128 threads, 8 independent chains per lane; each timed l | `replay_sm_residency`、`run_r01`、`run_r06` | — |
| `r07_probe.cu` | R07: launch-overhead split and clock-under-load probe.  Backends: the fixed R00 CUTLASS kernel (same type definitions as r00_cutlass.cu), | `r07_analyze`、`r07_run` | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |
| `r08_probe.cu` | R08: wave/tail scaling and tile traversal (raster, swizzle, cluster) of the fixed CUTLASS GEMM.  Kernel types are identical to r00_cutlass.cu / r07_pr | `r08_run` | — |
| `r09_plain_prefix.cu` | Build as an isolated shared module, without trace overlay or observer macros. | `prepare_r09_guarded_clock`、`r09_plain_prefix_audit` | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |
| `r09_plain_prefix.h` | — | `prepare_r09_guarded_clock`、`r09_plain_prefix_audit` | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |
| `r09_plain_prefix_host.hpp` | — | `prepare_r09_guarded_clock`、`r09_plain_prefix_audit` | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |
| `r09_probe.cu` | R09: in-kernel clock and stage split of the fixed CUTLASS kernel (NDEBUG build).  Same kernel types as r00_cutlass.cu / r07_probe.cu. Two builds of th | `r09_run` | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |
| `r10.cu` | R10: fixed 2600x3000x2000 GEMM, explicit physical B row stride. | `analyze_r10_address_feedback`、`analyze_r10_boundary_binding`、`analyze_r10_boundary_ledger`、`analyze_r10_boundary_lifecycle`、`analyze_r10_boundary_transfer`、`analyze_r10_clock_consistency` 等 17 个 | — |
| `r11.cu` | R11: identical action counts, state, consumers and two phase boundaries per unit. Order is a runtime input: all organizations of one pair use one kern | `plot_r11`、`replay_sm_joint`、`run_r11` | — |
| `r12.cu` | R12: independent TMA, WGMMA and STS regions; fixed 384-thread role assignment. | `plot_r12`、`replay_sm_joint`、`replay_smem_protocols`、`run_r12` | — |
| `r13.cu` | R13: TMA supply, consumer completion, slot retirement. Four slots always reserved. | `analyze_r13_release_interpretation`、`analyze_r13_supply`、`analyze_r18`、`decode_r13_wait_feedback`、`decode_r13_wait_sum`、`prepare_r13_wait_feedback` 等 10 个 | — |
| `r14.cu` | R14: 4-CTA persistent GEMM, per-role/per-tile tracing. See run_r14.py. | `analyze_r15_lifecycle`、`run_r14`、`run_v05`、`v05_calibrate` | — |
| `r14_trace.hpp` | — | `analyze_r15_lifecycle`、`run_r14`、`run_v05` | — |
| `r15.cu` | R15: 16KiB output chunks, explicit source-read / release / full-write boundaries. | `analyze_r15`、`analyze_r15_output_ns`、`q00_a044_legacy_order`、`r09_v08_clock`、`run_r15`、`run_r15_output_ns` | — |
| `r15_fragment_store.cuh` | One active output warp pair stores64 fragment registers per thread. | `run_r15` | — |
| `r16_quota.cu` | R16 quota: release before/after a dependent integer delay; two consumer WGs. | `replay_delivery`、`run_r16` | — |
| `r16_residency.cu` | R16 residency: two warpgroup CTA, stage ring, identical 1/2-CTA capacity controls. | `run_r16` | — |
| `r17.cu` | R17: copy of v06.cu (same GEMM, same light per-CTA stamps via the V06 overlay) with one added config cfg_a1 = cfg_a without the cluster (128x128x64 co | `r17_analyze`、`r17_rule`、`r17_run` | — |
| `r18.cu` | R18/R19: physical-grid matched boundaries, explicit zeros, and recorded work coordinates. Copied from the earlier probe: explicit lda/ldb/ldd, warm pr | `analyze_cfg_a_supply`、`analyze_cfg_c_dependencies`、`analyze_cfg_c_supply`、`analyze_r10`、`analyze_r10_boundary_transfer`、`analyze_r13_supply` 等 34 个 | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |
| `r18_first_output.hpp` | Five local control-flow cuts, deferred until the consumer exits its work loop. No completion/arrival or hardware clock-enclosure claim is made by this | `prepare_r18_first_output` | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |
| `r18_trace.hpp` | Default: clock64 events, globaltimer at CTA entry/final. R18_DUAL_CLOCK adds globaltimer to the same four per-tile sites and to producer first work. | `analyze_r13_supply`、`prepare_r13_wait_feedback`、`prepare_r13_wait_sum`、`run_r18`、`run_v08` | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |
| `tma_gemm_sources.cu` | cfg_b source coordinates with the tensor probe controller; warm both output work lists. | `q02_a_stride`、`q02_control_floor`、`q02_descriptor_factor`、`q02_prefill_active`、`q02_prefill_observer`、`q02_prefill_shared_prefix` 等 13 个 | [read-single-cta-tma](../read-single-cta-tma/README.md) |
| `tma_tensor_background.cu` | cfg_b tensor requests with a separate warpgroup issuing controlled WGMMA background. | 无；入口在轮次目录中 | [read-multicta-tensor](../read-multicta-tensor/README.md) |
| `tma_tensor_multicta.cu` | cfg_b tensor requests with a separate, explicit probe control program. | `analyze_tma_tensor_candidates`、`cfg_b_request_model`、`fit_tma_tensor_two_tier`、`fit_tma_tensor_workingset`、`prepare_tma_tensor_discrimination`、`q02_source_isolation` 等 8 个 | [read-multicta-tensor](../read-multicta-tensor/README.md) |
| `v01.cu` | V01 held-out combinations on GH200 (sm_90a).  kind 0  LDS -> FFMA, 128 threads, inputs preloaded in SMEM. | `analyze_v01`、`analyze_v01_followup`、`q00_a044_legacy_order`、`run_v01`、`v01_followup`、`v01_predict` | — |
| `v01_wg_inline.cuh` | Legacy small-TC lifecycle (kind 1 of v01.cu), unchanged since 2026-10-06: each of the four K16 m64n64 WGMMA is fenced, committed and drained with wait | `run_v01` | — |
| `v02_probe.cu` | V02: predict-then-measure run of the fixed CUTLASS kernel (NDEBUG build), copied from r09_probe.cu. Kernel types, warm protocol and sampled FP64 check | `v02_run` | — |
| `v03_probe.cu` | V03: in-call SM clock calibration and predict-then-measure run of the fixed CUTLASS kernel (NDEBUG build), copied from v02_probe.cu. Kernel types, war | `v03_run` | — |
| `v04_probe.cu` | V04: config-transfer test. Three CUTLASS v3.9.2 sm90 FP16->FP32 GEMMs (ElementC=void, persistent tile scheduler, -DNDEBUG), selected at compile time b | `v04_predict`、`v04_run` | — |
| `v05.cu` | V05: held-out full-grid GEMM. Never time before predictions are frozen. | `q00_a044_legacy_retry`、`run_v05`、`v05_calibrate` | — |
| `v05_trace.hpp` | — | `run_v05` | — |
| `v06.cu` | V06: full-grid GEMM, plain or light per-CTA stamps (v06_trace.hpp via include overlay). Copied from the earlier probe: explicit lda/ldb/ldd, warm prot | `r17_rule`、`v06_analyze`、`v06_run`、`v07_model` | — |
| `v06_trace.hpp` | V06 light per-CTA stamps (clock64; globaltimer only at CTA entry and final release). Per CTA (uint64 words): | `analyze_r13_supply`、`cfg_b_composition`、`cfg_b_pingpong_chain`、`prepare_cfg_c_gt_bracket`、`prepare_cfg_c_producer_tail`、`prepare_r09_guarded_clock` 等 15 个 | [read-cfg-b-gemm-trace](../read-cfg-b-gemm-trace/README.md) |

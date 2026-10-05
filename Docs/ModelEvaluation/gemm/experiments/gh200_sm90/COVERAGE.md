# GH200 本轮资源覆盖约定

状态是工作范围，不代表测量已经完成。

| 资源 | 阶段 | 处理 | 边界 |
|---|---|---|---|
| compute | S05,S06,S07,S08 | measure | FMA/MMA/WGMMA，包括固定来源差异和稠密FP8/INT8代表形式 |
| shared_access | S04,S09 | measure | scalar/vector/stride/broadcast/联合读写 |
| matrix_and_warp | S10 | measure | ldmatrix/stmatrix/shfl |
| sync | S11 | measure | warp/CTA/mbarrier/fence，等待条件区分 |
| async_copy | S12 | measure | cp.async分组与可见性 |
| global_path | S13 | measure | 请求吞吐与读写联合服务 |
| tma | S14,S15,S16 | measure | bulk/tensor双向与有限并发 |
| cluster | S17 | measure | DSM/多播/cluster同步 |
| occupancy | S18 | measure_constraints | 分配/线程/寄存器/SMEM、carveout；API上限与观测分开 |
| local_spill | S18 | measure | 受控local/spill代表路径 |
| setmaxnreg | S18 | constraints_only | 合法性和资源条件，不承诺通用成本模型 |
| auxiliary | S18 | measure | 整数地址、转换、同址/无争用原子代表点 |
| param_const_descriptor | S18,S19,S20 | account_in_probes | 记录实际需求与生成指令，不扫描全部性能 |
| pipeline | S19,S20 | measure | 固定FP32 SIMT和BF16 WGMMA受控组合 |
| physical_queue_depth | none | deferred | 软件stage扫描不证明物理在途容量 |
| full_gemm_and_autotune | none | out_of_scope | 后续独立目标 |

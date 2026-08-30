# 阶段 0：工作区、固定源码与覆盖分母

这份报告记录 59 个显式 Tensor Schedule Tag、11 个 `KernelScheduleAuto` 控制项和固定
CUTLASS 源码的清点结果。它属于工程证据，不代替教学正文，也不产生 Thor 运行、数值或
性能结论。

## 阶段开始

第一性原理目标：证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

这里的“同一目标函数”按单个实例解释：能够生成代码的实例，其 PTX 和 SASS 必须归属于
该实例唯一的 kernel symbol；静态拒绝项在有源码约束和编译诊断支持的失败层结束，不要求
产生目标函数。不同实例之间不共用函数归属结论。

阶段入口固定在隔离分支 `codex/sm110a-codegen-closure`。CUTLASS submodule HEAD 为
`e05f953a5b3d38adc240df2ff928e0421c2abba3`，checkout 为 clean；C++ recipe key 固定为
`cutlass::arch::Sm100`，编译目标固定为 `compute_110a/sm_110a`。本阶段只建立分母、来源、
证据边界和门禁，不编译新的 CUDA kernel。

固定清点得到 59 个 public leaf Tag，分为 11 组。每个 Tag 只承诺一个基准实例：39 个能在
官方 C++ test/example 中找到显式或条件式引用，1 个只在官方注释中给出 Auto 候选映射，
5 个可由 `generator.py` 的配置还原，其余 14 个从相邻合法配置沿一个明确变化轴派生。
14 个派生项都记录了直接父项和唯一变化轴；校验器会确认父子项使用同一来源锚点，并拒绝
循环依赖。

11 个 Auto 控制项同样逐组建账。固定源码给出的待验证推测是 4 项预计能够完成类型构造，
7 项预计会在 Builder 或 kernel 组合阶段被拒绝；这些不是本阶段终态。

## 实现与正向验证

清单和校验器绑定了以下身份：

| 对象 | SHA-256 |
|---|---|
| `dispatch_policy.hpp` | `fcce5fffb3118b15fea5aa59e39bea15ddd18c28c858b434ced889045968117c` |
| 来源清单 | `1370b33695542ceae8103a6280dd422e3122dcdd53cd0c69d55a92853d3cd296` |
| Auto 控制项清单 | `de7032f8234737b747b0d962c37809bb1b4b843adf7d26b12f736a0e0b15bfc3` |
| 完整工具链锁文件 | `d0c38e0759646fdb5950669aa055f564d43c7bb60984bbc5cacbff4bfcdf242a` |
| 工具链锁 canonical self-hash | `b74d16f06401b694daf94aa1d896238873ea29302da60bd49344f2eda0c0fe63` |

正向校验重新提取固定源码中的 59 个 Tag，核对 Tag→group、case 中实际
`mainloop_schedule`、文档的分组/状态/case、官方 C++ 确定性 occurrence、generator enum 到
C++ Tag 的映射、source-derived parent anchor、Auto 假设源码范围和所有 SHA。当前状态仍是
6 个 `HISTORICAL_STATIC_PASS`、53 个 `NOT_CHECKED`；没有把旧快照升级为 fresh 结果。

## 对抗审查

第一性原理目标：证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

本阶段最终保留 56 个故障注入用例，另有 2 个用于构造基准状态的辅助用例。反例覆盖目标漂移、Tag
漏项/重复/跨组交换、Auto 混入显式分母、错误 case Schedule、历史证据标志缺失、文档状态
与汇总数字漂移、官方源码引用位置选错、generator enum 映射串线、源码派生项与父实例的
来源锚点错配、Auto
假设 hash 漂移、CUTLASS HEAD/dirty/source hash、工具链锁漂移、空壳 result 和阶段完成报告
损坏。

独立红队审查实际找到并推动修复了以下问题：清单能够自行扩大允许状态、缺源码仍通过、错误 case
映射、官方 Mxf4 注释被误当显式实例、来源分类从 `40/7/12` 修正为 `40/5/14`、Auto 11 项
缺少结构化清单、Auto 被错误要求解析成 public leaf、2SM pointer FastFP32 Smem parent
错配、文档表格与摘要可漂移、工具链锁未冻结、阶段状态可以解锁空壳 result，以及
`load_result_record` 一次补丁拼接错误。

## 修复

第一性原理目标：证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

修复后的门禁由 validator 自己持有允许状态、固定目标 hash、Tag 集合 hash、Tag→group hash、
来源清单 hash、Auto 清单 hash 和工具链锁 hash。`--require-source` 同时要求 exact HEAD、clean
checkout 和固定源文件 hash。官方 40 项按 test 优先、example 次之、最短路径、字典序和文件
内第一次 occurrence 重算；generator 配置还要闭合 enum→C++ Tag；source-derived 项还要与
immediate parent 使用同一个 seed anchor。

Auto 控制项记录待验证结果、失败层和源码范围 hash。成功时后续填写 resolved Builder
specialization 与内部 `DispatchPolicy`，不强制物化成 public leaf Tag。结果合同在阶段 1 完成
前，校验器拒绝接收任何依赖新结果的终态；仅修改 phase JSON 不能提交新的 `STATIC_PASS`
或其他 result-backed 状态。

教学文档的 59 行表、11 行 Auto 表、状态汇总和来源汇总均与机器清单逐项对账。完成阶段还
必须绑定带 hash 的阶段报告，并在“阶段开始、对抗审查、修复、阶段完成”四处原样重复目标。

## 阶段完成

第一性原理目标：证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

最终从新的 `/tmp/sm110a-phase0-final.*` 构建目录重新 configure、build 和运行 6 个 host
CTest；随后单独重跑固定源码正向校验、56 个故障注入用例、2 个辅助用例和
`git diff --check`。阶段门禁记录为：

```text
positive_validation: PASS
deliberate_breakage: PASS
adversarial_audit: PASS
full_phase_rerun: PASS
unexplained_failures: 0
omissions: 0
evidence_promotions: 0
```

阶段 0 的结论止于分母和证据入口已经闭合。59 个 Tag 中仍有 53 个 `NOT_CHECKED`，另外 6 个
仍是历史快照；11 个 Auto 控制项也仍是 `NOT_CHECKED`。阶段 1 将先建立函数唯一归属、完整
result schema、fingerprint、artifact manifest 和 resume 门禁，再 fresh 重放历史实例。本报告
没有产生 Thor launch、数值正确性或性能证据。

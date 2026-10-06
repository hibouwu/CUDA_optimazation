# 旧计算家族 B 阶段校准接口草案

状态：待独立 A 审查与实现。本补充不修改已审 INTERFACE.md，不表示 `run_suite.py` 当前已经支持动态校准。当前只有固定迭代家族可运行；未知 adapter 或未实现校准须拒绝。

旧 FMA/mma.sync/WGMMA 的 sustained 点保留其原有校准语义：8192 次 pilot，取 CUDA event 毫秒数，`int(8192 * 100 / max(event_ms, 0.01))` 后夹紧至各 case 的冻结最小/最大迭代数。FMA 上限 1048576，矩阵上限 65536；取整规则名为 `legacy_event_ms_truncate_then_clamp_v1`。具体点、上限和计量仍由各家族 A 合同与独立 adapter 校验。

case 的 `iteration_policy` 使用 `kind="calibrated"`、`pilot_iterations`、`target_ns`、`min_iterations`、`max_iterations`、`rounding_model`；pilot 最短计时保护为 10000 ns。adapter 的 `resolve_iterations(pilot_raw, case, device)` 是纯函数，返回解析后的整数及审查说明，不自行运行进程。总控只在 B preflight 进行受 UUID 锁保护的有界 pilot，原始输出、进程收据、编译 binary 与设备身份全部保留。

`resolved_cases.json` 包含 `schema_version=2`、`family`、`phase="B-calibration"` 和 `cases`。每项记录 `case_id`、`pilot_iterations`、`pilot_event_ms`、`resolved_iterations`、`legacy_observed_iterations`、`rounding_model`、`binary_sha256`、`contract_sha256`、`device_identity`、`pilot_raw={path,sha256}`、`pilot_receipt={path,sha256}`。pilot 路径相对校准归档根，禁止逃逸路径和符号链接。

正式运行必须显式导入被 family B 独立 review 直接绑定的 resolved 文件、pilot raw 与收据；这不是将 pilot 的临时统计当正式样本。正式 run 冻结 resolved 文件和证据，并在自己的运行规格中固定每个 case 的迭代数。10 个进程、全部重测批和 resume 不重新校准，不把 8192 占位数自动当正式长度。修改迭代数、规则或 binary 必须另起 run 并重新取得相应 B 证据。

`compute_empty_window_v1` 家族控制点可显式使用 0 次迭代；这项例外只由相应 adapter 允许，不能推广到普通工作量探针。`sms_capped_occupancy` 使用实际查询的 occupancy 及冻结规则选择 grid；不能在 CPU plan 阶段猜设备驻留数，也不能默默等价成固定 4 CTA/SM。

现有 S04/S09 固定迭代数不经过该钩子。校准实现、resolved 导入的身份绑定、负例以及实际 B 设备验证全部通过独立审查后，才可注册相应动态 adapter 并授权正式采样。

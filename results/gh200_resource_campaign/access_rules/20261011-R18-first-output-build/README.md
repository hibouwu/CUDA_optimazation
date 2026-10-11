# cfg_a首输出五切点构建

固定logical=960x2816x4096，storage_m=1024、zero_m=960、input_map_m=960/1024、stage6、swizzle1。独立prepare入口生成公共R18基线的派生源码，旧默认runner/probe保持原样。选定12CTA×2WG leader，新增commit0/16/32/63与mma-return五个cycle/ns戳；局部值延迟至工作循环结束后写回。每CTA追加22word（两角色各1marker+10timestamp），完整清零容量1391808bytes，额外23232bytes，其中有效新时间戳载荷1920bytes。

off/on同一二进制，16次同二进制前序调用固定off，目标才切flag。decoder区分off零记录与on完整24角色，拒绝遗漏、非目标写入、布局/SM/坐标/原生uint64/源码顺序错误，始终保留observation_qualified=false。check-first-output模式在计时外完整比较off/on输出；尚未执行。

11项CPU测试通过。派生源码与原overlay的release/wait/commit/syncthreads/mma_tail调用数完全相同，详见source-port-audit.json；不是SASS或数值证明。此次仅申请目标工具链CPU构建，无GPU申请和计时样本。SASS资源与控制流、数值及独立扰动pilot仍需后续验证；具体ns门槛必须在目标采样前由独立控制冻结。

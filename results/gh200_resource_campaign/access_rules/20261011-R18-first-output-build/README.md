# cfg_a首输出五切点构建

固定logical=960x2816x4096，storage_m=1024、zero_m=960、input_map_m=960/1024、stage6、swizzle1。独立prepare入口生成公共R18基线的派生源码，旧默认runner/probe保持原样。选定12CTA×2WG leader，新增commit0/16/32/63与mma-return五个cycle/ns戳；局部值延迟至工作循环结束后写回。每CTA追加22word（两角色各1marker+10timestamp），完整清零容量1391808bytes，额外23232bytes，其中有效新时间戳载荷1920bytes。

off/on同一二进制，16次同二进制前序调用固定off，目标才切flag。decoder区分off零记录与on完整24角色，拒绝遗漏、非目标写入、布局/SM/坐标/原生uint64/源码顺序错误，始终保留observation_qualified=false。check-first-output模式在计时外完整比较off/on输出；尚未执行。

11项CPU测试通过。派生源码与原overlay的release/wait/commit/syncthreads/mma_tail调用数完全相同，详见source-port-audit.json；不是SASS或数值证明。此次仅申请目标工具链CPU构建，无GPU申请和计时样本。SASS资源与控制流、数值及独立扰动pilot仍需后续验证；具体ns门槛必须在目标采样前由独立控制冻结。

740064首次CPU构建在头文件重复定义处失败，无kernel执行；attempts/740064保存源码清单、冻结身份与日志。revision2仅统一扩展trace头文件及overlay转发入口后重新构建。


740065在a057用CUDA12.9完成CPU构建，34秒、4CPU/16GiB，无GPU。目标kernel168寄存器/2barrier，零stack/spill。与原dual的HGMMA、TMA load/store、arrival及wait静态数量相同。五处新读取PC、原wait/release与延迟写回记录于sass-audit.json：commit0为1e20/1e30，commit16为2190/21a0，commit32为2240/2250，commit63为22f0/2300，MMA_RETURN为24b0/24c0；wait1=2340、release=23e0、wait0=24f0，十个寄存器戳写回在3eb0..3f50。静态位置不证明硬件读取包围或扰动合格。完整数值和独立控制pilot仍未运行。

C的完整调用/配对接口4dd64fd已合入c8f6f76；合并后16项first-output测试通过，保留全部24角色、负分段差及跨调用SM改变。source_ports的原静态矩阵不变。

C独立源码审查发现740065主入口未放行check-first-output；主集成另发现check-output导出限制同样遗漏。该binary不用于后续数值检查，完整构建移至attempts/740065。revision3修复两处host入口并增加生成后可达性检查，GPU kernel源码不变。

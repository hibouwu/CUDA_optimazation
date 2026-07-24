# GMEM/DRAM 设计对抗式审查

## 攻击点

1. **工作集不够大导致 L2 hit。**

   设计响应：正式 baseline 使用 256 MiB，远大于 32 MiB L2，并报告 64/128/256/512 MiB 容量 sweep。NCU 必须检查 LTS miss-sector 或 DRAM bytes。

2. **L1 影响读结果。**

   设计响应：所有 global load/store 使用 `.cg`，SASS 必须显示 `STRONG.GPU` 128-bit op。

3. **写路径可能只停在 L2。**

   设计响应：write/copy stop 前执行 `__threadfence()`，并用大工作集制造 dirty eviction。若无 `dram__bytes*`，只能以 LTS write miss/sector 作为 DRAM-path proxy，不能声称直接 DRAM byte counter 已验证。

4. **多 wave 计时高估。**

   设计响应：使用 occupancy 检查，默认 blocks/SM=4 不超过可驻留上限，grid 为单 resident wave。

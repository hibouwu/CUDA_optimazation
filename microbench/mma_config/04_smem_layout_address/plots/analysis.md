# 04_smem_layout_address 分析

## 观察

- 有效 cases: 54
- 无效 cases: 90
- 最快 median cycles case: `layout_fp16_m128n256k16_sw32_off0` = 3387342.000 cycles
- 最高 TFLOP/s case: `layout_fp16_m128n256k16_sw32_off0` = 39.940304
- 本次运行中的 valid SMEM base offsets: 0, 128, 256.
- invalid SMEM base offsets/descriptors 已隔离到 `invalid_cases.csv`: 0, 16, 32, 64, 128, 256.
- invalid reason counts: max_abs_error>0.05:9, max_abs_error>0.25:9, misaligned address:72

## 推断

- row 只报告软件可见行为。`pending_mbarriers` 被视为累计 completion-prefix tracking，而不是独立 async group queue。
- 如存在 effective SMEM rate，它只表示 collector-discard 条件下 logical operand bytes / measured cycle，不是物理 port width。

## 不支持的说法

- 这些结果不能识别物理 SMEM bank count、物理 TMEM bank width 或 hidden collector depth。

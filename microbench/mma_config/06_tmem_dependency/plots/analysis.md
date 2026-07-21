# 06_tmem_dependency 分析

## 静态重新标定（主证据）

- CSV: `plots/static_tmem_benchmark.csv`.
- 图: `plots/static_tmem_dependency.svg`.
- aggregate rows: 8 valid, 0 invalid.
- 固定控制变量：static kernel、Q16、collector discard、same operand address、wait_hint=0、full-grid launch blocks。

BF16 观察：

| Shape | same/input_d=0 | same/input_d=1 | ring/input_d=0 | ring/input_d=1 |
| --- | ---: | ---: | ---: | ---: |
| N128 | 86.723 | 86.799 | 86.796 | 86.721 |
| N256 | 154.828 | 154.771 | 154.746 | 154.758 |

## 推断

- same-D、合法 D-ring、`input_d=0` 和 `input_d=1` 在当前 Q16 静态窗口中不可区分。
- 这不证明 TMEM dependency tracking 免费，只说明在 wait、dispatch 和 descriptor work 被控制后，本测试没有暴露可测成本。

## 不支持的说法

- 这些 row 不能识别 TMEM bank count、bank width、write bandwidth 或 hidden dependency scoreboard size。

## 旧 Runtime-Dispatch 负控制

旧 TMEM dependency sweep 保留用于审计，但它的 timed loop 混入 D address arithmetic 和 runtime control。dependency 结论应使用静态 row。

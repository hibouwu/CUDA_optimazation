# 01_collector_protocol 分析

## 静态重新标定（主证据）

- CSV: `plots/static_collector_benchmark.csv`.
- 图: `plots/static_collector_protocol.svg`.
- SASS dump: `../02_latency_throughput/plots/static_sass/`.
- aggregate rows: 32 valid, 0 invalid.
- 固定控制变量：Q16、合法 D-ring、`input_d=1`、wait_hint=0、full-grid launch blocks、相同总 MMA 指令数。

BF16 观察：

| Shape | discard | 成对 fill/lastuse | fill/use 范围 |
| --- | ---: | ---: | ---: |
| N128 | 86.752-86.778 | 85.471-85.506 | 89.683-89.777 |
| N256 | 154.707-154.718 | 150.838-150.867 | 148.616-148.654 |

## 推断

- ISA-visible collector protocol 在固定 Q 和固定 D dependency 下会改变性能，因此 collector path 是可观察的。
- 成对 fill/lastuse、fill/use/lastuse 和 fill/use/discard 是静态 runner 中不同的 SASS-level protocol。它们的符号随 shape 改变，所以不能简化为“collector 一定更快”或“collector 一定更慢”。
- 当前固定 Q 数据不能识别 hidden collector depth，只能报告 Q16 窗口中的 protocol-specific cost。

## 不支持的说法

- 不能从这些 row 推断 hidden collector entry count 或 depth。
- 除非已经排除 dispatch、wait 和 D dependency，否则不能把 batch-size throughput kink 称为 collector depth。

## 旧 Runtime-Dispatch 负控制

旧 `benchmark_results.csv` 保留用于审计，但它使用了不同 Q 和 runtime-dispatch kernel。其 discard/fill/use 差异不用于 collector-depth 推断。

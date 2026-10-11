# 主线复核 R09 历史预测基线

A专属提交已串行合入至f950ca5，63项R09测试通过。主线独立执行r09_history_forecast.py，80进程、2320个event预测和601920个CTA/原角色分解完整复现。history-forecast.json和call-forecasts.csv字节相同，role-forecasts.csv.gz解压后的86,515,980字节内容也完全相同。

| 保持紧前调用的条件 | event绝对误差中位数 | 最大绝对误差 | RMS |
|---|---:|---:|---:|
| r09_trajectory_single_sw1 | 2.350% | 5.623% | 2.871% |
| r09_trajectory_series_sw1 | 2.020% | 10.043% | 3.338% |
| r09_trajectory_single_sw8 | 23.949% | 39.086% | 28.197% |
| r09_trajectory_series_sw8 | 0.440% | 40.620% | 6.503% |
| r09_source_same_slot0 | 2.103% | 15.384% | 3.625% |
| r09_source_mirror_slot0 | 2.341% | 11.624% | 3.366% |
| r09_source_same_slot1 | 2.444% | 15.132% | 3.998% |
| r09_source_mirror_slot1 | 1.960% | 17.625% | 3.668% |

两种基线、全部首个切换和后续目标均保留。局部角色checkpoint只在批末回传，不能称为当前协议已支持的在线输入；event在下一调用前可用，也不使其变成纯静态预测。此为开发基线失败界限，不更新时钟价、Q/S或V09冻结。

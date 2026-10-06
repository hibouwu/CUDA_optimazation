# S10 全配置短正确性接口（草案）

本接口把已实现探针的计时外检查拆成独立入口。原 22 个正式配置、计量公式和目标指令不变；整个家族短检查通过并独立审查后，才能进行预热和正式采样。本文及 [profile 草案](../../../../../microbench/gh200_resource_campaign/contracts/matrix_exchange_validation_profiles_v1.draft.json) 尚未授予运行资格。

## 有限启动集合

| 适用配置 | profile | 每进程 target 启动 | 实际 kernel 参数 |
|---|---|---:|---|
| 18 个 load/store/roundtrip，x1/x2/x4、normal/trans | `matrix_short_1_2_v1` | 2 | `iterations=1,2`；每轮仍含原 8 个矩阵访问位置 |
| 4 个 shuffle，32/128 线程、1/4 streams | `shuffle_short_1_8_33_v1` | 3 | `steps=1,8,33`，直接传给 kernel，不再乘 8 |

顶层 seed 固定为 3，保持当前可识别坐标的输入公式。`target_iterations` 是实际传入 kernel 的计数；shuffle 的计数单位是交换步数，与正式入口把 8192 乘 8 的转换明确区分。1 步检查尾部，8 步执行一次完整展开循环，33 步检查跨越 32 步轮转后的尾部。只检查 1/33 步可能遗漏展开循环中的排列错误，因此保留 8 步对照。

每个配置独立进程，不在一个进程内先预热再检查别的配置。家族总计 22 个进程、48 次 target 启动，无辅助 kernel。matrix 与 shuffle profile 不可互换；不运行两个 profile 的笛卡尔积。

## 输出和完成条件

matrix 每次保存 `32+256×matrices+2048` 个 uint32：32 个 lane 的校验和、全部片段、完整 padded 输出区域。store 的零校验和属于完成检查，不代表读取了矩阵。shuffle 保存 `threads×streams` 个 uint32，逐 warp、lane、stream 比较，不只比较总和。

启动前逐元素以预期输出的按位反相初始化目的数组；stamp 全部置为未写入标记。目标完成、事件同步、全部 stamp 改写及所有输出比较必须成功。短窗口可为零，不要求正的吞吐率或访问所有物理 SM。实际 CUDA 失败、未完成输出或数值错误停止余下短启动并保留已有证据。

每次启动保存独立的 little-endian uint32 文件，记录相对路径、shape 和 SHA256。现有 ABI 的通用检查只证明文件完整性；S10 的独立结果审查还须读取全部值，使用冻结的 Python 坐标参考重算。保存完整值可以让后续参考修正先进行离线复核。单个文件最多 3104 个 word，这里无需用抽样或仅错误计数压缩证据。

profile 的 reference identity 指向独立 Python 参考。实际 C++ `mx_reference` 仍需在 source-B 中逐坐标与其比较；声明同一模型名称不能替代这个检查。设备源和编译结果由独立 B2 绑定，新入口只能改变 host 分流、poison 和工件记录；若实际设备函数变化，必须审查其影响。

## 入口与恢复

沿用 `validate_suite.py` 的 case-only 入口和三函数 adapter ABI，计划增加 `validate-only CASE PROFILE SEED` host 分支。正式旧入口继续保留其原检查与实验含义，独立短入口不能落入预热或测量分支。

适配器拒绝未知 case/profile、错误计数单位、非固定 seed、缺 lane/片段/输出 padding、重复工件路径和输出字节数不符。正式范围内的寄存器、静态/动态 SMEM 和 occupancy 保留完整身份。先完成全部 22 点的短结果及完整值重算，再通过独立 B3 汇总授予后续 B4 资格。

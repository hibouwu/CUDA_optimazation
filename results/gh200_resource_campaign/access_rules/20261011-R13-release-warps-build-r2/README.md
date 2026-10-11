# R13 四 warp release 观察器构建

目标：cfg_b、2816×1536×4096、stage 4/6；只准备同一二进制的 runtime off/on。CTA0 第二输出 global group 80 的 release 在 local k17 记录，warp leader 为 256/288/320/352。新增 clock64 对在 consumer 工作循环结束后写入 words 40:47，所有原始 consumer_release、warpgroup_wait 和 commit 调用保留。

协议 `r13-release-warps-v1`，magic `0x52313307`。原单 leader 对保留在 words 19:20，并要求与 40:41 精确一致。解析拒绝缺失、倒置、非 uint64、其他 CTA 写入、错误 slot/phase、关闭时额外 warp 非零及 producer 先于必要 arrival 的记录。预热开关恒为 0，目标开关由 setup/call/清单共同核对。

740051 成功编译四个二进制，退出码 0，耗时 131 秒。仅申请 CPU 4 / 16 GiB，不申请 GPU，使用 CUDA 12.9。`input.tar.gz` 每个路径只打包一次，overlay trace header 转发到同一 probe header。`source_hashes.json`、`build/commands.json` 和 `command.json` 记录输入。

`check-release-warps` 入口对 runtime off/on 的全部 4,325,376 个 FP32 输出做逐位比较；尚未在 GPU 执行。它自身也不代替与 plain 的全 D 对照。

本目录只证明准备与编译。SASS 必须另核对每 warp 的 clock/arrival 顺序及全 active lane 覆盖；尚无数值正确性、扰动门槛或正式 40 进程测量结果。时钟对只可约束经核验的 arrival 发出区间，不等于远端 barrier 可见时延。正式清单仍须冻结预热、配对顺序、二进制 SHA、同卡参照和 B 已声明的绝对差门槛。

26 项针对性 CPU 检查通过。SASS 初查定位到 local k==17 附近的 release 读取与非谓词 arrival（stage4 0x2240/0x22a0/0x22b0，stage6 arrival 0x2330、post 0x2340）；完整开关、lane 谓词及 active mask 追踪仍未完成，不据此宣称四 warp 覆盖已合格。二进制与 SASS SHA 见 `build-summary.json`。

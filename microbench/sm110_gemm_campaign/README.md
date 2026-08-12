# Thor/SM110 GEMM 上限模型硬件回传入口

这个目录用于 Git 往返式硬件采集：模型端提交 runner，Thor 端 pull 后运行，
再把不可替代的硬件证据提交到一个独立结果分支。第一阶段只测 dense
`tcgen05.mma` compute-only，不把它误称为完整 GEMM。当前 run contract 固定为
T5000 的 20-SM 配置；full-SM case 会回读每个 block 的 `%smid`，只有 20 个
block 覆盖 20 个不同 SM 才接收其全 GPU work accounting。

## Thor 端命令

从仓库根目录运行：

```bash
git fetch origin
git switch -c thor-results/<run-id> origin/codex/thor-sm110-gemm-bounds
bash microbench/sm110_gemm_campaign/launch_compute_campaign.sh <run-id>
```

运行前请先把 Thor 切到 `MAXN`（具体 `nvpmodel` mode ID 以该机器配置为准）并确认
`nvpmodel -q` 的输出包含 `MAXN`。runner 会保存这段输出，回传审计在无法证明
MAXN 时 fail closed，因为文档中的 1035/517 TFLOP/s 规格只在该功耗合同下比较。

`<run-id>` 必须稳定且只包含字母、数字、点、下划线或连字符，例如
`thor-t5000-maxn-20260812-a`。如果进程中断，原命令重跑即可：已通过且 fingerprint
一致的 case 会跳过；不一致的 case 会被拒绝，避免混入旧 binary 或旧配置。

launcher 会立刻返回，后台 runner 的 PID 和日志分别写入 `launcher.pid` 和
`launcher.log`。查看进度：

```bash
tail -f results/sm110_gemm_campaign/<run-id>/launcher.log
cat results/sm110_gemm_campaign/<run-id>/campaign_status.json
```

如果进程中断，重复同一条 launch 命令即可安全续跑。不要换 `run-id`，也不要修改
`--trials` 或 `--iters`。

基础吞吐闭环不强制 NCU。如果确认 counter 权限可用，可在基础 campaign 完成后
用同一个 `run-id` 增补 NCU：

```bash
bash microbench/sm110_gemm_campaign/launch_compute_campaign.sh <run-id> --ncu
```

`--ncu` 只采集每种精度一个 full-SM M128N256 case，使用最小指标集合，避免生成
不必要的大型报告；counter 权限或 metric 错误会 fail closed。若 NCU 不可用，
重新运行不带 `--ncu` 的基础命令即可恢复完整的 compute-only 结果。

只有后台进程退出且以下审计通过，才提交结果：

```bash
python3 microbench/sm110_gemm_campaign/audit_campaign.py \
  results/sm110_gemm_campaign/<run-id>
git add -f results/sm110_gemm_campaign/<run-id>
git commit -m "results: Thor SM110 GEMM campaign <run-id>"
git push -u origin thor-results/<run-id>
```

若增补 NCU 成功，把审计命令加上 `--require-ncu`。把最后的结果分支名返回给
模型端即可。runner 在记录 binary SHA-256 后移除 binary，
因此回传内容包含可复现源码、精确编译命令、SASS 与 hash，但不会把 72 个可执行文件
塞进 Git 历史。

## 返回内容

结果位于 `results/sm110_gemm_campaign/<run-id>/`，包括：

- `run_spec.json`：不可变运行合同和 case manifest；
- `environment.json`：首次启动的 GPU、CUDA、driver、频率、电源模式和 Git 状态；
- `environment_snapshots.jsonl`：每次安全续跑追加的环境快照；审计要求 GPU identity
  一致且所有快照都证明 MAXN；
- `cases/<case-id>/source.cu`：实际编译源码；
- `cases/<case-id>/descriptor.json`：官方 PTX ISA 字段与最终 idesc；
- `cases/<case-id>/compile_command.json`、`compile.log`、`sass.txt`、
  `binary.sha256`；
- `cases/<case-id>/trials.jsonl`：每次原始运行输出；
- `cases/<case-id>/result.json`：median/min/max 和静态审计；
- 可选 `ncu/` 目录；
- `summary.json` 与 `COMPLETE`：只有所有 case 通过才生成完成标志。

这一批结果只能校准 Tensor Core compute resource。TMA、TMEM readback、epilogue
和完整 GEMM correctness 会使用后续独立 campaign；不能用本批结果替代。

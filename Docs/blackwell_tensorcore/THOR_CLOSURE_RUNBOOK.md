# Thor/SM110 GEMM closure 运行手册

本文档与 closure runner、独立 auditor 和模型导入器在同一 Git 提交中维护。
不要从聊天记录复制旧的 commit hash；脚本会直接把当前 `HEAD` 冻结到
`run_contract.json`，并把同一个值传给所有 runner。

## 1. 更新到待测提交

在 Thor 仓库根目录执行：

```bash
git fetch origin
git switch codex/thor-sm110-gemm-bounds-v2
git pull --ff-only
git status --short --untracked-files=no
git rev-parse HEAD
```

最后一条命令显示的提交就是本次唯一允许的 `EXPECTED_COMMIT`。不要继续使用
shell 中遗留的同名变量。

## 2. 配置并验证 Thor

正式数据必须在 MAXN、GPU 1.575 GHz 锁频和 performance governor 下采集：

```bash
sudo /usr/sbin/nvpmodel -m 0
sudo /usr/bin/jetson_clocks
/usr/sbin/nvpmodel -q
sudo /usr/bin/jetson_clocks --show
```

`nvpmodel -q` 必须显示 `NV Power Mode: MAXN`；`jetson_clocks --show` 必须显示
`gpu-gpc-0 MinFreq=1575000000 MaxFreq=1575000000 CurrentFreq=1575000000`。
若机器刚发生 GPU fence hang、bounded termination failure 或需要 reboot 才恢复的
驱动异常，先重启，不要在异常状态下继续采数。

## 3. 启动新的 evidence suite

每次都使用从未出现过的新 `SUITE_ID`。失败目录也不能复用；这是为了避免旧
`environment.json`、旧 OC 基线或旧 summary 被误当成本次证据。

```bash
SUITE_ID=thor-t5000-closure-maxn-YYYYMMDD-a
bash microbench/sm110_closure_campaign.sh start "$SUITE_ID"
```

`start` 会机械执行以下门禁：

- 分支必须是 `codex/thor-sm110-gemm-bounds-v2`；
- tracked worktree 必须干净；
- 从当前 `HEAD` 生成本次 commit 合同；
- MAXN、1.575 GHz min/max/current 和 performance governor 必须同时成立；
- 保存 `preflight.txt`、`oc_before.tsv` 和 `run_contract.json`；
- 使用 NCU 并以 detached 方式运行 bounded epilogue preflight、compute、component
  和 full-GEMM，三批严格串行。

启动命令返回后可以关闭 SSH。不要对 campaign PID 发送 `Ctrl-C`。

## 4. 查看状态

一次性查看：

```bash
bash microbench/sm110_closure_campaign.sh status "$SUITE_ID"
```

持续查看：

```bash
watch -n 10 bash microbench/sm110_closure_campaign.sh status "$SUITE_ID"
```

也可以只跟踪日志：

```bash
tail -f "results/sm110_closure_suite/$SUITE_ID/suite_launcher.log"
```

退出 `watch` 或 `tail -f` 时按 `Ctrl-C` 只会退出查看工具，不会中止 detached
suite。只有日志出现独立一行 `SUITE_COMPLETE` 才能进入收尾。

## 5. 收尾、独立审计和模型导入

```bash
bash microbench/sm110_closure_campaign.sh finish "$SUITE_ID"
```

`finish` 会：

1. 再次验证 checkout 与 `run_contract.json` 中的 commit 一致；
2. 保存不可覆盖的 `oc_after.tsv`；
3. 重新运行 compute、component、full-GEMM 三个独立 auditor；
4. 要求 compute NCU、full-GEMM NCU、三个 epilogue profile、全部 raw trial、hash、
   SASS、环境和完成标志通过；
5. 生成
   `results/sm110_model_closure/$SUITE_ID/model_inputs.json`；
6. 运行模型 provenance audit 和 precision/resource coverage audit；
7. 保存关键 artifact 的 SHA-256 清单。

`finish` 是可恢复的：一旦 `oc_after.tsv` 已保存就不会覆盖它；若导入成功但后续
coverage 输出中断，再次执行会复用同一份不可变 counter 和 `model_inputs.json`。

若任一 overcurrent counter 增长，导入器会把增量作为 warning 和平台运行条件
保存；它描述 MAXN 下可能存在的功耗限制，但不单独否定已经通过数值、计时和
artifact 审计的结果。counter 倒退表示证据区间内发生过重置或重启，导入失败。

## 6. 回传结果

只有 `finish` 成功后才创建结果分支：

```bash
RESULT_BRANCH="thor-results/$SUITE_ID"
git switch -c "$RESULT_BRANCH"
git add -f \
  "results/sm110_closure_suite/$SUITE_ID" \
  "results/sm110_epilogue_probe/$SUITE_ID-epilogue-preflight" \
  "results/sm110_gemm_campaign/$SUITE_ID-compute" \
  "results/sm110_gemm_component_campaign/$SUITE_ID-components" \
  "results/sm110_full_gemm_campaign/$SUITE_ID-full" \
  "results/sm110_model_closure/$SUITE_ID"
git commit -m "results: Thor SM110 GEMM closure $SUITE_ID"
git push -u origin "$RESULT_BRANCH"
```

然后返回结果分支名和 commit hash。不要把结果提交到代码分支。

## 7. 恢复默认设置

全部 GPU 采数和 `finish` 完成后就不再需要 MAXN/锁频，可以恢复。该 Thor 在本次
campaign 前记录的默认功耗模式是 120W、mode 1：

先切回 mode 1，然后验证：

```bash
sudo /usr/sbin/nvpmodel -m 1
/usr/sbin/nvpmodel -q
sudo /usr/bin/jetson_clocks --show
```

应看到 120W mode，而不是 MAXN；GPU min/max 不应继续都固定为 1.575 GHz。
只有在切换 mode 后 GPU 时钟仍被 `jetson_clocks` 锁定、且没有可用的原配置
store 文件时，再执行 `sudo reboot`。
若运行前曾使用 `jetson_clocks --store` 保存过原配置，也可按该工具的
`--restore` 合同恢复，但不要在没有对应 store 文件时假定 restore 成功。

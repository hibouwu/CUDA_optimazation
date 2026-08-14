# Thor/SM110 GEMM closure 运行手册

本文档与 closure runner、独立 auditor 和模型导入器在同一 Git 提交中维护。
不要从聊天记录复制旧的 commit hash；脚本会直接把当前 `HEAD` 冻结到
`run_contract.json`，并把同一个值传给本次启动的 runner。第 8 节的增量路径会
另外保留基础 suite 的旧 commit；两者分别校验，绝不覆盖成同一个值。

## 0. 提交与执行指令必须成对

凡是会改变 Thor 采集程序、case 合同、审计器、模型导入或最终报告含义的提交，
都必须在**同一个 commit** 中同步更新本手册，给出针对该提交可直接复制执行的：

- 分支和 `HEAD` 校验；
- 从未使用过的 `SUITE_ID` 命名；
- 启动、一次性状态、持续状态和日志命令；
- `finish`、独立审计、结果提交与回传命令；
- 全部采集结束后的平台恢复命令。

发布此类提交时，交付消息还必须明确给出精确 commit hash，并说明该提交需要
“完整重跑”“仅补跑某个 campaign”或“无需 Thor 重跑”。不能先提交代码、再仅在
聊天中补充运行指令。纯文档、测试或不改变任何 Thor 证据合同的提交，也必须在
交付消息中明确写出“无需 Thor 重跑”及其理由。

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

总时长依赖 NCU 启动开销和当时机器状态，不能由 case 数简单线性外推。此前同一
Thor 的 `thor-t5000-closure-maxn-20260812-b-compute` 状态回传显示 72 个 compute
case 用时约 4 分 18 秒；这只是操作估计，不是本轮资格证据。完整 suite 通常应按
“几十分钟”而不是数小时预留。每个普通 trial 的硬超时是 120 秒，NCU case 是
300 秒；若某个 `current_case` 超过其合同仍没有更新，应查看日志，不能继续用
“平均每 case”估计剩余时间。

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
7. 生成 36 个 shape-qualified compute capacity、18 个 component capacity、15 个
   candidate/reference 对比、条件上界检查和 N=4096 holdout 经验预测偏差；
8. 保存关键 artifact 的 SHA-256 清单。

`finish` 是可恢复的：一旦 `oc_after.tsv` 已保存就不会覆盖它；若导入成功但后续
coverage 输出中断，再次执行会复用同一份不可变 counter 和 `model_inputs.json`。
数值报告位于同一目录下的 `closure_analysis.json` 和 `closure_summary.md`。

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

## 8. 已有基础 suite 时只补跑 component

本节**不适用于基础 suite 尚未启动的情况**。如果旧提交曾因 `wrong checkout` 在
启动门禁处退出、没有形成完整基础证据，应回到第 1–7 节，在当前提交上完整运行
closure suite；不要人为制造跨提交的组合证据。

若一个旧提交已经完整取得 epilogue preflight、36 个 compute capacity、15 个
full-GEMM observation 和对应 NCU，而新提交只改变 component microbenchmark、模型
工作量或导入/报告代码，则不需要重复 compute 和 full-GEMM。使用
`sm110_component_supplement.sh` 采集新的 18-case component campaign；组合导入器会：

- 以 `BASE_EXPECTED_COMMIT` 重新审计基础 compute/full 的环境、源码、生成器、
  二进制、SASS、NCU 和平台区间；
- 以当前提交重新审计 component 的独立环境、源码、二进制、SASS、数值和平台
  区间；
- 在 `campaign_sources` 中保留两个实际 commit 和各自提供的证据，不把两段运行
  伪装成单提交 suite；
- 用新的 `SUPPLEMENT_ID` 生成一套无重复 capacity/observation ID。

只有基础 suite 已经执行 `finish`、存在不可变 `oc_after.tsv` 且日志包含
`SUITE_COMPLETE` 时才能使用此路径。若 compute/full runner、源依赖或支持 manifest
已经改变，它们的独立 auditor 会因哈希不一致拒绝复用。

在 Thor 代码仓库根目录设置三个不会从旧 shell 误继承的值：

```bash
unset BASE_SUITE_ID BASE_EXPECTED_COMMIT SUPPLEMENT_ID EXPECTED_COMMIT
BASE_SUITE_ID=<已完成基础suite的ID>
BASE_EXPECTED_COMMIT=<基础suite的40位commit>
SUPPLEMENT_ID=thor-t5000-component-supplement-maxn-YYYYMMDD-a
EXPECTED_COMMIT=$(git rev-parse HEAD)

test "$(git branch --show-current)" = "codex/thor-sm110-gemm-bounds-v2"
test "$(git status --short --untracked-files=no)" = ""
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test ! -e "results/sm110_closure_suite/$SUPPLEMENT_ID"
test ! -e "results/sm110_gemm_component_campaign/$SUPPLEMENT_ID-components"
```

配置 MAXN/锁频后启动：

```bash
sudo /usr/sbin/nvpmodel -m 0
sudo /usr/bin/jetson_clocks
bash microbench/sm110_component_supplement.sh start \
  "$SUPPLEMENT_ID" "$BASE_SUITE_ID" "$BASE_EXPECTED_COMMIT"
```

一次性查看、持续查看和只看日志分别为：

```bash
bash microbench/sm110_component_supplement.sh status "$SUPPLEMENT_ID"
watch -n 10 bash microbench/sm110_component_supplement.sh status "$SUPPLEMENT_ID"
tail -f "results/sm110_closure_suite/$SUPPLEMENT_ID/supplement_launcher.log"
```

退出 `watch` 或 `tail -f` 的 `Ctrl-C` 不会中止 detached supervisor。18 个 case 各
10 次，正常运行应按数分钟预留；每个 trial 的独立硬超时仍是 120 秒。只有日志
出现独立一行 `COMPONENT_SUPPLEMENT_COMPLETE` 后执行：

```bash
bash microbench/sm110_component_supplement.sh finish "$SUPPLEMENT_ID"
```

`finish` 生成组合 `model_inputs.json`、coverage 和数值报告。成功后在单独结果分支
同时提交基础证据、增量证据和组合报告，因为组合 artifact 路径需要两部分都可解析：

```bash
RESULT_BRANCH="thor-results/$SUPPLEMENT_ID"
git switch -c "$RESULT_BRANCH"
git add -f \
  "results/sm110_closure_suite/$BASE_SUITE_ID" \
  "results/sm110_epilogue_probe/$BASE_SUITE_ID-epilogue-preflight" \
  "results/sm110_gemm_campaign/$BASE_SUITE_ID-compute" \
  "results/sm110_full_gemm_campaign/$BASE_SUITE_ID-full" \
  "results/sm110_closure_suite/$SUPPLEMENT_ID" \
  "results/sm110_gemm_component_campaign/$SUPPLEMENT_ID-components" \
  "results/sm110_model_closure/$SUPPLEMENT_ID"
git commit -m "results: Thor SM110 composite closure $SUPPLEMENT_ID"
git push -u origin "$RESULT_BRANCH"
```

回传结果分支名和 commit 后，即可按第 7 节恢复 120W mode 1。后续不再需要保持
MAXN 或 GPU 锁频。

## 9. `d382b57` 已完成基础证据的定向恢复

suite `thor-t5000-closure-maxn-20260814-d382b57-a` 已在 commit
`d382b57eae289b458c5290e3d2b7e0daf1b7d7c8` 完成 epilogue preflight、compute、
旧 14-case component 和 full-GEMM，三批独立 auditor、50 capacity/15 observation
导入和 campaign coverage 均通过；`oc_after.tsv` 已冻结，日志包含
`SUITE_COMPLETE`。最终报告暴露了两个模型问题：经验 read/write 绕过共享
`hbm.total`，以及串行 TMA L2 probe 低估 tc5a 四 stage、A/B 双请求的 per-SM
ingress。前者是纯模型修复；后者需要新的 18-case component campaign。旧
component 不复用，但 compute/full 不应重跑。

拉取发布该修复的提交后，在 Thor 仓库根目录执行以下定向 supplement。当前提交
必须由 `git rev-parse HEAD` 取得，交付消息会同时给出其精确 40 位 hash：

```bash
unset BASE_SUITE_ID BASE_EXPECTED_COMMIT SUPPLEMENT_ID EXPECTED_COMMIT

BASE_SUITE_ID=thor-t5000-closure-maxn-20260814-d382b57-a
BASE_EXPECTED_COMMIT=d382b57eae289b458c5290e3d2b7e0daf1b7d7c8
SUPPLEMENT_ID=thor-t5000-tma-ingress-supplement-maxn-20260814-a
EXPECTED_COMMIT=$(git rev-parse HEAD)

test "$(git branch --show-current)" = \
  "codex/thor-sm110-gemm-bounds-v2"
test "$(git status --short --untracked-files=no)" = ""
test -f "results/sm110_closure_suite/$BASE_SUITE_ID/oc_after.tsv"
grep -x 'SUITE_COMPLETE' \
  "results/sm110_closure_suite/$BASE_SUITE_ID/suite_launcher.log"
test ! -e "results/sm110_closure_suite/$SUPPLEMENT_ID"
test ! -e \
  "results/sm110_gemm_component_campaign/$SUPPLEMENT_ID-components"
```

配置平台并启动：

```bash
sudo /usr/sbin/nvpmodel -m 0
sudo /usr/bin/jetson_clocks
/usr/sbin/nvpmodel -q
sudo /usr/bin/jetson_clocks --show

bash microbench/sm110_component_supplement.sh start \
  "$SUPPLEMENT_ID" "$BASE_SUITE_ID" "$BASE_EXPECTED_COMMIT"
```

状态、持续状态和日志：

```bash
bash microbench/sm110_component_supplement.sh status "$SUPPLEMENT_ID"
watch -n 10 bash microbench/sm110_component_supplement.sh status "$SUPPLEMENT_ID"
tail -f \
  "results/sm110_closure_suite/$SUPPLEMENT_ID/supplement_launcher.log"
```

只有日志出现独立一行 `COMPONENT_SUPPLEMENT_COMPLETE` 后执行：

```bash
bash microbench/sm110_component_supplement.sh finish "$SUPPLEMENT_ID"
```

`finish` 会用旧 commit 的 compute/full 和当前 commit 的 18 个 component
capacity 重新生成组合报告。成功后严格按第 8 节的结果分支命令提交基础证据、
supplement 证据和组合 `sm110_model_closure/$SUPPLEMENT_ID`；不要把结果提交到代码
分支。结果推送完成后按第 7 节恢复 120W mode 1。

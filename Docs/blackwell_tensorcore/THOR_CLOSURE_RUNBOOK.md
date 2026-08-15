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

本节的 `-c` supplement 替代未充分隔离的早期 `-a` 指令以及输出 schema 不完整的
`-b` 运行。`-b` 在首个 TMA case 的 fail-closed 字段校验处停止，没有产生可导入的
component 证据；必须保留其失败目录且不能复用该 ID。L2-hit TMA 容量只
启动一个 CTA，避免共享 L2 总线污染 per-SM 出口测量；正式 tc5a case 使用四个
stage 的 A=16 KiB、B=32 KiB、2D SW128、四个 48 KiB stage barrier 和八笔
在途 TMA 精确混合请求。串行和 uniform
inflight=4 结果只保留为 diagnostic resource，不能覆盖正式 tc5a 容量。

拉取发布该修复的提交后，在 Thor 仓库根目录执行以下定向 supplement。当前提交
必须由 `git rev-parse HEAD` 取得，交付消息会同时给出其精确 40 位 hash：

```bash
unset BASE_SUITE_ID BASE_EXPECTED_COMMIT SUPPLEMENT_ID EXPECTED_COMMIT

BASE_SUITE_ID=thor-t5000-closure-maxn-20260814-d382b57-a
BASE_EXPECTED_COMMIT=d382b57eae289b458c5290e3d2b7e0daf1b7d7c8
SUPPLEMENT_ID=thor-t5000-tma-ingress-supplement-maxn-20260814-c
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

## 10. 结果分支异路径复审

结果提交 `ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c` 已完成 Thor 采集，不需要再次
运行 GPU。后续 auditor 必须允许把该结果分支 checkout 到不同绝对目录，同时继续
验证原始 self-test 命令确实指向同一个 run ID 下的
`results/sm110_full_gemm_campaign/<run-id>/build/extended --self-test`。不能把原始
Thor 仓库前缀与复审 checkout 前缀作字符串相等比较。

在任意干净 checkout 中复审已有结果时执行：

```bash
BASE_SUITE_ID=thor-t5000-closure-maxn-20260814-d382b57-a
SUPPLEMENT_ID=thor-t5000-tma-ingress-supplement-maxn-20260814-c

python3 microbench/sm110_gemm_campaign/audit_campaign.py \
  "results/sm110_gemm_campaign/$BASE_SUITE_ID-compute" \
  --require-ncu
python3 microbench/sm110_gemm_component_campaign/audit_campaign.py \
  "results/sm110_gemm_component_campaign/$SUPPLEMENT_ID-components"
python3 microbench/sm110_full_gemm_campaign/audit_campaign.py \
  "results/sm110_full_gemm_campaign/$BASE_SUITE_ID-full"
sha256sum -c \
  "results/sm110_model_closure/$SUPPLEMENT_ID/artifact_sha256.txt"
```

这一提交只修复离线审计的路径可移植性并更新最终报告，不改变任何 GPU-facing
源码、case、工作量或容量语义，因此**无需 Thor 重跑**。已有 evidence 仍分别绑定
到 `d382b57eae289b458c5290e3d2b7e0daf1b7d7c8` 和
`25d8cf71fa566150b64f2eb1dc7f814ce70fa354`。

## 11. 当前严格模型的精确 TMA resource supplement

本节只补采 generic、byte-container、block-scaled 和 tc5a schedule 的精确 TMA
resource capacity。它不重跑已知的共享 L2 `1024 B/cycle` read、`512 B/cycle`
write 上限，也不重跑历史 compute/full-GEMM。它包含 54 个 case、每 case 10 个
外部 trial，以及 18 份 NCU report。它会关闭 resource-envelope 的实验输入缺口，
但不会单独关闭 causal DAG 或缺失精度的 full-GEMM/reference/denominator 缺口。

### 11.1 拉取并固定同一提交

交付消息会给出唯一的 40 位 `EXPECTED_COMMIT`。在 Thor 仓库根目录执行：

```bash
unset EXPECTED_COMMIT SUITE_ID RESULT_BRANCH
git fetch origin
git switch codex/sm110-all-precision-closure
git pull --ff-only

EXPECTED_COMMIT=<交付消息中的40位提交>
test "$(git branch --show-current)" = \
  "codex/sm110-all-precision-closure"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git status --short --untracked-files=no)" = ""
```

不要用旧 shell 中遗留的 hash，也不要在结果分支或带 tracked 修改的 checkout 上
启动。

### 11.2 配置 Thor 并启动

```bash
sudo /usr/sbin/nvpmodel -m 0
sudo /usr/bin/jetson_clocks
/usr/sbin/nvpmodel -q
sudo /usr/bin/jetson_clocks --show

SUITE_ID=thor-t5000-exact-resource-maxn-YYYYMMDD-a
test ! -e "results/sm110_resource_suite/$SUITE_ID"
test ! -e \
  "results/sm110_gemm_resource_campaign/$SUITE_ID-resources"
bash microbench/sm110_resource_supplement.sh start "$SUITE_ID"
```

`start` 会冻结 branch/commit、MAXN、GPU 1.575 GHz min/max/current、performance
governor 和 OC counter 基线，然后 detached 串行运行全部 case。每个普通 trial 的
进程组 timeout 为 120 s；NCU 为 300 s；失败会保存 timeout 证据并停止，不能把
timeout 当成低吞吐结果。supervisor 在最后一个独立 auditor 通过后立即冻结
`oc_after.tsv`，尽量避免把等待人工 `finish` 的空闲时间计入 OC 区间。

### 11.3 查看状态

一次性查看、持续查看和只看日志分别为：

```bash
bash microbench/sm110_resource_supplement.sh status "$SUITE_ID"
watch -n 10 bash microbench/sm110_resource_supplement.sh status "$SUITE_ID"
tail -f "results/sm110_resource_suite/$SUITE_ID/suite_launcher.log"
```

退出 `watch`/`tail -f` 只停止查看，不会中止 detached supervisor。只有日志中出现
独立一行 `RESOURCE_SUPPLEMENT_COMPLETE` 才能进入 finish。

若 supervisor 因 shell/主机进程异常退出、但 GPU/driver 未重启、OC counter 未
reset 且 `oc_after.tsv` 尚未生成，可在同一冻结 checkout 恢复：

```bash
bash microbench/sm110_resource_supplement.sh resume "$SUITE_ID"
```

`resume` 会再次验证 MAXN/锁频/clean commit，保存新的 preflight 与 OC snapshot，
不会重新编译首次运行冻结的 binary。runner 会先复核 source dependency、原 compile
command、binary/hash record、函数级 SASS，并重放 54 个 `--contract-only` 合同；
全部一致后才复用 fingerprint、10 个 trial 和 NCU artifact 都完整可复审的 case。
若检测到 retained artifact 改变、counter reset、已有活进程或已关闭
证据区间，它会拒绝恢复；此时必须换新 `SUITE_ID`，不能拼接跨 reboot 的区间。

### 11.4 finish 与双层独立审计

```bash
bash microbench/sm110_resource_supplement.sh finish "$SUITE_ID"
```

`finish` 不覆盖已存在的 `oc_after.tsv`。它会：

1. 核对当前 checkout 与冻结 commit；
2. 保存 OC counter 终点并拒绝 counter reset；
3. 用 campaign auditor 从冻结 Git blob 独立重建 54-case matrix；
4. 验证 540 个 raw trial、18 份 NCU、retained binary、function-scoped SASS、
   environment/progress/COMPLETE 和逐文件 SHA-256；
5. 用 platform auditor 验证 branch、MAXN、锁频和 OC interval；
6. 把非零 OC 增量保留为 warning，而不是隐藏为全绿。
7. 重新审计后生成
   `results/sm110_model_closure/$SUITE_ID/resource_capacities.json`，其中 54 个
   capacity 都保留 family、A/B packed row stride 和 hot-per-SM/cold-device
   scope；不会把实测速率提升成物理上界。

也可手工重放两层审计：

```bash
python3 microbench/sm110_gemm_resource_campaign/audit_campaign.py \
  "results/sm110_gemm_resource_campaign/$SUITE_ID-resources" \
  --require-ncu --expected-commit "$EXPECTED_COMMIT"
python3 \
  microbench/sm110_gemm_resource_campaign/audit_resource_suite.py \
  "results/sm110_resource_suite/$SUITE_ID" \
  --expected-commit "$EXPECTED_COMMIT"
```

### 11.5 提交并回传结果

只有 `finish` 成功后执行：

```bash
RESULT_BRANCH="thor-results/$SUITE_ID"
git switch -c "$RESULT_BRANCH"
git add -f \
  "results/sm110_resource_suite/$SUITE_ID" \
  "results/sm110_gemm_resource_campaign/$SUITE_ID-resources" \
  "results/sm110_model_closure/$SUITE_ID/resource_capacities.json"
git commit -m "results: Thor SM110 exact resources $SUITE_ID"
git push -u origin "$RESULT_BRANCH"
git rev-parse HEAD
```

回传 `RESULT_BRANCH`、结果 commit 和 `suite_audit.json` 的 `pass/warnings`。不要把
结果提交到代码分支。

拉回结果分支后，在 evaluate/report 命令中显式附加：

```bash
--resource-import \
  "results/sm110_model_closure/$SUITE_ID/resource_capacities.json"
```

模型只在 packed、非转置且 `K=N` 时使用本轮共同 A/B row-stride 合同：A 的
leading dimension 为 `K`，B 的 leading dimension 为 `N`。`K != N` 或 stride
不在 1024/2048/4096 时会返回 `insufficient_evidence`，不会借用邻近尺寸数据。

### 11.6 恢复默认平台设置

若还要立即执行第 12 节 causal suite，先保持 MAXN/锁频并直接进入第 12 节；两个
suite 不能并行，但可以顺序复用平台设置。只有本轮不再运行其他 GPU evidence 时，
才按本机既有默认 120W mode 1 恢复：

```bash
sudo /usr/sbin/nvpmodel -m 1
/usr/sbin/nvpmodel -q
sudo /usr/bin/jetson_clocks --show
```

应看到 120W mode，不再是 MAXN，GPU min/max 也不应继续都固定为 1.575 GHz；若
mode 切换后锁频仍残留且没有对应 `jetson_clocks --store` 快照，再重启恢复。

## 12. tc5a persistent-worker causal pipeline suite

本节分别采集与 FP16、BF16 `tc5a_m128n256k64_stage4` 完全匹配的两个因果时序
profile。定义 `precision_id` 为一份 profile 直接验证的精度标识，无单位；两个输出
必须分别是 singleton `precision_ids=["fp16_f32"]` 和
`precision_ids=["bf16_f32"]`。同一个 binary 为两种精度生成不同模板实例，并分别
冻结 tensor-map 数据类型和 MMA instruction descriptor；不能因二者都是
2 B/element 而共享时序。它不是另一个带宽峰值实验：每种精度 91 个 case 用 raw
`%globaltimer` 事件分离 TMA-only、MMA-only、joint overlap 和完整 persistent
worker 的 startup、稳态 interval、双 accumulator 复用与 readback/store drain；
每 case 10 个外部 trial。总计 182 case、1,820 条 raw trial，另有 8 份预声明 NCU
report（每种精度 4 份）。

本组与第 11 节 resource suite 共用 `results/sm110_campaign.lock`。两组必须串行：
先等一个 suite 的日志出现完成 marker 并执行 `finish`，再启动另一个。不要通过删除
lock 文件强行并行；活进程持有的是内核文件锁，删除 pathname 只会破坏证据纪律。

### 12.1 拉取并固定同一提交

在 Thor 仓库根目录执行：

```bash
unset EXPECTED_COMMIT CAUSAL_SUITE_ID CAUSAL_RUN_ID RESULT_BRANCH
git fetch origin
git switch codex/sm110-all-precision-closure
git pull --ff-only

EXPECTED_COMMIT=<交付消息中的40位提交>
test "$(git branch --show-current)" = \
  "codex/sm110-all-precision-closure"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git status --short --untracked-files=no)" = ""
```

### 12.2 配置平台并启动

如果第 11 节 resource suite 刚刚完成且尚未恢复默认设置，可复用当前 MAXN/锁频；
仍必须运行查询命令确认。否则重新配置：

```bash
sudo /usr/sbin/nvpmodel -m 0
sudo /usr/bin/jetson_clocks
/usr/sbin/nvpmodel -q
sudo /usr/bin/jetson_clocks --show

CAUSAL_SUITE_ID=thor-t5000-tc5a-causal-maxn-YYYYMMDD-a
CAUSAL_RUN_ID="$CAUSAL_SUITE_ID-causal"
test ! -e "results/sm110_causal_suite/$CAUSAL_SUITE_ID"
test ! -e "results/sm110_gemm_causal_campaign/$CAUSAL_RUN_ID"
bash microbench/sm110_causal_suite.sh start "$CAUSAL_SUITE_ID"
```

`start` 冻结 clean branch/commit、MAXN、GPU 1.575 GHz min/max/current、performance
governor 和 OC counter 基线，并 detached 启动 supervisor。普通 trial 的进程组
timeout 为 120 s，NCU 为 300 s；超时或 NCU 权限失败会保留证据并停止。

### 12.3 状态与安全恢复

```bash
bash microbench/sm110_causal_suite.sh status "$CAUSAL_SUITE_ID"
watch -n 10 bash microbench/sm110_causal_suite.sh status "$CAUSAL_SUITE_ID"
tail -f \
  "results/sm110_causal_suite/$CAUSAL_SUITE_ID/suite_launcher.log"
```

退出 `watch` 或 `tail -f` 不会停止 detached supervisor。日志只有出现独立一行
`CAUSAL_SUITE_COMPLETE` 才算 campaign、独立审计和立即 OC 终点采集全部完成。

若 supervisor 意外退出，但 GPU/driver 未重启、OC counter 未 reset 且
`oc_after.tsv` 尚未生成，可执行：

```bash
bash microbench/sm110_causal_suite.sh resume "$CAUSAL_SUITE_ID"
```

恢复不会重新编译已经冻结的 binary：两次相同 `nvcc` 调用的字节级 binary hash 不作
可复现性假设。runner 会重新核对 source/helper/manifest、原 compile command、
retained binary hash、stage-1/2/4 函数级 SASS、CSV header 和 binary 自身的 header
输出；全部一致后，才复用 fingerprint、10 条 raw trial、derived timestamp
arithmetic 和必需 NCU artifact 都可复审的 case。任一 retained artifact 改变都会
fail closed。若 counter reset、checkout 改变、已有活进程或 OC 区间已经关闭，必须
换新 suite ID。

### 12.4 finish、双层独立审计与 profile 门禁

```bash
bash microbench/sm110_causal_suite.sh finish "$CAUSAL_SUITE_ID"
```

`finish` 会验证 182-case/1,820-trial/8-NCU、binary/SASS/env/hash、不可变 Git
blob、MAXN/clock/OC interval，并为 FP16、BF16 分别重建 component linear fit 与
full-worker validation。每份 profile 的预声明门槛为：TMA/MMA/joint 三个 fit 的
决定系数均不低于 0.98，且 calibration/holdout 最大相对误差都不超过 10%。

注意两个不同结论：

- `suite_audit.json.pass=true` 表示 raw acquisition 和审计合同完整；
- `profile_qualified_by_precision.fp16_f32=true` 与
  `profile_qualified_by_precision.bf16_f32=true` 分别表示对应 fit 可用；聚合的
  `profile_qualified=true` 只在二者都通过时成立。

若第一项为 true、第二项为 false，结果仍应完整回传；auditor 会保留
对应的 `quarantined` profile 和 warning，禁止模型用它预测，但另一种精度是否可用
仍由自己的门禁决定，raw 数据也可用于分析下一版模型。不得调宽阈值后原地篡改
同一 run ID。

也可手工重放：

```bash
python3 microbench/sm110_gemm_causal_campaign/audit_campaign.py \
  "results/sm110_gemm_causal_campaign/$CAUSAL_RUN_ID" \
  --require-ncu --expected-commit "$EXPECTED_COMMIT"
python3 \
  microbench/sm110_gemm_causal_campaign/audit_causal_suite.py \
  "results/sm110_causal_suite/$CAUSAL_SUITE_ID" \
  --expected-commit "$EXPECTED_COMMIT"
```

### 12.5 提交并回传结果

只有 `finish` 成功后执行：

```bash
RESULT_BRANCH="thor-results/$CAUSAL_SUITE_ID"
git switch -c "$RESULT_BRANCH"
git add -f \
  "results/sm110_causal_suite/$CAUSAL_SUITE_ID" \
  "results/sm110_gemm_causal_campaign/$CAUSAL_RUN_ID"
git commit -m "results: Thor SM110 tc5a causal profile $CAUSAL_SUITE_ID"
git push -u origin "$RESULT_BRANCH"
git rev-parse HEAD
```

回传 `RESULT_BRANCH`、结果 commit、`suite_audit.json` 的 `pass/warnings`、
`pipeline_profiles.json` 中两份 profile 的 `qualification`，以及
`profile_qualified_by_precision` 和聚合 `profile_qualified`。结果分支拉回分析
checkout 后，模型导入命令为：

```bash
MODEL_DIR="results/sm110_model_closure/$CAUSAL_SUITE_ID"
mkdir -p "$MODEL_DIR"
python3 -m scripts.sm110_gemm_model.cli import-causal-profile \
  --repo-root . \
  --run-id "$CAUSAL_RUN_ID" \
  --expected-commit "$EXPECTED_COMMIT" \
  --output "$MODEL_DIR/pipeline_profiles.json"
```

导入器会再次运行 campaign independent auditor，并逐条验证 profile gate、validation
算术和 repository-relative artifact path；不能手工把 `pipeline_profiles.json`
中的拟合数值复制到默认 profile 文件。

### 12.6 恢复默认平台设置

如果第 11、12 节需要顺序执行，应在两组都完成并 push 后再恢复。最终执行：

```bash
sudo /usr/sbin/nvpmodel -m 1
/usr/sbin/nvpmodel -q
sudo /usr/bin/jetson_clocks --show
```

应看到本机默认 120W mode 1，且 GPU min/max 不再都固定为 1.575 GHz。

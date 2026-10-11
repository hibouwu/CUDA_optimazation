# 局部中位数等价：候选协议与判定器

本候选验证固定采样协议下两臂**记录 ns 的总体中位数之差**。它不验证逐次点值相同，也不从分项中位数相加、相除或取最大得到完整时间。关键 CTA 的逐次排序、物理时间戳准确度、完整时间和跨卡迁移继续单独验证。

状态为提案，尚未冻结执行或提交 GPU。统计判定器已实现，原始记录提取复放已执行；未来采集日志、实验单元身份及采样假设的验收仍需完成。`evaluate.py` 是数值统计内核，不是完整实验资格判定器。

## 判定

- 固定原 6372 个控制系列，off/on、off/off 两批，合计 25488 个臂中位数。主判定只用 ns；cycles 留作独立诊断。
- 候选同时覆盖率 95%，使用未插值次序统计量区间与 Bonferroni 并集界。
- 候选中位数差容差为 ±64 ns，保留原绝对量级作为待评审起点。它与原“配对绝对差中位数≤64 ns”不是同一指标，不能追认旧控制通过。
- 若每臂 64 次，取每臂第 14 至第 51 次序统计量。两臂区间相减后，**整个差值区间**在 ±64 ns 内才报告 `conditional equivalence`；整个区间落在容差外报告 `conditional non-equivalence`；其余为 `inconclusive`。无法构造所需同时覆盖区间时单列 `insufficient simultaneous coverage`。
- 原逐次配对 64/256 ns 判定同时输出，始终保留，不用中位数结论覆盖。

`conditional equivalence` 只在采样假设成立时描述记录值中位数差。判定器无论得到何种统计结果，都输出 `observation_qualified=false`、`prediction_qualified=false`；采样假设、时间戳准确度和关键 CTA 排序三个标记也固定为 false。不能靠输入一个“已批准”布尔值获得资格。

## 候选采样表

[protocol-proposal.json](data/protocol-proposal.json) 保存完整系列名单、原 GPU/binary 身份、容差与停止规则；[schedule-proposal.json](data/schedule-proposal.json) 保存 256 条命令及顺序，未执行。

64 个 block，每个 block 含四个独立进程：off/on 的两个臂、off/off 的两个臂。每进程沿用原 16 次同 buffer、关闭新增 observer 的前序，producer-tail 恒关。每个 block 内打散四臂顺序；启用 observer 的目标在四种顺序位置各出现 16 次，随机种子 20261011。这样不再将两个比较分成先后大批，但随机化本身不证明跨进程独立同分布或卡状态恒定。

固定 64 个 block，不因中间结果看起来通过而提前结束，也不因区间过宽而追加样本。出现进程、数值、设备或协议身份错误时停止采集，保留不完整数据并标记无效，不替换或过滤失败记录。64 次只是具体候选规模；它满足有限区间的条件，但没有精度或统计功效保证，不宣称足以使 ±64 ns 等价成立。

未来正式采集/提取需将以下内容一同冻结：block/臂/进程 ID 与原始文件 SHA、实际执行顺序、setup/参数身份、各 CTA 的 SM 映射、全部同调用端点及前序记录。多个坐标必须保留同一进程的联合向量。不能逐列独立重排、对不同分项挑不同进程、剔除慢窗口，或将局部中位数组装成一条不存在的执行轨迹。新采集只有身份和数量完整仍不足以建立采样假设。

## 已执行的核对

从原 40 个 raw 进程，经冻结 `analyze_cfg_c_gt_control.process` 重新校验并提取全部系列，核对 52 个输入 SHA。全部 12744 个“批次×系列”的原配对中位绝对差、最大绝对差和通过标记逐项相同：off/on 4912 条失败，off/off 4729 条失败。

将同一候选的重复数仅设为旧数据实际的 10 次作回放，12744 个系列全部返回 `insufficient simultaneous coverage`，没有任何系列被追认为等价。见 [replay-summary.json](data/replay-summary.json) 和 [完整旧数据判定](data/old-control-replay.json.gz)。这个回放不是新实验，也没有把旧批次称为随机 block。

八项针对性测试覆盖：统计等价不授观察资格、含零宽区间不能判等价、区间整体越界、局部中位数相同但旧尾差失败、缺坐标、缺重复、NaN/布尔数值，以及十次全坐标不能复用边际覆盖率。旧原始数据中所有数值检查由冻结解码器重新执行。

## 使用

在仓库根目录运行测试：

```sh
python -m unittest discover \
  -s results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-median-protocol-v1 \
  -p test_evaluate.py -v
```

重新生成候选及旧数据回放，输出目录须不存在：

```sh
python results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-median-protocol-v1/prepare_and_replay.py \
  --run results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls \
  --output /tmp/gh200-median-proposal
```

`evaluate.py` 的输入是 `{off_on: {series: {arm0: [ns…], arm1: [ns…]}}, off_off: …}`。数组按共同 block 顺序排列，配对身份必须由提取器先验证；不能仅凭数值数组证明身份。CLI 要求调用者提供协议 SHA，拒绝身份不符和覆盖已有输出。此工具不能代替上述尚未完成的执行与观测验收。

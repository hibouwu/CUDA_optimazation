# cfg_c G2 的共享参数与时钟联合区间

固定 V09-r2 的事件、输出、时钟、zero-activity mixing 和 plain transfer，只让首轮供给参数在原训练预测/分支不变的非负可行集中变化，得到 plain 完整时间区间 **[17.0990592544, 17.4428958356] µs**。上下界之比相差 **2.010851%**。这是模型内的条件参数区间，不是置信区间；原冻结 `unsupported` 保留。

| 量 | 联合下界端点 | 联合上界端点 |
|---|---:|---:|
| CTA 包络 / µs | 14.4235645675 | 14.7660527785 |
| 闭合频率 / GHz | 1.8586933211 | 1.8612801225 |
| plain 完整时间 / µs | 17.0990592544 | 17.4428958356 |
| 局部 L 最小 / ns | 9066.1157460 | 9053.8649713 |
| 局部 L 最大 / ns | 9066.1157460 | 9408.6039570 |
| 模型内达到最大时长的 CTA | 0–63 全部并列 | 62、63 并列 |

端点 CTA 集合只描述旧零 entry-skew 模型的代数并列，不是观测到的最后 CTA，也未获关键 CTA 资格。没有将两端集合并作新候选集合。

## 联合求解

条件为 `v09d_cfg_c_G2_base`，512×3968×1024、64 个 CTA，各一个 cooperative 输出，K 组数 16。全部请求只有两个特征向量；它们共用同一个五维首轮参数 `θ`，而不是逐 CTA 独立选参数。原训练约束有四个独立等式，非负与分支不等式将剩余一维自由度限制为有界线段。

单输出递推可直接写为

\[
 D_i(f,\theta)=b+(1+x_k)\left[\theta_0+16\max(1024/f,z_i^T\theta_{1:})\right],
 \qquad E(f,\theta)=\max_iD_i(f,\theta)/1000.
\]

`b` 包含冻结的 producer/prefill、w、单输出 epilogue、tail 和退出项；源程序中 `x_k=0`。供给外各项全部固定。对给定 f，E 是 θ 的若干仿射函数之最大值：最小值用一个共享 θ 的 epigraph LP 求解，最大值分别最大化有限个仿射分支后取最大。这里没有相加各 CTA 各自达到的极值。

随后分别求解 `f=F(E_min(f))` 和 `f=F(E_max(f))`。原非负时钟成本使 F 随窗口 W 非递减；每个 E 随 f 非递增，因此 `E(F(W))-W` 严格递减。所有可行 θ 的闭合 W 被两条极值包络的根夹住。两个根处的 LP 参数见证又分别在原完整预测器中重现根，故这两个界在该条件参数集中可达，不只是独立局部区间拼成的外包络。

读取的 15 个输入由冻结 manifest 直接或经 `source_hashes.json` 绑定；计算仅导入冻结 source 中的模块，不读取目标 samples。两个见证的原训练预测最大变化分别为 `9.29e-9`、`1.01e-8 ns`。这些是浮点 LP/求根检查，不宣称形式化精确算术证明。

## 复核与事后比较

[analyze.py](analyze.py) 使用 163 次 LP 完成联合求解。[verify.py](verify.py) 不调用 LP：将四个等式的零空间化为一条参数直线，逐不等式求完整可行线段，再枚举仿射分支交点与两端点求极值。两种方法所得包络端点差不超过 `2.65e-10 µs`；另有 21 组可行参数经原预测器检验落在区间内。详细参数、残差和输入 SHA 见 [report.json](data/report.json) 与 [verification.json](verification.json)。

[compare.py](compare.py) 在独立步骤读取原 job739011 的十个 plain 进程，核对预测绑定、20 个样本/原始文件 SHA、setup 与 elapsed_us 一致性，得到中位数 **17.872000115 µs**。条件预测区间的有符号误差为 **[−4.324870%, −2.400986%]**，实测中位数不在区间内，见 [comparison.json](comparison.json)。这说明仅在此等价供给参数集中选值，不能消除完整时间低估。未选择中点、回填冻结预测或修改 V09 评分。

区间没有包含其他组件参数不确定性、模型缺项、时钟读数误差、observer transfer 误差或跨卡变化。L 控制失败与关键 CTA 观测缺口均未解除；窄的完整时间区间不能替代分项资格。下一步不应为这一条件继续从等价供给参数中挑选有利值，而应定位剩余组件/模型误差，并补足有效局部观测。

## 复现

仓库根目录执行；三个输出位置均须不存在：

```sh
python results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/main-g2-coupled-bounds-v1/analyze.py \
  --run results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2 \
  --output /tmp/gh200-g2-coupled-data
python results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/main-g2-coupled-bounds-v1/verify.py \
  --run results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2 \
  --report results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/main-g2-coupled-bounds-v1/data/report.json \
  --output /tmp/gh200-g2-verification.json
python results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/main-g2-coupled-bounds-v1/compare.py \
  --heldout results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011 \
  --report results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/main-g2-coupled-bounds-v1/data/report.json \
  --output /tmp/gh200-g2-comparison.json
```

没有新增 GPU 实验或公共模型修改；所有观测/预测资格仍为 false。

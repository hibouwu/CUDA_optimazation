# R18：cluster 边界、有效零数据与影响范围

24个plain条件均完成10进程采样并独立复算，1,966,080个抽样输出值正确。完整trace有22个条件合格；另外两个短K条件原扰动5.34%/5.47%，经减少重复坐标写出的轻量补查后通过。轻量组含8个条件（两组边界及普通对照、两个K），再检查655,360个值。24个代表条件现均有可用的plain和合格范围内的事件证据。

job737122、romeo-a043、GPU-099dda56-d7af-f60e-c285-aa2dc7ddfcfe，CUDA12.9.41、sm_90a、CUTLASS3.9.2、NDEBUG。该设备身份与本轮R16配额、R19相同。

## 对照与不变条件

cfg_a沿M测试：普通M=1536、整块越界M=1408、偶数部分边界M=1504、奇数部分边界M=1376；N=2048。cfg_c沿N使用同样四个边界长度，M=3072。K=1024/8192。显式零组保留1536的有效descriptor范围，最后128行A（cfg_a）或128列B（cfg_c）置零。

整块组的普通、越界、显式零数据具有相同物理tile列表、grid、stride和分配容量；cfg_a保持6 stage、cfg_c保持4 stage，编译无spill、无C7510。两种输入仍为FP16，FP32累加/输出、α=1、β=0。输入和输出准备在计时外。

- 越界与有效零：二者都能让参与乘法的数据为零，但前者由descriptor边界导致TMA零填充，后者从有效全局地址读取显式零。能区分“值为零”与“边界/运输路径变化”，不能仅凭时间认定某个物理缓存机制。
- 部分边界：分别保留偶数或奇数个逻辑tile，区分部分有效tile与额外整块补齐。
- cluster1×1：只在长K增加两个条件。它同时改变多播、补齐和工作分配，不能用总时间差单独测多播成本。

## 完整GEMM时间

| 配置/条件 | K=1024 μs | K=8192 μs |
|---|---:|---:|
| cfg_a / ordinary | 19.456 | 90.080 |
| cfg_a / oob | 20.256 | 108.192 |
| cfg_a / explicit_zero | 19.008 | 89.776 |
| cfg_a / partial_even | 19.184 | 92.016 |
| cfg_a / partial_odd | 20.448 | 108.560 |
| cfg_a1 / oob | — | 89.456 |
| cfg_a1 / partial_odd | — | 88.928 |
| cfg_c / ordinary | 31.104 | 172.240 |
| cfg_c / oob | 30.816 | 171.440 |
| cfg_c / explicit_zero | 31.120 | 171.856 |
| cfg_c / partial_even | 31.136 | 172.256 |
| cfg_c / partial_odd | 30.688 | 170.960 |
| cfg_c1 / oob | — | 90.096 |
| cfg_c1 / partial_odd | — | 90.032 |

cfg_a长K整块越界比普通数据慢`108.192/90.080−1=20.1%`，有效零组89.776 μs与普通组接近。该条件不支持“只要数据为零就变慢”的解释。cfg_c的普通/越界/有效零分别172.240/171.440/171.856 μs，未呈现cfg_a的正惩罚，因此不能把cfg_a系数移植给cfg_c。

cfg_c切为cluster1后约90 μs，不能解释成多播翻倍成本：cluster1×2把12×11个逻辑tile补成12×12=144个物理tile，关键CTA执行两轮；cluster1×1恰好12×11=132个tile，一轮即可覆盖整卡。必须先算实际工作分配。

![边界完整服务](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/r18-service.png)

误差条为10个独立进程的标准差；橙色表示完整trace扰动超过5%，对应plain时间本身仍合格。

## 同物理tile的窗口增量

打点记录CTA入口、SM、producer首次工作、每consumer每tile的FIRST_MMA/MAIN_END/EPI_PERMIT/EPI_DONE和最终退出；首末SM相同。MAIN_END在mma_tail之后。它是包含等待的主循环窗口，不是单条MMA延迟。

```text
CTA entry → producer first work → FIRST_MMA → MAIN_END → EPI_PERMIT → EPI_DONE → next tile
                                  └── 主循环窗口 ──┘     └── 输出窗口 ──┘
```

先比较相同CTA编号、相同轮次和相同物理(mi,ni)的窗口差，再按边界双方、同列/同行其他CTA、其余CTA归组；分析器逐进程检查整份物理工作列表相同。两个不同进程的绝对clock64不能直接相减，必须各自先取窗口长度。

真实手算：cfg_a长K、trial0、CTA11、tile(11,0)，越界窗口为`4477056817549299−4477056817451969=97,330 cycle`；匹配普通组为`4477062071866019−4477062071800069=65,950 cycle`，增量31,380 cycle。该单点不是组中位数，也不代表最慢CTA。

10进程的首tile配对增量中位数如下，括号为进程标准差：

| 长K对照 | 边界tile cycle | cluster伙伴 cycle | 同列/同行其他CTA cycle |
|---|---:|---:|---:|
| cfg_a 整块越界−普通 | 21668 (1005) | 21776 (975) | 8091.5 (504) |
| cfg_a 有效零−普通 | −90 (261) | −86 (278) | −128.25 (207) |
| cfg_c 整块越界−普通 | 6.5 (131) | 1 (128) | −2.75 (136) |

cfg_a影响扩展到同列其他CTA，和共享B运输/服务竞争的解释相容；本组没有计数器与能唯一定位物理共享路径的第二干预，因此仍标为机制推断。cfg_c的微小正负差落在波动量级内，保留零差范围，不能解释成加速或端口规律。

每CTA每Ktile的工作量按真实tile计算：cfg_a为`2×128×128×64=2,097,152 FLOP`；cfg_c为`2×256×128×64=4,194,304 FLOP`。K=8192有128个Ktile。边界物理tile也执行指令，但完整GEMM的有效数学工作仍为2MNK，二者分开记。

主循环窗口差可以作为对应边界条件的候选增量。两个K点只能给出插值假设，不能凭零残差宣称可迁移斜率；V07之前冻结候选并用新形状检验，不把平均cycle/Ktile再叠一次截距。

## 两个超限条件的有界补查

轻量版本仅去掉consumer每tile的两个坐标写出，保留双方相同四事件与首末SM。cfg_a/c的plain GEMM SASS与完整版本逐字一致，[核对结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/reanalysis/local-independent/plain-sass-comparison.json)保存摘要。坐标由已在完整组逐进程验证的静态调度映射推得，并再次检查每CTA的tile计数；本组不声称重新直接记录了坐标。

| 轻量条件 | plain μs | 扰动 |
|---|---:|---:|
| cfg_a普通，K1024 / K8192 | 18.960 / 89.552 | 3.63% / 0.82% |
| cfg_a偶数部分边界，K1024 / K8192 | 19.168 / 92.352 | 3.09% / 0.97% |
| cfg_c普通，K1024 / K8192 | 31.136 / 171.664 | 1.34% / 0.69% |
| cfg_c奇数部分边界，K1024 / K8192 | 30.512 / 170.224 | 2.78% / 0.63% |

全部8条件各10组plain/trace通过，完整版本的两条超限记录仍不参与拟合。需要这两类的K插值时，使用轻量组自己的短/长K配对增量，不混搭完整组基线。[轻量归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/)、[独立检查结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/reanalysis/local-independent/rules.json)、[配对增量](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-lite-job737122-v1/reanalysis/local-independent/paired-tile-increments.json)。复现入口为`run_r18_lite.py`，prepare/build/setup/sample参数与完整组一致。

## 证据与复现

[归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/)、[条件与检查结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/rules.json)、[逐组窗口增量](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/paired-tile-increments.json)、[关键CTA和分布](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R18-job737122-v1/reanalysis/formal-v1/tails.json)。所有原始坐标、时间戳和4096个检查位置保存在每进程压缩stdout中。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r18.py prepare --output <新目录> --cutlass-root <CUTLASS3.9.2>
# 将目录搬到单GPU GH200分配内；以下三步顺序执行。
python3 <目录>/source/run_r18.py build --output <目录>
python3 <目录>/source/run_r18.py setup --output <目录>
python3 <目录>/source/run_r18.py sample --procs 10 --output <目录>
python3 microbench/gh200_resource_campaign/access_rules/analyze_r18.py --input <归档> --output <新分析目录>
```

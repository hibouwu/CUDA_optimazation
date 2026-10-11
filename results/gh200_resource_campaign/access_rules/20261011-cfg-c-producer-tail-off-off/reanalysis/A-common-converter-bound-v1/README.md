# R09：一个共同换算倍率的条件精度边界

使用740091 on/off、740092 off/off共40份原raw。固定cfg_c 2304×3072×4096、S4、tile256×128×64、cluster1×2、grid[66,2,1]、GPU-ef8692f3、dyadic seed17、16次同buffer observer-off前序。原局部ns判定失败、event判定通过保持不变。

每个调用对单独允许一个任意正α，构造条件预测：

`N1_i = α × (C1_i/C0_i) × N0_i`。

C0/C1来自两臂同CTA/同WG/同输出的原生cycle差，N0/N1为同标签globaltimer差；共同计算窗口另列。该模型显式读取两臂目标C与arm0目标N，α还事后读取全部目标窗口。它是对所有共同倍率的条件oracle，不是自由预测；交替的实际执行顺序全部保留，不声称arm0在arm1前可获得。

## 精确下界和可取见证

记q_i=N0_i×C1_i/C0_i、y_i=N1_i。要求所有绝对残差≤δ，等价于各倍率区间`[(y_i−δ)/q_i, (y_i+δ)/q_i]`有公共交集。任意两区间给出必要界：

`δ ≥ |y_i q_j−y_j q_i|/(q_i+q_j)`。

全部窗口对的最大值δ*同时给出全体区间的非空交集。导出该交集内α，并逐项验证`max|αq_i−y_i|=δ*`，结合两窗口下界即证明最优且可取。没有调用优化器。

相对目标另计算z_i=q_i/y_i，最优倍率为`2/(z_min+z_max)`、最大相对残差为`(z_max−z_min)/(z_max+z_min)`，也保存两极值见证与全窗口可取验证。绝对和相对最优倍率不同，不能组合成一个预测器或因果分解。

| 对照／窗口，各10对 | δ*最小／中位／最大，ns | 相对下界最小／中位／最大，% | δ*>256ns对数 |
|---|---:|---:|---:|
| on/off，原WG L | 125.488 / 329.885 / 680.062 | 0.3279 / 0.8337 / 1.7205 | 6 |
| on/off，共同计算窗口 | 108.862 / 324.452 / 663.543 | 0.2835 / 0.8217 / 1.6780 | 6 |
| off/off，原WG L | 97.849 / 241.832 / 363.652 | 0.2452 / 0.6096 / 0.9199 | 5 |
| off/off，共同计算窗口 | 84.434 / 239.481 / 356.679 | 0.2193 / 0.6006 / 0.9052 | 5 |

这里的256ns只用于诊断绝对残差尺度；残差不是原on/off观察差，不重判64/256ns gate，也不输出修正后的通过率。每对调用的最优最大相对残差很小，因此这些数据不排除一个共同倍率作为较粗相对精度的近似；但它们也没有证明预测前可得的状态能选出所需α。不能从绝对观察尺度失败推导完整时间5%/10%目标不可能。

## 两个绝对界见证

on/off pair1，δ*=680.06243571009ns，来自以下原WG窗口：

| CTA／WG／ordinal／SM | C0 | C1 | N0，ns | N1，ns |
|---|---:|---:|---:|---:|
| 114／0／0／89 | 65943 | 65874 | 38368 | 38848 |
| 13／0／1／18 | 65775 | 65909 | 41024 | 40256 |

off/off pair7，δ*=363.6520752475903ns：

| CTA／WG／ordinal／SM | C0 | C1 | N0，ns | N1，ns |
|---|---:|---:|---:|---:|
| 63／1／0／118 | 65582 | 65635 | 37920 | 38336 |
| 31／0／1／54 | 65794 | 65775 | 41184 | 40832 |

每个C都先在各自同SM记录内相减；见证比较没有跨SM原始clock相减。不同CTA/轮次在此只约束一个共同倍率，不定位唯一空间或相位机制，也不排除任意共同的可变F(t)。

## 端点和前序代理核对

两WG共同计算窗口先以clock选最早FIRST和最晚MAIN的WG标签，再使用相同标签的ns。8640个原窗口中，该ns差全部恰好等于ns域min-FIRST到max-MAIN包络；不会因分别选择不同域极值引入未声明的比值。此一致性不证明读取同时性或硬件精度。off/off全部4320个共同native cycle窗口与主集成main-cta-compute-envelope-v1逐值相同；没有采用其4096 FLOP/SM-cycle条件参考或转移服务价格。

`proxy-gap.py`只复用主集成已选定的末次前序event代理α=prefix1/prefix0，不新增状态函数。WG条件残差的每对最大值跨10对的最小／中位／最大为：on/off 0.8449%/1.2215%/5.9347%，off/off 0.3950%/1.4676%/3.0678%。off/off全部4320项与主集成main-prefix-scalar-diagnostic-v1的有理数残差精确一致。它与oracle之间的差距说明代理误差和共同倍率的表达精度需要分别评估；仍使用目标信息，不是局部自由预测。

## 身份与复验

绑定119项输入SHA，核对两份冻结清单的原/returned文件、完整output.tar.gz回执、binary、完整有序20记录、原raw、固定GPU/几何/trace、同buffer前序、flag身份、同SM与工作坐标。40份原4096样本检查通过标记保留，本轮不重算数值参考。

`windows.json.gz`保存8640个WG匹配窗口及4320个共同匹配窗口；`endpoint-selections.json.gz`保存8640个原共同窗口的端点选择核对；`report.json`保存40个最优问题及全部见证。`verify.py`不导入分析器或公共模型，独立重新读取40份原raw，验证12960项窗口、8640项端点选择、40份最优证书和20个前序代理配对；错误α、错误下界见证、错误相对界三个损坏证书全部拒绝。

复放：

```sh
r09_converter_output=$(mktemp -d /tmp/r09-converter-XXXXXX)
cp /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/A-common-converter-bound-v1/{analyze,proxy-gap,verify}.py "$r09_converter_output/"
python "$r09_converter_output/analyze.py" --base /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules --output "$r09_converter_output"
python "$r09_converter_output/proxy-gap.py"
python "$r09_converter_output/verify.py"
```

直接复验：

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/A-common-converter-bound-v1/verify.py
```

on/off和off/off分别保留，没有跨批相减或抵消失败；C/N仍含端点效应。没有新物理频率、误差界、源价格、公共模型修改、GPU工作、新冻结留出或跨卡资格。

# R16：TC 驻留和动态寄存器配额

驻留12条件与配额6条件均已完成正式采样、SASS检查和独立数值重算。这里交付容量、窗口重叠和释放时序三种条件服务；配额使用新分配 job737122，驻留保留 job736989 的有效证据，不拼接两张卡拟合。

## 驻留：容量与服务分开

64×64×64 FP16 TC探针，256线程（producer/consumer各一个warpgroup），每CTA处理32个Ktile，grid固定2×SM数=264 CTA。stage=1/2/4，全部预留四槽；总动态SMEM分别128/96 KiB，查询得到1/2 CTA/SM上限。三种stage使用同样预留和输出，58 reg/thread，无spill。

共享小源为16 KiB，全部CTA复用；独立大源为264×32×16 KiB=138412032 B，每CTA/每tile独立，比60 MiB L2大。数据预先按SW128打包，TMA普通bulk复制整块16 KiB；每Ktile四条m64n64k16、commit/wait0后释放输入槽。这里测这种依赖与等待条件的服务，不代表峰值WGMMA mainloop。

| stage | 小源1 CTA上限 ns | 小源2 CTA上限 ns | 大源1 CTA上限 ns | 大源2 CTA上限 ns |
|---|---:|---:|---:|---:|
| 1 | 48112 | 34128 | 76928 | 58560 |
| 2 | 34544 | 27888 | 59296 | 56192 |
| 4 | 32272 | 27792 | 55024 | 55312 |

- 小源stage1从容量1增至2，时间缩短29.1%；stage4仍缩短13.9%。更多软件stage不能完全替代另一个CTA的工作覆盖。
- 大源stage1增加容量有收益，stage4却基本无收益（55024/55312差0.5%，对应CV约0.65–1.52%）。该条件下不能把更高occupancy自动算成更高吞吐。
- 记录到的每SM执行窗口重叠最大为1/2，与容量上限相容；它不是完整CTA存活区间，因此不推“始终驻留两个”。

真实工作量：每CTA`2×64×64×64×32=16777216 FLOP`，264个CTA共4429185024 FLOP；小源stage4/容量2的globaltimer包络27792 ns，故有效速率约159.37 TFLOP/s。分子只计矩阵工作，分母含搬运、控制、消费和结果处理，不是整卡TC峰值。

数值检查覆盖每CTA全部4096个输出与末输入槽8192个半字。24个1/2-Ktile短检查、120正式进程，共144进程、467140608个值独立重算正确，全部预热收敛，正式最大CV1.52%。clock64只同SM相减；整卡结果使用globaltimer包络，保存CTA首末SM及时间区间。

[驻留归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R16-residency-job736989-v1/)、[结构化条件](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R16-residency-job736989-v1/analysis/rules.json)。本轮job736989、romeo-a048、GPU-009a8880、CUDA12.9.41。

## 配额：先释放还是先做工作

384线程，producer一个warpgroup、consumer两个。实际初始168 reg/thread：`384×168=64512`，低于65536；目标稳态`128×40+256×232=64512`。producer释放`128×(168−40)=16384`，能支持两个consumer各增加`128×(232−168)=8192`。编译无spill，consumer在inc后持有192个可识别FP32值并全部消费、保存。

| producer依赖整数动作/线程 | delay在dec之后：inc窗口 cycle | delay在dec之前：inc窗口 cycle | 两种次序的192次依赖FADD窗口 cycle |
|---|---:|---:|---:|
| 0 | 144.5 | 190.0 | 787.75 / 787.0 |
| 64 | 144.5 | 2106.5 | 787.5 / 788.0 |
| 256 | 144.5 | 7866.5 | 788.0 / 787.0 |

每条件10个独立进程，表内先取每进程8个consumer warp的中位窗口，再取进程中位数。inc进程CV最大0.45%，归约CV最大0.089%。先释放时，0/64/256动作的inc中位数相同；进程标准差约0.40/0.46/0.55 cycle，这只是当前分辨率内未见延长。先工作时，consumer等待包含producer释放前的延迟。64/256动作的producer延迟窗口中位数为1978.5/7738.5 cycle，包含循环控制和打点，不是单条IMAD延迟。

### 计时与工作量

```text
producer（先释放）  start → dec_enter → dec_return → delay_start → delay_end
producer（后释放）  start → delay_start → delay_end → dec_enter → dec_return
consumer           start → inc_enter → [重试分配] → inc_return
                                                 → 加载192值 → compute_start
                                                 → 192次依赖FADD → compute_end → 保存结果
```

`.inc`窗口包括warp同步、请求、重试、资源释放等待和时钟边界；`.dec`窗口约65–66 cycle。SASS中inc时钟为`0x1e0/0x240`，请求位于`0x210`，失败分支回到`0x1f0`，不会重复起点读数；dec时钟为`0x2930/0x2970`，释放位于`0x2960`。归约时钟为`0xee0/0x1b50`，192条FADD位于两者之间，加载指令在前、结果存储在后。入口时钟依赖最后一项加载值；该窗口描述这种加载组织后的归约服务，可能残留的操作数等待也在分母内。末时钟之前有依赖最终和的整数/谓词检查。全部warp记录首末SM一致，只在同SM时钟域相减。

每CTA consumer共`256×192=49,152`次FP32加法、49,152个保留值，输入为196,608 B。producer共`128×delay`次依赖IMAD，其中delay=256对应32,768次；不能把consumer的单warp中位窗口当整CTA同步包络，进而用总工作量除它推整卡吞吐。

例如`quota_delay64_before_dec/formal-00`的consumer首warp：原始`4473079253278769−4473079253276678=2091 cycle`可在[进程记录](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R16-quota-job737122-v2/samples/quota_delay64_before_dec/formal-00/result.json)手算；八warp中位数与每进程重算值保存在[processes.json](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R16-quota-job737122-v2/reanalysis/local-ten-processes/processes.json)。相对于先释放，64动作条件中位窗口多`2106.5−144.5=1962 cycle`，与释放前延迟同一量级。192次加法窗口约788 cycle，数量级接近`192×4=768`再加边界开销；本实验没有单独隔离裸FADD延迟，不据此授予4 cycle硬件常数。

本轮6个短检查、60个正式进程共66进程，3,269,376个值精确重算通过，包括所有consumer活跃值、逐步FP32归约与producer整数结果。预热8–30次，监测consumer inc窗口，末5次CV≤2%。job737122、romeo-a043、GPU-099dda56-d7af-f60e-c285-aa2dc7ddfcfe、功率上限900 W、CUDA12.9.41；NCU单次检查仍为`ERR_NVGPUCTRPERM`。

![驻留容量、配额等待与分配后归约](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R16-quota-job737122-v2/reanalysis/local-ten-processes/r16-services.png)

误差条为独立进程标准差。左图整卡globaltimer微秒与右侧同SM周期分属不同分配、不同计量边界，不互相换算。

[正式配额归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R16-quota-job737122-v2/)、[本地独立资格与条件](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R16-quota-job737122-v2/reanalysis/local-ten-processes/rules.json)。原3进程分析留在`analysis/`，10进程补样说明和脚本在`reanalysis/`；旧18进程代表记录及其独立重算仍保留在[旧归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R16-quota-dev-job736989/reanalysis/20261008-offline-quota-v1/)。旧版加载与归约交错，未记录归约末端/末SM，预热监测producer延迟，也未按相邻配对采样，因此不混入新版正式数据。

## 模型适用范围

- 容量输入：使用实测编译资源与CUDA查询上限；stage增加和CTA容量增加可分别覆盖等待，不能只按occupancy比例加速。
- 配额依赖：consumer的请求必须能从producer释放获得所需资源；把释放前工作保留在事件依赖中，不能给所有setmaxnreg附加同一固定成本。
- 后续消费：本组的192次依赖FADD与其输入/输出边界明确，仅作为活跃值和分配后执行见证，不替代CUTLASS实际consumer计算服务。
- 未得到物理RF bank、端口容量、池仲裁公式，也未验证其他配额比例或tile形状。

## 复现

```bash
# 默认入口仍为驻留12条件。
python3 microbench/gh200_resource_campaign/access_rules/run_r16.py \
  --subset residency --cutlass-root <CUTLASS3.9.2> --output <新驻留目录>
python3 microbench/gh200_resource_campaign/access_rules/run_r16.py \
  --subset quota --quota-procs 10 --cutlass-root <CUTLASS3.9.2> --output <新配额目录>
python3 microbench/gh200_resource_campaign/access_rules/analyze_r16.py \
  --input <归档> --output <新分析目录>
```

`--prepare-only`冻结源码、完整依赖头和矩阵；搬至节点后用归档内`source/run_r16.py --resume`运行。当前入口的`--quota-procs 10`直接取得10进程；已冻结v2入口初始为3进程，确切补样命令为`python3 <归档>/reanalysis/supplement.py <归档>`，不可用新版脚本冒充旧编译源码。重要归档及CUTLASS依赖有树外备份和SHA256核验。

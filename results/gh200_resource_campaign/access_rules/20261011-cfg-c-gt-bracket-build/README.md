740156 在 ROMEO a057 以 4 CPU / 16 GiB、CUDA 12.9、SM90a 完成构建，未申请或运行 GPU。目标为固定 cfg_c 2304×3072×4096、tile 256×128×64、stage 4、cluster 1×2、132 CTA、原 dyadic 输入和 16 次同 buffer 前序。二进制 SHA：`c5476d9b00679859eef64bff34e87532e4936ea73f7f432c476597b8c36a6480`。

`prepare_cfg_c_gt_bracket.py` 从 SHA 固定的 740087 源码构造目标。856 个原文件中仅 host 入口 `probes/r18.cu` 和实际使用的 `probes/v06_trace.hpp` 改动；另新增读取 helper 与准备脚本。`source.patch`、`source-audit.json` 和完整输入归档保存差异。首次 740155 错改未引用的 `r18_trace.hpp` 副本，出现三个 host 标识未定义错误，保存在相邻 `20261011-cfg-c-gt-bracket-build-failed-header-740155`；本次显式检查覆盖头到 v06_trace 的 include。

原 FIRST_MMA / MAIN_END 标记线程 128、256 读取 GT_before → clock64 → GT_after。GT_before 的真实返回值经零值判断控制 clock 读，clock 返回值经零值判断控制后 GT 读。任一前置值为零时整个三元组清零，无重试。FIRST 的 SASS PC 为 1790 / 17d0 / 1800，MAIN 为 2450 / 2490 / 24c0：前依赖保留实际分支，后依赖保留带谓词的 CS2R。`audit.py` 核对完整指令链、无绕过前置检查的跳转、无效路径清零和输入身份；`build-audit.json` 保存结果。

原 cycles 和 GT_after 仍在原 event 槽。GT_before 写独立 shadow，每 CTA 9 个 uint64：`((role-1)*2+output_ordinal)*2+event`，event 仅 FIRST=0 / MAIN=1，最后一格为越界标记。固定最多两个输出序号，额外 9,504 bytes；原 1,368,576 bytes trace 不变。关闭时 shadow 全零；每次调用在计时外清零；16 次前序固定关闭；target `--gt-bracket-observer 0/1`，producer-tail 强制关闭。`check-gt-bracket` 复用全 D off/on 逐位比较和导出入口，目前仅编译成功，尚未执行。新协议版本明确拒绝冒用旧 prefix 规则的 binary 身份。

资源仍为 168 registers、2 barriers、无 stack/spill。资源行与旧构建相同；静态 HGMMA=16、WARPGROUP=4、SYNCS.ARRIVE=4。资源相同不代表无扰动。实际 off 路径也有新 flag 分支和不同代码布局，新增 shadow 的清零也改变前序准备；旧 binary 与旧 prefix 校准不自动迁移。

区间公式以实际采样顺序及 GT 参考域为条件：单个 FIRST/MAIN 的 elapsed ∈ [MAIN_before−FIRST_after, MAIN_after−FIRST_before]。需要由新 reader 校验完整 132 CTA、SM/线程/工作坐标、前后读顺序、无零值和无越界；当前没有新设备记录。不得用三读数值单调、资源相同或源码/PC 顺序单独授硬件采样包围资格。条件域与额外 GT 误差预算、off/on 扰动、off/off 背景、数值正确性和最终预测分别判断，旧失败门槛保留。

[NVIDIA inline-PTX 文档](https://docs.nvidia.com/cuda/archive/12.9.1/inline-ptx-assembly/index.html#incorrect-optimization) 将 volatile 和 memory clobber 的保证限定在 PTX 生成与内存优化；[PTX globaltimer 说明](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi) 声明其行为依目标而定。当前结果只证明本二进制保留结果依赖，不提供连续物理 ns 精度、硬件读延迟界或有效观测证书。全部实测与预测资格保持 false。

复放：输入归档解包后执行保存的 `run.sh`（需要相同 CUDA/编译环境）；返回结果按 `returned/` 布局解包，执行 `python audit.py`。不重复构建即可从归档源码、SASS 和完整 hash 链完成静态检查。

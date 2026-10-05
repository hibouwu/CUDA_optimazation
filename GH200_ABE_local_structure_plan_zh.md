# A / B / E 本地文件结构规划

日期：2026-09-29。状态：已建立 [GPUResearch 导航入口](../../GPUResearch/README.md) 与 A/E 的计划文档；其余仍为目标结构，尚未初始化 Git、创建完整工程或迁移代码。

A 的最新范围与选型依据见[公开文献综述](../../GPUResearch/vpsr-gpu/literature/review_zh.md)和[框架比较](../../GPUResearch/vpsr-gpu/literature/frameworks_zh.md)；不再预先绑定 Turbulence，目录名暂留以保持链接。

建议采用“一个导航入口、三个项目边界”：A 研究低精度数值语义与 GPU 编译实现，B 延续现有 CUDA 算子仓库，E 管理 Qwen3 推理框架实验。学习资料、上游源码、自己写的代码和实验产物各有归属。

## 1. 已有目录与落点

本次检查确认 `/home/jianyeshi/Note` 和 `/home/jianyeshi/Note/SGLang` 不是 Git 仓库；`CUDA_optimazation` 与 `SGLang/sglang` 各自是 Git 仓库。

| 对象 | 建议路径 | 处理方式 |
|---|---|---|
| 总入口 | `/home/jianyeshi/Note/GPUResearch/` | 新建普通工作目录，不在这一层初始化 Git |
| A：低精度与编译实验 | `/home/jianyeshi/Note/GPUResearch/vpsr-gpu/` | 独立项目；公开研究与后续可能取得的私有资料分别管理 |
| B：GH200 算子 | `/home/jianyeshi/Note/CUDA/CUDA_optimazation/GH200/` | 加入现有 CUDA 仓库，不创建嵌套 Git 仓库 |
| E：推理框架实验 | `/home/jianyeshi/Note/GPUResearch/qwen3-inference/` | 独立实验项目 |
| 上游开发源码 | `/home/jianyeshi/Note/GPUResearch/upstream/` | Turbulence、SGLang、vLLM 各自的 checkout |
| SGLang 学习资料 | `/home/jianyeshi/Note/SGLang/Docs/` | 保留原位，通过链接引用 |
| SGLang 课程源码 | `/home/jianyeshi/Note/SGLang/sglang/` | 保留课程阅读用途，不与实验版本强行合并 |

总入口通过 README 和编辑器工作区引用 B 的真实位置，不复制 CUDA 仓库，也不依赖软链接搬运它。

## 2. 总体结构

下面都是目标结构；目录按实际任务逐步创建。

```text
/home/jianyeshi/Note/GPUResearch/          # 普通工作目录，不是总 Git 仓库
├── README.md                            # A/B/E 入口、当前任务、已有计划链接
├── research.code-workspace              # 打开 A、B、E 的真实目录；本机路径
├── paths.local.env                      # 本机代码、依赖、数据路径，不含密钥
├── vpsr-gpu/                            # A：独立实验目录，名称暂留
├── qwen3-inference/                      # E：独立实验仓库
├── upstream/                            # 各 checkout 独立管理版本与修改
│   ├── turbulence/                      # 获得导师仓库后，按其真实结构开发
│   ├── sglang/                          # 实验/修改用源码，后续需要时获取
│   └── vllm/                            # 对照源码，需要读改源码时再获取
└── artifacts/                           # 生成产物，不纳入代码仓库
    ├── a-vpsr/<run-id>/
    ├── b-gh200/<run-id>/
    └── e-inference/<run-id>/
```

`upstream/` 表示代码来源，不表示只能读：实际修改放在对应 checkout 的开发分支，保留基准提交与补丁记录。A/E 仓库记录依赖的准确提交和未提交差异，不能只写版本标签。需要隔离同时进行的开发时再创建工作树。

不把这些 checkout 复制进 A/E 的 `src/`，第一版也不建立统一依赖下载器或公共 Python 包。

## 3. A：低精度、编译与数值验证

```text
/home/jianyeshi/Note/GPUResearch/vpsr-gpu/
├── README.md                            # 课题范围、最小运行方式、当前状态
├── literature/                          # 已建立：综述、索引、BibTeX、公开 PDF
├── plans/turbulence-original.md          # 已归档：导师路线的原始方案
├── docs/
│   ├── algorithm-notes.md               # 论文理解、未确认的问题
│   ├── numerical-contract.md            # 精度、舍入点、特殊值、随机数语义
│   └── compiler-path.md                 # 实际 IR/pass 路径与源码位置
├── configs/                             # 精度/舍入配置、小模型实验配置
├── src/                                 # 自己的参考实现、适配与实验驱动
├── tests/                               # 边界、统计性质、编译前后回归
├── scripts/                             # 构建/运行入口、Slurm 提交脚本
├── env/                                 # 依赖版本、ROMEO 环境说明
└── reports/<experiment-id>/             # 可复现报告、精选表格与图
```

**选用上游框架后，核心 GPU/编译改动放在对应 checkout 的原生目录中。若选择 Turbulence，同样遵守此约定。** 尚未拿到私有代码，不能提前杜撰它的 `lib/`、`passes/` 或 kernel 结构。A 的 `src/` 放框架外的参考与实验代码，避免维护两份核心算法。

原文论文、导师资料与源码按各自许可保留；本地实验仓库默认私有，公开材料需要单独筛选。D 的语义差分验证直接并入这里的 `tests/`，不另起第四个项目。

## 4. B：GH200 算子实现与性能建模

当前目录按已有 GEMM 模型和微基准实际位置整理；下列结构替代早期的单文件硬件与重复 microbench 规划。

```text
/home/jianyeshi/Note/CUDA/CUDA_optimazation/
├── microbench/
│   ├── gh200_l0/                        # 历史 FMA/MMA/WGMMA 探针与原始归档
│   └── gh200_resource_campaign/         # 新资源探针、运行、独立审查和绘图
├── results/gh200_resource_campaign/
│   └── <run-id>/                        # 配置、工件、原始记录、报告与图表
├── Docs/ModelEvaluation/gemm/
│   ├── problem.md、workloads.yaml       # 通用问题定义与具体实例
│   ├── model/L0.md–L4.md                # 通用分层模型与统一接口
│   ├── hardware/gh200_sm90/             # README 与 L0–L3 硬件资源
│   ├── schemes/                        # 具体实现选择、资源需求和预测
│   └── experiments/gh200_sm90/          # 资源实验说明、审查与结果入口
└── GH200/                               # 未来完整算子实现需要时创建
    ├── kernels/gemm/                   # 实际 GEMM 实现
    ├── tests/                          # 数值、边界、布局与支持范围
    ├── benchmarks/                     # kernel 计时与 cuBLAS 等基线
    └── scripts/                        # 完整算子的构建和 Slurm 入口
```

资源微基准统一使用 `microbench/`，不再增加 `GH200/microbench/`。新 run 归档只保存一份到 `results/gh200_resource_campaign/`，历史 L0 数据保留原路径；当前 Git 忽略结果目录，文档链接可本地访问不代表结果已纳入版本控制。

[GH200 实验入口](Docs/ModelEvaluation/gemm/experiments/gh200_sm90/README.md)记录目录决策、旧测量复审和后续顺序。`problem.md` 保留硬件无关的候选域；具体 GH200 精度、tile 与流水方案由 workload 和方案确定，不从旧 Thor 实现直接继承。

第一阶段仅补全与审查组件服务，不提前建立空的算子目录。完整 GEMM 实现启动后再创建所需 `GH200/` 文件；构建产物使用架构独立目录。

## 5. E：Qwen3 推理框架研究

```text
/home/jianyeshi/Note/GPUResearch/qwen3-inference/
├── README.md                            # 固定模型、两框架入口、当前主问题
├── docs/
│   ├── request-path.md                  # 带版本/源码位置的请求执行链
│   └── comparison-contract.md           # 精度、缓存、负载、指标与计时口径
├── configs/
│   ├── models/                          # 模型 revision、模板、生成设置
│   ├── frameworks/                      # vLLM/SGLang 的有效配置
│   └── workloads/                       # 请求长度、到达模式、前缀关系
├── src/                                 # 请求生成、客户端、指标与 trace 分析
├── integrations/                        # 框架外适配代码、B 算子接入说明
├── tests/                               # 指标、请求语义、改动回归
├── scripts/                             # 服务启动、测量、Slurm 与结果整理
├── env/                                 # 依赖、源码提交、ROMEO 环境
└── reports/<experiment-id>/             # 机制分析、精选数据、图与结论
```

**修改 scheduler、KV cache 或模型内部 dispatch 时，代码放对应框架 checkout。** E 保存如何启用改动、基准/修改提交以及回归与测量结果，不另复制一套框架内部源码。`integrations/` 仅在确有框架外适配代码时创建。

B 的 kernel 通过构建产物或可安装扩展接入 E；E 记录 B 的提交和构建参数，不拷贝 `.cu` 文件。先接入一个实际调用点，再按需要完善打包，不预先设计通用插件系统。

## 6. 版本与实验产物约定

| 内容 | 保存位置 | Git 策略 |
|---|---|---|
| 自写代码、测试、小型配置、报告 | 所属 A/B/E 项目 | 纳入各自仓库 |
| Turbulence/框架源码与修改 | `upstream/` 下对应 checkout | 使用上游各自的 Git 历史 |
| 原始 timing、日志、IR、PTX/SASS、NCU/NSYS 文件 | `artifacts/<project>/<run-id>/` | 不纳入代码仓库，重要记录另做持久备份 |
| 精选数据、图、产物索引 | A/E 的 `reports/`，B 的 `measurements/` | 小而可复现的部分纳入 Git |
| 模型权重、数据集、Python 环境 | 配置指定的本机/集群缓存位置 | 不纳入 Git，不随代码同步 |

每次运行创建新 `run-id`，目录内至少保存 `manifest.json`、有效配置、日志与原始结果。Manifest 记录代码提交和本地差异、依赖版本、命令、设备/节点、随机种子和状态；报告通过 run-id 与文件校验值指向原始证据。

同一个 run-id 只追加该次运行的状态与输出。修复代码或改变配置后重新分配 run-id，重试单独编号；不能把新结果覆盖成旧配置的结果。

**第一版仅共享记录格式和路径约定。** 每个项目保留自己的小型脚本；实际重复出现后再抽取公共代码。根目录不建大一统工具包。初始化时逐项检查 `.gitignore` 与跟踪状态；现有 CUDA 仓库对 `results/`、profiler 和二进制已有忽略规则，报告应使用上述明确路径。

## 7. A/B/E 之间传递什么

- **E → B**：实际 shape、dtype、stride、batch 和频次，导出小型 workload 文件，并记录来源模型、框架提交和 trace；B 冻结一份作为自己的实验输入。
- **B → E**：可构建的 kernel/扩展接口、支持范围与回退条件，引用明确提交和构建配置。
- **A → B/E**：精度/舍入契约、质量结果及配置。只有实现了相应原生格式与计算路径后，才进入端到端性能比较；不能仅凭模拟配置就切换框架精度。

共享的是明确的数据与接口，不让三个项目互相导入整套源码目录。

## 8. 本地与 ROMEO 的对应

本地负责编辑、阅读、分析与归档，ROMEO 负责 GPU 构建和运行。集群目录根路径尚未核实，运行时使用 `$PROJECT_ROOT`、`$ARTIFACT_ROOT`、`$MODEL_CACHE` 等配置，不假设存在某个 `/scratch` 或配额目录。

同步源码与配置，排除 `.venv/`、`build/`、模型缓存和原始 profiler 大文件；本地与 AArch64 集群分别创建环境和构建产物。优先回传 manifest、日志、摘要和图，完整原始证据按重要性归档。不要把节点临时存储当作长期备份。

## 9. 第一批真正需要创建的文件

实施时先创建总入口 README、A/E 的 README 与忽略规则、B 的 GH200 README，以及各项目的第一个计算/实验契约。随后跟随第一项真实实现增加目录：A 的一个参考测试、B 的一个 GEMM、E 的一个最小请求基线。

不为整棵树创建空文件；不移动现有 SGLang 课程、CUDA 内核或 Thor 结果；不同时复制所有上游源码。当前项目路线文档继续作为唯一计划入口之一，由新的总入口链接，避免维护两份相同计划。

本方案基于本次本地目录与 Git 边界检查。保存这份规划不代表上述项目已创建、已安装依赖或在集群运行。

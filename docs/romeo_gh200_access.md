# ROMEO GH200 远程访问

记录日期：2026-09-29。登录信息来自本次会话，Slurm 流程对照 ROMEO 官方文档；本次未发起 SSH 连接或申请 GPU。

```text
本地终端
  └─ SSH → hibouwu@romeo1.univ-reims.fr（登录节点）
       └─ salloc 申请 armgpu → srun 进入当次分配的 GH200 计算节点
```

## 1. 从本地登录

```bash
ssh -i ~/.ssh/id_ed25519 hibouwu@romeo1.univ-reims.fr
```

| 项目 | 当前记录 |
|---|---|
| 登录主机 | `romeo1.univ-reims.fr` |
| 用户 | `hibouwu` |
| 本地私钥路径 | `~/.ssh/id_ed25519`，仅引用路径，不放入仓库 |
| Slurm 项目账户 | `r260073`，来自此前账户关联查询 |
| GPU 节点约束 | `armgpu` |

登录后在 `romeo1` 上使用 Slurm。登录节点不是 GH200 计算节点，那里没有 `nvidia-smi` 不代表 GPU 环境损坏。本地终端也不需要安装 `srun`。

## 2. 在登录节点申请并进入 GH200

以下是三小时、单 GPU 的交互式开发示例。`--mem=16G` 和四个 CPU 核是起步配置，按实际编译/运行需求调整；`--mem` 不是 GPU 显存申请。

```bash
salloc \
  --account=r260073 \
  --partition=short \
  --time=03:00:00 \
  --constraint=armgpu \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --gpus-per-node=1
```

等待出现分配成功信息后，在同一个会话中执行：

```bash
srun --pty bash -i
hostname
nvidia-smi -L
```

用实际输出确认节点和 GPU。`romeo-a057` 只是此前分配过的节点，不能作为固定入口；历史作业 `725075` 已到期，不应继续使用。

需要从另一个终端进入仍有效的作业时，先执行第 1 节的 SSH 登录，再查询作业：

```bash
squeue -u "$USER"
```

将下方 `JOB_ID` 替换为自己当前运行中的作业号：

```bash
srun --jobid=JOB_ID --overlap --pty bash -i
```

该命令不会延长分配。到时后作业及其 step 会被终止；重新申请会得到新的作业号。结束使用时先退出计算节点 shell，再退出 `salloc` 分配 shell。

## 3. 在计算节点加载 CUDA

以下版本来自此前会话的环境记录，使用时确认当前可用性：

```bash
romeo_load_armgpu_env
spack load cuda@12.9.0
command -v nvcc
command -v ncu
nvcc --version
ncu --version
```

若 CUDA 版本不存在，先用 `spack find cuda` 查询安装版本；`ncu` 是否可用以当前节点的命令输出为准。环境加载在计算会话或作业脚本里进行，不把 `romeo_load_armgpu_env` 加进通用 `.bashrc`。

## 4. 本仓库的工作入口

- 本地仓库：`/home/jianyeshi/Note/CUDA/CUDA_optimazation`。
- 继续定义 GEMM：[problem.md](../Docs/ModelEvaluation/gemm/problem.md)。
- 硬件、方案与测量导航：[GEMM 设计与性能建模](../Docs/ModelEvaluation/gemm/README.md)。
- 远程仓库实际路径：**尚未核实**。SSH 登录位置与远程代码目录是两回事，不能把本地绝对路径直接当成集群路径；在远程 checkout 内通过 `pwd` 和 `git rev-parse --show-toplevel` 确认。

参考：[ROMEO 作业与交互式使用说明](https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/lancer_un_calcul/)。账户权限、排队情况和可用软件以登录后的实际状态为准。

## 5. NCU 硬件计数器权限实查（2026-09-30）

SSH → Slurm → `romeo_load_armgpu_env` → CUDA 12.9.0 的连接与计算流程已实际成功。NCU 性能计数器需要额外的驱动授权，GPU 计算权限不自动包含该权限。

`short` 普通作业在 `romeo-a043`、`short --exclusive` 作业在 `romeo-a046` 均读到 `RmProfilingAdminOnly: 1`，实际 NCU 返回 `ERR_NVGPUCTRPERM`。此前 `instant` 的 `romeo-a057` 也返回相同错误。独占分配未自动开启计数器。

`sudo -n -l` 要求密码，尚未确认该账户的交互 sudo 授权范围。若有专用 profiling 节点、reservation、管理员 wrapper 或已有 sudo 授权，应使用其明确入口；不能把更换 SSH 参数或分区当作已获得权限。

详细证据见 [计数器权限诊断](../microbench/gh200_l0/results/20260930-counter-access/README.md)。

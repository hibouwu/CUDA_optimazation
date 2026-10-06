# S20 冻结实验脚本

本目录保留正式采样的源码、配置、参考检查、审查器和公共依赖的原始字节及相对路径。首先阅读 [实验导航](../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/README.md)。数据与发布资格以导航中的独立审查和结果记录为准；源码交付不代表参数发布已完成。

- `run_s20.py`：该家族有限配置的短检查、自身校准和正式采样入口。
- `audit_s20.py`：独立重算已保存的原始数据。
- `run_node_s20.py`、`collect_node_s20.py`：原节点运行与收集。
- `report_service.py`：从重算结果生成表、图和中文报告。
- `source-manifest.json`、`source-review.json`：原冻结身份与独立源码签录。

原 ARM ELF 不复制进源码目录；完整可审查的运行包保存在 `results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/repo` 及对应 `local-collection` 归档。原 manifest 包含 `bin/probe` 的精确哈希，因此直接从本源码目录运行正式采样会拒绝缺失的 ELF。不要删去哈希校验或将新编译文件冒充原文件；新编译是新运行，需要重新绑定其二进制和受影响的审查证据。

对既有结果的离线复核，使用原归档中的完整 repo 与原始数据。复核命令格式：

```bash
python3 -B /已解包运行目录/repo/audit_s20.py --run /已解包运行目录
```

原采集使用 Python 3.9.21；其他版本的统计末位可能不同。离线重算入口为 ../../replay_frozen_family.py，原环境重算已经保留；本目录不声称该示例在任意解释器下逐位相同。归档保留原二进制不要求执行 GPU，审查器只读取证据。

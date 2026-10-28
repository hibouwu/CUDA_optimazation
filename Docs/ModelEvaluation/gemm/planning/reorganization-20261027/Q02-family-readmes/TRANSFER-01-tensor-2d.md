# TRANSFER-01-tensor-2d（草稿）

主题：TRANSFER-01 tensor 2D 传输。

5 个目录，其中 0 个由别处移入（原族列出）。“记录性质”按目录内是否有 Slurm 日志、receipt、submission、allocation 或回传目录判定；“卡”由文本中的 GPU UUID 识别，未注明为 —；“结论摘录”取自各轮正文，最终版按原记录改写。

| 轮次 | 原族 | 问题 | 记录性质 | 卡 | 结论摘录 |
|---|---|---|---|---|---|
| `20261014-tma-pitch-discrimination-001` | TRANSFER-01-tensor-2d | B 行跨度冻结对照 | 含 GPU 执行 | — | 原始包、返回凭据、逐条件统计与绑定保留原样。原参考卡 UUID 见 plan.json。 |
| `20261014-tma-tensor-development` | TRANSFER-01-tensor-2d | cfg_b tensor 探针开发记录 | 含 GPU 执行 | a057 | 第二版校准与64 CTA 校准均属于开发。共享源 1 MiB 与独立源 64 MiB 的工作集不同；候选比较同时保留在途合并与工作集驻留解释。拟用 128 槽源环、128 组、固定共享 4 MiB 的相位对照区分，两种相位的每 CTA 地址集合相同。 |
| `20261014-tma-tensor-discrimination-001` | TRANSFER-01-tensor-2d | Tensor 多 CTA 首批冻结判别 | 含 GPU 执行 | a057 | 残差集中在 Q1 完成等待、独立源在 Q1/Q4 下的吞吐响应和固定工作集的相位差。后续开发分开驻留/流式路径的服务率与完成延迟，并保留在途共享候选；本批结果转为后续开发证据，新版本另做独立留出。 |
| `20261014-tma-tensor-heldout-002` | TRANSFER-01-tensor-2d | tensor 第二轮冻结留出 | 含 GPU 执行 | — | 评分：相位响应预测 +8.96496%，实测 −0.18727%，未记录模式为 −0.25554%。零响应竞争解释保留，P16 候选被反证。全部局部时间最大绝对误差 41.45%，未通过 5% 标准；两项消费者释放延迟条件误差分别为 −0.6513% 与 +0.4153%。 |
| `20261017-TRANSFER-01-replay-001` | TRANSFER-01-tensor-2d | TRANSFER-01 离线交付 | CPU／已有数据 | — | S16本轮只核发布数字与原审查身份，没有重做全payload审查。tensor性能模式的released为issuer的软件许可标记；旧两套冻结集合均未通过。实际cfg_b的源／目的价格与信用规则迁移仍待测，原组合C2/C3资格保持未完成。 |

## 族内分组（待定稿）

- （按假设或端点分小节，审阅时补）


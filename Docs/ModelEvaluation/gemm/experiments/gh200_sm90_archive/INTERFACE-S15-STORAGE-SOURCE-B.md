# S15 配额查询与完整归档工具

状态：source-B 实施，待独立审查；没有执行 GPU target，没有 S15 B3 或性能资格。

`runners/s15_storage.py` 只负责已有 S15 短验证的查询与运输。每个包保留一个完整 run；固定 68×3 坐标与原探针不变。controller 在新的 `S15-short-controller-A-review-r3.json` 和自己的 source-B 门禁通过前仍拒绝 `run`；旧A与source初稿保持不改。

## 查询

`quota --repo <scratch-repo> --output <receipt>` 实际执行 GPFS `mmlsquota -e -u <当前UID> -Y gpfs`，按原 HEADER 名称解析唯一 UID/gpfs/scratch 行；`-Y` 的 KB 乘 1024，使用量加 blockInDoubt，同时拒绝 block/inode soft grace 到期，另检查 inode 剩余至少 65536。receipt 保留原 stdout SHA、当前时间、UID、scratch fileset 根和 512 MiB 余量。controller 每个新 target 前重新调用查询，不复用过期 headroom。

这里不能用 `/gpfs` 总空闲量替代用户 scratch 配额。IBM 明确 `-Y` 返回 KB、in-doubt 会约束剩余空间；`-e` 刷新各节点使用量。[IBM mmlsquota 文档](https://www.ibm.com/docs/en/storage-scale/6.0.0?topic=reference-mmlsquota-command)

`device --binary <已审S15实际binary> --output <device.json>` 只允许匹配既有 S15 source-B 的实际 binary SHA，在有效 GH200 Slurm 分配和 UUID 锁内执行 `probe device`。保存 bounded 30 秒 receipt、原 stdout、环境、binary 与 source-review SHA；controller 再核 argv 恰为 device、成功与清理、stdout 和分配 UUID。它不启动 target kernel。

## 封包、转存与恢复

1. `seal --index <0..203> --output <新目录>` 在 family 锁下核对原 manifest 所有字节与完整文件集合，再以固定普通文件清单无损写 tar.xz。拒绝链接、路径越界、额外文件、未完成诊断或未确认进程清理；封包后复核源成员和字节不变，再完整解码验证每个成员，生成 remote receipt。不会覆盖旧包。
2. `collect --remote-repo <ROMEO scratch repo> --entry <相对entry.json> --repo <本地新repo>` 仅取 entry 指向的 archive/index/closure/remote receipt/origin。远端逐文件 SHA/大小前后相同、传到本地字节相同后，本地完整解码，生成 offhost receipt/origin 与 collection 记录。收集前用受审本地合同核204中的具体坐标/run；将全部传送文件和三份待生成输出一次预检，所有已存在父目录必须无symlink、规范相对路径且resolve仍在local_repo，禁止覆盖，之后每次mkdir/SCP前及SCP后复检普通文件。已有symlink反例在任何mkdir或SCP前拒绝，仓库外零写入。不会覆盖本地既有文件，不在远端删除原始 run。
3. `admit --entry ... --offhost-receipt ... --offhost-origin ...` 把已转回远端的两个验证收据绑定到原 `resident_passed` ledger 的 coordinate/run/manifest。要求不同 verifier host，核实际包全成员、包内 spec/raw 坐标、原状态与两收据闭包，才记 archived_verified。
4. controller 恢复已封存点时执行 archive audit；展开目录缺失不会触发 GPU 重测。数值 B3 仍需要独立检查器，完整字节运输不等于数值通过。
5. `retire --index ...` 是单独显式命令，只在 `/gpfs/scratch` 部署内允许，先重新验证完整包/两地身份/原 resident 字节；UUID 锁内要求无 active registry，才删除此 run。包、ledger、收据和本地归档保留。controller 不自动调用 retire。

core 的原 validation_manifest 故意不包含可变 validation_state。source-B 接口将该文件作为唯一额外必要成员，单独绑定字节并核 case_diagnostic_passed；原 manifest 加自身及该 state 必须恰好等于包成员集合。此增量没有改写原 core 或历史 manifest。

## 实证与时间边界

CPU 测试覆盖实际小包写入和两次完整解码、配额字段/UID/fileset/in-doubt、缺失/额外/变更/链接证据拒绝、规范 run 与包内坐标、错误收据和本地 canonical retire 拒绝。ROMEO 的真实 CPU fixture 在单独新 scratch 部署中生成 1 MiB payload，实际执行配额查询与封包，再真实转到本地并全成员解码；它明确标记 synthetic_CPU_fixture，不能当作 GPU 数值或 B3 证据。初次 fixture 同源同目标 copy 的失败保留，没有 GPU 重测。

最大配置仍是 64 KiB padding、33 轮、396 CTA 的必要容量上界，artifact 共 2,595,257,664 B。相同字节量在本地 `/tmp` 完整写入、fsync 和全读 SHA 耗约 2.25 秒；原始输出位于 `implementation/s15-storage-source-B/maximum-artifact-CPU-io.json`。这是本地 CPU I/O 证据，不能作为 ROMEO target 的 30 秒上界。实际 source 仍沿用 core 的 30 秒 bounded target、180 秒编译和 Slurm 剩余时间门槛；首个最大配置须获得真实成功 receipt 才能证明该时间预算适用，失败必须保留并停止，不能改轮数或伪报完成。source-B 不授予这个未知时间参数。

最大 raw 加最坏封包临时空间约需 4.84 GiB，另有元数据和日志。真实配额不足就 checkpoint，不根据实测压缩比放行。本轮没有自动清理任何既有 S11/S14 数据。

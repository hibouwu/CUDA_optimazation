# S15 连续配额查询的证据归档修订

实际 job731148 在 wrapper 初次查询后，controller 对同一个 alias 刷新配额，固定 stdout 已存在导致 Errno17，目标launch为0；第二次写入 quota.process.json 还覆盖了初次process receipt。原ROOT b、首次GPU432身份、空ledger、提交与失败记录保持原文。修订只解决查询证据的文件身份，不更改测量协议。

每次 `query_quota(repo, output)` 创建 `output.name + '.queries' / <time_ns>-<pid>-<nonce>`。新目录必须不存在，不能是外部symlink；stdout、stderr、process.json、result.json均保存在此目录。成功result保持原配额schema，`query_stdout.path/sha256`引用本次不可覆盖的stdout。该目录与raw/process/result长期保留，current alias不能代替本次证据身份。

只有真实bounded进程成功、清理确认、UID/gpfs/scratch对应行与数值解析通过，并且本次result归档完成后，才原子更新output这个current alias。失败也归档明确rejected result，保留当前alias原字节；目录身份冲突在执行前拒绝，不覆盖任何旧工件。单次仍30秒，命令仍 `mmlsquota -e -u UID -Y gpfs`；KB换算、in-doubt、block/files grace、inode reserve、60秒freshness、512MiB余量、4GiB raw与最坏seal预算不变。

wrapper初次查询与controller每个候选前的freshquery均调用同一修订函数，连续配置不共享raw/receipt。无需删除首次查询，也不通过减少配置或扩大时限掩盖问题。

新contract v3、manifest v4与early A/source-B r4、controller A r6/source-B r4仅派生当前依赖路由。probe、profile、68×3配置、固定index101最大点、n33/seedUINTMAX、actual22目标SASS与shared core r5保持原SHA。旧contract、manifest、源码、门禁和部署包精确保全；新版本必须新ROOT/新source identity，不能续写旧b。当前缺新门禁时入口主动拒绝。

CPU证据使用实际bounded进程和文件IO，GPFS挂载与mmlsquota输出是明确的模拟提供者，不声称发生ROMEO真实query或GPU。保留六个完整目录：wrapper初次、连续配置101与0、exit13失败、错误UID失败，以及下一次fresh成功。两失败保持alias不变；过期alias先拒绝再经新query恢复。新增六个测试还覆盖nonce碰撞无覆写、四配置连续refresh和symlink零外写，旧十四个S15测试仍通过。

独立复审需核每个完整process与result SHA、当前alias与本次immutable result对应、失败不改alias、先前目录所有SHA未变，以及metadata路由只改允许的字段。此修订不授GPU数值/B3，不自动提交或retire。新门禁通过后需真实CPU initialize/freeze与新部署包装审查，再继续固定首次短验证。

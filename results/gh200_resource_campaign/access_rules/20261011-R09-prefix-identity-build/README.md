# Plain 前序准备身份

协议`r09-plain-module-sw1-16/v1`在setup和call同时记录ABI、每次trace清零字节数以及`trace-memset-device-sync-per-call`规则。case必须显式声明协议和清零字节数，call必须保留16次前序、同buffer标识与16个有限非负间隔；解析器拒绝缺失、未声明或不一致的身份。

这与旧common-sw1-16不是同一完整历史契约：后者调用当前编译变体；共享plain前序另有独立模块身份。二进制/模块哈希仍由构建清单绑定，不把ABI版本当内容哈希。跨观察变体还须检查清零容量与前序模块一致，单进程身份通过不代表两变体匹配或无扰动。

740050在romeo-a057完成，退出0:0，67秒，仅4CPU/16GiB、无GPU分配。两个host重编成功，协议字符串在制品中；GEMM SASS分别与冻结V09 plain/dual逐条相同。3项新身份测试、18项既有历史测试通过；此轮没有验证新增字段的GPU运行输出或采集时间。

旧740048/r5仍保留两进程数值smoke结论。其plain/dual scratch分别827904/1368576字节，setup缺新协议字段，因此严格解析器不接受它们作为已声明新协议样本；不补写旧raw、不把它们升级为观察有效性证据。

[验证记录](verification.json)、[编译脚本](run.sh)、[启动参数](command.json)、[输入哈希](input-sha256.json)、[回传身份](receipt.json)可复核。完整输入/输出archive及二进制在当前结果目录保留。

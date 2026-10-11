# R13 四 warp release 数值与协议检查

作业740054在romeo-a057 / GPU-43269fbc-449d-3e0f-908a-9c81229546d3完成，134秒，分配4CPU、16GiB、1GPU。该作业不独占，仅作编译、数值检查及每条件一次的轨迹smoke，不纳入正式时间样本或ef卡机制价格。

stage4/6均运行plain和stamped的 `check-release-warps`：每次对4,325,376个FP32输出做off/on逐位比较，different=0、nonfinite=0。随后在同作业内用 `cmp` 逐字节比较plain与stamped-off的完整D；两stage都通过。四份导出D的SHA256相同：069572268437ecad997fdc04c96f39175b9e0fa816cf01e83742a908dd7cf533。全D文件只用于compute节点比较；本地保留命令、成功标记、SHA和检查输出，未回传四份大型D文件。

四份smoke使用common-sw1-16，预热观察开关为0。setup/call/清单身份、16次前序、trace布局、slot/phase和四warp记录均由当前严格解析器验证。每次另有4096个输入参考值检查，nonfinite/padding/reference_error均为0。

| stage | 观察开关 | 四warp前后戳宽度（cycle） |
|---|---|---|
| 4 | 0 | 22 / 0 / 0 / 0 |
| 4 | 1 | 23 / 22 / 22 / 22 |
| 6 | 0 | 22 / 0 / 0 / 0 |
| 6 | 1 | 25 / 22 / 22 / 22 |

单次读取宽度均不超过B预先给出的32cycle门槛，但每条件只有一次，不能据此判定局部或整调用的配对扰动门槛通过。完整数值输出一致也不能证明计时无扰动。

本次只增加host端 `--check-output` 导出；四个GEMM的指令序列与740051逐指令相同，SHA见 `kernel-identity.json`。740051的static-release-audit.json记录开关常量、lane选择与k17处同谓词前后clock、中间非谓词arrival及区间内无分支/写回的静态证据；尚未运行active mask观测，不把它写成物理到达或完成证明。

40进程正式机制观察未启动；预热/二进制/同卡参照/配对顺序及扰动门槛仍须统一冻结。V09失败、cfg_a/b/c总目标和新独立留出标准不变。

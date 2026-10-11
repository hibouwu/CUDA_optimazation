# 主线复核 cfg_a 输出源保证

合入 C 的 287930d 为 6854ee0 后，主线以当前 analyze_r15_lifecycle.py 独立复放历史 R14。852 份归档输入及两份二进制重新核对，40 进程的 11,796,480 个检查值保持通过；20 份 trace 给出 90 个输出源保证，其中 70 个中间输出保证记录在下一输出 permit 之后。

与 C-20261011-output-lifecycle-v4 对照：由于主线传入相对目录，JSON 顶层目录、逐进程 source、raw_sha256 的路径键，以及两个 CSV 中的 source 路径均有绝对/相对差别。只删除 C 文件中的当前工作目录绝对前缀后，JSON 和两个 CSV 的完整文本精确一致；未改数值或哈希。13 项 output dependency/lifecycle 测试及26项 release/历史协议回归共39项通过。

8 输出条件先取每进程7个中间窗口中位数，再聚合10进程：permit 到源保证为36598.5 cycle；下一输出 permit 到该保证为657.5 cycle。该记录提供 source read completion 的上界，不能把整个窗口解释为独立输出价格，也不转换为 ns 或迁到新卡/全GPU。

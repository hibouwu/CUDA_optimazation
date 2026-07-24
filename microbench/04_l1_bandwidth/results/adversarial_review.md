# L1 运行对抗式审查

## 结论

通过：`read-ca` 的 L1TEX load bytes 接近期望、lookup-hit 占比高，且 LTS 流量远小于逻辑请求。

写路径只作为 L1TEX store path 报告，不解释为纯 L1 cache write-port 峰值。

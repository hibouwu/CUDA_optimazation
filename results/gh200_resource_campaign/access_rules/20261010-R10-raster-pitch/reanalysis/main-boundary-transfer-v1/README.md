B 的旧／新 R10 共同接口审计已在主线完整重算。`59b1ee18` 接入为 `3ac0938d`；两侧各 37 项输入 SHA 核对，完整报告除路径外相同，四份完整账本与说明文件逐字节相同。未重跑未修改 helper 的既有测试。

旧、新 12 份核心归档头文件逐字节相同，且分别匹配各自 manifest。新四条件的源 keys 与原 full-tile source API 及先前完整工作审计一致。可以复用 `g=j*ceil(K/64)+q`、slot=g%6、同槽前驱 g−6 和 WG=j%2；旧 Kt32/504 输出、T4=108/T3=24，新 Kt16/432 输出、T4=36/T3=96，持续状态必须各自重建。

四个新条件的完整完成工作与 ring 拓扑相同。同 raster 改 pitch 时，源身份等价类也相同，但实际 keys/地址不同。改 raster 则改变源等价类：CTA0/1 首输出在 AlongM 复用 B0/B1，在 AlongN 复用 A。因此，源相等关系可以给 raster 提供不同输入，但单凭它仍不能区分 pitch；没有据此认定实际物理共享、owner 或仲裁。

本审计没有转移服务价格、卡间换算、旧局部观测或时间资格。新的有效观测、服务识别和独立留出仍未完成。

```sh
python microbench/gh200_resource_campaign/access_rules/analyze_r10_boundary_transfer.py \
  --root results/gh200_resource_campaign/access_rules \
  --output NEW_OUTPUT_DIRECTORY
python results/gh200_resource_campaign/access_rules/20261010-R10-raster-pitch/reanalysis/main-boundary-transfer-v1/verify.py
```

# L1 设计对抗式审查

## 攻击点

1. **`.ca` 读可能没有 L1 hit。**

   设计响应：每个 CTA 的 working set 默认会提升到 32 KiB，以覆盖 256 线程 * 8 unroll 的首轮唯一访问，并在同一个 kernel 中先用 `.ca` 预热；NCU 必须验证 L1TEX lookup-hit bytes 接近期望请求，且 LTS bytes 显著低于逻辑请求。

2. **block 不一定一一分布到 SM。**

   设计响应：默认 blocks=SM count；运行报告记录 blocks 和 SM count。若 NCU 显示 waves/SM 异常或 L1 hit 不成立，则该结论不通过。

3. **写路径不能叫纯 L1 cache write bandwidth。**

   设计响应：`write-wb` 只解释为 L1TEX global-store front-end / end-to-end store path。

4. **编译器可能改变 cache op 或访问宽度。**

   设计响应：使用 inline PTX `ld.global.ca/cg.v4.u32` 和 `st.global.wb/cg.v4.u32`；SASS 摘要必须显示 128-bit global ops。

## 设计结论

设计可以运行；通过条件是 NCU hit/miss/traffic 支持 `read-ca` 是 L1-hit，SASS 保持 128-bit op，并且写路径按 store path 限定表述。

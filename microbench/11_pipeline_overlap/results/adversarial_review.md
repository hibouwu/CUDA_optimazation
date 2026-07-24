# Pipeline overlap 运行对抗式审查

结论：通过。

## 判据结果

- overlap gain serial/overlap = 1.550
- mainloop cycles/cp 19.881 close to overlap 19.901
- SASS needles present
- NCU summary present

## 证据和解释

- `cp-only` measured cp ingress is `859.024 B/cycle/GPU`.
- `ts-mma-only` estimated TMEM consume demand is `115.699 B/cycle/GPU`.
- `serial-a1` cycle/tile is `30.839`; `overlap-a2` is `19.901`.
- `overlap-a2` cp payload is `102.907 B/cycle/GPU`, `11.98%` of cp-only.
- `overlap-a2` consume demand is `102.907 B/cycle/GPU`, `88.94%` of TS-MMA-only.
- NCU overlap-a2 SM throughput is `51.120% peak`; rough cp upper from SM peak is `201.305 B/cycle/GPU`, rough consume upper is `201.305 B/cycle/GPU`.

## 保留边界

- `bytes_per_cycle` 是 cp payload，不是总 TMEM 双向带宽。
- consume bandwidth 是按 TS MMA operand demand 估计，不是 raw TMEM read-port counter。
- NCU `sm__inst_executed_pipe_tmem.*` 在当前工具链可能为 0；不把它作为 UTCCP 是否执行的判据。

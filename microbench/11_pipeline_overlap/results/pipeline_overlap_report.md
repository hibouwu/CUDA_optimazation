# Pipeline overlap app report

|case|role|TFLOP/s|cp B/cycle|consume B/cycle est.|cycles/cp|cycles/tile|serial gain|ideal-overlap eff.|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|cp-only|component-cp-ingress|0.000|859.024|0.000|2.384|0.000|0.000|0.000%|
|ts-mma-only|component-tmem-consume|373.198|0.000|115.699|0.000|0.000|0.000|0.000%|
|serial-a1|serial-cp-mma|214.210|66.409|66.409|30.839|30.839|1.000|57.398%|
|overlap-a2|double-buffer-overlap|331.938|102.907|102.907|19.901|19.901|1.550|88.944%|
|warp-split-a2|split-issuer-overlap|300.601|93.192|93.192|21.976|21.976|1.403|80.547%|
|mainloop-a2-k16|steady-state-overlap|332.272|103.011|103.011|19.881|318.102|1.551|89.034%|

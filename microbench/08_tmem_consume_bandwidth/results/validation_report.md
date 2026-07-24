# TMEM consume validation report

|case|role|TFLOP/s median|estimated TMEM consume B/cycle median|
|---|---|---:|---:|
|ss-mma-mainloop-k16|smem-baseline|921.885|0.000|
|ts-cp-mma-a2-k16|tmem-consume-plus-cp|332.272|103.011|
|ts-mma-only|tmem-consume|373.198|115.699|

## NCU

|case|SM %peak|tensor inst %peak|tmem pipe inst %peak|rough consume upper B/cycle|
|---|---:|---:|---:|---:|
|ts-mma-only|44.770|0.560|0.000|258.429|
|ts-cp-mma-a2-k16|51.280|0.500|0.000|200.879|
|ss-mma-mainloop-k16|91.780|1.390|0.000|0.000|

# DSMEM 运行对抗式审查

## NCU 后审查

|mode|app B/cycle|dshared/expected|dshared %peak|estimated peak B/cycle|
|---|---:|---:|---:|---:|
|local-read|2400.191|0.000|0.000||
|local-write|2173.339|0.000|0.000||
|remote-read|239.181|1.032|2.410|10231.535|
|remote-write|283.116|1.032|2.850|10241.053|

## 结论

通过：remote-read/remote-write 的 NCU dshared bytes 与预期请求字节在 25% 容差内，说明 remote 模式确实产生 DSMEM traffic。

local shared 没有本机 direct byte counter，因此 local 结果只作为 app clock + SASS + wavefront proxy 支撑。

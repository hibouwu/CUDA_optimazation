# NCU GMEM/DRAM validation report

|mode|app B/cycle|LTS/expected|DRAM proxy/expected|LTS %peak|LTS B/cycle|estimated LTS peak B/cycle|
|---|---:|---:|---:|---:|---:|---:|
|read-stream|125.440|1.031|0.949|12.670|129.740|1023.994|
|write-stream|82.179|1.031|0.860|7.770|79.590|1024.324|
|copy-stream|71.889|1.292|0.970|9.440|96.620|1023.517|

When `dram__bytes*` is missing, DRAM proxy is `(LTS read miss sectors + LTS write miss sectors) * 32 B`.

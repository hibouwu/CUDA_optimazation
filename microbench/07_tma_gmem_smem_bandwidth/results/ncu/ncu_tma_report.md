# NCU TMA GMEM-to-SMEM validation report

|mode|app B/cycle|TMA bytes/expected|TMA %peak|TMA B/cycle|estimated TMA peak B/cycle|LTS/expected|DRAM proxy/expected|LTS %peak|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|l2-hit|775.178|1.008|29.450|753.860|2559.796|1.008|0.006|73.620|
|dram-stream|164.159|1.008|6.150|157.320|2558.049|1.008|1.008|15.370|

When `dram__bytes*` is missing, DRAM proxy is LTS read miss-sector bytes.

# TMEM cp ingress validation report

- cp-only rows: 9
- cp-only throughput: 859.024 B/cycle/GPU
- rough cp ingress upper used here: 859.024 B/cycle/GPU (cp-only sustained)
- cp-only cycles/cp: 2.384
- effective bytes/cp: 2048
- cp instructions in app timing row: 200000
- interference rows with cp traffic: 36
- max cp traffic during MMA interference: 773.397 B/cycle/GPU
- worst slowdown vs control in interference sweep: 0.287
- SASS summary: present
- NCU key report: present
- static UTCCP instructions in representative cp-only SASS: 29
- NCU tmem pipe metric value: 0.000 inst/cycle active
- NCU memory throughput pct peak active: 8.300%
- NCU tmem pipe metric is not used as cp proof when it reports zero for UTCCP.

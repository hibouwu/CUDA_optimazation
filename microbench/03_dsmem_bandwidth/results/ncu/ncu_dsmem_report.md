# NCU DSMEM validation report

|mode|app B/cycle|dshared B/cycle elapsed|dshared %peak elapsed|dshared/expected|estimated dshared peak B/cycle|shared wavefront %peak|LGDS wavefront %peak|
|---|---:|---:|---:|---:|---:|---:|---:|
|local-read|2400.191|0.000|0.000|0.000||93.280|0.010|
|local-write|2173.339|0.000|0.000|0.000||84.500|0.010|
|remote-read|239.181|246.580|2.410|1.032|10231.535|0.000|12.270|
|remote-write|283.116|291.870|2.850|1.032|10241.053|0.000|12.560|

Direct byte counters exist for `mem_dshared`; local shared memory is checked with LSU wavefront and bank-conflict counters on this machine.

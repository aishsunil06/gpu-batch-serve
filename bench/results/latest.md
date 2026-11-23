| Arm | Batch | Delay ms | Conc. | RPS | p50 ms | p95 ms | p99 ms | Mean batch | GPU % |
|---|---|---|---|---|---|---|---|---|---|
| A: no batching | 1 | 0.0 | 1 | 107.4 | 8.6 | 14.7 | 18.4 | 1.0 | n/a |
| A: no batching | 1 | 0.0 | 8 | 120.3 | 62.9 | 84.9 | 92.6 | 1.0 | n/a |
| C: batch 8, 5ms wait | 8 | 5.0 | 1 | 60.1 | 15.9 | 27.5 | 29.7 | 1.0 | n/a |
| C: batch 8, 5ms wait | 8 | 5.0 | 8 | 241.0 | 33.2 | 37.3 | 39.4 | 8.0 | n/a |

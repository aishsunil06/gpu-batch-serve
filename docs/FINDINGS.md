# Findings

Every figure here names the command that produced it. Where a number is
from a single short run rather than the full grid, it says so.

## Setup

| | |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2`, 6 layers, 384 dimensions |
| Workload | 64-512 deterministic mixed-length queries, seed 17 |
| Load | Closed loop, fixed concurrency, 2-5 s warmup discarded |
| Percentiles | Nearest rank |
| Repeats | 3 for the full grid, 1 for smoke runs |

## 1. Below the crossover, batching is pure cost

```bash
python scripts/run_bench.py --device cpu --quick --arms A,C --concurrencies 1,8 --workload-size 64
```

CPU, 1 repeat, 6 s windows. Single run, so no standard errors.

| Arm | Concurrency | RPS | p50 ms | p95 ms | p99 ms | Mean batch |
|---|---:|---:|---:|---:|---:|---:|
| A: no batching | 1 | 107.4 | 8.6 | 14.7 | 18.4 | 1.0 |
| C: batch 8, 5 ms | 1 | 60.1 | 15.9 | 27.5 | 29.7 | 1.0 |

Throughput fell 44% and median latency rose 85%. The mean batch size
column is what makes this interpretable: it stayed at 1.0, so the server
spent its grace period waiting for a second request that a single-client
workload can never supply, then ran a batch of one.

**Why this matters operationally:** the failure is invisible from latency
alone. A service whose p50 doubled after a config change looks like a
regression in the model, the host, or the network. The mean batch size is
the only signal that distinguishes "batching cost us" from every other
explanation, and it is not a metric most serving stacks expose by default.

## 2. Above the crossover, batching improves throughput and tail together

Same command.

| Arm | Concurrency | RPS | p50 ms | p95 ms | p99 ms | Mean batch |
|---|---:|---:|---:|---:|---:|---:|
| A: no batching | 8 | 120.3 | 62.9 | 84.9 | 92.6 | 1.0 |
| C: batch 8, 5 ms | 8 | 241.0 | 33.2 | 37.3 | 39.4 | 8.0 |

2.0x throughput, and p95 fell from 84.9 ms to 37.3 ms at the same time.

This is the result the usual framing gets backwards. Dynamic batching is
normally described as trading latency for throughput. That description is
correct only below the crossover. Once the server is the bottleneck, an
unbatched server's requests queue behind each other anyway, and that queue
is a worse source of latency than a 5 ms batch delay. Arm A at concurrency
8 has a p50 of 62.9 ms for a forward pass that took 8.6 ms when it ran
alone; the other 54 ms is queueing.

Note also that arm A barely gained throughput from 8x the concurrency
(107.4 to 120.3 RPS) while its latency grew 7x. That is the signature of a
saturated serial resource, and it is the condition under which batching
has something to work with.

## 3. GPU run

**Not yet measured.** `notebooks/colab_run.ipynb` runs the full grid on a
Colab T4. Three things can only be answered there:

- **Where the crossover sits on a GPU.** A GPU's cost per batch is flatter
  in batch size than a CPU's, so the crossover should move left, possibly
  to a concurrency low enough that batching is worth enabling by default.
  This is a prediction, not a result.
- **Whether utilisation explains the gain.** If arm A already saturates the
  GPU, batching cannot help, and the utilisation column will say so
  directly. If arm A sits at low utilisation, the headroom is the gain.
- **What fp16 adds on top of batching (arm E).** Half precision and
  batching both reduce per-request cost, and whether they compose or
  overlap is not predictable from either alone.

Record the numbers here with the exact command and GPU name once the run
completes, and update the README's GPU section at the same time.

## 4. Threats to these results

- **The CPU numbers are a single run each.** Arms were run in one
  contiguous block in the order A then C, on a thermally managed laptop.
  Block ordering is a plausible confound for the tail columns
  specifically. The p50 and mean-batch-size columns, and the direction of
  every effect above, are large enough not to be ordering artefacts; the
  p99 column should be treated as provisional until the grid runs with
  three repeats.
- **The batcher is not a production batcher.** It is roughly 60 lines and
  does not implement priority, request cancellation, sequence-length
  bucketing, or per-batch timeout. Sequence-length bucketing in particular
  would change the results, since a padded batch runs at the length of its
  longest member and one long query inflates the cost for everyone sharing
  its batch.
- **One model, one size.** MiniLM is small. The crossover for a model large
  enough that a single request already saturates the device will sit
  somewhere else, and plausibly does not exist at all.
- **No accuracy check on fp16.** Arm E measures speed only.

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

## 3. On a GPU the gain is 2.3x, and the GPU is not what limits it

Measured on a Colab **Tesla T4**. Full grid: 5 arms x 4 concurrencies x 3
repeats of 20 s, after a 5 s discarded warmup, on a 512-query workload.
26 minutes, 222,333 requests, **0 failed**.

```bash
python scripts/run_bench.py --device cuda --concurrencies 1,8,32,64     --workload-size 512 --repeats 3 --duration 20 --warmup 5
```

Every cell is in `bench/results/gpu_t4.md`. The rows that carry the argument:

| Arm | Conc. | RPS | p50 ms | p95 ms | Mean batch | GPU % |
|---|---:|---:|---:|---:|---:|---:|
| A: no batching | 1 | 143.4 (SE 0.7) | 6.4 | 9.8 | 1.00 | 26.9 |
| B: cap 8, no wait | 1 | 144.2 (SE 0.7) | 6.3 | 9.6 | 1.00 | 27.0 |
| C: cap 8, 5 ms wait | 1 | 83.0 (SE 0.0) | 11.3 | 15.3 | 1.00 | 15.6 |
| A: no batching | 8 | 143.5 (SE 0.6) | 50.9 | 76.8 | 1.00 | 27.6 |
| B: cap 8, no wait | 8 | **336.6** (SE 0.4) | **21.4** | **35.5** | 3.89 | 25.7 |
| C: cap 8, 5 ms wait | 8 | 326.5 (SE 2.0) | 24.0 | 36.4 | 4.70 | 24.3 |
| D: cap 32, 5 ms wait | 8 | 308.1 (SE 2.3) | 25.5 | 39.2 | 4.67 | 23.1 |
| E: cap 32, 5 ms, fp16 | 8 | 327.9 (SE 1.2) | 24.3 | 36.5 | 4.77 | **11.9** |
| A: no batching | 32 | 139.9 (SE 0.8) | 209.5 | 304.6 | 1.00 | 27.0 |
| B: cap 8, no wait | 32 | 168.5 (SE 3.8) | 104.3 | 600.8 | **1.96** | 19.6 |
| C: cap 8, 5 ms wait | 32 | **220.0** (SE 4.3) | 78.2 | 462.3 | 3.79 | 17.0 |

### Batching improves throughput and median latency together

At concurrency 8, arm B does **2.3x** the throughput of arm A (336.6 against
143.5 RPS) at **2.4x lower** median latency (21.4 ms against 50.9 ms). It is
not a trade.

Arm A is why. Its throughput is flat at roughly 143 RPS from concurrency 1
to 32 while its median latency grows from 6.4 ms to 209.5 ms, a 33x increase
for no throughput at all. Every request after the first is queued behind a
forward pass it could have shared.

### The GPU was never the limit, and fp16 is the proof

**Mean GPU utilisation never exceeded 27.6% in any of the 20 cells**, and the
highest instantaneous sample across the whole grid was 33%. That is
suggestive but not conclusive on its own, because `nvidia-smi` reports the
fraction of time a kernel was resident, not how much of the device that
kernel used.

Arm E settles it. It is arm D in half precision, and it did what half
precision is supposed to do: mean utilisation fell from 23.1% to 11.9% and
peak memory from 265 MiB to 217 MiB. **The device's work was roughly halved
and throughput rose 6%** (327.9 against 308.1 RPS, combined SE 2.6) rather
than approaching the near-doubling a device-bound server would show.

This is the most operationally useful result on this page. A team told their
embedding service is slow will reach for a faster GPU, a smaller model, or
quantisation. On this workload all three were available and none of them was
the answer, because the forward pass is small and the cost is in getting
requests to it. **The intervention that paid 2.3x cost nothing but a queue.**

### Past concurrency 8, achieved batch size falls as load rises

Arm B peaks at concurrency 8 and then degrades: 336.6 RPS to 168.5 at
concurrency 32, with p95 rising from 35.5 ms to 600.8 ms. Its achieved batch
size falls at the same time, 3.89 to 1.96.

That direction is backwards. More concurrent clients should mean fuller
batches, not emptier ones, and the collapse is therefore not the batcher
running out of room. Combined with flat GPU utilisation, it places the
bottleneck inside the server process rather than on the device. **The
specific mechanism is not isolated here** and would need per-stage timing to
pin down; the single-threaded request path and the GIL are the obvious
suspects and are untested.

### The queue delay is insurance priced at 42%

Separating the batch cap from the queue delay is what makes the two arms
worth running, and they behave differently at every load:

| | Conc. 1 | Conc. 8 | Conc. 32 | Conc. 64 |
|---|---:|---:|---:|---:|
| B: cap only, no wait | 144.2 | **336.6** | 168.5 | 145.4 |
| C: cap plus 5 ms wait | 83.0 | 326.5 | **220.0** | **160.8** |

The cap is free. Arm B matches arm A exactly at concurrency 1 (144.2 against
143.4), because a cap that is never reached costs nothing, and it is the
fastest arm in the grid at concurrency 8.

The delay is not free. It costs **42%** of idle throughput, for the reason
section 1 gives on CPU: the server waits 5 ms for a second request that a
single client cannot supply, then runs a batch of one. But it is the only
thing that holds batch size up under overload, 3.79 against 1.96 at
concurrency 32, and it buys **31% more throughput** there.

So the delay is insurance against overload, and 42% of idle throughput is
its premium. Which way that priced trade goes depends on whether the service
spends more of its life idle or saturated, and that is a traffic question,
not a serving question.

### A prediction this grid refuted

Section 3 of an earlier revision predicted the crossover would **move left**
on a GPU, since a GPU's cost per batch is flatter in batch size than a CPU's.

It did not move. Arm C costs 42% of throughput at concurrency 1 on the T4
against 44% on the CPU, which is the same answer. The prediction treated
"batching" as one knob; the grid shows the cap and the delay are separate
knobs and only the delay has a crossover at all. The practical conclusion
the prediction was reaching for does hold, but for a different reason than
it gave: batching is worth enabling by default on this GPU because the cap
is free, not because a GPU moved the crossover.

### What this grid does not answer

Arm E measures speed and memory only; no accuracy check was run on fp16, so
"half precision was free" is a statement about throughput, not about
embedding quality. Section 5 lists the rest.

## 4. A larger batch can mean a worse server

Measured. Triton 24.08 CPU-only in Docker Desktop, the local arm native on
the host, three repeats of 20 s per cell after a 5 s discarded warmup, all
arms on byte-identical ONNX weights verified to agree to 1.3e-07.

```bash
python scripts/compare_servers.py --arms local,triton,triton-grpc     --concurrencies 1,8,32,64 --repeats 3 --duration 20 --warmup 5
```

| Arm | Conc. | RPS | p50 ms | p95 ms | Mean batch |
|---|---:|---:|---:|---:|---:|
| local | 1 | 62.9 (SE 0.8) | 15.6 | 19.0 | 1.0 |
| triton | 1 | **75.2** (SE 0.7) | **13.2** | **14.7** | 1.0 |
| triton-grpc | 1 | 75.1 (SE 0.3) | 13.2 | 14.6 | 1.0 |
| local | 8 | 169.1 (SE 0.3) | 46.9 | 51.7 | **8.0** |
| triton | 8 | **194.6** (SE 1.7) | **40.6** | **47.0** | 5.4 |
| triton-grpc | 8 | 190.3 (SE 9.9) | 41.8 | 49.3 | 6.2 |
| local | 32 | **238.0** (SE 1.8) | **133.9** | **142.7** | **31.7** |
| triton | 32 | 207.7 (SE 2.2) | 152.9 | 181.8 | 16.4 |
| triton-grpc | 32 | 215.0 (SE 4.4) | 146.3 | 168.8 | 15.9 |
| local | 64 | **248.8** (SE 2.3) | **256.7** | **269.7** | 31.8 |
| triton | 64 | 209.5 (SE 0.8) | 287.8 | 420.8 | 29.3 |
| triton-grpc | 64 | 214.0 (SE 2.9) | 284.2 | 408.9 | 27.8 |

### The result that matters

**Triton wins below the crossover while forming batches a third smaller,
and the naive collector wins above it while forming batches twice as
large.** At concurrency 8 Triton is 15% faster on throughput and better on
both percentiles, on a mean batch of 5.4 against the collector's 8.0.

Part 1 of this repository treats mean batch size as the metric that
explains a throughput result. This table shows that reading is wrong, and
the mechanism is the interesting part.

The collector holds one queue and runs one batch at a time. While a forward
pass executes, nothing is dequeued, so arrivals pile up and the next batch
is maximally full. Its batch of **31.7 at concurrency 32 is not a batcher
working well; it is 32 clients stalled behind one forward pass.** Triton
overlaps input handling with execution, so at any dispatch moment only part
of the client population is queued and the rest are in flight. Its batch
size converging to roughly half the concurrency is the signature of a
server that is not blocking.

So a larger achieved batch size is not evidence of a better batcher. It can
be evidence of a worse one, and distinguishing the two needs the latency
columns, not the batch column alone.

### A prediction this made, and the experiment that refuted it

Triton's batch of 16.4 at concurrency 32 sat suspiciously close to the
middle value of `preferred_batch_size: [8, 16, 32]`, suggesting Triton was
dispatching at a preferred size instead of filling the batch. If so,
narrowing the setting to `[32]` should raise achieved batch size toward 32
and recover the throughput gap.

```bash
# preferred_batch_size: [ 32 ]
python scripts/compare_servers.py --arms triton --concurrencies 32,64 --repeats 3
```

| Config | Conc. | RPS | p50 ms | p95 ms | Mean batch |
|---|---:|---:|---:|---:|---:|
| preferred [8,16,32] | 32 | 207.7 (SE 2.2) | 152.9 | 181.8 | 16.4 |
| preferred [32] | 32 | 224.0 (SE 5.0) | 134.0 | 227.0 | **14.9** |
| preferred [8,16,32] | 64 | 209.5 (SE 0.8) | 287.8 | 420.8 | 29.3 |
| preferred [32] | 64 | 211.4 (SE 1.5) | 291.5 | 412.3 | 30.0 |

**The prediction was wrong.** Achieved batch size did not rise toward 32; it
fell slightly, to 14.9. `preferred_batch_size` was therefore not the cap,
which leaves the overlap explanation above as the one consistent with both
experiments. Throughput at concurrency 32 did improve, by about 16 RPS
against a combined standard error near 5.4, but p95 got 25% worse, so the
setting is a latency-throughput trade rather than the free win the
hypothesis predicted.

Recording this is the point. A comparison that only reported the first
table would have shipped the preferred-batch-size story as an explanation,
and it is not one.

### gRPC against HTTP

Within noise at every concurrency: 215.0 against 207.7 RPS at concurrency
32, 214.0 against 209.5 at 64. **No transport effect is detectable here,
and this pairing cannot cleanly measure one anyway** (see the caveat
below). On an 86 MB model returning 1.5 KB responses, there is little for a
protocol to win.

The incidental finding is more useful: the two rows agree to within about
3% despite one being driven by Python threads and the other by Go
goroutines, which is evidence that the server rather than the client
dominates these measurements.

## 5. Threats to these results

- **The local arm runs natively; Triton runs in a virtual machine.** This is
  the most serious threat on this page. Docker Desktop for Windows runs
  containers inside a WSL2 VM, so every Triton request crosses a virtualised
  network boundary that the local arm's requests do not. The local arm
  therefore has a structural advantage that has nothing to do with batching,
  and it is a plausible contributor to its high-concurrency throughput win.
  **The cross-arm numbers at concurrency 32 and 64 should not be read as a
  verdict on Triton.** Three things survive this confound: the within-Triton
  preferred-batch-size experiment, since both configs ran in the same
  container; the achieved-batch-size columns, which are counted server-side
  and are unaffected by transport; and Triton's *win* at low concurrency,
  which holds despite the handicap. Settling the rest needs both arms in
  containers or both on the host.
- **The CPU numbers in sections 1 and 2 are a single run each.** Arms were
  run in one
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
- **71% of every forward pass is padding.** Sequence length is fixed at 64
  while the workload's real token counts run 11 to 30. Fixed shapes are what
  let Triton batch without ragged-batching support and what put every arm on
  equal footing, but they mean roughly three quarters of the compute in every
  measurement above is spent on padding. Sequence-length bucketing would
  change every absolute number here, though it applies equally to all arms
  and so is unlikely to reverse a direction.
- **One CPU instance everywhere.** `instance_group` count is held at 1, and
  Ray's replica count at 1, so none of these results say anything about
  scaling out model instances, which is the other half of a real serving
  configuration.
- **Ray Serve is not measured at all.** It could not be run on this host: the
  Python installation is from the Microsoft Store, which executes inside an
  AppContainer sandbox that denies the `AssignProcessToJobObject` privilege
  Ray's process manager requires. The arm is implemented and the comparison
  script supports it; it needs a non-Store Python, a Linux host, or WSL.

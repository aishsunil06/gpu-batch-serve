# gpu-batch-serve: what dynamic batching actually buys

**Dynamic batching is sold as a throughput win. It is really a trade: a
latency cost paid on every request in exchange for a throughput gain that
only arrives once traffic is concurrent enough to fill a batch. Below the
crossover the cost is all you get. On CPU at concurrency 1, turning
batching on cost 44% of throughput and nearly doubled median latency, and
the mean batch size never rose above 1.0 — the server waited for company
that never came.**

An embedding service is the clearest place to measure this, because one
request is one small forward pass and a GPU is nowhere near saturated by
it. That is also the shape of real ranking and recommendation traffic:
many independent clients each asking for one vector, none of them in a
position to batch their own work.

The repository exists to answer one question with measurements:

> At what concurrency does dynamic batching start paying for itself, and
> what does each knob cost on the way there?

Full numbers, with the command that reproduces each, are in
[docs/FINDINGS.md](docs/FINDINGS.md).

## What is being served

`sentence-transformers/all-MiniLM-L6-v2`, a 6-layer encoder, behind a
FastAPI service. One HTTP request carries one text and returns one
mean-pooled, L2-normalised vector. Padding tokens are excluded from the
pooling, so an embedding never depends on the length of whatever else
shared its batch — without that, every result would be batch-size
dependent and the experiment would be measuring its own artefact.

## The arms

Each arm is a separate server process, so no arm inherits a warm allocator
pool or another arm's statistics.

| Arm | Max batch | Queue delay | Precision | What it isolates |
|---|---:|---:|---|---|
| A | 1 | 0 ms | fp32 | Baseline: one request, one forward pass |
| B | 8 | 0 ms | fp32 | Coalescing with no waiting at all |
| C | 8 | 5 ms | fp32 | What the grace period adds on top of B |
| D | 32 | 5 ms | fp32 | What a wider cap adds on top of C |
| E | 32 | 5 ms | fp16 | What half precision adds on top of D |

Each arm runs at concurrency 1, 8, 32 and 64, three times, on the same
seeded workload, so every arm meets exactly the same input.

## Headline results

### CPU reference run

Verified on an Intel Core laptop, 2 arms, 1 repeat, short windows — enough
to establish the crossover exists, not enough for standard errors.

| Arm | Concurrency | RPS | p50 ms | p95 ms | Mean batch |
|---|---:|---:|---:|---:|---:|
| A: no batching | 1 | 107.4 | 8.6 | 14.7 | 1.0 |
| C: batch 8, 5 ms | 1 | **60.1** | **15.9** | 27.5 | **1.0** |
| A: no batching | 8 | 120.3 | 62.9 | 84.9 | 1.0 |
| C: batch 8, 5 ms | 8 | **241.0** | 33.2 | **37.3** | **8.0** |

Two things in that table are the whole point of the project:

1. **At concurrency 1, batching is pure cost.** Throughput fell 44% and
   median latency rose 85%, and the mean batch size column explains why:
   it stayed at 1.0, so every request paid the 5 ms grace period and then
   ran alone anyway. A deployment that enables batching for a low-traffic
   model is strictly worse off, and nothing in its latency dashboard will
   say batching is the reason.

2. **At concurrency 8, batching improved throughput and tail latency at
   the same time.** 2.0x the throughput with p95 falling from 84.9 ms to
   37.3 ms. This is the result that is easy to state backwards: batching
   is usually described as trading latency for throughput, but once the
   server is the bottleneck, the queue in front of an unbatched server is
   a worse source of latency than the batch delay ever was.

### GPU run

**Not yet measured.** The harness is GPU-ready and `notebooks/colab_run.ipynb`
runs the full grid on a Colab T4, including GPU utilisation sampling and
the fp16 arm, which cannot run on CPU. Until that run is recorded in
[docs/FINDINGS.md](docs/FINDINGS.md), every number in this README is a CPU
number and the fp16 arm is unevaluated. The expectation is that the
crossover moves *left* on a GPU, because a GPU's cost per batch is flatter
in batch size than a CPU's, but an expectation is not a measurement.

## Reproducing

```bash
pip install -r requirements-dev.txt

python -m pytest tests/ -q                   # 30 tests
python scripts/run_bench.py --device cpu --quick --arms A,C --concurrencies 1,8
python scripts/run_bench.py --device cuda    # full grid, needs an NVIDIA GPU
```

Results land in `bench/results/latest.json` with every individual run
preserved, and a rendered table in `bench/results/latest.md`.

On Colab, open `notebooks/colab_run.ipynb` and set
`Runtime > Change runtime type > T4 GPU`.

## Why it is set up this way

**Closed-loop load generation, not open-loop.** A fixed number of virtual
clients each wait for a response before sending again, which holds exactly
`concurrency` requests in flight. An open-loop generator at a fixed arrival
rate would let a slow arm build an unbounded queue and then report a
latency that describes the backlog rather than the server, which would make
arm A look arbitrarily bad at high load for reasons unrelated to batching.

**Nearest-rank percentiles, not interpolated.** An interpolated p99 over a
few hundred samples reports a number no request actually experienced.

**Warmup is discarded, and every batch shape is warmed before the port
opens.** cuDNN selects algorithms per input shape on first sight and caches
them. An unwarmed first request pays a one-off cost that has nothing to do
with steady-state serving, and at concurrency 1 over a short window that
single request is a visible fraction of the sample.

**Mean batch size is reported in every row.** Without it, "batching did not
help here" and "batching did not happen here" are indistinguishable, and
they call for opposite fixes.

**Standard error, not standard deviation.** The question a reader has is
whether two arms differ, and the error on the mean is what answers it.

## What this does not measure

- **Concurrent model instances.** Multiple copies of the model on one GPU
  is the other half of a serving configuration, and this project varies
  only batching and precision. One instance per process is the control.
- **Accuracy under fp16.** The fp16 arm measures speed only. Half precision
  does perturb embeddings, and a cosine-similarity comparison against the
  fp32 vectors is the missing check before anyone would ship arm E.
- **A production inference server.** NVIDIA Triton and vLLM implement
  dynamic batching with far more sophistication than the ~60-line collector
  here. The batcher in this repository exists to make the trade visible and
  measurable, not to compete with them.
- **Cold start.** Model load time is excluded from every measurement, and
  it is the dominant cost of an autoscaling event.

## Layout

```
src/gpu_batch_serve/
    config.py        environment-driven configuration, one knob per arm
    engine.py        batch of strings in, batch of vectors out
    batcher.py       the queue and the collector loop: the experiment
    api.py           FastAPI surface, one text per request
    gpu.py           nvidia-smi utilisation sampling
    bench/
        workload.py  deterministic mixed-length queries
        load.py      closed-loop generator, percentiles
        report.py    aggregation and markdown rendering
scripts/run_bench.py the sweep: one server process per arm
notebooks/           Colab runner
tests/               30 tests, no GPU or model download required
```

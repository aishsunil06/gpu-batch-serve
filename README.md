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

## Part two: is a hand-rolled batcher worth keeping?

Having built a batcher and measured what it buys, the obvious next question
is whether building it was a good idea. Two production systems implement the
same policy, so the comparison is available.

All four arms load **byte-identical ONNX weights**, with batch cap and queue
delay set to the same values, driven by the same pre-tokenized fixture.
Anything that differs between them belongs to the serving implementation
rather than the model, the runtime, or the work.

| Arm | Implementation | Transport |
|---|---|---|
| `local` | the ~60-line collector in this repository | HTTP |
| `ray` | Ray Serve's `@serve.batch` | HTTP |
| `triton` | NVIDIA Triton's `dynamic_batching` | HTTP |
| `triton-grpc` | the same Triton | gRPC |

The knobs correspond exactly, which is the point:

```
DynamicBatcher(max_batch_size=32, max_queue_delay_ms=5)
@serve.batch(max_batch_size=32, batch_wait_timeout_s=0.005)
dynamic_batching { max_queue_delay_microseconds: 5000 }   # max_batch_size: 32
```

### Three decisions that make it a fair comparison

**Pooling is fused into the ONNX graph.** Were the model to stop at
`last_hidden_state`, every arm would return a 64x384 tensor, roughly 98 KB
serialised, against 1.5 KB for a pooled vector. The benchmark would then be
measuring JSON encoding, and Triton would be penalised for nothing it did.

**Tokenization happens offline, once, into a shared fixture.** It is CPU work
that precedes any batching decision and is identical in every arm, so leaving
it in the request path would add a constant per-request cost that shrinks the
measured difference between arms for an irrelevant reason. It also lets a Go
client participate without a tokenizer.

**Sequence length is fixed at 64.** A fixed input shape is what lets Triton's
dynamic batcher coalesce arbitrary requests without ragged-batching support,
and it puts every arm on identical footing. The cost is real: **71% of every
forward pass is padding** on this workload, which is exactly what
sequence-length bucketing exists to fix and which this project does not do.

### The achieved-batch-size problem

Every arm must report the batch size it actually formed, or "batching did not
help" and "batching did not happen" are indistinguishable. Each exposes it
differently:

- `local` and `ray` count batches and requests directly.
- `triton` does not report it at all. It is derived from two Prometheus
  counters, `nv_inference_request_success / nv_inference_exec_count`, and
  because those counters are monotonic, a per-window figure requires
  differencing a before and after snapshot. Dividing the raw totals would
  report the batch size since the container started.

That derivation is the most error-prone part of the harness, so it is
covered by tests against real Triton metrics output.

### Running the comparison

Each arm has a different lifecycle, so the comparison script does not start
them. Start what you want to measure, then run it; unreachable arms are
skipped with a note rather than failing the run.

```bash
pip install -r requirements-compare.txt

python scripts/export_onnx.py          # 86 MB graph, verified against torch
python scripts/prepare_fixture.py      # shared pre-tokenized workload

# Arm 1, the control
GBS_MAX_BATCH_SIZE=32 GBS_MAX_QUEUE_DELAY_MS=5 \
  python -m uvicorn serving.local.app:get_app --factory --port 8100

# Arm 2
python serving/ray/app.py --max-batch-size 32 --batch-wait-ms 5 --block

# Arms 3 and 4, plus Prometheus and Grafana
docker compose -f deploy/docker-compose.yml up -d

python scripts/compare_servers.py --quick --arms local,triton
python scripts/compare_servers.py --repeats 3
```

Grafana is on `http://localhost:3000` (anonymous viewer, no login) with a
provisioned dashboard whose first panel is achieved batch size. Prometheus is
on `:9090`, Triton's raw metrics on `:8002/metrics`.

**Disk:** the Triton image is 9.16 GB compressed and roughly 17 GB extracted.
NVIDIA does not publish a slim CPU-only tag; current docs say CPU-only
containers require a custom build. The standard `py3` image runs CPU-only
when no `--gpus` flag is passed, which is how `deploy/docker-compose.yml`
uses it.

### Results

**A larger achieved batch size turned out to be evidence of a worse server,
not a better one.** Full numbers in [docs/FINDINGS.md](docs/FINDINGS.md)
section 4.

| Arm | Conc. 8 RPS | Conc. 8 batch | Conc. 32 RPS | Conc. 32 batch |
|---|---:|---:|---:|---:|
| local | 169.1 | **8.0** | **238.0** | **31.7** |
| triton | **194.6** | 5.4 | 207.7 | 16.4 |
| triton-grpc | 190.3 | 6.2 | 215.0 | 15.9 |

Triton is 15% faster at concurrency 8 while forming batches a third
smaller. The collector only forms bigger batches because it blocks: nothing
is dequeued while a forward pass runs, so arrivals pile up and the next
batch is maximally full. Its batch of 31.7 at concurrency 32 is 32 clients
stalled behind one forward pass, not a batcher working well. Triton overlaps
input handling with execution, so its batch size converges to about half the
concurrency, which is the signature of a server that is not blocking.

Part 1 of this README treats mean batch size as the number that explains a
throughput result. Part 2 shows that is wrong, which is the most useful
thing the comparison produced.

It also produced a **refuted prediction**, recorded in full because the
first table alone would have shipped a wrong explanation: Triton's 16.4
looked like it was tracking `preferred_batch_size: [8,16,32]`, so narrowing
it to `[32]` should have raised the batch toward 32. It fell to 14.9
instead, ruling preference out and leaving overlap as the only explanation
consistent with both experiments.

**Read the cross-arm high-concurrency rows with care.** The local arm runs
natively on the host while Triton runs inside Docker Desktop's WSL2 VM, so
Triton pays a virtualised network hop the local arm does not. That handicap
plausibly contributes to the collector's win at concurrency 32 and 64. The
within-Triton experiment, the server-side batch counts, and Triton's
low-concurrency win all survive it; the rest does not.

`ray` is unmeasured: Microsoft Store Python runs in an AppContainer sandbox
that denies a privilege Ray's process manager needs. The arm is implemented
and needs a non-Store Python, Linux, or WSL.

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
    engine.py        torch engine: batch of strings in, vectors out
    onnx_engine.py   ONNX engine over pre-tokenized input, shared by all arms
    batcher.py       the queue and the collector loop: the experiment
    api.py           FastAPI surface, one text per request
    gpu.py           nvidia-smi utilisation sampling
    bench/
        workload.py  deterministic mixed-length queries
        load.py      closed-loop generator, percentiles
        report.py    aggregation and markdown rendering
serving/
    local/app.py     arm 1, the hand-rolled batcher over the ONNX graph
    ray/app.py       arm 2, Ray Serve's @serve.batch
    triton/          arm 3 and 4, model repository and config.pbtxt
deploy/              Triton, Prometheus, Grafana, and the dashboard
loadgen/main.go      Go closed-loop generator, HTTP, all arms
scripts/
    export_onnx.py   pooled encoder to ONNX, verified against torch
    prepare_fixture.py  offline tokenization into a shared fixture
    run_bench.py     the batching sweep: one server process per arm
    compare_servers.py  the four-arm comparison
notebooks/           Colab runner
tests/               52 tests, no GPU, Docker or model download required
```

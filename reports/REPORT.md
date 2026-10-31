# Stageplane implementation and measured evaluation

> Retained earlier experiment. Project identifiers and paths were updated during migration; measured values and token IDs were preserved. See [the fresh Stageplane verification](migration/REPORT.md).

Measured on **31 October 2026**. The end-to-end system downloaded the official
Qwen3.5-0.8B checkpoint, partitioned all its weights, generated with four workers,
and compared output with its native unpartitioned Transformers model.

**Correctness:** generated token IDs match across native, one-worker, and
four-worker execution in all five trials: **True**. The real-model worker
crash experiment also preserved the exact generated token sequence:
**True**. Four Docker workers participated in a separate matching
real-model run: **True**.

## What was built

1. A distributor for compatible HF repositories/URLs, local checkpoints, static
   HTTP checkpoint directories, and explicit Ollama Qwen3.5 aliases. It stores
   complete tensors in immutable SHA-256-verified Safetensors artifacts and a
   checkpoint/tokenizer-pinned execution manifest.
2. An executor with durable deployment metadata, bounded admission, weighted
   placement, configurable stage replicas, group-scoped membership rebalancing,
   leased session routes, committed-prefix recovery, OpenAI text chat/SSE, model
   import jobs, and a light dashboard with live Three.js topology and stage traces.
3. Workers that register/heartbeat, download and verify their assigned artifacts,
   materialize only assigned stages, retain native attention/recurrent state, and
   forward tensors directly over REST, gRPC, or WebSocket. Bearer auth and mutual
   TLS are exercised by real network tests.

Chunks are independent of node count. This model's 24 language blocks became
12 two-block chunks, plus embedding and normalization/head stages: **14 executable
stages**. A separate auxiliary artifact preserves the remaining checkpoint tensors.
One worker ran every stage; four workers hosted the same artifacts. Planner tests
also cover 100 workers without changing chunk count.

## Model and setup

- Source: [official Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B).
- Pinned revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Stored checkpoint artifacts: **1,746,941,336 bytes**
  (1.746941 GB; includes auxiliary weights).
- Checkpoint weights retain their original BF16/F32 values; execution used
  **float32 CPU** activations and parameters, greedy sampling, batch size one,
  and two PyTorch threads per executing process.
- Host: macOS-27.0.1-arm64-arm-64bit-Mach-O, 10 logical CPUs,
  16 GiB RAM. Exact software and memory/swap
  observations are in [environment.json](environment.json).
- Five measured trials after one warmup in each mode. Three prompts cycle across
  the trials. Output is capped at 16 tokens; the arithmetic prompt ends after two
  tokens. Identical prompt/output-length behavior is used in all modes.
- Qwen3.5-0.8B is the small current-generation checkpoint selected for this host;
  this report does not assert that it is the newest Qwen release at every size.

## Latency and memory

All times below are milliseconds. GB means 10^9 bytes. TTFT is time to first
token; decode excludes the first token. Throughput includes prefill and the actual
output length. Distributed TTFT is observed at the SSE client; native timing is
around the native forward loop. The one-worker API baseline helps expose serving
and stage-dispatch overhead independently of worker-to-worker network boundaries.

| Execution | TTFT p50 | TTFT p95 | Decode p50 | Decode p95 | Median tokens/s | Peak aggregate RSS GB | Largest worker/process RSS GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct-native | 1145.1 | 1184.8 | 1107.9 | 1610.2 | 0.839 | 4.178 | 4.178 |
| distributed-1 | 1439.0 | 1733.5 | 1091.3 | 1448.9 | 0.825 | 4.415 | 4.074 |
| distributed-4 | 1689.4 | 2078.3 | 1553.5 | 2190.1 | 0.617 | 5.163 | 1.672 |

The four-worker decode median is **1.40×** the direct baseline.
Its largest worker uses **60.0% less peak RSS** than the direct
process, while aggregate serving RSS includes extra runtimes, serialization,
and a second tied embedding copy. This run demonstrates a lower per-worker
residency requirement, with measured latency and total-memory costs.

RSS is sampled every 20 ms using psutil. Aggregate RSS includes the executor and
all serving workers and can count shared library pages more than once. It is not
system-wide physical RAM or GPU allocation. Warm inference and model-load peaks
are separated in native raw results; startup timings have different boundaries
and are not treated as an equal cold-network-download comparison.

## Storage and communication

| Execution | Logical serving artifact storage GB |
|---|---:|
| direct-native | 1.746941 |
| distributed-1 | 1.504832 |
| distributed-4 | 2.013391 |

Native storage above includes the complete preserved checkpoint. Worker storage
contains only assigned text-path artifacts. Four-worker storage grows because
the head and embedding live on different hosts and both require the tied matrix.
The registry additionally retains 1.746941 GB, so a deployment
with genuinely separate worker disks stores about
3.760332 GB in registry plus
worker artifacts, excluding runtime images, tokenizer metadata and filesystem
overhead. Additional replicas add their actual stage artifacts and session state.

The process benchmark hardlinks local cache files to avoid redundant same-disk
copies. The values above are **logical deployment storage**, not an assertion
that hardlinks consumed those bytes twice. Docker validation instead used actual
HTTP downloads into separate worker volumes.

Worker caches retain verified artifacts from previous placements. The four-worker
run's observed logical cache inventory was
3.438403 GB, compared with
2.013391 GB required by its final placement.
Both values are retained in raw results. Cache eviction and global cold-start
transfer budgets are not implemented; operators must account for this high-water
storage instead of treating final placement size as a cache limit.

The four-worker trials transferred 20.88–161.53 MB
of application data-plane payload per request. Raw trials contain exact values.
Counters sum outgoing request and returning response frames across gateway/worker
edges; they include envelopes and returned logits, exclude HTTP/TCP/TLS overhead,
and exclude control-plane heartbeats/artifact transfers. Colocated chunk calls
do not contribute network bytes. Final logits currently return along the RPC
chain, which is a material opportunity for further transport optimization.

## Failure and Docker experiments

The real-model fault probe removed **node-2** after output started.
It used one weight owner per stage, so recovery required cold replacement
placement rather than a preloaded full replica. The gateway performed
**1** recovery attempt(s), rebuilt the
committed prefix under a new epoch, and completed all
16 output tokens with exact token-ID parity. The largest
observed output gap was **3.594 s**, including
failure detection/replacement/replay as well as stage computation. Full events
and timestamps are in [fault.json](qwen/fault.json).

The four-container validation used the same 14 stages, separate downloaded
worker caches, mixed REST/gRPC/WebSocket transports, torch 2.15.0+cpu and
Transformers 5.22.1 inside Linux ARM64 containers. All four containers executed
stages. The arithmetic response was `4`
and token IDs matched native execution. Client elapsed time was
**33.547 s** for two generated tokens. This one cold/first
container request is reported separately from warm macOS process trials; it is
not used to infer virtualization overhead from a controlled comparison.
See [docker-validation.json](docker-validation.json) and
[service logs](docker-services.log). Test containers were stopped afterward.

## Verification evidence

- [pytest output](test-results.txt): 19 passed. Native prefill and cached-decode
  logit parity covers Qwen2, Qwen3, Qwen3.5 text, Llama and Mistral. Tests include
  mixed transports, one-worker placement, live worker loss/rejoin, group isolation,
  replica failure domains, stale epochs, leases, offsets, tensor bounds, artifact
  corruption, tenant cache separation, imports, removed-model metadata, mutual
  TLS, and actual OpenAI SDK streaming/non-streaming calls.
- [OpenAPI syntax lint](spec-lint.txt) and
  [generated collection lint](collection-lint.txt): no issues.
- [Postman smoke run](postman-smoke.txt): 3 requests and 9 assertions, 0 failures.
  The unmodified CLI output is retained. Governance lint/cloud publication were
  not run because this repository has no signed-in/bound Postman workspace.
- Browser QA verified a real streamed response, node inventory, stage assignments
  and Three.js rendering. [Current Stageplane dashboard](dashboard.jpg).
- Raw measurement files: [direct](qwen/direct.json),
  [one worker](qwen/distributed-1.json), [four workers](qwen/distributed-4.json),
  and [summary](qwen/summary.json).

## Interpretation and remaining boundaries

The experiment uses one shared Mac and loopback networking, with independent
worker processes. It does not establish physical multi-host network performance,
GPU performance, concurrent goodput, energy savings, task-quality equivalence
beyond tested outputs, or enterprise cost/productivity. The host had substantial
pre-existing and accumulated swap use; memory pressure and other applications
can affect timings. CPU hybrid-attention kernels used Transformers' correct
PyTorch fallbacks instead of optional CUDA-oriented optimized kernels.
Five trials provide limited tail-latency evidence: the displayed p95 values are
sample interpolations, and no p99/service-level conclusion is justified.

This is the implemented serving baseline, not every research extension in the
paper. The coordinator is single-writer SQLite, not consensus HA. Failover uses
prefix replay, not proactive cache replication. Public access is a shared token
and one default tenant; role-based tenant administration, device thermal/battery
policies, active link profiling, continuous batching, quantization, speculative
decoding, tensor/expert parallelism, and encoder-decoder/multimodal execution are
not implemented. Unsupported architectures fail explicitly. Ollama aliases use
original full-precision HF weights rather than reinterpreting GGUF quantization.
The [implementation map](../docs/architecture.md) and
[protocol](../docs/protocol.md) record these boundaries.

## Reproduce

```sh
uv sync --extra test
uv run --extra test pytest -q
uv run stageplane distribute Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 --layers-per-chunk 2
uv run python -m scripts.benchmark --model Qwen--Qwen3--5-0--8B-2fc06364715b --trials 5 --tokens 16
uv run python -m scripts.fault_probe --model Qwen--Qwen3--5-0--8B-2fc06364715b
uv run python -m scripts.write_report
```

Existing immutable artifacts should be reused instead of distributing the same
revision a second time. Docker validation is described in the repository README;
the final report renderer requires its retained evidence file.

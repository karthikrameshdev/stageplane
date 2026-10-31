# Qwen3 4B: 50 strictly limited heterogeneous workers

> Retained earlier experiment. Project identifiers and paths were updated during migration; measured values and token IDs were preserved. See [the fresh Stageplane verification](../migration/REPORT.md).

**The requested pool was actually started. It could not admit this model.**
Deployment returned HTTP 503 and inference returned HTTP
503. No model inference completed, so TTFT, decode latency
and tokens/second are **N/A**, not zero. Admission-response time is not model
inference latency.

## Actual configuration and outcome

- Requested workers: 50; attempted workers: 50.
- Memory tiers: 10 each at 150, 250, 350, 400 and 500 decimal MB for 50 workers.
- Every worker: Docker hard CPU quota 0.05 cores (50 millicores), one compute
  thread; memory+swap limit equals RAM limit, so **no swap** is permitted.
- Requested nominal pool capacity: 16.5 GB RAM and 2.5 CPU cores. PyTorch/runtime
  memory and availability reduce the capacity that can hold weights.
- Checkpoint, manifest, BF16 precision and `bounded-f32` kernel are the same as
  the [four-worker test](FOUR_NODE_REPORT.md). Image: `stageplane:qwen4-review`.
- Executor is separate: 768 MB / one CPU, not counted among worker resources.
- Registered workers at capture: 40.
- Docker-confirmed OOM-killed workers: 10.
- Storage fallback to 25 workers: **False**. A fallback is
  permitted only for an actual Docker no-space error, never for RAM/CPU failure.
- Capacity reason: `memory`.
- Console/API caution: **Not enough memory available to run this model. Kindly add more nodes.**

| RAM limit (decimal MB) | Containers attempted | Registered | OOM killed |
|---|---:|---:|---:|
| 150 | 10 | 0 | 10 |
| 250 | 10 | 10 | 0 |
| 350 | 10 | 10 | 0 |
| 400 | 10 | 10 | 0 |
| 500 | 10 | 10 | 0 |

## Why aggregate RAM did not suffice

The smallest supported execution unit is a complete compatible stage. The
embedding stage requires 777,912,456 weight bytes and the norm/output stage
777,917,696 bytes. Neither fits a 500 MB worker even before runtime/workspace.
Decoder stages each require about 201.9 MB of weights plus runtime and session
state. Some of the smallest workers cannot even initialize the PyTorch runtime
within their cap; the table reports the actual Docker OOM result.

Splitting the archive into 8,000 byte pieces would not make 8,000 executable
stages. This implementation supports multiple complete stages per worker and
does not implement out-of-core operator streaming or tensor-parallel embedding/
head execution. More workers with the same small maximum RAM cannot resolve
that per-stage requirement. Add workers with enough free RAM for the largest
stage and runtime, or implement a backend that subdivides those operators.

The executor retains a pending deployment and emits the memory caution. It
withholds partial-route assignments, avoiding downloads that cannot produce a
complete model route. CPU/storage cautions are separate and covered by automated
tests. No precision change, hidden larger worker or swapped container was used
to claim a successful 50-worker inference.

## Evidence and reproduction

- [Raw configuration, Docker states and executor snapshot](heterogeneous-50/result.json)
- Console and per-node logs: `reports/qwen3-4b/heterogeneous-50/`.
- [Local testing commands and resource semantics](../../docs/LOCAL_TESTING.md)
- [Retained artifact paths and manifest digest](artifacts.json)

Every container's Docker `Memory`, `MemorySwap` and `NanoCpus` values are in the
raw result, alongside exact worker registration/telemetry. Owned containers and
the private network are removed after capture. The shared imported chunks remain
at `/Users/karthikramesh/Projects/Self/OSS/stageplane/data/review/qwen3-4b/executor-models/Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1`.

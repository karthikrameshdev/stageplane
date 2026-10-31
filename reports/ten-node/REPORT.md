# Qwen3 4B: ten Docker workers

The ten-worker pool fits this model in memory with 2.5–3.5 GB RAM ceilings per
worker. The original **50-millicore CPU cap fails** the execution deadline; a
separate **one-core-per-worker comparison succeeds**. Both configurations have
zero OOM kills.

| Ten-worker configuration | Inference | Server TTFT | Server generation time | OOM kills |
|---|---|---:|---:|---:|
| 50m per worker | HTTP 503, chunk deadline | N/A | N/A | 0 |
| 1,000m per worker | HTTP 200, exact reference token parity | 92.908 s | 639.133 s | 0 |

The successful answer was **2 plus 2 is 4.** (nine tokens including EOS). This
is one functional request in each final CPU configuration, not a general latency
or task-quality benchmark. The successful configuration remains very slow on
this Docker CPU backend. Worker-stage execution dominates the recorded time;
these results do not establish general distributed-inference performance.

Generated 2026-10-31 19:17 IST.

**At 50 millicores: inference failed. At one CPU core: inference completed** (see the separately measured comparison below).

**50-millicore inference completed: False.** All 10
workers registered. Docker reported 0 OOM-killed workers. Deployment
returned HTTP 200; inference returned HTTP 503.
This is an actual ten-worker attempt; a 25-worker benchmark was not run.

## Configuration

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Pinned checkpoint: `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Manifest SHA-256: `5eab0bbfa7ed6cf7b4dc33c1aa9ce11e7592469d84d7a30ec4ffd1690572dddd`.
- Original BF16 weights, without quantization or a substitute model;
  `bounded-f32` CPU kernel. Linear weight tiles of at most 16 MiB use FP32
  accumulation and return BF16. Every worker uses one compute thread.
- 38 executable stages: 36 decoder blocks, embedding, and norm/output.
- Ten workers; each has a hard **50 millicore (0.05 CPU core)** quota, for
  **0.5 nominal worker CPU cores in total**.
- RAM ceilings: four workers at 2,500 MB, three at 3,000 MB and three at 3,500 MB
  (decimal units). Total configured ceilings are 29.5 GB, not reserved physical RAM.
- Container swap disabled: each worker's memory+swap limit equals its RAM limit.
- Docker VM: 14.636 GB RAM; host: 17.180 GB.
  The executor is separate, with a 768 MB RAM / one CPU limit.
- REST tensor transport, one group and one replica. Workers download and verify
  their assigned artifacts. Multiple stages can reside on each worker.
- Startup deadline 300 s; deployment-to-ready deadline 1,800 s. Execution-service
  chunk, RPC and retry defaults were retained. The report does not silently
  increase CPU quotas or change precision to obtain a successful result.

## Loading stability and validation

The first attempt registered all ten workers without OOM but reached only 37 of
38 ready stages before its 900-second readiness deadline. A resource-triggered
replan moved six stages while other workers were still loading. The executor now
defers resource-driven replans while a peer in the same placement group is
loading; other groups and membership changes remain eligible. The regression
test and full suite passed **36 tests**. The final retry keeps the same RAM, CPU
and model, with a 1,800-second loading allowance.

[First attempt evidence](../ten-node-first-attempt/result.json),
[final test output](tests.txt).

## Outcome and latency

Inference response: `{'detail': 'No ready replica for blocks-000'}`.

No inference completed in this 50-millicore case, so generation TTFT, token throughput and token parity
are **N/A**. Request/rejection and measurement-capture time was
52.794 s; that is not model inference latency.
Deployment-to-readiness wait was 1446.868 s.

## Memory, storage and stage distribution

Memory metrics are reported by the workers inside the VM. Cgroup usage includes
anonymous runtime/weights, kernel charges and file cache. RSS is a different
metric and should not be compared directly with cgroup usage or host swap.
Container swap is disabled; macOS can still compress/swap the VM's backing pages.
Firefox and Codex remained open, so host memory pressure affects timing.

| Worker | RAM ceiling MB | Assigned stages | Ready stages | Loaded weight GB | Last cgroup use GB | Last RSS GB | Observed cgroup peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| node-00 | 2500 | 4 | 4 | 0.807 | 1.185 | 1.163 | 1.503 |
| node-01 | 3000 | 0 | 0 | 0.000 | 1.449 | 1.415 | 1.714 |
| node-02 | 3500 | 5 | 5 | 1.009 | 1.451 | 1.423 | 1.849 |
| node-03 | 2500 | 5 | 5 | 1.009 | 1.383 | 1.391 | 1.743 |
| node-04 | 3000 | 5 | 3 | 1.182 | 1.703 | 1.563 | 2.115 |
| node-05 | 3500 | 2 | 2 | 0.404 | 0.787 | 0.771 | 1.230 |
| node-06 | 2500 | 2 | 2 | 0.404 | 0.781 | 0.771 | 1.224 |
| node-07 | 3000 | 7 | 7 | 1.989 | 2.495 | 2.363 | 2.998 |
| node-08 | 3500 | 5 | 3 | 0.606 | 1.088 | 0.977 | 1.573 |
| node-09 | 2500 | 3 | 2 | 0.404 | 0.853 | 0.769 | 1.228 |

The original request route used 10 workers. The table shows the final
post-failure placement: the timed-out worker was quarantined and its chunks
were assigned to cold replacements. An unassigned quarantined worker is not a
healthy spare. Having ten registered workers does not require exactly ten
chunks; one worker can host several stages.


- Unique producer weight artifacts: 8.045 GB.
- Last reported allocated worker weights: 7.814 GB.
- Last reported total worker cache files: 8.823 GB.
- Largest sampled aggregate worker cgroup usage: 13.587 GB.
- Largest sampled aggregate worker RSS: 12.752 GB.

Observations were collected about every 10 seconds; diagnostic timeouts can
create gaps. Per-container `memory.peak` was read periodically, including loading.
Those values are observed high-water marks up to each successful read, not a
complete final peak capture. A zero table value means no successful peak read.
The aggregate values exclude executor, Docker daemon and VM kernel memory and
can miss a transient maximum. The different workers' peaks need not occur together.
Tied embedding/output weights may require separate copies on separate workers.

## Comparison and limits

The prior 50-worker test exceeded physical VM memory, even after per-worker
RAM ceilings were increased. The ten-worker test reduces the number of Python/
PyTorch runtimes, while retaining the same full checkpoint and 50-millicore
worker quota. See [the previous report](../migration/REPORT.md).

The earlier direct/four-worker measurements used a different CPU allocation
and five requests. They cannot isolate the effect of changing only node count;
no general speedup or tail-latency conclusion follows from this single run.
The current backend executes whole compatible stages, with epoch/prefix replay
on bounded recovery. It does not implement out-of-core operators or tensor
parallel splitting of one stage.

## Evidence and reproduction

- [Raw executor, Docker settings and final state](result.json)
- [Sampled progress and memory](observations.jsonl)
- [Kernel cgroup peak observations](memory-peaks.jsonl)
- [Latest independent executor snapshot](snapshot-latest.json)
- Worker/executor logs remain in this directory locally; they are ignored by Git.

```sh
python -m scripts.resource_profile \
  --store data/review/qwen3-4b/executor-models \
  --model Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1 \
  --nodes 10 --memory-tiers-mb 2500,3000,3500 \
  --startup-timeout 300 --load-timeout 1800 --request-timeout 1800 \
  --output reports/ten-node
```

The ten-worker limit changes RAM ceilings only; it keeps the 50-millicore quota.
Model chunks remain at:

```text
/Users/karthikramesh/Projects/Self/OSS/stageplane/data/review/qwen3-4b/layer-models/Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1
```

The imported consumer copy remains at:

```text
/Users/karthikramesh/Projects/Self/OSS/stageplane/data/review/qwen3-4b/executor-models/Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1
```

Test containers and the private network are removed after evidence capture.

## Separate CPU comparison: ten workers with one core each

The 50-millicore run demonstrated that the model fits in memory with ten
workers, but the first decoder chunk (`blocks-000`, `node-01`) exceeded the
30-second compute/queue deadline. The execution service quarantined the failing
node and attempted one recovery; with one replica there was no ready replacement
for that chunk, so the request returned HTTP 503 without output tokens.
This was a compute/deadline failure, **not an OOM**.

The diagnostic comparison changes the per-worker CPU cap to **1 core (1,000
millicores)**, keeps one compute thread, the same RAM limits, precision, kernel,
checkpoint, stage count and execution deadlines. It does not claim a successful
50-millicore inference. A single request executes dependent stages in sequence;
ten 50-millicore workers do not behave like one 0.5-core CPU for its active stage.

| Measurement | 50m per worker | 1,000m per worker |
|---|---:|---:|
| Registered workers | 10 | 10 |
| Nominal worker CPU cores total | 0.5 | 10 |
| OOM-killed workers | 0 | 0 |
| Complete route loaded before inference | True | True |
| Deployment to ready (s) | 1446.868 | 121.408 |
| Inference HTTP status | 503 | 200 |
| Inference completed | False | True |
| Server TTFT (s) | N/A | 92.908 |
| Server generation elapsed (s) | N/A | 639.133 |
| Median decode interval (s) | N/A | 64.734 |
| Output tokens/s including prefill | N/A | 0.0141 |
| Generated tokens including EOS | 0 | 9 |
| Recoveries | 1, unsuccessful | 0 |

Answer: **2 plus 2 is 4.**

Exact generated-token parity with the retained native reference: **True**.

All 38 stages ran in the successful route. Sampled peak aggregate worker cgroup
usage was 13.444 GB; sampled peak
aggregate worker RSS was 13.131 GB.
These memory samples exclude the executor and VM kernel, and are subject to
sampling gaps. This remains a single functional request, not stable p95 evidence.

| Token | Token ID | Server token interval (s) | Sum of stage trace durations (s) |
|---|---:|---:|---:|
| 1 | 17 | 92.908 | 86.558 |
| 2 | 5519 | 82.302 | 80.517 |
| 3 | 220 | 68.778 | 67.776 |
| 4 | 17 | 62.783 | 61.811 |
| 5 | 374 | 55.100 | 54.243 |
| 6 | 220 | 64.895 | 64.025 |
| 7 | 19 | 64.572 | 63.621 |
| 8 | 13 | 63.215 | 62.469 |
| 9 | 151645 | 84.294 | 82.694 |

Stage traces account for approximately 97.6% of
recorded per-token wall time in this request. These are stage execution wall
intervals, including thread/resource scheduling and memory effects, not measured
CPU utilization. Remaining time includes routing, serialization and gateway work.
The Docker run uses Linux ARM64 CPU PyTorch, while the earlier direct/four-worker
baseline ran natively on macOS. Different CPU allocations, libraries, paging and
sample sizes prevent attributing that performance difference only to node count.

Reproduce the diagnostic CPU allocation separately:

```sh
python -m scripts.resource_profile \
  --store data/review/qwen3-4b/executor-models \
  --model Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1 \
  --nodes 10 --cpu-cores 1 --memory-tiers-mb 2500,3000,3500 \
  --startup-timeout 300 --load-timeout 1800 --request-timeout 1800 \
  --output reports/ten-node/cpu-1000m
```

Evidence: [comparison raw result](cpu-1000m/result.json),
[comparison summary](cpu-1000m/summary.json),
[comparison observations](cpu-1000m/observations.jsonl),
[Docker runtime versions](docker-environment.json).

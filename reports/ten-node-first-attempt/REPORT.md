# Qwen3 4B: ten Docker workers

Generated 2026-10-31 18:45 IST.

**Inference completed: False.** All 10
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
- Startup deadline 300 s; deployment-to-ready deadline 900 s. Execution-service
  chunk, RPC and retry defaults were retained. The report does not silently
  increase CPU quotas or change precision to obtain a successful result.

## Outcome and latency

Inference response: `{'detail': 'No ready replica for blocks-034'}`.

No inference completed, so generation TTFT, token throughput and token parity
are **N/A**. Request/rejection and measurement-capture time was
0.145 s; that is not model inference latency.
Deployment-to-readiness wait was 901.790 s.

## Memory, storage and stage distribution

Memory metrics are reported by the workers inside the VM. Cgroup usage includes
anonymous runtime/weights, kernel charges and file cache. RSS is a different
metric and should not be compared directly with cgroup usage or host swap.
Container swap is disabled; macOS can still compress/swap the VM's backing pages.
Firefox and Codex remained open, so host memory pressure affects timing.

| Worker | RAM ceiling MB | Assigned stages | Ready stages | Loaded weight GB | Last cgroup use GB | Last RSS GB | Observed cgroup peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| node-00 | 2500 | 5 | 5 | 1.009 | 1.607 | 1.613 | 1.842 |
| node-01 | 3000 | 4 | 4 | 0.807 | 1.176 | 1.181 | 1.649 |
| node-02 | 3500 | 7 | 7 | 1.413 | 1.814 | 1.818 | 2.034 |
| node-03 | 2500 | 4 | 4 | 0.807 | 1.625 | 1.630 | 1.951 |
| node-04 | 3000 | 5 | 5 | 1.585 | 2.172 | 2.174 | 2.613 |
| node-05 | 3500 | 2 | 2 | 0.404 | 0.786 | 0.791 | 1.226 |
| node-06 | 2500 | 0 | 0 | 0.000 | 0.760 | 0.753 | 1.257 |
| node-07 | 3000 | 6 | 6 | 1.787 | 2.158 | 2.155 | 2.391 |
| node-08 | 3500 | 3 | 3 | 0.606 | 0.954 | 0.962 | 1.216 |
| node-09 | 2500 | 2 | 1 | 0.202 | 0.691 | 0.556 | 0.705 |

The placement uses 9 workers and leaves 1 as spares.
Having ten registered workers does not require exactly ten chunks or force a
stage onto every worker.

- Unique producer weight artifacts: 8.045 GB.
- Last reported allocated worker weights: 8.621 GB.
- Last reported total worker cache files: 9.832 GB.
- Largest sampled aggregate worker cgroup usage: 13.872 GB.
- Largest sampled aggregate worker RSS: 13.685 GB.

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
  --startup-timeout 300 --load-timeout 900 \
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

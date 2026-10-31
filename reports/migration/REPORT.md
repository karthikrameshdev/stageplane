# Stageplane migration and fresh test report

The implementation and retained artifacts now live at:

```text
/Users/karthikramesh/Projects/Self/OSS/stageplane
```

The existing repository's Git history, Apache license and
`https://github.com/karthikrameshdev/stageplane.git` remote were preserved.
No Git push or public artifact upload was performed.

## Complete project identity

- Python distribution, import package, installed CLI and Docker entrypoint:
  `stageplane` (`python -m stageplane` also works).
- Runtime settings and authentication: `STAGEPLANE_*` environment variables.
- gRPC service: `/stageplane.Worker/Forward`; internal error marker:
  `STAGEPLANE_ERROR` followed by newline.
- Published bundles: `.stageplane`; catalog format: `stageplane-catalog-v1`.
- Dashboard, API title/model ownership, OpenAPI specification, generated
  collections, Compose names/images, test certificates, docs and scripts all
  use Stageplane. Every component must use these current identifiers.
- A new virtual environment and wheel were built from this checkout. The wheel
  includes the package, console entrypoint and vendored UI assets.
- Retained historical results have their project identifiers/path labels updated;
  their timing values and token IDs were preserved. The results below are fresh
  executions from this checkout. Current screenshots replace the earlier branding.

## Automated and contract validation

The complete automated suite passed **35 tests** from the new virtual environment.
Coverage includes native prefill/decode parity, bundle imports and integrity,
resource telemetry/default publication, placement/rebalancing, bounded retries,
mixed REST/gRPC/WebSocket transport, OpenAI SDK calls and mutual TLS.
The increased-RAM run exposed transient CPU exhaustion during weight loading;
the fix retains existing loading/session assignments and excludes busy workers
from new assignments. A regression test covers this behavior.

- [Original pytest output](tests.txt)
- [OpenAPI syntax lint](spec-lint.txt): no issues.
- [Regenerated collection lint](collection-lint.txt): no issues, 14 requests.
- [Live executor smoke output](postman-smoke.txt): 3 requests, 9 assertions,
  zero failures.
- [Installed package and wheel identity](package.json)
- [Model/hash/archive verification](artifact-verification.json)
- [Identity audit](identity-audit.json)

Host runtime: Python `3.12.14`, PyTorch
`2.15.0`, Transformers `5.22.1`.
The Docker image uses Python 3.13. The host has 10 logical CPUs and 16 GiB RAM.

## Fresh Qwen3 4B: direct versus four workers

Checkpoint: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
BF16 weights; no quantization or substitute model. Both modes use the explicit
`bounded-f32` CPU kernel: at most 16 MiB weight tiles are converted to FP32 for
linear accumulation, then outputs return to BF16. This is not a comparison of
vanilla BF16 kernel performance. Direct mode is an unpartitioned Transformers
model with two compute threads. Four workers use REST, two threads each, one
group/replica, verified local hardlinks and 6 GiB serving budgets per worker.
These worker budgets share one physical Mac and do not reserve 24 GiB of RAM.
The four-worker timings were collected before the loading-state scheduling guard
was added; serving kernels and wire format are unchanged. The full 35-test suite
and final higher-RAM Docker retry exercise that guard.

**Exact generated-token parity: True across all five trials.**
Each mode had an excluded warmup; measured requests generated at most 16 new
tokens, stopping at EOS. Math requests returned `2 plus 2 is 4.`.

| Metric | Direct | Four workers |
|---|---:|---:|
| Median first-token latency (s) | 0.936 | 1.512 |
| Interpolated p95 first-token latency (s) | 1.233 | 2.410 |
| Median decode interval (s) | 0.651 | 1.389 |
| Interpolated p95 decode interval (s) | 1.845 | 11.065 |
| Median whole request (s) | 11.310 | 19.380 |
| Median output tokens/s, including prefill | 1.415 | 0.755 |
| Observed aggregate peak RSS (decimal GB) | 8.201 | 8.823 |
| Largest worker peak RSS (decimal GB) | 8.201 | 2.699 |
| Logical serving storage (decimal GB) | 8.045 | 8.823 |
| Load / complete-route-ready time (s) | 19.232 | 9.687 |

| Trial | Tokens | Direct request (s) | Four-worker request (s) | Exact parity |
|---|---:|---:|---:|---|
| 1 | 9 | 6.342 | 13.604 | True |
| 2 | 16 | 11.310 | 21.185 | True |
| 3 | 16 | 12.062 | 19.380 | True |
| 4 | 9 | 14.319 | 9.226 | True |
| 5 | 16 | 10.799 | 133.092 | True |

Raw evidence: [summary](four-node/summary.json), [direct](four-node/direct.json),
[four workers](four-node/distributed-4.json). RSS excludes macOS compressed and
swapped pages; paging and background applications affect latency. Five trials
do not establish stable tail latency or performance on separate physical hosts.
Tied embedding/head weights share storage in direct mode; separate worker
processes may require separate copies. Hardlinks reduce local disk duplication,
while remote hosts need their own assigned files.

## Fifty workers, including the higher-RAM retry

Both runs keep 50 workers and a strict **0.05 CPU cores per worker**, one compute
thread, BF16 precision, and container swap disabled. RAM limits are explicit
ceilings rather than reserved physical RAM. The higher-RAM run is a separate
attempt requested after the original OOMs; it is not presented as the same profile.

An initial higher-RAM loading probe registered all 50 nodes without OOM, then
exposed the loading-state scheduling issue described above. Its unmodified
[raw evidence](heterogeneous-50-increased-ram-loading-probe/result.json) is
retained separately. The table's increased-RAM row is the subsequent fixed run.

| Attempt | RAM tiers (decimal MB) | Registered | OOM killed | Deployment / inference HTTP | Inference completed |
|---|---|---:|---:|---|---|
| Original | 150, 250, 350, 400, 500 | 40 | 10 | 503 / 503 | False |
| Increased RAM | 750, 1000, 1250, 2000, 2500 | 50 | 5 | 200 / 503 | False |

### Original result

Docker VM RAM: 14.636 GB; host RAM: 17.180 GB. Storage fallback to 25: False.

Inference did not complete: `{'detail': 'Not enough memory available to run this model. Kindly add more nodes.'}`. TTFT and throughput are **N/A**. HTTP rejection/response capture time is not inference latency.

[Full Docker limits/states, placement and telemetry](heterogeneous-50/result.json). Executor and individual worker logs are retained beside that file.

### Increased RAM result

Docker VM RAM: 14.636 GB; host RAM: 17.180 GB. Storage fallback to 25: False.

**Partial capture:** Docker log collection timed out after the inference request. The status codes above come from the retained executor log, the 50 registrations from a captured live observation, and the five OOM kills from Linux kernel records. The complete final registry and timing snapshot was lost; no such snapshot or inference timing is claimed. The harness now persists measurements before optional log collection, and a regression test verifies preservation on log timeout.

Inference did not complete: `{'capture_note': 'HTTP status recovered from executor log; response body not retained.'}`. TTFT and throughput are **N/A**. HTTP rejection/response capture time is not inference latency.

[Reconstructed summary and evidence limitations](heterogeneous-50-increased-ram/result.json). Executor and individual worker logs are retained beside that file.

The largest stages require about 778 MB of weights each; decoder stages require
about 202 MB each, in addition to runtime, loading and session workspace. A route
needs every complete stage. Increasing container caps does not create physical
RAM in the Docker VM, and arbitrary byte slices are not executable stages.
The backend does not implement out-of-core operators or tensor-parallel splitting
of the embedding/output stage. State recovery uses bounded executor retries and
committed-prefix replay, not live KV-state migration.

The final increased-RAM run hit **VM-wide physical exhaustion**. The kernel
recorded `constraint=CONSTRAINT_NONE` and `global_oom`, rather than a container
exceeding its own RAM cap. Five Stageplane processes were killed. See
[the kernel evidence](vm-global-oom.json), [the live observation](loading-observation.json),
and [one retained Docker OOM inspect record](remaining-oom-container.json).
Increasing container limits further cannot create RAM inside the 14.636 GB VM.
The successful four-worker benchmark is valid; there is no successful 50-worker
inference benchmark on this host. More physical worker/VM memory is required.

## Retained artifacts and reproduction

All five retained model directories passed manifest/file hash verification; the
portable archive passed ZIP CRC verification. The Qwen4 manifest SHA-256 remains
`5eab0bbfa7ed6cf7b4dc33c1aa9ce11e7592469d84d7a30ec4ffd1690572dddd`.

Chunks:

```text
/Users/karthikramesh/Projects/Self/OSS/stageplane/data/review/qwen3-4b/layer-models/Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1
```

Consumer copy:

```text
/Users/karthikramesh/Projects/Self/OSS/stageplane/data/review/qwen3-4b/executor-models/Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1
```

Uploadable archive:

```text
/Users/karthikramesh/Projects/Self/OSS/stageplane/data/review/qwen3-4b/public/qwen3-4b.stageplane
```

See [the detailed local testing README](../../docs/LOCAL_TESTING.md) for install,
publishing, imports, worker limits and recovery. Fresh four-worker benchmark:

```sh
python -m scripts.benchmark \
  --store data/review/qwen3-4b/executor-models \
  --model Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1 \
  --output reports/migration/four-node \
  --dtype bfloat16 --cpu-kernel bounded-f32 --memory-gb 6 \
  --skip-single --trials 5 --tokens 16
```

The higher-RAM command is in the local guide. Render this report again with
`python -m scripts.write_migration_report`. Model artifacts remain on disk;
test services/containers are cleaned up afterward. Firefox and Codex remain open.

Current UI proof: [dashboard](../dashboard-full.jpg),
[published chunk import](../qwen3-4b/bundle-import.jpg).

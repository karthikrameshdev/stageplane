# Stageplane 0.2.0: enterprise readiness report

Measured 3 November 2026 on macOS arm64 (10 CPUs, 16 GiB RAM, about 3 GB free during runs).
Model: official Qwen3.5-0.8B (`Qwen--Qwen3--5-0--8B-2fc06364715b`, 24 layers, 14 stages), bfloat16
on CPU, greedy decoding, 12 output tokens.

## 1. Gaps found in the audit and how each was closed

| Area | Gap in 0.1.0 | Change in 0.2.0 |
|---|---|---|
| Placement | Contiguous hash placement; node size only weighted the hash | `capacity` strategy fills nodes proportionally, interleaving layers (verified below) |
| Routing | Workers forward to each other in a fixed chain | `hub` routing: executor forwards every cross-node hop; `chain` retained |
| Small devices | Every stage fully resident in RAM | `--stream-weights`: layers on disk, node-wide pinned cache, one-layer prefetch |
| Streaming cache | (first version used LRU, which never hits on a cyclic layer sweep) | Pin-until-full policy: hit rate equals cache size divided by working set |
| Long context | 2,048-token cap, prompt sent as one frame, float32 full-context KV reservation | 32,768 default (bounded by the model), chunked prefill, dtype-aware reservation |
| OpenAI API | No `stop`, penalties, `max_completion_tokens`, usage-in-stream, `/v1/models/{id}`, structured errors; only one EOS id | All added; multi-EOS from `generation_config.json`; `error` object; `Retry-After` |
| Observability | No metrics, no request correlation, ad-hoc logging | Prometheus `/metrics` on executor and workers, `X-Request-ID`, JSON logs |
| Health | One `/health`; worker health needed credentials | `/livez` and `/ready` (unauthenticated) on both, `stageplane probe` healthcheck |
| Governance | No record of who changed what | Durable `audit` table and `/api/audit` |
| Lifecycle | Deployments could not be listed or removed | `GET /api/deployments`, `GET`/`DELETE /api/deployments/{model}` |
| Multi-tenancy | One tenant could occupy every admission slot | `STAGEPLANE_TENANT_MAX_REQUESTS`, counted before waiting for a slot |
| Shutdown | Nodes vanished only after the heartbeat timeout | Workers deregister on graceful shutdown; configurable grace period |
| Containers | Root user, no healthcheck, writable root | UID 10001, healthcheck, Compose `read_only`, `cap_drop: ALL`, `no-new-privileges` |
| Orchestration | Compose only | Kubernetes Deployment, StatefulSet, Services, probes, restricted security context |
| Quality gates | No linter, unused imports | Ruff in CI and `pyproject.toml`; repository is clean |
| Release hygiene | Version only in code | 0.2.0 across package, lock, and API; `--version`; `CHANGELOG.md` |

A code review of the diff found three real defects, and all three are fixed with regression tests.
Each test fails when its fix is reverted and passes when it is restored.
1. The tenant quota could be exceeded by requests waiting for admission.
2. A stage that failed its readiness fixture leaked the streamed-layer budget it had pinned.
3. The planner did not reserve RAM for a streaming node's layer cache.

## 2. Real-model results (final code)

Raw data: [heterogeneous.json](heterogeneous.json). Prompts: two short ones and one with 2,154 tokens.

| Scenario | Nodes and budgets | Identical tokens to resident | Decode tok/s | Peak RSS per worker (GB) | Long-prompt TTFT (s) |
|---|---|---|---:|---|---:|
| single-resident (reference) | 1 x 3 GiB, chain | reference | 0.44 | 1.69 | 77.9 |
| single-resident-unchunked | same, prompt sent in 1 frame | 3/3 | 0.44 | 1.71 | 76.2 |
| heterogeneous-hub-interleaved | 1.4 GiB + 1.4 GiB + 0.7 GiB, all streaming | 3/3 | 0.64 | 0.37 / 0.87 / 0.20 | 81.9 |
| single-streaming-edge | 1 x 1.8 GiB, 96 MiB layer cache | 3/3 | 0.44 | 0.84 | 79.8 |

The interleaved route for a single token was
`node-0, node-0, node-1, node-2, node-2, node-1, node-0, node-2, node-0, node-2, node-1, node-1, node-0, node-1`.
Ownership was non-contiguous and followed node size:

- node-0 (1.4 GiB) held blocks 000, 005, 007, 011 and the embedding.
- node-1 (1.4 GiB) held blocks 001, 004, 009, 010 and the head.
- node-2 (0.7 GiB) held blocks 002, 003, 006 and 008. Its 32 MiB cache pinned nothing, so it streamed every layer from disk.

The 2,154-token prompt ran as 9 prefill frames of 256 tokens, and the output matched the single-frame run exactly.

What this shows:
- None of the three heterogeneous nodes could hold the 2 GB model. Together they served it correctly, and the largest worker used 48% less peak RSS than the resident reference.
- A single 1.8 GiB device with a 96 MiB layer cache ran the whole model with 50% less peak RSS at the same decode rate. The decode rate was this good only because the OS page cache served the repeated disk reads.
- Hub routing added no measurable decode penalty on loopback; the three-process pool decoded faster because it used more cores. This does not predict performance over a real network.
- On this host, long-prompt TTFT is dominated by CPU prefill, which is about 78 to 82 s for 2,154 tokens in every configuration.

## 3. Verification

| Check | Result |
|---|---|
| `uv run --extra test pytest -q` | **187 passed** (167 from before this work, 20 new) |
| `uvx ruff check .` | All checks passed |
| `node --check stageplane/ui/app.js` | OK |
| `postman spec lint ... --issueType=syntax` | No issues. Governance lint needs `postman login`, which was not available |
| `postman collection lint postman/collections/Smoke` | No issues (8 requests) |
| `postman collection run` Smoke against a live 2-node cluster | **8 requests, 32 assertions, 0 failed** ([output](postman-smoke.txt)) |
| Official `openai` Python SDK against the live cluster | Model retrieval, `create`, streaming with usage chunk, typed 404/422 errors ([output](openai-sdk.txt)) |
| `scripts.artifact_delivery_validation` with the new non-root image | Passed ([evidence](artifact-delivery)) |
| Hardened container (`--read-only --cap-drop ALL`, UID 10001) | Healthy; `/livez` 200 with `X-Request-ID`; `/metrics` without a token returns 401 |
| `kubectl apply --dry-run=client -f deploy/kubernetes/stageplane.yaml` | All six objects valid |
| Real-model heterogeneous validation | 12 of 12 generations token-identical |

## 4. Remaining limits

- **Executor high availability.** There is one executor using SQLite, so it remains the single point of control. Consensus and stream continuation after an executor failure are not implemented.
- **Throughput.** Batch size is one, with no continuous batching, paged KV, tensor parallelism or prefix caching across requests. TTFT for long prompts is limited by prefill.
- **Hub routing on a real network.** It adds one executor round trip per node boundary. It was measured only on loopback; for multi-host latency-sensitive pools, use `chain` with contiguous placement.
- **Streaming cache policy.** Pinning is static first-come; the learned hot-layer set (the colibri idea) is not built. A node with too small a cache reads every layer from disk on every token.
- **Supported models.** Only the architectures and checkpoints listed in the README are supported. Tools, the Responses API and multimodal inputs are not implemented.
- **Postman.** Governance lint and publishing to a Postman workspace need a login and a workspace id, neither of which is configured.
- **Upgrade behaviour.** Existing persisted deployments default to `hub` routing after upgrade. Re-apply them with `"routing": "chain"` to keep the 0.1.0 behaviour; `CHANGELOG.md` explains this.

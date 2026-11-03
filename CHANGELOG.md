# Changelog

All notable changes are recorded here. The project follows [Semantic Versioning](https://semver.org/);
before 1.0 a minor version may change defaults.

## 0.2.0

### Placement and execution
- `strategy: "capacity"` (new default) interleaves layer chunks across nodes in proportion to free
  memory, so a node can own non-contiguous layers (for example 1, 6, 8) and tiny devices can host one layer.
  `strategy: "rendezvous"` keeps the previous weighted hash.
- `routing: "hub"` (new default) returns activations to the executor at every node boundary;
  `routing: "chain"` keeps direct worker-to-worker forwarding.
- `node --stream-weights --layer-cache SIZE`: layers stay on disk; a node-wide budget pins as many
  materialized layers as fit and streams the rest with one-layer prefetch. Output is bit-identical.
- Chunked prefill (`STAGEPLANE_PREFILL_CHUNK`, default 512). Default `STAGEPLANE_MAX_CONTEXT` raised from
  2,048 to 32,768 and further bounded by the model's `max_position_embeddings`.
- Worker session reservations use the stage's real element size and the prefill chunk, instead of a
  float32 full-context estimate.

### OpenAI compatibility
- `max_completion_tokens`, `stop`, `presence_penalty`, `frequency_penalty`, `n=1`, `user`,
  `stream_options.include_usage`, `developer` role, text-part message content, and `/v1/models/{id}`.
- Generation stops on every `eos_token_id` from `generation_config.json` and reports `finish_reason` correctly.
- `/v1/*` errors use the OpenAI `error` object (the `detail` field is retained); 429/503 include `Retry-After`.
- Streams start with `role: "assistant"`; responses include `system_fingerprint`.

### Operations
- `/livez`, `/ready` and Prometheus `/metrics` on the executor and workers.
- `X-Request-ID` propagation, `--log-format json`, `--log-level`, and `--version`.
- Durable audit log (`/api/audit`) for imports, deployments, drains, and resumes.
- `GET /api/deployments`, `GET` and `DELETE /api/deployments/{model}`.
- Per-tenant concurrency quota (`STAGEPLANE_TENANT_MAX_REQUESTS`) and `STAGEPLANE_ADMISSION_WAIT`.
- Workers deregister on graceful shutdown; uvicorn graceful shutdown is configurable.
- Container runs as UID 10001 with a `stageplane probe` healthcheck; Compose drops capabilities and uses
  read-only root filesystems. Kubernetes manifests under `deploy/kubernetes/`.
- Dashboard exposes placement strategy, routing, and streaming-node residency.

### Changed defaults to review when upgrading
- New deployments use `capacity` placement and `hub` routing. Existing persisted deployments keep
  `rendezvous` placement until re-applied; their routing becomes `hub`. Re-apply with `"routing": "chain"`
  to keep direct forwarding.
- `/health` now also returns `version`.

## 0.1.0
- Initial release: immutable stage artifacts, weighted placement, replicas, recovery, OpenAI chat, dashboard.

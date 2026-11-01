# Execution protocol

## Immutable artifacts

`data/models/<key>/manifest.json` identifies the resolved checkpoint revision,
configuration, tokenizer hash, layout/dtype, layer intervals, and every file's
size and SHA-256. The manifest digest hashes its sorted JSON without the digest
field. Assignments pin that digest. Workers verify it and each artifact before
loading. The trusted executor and its authenticated channel are the authority;
hashes are integrity checks, not signatures or remote execution attestations.

Published `.stageplane` archives, directories and catalogs retain this same manifest.
Executors can import them from local paths, HTTP, Git/LFS and public Google Drive
files. Every artifact is verified before an atomic install; an optional external
manifest digest pins publisher identity. Imports never execute repository code.

Chunks keep complete tensors and original names. Native layers retain their
original indices and position encodings. Embedding and head stages reference the
same tied-weight artifact. Colocated stages share materialized tensor storage;
separate hosts need separate copies. Auxiliary tensors are preserved but are not
placed on a text route.

## Control plane

Registration and heartbeats carry endpoints, selected transport, group, allocatable
serving memory, exact CPU cores, speed, failure domain, ready stages, and metrics.
Omitted resource flags resolve to 100% of the detected host/container capacity;
registration still carries numeric byte/core values. Metrics distinguish those
limits from current usage, available memory/CPU/storage and loaded weight bytes.
Workers also publish `loading` while initializing a stage and for a two-second
settling interval afterward. Existing loading/session reservations survive a
transient CPU-headroom dip; new assignments still require available compute.
Resource-driven replans wait while a worker in the same placement group is
loading, so a spare's falling headroom does not restart an otherwise healthy
bootstrap. Other groups, membership changes and explicit drains remain eligible.
Responses contain
`{model, chunk, manifest_sha256}` assignments. Loading runs separately from
heartbeats. A shape/finite-output fixture precedes readiness. Whole-route numerical
parity comes from tests, not process health.

Deployments select model, replica target, and group count. Weighted rendezvous
uses current CPU headroom, declared speed and observed stage cost; conservative
weight/workspace and storage estimates constrain placement. Membership changes
replan only affected groups. Idle workers with material resource changes can
trigger a group-local replan at most once per 30 seconds. Missing capacity stays
unready and emits distinct memory, compute or storage cautions. Partial routes
receive no assignments. Healthy requests retain their route while it is in use. Previous ready owners
remain available during feasible replacement loading, with overlapping capacity
reserved until new targets are ready and active sessions have released old copies.

## Data plane

All transports carry identical bytes:

1. Four-byte little-endian unsigned JSON-header length.
2. UTF-8 JSON envelope of at most 64 KiB.
3. Safetensors containing a tensor named `value`.

Frames are bounded to 64 MiB and batch size is one. Embeddings receive `[1,S]`
int64 token IDs; blocks/head receive `[1,S,H]` floating-point tensors. The head
returns `[1,1,V]` float32 logits for the last sequence position. No pickle is used.
Invalid frame size, dimensions, IDs, revisions, offsets, stale epochs, and expired
leases fail before computation. A large hidden width can make the frame bound
stricter than the nominal context limit.

Envelope fields: `tenant`, `session`, `model`, `revision`, `manifest_sha256`,
`epoch`, `offset`, `context_budget`, `lease_until`, ordered `route`, current
`index`, and timing/shape-only `trace`. Each hop names node, chunk, URL, gRPC
endpoint, and transport. The first worker receives tokens and forwards activations
directly to successors. Colocated chunks execute locally. Only final logits and
trace metadata return to the gateway.

REST uses `POST /v1/forward` with `application/octet-stream`. gRPC uses the generic
byte unary method `/stageplane.Worker/Forward`. WebSocket uses `/ws/forward`, one binary
request and response. Errors propagate through REST 409, gRPC failed-precondition,
or a binary `STAGEPLANE_ERROR\n` JSON error frame on WebSocket. Internal errors carry
the actual failed node/chunk through preceding workers to the gateway.

## Session ownership, commitment and recovery

State is keyed by `(tenant, session, model:chunk)` plus a session epoch. Native
DynamicCache stores owned attention and recurrent state. Decode offsets must
match the cached prefix. A new epoch clears old state before replay. Admission
reserves estimated KV, recurrent state, and workspace within serving memory.
This is a conservative budget, not a paged-cache allocator.

The gateway samples/commits only after the full route succeeds. A failed forward
identifies the failed node/chunk, quarantines that node from replacement routes
for 30 seconds, increments the epoch, and replays the prompt plus committed output
on a complete ready route. A ready alternative is used immediately. The executor
allows three retries by default (`STAGEPLANE_EXECUTION_RETRIES`); replacement readiness
waits at most 10 seconds (`STAGEPLANE_RECOVERY_WAIT_SECONDS`). Per-chunk compute/queue
deadlines default to 30 seconds (`STAGEPLANE_CHUNK_TIMEOUT`), and transport operations
to 180 seconds (`STAGEPLANE_ROUTE_RPC_TIMEOUT`). Delivered tokens are never resampled
or duplicated. Computation may repeat. Without replacement capacity, the request
fails explicitly. A timed-out native kernel cannot be forcibly cancelled: its
compute permit is retained until it exits, and its abandoned session is discarded.

Healthy sessions retain routes during rebalancing. Draining workers retain stages
used by live sessions. Cancellation/completion release caches and reservations;
stale-epoch tombstones remain for a bounded TTL (default 120 seconds). Route leases
also expire so workers cannot execute indefinitely without authority.

## Operating boundary

Non-loopback listeners require `STAGEPLANE_TOKEN`. The internal token grants worker and control access.
`STAGEPLANE_API_KEYS_FILE` adds separate admin, viewer and tenant inference keys;
see [the operator guide](OPERATIONS.md). mTLS verifies channels and certificates on all
three transports. Admin/viewer credentials are cluster-wide operator roles. Inference credentials
select a configured tenant for cache ownership. Certificate revocation/discovery,
per-tenant quotas and model entitlements remain future work. Managed workers and artifact
sources must be trusted. TLS does not hide activations from participating workers.

SQLite preserves deployment metadata across executor restart. In-flight client
streams do not transfer to a replacement gateway. The coordinator is single-writer,
not a Raft cluster. Cache state is not proactively replicated; recovery uses
prefix replay. Unsupported model architectures fail instead of being silently
routed through incompatible layers.

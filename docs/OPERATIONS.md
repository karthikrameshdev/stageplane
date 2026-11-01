# Operating Stageplane

Stageplane executes supported dense text transformer stages across a trusted
worker pool. The control plane assigns stages and admits requests; workers keep
their own session state and forward activations. Increasing worker count can add
capacity and replicas. A single autoregressive request still traverses its layers
in order, so adding workers does not imply lower token latency.

## Credentials and organization boundaries

Provision a random `STAGEPLANE_TOKEN` on the executor and managed workers. This
internal credential can register workers and download artifacts. Keep it separate
from credentials given to API consumers. Enable TLS or mutual TLS as described
in the [main README](../README.md#mutual-tls).

On the executor, set `STAGEPLANE_API_KEYS_FILE` to a file outside the repository:

```json
[
  {"token": "REPLACE_WITH_RANDOM_OPERATOR_KEY", "tenant": "operations", "role": "admin"},
  {"token": "REPLACE_WITH_RANDOM_VIEWER_KEY", "tenant": "operations", "role": "viewer"},
  {"token": "REPLACE_WITH_RANDOM_ORG_A_KEY", "tenant": "org-a", "role": "inference"},
  {"token": "REPLACE_WITH_RANDOM_ORG_B_KEY", "tenant": "org-b", "role": "inference"}
]
```

Use independent random secrets, restrict file access to the service account, and
restart the executor after changing the file. The placeholders above are not
production credentials. Invalid or duplicate entries stop startup. The file
requires the internal worker token to be configured.

| Credential | Access |
| --- | --- |
| Internal token | Worker registration, heartbeats, artifacts, control and inference |
| Admin key | Model import/export, deployments, drains, operations and inference |
| Viewer key | Cluster operations, request metadata, events and model list |
| Inference key | Model list and chat completions for its configured tenant |

Admin and viewer are **cluster operator roles**: they can inspect operational
metadata across organizations. Tenant inference keys cannot access the operations
snapshot or events. The authenticated tenant is used in worker cache ownership;
the caller cannot select a different tenant through its chat payload. Model
availability remains shared across tenants. Per-tenant model entitlements,
quotas, billing, SSO and immutable audit logs are not implemented.

Without an API key file, the legacy internal token continues to work. Without
either token or key file, only the loopback development configuration is intended.
The CLI requires an internal token for non-loopback listeners. Executor JSON
bodies default to a 2 MiB limit, adjustable with `STAGEPLANE_MAX_REQUEST_BYTES`
between 1 KiB and 64 MiB.

## Placement and rebalancing

Workers publish exact memory and CPU budgets, current headroom, available disk,
loaded stages, session count, session capacity and loading state. Set
`--max-memory` and `--max-cpu` when starting each node; omitted limits resolve to
the detected host/container capacity. Bare-process memory budgets control
admission; use container or OS limits for hard enforcement.

The planner reserves capacity across all deployments. Stages hash to stable
placement groups. Membership changes and drains replan affected groups; resource
pressure is compared with the last planning baseline so gradual changes are not
missed. Resource changes are rate limited to avoid repeated movement during load.
Use the same failure-domain identifier for workers sharing a physical failure
unit, and distinct identifiers for independent hosts or racks. Colocated processes
also share host RAM and CPU; enforce aggregate budgets with your container scheduler.

During a feasible migration, healthy previous copies remain assigned while new
copies load and verify. They retire after replacements become ready and live
sessions release their pins. Migration needs spare capacity for overlapping
copies. Adding a node does not create capacity inside a different placement group.
The dashboard distinguishes blocked, loading, degraded, rebalancing and ready
deployments, including target readiness and remaining transitions. A serving
route can be degraded: redundancy counts only active, non-draining workers that
have verified their copy. Target readiness waits for every assigned copy to load.

The executor owns bounded retries. A failed or timed-out chunk moves the request
to a ready route and replays committed context under a new epoch. Worker REST
rejections and transport failures preserve the failing node/chunk identity;
transports do not retry independently. Native kernels cannot be forcibly stopped
by Python cancellation. Their compute permits and memory reservations remain
accounted for until completion; supervise worker processes to restart genuinely
stuck native runtimes.

## Model publication

Use the [local testing guide](LOCAL_TESTING.md) for producer, bundle publishing,
remote import and execution commands. Checkpoints are read in bounded ranges,
validated, written to hidden staging directories, and atomically published only
after the manifest is complete. Malformed shapes/offsets, duplicate tensors,
missing language layers and insufficient storage fail before publication.
Failed attempts are cleaned up; interrupted whole-model builds currently restart.

### Central store and worker copies

The producer writes a model registry with `--store /srv/stageplane/models`.
An executor can read those same files from its own mount with
`--store /mnt/central/models`; the path names need not match across machines.
Provision the mount and permissions outside Stageplane. For dedicated roles,
three machines are one producer, one executor, and one worker; ten machines are
one producer, one executor, and eight workers. More workers do not change the
immutable stage boundaries.

Set `STAGEPLANE_STATE_FILE` to a writable **local** path on the executor. Its
default is inside `--store`; using a separate path avoids putting SQLite/WAL on
a shared filesystem or a read-only artifact mount. The executor needs write
permission to `--store` only when it imports/distributes models there. Keep the
model registry available for manifest discovery and authenticated fallback even
when all workers normally download from a central HTTP server.

An executor advertises its absolute store path by default. Repeat
`--artifact-source ROOT` to advertise explicit central filesystem/HTTP(S) roots
instead. Each root must contain `MODEL_KEY/FILENAME`. A worker can set its own
roots to account for mount aliases or preferred mirrors, for example
`--artifact-source /mnt/models --artifact-source https://models.example/models`.
The worker checks its verified cache first, then its own roots, then advertised
roots, then the authenticated executor artifact endpoint. Explicit roots never
disable that final fallback. Sources are bounded to eight roots per list.

Workers keep copies under `--cache/NODE_ID/MODEL_KEY/`; changing the source does
not change cache ownership. Normal transfers verify SHA-256 and size before
atomic publication, including local filesystem copies. A worker only fetches
artifacts needed for assigned stages. Budget local cache disk separately from
central model storage, serving weight memory, and temporary transfer space.

Direct HTTP sources receive no executor bearer token, client certificate,
cookies, or proxy settings. URLs with credentials, query parameters, or fragments
are rejected. Provision a worker-reachable endpoint with trusted HTTPS
certificates or a read-only filesystem mount; otherwise rely on executor fallback.
Do not place tokens, private configuration, or executor SQLite state in a publicly
served model root. Direct artifact access does not replace control-plane trust.

Local filesystem operations run off the worker event loop, so slow copies do not
block heartbeats. A genuinely unresponsive NFS/SMB syscall is governed by OS/mount
timeouts; cancellation waits for in-flight file I/O before cleaning up.

The direct client's connect/read timeouts are three/ten seconds. A network-failed
HTTP origin cools down for 30 seconds to avoid repeating a slow mirror attempt
for every chunk. Missing or corrupt files are skipped and later sources tried.
Failures writing the local cache are reported as destination failures, not
hidden by repeatedly downloading the same bytes from different sources.

Workers must reach the executor for registration, manifests, heartbeats, and
fallback downloads. Executors and workers must reach the advertised worker
HTTP endpoints and selected gRPC/WebSocket transport endpoints for execution;
central storage connectivity alone does not establish a usable inference route.
See [network requirements](LOCAL_TESTING.md#network-reachability) and the
[HTTPS directory example](LOCAL_TESTING.md#central-https-directory).

Inspect `nodes[].metrics.artifact_transfers` in authenticated `/api/state` to
distinguish cache reuse, direct-local/HTTP copies, and executor transfers.
`fallbacks` counts attempts; successful executor copies are counted separately.
Counters reset on worker restart. Pair these counters with readiness and loading
errors, rather than interpreting an empty cache after reassignment as proof that
central downloads succeeded. See the [counter definitions](LOCAL_TESTING.md#confirm-which-source-served-a-worker).

The default import budget is 256 GiB (`STAGEPLANE_BUNDLE_MAX_BYTES`). Header metadata
is capped separately by `STAGEPLANE_MAX_TOTAL_HEADER_BYTES` (128 MiB by default).
Increase limits deliberately for larger trusted models and provision the required
storage. Original checkpoints, staged copies, published archives and worker
caches can each consume disk; hardlinks only save physical space on the same
filesystem. The dataset/model license continues to apply to published weights.

## Large-model capacity and current boundaries

These are weight-only lower bounds at two bytes per parameter:

| Parameter count | One BF16 weight copy | Two independent replicas |
| --- | --- | --- |
| 100 billion | 200 GB | 400 GB |
| 1 trillion | 2 TB | 4 TB |
| 10 trillion | 20 TB | 40 TB |

These decimal byte totals exclude KV/recurrent caches, activations, runtime
workspaces, process overhead, duplicated tied tensors and staging storage. FP32
materialization uses about twice the BF16 weight memory. CPU cores or aggregate
RAM alone do not determine whether a model fits: **each complete assigned stage
must fit on an eligible worker**. A large dense layer cannot be split across many
small-memory workers by cutting its file into arbitrary byte fragments.

The runtime currently supports whole-layer pipeline execution, batch size one,
64 MiB tensor frames, 64 KiB metadata envelopes and at most 1,024 route hops.
The envelope limit can become restrictive before the hop limit. Very wide models
or long prefills can exceed frame limits. The runtime builds a model skeleton on
the meta device when constructing a stage; metadata costs still grow with model
architecture size.

Tensor parallelism, expert parallelism/MoE, paged KV caches, continuous batching,
distributed prefill and route-ID-based metadata are not implemented. The executor
is a single writer with SQLite persistence. Multiple independent executors must
use independent worker pools; launching several against one pool is not a
supported HA deployment. Consensus leadership, durable request ownership and
stream continuation across executor failure remain future work.

Consequently, the synthetic 100B/1T/10T tests establish **placement arithmetic and
group isolation**, not execution support or an enterprise service-level guarantee.
Actual deployment certification requires a supported checkpoint, sufficiently
large workers, real network/storage measurements and failure/load testing on the
target hardware. See the [validation report](../reports/hardening/REPORT.md).

## Verification and rollout

```sh
uv sync --extra test --locked
uv run --extra test pytest -q
uv run python -m scripts.scale_validation
STAGEPLANE_TOKEN=configuration-check-only docker compose config --quiet
```

Start with the tiny offline fixture and verify a complete ready route. Inspect
active request age, first-token latency, token rate, worker CPU relative to its
published cores, RAM headroom, background computations and transition status.
Drain a worker, verify replacement readiness, then test a hard loss with replicas.
Measure p50/p95/p99 latency and failure rates under representative concurrent
traffic before setting an admission limit or advertising an SLA. The dashboard's
CPU/RAM utilization is measured resource use; it is not a model-quality or energy
efficiency score. The topology intentionally shows a bounded subset at large
cluster sizes; use the paginated inventory for all workers.

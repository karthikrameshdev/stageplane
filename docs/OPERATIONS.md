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

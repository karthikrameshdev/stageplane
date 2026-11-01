# Test central artifact delivery and published models

Choose one of two workflows:

- **Central store:** the producer writes chunks once, the executor reads them
  through its own filesystem mount, and workers fetch directly from mounted or
  HTTP(S) sources before falling back to the executor.
- **Portable publication:** export a `.stageplane` bundle, directory, or catalog;
  another executor imports it from local storage, HTTP, Git/LFS, or public Google
  Drive. Imported files can then be served through the same direct-first workflow.

The [main README](../README.md#quick-start) contains the complete multi-machine
Qwen3.5-0.8B path, from distribution through an OpenAI-compatible request. This
guide explains source selection, local reproduction, and larger Qwen3 4B tests.

## Central artifact workflow

### Machine roles and directory layout

Machine counts include the roles you deploy. With dedicated roles, three machines
mean **one producer, one executor, and one worker**. Ten machines mean **one
producer, one executor, and eight workers**. A three-worker Docker test instead
uses three worker containers plus its executor and publisher on one host; it does
not measure three independent physical machines.

The producer can write to any provisioned directory:

```sh
stageplane distribute Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --layers-per-chunk 2 --store /srv/stageplane/models
```

The resulting central layout is:

```text
/srv/stageplane/models/
└── Qwen--Qwen3--5-0--8B-2fc06364715b/
    ├── manifest.json
    ├── embedding.safetensors
    ├── blocks-000.safetensors
    ├── ...
    └── configuration, tokenizer, license, and other declared artifacts
```

Use the actual filenames in `manifest.json`; tied embedding/head stages may
reference one artifact. Do not rename chunks or alter the immutable manifest.

| Option | Process | Meaning |
| --- | --- | --- |
| `--store /srv/stageplane/models` | Producer | Write the complete immutable model registry here |
| `--store /mnt/central/models` | Executor | Read the registry visible at this local mount; import operations also write here |
| `--artifact-source ROOT` | Executor | Advertise a worker-readable directory root; repeat for mirrors |
| `--artifact-source ROOT` | Worker | Try this machine's preferred root before advertised roots; repeat for mirrors |
| `--cache data/node-cache` | Worker | Keep verified local copies under `data/node-cache/NODE_ID/MODEL_KEY/` |

`--store` already existed. Direct-first worker acquisition is the additional
provision supplied by `--artifact-source`. Neither flag mounts NFS/SMB, creates a
shared volume, nor infers how another host names its mounts. Provision that
infrastructure before starting the services.

### Exact fetch order

For each required artifact, a worker follows this order:

1. Its existing local cache, if the manifest's SHA-256 **and byte size** match.
2. Its own `--artifact-source` roots, in command-line order.
3. Roots advertised by the executor, in command-line order.
4. Authenticated `GET /api/artifacts/MODEL_KEY/FILENAME` on its executor.

Duplicate roots are tried only once. Each list accepts at most eight roots. Without
explicit executor roots, the executor advertises the absolute path of its
`--store`. Supplying executor roots replaces that default advertisement, but
**never removes executor HTTP fallback**. A worker's own roots precede advertised
ones; they do not replace them. Assignment and manifest authority remains with
the authenticated executor; a storage mirror cannot select a different model.

A root is an absolute filesystem path, a local `file://` directory URL, or an
HTTP(S) directory URL. It must resolve `ROOT/MODEL_KEY/FILENAME`. Do not pass a
single-model directory, `manifest.json`, archive, Git URL, or Drive share link as
a direct root. Those published-model forms use `import-bundle` instead.

Missing files, invalid hashes/sizes, and inaccessible sources cause the worker to
try the next source. Direct HTTP connects time out after three seconds, with a
ten-second read timeout; connection/network failures cool down that origin for
30 seconds. Source errors do not prevent executor fallback. A full or unwritable
**destination cache** still fails explicitly; changing the source cannot fix it.
All normal local/HTTP copies use temporary files, verify size and hash, and
atomically replace the cache file only after success. Shared-source files are
copied into the cache; workers do not execute directly from mutable shared files.

Direct HTTP uses a separate client without the executor's bearer token, client
certificate, cookies, or environment proxies. Root URLs cannot contain embedded
credentials, query strings, or fragments, so signed query URLs are not direct
roots. Use a network-accessible central endpoint with trusted HTTPS certificates,
a mounted filesystem for private storage, or the authenticated executor fallback.
An HTTPS mirror does not replace worker/executor authentication.

### Different mount paths

Suppose the same central files are visible at these locations:

| Machine | Local view of the central model root |
| --- | --- |
| Producer | `/srv/stageplane/models` |
| Executor | `/mnt/central/models` |
| Worker A | `/mnt/models` |
| Worker B | No shared filesystem mount |

Start the executor with a local state file outside the shared registry:

```sh
export STAGEPLANE_TOKEN='replace-with-a-random-shared-internal-token'
export STAGEPLANE_STATE_FILE="$PWD/data/executor-state/executor.sqlite"
stageplane executor --store /mnt/central/models --host 0.0.0.0 --port 8080
```

Worker A maps its different mount explicitly:

```sh
export STAGEPLANE_TOKEN='replace-with-a-random-shared-internal-token'
stageplane node --id worker-a --executor http://executor.example:8080 \
  --host 0.0.0.0 --port 8101 --advertise-url http://worker-a.example:8101 \
  --artifact-source /mnt/models --cache data/node-cache \
  --max-memory 8GiB --max-cpu 2 --group 0
```

Worker B can omit `--artifact-source`; the nonexistent advertised filesystem root
fails locally and the executor serves its assigned files. It still requires local
cache disk space. These are independent copies: a shared model directory does not
eliminate worker cache or weight-memory requirements.

### Central HTTPS directory

An HTTP(S) source supplements the executor's local registry. The executor still
needs its mounted or independently imported copy so it can read manifests and
serve fallback downloads. Expose **only model artifacts**, not tokens, SQLite
state, private configuration, or unrelated directories.

For example, with an existing Nginx installation and a certificate trusted by
workers, map `/models/` to the producer's `/srv/stageplane/models/`:

```nginx
server {
    listen 443 ssl;
    server_name models.example;
    ssl_certificate /etc/nginx/tls/models.crt;
    ssl_certificate_key /etc/nginx/tls/models.key;
    root /srv/stageplane;
    location /models/ {
        autoindex off;
        limit_except GET { deny all; }
    }
    location / { return 404; }
}
```

The server account needs read access. An artifact is reachable at
`https://models.example/models/MODEL_KEY/FILENAME`. A normal static GET server is
sufficient; direct chunk downloads do not require range support or a Stageplane
service on the storage host.

Advertise the central HTTPS root, optionally followed by another mirror:

```sh
stageplane executor --store /mnt/central/models --host 0.0.0.0 --port 8080 \
  --artifact-source https://models.example/models \
  --artifact-source https://models-backup.example/models
```

A worker can prefer its local mount and use the advertised HTTP mirrors if that
mount is missing or has an invalid copy:

```sh
stageplane node --id worker-a --executor http://executor.example:8080 \
  --host 0.0.0.0 --port 8101 --advertise-url http://worker-a.example:8101 \
  --artifact-source /mnt/models --cache data/node-cache \
  --max-memory 8GiB --max-cpu 2
```

Keep the token and local `STAGEPLANE_STATE_FILE` environment settings from the
previous examples. Configure TLS/mTLS for control and worker links separately.

### Network reachability

Direct model download and activation forwarding are separate connections.
Central HTTP availability alone is not enough to run inference.

| Connection | Required access |
| --- | --- |
| API client/browser → executor | Executor HTTP(S), normally port 8080 |
| Worker → executor | Registration, heartbeats, manifests, artifact fallback |
| Worker → central storage | Its chosen filesystem mount or HTTP(S) source |
| Executor/worker → worker | Advertised HTTP(S) endpoint; required for REST/WebSocket execution and management |
| Executor/worker → gRPC worker | Advertised gRPC endpoint, in addition to the worker's HTTP listener |

Use routable hostnames/IPs in `--advertise-url` and `--grpc-url`. `127.0.0.1` refers
to the current machine/container and is only appropriate when services truly
share that network namespace. For Docker, use service/container DNS names on a
shared network. For gRPC, set both `--grpc-port 9101` and a reachable
`--grpc-url worker-a.example:9101 --transport grpc`. For WebSocket,
`--transport websocket` uses the worker's advertised HTTP listener. Firewall and
certificate names must cover every required link; Stageplane does not create
firewall rules or discover NAT mappings.

### Confirm which source served a worker

An authenticated `GET /api/state` includes each node's
`metrics.artifact_transfers`. Inspect the counters after assignments become ready:

```sh
curl --fail-with-body http://127.0.0.1:8080/api/state \
  -H "Authorization: Bearer $STAGEPLANE_TOKEN" \
  | python -c 'import json,sys; s=json.load(sys.stdin); print(json.dumps({n["id"]:n.get("metrics",{}).get("artifact_transfers",{}) for n in s["nodes"]},indent=2))'
```

`direct_local` and `direct_http` count verified files copied from central roots;
`executor` counts verified files received from the executor. `cache_hits` counts
successful reuse of existing verified cache files. `direct_failures` records
failed source attempts, while `fallbacks` counts entry into executor fallback
after configured direct sources were exhausted or skipped. A fallback attempt
does not itself prove a successful transfer; check `executor`, loading errors,
and stage readiness. `bytes_direct` and `bytes_executor` count successfully
verified bytes. Counters reset on worker restart, so a warm-cache restart can
show cache hits without new downloads. They are counts, not throughput metrics.

### Three-worker Docker source and fallback test

This test creates a tiny offline checkpoint and three Docker workers, then checks
actual mounted-file, central HTTP, and executor fallback paths in an isolated
network. It is a delivery/correctness test, not a Qwen performance measurement.
It does not download a public checkpoint or reuse an existing live demo.

```sh
uv sync --extra test --locked
docker build -t stageplane:local .
uv run python -m scripts.artifact_delivery_validation \
  --image stageplane:local --output reports/artifact-delivery
```

The script removes only its own containers/network by default and retains its
report artifacts. To keep its test services available for inspection on port
8082, use `--keep`; ensure that port is free first:

```sh
uv run python -m scripts.artifact_delivery_validation \
  --image stageplane:local --output reports/artifact-delivery --keep
```

Read the generated report for actual source counters, inference results, and
failures. The commands above are reproduction instructions, not a claim that a
particular machine has passed them. The separate procedures below preserve the
published Qwen3 4B review workflow and historical resource measurements.

## Published Qwen3 4B workflow

## Requirements and fixed review locations

Use Python 3.11+ (3.12 and 3.13 tested), Git, and `uv`. Docker Desktop is needed only for the strictly
limited heterogeneous container test. The tested model is
`Qwen/Qwen3-4B-Instruct-2507`, pinned to
`cdbee75f17c01a7cc42f958dc650907174af0554`. Its original BF16 weights occupy about
8.04 GB. Use at least 35 GB free disk space for a producer, archive, consumer copy,
and temporary import download. Keep more headroom when Docker shares that disk.

The working tree's persistent review root is:

```text
/Users/karthikramesh/Projects/Self/OSS/stageplane/data/review/qwen3-4b/
├── models/             # Initial two-layer producer artifacts retained for review
├── layer-models/       # Active one-layer producer chunks and manifest
├── public/             # Portable qwen3-4b.stageplane archive for uploading
└── executor-models/    # Independently imported consumer artifacts
```

These paths are ignored by Git so multi-gigabyte weights are never accidentally
committed. They remain on disk after the tests. Other users can use the same
relative paths inside their checkout. Results and exact configurations are under
`reports/qwen3-4b/`.

## 1. Install

```sh
cd /path/to/stageplane
uv sync --extra test --locked
source .venv/bin/activate
```

`stageplane` is installed as the console command. All steps can alternatively use
`uv run stageplane ...` or `python -m stageplane ...`.

## 2. Producer: download and create reusable stages

For a fresh run, create one decoder block per chunk directly:

```sh
stageplane distribute Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --layers-per-chunk 1 \
  --store data/review/qwen3-4b/layer-models
```

The printed immutable model key identifies the directory containing all stage
files, tokenizer/configuration/license files, and `manifest.json`. There are 36
decoder-block stages, one embedding stage, and one output stage: 38 executable
stages. A chunk is a complete compatible block interval, not an arbitrary byte
slice of a checkpoint. Tied embeddings are referenced by both ends of the route.

If a two-block version was already created, reuse its tensors instead of
redownloading:

```sh
stageplane rechunk \
  data/review/qwen3-4b/models/Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0 \
  --layers-per-chunk 1 --store data/review/qwen3-4b/layer-models
```

The retained review run uses the rechunked key:

```text
Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1
```

A fresh direct one-block distribution prints the key without `-l1`. Use the key
printed by your own command in subsequent examples. Rechunking preserves the
checkpoint identity and records the original manifest digest. Existing immutable
model revisions are reused rather than overwritten.

## 3. Publisher: export the verified model

```sh
stageplane export Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1 \
  --store data/review/qwen3-4b/layer-models \
  --output data/review/qwen3-4b/public/qwen3-4b.stageplane
```

The `.stageplane` file is an uncompressed ZIP64 archive, so hosting does not need any
model-specific server. It contains the immutable manifest and every declared
artifact. Export verifies the source file hashes before packaging and prints the
manifest SHA-256. Share that digest separately when consumers need to pin the
publisher's exact manifest. A self-reported hash alone does not authenticate an
unknown publisher.

Upload the archive to a public object store, static HTTP server, Git release, or
Google Drive file with public download permission. For GitHub releases, upload
individual stage files through a catalog rather than an 8 GB archive: each
[release asset must be under 2 GiB](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases). Stageplane does not change sharing
permissions or upload files to another user's account automatically. Keep the
model's license with its artifacts.

To publish a directory instead of an archive:

```sh
stageplane export MODEL_KEY --store data/review/qwen3-4b/layer-models \
  --format directory --output data/review/qwen3-4b/public/model-directory
```

Directory hosts need only ordinary GET requests. Range requests are required by
the original-checkpoint distributor, but not by the published-bundle importer.

## 4. Simulate a public host locally

Keep this terminal open:

```sh
python -m http.server 8801 --bind 127.0.0.1 \
  --directory data/review/qwen3-4b/public
```

Another machine needs the server's reachable address instead of `127.0.0.1`.
For a real public deployment, use a storage service or authenticated web server
appropriate to your environment.

## 5. Consumer: import without the producer's model directory

In another terminal:

```sh
stageplane import-bundle http://127.0.0.1:8801/qwen3-4b.stageplane \
  --store data/review/qwen3-4b/executor-models \
  --manifest-sha256 PUBLISHERS_MANIFEST_SHA256
```

Replace the hash placeholder with the producer's printed digest, or omit that
option when deliberately trusting the chosen host. The review run's digest is
recorded in `reports/qwen3-4b/artifacts.json`.

The importer downloads into temporary storage, rejects unsafe archive paths,
validates the complete manifest and file inventory, checks every artifact's
SHA-256 and size, and publishes the model atomically only after verification.
Corruption or failure never exposes a partially imported model. Reimporting an
identical verified manifest is idempotent; a conflicting immutable key is rejected.

Other supported sources:

```sh
# Local directory, manifest file, archive, or file:// URL
stageplane import-bundle /srv/models/model-directory --store data/consumer-models
stageplane import-bundle /srv/models/model-directory/manifest.json --store data/consumer-models
stageplane import-bundle /srv/models/model.stageplane --store data/consumer-models

# Static directory or manifest URL
stageplane import-bundle https://models.example/model-directory/ --store data/consumer-models
stageplane import-bundle https://models.example/model-directory/manifest.json --store data/consumer-models

# Public Google Drive archive file; must permit downloading
stageplane import-bundle 'https://drive.google.com/file/d/FILE_ID/view?usp=sharing' \
  --store data/consumer-models

# Git repository containing a bundle directory, optionally using Git LFS
stageplane import-bundle git+https://github.com/OWNER/MODEL-REPO.git \
  --revision COMMIT_OR_BRANCH --subdir bundles/qwen4b --store data/consumer-models
```

Git resolves files but never runs repository setup scripts or model Python code.
Git LFS files require the Git LFS client. Ordinary Git hosting often limits large
files; prefer a small Git catalog plus externally hosted artifacts, Git LFS, or a
release download URL. An HTTPS Git repository URL needs the `git+` prefix so it
is distinguished from an HTTP artifact URL.

Google Drive uses `gdown` with browser-cookie reuse disabled. It handles public
file share URLs and large-file confirmation pages. Private login-only links,
provider download quotas, and revoked sharing fail explicitly. Host one archive
file on Drive; recursive Drive folder import is not implemented. An archive from
any provider must contain the Stageplane manifest and declared files.

## Git catalog with artifacts hosted elsewhere

A Git repository can contain `manifest.json` and `catalog.json`, while the large
weights live on any combination of public HTTP URLs and Drive file links:

```json
{
  "format": "stageplane-catalog-v1",
  "manifest": "manifest.json",
  "manifest_sha256": "PUBLISHERS_MANIFEST_SHA256",
  "files": {
    "embedding.safetensors": "https://storage.example/qwen/embedding.safetensors",
    "blocks-000.safetensors": "https://drive.google.com/file/d/ARTIFACT_FILE_ID/view"
  }
}
```

Include an override for every file not stored alongside the manifest, including
config/tokenizer/license files. Omitted entries are resolved relative to the
manifest. Catalogs can also be imported from HTTP URLs or local paths. This keeps
hosting locations separate from immutable model identity.

## 6. Start the executor directly from a remote bundle

```sh
export STAGEPLANE_TOKEN='choose-a-local-test-token'
stageplane executor --store data/consumer-models \
  --bundle http://127.0.0.1:8801/qwen3-4b.stageplane --port 8080
```

`--bundle` imports before serving and can be repeated. The executor retains the
imported model in `--store`. Workers obtain the authoritative manifest from the
executor, try their direct artifact roots, and use executor HTTP fallback when
needed. They do not require access to the original bundle publisher. To enable
direct publisher downloads, expose a directory with the root layout described
above and advertise it with `--artifact-source`; an archive URL is not a direct root.

An already running executor supports asynchronous import:

```sh
curl http://127.0.0.1:8080/api/models/import \
  -H "Authorization: Bearer $STAGEPLANE_TOKEN" -H 'Content-Type: application/json' \
  -d '{"kind":"bundle","source":"http://127.0.0.1:8801/qwen3-4b.stageplane"}'
# Poll GET /api/jobs/RETURNED_JOB_ID until completed or failed.
```

Use `kind: checkpoint` for original Hugging Face/local checkpoint distribution.
The dashboard import dialog exposes the same two modes. To download a bundle from
an executor for republishing, use authenticated `GET /api/bundles/MODEL_KEY`, e.g.
`curl ... -o model.stageplane`; the CLI export avoids a duplicate server-side archive.

## 7. Worker resource allocation

If omitted, both resource options resolve to **100% of that node/container's
capacity**. The node prints and publishes exact numeric byte/core values on
registration and includes current usage/headroom in every heartbeat.

```sh
# Entire node/container capacity
stageplane node --id node-a --executor http://127.0.0.1:8080

# Explicit maximum serving budgets
stageplane node --id node-b --executor http://127.0.0.1:8080 \
  --max-memory 500MB --max-cpu 50m --dtype bfloat16

# Percentages are resolved before publishing, not sent as "100%"
stageplane node --id node-c --executor http://127.0.0.1:8080 \
  --max-memory 50% --max-cpu 25%
```

Memory accepts bytes, KB/MB/GB, KiB/MiB/GiB, or a percentage. CPU accepts cores,
`m` millicores, or a percentage. `50m` is 0.05 cores; `500MB` is 500,000,000 bytes.
Percentages use detected node/container limits rather than current free memory;
free memory and current CPU use are separate published measurements. The legacy
`--memory-gb` option still accepts GiB. `--threads` cannot exceed allocated whole
compute threads. Explicit CPU budgets below available capacity use average process CPU throttling on
bare Linux/macOS; cgroups provide their own hard quota in Docker.

The memory option is a serving admission budget including estimated runtime,
weights, cache and workspace. It is not an OS RSS limiter on a bare Python
process. Use Docker/Kubernetes memory limits for strict process enforcement:

```sh
docker run --memory 500000000 --memory-swap 500000000 --cpus .05 ... \
  node --max-memory 500MB --max-cpu 50m
```

Registration includes resolved `memory_bytes` and `cpu_cores`; metrics include
`memory_limit_bytes`, `memory_used_bytes`, `available_memory_bytes`,
`loaded_weight_bytes`, `cpu_limit_cores`, `cpu_usage_percent`,
`available_cpu_cores`, and free storage. Container telemetry respects cgroup v2
limits. Placement uses present headroom, precision, declared failure domains,
observed stage cost, and CPU availability. The default CPU eligibility floor is 10 millicores of available capacity, configurable
with `STAGEPLANE_MIN_CPU_CORES`; it does not establish a latency SLO. Resource-triggered replans are scoped
to the affected group and rate-limited; healthy live routes remain pinned.

One worker may host many stages; node count never determines chunk count.
Storing 160 one-megabyte pieces does not mean executing 160 independent model
stages. The runtime currently needs complete compatible transformer blocks and
embedding/output modules in its serving budget. Arbitrary byte splitting, streamed
operator weights, and out-of-core sub-layer execution are not implemented.

If a full route cannot be placed, no partial route is advertised or dispatched.
The deployment stays pending for additional capacity. Console, dashboard and API
show the relevant caution:

```text
Not enough memory available to run this model. Kindly add more nodes.
Not enough compute available to run this model. Kindly add more nodes.
Not enough space available to run this model. Kindly add more nodes.
```

If the largest single stage cannot fit any worker, add a larger capable worker;
adding more equally small workers cannot execute that stage with this runtime.

## 8. Four-worker Qwen3 4B test

Stop a separately started executor first if it uses port 8080, then run:

```sh
python -m scripts.dev_cluster \
  --store data/review/qwen3-4b/executor-models \
  --model Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1 \
  --work data/review/qwen3-4b/four-node-demo \
  --nodes 4 --dtype bfloat16 --cpu-kernel bounded-f32 --memory-gb 6 --local-artifacts
```

Open http://127.0.0.1:8080 and enter `local-stageplane-development` through settings.
Inspect all 38 stages across four workers; each worker may host multiple stages.
The local script hardlinks verified cache files to reduce disk duplication. Omit
`--local-artifacts` to exercise ordinary verified copies; with a readable default
store path on the same host these can be local copies rather than network
downloads. Use the isolated Docker source/fallback test above to distinguish
direct HTTP from executor HTTP delivery. Ctrl-C stops owned services.

To reproduce the measured native/four-worker comparison:

```sh
python -m scripts.benchmark \
  --store data/review/qwen3-4b/executor-models \
  --model Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1 \
  --output reports/qwen3-4b/four-node \
  --dtype bfloat16 --cpu-kernel bounded-f32 --memory-gb 6 --skip-single --trials 5 --tokens 16
```

On this ARM CPU, `bounded-f32` keeps BF16 weights but uses FP32 accumulation in
weight tiles of at most 16 MiB. This avoids a slow generic BF16 CPU dot kernel
without materializing a second FP32 model. It is explicitly selected for both
the native unpartitioned architecture and workers; use `native` to compare the
unmodified PyTorch kernel instead. The native model and workers use the same BF16 tensors, prompts, greedy decoding,
and output cap. Raw token IDs are compared. Capture real timing rather than
assuming distribution improves single-session latency. All local workers share
one host's CPU and RAM. RSS does not include all compressed/swap memory.

## 9. Strict 50-node heterogeneous test, with storage fallback to 25

The requested profile has 50 workers: 10 each at **150, 250, 350, 400 and 500 MB**,
all with **50 mCore**. Docker enforces each worker's exact memory/CPU allocation
with swap disabled. Aggregate configured RAM is 16.5 GB and CPU is 2.5 cores,
but runtime overhead, indivisible stage sizes, and current availability still
control whether an actual inference route can fit.

```sh
docker build -t stageplane:qwen4-review .
python -m scripts.resource_profile \
  --store data/review/qwen3-4b/executor-models \
  --model Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1 \
  --nodes 50 --output reports/qwen3-4b/heterogeneous-50
```

This runs real containers and records Docker-enforced byte/core limits, actual
registrations, OOM exits, placement decisions, cautions and inference response.
A rejected route has no token-latency result; admission response time is recorded
separately. The test does not raise memory limits or change precision to claim
success. If Docker reports exhausted storage while creating 50 containers, it
cleans up owned containers and retries with 25, recording the reason. An
insufficient-memory result does not silently change the requested node count.

To repeat an OOM-limited run with explicitly larger RAM allocations, keep the
node count and CPU quotas and record it in a separate output directory:

```sh
python -m scripts.resource_profile \
  --store data/review/qwen3-4b/executor-models \
  --model Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1 \
  --nodes 50 --memory-tiers-mb 750,1000,1250,2000,2500 \
  --output reports/migration/heterogeneous-50-increased-ram
```

These are per-container ceilings, not reserved physical RAM. Increasing a worker
limit cannot increase the Docker VM's or host's physical capacity. The result
records both physical totals, registration and OOM outcomes, loading readiness,
and the inference response. The test waits up to `--load-timeout` seconds for a
complete ready route before attempting inference; default 300 seconds.

The script removes owned containers/network afterward and preserves the model,
raw report, and service logs. It leaves other projects' containers untouched.

## 10. Ten-worker test

For a smaller pool with larger per-worker RAM limits, reproduce the ten-worker
test while keeping the original 50-millicore CPU quota:

```sh
python -m scripts.resource_profile \
  --store data/review/qwen3-4b/executor-models \
  --model Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1 \
  --nodes 10 --memory-tiers-mb 2500,3000,3500 \
  --startup-timeout 300 --load-timeout 1800 --request-timeout 1800 \
  --output reports/ten-node
```

The RAM ceilings total 29.5 GB across containers, but do not reserve that amount
of physical memory. Actual usage must fit the Docker VM. Loading can take many
minutes at 50 millicores. `--request-timeout` changes the benchmark client's HTTP
deadline; the execution service's per-chunk deadlines and bounded retries remain
separate. See [the actual ten-worker report](../reports/ten-node/REPORT.md).

The measured 50m run loaded all stages but timed out during execution. Its
separate successful CPU comparison uses `--cpu-cores 1` with the same RAM tiers
and `--output reports/ten-node/cpu-1000m`. The default harness CPU cap remains
0.05 cores; standalone `stageplane node` still defaults to 100% of detected
capacity when resource options are omitted. The comparison does not claim that
50 millicores can meet the default chunk execution deadline.

## 11. Bounded execution recovery

`STAGEPLANE_CHUNK_TIMEOUT` bounds a chunk's compute/queue wait (default 30 seconds).
All transports have a bounded route RPC timeout (default 180 seconds).
`STAGEPLANE_EXECUTION_RETRIES` limits recovery retries (default three), and
`STAGEPLANE_RECOVERY_WAIT_SECONDS` bounds waiting for a replacement route (default 10).
These retries belong to Stageplane's execution service, not an external API client.

A timed-out worker/chunk is identified in the error envelope, quarantined for
that session, and excluded from replacement selection. A ready equivalent
replica is selected without a retry sleep. Decode cannot move blindly to a
weights-only replica: the execution service starts a new epoch and replays the
committed prefix to reconstruct attention/recurrent state. Committed output is
never resampled. The present recovery unit is the leased execution route, not a
claim of stage-local KV migration. No replica/capacity or exhausted retries
produces an explicit bounded failure.

Native compute cannot safely be interrupted in the middle of a PyTorch kernel.
After a deadline, the abandoned state/result is discarded while its compute
permit remains held until that kernel finishes. This prevents timed-out requests
from creating unbounded background compute. Use `max_retries=0` in an OpenAI
client if you want to observe only the execution service's retry policy.

## 12. Automated checks

```sh
uv run --extra test pytest -q
postman spec lint postman/specs/stageplane.openapi.json --issueType=syntax --no-report-events
postman collection lint postman/collections/Stageplane
```

Tests exercise local/HTTP/Git bundle imports, catalog mapping, corrupt artifacts,
unsafe ZIP paths, pinned hashes, BF16 cached-decode parity, resource resolution,
cgroup telemetry, capacity cautions, bounded retries, stage deadlines, real mixed
transports, OpenAI compatibility and mTLS. The public Drive adapter is tested
against a deterministic download fixture; a live user-hosted Drive model has not
been exercised without a supplied public bundle link. Separate provider failures
from model execution failures when interpreting results.

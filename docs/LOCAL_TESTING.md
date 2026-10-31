# Local producer, publisher, executor, and worker test

This guide tests a model produced by one user and consumed by another executor.
The consumer imports existing Stageplane stages; it does not download the original
checkpoint or repeat chunking. Anyone with access to the published files can
import the bundle into their own executor and deploy their own worker pool.

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

`--bundle` imports before serving and can be repeated. Workers do not need access
to the publisher: they receive the imported immutable manifest and assigned files
from the executor.

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
`--local-artifacts` to exercise real worker downloads. Ctrl-C stops owned services.

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

## 10. Bounded execution recovery

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

## 11. Automated checks

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

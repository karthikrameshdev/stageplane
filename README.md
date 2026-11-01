<h1 align="center">Stageplane</h1>

<p align="center">Run transformer layers across a trusted pool of heterogeneous workers.</p>

<p align="center">
  <a href="https://github.com/karthikrameshdev/stageplane/actions/workflows/test.yml"><img src="https://github.com/karthikrameshdev/stageplane/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="CODE_OF_CONDUCT.md"><img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant 2.1"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/LOCAL_TESTING.md">Local testing</a> ·
  <a href="#contributing">Contributing</a> ·
  <a href="#license">License</a>
</p>

Stageplane is an **Apache-2.0-licensed open-source research project** for distributed
transformer inference. You can use, modify, and redistribute the project under
the terms of its [license](LICENSE). Model weights retain their own upstream
licenses and access conditions.

Operational setup and limits: [operator guide](docs/OPERATIONS.md).
Hardening evidence: [validation report](reports/hardening/REPORT.md).
Direct-first storage and fallback: [Docker validation report](reports/artifact-delivery/REPORT.md).

Validation reports: [ten-worker Qwen3 4B test](reports/ten-node/REPORT.md) and
[repository migration with the direct/four-worker comparison](reports/migration/REPORT.md).

Run the layers of one open-weight language model across a trusted worker pool.
Stageplane separates **model distribution**, **execution coordination**, and **worker
execution**. Its light dashboard shows a Three.js topology, actual assignments,
worker health and memory, token commitments, stage timings, and recovery.

Implemented from [Distributed Transformer Execution: Orchestrated Inference Across
Heterogeneous Devices](http://127.0.0.1:4179/thesis/distributed-transformer-inference).
This is a runnable research implementation. See the [implementation map](docs/architecture.md)
for the distinction between implemented mechanisms and research extensions.

## Quick start

The producer chunks a model once into a central directory. The executor reads
that directory through its own mount and assigns stages. Workers first reuse
verified cache files, then try configured central sources directly, and finally
download through the authenticated executor if direct access fails.

| Role | Example location | Responsibility |
| --- | --- | --- |
| Producer | `/srv/stageplane/models` | Download original weights and publish immutable chunks |
| Executor | `/mnt/central/models` | Read the same central files, plan placement, route requests |
| Worker | `/mnt/models` → local `data/node-cache` | Copy assigned artifacts into a verified local cache and execute stages |

These paths can name the **same storage mounted differently on each machine**.
Provision the shared storage, mount it, and grant the producer write access before
starting Stageplane. Stageplane does not mount NFS/SMB or copy directories between
machines automatically. The executor must have a local filesystem view of its
`--store`; an HTTP source is a worker download location, not a replacement for
its model registry.

With separate roles, **3 machines = 1 producer + 1 executor + 1 worker**;
**10 machines = 1 producer + 1 executor + 8 workers**. A three-worker Docker test
has three worker containers in addition to its executor and artifact publisher.

### 1. Install on each machine

Python 3.11+ is required; Python 3.12 and 3.13 were tested on Apple Silicon.

```sh
git clone https://github.com/karthikrameshdev/stageplane.git
cd stageplane
uv sync --locked
source .venv/bin/activate
```

### 2. Producer: create the central model files

On the producer, with `/srv/stageplane/models` provisioned and writable:

```sh
stageplane distribute Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --layers-per-chunk 2 --store /srv/stageplane/models
```

This pinned checkpoint produces the model key
`Qwen--Qwen3--5-0--8B-2fc06364715b`. Use the key printed by your own distribution
command when choosing another checkpoint or revision. Files are published under
`/srv/stageplane/models/MODEL_KEY/`, including `manifest.json` and stage files.
Chunk boundaries follow model layers; they do not depend on worker count.

### 3. Executor: read the central store

Mount the producer's model directory at `/mnt/central/models` on the executor.
Provision the same internal token on the executor and every trusted worker.
The following HTTP addresses assume a private test network; use
[TLS/mTLS](#mutual-tls) for deployment across real networks.

On the executor, in a terminal that stays running:

```sh
export STAGEPLANE_TOKEN='replace-with-a-random-shared-internal-token'
# Keep SQLite on local disk, outside the shared/read-only model directory.
export STAGEPLANE_STATE_FILE="$PWD/data/executor-state/executor.sqlite"
stageplane executor --host 0.0.0.0 --port 8080 --store /mnt/central/models
```

The executor advertises `/mnt/central/models` to workers by default. If workers
see that storage under another path, set their own `--artifact-source` as below.
Alternatively, advertise a central HTTP(S) directory with
`--artifact-source https://models.example/models`; see the
[central HTTPS example](docs/LOCAL_TESTING.md#central-https-directory).

### 4. Worker: fetch assigned chunks and execute

On the worker, mount the same central directory at `/mnt/models`. Replace the
example DNS names with addresses reachable from the executor **and other workers**:

```sh
export STAGEPLANE_TOKEN='replace-with-a-random-shared-internal-token'
stageplane node --id worker-a \
  --executor http://executor.example:8080 \
  --host 0.0.0.0 --port 8101 --advertise-url http://worker-a.example:8101 \
  --artifact-source /mnt/models --cache data/node-cache \
  --max-memory 8GiB --max-cpu 2 --dtype float32 --group 0
```

`--artifact-source` is repeatable on both executor and worker. Each root must
contain `MODEL_KEY/FILENAME`; it is not the path to one model or a `.stageplane`
archive. A worker tries its own roots before the executor's advertised roots.
Missing, unreachable, or corrupt direct copies fall back to the executor's
authenticated artifact endpoint. Executor credentials are never sent to central
HTTP(S) sources. `--cache` remains a local worker copy, even with shared storage.

Repeat the worker command on further machines with unique IDs and reachable
endpoints. Omit `--artifact-source` on a worker without a mount: executor downloads
still work. The example memory/CPU budgets are starting values, not guaranteed
model requirements; use the readiness and capacity messages to size each worker.

### 5. Deploy and call the OpenAI-compatible endpoint

In another terminal on the executor:

```sh
export STAGEPLANE_TOKEN='replace-with-a-random-shared-internal-token'
export MODEL_KEY='Qwen--Qwen3--5-0--8B-2fc06364715b'

curl --fail-with-body http://127.0.0.1:8080/api/deployments \
  -H "Authorization: Bearer $STAGEPLANE_TOKEN" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_KEY\",\"replicas\":1,\"groups\":1}"

curl --fail-with-body http://127.0.0.1:8080/api/state \
  -H "Authorization: Bearer $STAGEPLANE_TOKEN"
```

Open **http://executor.example:8080** and enter the token in **Access settings**.
Wait for **Route ready** and all target stages to load. One capable worker can
hold every stage; multiple workers may each hold several stages.

```sh
curl --fail-with-body -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $STAGEPLANE_TOKEN" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_KEY\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello briefly.\"}],\"max_tokens\":32,\"temperature\":0,\"stream\":true}"
```

For a quick offline correctness fixture on one computer:

```sh
uv sync --extra test --locked
python -m scripts.tiny_checkpoint data/fixture
stageplane distribute data/fixture --store data/fixture-models --layers-per-chunk 1
python -m scripts.dev_cluster --store data/fixture-models --work data/fixture-run
```

The development launcher starts an executor and four worker processes; enter
`local-stageplane-development` in its dashboard at `http://127.0.0.1:8080`.
Ctrl-C stops its owned processes. The random fixture establishes machinery
correctness, not language quality.

## Local testing and published models

See [the detailed local testing guide](docs/LOCAL_TESTING.md) for source ordering,
mount mappings, central HTTP(S) hosting, Docker testing, Qwen3 4B commands,
resource allocations, and bounded recovery. `--store` already selects the
producer/executor model directory; `--artifact-source` adds direct worker access
before executor fallback. It does not change model chunking.

When a consumer cannot mount the central store, publish a portable bundle instead:

```sh
stageplane export MODEL_KEY --store /srv/stageplane/models --output dist/model.stageplane
stageplane import-bundle https://models.example/model.stageplane --store data/consumer-models
stageplane executor --store data/consumer-models
```

Executors can import local directories/archives, HTTP directories/archives,
Git/Git LFS/catalogs, or public Google Drive files. Imports preserve and verify
existing chunks without rechunking or fetching the original checkpoint.
Git and Drive resolve through this import workflow; worker direct sources are
filesystem or HTTP(S) directory roots.

Retained Qwen3 4B chunks are under `data/review/qwen3-4b/`; the guide lists exact
paths. See the [four-worker report](reports/qwen3-4b/FOUR_NODE_REPORT.md) and
[50-worker resource report](reports/qwen3-4b/FIFTY_NODE_REPORT.md). The four-worker
run matched direct output tokens in all five trials. The 50-worker pool rejected
the model because its largest complete stages exceed those workers' RAM limits;
the report retains the actual admission and OOM evidence.

## Model distribution

Accepts a Hugging Face repository and optional revision, a local HF-format
Safetensors checkpoint, or a trusted HTTP directory with `config.json`, tokenizer
files, Safetensors, and a `files.json` inventory containing `revision` and `files`.
Every tensor is retained. Contiguous whole language blocks, embeddings, the output
stage, and auxiliary weights become separate artifacts. The immutable manifest
pins checkpoint identity, tokenizer, file hashes, layout, and layer intervals.
Four bounded HTTP range downloads overlap without a second complete disk copy.

```sh
stageplane distribute Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --layers-per-chunk 2 --store data/models
```

Supported adapters: Qwen2, Qwen3, Qwen3.5 text, Llama, and Mistral. Other model
architectures require an adapter and parity tests. Auxiliary multimodal weights
are preserved, but image/video execution is not implemented. No remote Python
model code or pickle weights are executed.

Ollama tags commonly identify quantized GGUF. Explicit Qwen3.5 dense size tags
are resolved to their original Hugging Face open weights with one identifier:

```sh
stageplane distribute ollama:qwen3.5:0.8b
```

For other Ollama tags, supply `--hf-source <exact-repository>`. This executes
original open weights and does not claim numerical parity with Ollama quantization. A dashboard import form and `POST /api/models/import` also
support asynchronous imports with status at `/api/jobs/{id}`.

## Executor and nodes

The executor manages registry, placement, node health, route leases, admission,
sampling, committed output, cancellation, recovery, and live graph events.
Deployment metadata persists in SQLite. Prompts and tensor contents are not logged.
The authoritative contract is `postman/specs/stageplane.openapi.json`; live docs are
at `/docs`.

Nodes register, heartbeat each second, fetch assignments, download and verify
artifacts, and materialize only assigned layers. A real readiness fixture precedes
advertising a stage. Workers retain their attention/recurrent state and forward
activations directly to the next worker. REST, generic unary gRPC, and WebSocket
share the same bounded Safetensors protocol. All support TLS and bearer auth.

```sh
export STAGEPLANE_TOKEN='choose-a-secret-and-provision-it-on-all-services'
stageplane executor --host 0.0.0.0 --port 8080 --store data/models

# On each worker, advertise endpoints reachable by all other workers.
stageplane node --id worker-a --executor http://executor.example:8080 \
  --host 0.0.0.0 --advertise-url http://worker-a.example:8101 \
  --grpc-port 9101 --grpc-url worker-a.example:9101 \
  --transport grpc --memory-gb 8 --group 0
```

Non-loopback listeners require a token. Use mutual TLS below on real networks.
`--max-memory` and `--max-cpu` allocate maximum serving budgets; omitted values
resolve to 100% of node/container capacity and are published as exact bytes/cores.
Current memory, CPU use, and storage are reported separately. `--memory-gb` remains
a legacy GiB option, and `--speed` is a relative throughput hint. CPU defaults to float32. Other PyTorch devices are selected through
`--device`; their performance is not established by the local report.

Deploy using the dashboard or API:

```sh
curl http://127.0.0.1:8080/api/deployments \
  -H "Authorization: Bearer $STAGEPLANE_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-0.8B","replicas":2,"groups":1}'
```

One node can host all chunks. Adding 100 nodes does not change chunk boundaries;
excess capacity supplies replicas or spares. `groups:1` is an ordinary pool.
For multiple groups, stages hash to groups and nodes declare `--group`.
Membership changes replan only affected groups. Provision capable capacity in
every group: an uncovered group remains unready instead of triggering global
movement. Replicas use distinct declared `--failure-domain` values when possible.

Healthy sessions pin their route. Drain changes new assignments while retaining
stages used by live sessions. A crash increments the epoch and replays the prompt
plus committed output on a ready route. Loaded weights alone never count as a
complete session cache. No tokens are resampled after they have been delivered.

## OpenAI protocol

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-stageplane-development")
model = client.models.list().data[0].id
for chunk in client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Explain distributed inference briefly."}],
    max_tokens=32, temperature=0, stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Supported chat fields: text `messages`, `model`, `max_tokens`, `temperature`,
`top_p`, `seed`, and `stream`. Unknown fields are rejected. Tools, Responses API,
multimodal inputs, arbitrary OpenAI extensions, and batched tensors are not
implemented. Context defaults to 2,048 tokens including output, controlled by
`STAGEPLANE_MAX_CONTEXT`. Default admission: four gateway requests, eight resident
sessions per worker, one compute operation per worker.

## Four Docker workers

Prepare `data/models` first; Compose never silently downloads a checkpoint.

```sh
export STAGEPLANE_TOKEN='choose-a-secret'
docker compose up --build -d
# Deploy via /api/deployments or the dashboard.
docker compose logs -f
docker compose down
```

Worker volumes receive their own verified downloads. The example's aggregate
container memory limits are about 20 GB; actual use depends on model and placement.

## Mutual TLS

```sh
sh scripts/dev_certs.sh certs
export STAGEPLANE_CA="$PWD/certs/ca.crt"
export STAGEPLANE_TLS_CERT="$PWD/certs/worker.crt"
export STAGEPLANE_TLS_KEY="$PWD/certs/worker.key"
export STAGEPLANE_CLIENT_CERT="$PWD/certs/worker.crt"
export STAGEPLANE_CLIENT_KEY="$PWD/certs/worker.key"
export STAGEPLANE_TOKEN='choose-a-secret'
```

The short-lived test certificates cover localhost only. Use distinct workload
certificates with appropriate SANs from your PKI on actual hosts, and advertise
`https://` URLs. REST/WebSocket verify Python TLS contexts; gRPC also requires and
verifies client certificates. A browser reaching an mTLS listener needs a valid
client certificate or a separately authenticated gateway.

## Tests and measurements

```sh
uv run --extra test pytest -q
uv run --extra test python -m scripts.benchmark \
  --model Qwen--Qwen3--5-0--8B-2fc06364715b --trials 5 --tokens 16
postman spec lint postman/specs/stageplane.openapi.json --issueType=syntax --no-report-events
postman collection lint postman/collections/Smoke
postman collection run postman/collections/Smoke --env-var baseUrl=http://127.0.0.1:8080
```

The benchmark owns its services, retains raw direct/one-worker/four-worker trials,
compares token IDs, samples RSS every 20 ms, and separates logical worker storage
from hardlinked disk use. Local workers share one CPU, memory subsystem, and
loopback network. This establishes correctness and local costs, not multi-host
speedup or enterprise productivity. See [the report](reports/REPORT.md),
[protocol](docs/protocol.md), and [implementation map](docs/architecture.md).

## Contributing

Contributions are welcome: bug reports, reproducible measurements, documentation,
model adapters, and fixes. Start with [CONTRIBUTING.md](CONTRIBUTING.md), fork the
repository, and open a pull request against `main`. The guide covers the offline
fixture, required checks, and review expectations.

[Karthik Ramesh (@karthikrameshdev)](https://github.com/karthikrameshdev) is the
maintainer and code owner. See [MAINTAINERS.md](MAINTAINERS.md) and
[CODEOWNERS](.github/CODEOWNERS) for ownership. Community contributions follow
code-owner review; GitHub enforcement depends on repository branch settings.

Please follow the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
Use the [issue templates](https://github.com/karthikrameshdev/stageplane/issues/new/choose)
for bugs, feature requests, and questions. Report vulnerabilities privately using
the contact in [SECURITY.md](SECURITY.md).

## License

Stageplane is licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE)
and [NOTICE](NOTICE) for the license and attribution information.

The bundled Three.js code uses the [MIT license](stageplane/ui/vendor/THREE-LICENSE.txt).
Other dependencies and downloaded model weights retain their respective licenses;
Stageplane's license does not replace those terms.

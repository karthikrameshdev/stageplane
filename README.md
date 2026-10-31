# Stageplane

Latest validation: [repository migration and fresh Qwen3 4B test report](reports/migration/REPORT.md).

Run the layers of one open-weight language model across a trusted worker pool.
Stageplane separates **model distribution**, **execution coordination**, and **worker
execution**. Its light dashboard shows a Three.js topology, actual assignments,
worker health and memory, token commitments, stage timings, and recovery.

Implemented from [Distributed Transformer Execution: Orchestrated Inference Across
Heterogeneous Devices](http://127.0.0.1:4179/thesis/distributed-transformer-inference).
This is a runnable research implementation. See the [implementation map](docs/architecture.md)
for the distinction between implemented mechanisms and research extensions.

## Local testing and published models

See [the detailed local testing README](docs/LOCAL_TESTING.md) for the complete
producer → publisher → remote executor → worker workflow, Qwen3 4B commands,
exact resource allocations, 50 heterogeneous containers, and bounded recovery.
The retained chunks are under `data/review/qwen3-4b/`; the guide lists every path.
The measured Qwen3 4B results are in the
[four-worker report](reports/qwen3-4b/FOUR_NODE_REPORT.md) and the
[50-worker resource report](reports/qwen3-4b/FIFTY_NODE_REPORT.md).
The four-worker run matched direct output tokens in all five trials. The 50-worker
pool rejected this model because its largest complete stages exceed the workers'
RAM limits; the report records the actual admission and OOM evidence.

A producer can export a `.stageplane` ZIP64 bundle and upload it to any public host.
Executors import the same verified stages from a local path, HTTP directory or
archive, Git/Git LFS/catalog, or public Google Drive file. Importing a published
bundle does not rechunk or fetch the original checkpoint.

```sh
stageplane export MODEL_KEY --store data/models --output dist/model.stageplane
stageplane import-bundle https://models.example/model.stageplane --store data/consumer-models
stageplane executor --store data/consumer-models --bundle https://models.example/model.stageplane
```

## Quick start

Python 3.11+ is required; Python 3.12 and 3.13 were tested on Apple Silicon.

```sh
uv sync --extra test
source .venv/bin/activate

# Download every checkpoint tensor into hashed Safetensors artifacts.
# One complete decoder block per chunk, independent of node count.
stageplane distribute Qwen/Qwen3-4B-Instruct-2507 --layers-per-chunk 1

# Four worker processes, mixed REST/gRPC/WebSocket links, one executor.
python -m scripts.dev_cluster --nodes 4 --port 8080 --dtype bfloat16 --cpu-kernel bounded-f32 --local-artifacts
```

Open **http://127.0.0.1:8080**. Enter the default development token,
`local-stageplane-development`, through the dashboard settings button.
`--local-artifacts` hardlinks verified files for same-disk testing; omit it to
exercise worker downloads. Ctrl-C stops all services owned by the script.

For an offline correctness fixture:

```sh
python -m scripts.tiny_checkpoint data/fixture
stageplane distribute data/fixture --store data/fixture-models --layers-per-chunk 1
python -m scripts.dev_cluster --store data/fixture-models --work data/fixture-run
```

The random fixture establishes machinery correctness, not language quality.

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

# Direct-first artifact delivery validation

Validated 2026-11-01 against base `86cbe89ddd1b1c106872e5c744c0a6962ade3e02`
plus the accompanying artifact-source implementation.

## Implemented workflow

The producer chooses its output root with `distribute --store PATH`. The executor
opens that store through its own filesystem mount using `executor --store PATH`.
It advertises its store path by default, or explicit repeatable `--artifact-source`
roots. A node can provide its own mount aliases or mirrors with the same option.

For every assigned file, the node checks verified local cache, its configured
sources, executor-advertised sources, and finally the authenticated executor file
endpoint. Local paths, local `file://` URLs and HTTP(S) roots are supported.
Source directories have the layout `ROOT/MODEL_KEY/FILENAME`. Copies are size/hash
checked and atomically published. Central HTTP requests never use the executor's
authentication headers or client certificate. Slow filesystem I/O stays off the
heartbeat event loop. OS/mount timeouts still govern a stuck filesystem syscall.

## Validation evidence

- Full regression suite: **142 passed**, zero failures/errors/skips, 51.57 seconds.
- Additional executor/CLI contract regressions: **16 passed**, 2.89 seconds.
- **158 unique tests passed** across those runs. Tests cover corrupt/denied/missing
  sources, timeout cooldowns, credential isolation, partial cleanup, concurrent
  downloads, cancellation, stalled file I/O, disk-full behavior and source ordering.
- OpenAPI syntax and collection lint passed. Local API collections: **4 requests,
  15 assertions, zero failures**.
- The Docker image built and installed the complete updated application.

The Docker test used an offline deterministic Qwen3 fixture with six stages,
three workers, an executor and a central HTTP publisher. It used separate Docker
networks and read-only mounts to prove access restrictions rather than mocking them.
This is a delivery/execution correctness test, not a large-model performance result.

| Phase | Node with shared directory | HTTP-capable node | Executor-only node |
| --- | --- | --- | --- |
| Initial load | 4 direct local files, 0 executor files | 2 direct HTTP files, 0 executor files | 1 executor file after direct failure |
| HTTP publisher stopped; HTTP node given empty cache | Existing verified files | 2 executor files, 0 successful direct HTTP files | Existing verified files |
| Shared mount removed; local node restarted with its cache | 4 cache hits, 0 direct or executor transfers | Existing verified files | Existing verified files |

Inference succeeded in every phase across all three workers. Streaming output
matched the deterministic non-streaming baseline. All requests completed and
worker sessions returned to zero. The central HTTP log confirmed that it received
no Authorization or Cookie headers. Raw evidence and immutable Docker image ID:
[validation.json](validation.json). Suite totals: [tests.json](tests.json).

## Reproduce

From the repository root with Docker running:

```sh
uv sync --locked --extra test
docker build -t stageplane:local .
uv run python -m scripts.artifact_delivery_validation --image stageplane:local \
  --output reports/artifact-delivery
```

The script owns its isolated containers, volumes and temporary fixture directory,
and removes them after the run. `--keep` leaves the diagnostic fixture on port
8082; its final phase intentionally has the publisher stopped and shared mount
removed. The script prints the Compose file needed to stop that fixture. Port 8080
and existing user containers are unaffected. CI now builds the image and executes
the same Docker assertions on each change.

The main [README](../../README.md#quick-start) has the complete producer, executor,
worker, deployment and OpenAI request commands. The [local guide](../../docs/LOCAL_TESTING.md)
includes shared mount mappings, central HTTP(S), worker counters and failure tests.

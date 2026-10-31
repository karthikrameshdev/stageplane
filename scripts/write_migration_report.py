"""Render the current repository's migration and fresh execution evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/migration"


def read(path):
    return json.loads(path.read_text())


def main():
    summary = read(OUT / "four-node/summary.json")
    native = read(OUT / "four-node/direct.json")
    four = read(OUT / "four-node/distributed-4.json")
    env = read(OUT / "environment.json")
    artifacts = read(ROOT / "reports/qwen3-4b/artifacts.json")
    x, y = summary["results"]
    text = f"""# Stageplane migration and fresh test report

The implementation and retained artifacts now live at:

```text
{ROOT}
```

The existing repository's Git history, Apache license and
`https://github.com/karthikrameshdev/stageplane.git` remote were preserved.
No Git push or public artifact upload was performed.

## Complete project identity

- Python distribution, import package, installed CLI and Docker entrypoint:
  `stageplane` (`python -m stageplane` also works).
- Runtime settings and authentication: `STAGEPLANE_*` environment variables.
- gRPC service: `/stageplane.Worker/Forward`; internal error marker:
  `STAGEPLANE_ERROR` followed by newline.
- Published bundles: `.stageplane`; catalog format: `stageplane-catalog-v1`.
- Dashboard, API title/model ownership, OpenAPI specification, generated
  collections, Compose names/images, test certificates, docs and scripts all
  use Stageplane. Every component must use these current identifiers.
- A new virtual environment and wheel were built from this checkout. The wheel
  includes the package, console entrypoint and vendored UI assets.
- Retained historical results have their project identifiers/path labels updated;
  their timing values and token IDs were preserved. The results below are fresh
  executions from this checkout. Current screenshots replace the earlier branding.

## Automated and contract validation

The complete automated suite passed **35 tests** from the new virtual environment.
Coverage includes native prefill/decode parity, bundle imports and integrity,
resource telemetry/default publication, placement/rebalancing, bounded retries,
mixed REST/gRPC/WebSocket transport, OpenAI SDK calls and mutual TLS.
The increased-RAM run exposed transient CPU exhaustion during weight loading;
the fix retains existing loading/session assignments and excludes busy workers
from new assignments. A regression test covers this behavior.

- [Original pytest output](tests.txt)
- [OpenAPI syntax lint](spec-lint.txt): no issues.
- [Regenerated collection lint](collection-lint.txt): no issues, 14 requests.
- [Live executor smoke output](postman-smoke.txt): 3 requests, 9 assertions,
  zero failures.
- [Installed package and wheel identity](package.json)
- [Model/hash/archive verification](artifact-verification.json)
- [Identity audit](identity-audit.json)

Host runtime: Python `{env['python'].split()[0]}`, PyTorch
`{env['packages']['torch']}`, Transformers `{env['packages']['transformers']}`.
The Docker image uses Python 3.13. The host has 10 logical CPUs and 16 GiB RAM.

## Fresh Qwen3 4B: direct versus four workers

Checkpoint: `{summary['source']}`, revision `{summary['revision']}`.
BF16 weights; no quantization or substitute model. Both modes use the explicit
`bounded-f32` CPU kernel: at most 16 MiB weight tiles are converted to FP32 for
linear accumulation, then outputs return to BF16. This is not a comparison of
vanilla BF16 kernel performance. Direct mode is an unpartitioned Transformers
model with two compute threads. Four workers use REST, two threads each, one
group/replica, verified local hardlinks and 6 GiB serving budgets per worker.
These worker budgets share one physical Mac and do not reserve 24 GiB of RAM.
The four-worker timings were collected before the loading-state scheduling guard
was added; serving kernels and wire format are unchanged. The full 35-test suite
and final higher-RAM Docker retry exercise that guard.

**Exact generated-token parity: {summary['parity']} across all five trials.**
Each mode had an excluded warmup; measured requests generated at most 16 new
tokens, stopping at EOS. Math requests returned `2 plus 2 is 4.`.

| Metric | Direct | Four workers |
|---|---:|---:|
"""
    rows = [
        ("Median first-token latency (s)", x["ttft_p50_ms"]/1000, y["ttft_p50_ms"]/1000),
        ("Interpolated p95 first-token latency (s)", x["ttft_p95_ms"]/1000, y["ttft_p95_ms"]/1000),
        ("Median decode interval (s)", x["decode_p50_ms"]/1000, y["decode_p50_ms"]/1000),
        ("Interpolated p95 decode interval (s)", x["decode_p95_ms"]/1000, y["decode_p95_ms"]/1000),
        ("Median whole request (s)", x["request_p50_ms"]/1000, y["request_p50_ms"]/1000),
        ("Median output tokens/s, including prefill", x["tokens_per_second_median"], y["tokens_per_second_median"]),
        ("Observed aggregate peak RSS (decimal GB)", x["peak_rss_bytes"]/1e9, y["peak_rss_bytes"]/1e9),
        ("Largest worker peak RSS (decimal GB)", x["max_worker_rss_bytes"]/1e9, y["max_worker_rss_bytes"]/1e9),
        ("Logical serving storage (decimal GB)", x["logical_storage_bytes"]/1e9, y["logical_storage_bytes"]/1e9),
        ("Load / complete-route-ready time (s)", x["load_s"], y["load_s"]),
    ]
    text += "".join(f"| {name} | {a:.3f} | {b:.3f} |\n" for name, a, b in rows)
    text += "\n| Trial | Tokens | Direct request (s) | Four-worker request (s) | Exact parity |\n|---|---:|---:|---:|---|\n"
    for i, (a, b) in enumerate(zip(native["trials"], four["trials"]), 1):
        text += f"| {i} | {len(a['token_ids'])} | {a['elapsed_ms']/1000:.3f} | {b['elapsed_ms']/1000:.3f} | {a['token_ids'] == b['token_ids']} |\n"
    text += """
Raw evidence: [summary](four-node/summary.json), [direct](four-node/direct.json),
[four workers](four-node/distributed-4.json). RSS excludes macOS compressed and
swapped pages; paging and background applications affect latency. Five trials
do not establish stable tail latency or performance on separate physical hosts.
Tied embedding/head weights share storage in direct mode; separate worker
processes may require separate copies. Hardlinks reduce local disk duplication,
while remote hosts need their own assigned files.

## Fifty workers, including the higher-RAM retry

Both runs keep 50 workers and a strict **0.05 CPU cores per worker**, one compute
thread, BF16 precision, and container swap disabled. RAM limits are explicit
ceilings rather than reserved physical RAM. The higher-RAM run is a separate
attempt requested after the original OOMs; it is not presented as the same profile.

An initial higher-RAM loading probe registered all 50 nodes without OOM, then
exposed the loading-state scheduling issue described above. Its unmodified
[raw evidence](heterogeneous-50-increased-ram-loading-probe/result.json) is
retained separately. The table's increased-RAM row is the subsequent fixed run.

| Attempt | RAM tiers (decimal MB) | Registered | OOM killed | Deployment / inference HTTP | Inference completed |
|---|---|---:|---:|---|---|
"""
    profiles = []
    for name, label in [("heterogeneous-50", "Original"), ("heterogeneous-50-increased-ram", "Increased RAM")]:
        path = OUT / name / "result.json"
        if not path.exists():
            text += f"| {label} | Pending | — | — | — | — |\n"
            continue
        r = read(path)
        profiles.append((name, label, r))
        tiers = ", ".join(str(v) for v in r["memory_tiers_MB"])
        text += f"| {label} | {tiers} | {r['registered_nodes']} | {r['oom_count']} | {r['deploy_status']} / {r['inference_status']} | {r['completed_inference']} |\n"
    for name, label, r in profiles:
        text += f"\n### {label} result\n\n"
        text += f"Docker VM RAM: {r['docker_memory_bytes']/1e9:.3f} GB; host RAM: {r['host_memory_bytes']/1e9:.3f} GB. Storage fallback to 25: {r['storage_fallback']}.\n\n"
        if r.get("capture_complete") is False:
            text += "**Partial capture:** Docker log collection timed out after the inference request. The status codes above come from the retained executor log, the 50 registrations from a captured live observation, and the five OOM kills from Linux kernel records. The complete final registry and timing snapshot was lost; no such snapshot or inference timing is claimed. The harness now persists measurements before optional log collection, and a regression test verifies preservation on log timeout.\n\n"
        if r["completed_inference"]:
            text += f"Output: `{r['inference_response']['choices'][0]['message']['content']}`.\n\n"
            req = r["cluster"]["requests"][-1]
            text += f"Server first token: {req['ttft_ms']/1000:.3f} s; total inference: {req['elapsed_ms']/1000:.3f} s; output tokens/s: {req['tokens_per_second']:.3f}. This is one functional request, not five-trial performance evidence.\n\n"
        else:
            text += f"Inference did not complete: `{r['inference_response']}`. TTFT and throughput are **N/A**. HTTP rejection/response capture time is not inference latency.\n\n"
        title = "Reconstructed summary and evidence limitations" if r.get("capture_complete") is False else "Full Docker limits/states, placement and telemetry"
        text += f"[{title}]({name}/result.json). Executor and individual worker logs are retained beside that file.\n"
    text += f"""
The largest stages require about 778 MB of weights each; decoder stages require
about 202 MB each, in addition to runtime, loading and session workspace. A route
needs every complete stage. Increasing container caps does not create physical
RAM in the Docker VM, and arbitrary byte slices are not executable stages.
The backend does not implement out-of-core operators or tensor-parallel splitting
of the embedding/output stage. State recovery uses bounded executor retries and
committed-prefix replay, not live KV-state migration.

The final increased-RAM run hit **VM-wide physical exhaustion**. The kernel
recorded `constraint=CONSTRAINT_NONE` and `global_oom`, rather than a container
exceeding its own RAM cap. Five Stageplane processes were killed. See
[the kernel evidence](vm-global-oom.json), [the live observation](loading-observation.json),
and [one retained Docker OOM inspect record](remaining-oom-container.json).
Increasing container limits further cannot create RAM inside the 14.636 GB VM.
The successful four-worker benchmark is valid; there is no successful 50-worker
inference benchmark on this host. More physical worker/VM memory is required.

## Retained artifacts and reproduction

All five retained model directories passed manifest/file hash verification; the
portable archive passed ZIP CRC verification. The Qwen4 manifest SHA-256 remains
`{artifacts['manifest_sha256']}`.

Chunks:

```text
{artifacts['chunk_directory']}
```

Consumer copy:

```text
{artifacts['consumer_directory']}
```

Uploadable archive:

```text
{artifacts['public_archive']}
```

See [the detailed local testing README](../../docs/LOCAL_TESTING.md) for install,
publishing, imports, worker limits and recovery. Fresh four-worker benchmark:

```sh
python -m scripts.benchmark \\
  --store data/review/qwen3-4b/executor-models \\
  --model {summary['model']} \\
  --output reports/migration/four-node \\
  --dtype bfloat16 --cpu-kernel bounded-f32 --memory-gb 6 \\
  --skip-single --trials 5 --tokens 16
```

The higher-RAM command is in the local guide. Render this report again with
`python -m scripts.write_migration_report`. Model artifacts remain on disk;
test services/containers are cleaned up afterward. Firefox and Codex remain open.

Current UI proof: [dashboard](../dashboard-full.jpg),
[published chunk import](../qwen3-4b/bundle-import.jpg).
"""
    (OUT / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()

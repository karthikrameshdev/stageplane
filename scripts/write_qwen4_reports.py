"""Render the retained Qwen4 measurements; never substitutes missing inference data."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/qwen3-4b"


def read(path):
    return json.loads(path.read_text())


def main():
    receipt = read(OUT / "artifacts.json")
    summary = read(OUT / "four-node/summary.json")
    direct = read(OUT / "four-node/direct.json")
    distributed = read(OUT / "four-node/distributed-4.json")
    a, b = summary["results"]
    rows = [
        ("Median time to first token (s)", a["ttft_p50_ms"] / 1000, b["ttft_p50_ms"] / 1000),
        ("Interpolated p95 first token (s)", a["ttft_p95_ms"] / 1000, b["ttft_p95_ms"] / 1000),
        ("Median decode token interval (s)", a["decode_p50_ms"] / 1000, b["decode_p50_ms"] / 1000),
        ("Interpolated p95 decode interval (s)", a["decode_p95_ms"] / 1000, b["decode_p95_ms"] / 1000),
        ("Median whole request (s)", a["request_p50_ms"] / 1000, b["request_p50_ms"] / 1000),
        ("Median output tokens/s, including prefill", a["tokens_per_second_median"], b["tokens_per_second_median"]),
        ("Observed aggregate peak RSS (decimal GB)", a["peak_rss_bytes"] / 1e9, b["peak_rss_bytes"] / 1e9),
        ("Largest worker peak RSS (decimal GB)", a["max_worker_rss_bytes"] / 1e9, b["max_worker_rss_bytes"] / 1e9),
        ("Logical serving weights/storage (decimal GB)", a["logical_storage_bytes"] / 1e9, b["logical_storage_bytes"] / 1e9),
        ("Load / cluster-ready time (s)", a["load_s"], b["load_s"]),
    ]
    text = """# Qwen3 4B: direct versus four workers

All five measured requests produced identical token IDs in both modes. Four
workers reduced the largest worker's resident allocation but did not improve
median request latency in this experiment. These measurements are from one
memory-constrained shared Mac, not four independent physical machines.

## Model and execution

"""
    text += f"""- Checkpoint: `{receipt['checkpoint']}`.
- Pinned revision: `{receipt['revision']}`.
- Manifest SHA-256: `{receipt['manifest_sha256']}`.
- Model tensors: 4,022,468,096 parameters; BF16, no quantization.
- Artifacts: {receipt['artifact_bytes']:,} bytes. 36 whole decoder-block stages,
  one embedding stage, one norm/output stage: **38 executable stages**.
- Host: `{summary['host']['platform']}`, 10 logical CPUs, 16 GiB physical RAM.
- Direct mode: unpartitioned native Transformers model, two compute threads.
- Distributed mode: four worker processes plus executor, REST tensor transport,
  one group, one replica, two compute threads per worker, 6 GiB serving budget
  per worker. Each publishes the shared host's detected 10 CPU cores; these are
  not 40 independent cores. The four serving budgets are admission limits,
  not a claim that the host has 24 GiB RAM.
- Both modes explicitly use `bounded-f32`: BF16 weights are materialized in RAM;
  CPU linear operations cast weight tiles of at most 16 MiB to FP32 for
  accumulation and return BF16. This avoids this ARM CPU's slow generic BF16
  path. It is not a vanilla BF16-kernel benchmark. No model substitution occurred.
- Greedy decoding, at most 16 new tokens; one excluded warmup, five measured
  requests, three prompts rotated. EOS ended the math requests after 9 tokens.
- Worker files are verified hardlinks to the independently HTTP-imported bundle
  for this same-disk test. Historical tests separately exercise worker downloads.

## Measurements

| Metric | Direct | Four workers |
|---|---:|---:|
"""
    text += "".join(f"| {name} | {x:.3f} | {y:.3f} |\n" for name, x, y in rows)
    text += "\n| Trial | Output tokens | Direct request (s) | Four workers (s) | Exact token parity |\n|---|---:|---:|---:|---|\n"
    for i, (x, y) in enumerate(zip(direct["trials"], distributed["trials"]), 1):
        text += f"| {i} | {len(x['token_ids'])} | {x['elapsed_ms']/1000:.3f} | {y['elapsed_ms']/1000:.3f} | {x['token_ids'] == y['token_ids']} |\n"
    text += "\n| Worker | Executable stages | Materialized weight bytes |\n|---|---:|---:|\n"
    for node in sorted(distributed["nodes"].values(), key=lambda n: n["id"]):
        count = sum(node["id"] in v for v in distributed["placements"]["placements"].values())
        text += f"| {node['id']} | {count} | {node['metrics']['loaded_weight_bytes']:,} |\n"
    text += f"""
The math answer was `2 plus 2 is 4.`. Longer answers are intentionally truncated
at the output token limit; raw text, token IDs, individual token intervals and
wire-payload counts are retained in [direct.json](four-node/direct.json) and
[distributed-4.json](four-node/distributed-4.json). Summary: [summary.json](four-node/summary.json).

## Interpretation and limits

The direct median was {a['request_p50_ms']/1000:.2f} s and distributed median was
{b['request_p50_ms']/1000:.2f} s. Trial variation is substantial: the Mac compressed
and swapped model pages while Codex and Firefox remained open. Other visible
apps were already closed; the idle Docker VM and unused test publisher were
stopped during the comparison. Native and distributed models ran sequentially.
The direct warmup took {direct['warmup']['elapsed_ms']/1000:.2f} s and the distributed
warmup took {distributed['warmup']['elapsed_ms']/1000:.2f} s. Five samples do not
establish stable tail latency or a general performance ranking.

RSS is sampled every 20 ms. On macOS it excludes compressed and swapped pages;
it is not the complete physical or virtual footprint. No reliable per-request
CPU utilization, energy, or network-bandwidth utilization measurement is claimed.
Timing includes paging, routing and serialization. The distributed trace's
payload counters include internal framed tensors/logits; they are not NIC bytes.

Tied embedding/head weights can share one tensor in direct mode. Separate worker
processes require separate copies when those stages are placed apart, explaining
the larger 8.823 GB logical worker storage versus 8.045 GB unique artifacts.
Hardlinks avoid duplicating data blocks in this local cache; real machines need
their own assigned artifact files. Producer, consumer and export archive copies
are additional retained review storage and are not serving-weight RAM.

Earlier incomplete native-BF16 and file-backed-residency probes are retained as
[slow-native-probe.json](slow-native-probe.json) and
[mmap-residency-probe.json](mmap-residency-probe.json). They are excluded from
the five completed trials and all table statistics.

## Reproduce and review

See [the local testing guide](../../docs/LOCAL_TESTING.md), including the exact
benchmark command and public-bundle producer/consumer workflow. Run
`python -m scripts.write_qwen4_reports` after collecting both raw tests to render
these reports again. Artifacts remain at:

```text
{receipt['chunk_directory']}
```

The consumer directory and portable archive are recorded in
[artifacts.json](artifacts.json). Nothing was uploaded to a public account.
"""
    (OUT / "FOUR_NODE_REPORT.md").write_text(text)

    profile_path = OUT / "heterogeneous-50/result.json"
    if not profile_path.exists():
        return
    r = read(profile_path)
    c = r["cluster"]
    deployment = c["deployments"][receipt["model"]]
    capacity = deployment["capacity"]
    tiers = sorted({row["memory_bytes"] for row in r["containers"]})
    text = f"""# Qwen3 4B: 50 strictly limited heterogeneous workers

**The requested pool was actually started. It could not admit this model.**
Deployment returned HTTP {r['deploy_status']} and inference returned HTTP
{r['inference_status']}. No model inference completed, so TTFT, decode latency
and tokens/second are **N/A**, not zero. Admission-response time is not model
inference latency.

## Actual configuration and outcome

- Requested workers: {r['requested_nodes']}; attempted workers: {r['actual_nodes']}.
- Memory tiers: 10 each at 150, 250, 350, 400 and 500 decimal MB for 50 workers.
- Every worker: Docker hard CPU quota 0.05 cores (50 millicores), one compute
  thread; memory+swap limit equals RAM limit, so **no swap** is permitted.
- Requested nominal pool capacity: 16.5 GB RAM and 2.5 CPU cores. PyTorch/runtime
  memory and availability reduce the capacity that can hold weights.
- Checkpoint, manifest, BF16 precision and `bounded-f32` kernel are the same as
  the [four-worker test](FOUR_NODE_REPORT.md). Image: `{r['image']}`.
- Executor is separate: 768 MB / one CPU, not counted among worker resources.
- Registered workers at capture: {r['registered_nodes']}.
- Docker-confirmed OOM-killed workers: {r['oom_count']}.
- Storage fallback to 25 workers: **{r['storage_fallback']}**. A fallback is
  permitted only for an actual Docker no-space error, never for RAM/CPU failure.
- Capacity reason: `{capacity['reason']}`.
- Console/API caution: **{capacity['message']}**

| RAM limit (decimal MB) | Containers attempted | Registered | OOM killed |
|---|---:|---:|---:|
"""
    registered = {n["id"] for n in c["nodes"]}
    for tier in tiers:
        nodes = [row for row in r["containers"] if row["memory_bytes"] == tier]
        text += f"| {tier/1e6:.0f} | {len(nodes)} | {sum(n['node'] in registered for n in nodes)} | {sum(n['state']['OOMKilled'] for n in nodes)} |\n"
    text += f"""
## Why aggregate RAM did not suffice

The smallest supported execution unit is a complete compatible stage. The
embedding stage requires 777,912,456 weight bytes and the norm/output stage
777,917,696 bytes. Neither fits a 500 MB worker even before runtime/workspace.
Decoder stages each require about 201.9 MB of weights plus runtime and session
state. Some of the smallest workers cannot even initialize the PyTorch runtime
within their cap; the table reports the actual Docker OOM result.

Splitting the archive into 8,000 byte pieces would not make 8,000 executable
stages. This implementation supports multiple complete stages per worker and
does not implement out-of-core operator streaming or tensor-parallel embedding/
head execution. More workers with the same small maximum RAM cannot resolve
that per-stage requirement. Add workers with enough free RAM for the largest
stage and runtime, or implement a backend that subdivides those operators.

The executor retains a pending deployment and emits the memory caution. It
withholds partial-route assignments, avoiding downloads that cannot produce a
complete model route. CPU/storage cautions are separate and covered by automated
tests. No precision change, hidden larger worker or swapped container was used
to claim a successful 50-worker inference.

## Evidence and reproduction

- [Raw configuration, Docker states and executor snapshot](heterogeneous-50/result.json)
- Console and per-node logs: `reports/qwen3-4b/heterogeneous-50/`.
- [Local testing commands and resource semantics](../../docs/LOCAL_TESTING.md)
- [Retained artifact paths and manifest digest](artifacts.json)

Every container's Docker `Memory`, `MemorySwap` and `NanoCpus` values are in the
raw result, alongside exact worker registration/telemetry. Owned containers and
the private network are removed after capture. The shared imported chunks remain
at `{r['artifact_location']}`.
"""
    (OUT / "FIFTY_NODE_REPORT.md").write_text(text)
    (OUT / "REPORT.md").write_text("""# Qwen3 4B test results

- [Direct versus four workers: five completed trials, exact token parity](FOUR_NODE_REPORT.md)
- [50 capped heterogeneous workers: actual admission/OOM results](FIFTY_NODE_REPORT.md)
- [Model identity and retained chunks](artifacts.json)
- [Detailed local testing README](../../docs/LOCAL_TESTING.md)

The final automated run passed **33 tests**. The live executor Postman smoke
run made **3 requests with 9 assertions and 0 failures**; see
[the original CLI output](postman-smoke.txt). OpenAPI and generated collection
lint found no issues. Tests cover bundle integrity/source adapters, exact default
resource publication, capacity cautions, bounded retry/failover, native stage
parity, mixed transports, OpenAI calls and mTLS. A separate bare-host CPU governor
sample requested 0.1 cores and observed 0.1014 cores over two seconds; this is a
short functional check, not long-term quota precision.

The real 8 GB bundle was imported over local HTTP without rechunking. Git/local
and catalog imports were also exercised. Google Drive uses its public-file
adapter and a deterministic test fixture; a live user-owned Drive archive was
not supplied, so that external download path has not been demonstrated end to end.

Recovery immediately selects a ready alternative after a failed chunk when one
exists. Stateful decoding replays the committed prefix under a new epoch; this
is not stage-local KV-cache migration. Coordinator consensus, proactive KV
replication, tensor-parallel and out-of-core operators, and arbitrary architecture
support remain research extensions. See [the implementation map](../../docs/architecture.md).
""")


if __name__ == "__main__":
    main()

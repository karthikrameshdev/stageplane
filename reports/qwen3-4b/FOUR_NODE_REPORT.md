# Qwen3 4B: direct versus four workers

> Retained earlier experiment. Project identifiers and paths were updated during migration; measured values and token IDs were preserved. See [the fresh Stageplane verification](../migration/REPORT.md).

All five measured requests produced identical token IDs in both modes. Four
workers reduced the largest worker's resident allocation but did not improve
median request latency in this experiment. These measurements are from one
memory-constrained shared Mac, not four independent physical machines.

## Model and execution

- Checkpoint: `Qwen/Qwen3-4B-Instruct-2507`.
- Pinned revision: `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Manifest SHA-256: `5eab0bbfa7ed6cf7b4dc33c1aa9ce11e7592469d84d7a30ec4ffd1690572dddd`.
- Model tensors: 4,022,468,096 parameters; BF16, no quantization.
- Artifacts: 8,044,982,168 bytes. 36 whole decoder-block stages,
  one embedding stage, one norm/output stage: **38 executable stages**.
- Host: `macOS-27.0.1-arm64-arm-64bit-Mach-O`, 10 logical CPUs, 16 GiB physical RAM.
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
| Median time to first token (s) | 2.916 | 10.382 |
| Interpolated p95 first token (s) | 7.827 | 11.385 |
| Median decode token interval (s) | 2.368 | 2.627 |
| Interpolated p95 decode interval (s) | 34.441 | 14.004 |
| Median whole request (s) | 37.527 | 81.836 |
| Median output tokens/s, including prefill | 0.426 | 0.167 |
| Observed aggregate peak RSS (decimal GB) | 8.149 | 8.630 |
| Largest worker peak RSS (decimal GB) | 8.149 | 2.519 |
| Logical serving weights/storage (decimal GB) | 8.045 | 8.823 |
| Load / cluster-ready time (s) | 41.985 | 14.062 |

| Trial | Output tokens | Direct request (s) | Four workers (s) | Exact token parity |
|---|---:|---:|---:|---|
| 1 | 9 | 129.378 | 81.836 | True |
| 2 | 16 | 37.527 | 131.074 | True |
| 3 | 16 | 23.866 | 96.029 | True |
| 4 | 9 | 19.114 | 8.476 | True |
| 5 | 16 | 291.272 | 20.750 | True |

| Worker | Executable stages | Materialized weight bytes |
|---|---:|---:|
| node-0 | 11 | 2,796,528,640 |
| node-1 | 11 | 2,220,477,952 |
| node-2 | 7 | 1,413,031,424 |
| node-3 | 9 | 2,392,810,496 |

The math answer was `2 plus 2 is 4.`. Longer answers are intentionally truncated
at the output token limit; raw text, token IDs, individual token intervals and
wire-payload counts are retained in [direct.json](four-node/direct.json) and
[distributed-4.json](four-node/distributed-4.json). Summary: [summary.json](four-node/summary.json).

## Interpretation and limits

The direct median was 37.53 s and distributed median was
81.84 s. Trial variation is substantial: the Mac compressed
and swapped model pages while Codex and Firefox remained open. Other visible
apps were already closed; the idle Docker VM and unused test publisher were
stopped during the comparison. Native and distributed models ran sequentially.
The direct warmup took 209.19 s and the distributed
warmup took 150.09 s. Five samples do not
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
/Users/karthikramesh/Projects/Self/OSS/stageplane/data/review/qwen3-4b/layer-models/Qwen--Qwen3-4B-Instruct-2507-cdbee75f17c0-l1
```

The consumer directory and portable archive are recorded in
[artifacts.json](artifacts.json). Nothing was uploaded to a public account.

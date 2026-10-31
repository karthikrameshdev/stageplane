# Qwen3 4B test results

> Retained earlier experiment. Project identifiers and paths were updated during migration; measured values and token IDs were preserved. See [the fresh Stageplane verification](../migration/REPORT.md).

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

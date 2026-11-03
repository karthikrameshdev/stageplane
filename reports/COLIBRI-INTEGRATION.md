# Interleaved layers and disk-streamed layers: analysis and results

## Context (what the report and thesis establish)
`reports/REPORT.md` (Qwen3.5-0.8B, 14 stages) shows that 4 workers cut largest-worker RSS by 60%
but cost 1.40x decode latency and more total memory. The thesis proposes a coordinator, workers
hosting transformer blocks, direct activation exchange, and replicas. Stageplane's gaps were:
placement was chunk-granular but routing was a strict worker-to-worker chain, a stage had to be
fully resident in RAM, and `MAX_CONTEXT` is 2048.

## colibri (JustVugg/colibri, cloned to `reference/colibri`, git-ignored)
A C inference engine for huge MoE models on small hardware. Ideas relevant here:
- Weights are *placed*, not required to fit: dense parts in RAM, bulk on disk, a learned LRU cache.
- Placement may only change speed, never numerics (same tensors, same router decisions).
- Overlap I/O with compute (prefetch next layer), single-read units, mirrored disks.
- Long context: float16 KV, compressed KV, parking/reusing conversation state.

## What was implemented
1. **Interleaved, size-proportional layer placement** (`strategy="capacity"`): each layer chunk goes to the
   node with the lowest fill ratio, so a 1-layer device, an 8-layer node and a 15-layer node coexist
   and own non-contiguous layers (e.g. 1, 6, 8 / 2, 4, 7). Legacy `rendezvous` remains selectable.
2. **Hub routing** (`routing="hub"`, default): a node returns the activation to the executor at every node
   boundary and the executor forwards it to the next stage's owner. Colocated consecutive stages still run
   in-process. `routing="chain"` restores the old worker-to-worker behavior. Recovery/replay is unchanged.
3. **Disk-streamed layers** (`node --stream-weights --layer-cache 256MiB`): a `LayerStreamer` keeps
   layers on disk, holds a byte-bounded LRU in RAM and prefetches layer n+1 during layer n.
   Planner accounting charges a streaming node only one layer of working memory plus disk for the
   rest, so edge devices with little RAM can host any layer. Output is bit-identical to resident execution.

## Verification
- Full suite: **174 passed** (167 pre-existing + 7 new, `tests/test_layer_scheduling.py`).
- Placement with 1/8/15-layer budgets assigns 1/8/15 layers, all 24 placed, non-contiguous ownership.
- Streaming stage output is `torch.equal` to resident output across prefill and decode, with eviction forced.
- Live 3-node cluster (REST+gRPC+WebSocket) with hub routing: generated text equals the single-node chain
  baseline, and the route revisits nodes after other nodes ran (interleaved).
- Live streaming node with a 1 MiB cache: identical text, resident weights < 1 MiB, cache misses recorded.

## Not done / honest limits
- Not benchmarked on the real Qwen model; hub routing adds one executor round trip per node boundary
  (boundary bytes are accounted double), so latency is expected to rise versus chain.
- Long context: `MAX_CONTEXT` stays 2048 (env-configurable). Float16 KV, KV paging to disk and
  conversation-state parking from colibri are the next steps; the layer streamer is the base for them.
- Streamed layers are re-read from disk on cache miss with no learned hot set yet.

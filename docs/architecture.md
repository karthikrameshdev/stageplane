# Implementation map

The thesis is an architecture proposal. This repository implements its initial
decoder-only serving system and records the limits of the actual experiment.

| Mechanism | Implementation | Evidence / boundary |
|---|---|---|
| Compatible execution graph | Whole contiguous blocks, embedding, norm/head | Native prefill/decode logit parity for five adapters |
| Versioned execution contract | Manifest, pinned commit, tokenizer and artifact hashes | Integrity, revision, shape, offset, epoch and lease rejection tests |
| Distribution | Bounded overlapping range reads or local Safetensors; verified portable bundles | Full official Qwen3.5-0.8B and Qwen3-4B checkpoints partitioned; 4B bundle imported over HTTP |
| Control/data separation | Executor assignments; direct worker tensors | Live mixed REST/gRPC/WebSocket tests |
| Heterogeneous placement | Exact CPU/memory limits, live headroom/storage, speed hints, observed stage costs, declared failure domains | Planner tests, capped-container admission evidence and local CPU measurements |
| Local rebalancing | Stable stage groups and group-scoped rendezvous replans | Unaffected-group assignments remain identical in tests |
| Replicas | Configurable target constrained by memory and eligible hosts | Live replica recovery tests |
| Stateful decode | Native attention KV and hybrid recurrent state | Prefill and repeated decode match native logits |
| Recovery | Bounded executor retries, chunk deadlines, failed-host exclusion, epoch fencing and committed-prefix replay | Worker killed after output starts; completed output matches uninterrupted run; timeout/exhaustion tests |
| Admission/backpressure | Gateway request cap, worker sessions/compute caps, memory estimates | Validation tests; no continuous batching |
| Operations UI | Light dashboard, Three.js, node health/RSS/stages, traces, imports and drains | Live browser inference verified |
| OpenAI protocol | Model list and text chat, sampling, SSE | Actual OpenAI SDK streaming/non-streaming tests |
| Protected transport | Bearer authentication and mutual TLS | Real certificate-backed test; missing client certificate rejected |
| Evaluation | Native direct vs one/four workers; raw tokens, timings, RSS, storage | Raw JSON and report retained |
| CPU budgets | Explicit cores/millicores/percent; detected cgroup quota; bare-host CPU governor | Exact default-capacity registration test and measured governor sample; bare-host RAM flag is admission, Docker RAM limit is enforced |
| Coordinator failover | SQLite deployment persistence only | Consensus HA and transferable gateway sessions not implemented |
| Warm-state recovery | Prefix replay baseline | Cache snapshots and proactive cache replication not implemented |
| Device/network eligibility | Declared serving budgets and speed, observed stage timings | Active link profiling, thermal/battery/foreground policies not implemented |
| Optional optimizations | Extension points, not advertised features | Continuous batching, quantization kernels, speculative decode, split prefill/decode, MoE, tensor parallelism and encoder-decoder/multimodal execution not implemented |

More nodes can expand resident capacity and provide replica alternatives.
Autoregressive layers still depend on preceding layers. Single-session speedup
is neither assumed nor guaranteed. The local experiment does not establish the
paper's enterprise goodput, productivity, energy or cost hypotheses. Those require
physical heterogeneous hosts, calibrated service objectives, representative
concurrent workloads, and task-quality evaluation.

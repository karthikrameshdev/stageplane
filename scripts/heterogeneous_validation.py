"""Real-model validation of interleaved placement, hub routing, streamed layers and long context.

Runs a heterogeneous pool against an already-distributed checkpoint and compares every
configuration's greedy tokens with a single resident worker on the identical prompt.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import psutil

from scripts.cluster import Cluster


def chat(cluster, prompt, tokens):
    started = time.perf_counter()
    first, text = None, ""
    with cluster.http.stream("POST", "/v1/chat/completions", timeout=900, json={
            "model": cluster.model, "temperature": 0, "max_tokens": tokens, "stream": True,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}) as response:
        response.raise_for_status()
        usage = None
        for line in response.iter_lines():
            if not line.startswith("data: {"):
                continue
            item = json.loads(line[6:])
            if "error" in item:
                raise RuntimeError(item["error"])
            if item["choices"]:
                delta = item["choices"][0]["delta"].get("content", "")
                if delta and first is None:
                    first = time.perf_counter() - started
                text += delta
            else:
                usage = item["usage"]
    return {"text": text, "ttft_s": first, "total_s": time.perf_counter() - started, "usage": usage}


def worker_rss(cluster):
    result = {}
    for name, process in cluster.node_processes.items():
        try:
            result[name] = psutil.Process(process.pid).memory_info().rss
        except psutil.Error:
            pass
    return result


def scenario(store, work, name, nodes, node_args, deploy, prompts, tokens, env=None, dtype="bfloat16"):
    cluster = Cluster(store, Path(work) / name, nodes=nodes, transports=["rest", "grpc", "websocket"],
                      deploy_options=deploy, node_args=node_args, memory_gb=None, threads=2, dtype=dtype)
    cluster.env.update(env or {})
    try:
        cluster.start()
        state = cluster.wait_ready(timeout=900)
        runs, peak = [], {}
        for prompt in prompts:
            runs.append(chat(cluster, prompt, tokens))
            for k, v in worker_rss(cluster).items():
                peak[k] = max(peak.get(k, 0), v)
        state = cluster.http.get("/api/state").json()
        route = [h["node"] for h in state["requests"][-1]["route"]]
        nodes_info = {n["id"]: {"ready_stages": sorted(k.split(":")[-1] for k in n["ready"]),
                                "streaming": n.get("streaming", False),
                                "loaded_weight_bytes": n["metrics"].get("loaded_weight_bytes"),
                                "layer_stream": n["metrics"].get("layer_stream")} for n in state["nodes"]}
        return {"name": name, "deploy": deploy, "node_args": {str(k): v for k, v in node_args.items()},
                "route": route, "nodes": nodes_info, "peak_worker_rss_bytes": peak,
                "prefill_chunks": state["requests"][-1].get("prefill_chunks"),
                "runs": runs, "decode_tokens_per_s": statistics.median(
                    max(0, r["usage"]["completion_tokens"] - 1) / max(1e-9, r["total_s"] - r["ttft_s"]) for r in runs)}
    finally:
        cluster.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("data/models"))
    parser.add_argument("--work", type=Path, default=Path("/tmp/stageplane-heterogeneous"))
    parser.add_argument("--output", type=Path, default=Path("reports/enterprise/heterogeneous.json"))
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--resident-memory", default="3GiB",
                        help="Budget of the single resident reference worker (bfloat16 Qwen3.5-0.8B needs about 2.1 GB)")
    args = parser.parse_args()
    prompts = ["Explain in one sentence why the sky is blue.",
               "List three prime numbers.",
               # Roughly 1.5k tokens: exercises chunked prefill across several frames.
               "Summarise the following notes in one line. " + " ".join(
                   f"Note {i}: the service handled request batch {i} with latency {i % 17} ms." for i in range(110))]
    env = {"STAGEPLANE_PREFILL_CHUNK": "256", "STAGEPLANE_CHUNK_TIMEOUT": "300"}
    results = [scenario(args.store, args.work, "single-resident", 1, {0: ["--max-memory", args.resident_memory]},
                        {"routing": "chain", "strategy": "rendezvous"}, prompts, args.tokens, env)]
    # Heterogeneous pool of memory-constrained devices: two medium streaming nodes and a tiny
    # one. Interleaved placement and hub routing place 14 stages across them; none could hold
    # the 2 GB bfloat16 model resident.
    results.append(scenario(args.store, args.work, "heterogeneous-hub-interleaved", 3, {
        0: ["--max-memory", "1400MiB", "--stream-weights", "--layer-cache", "160MiB"],
        1: ["--max-memory", "1400MiB", "--stream-weights", "--layer-cache", "160MiB"],
        2: ["--max-memory", "700MiB", "--stream-weights", "--layer-cache", "32MiB"]},
        {"routing": "hub", "strategy": "capacity"}, prompts, args.tokens, env))
    results.append(scenario(args.store, args.work, "single-streaming-edge", 1, {
        0: ["--max-memory", "1800MiB", "--stream-weights", "--layer-cache", "96MiB"]},
        {"routing": "hub", "strategy": "capacity"}, prompts, args.tokens, env))
    # Same resident worker with prompts executed in one frame: chunked prefill must not change output.
    results.append(scenario(args.store, args.work, "single-resident-unchunked", 1, {0: ["--max-memory", args.resident_memory]},
                            {"routing": "chain", "strategy": "rendezvous"}, prompts, args.tokens,
                            {**env, "STAGEPLANE_PREFILL_CHUNK": "8192"}))
    reference = [r["text"] for r in results[0]["runs"]]
    for r in results:
        r["matches_single_resident"] = [run["text"] == ref for run, ref in zip(r["runs"], reference)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for r in results:
        print(json.dumps({"name": r["name"], "match": r["matches_single_resident"], "route": r["route"],
                          "decode_tok_s": round(r["decode_tokens_per_s"], 3),
                          "peak_rss_gb": {k: round(v / 1e9, 3) for k, v in r["peak_worker_rss_bytes"].items()},
                          "ttft_s": [round(x["ttft_s"], 2) for x in r["runs"]]}))


if __name__ == "__main__":
    main()

"""Same-checkpoint CPU comparison, with raw trials and explicit topology limits.

python -m scripts.benchmark --model MODEL_KEY --trials 5 --tokens 16
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from scripts.cluster import Cluster

PROMPTS = ["What is 2 plus 2? Answer briefly.",
           "Explain why distributed transformer inference needs to transfer activations between successive layers.",
           "Write one sentence about a reliable distributed system, mentioning worker health and recovery."]


class MemorySampler:
    def __init__(self, pids):
        self.pids = pids
        self.peak = {str(p): 0 for p in pids}
        self.total_peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.wait(.02):
            total = 0
            for pid in self.pids:
                try:
                    rss = psutil.Process(pid).memory_info().rss
                    self.peak[str(pid)] = max(self.peak[str(pid)], rss)
                    total += rss
                except psutil.NoSuchProcess:
                    pass
            self.total_peak = max(self.total_peak, total)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def direct(args):
    os.environ["STAGEPLANE_CPU_KERNEL"] = args.cpu_kernel
    import torch
    from transformers import AutoTokenizer
    from stageplane.runtime import load_reference
    torch.set_num_threads(2)
    directory = args.store / args.model
    manifest = json.loads((directory / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    with MemorySampler([os.getpid()]) as loading:
        start = time.perf_counter()
        model = load_reference(manifest, directory, args.dtype)
        load_s = time.perf_counter() - start

    def trial(prompt):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        value, generated, cache, timings = torch.tensor([ids]), [], None, []
        start = time.perf_counter()
        with torch.inference_mode():
            for i in range(args.tokens):
                tick = time.perf_counter()
                output = model(value, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                token = int(output.logits[0, -1].argmax())
                generated.append(token)
                timings.append((time.perf_counter() - tick) * 1000)
                value = torch.tensor([[token]])
                if token == tokenizer.eos_token_id:
                    break
        elapsed = (time.perf_counter() - start) * 1000
        return {"prompt": prompt, "prompt_tokens": len(ids), "token_ids": generated,
                "output": tokenizer.decode(generated, skip_special_tokens=True), "elapsed_ms": elapsed,
                "ttft_ms": timings[0], "token_ms": timings, "tokens_per_second": len(generated) / (elapsed / 1000)}

    print("Direct warmup started", flush=True)
    warmup = trial(PROMPTS[0])
    print(f"Direct warmup: {warmup['elapsed_ms']:.1f} ms", flush=True)
    with MemorySampler([os.getpid()]) as memory:
        trials = []
        for i in range(args.trials):
            trials.append(trial(PROMPTS[i % len(PROMPTS)]))
            (args.output / "direct-progress.json").write_text(json.dumps(trials, indent=2))
            print(f"Direct trial {i+1}/{args.trials}: {trials[-1]['elapsed_ms']:.1f} ms", flush=True)
    result = {"mode": "direct-native", "load_s": load_s, "warmup": warmup, "trials": trials,
              "peak_rss_bytes": memory.total_peak, "load_peak_rss_bytes": loading.total_peak,
              "threads": 2, "dtype": args.dtype, "cpu_kernel": args.cpu_kernel}
    (args.output / "direct.json").write_text(json.dumps(result, indent=2))


def distributed(args, count):
    work = args.output / f"cluster-{count}"
    c = Cluster(args.store, work, nodes=count, replicas=1, transports=["rest"], local_artifacts=True,
                dtype=args.dtype, memory_gb=args.memory_gb, cpu_kernel=args.cpu_kernel)
    try:
        tick = time.perf_counter()
        c.start(args.model)
        load_s = time.perf_counter() - tick

        def trial(prompt):
            def wire_totals():
                snapshot = c.http.get("/api/state").json()
                counters = [snapshot.get("transport", {})]
                for node in snapshot["nodes"]:
                    counters.append(c.http.get(node["url"] + "/health").json()["metrics"].get("transport", {}))
                return sum(x.get("request_payload_bytes", 0) + x.get("response_payload_bytes", 0) for x in counters)
            wire_before = wire_totals()
            text, first, timings = "", None, []
            tick = prev = time.perf_counter()
            usage = {}
            with c.http.stream("POST", "/v1/chat/completions", json={"model": args.model,
                    "messages": [{"role": "user", "content": prompt}], "max_tokens": args.tokens,
                    "temperature": 0, "stream": True}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    item = json.loads(line[6:])
                    if "error" in item:
                        raise RuntimeError(item["error"])
                    text += item["choices"][0]["delta"].get("content", "")
                    now = time.perf_counter()
                    if item["choices"][0]["finish_reason"] is None:
                        if first is None:
                            first = (now - tick) * 1000
                        timings.append((now - prev) * 1000)
                        prev = now
                    usage = item.get("usage", usage)
            elapsed = (time.perf_counter() - tick) * 1000
            state = c.http.get("/api/state").json()
            record = state["requests"][-1]
            token_events = [e for e in state["events"] if e["kind"] == "token_committed" and e["request"] == record["id"]]
            return {"prompt": prompt, "prompt_tokens": usage["prompt_tokens"],
                    "token_ids": [e["token_id"] for e in token_events], "output": text,
                    "elapsed_ms": elapsed, "ttft_ms": first, "token_ms": timings,
                    "wire_payload_bytes": wire_totals() - wire_before,
                    "tokens_per_second": usage["completion_tokens"] / (elapsed / 1000), "server_metrics": record}

        warmup = trial(PROMPTS[0])
        print(f"{count}-worker warmup: {warmup['elapsed_ms']:.1f} ms", flush=True)
        pids = [p.pid for p in c.processes]
        with MemorySampler(pids) as memory:
            trials = []
            for i in range(args.trials):
                trials.append(trial(PROMPTS[i % len(PROMPTS)]))
                (args.output / f"distributed-{count}-progress.json").write_text(json.dumps(trials, indent=2))
                print(f"{count}-worker trial {i+1}/{args.trials}: {trials[-1]['elapsed_ms']:.1f} ms", flush=True)
        state = c.http.get("/api/state").json()
        nodes = {n["id"]: n for n in state["nodes"]}
        logical_storage = sum(n["metrics"].get("cache_bytes", 0) for n in state["nodes"])
        manifest = json.loads((args.store / args.model / "manifest.json").read_text())
        current_storage = 0
        placements = state["deployments"][args.model]["placements"]
        for node in state["nodes"]:
            artifacts = {a for stage in manifest["stages"] if node["id"] in placements.get(stage["id"], [])
                         for a in stage["artifacts"]}
            current_storage += sum(manifest["artifacts"][a]["bytes"] for a in artifacts)
        return {"mode": f"distributed-{count}", "load_s": load_s, "warmup": warmup, "trials": trials,
                "peak_rss_bytes": memory.total_peak, "per_process_peak_rss_bytes": memory.peak,
                "max_worker_rss_bytes": max((memory.peak[str(p.pid)] for p in c.node_processes.values()), default=0),
                "logical_worker_storage_bytes": current_storage, "cached_worker_storage_bytes": logical_storage,
                "local_cache_hardlinks": True,
                "placements": state["deployments"][args.model], "nodes": nodes,
                "transports": ["rest"], "threads_per_worker": 2, "dtype": args.dtype, "cpu_kernel": args.cpu_kernel}
    finally:
        c.close()


def percentile(values, q):
    vals = sorted(values)
    index = (len(vals) - 1) * q
    lo = int(index)
    return vals[lo] + (vals[min(lo + 1, len(vals) - 1)] - vals[lo]) * (index - lo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", type=Path, default=Path("data/models"))
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, default=Path("reports/qwen"))
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--memory-gb", type=float, default=6)
    p.add_argument("--cpu-kernel", choices=["native", "bounded-f32"], default="native")
    p.add_argument("--direct-child", action="store_true")
    p.add_argument("--skip-single", action="store_true")
    p.add_argument("--reuse-direct", action="store_true", help="Reuse an unchanged direct.json while rerunning service trials")
    args = p.parse_args()
    args.store, args.output = args.store.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.direct_child:
        return direct(args)
    manifest = json.loads((args.store / args.model / "manifest.json").read_text())
    if not args.reuse_direct:
        subprocess.run([sys.executable, "-m", "scripts.benchmark", "--store", str(args.store), "--model", args.model,
                        "--output", str(args.output), "--trials", str(args.trials), "--tokens", str(args.tokens), "--dtype", args.dtype,
                        "--cpu-kernel", args.cpu_kernel, "--direct-child"], check=True)
    results = [json.loads((args.output / "direct.json").read_text())]
    for count in ([4] if args.skip_single else [1, 4]):
        result = distributed(args, count)
        (args.output / f"distributed-{count}.json").write_text(json.dumps(result, indent=2))
        results.append(result)
    parity = all(a["token_ids"] == b["token_ids"] for r in results[1:] for a, b in zip(results[0]["trials"], r["trials"]))
    summary = {"host": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                "memory_bytes": psutil.virtual_memory().total}, "model": args.model, "source": manifest["source"],
               "revision": manifest["revision"], "artifact_bytes": sum(a["bytes"] for a in manifest["artifacts"].values()),
               "parity": parity, "dtype": args.dtype, "cpu_kernel": args.cpu_kernel, "results": []}
    for r in results:
        ttfts = [t["ttft_ms"] for t in r["trials"]]
        decodes = [ms for t in r["trials"] for ms in t["token_ms"][1:]]
        summary["results"].append({"mode": r["mode"], "ttft_p50_ms": statistics.median(ttfts),
            "ttft_p95_ms": percentile(ttfts, .95), "decode_p50_ms": statistics.median(decodes),
            "decode_p95_ms": percentile(decodes, .95), "request_p50_ms": statistics.median(t["elapsed_ms"] for t in r["trials"]),
            "tokens_per_second_median": statistics.median(t["tokens_per_second"] for t in r["trials"]),
            "peak_rss_bytes": r["peak_rss_bytes"], "max_worker_rss_bytes": r.get("max_worker_rss_bytes", r["peak_rss_bytes"]),
            "logical_storage_bytes": r.get("logical_worker_storage_bytes", summary["artifact_bytes"]), "load_s": r["load_s"]})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not parity:
        raise SystemExit("Generated token IDs differ: inspect raw trials before interpreting performance")


if __name__ == "__main__":
    main()

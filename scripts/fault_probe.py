"""Crash a worker after real output, then compare the committed transcript."""
import argparse
import json
import time
from pathlib import Path

from scripts.cluster import Cluster


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", type=Path, default=Path("data/models"))
    p.add_argument("--model", required=True)
    p.add_argument("--reference", type=Path, default=Path("reports/qwen/direct.json"))
    p.add_argument("--output", type=Path, default=Path("reports/qwen/fault.json"))
    args = p.parse_args()
    reference = json.loads(args.reference.read_text())["trials"][1]
    c = Cluster(args.store, args.output.parent / "fault-cluster", nodes=4, replicas=1, local_artifacts=True)
    result = {"model": args.model, "prompt": reference["prompt"], "replicas": 1,
              "policy": "cold replacement and committed-prefix replay"}
    try:
        c.start(args.model)
        tokens, text, killed = [], "", False
        start = time.perf_counter()
        last_token = start
        pauses = []
        with c.http.stream("POST", "/v1/chat/completions", json={"model": args.model,
                "messages": [{"role": "user", "content": reference["prompt"]}],
                "max_tokens": 16, "temperature": 0, "stream": True}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                item = json.loads(line[6:])
                if "error" in item:
                    raise RuntimeError(item["error"])
                text += item["choices"][0]["delta"].get("content", "")
                pauses.append((time.perf_counter() - last_token) * 1000)
                last_token = time.perf_counter()
                if not killed:
                    state = c.http.get("/api/state").json()
                    event = next(e for e in reversed(state["events"]) if e["kind"] == "request_started")
                    victim = event["route"][len(event["route"]) // 2]["node"]
                    result["victim"] = victim
                    c.kill_node(victim)
                    killed = True
        state = c.http.get("/api/state").json()
        record = state["requests"][-1]
        events = [e for e in state["events"] if e.get("request") == record["id"]]
        tokens = [e["token_id"] for e in events if e["kind"] == "token_committed"]
        result.update(output=text, token_ids=tokens, parity=tokens == reference["token_ids"],
                      elapsed_ms=(time.perf_counter() - start) * 1000,
                      largest_output_gap_ms=max(pauses), server_metrics=record, events=events)
        args.output.write_text(json.dumps(result, indent=2))
        assert result["parity"] and record["recoveries"] >= 1 and record["status"] == "completed", result
        print(json.dumps({k: v for k, v in result.items() if k not in ("events", "server_metrics")}, indent=2))
    finally:
        c.close()


if __name__ == "__main__":
    main()

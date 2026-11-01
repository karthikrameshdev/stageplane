"""Check deterministic output and request cleanup on a running fixture cluster."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("reports/hardening/concurrency.json"))
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.tokens) <= 0:
        parser.error("Request, concurrency and token counts must be positive")
    token = os.environ.get("STAGEPLANE_TOKEN", "local-stageplane-development")
    body = {"model": args.model, "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": args.tokens, "temperature": 0}
    with httpx.Client(base_url=args.url, headers={"Authorization": "Bearer " + token}, timeout=90, trust_env=False) as client:
        response = client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        baseline = response.json()
        def trial(index):
            started = time.perf_counter()
            response = client.post("/v1/chat/completions", json=body)
            result = response.json() if response.status_code == 200 else {}
            return {"i": index, "status": response.status_code, "seconds": time.perf_counter() - started,
                    "parity": result.get("choices") == baseline["choices"] and result.get("usage") == baseline["usage"]}
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            trials = list(pool.map(trial, range(args.requests)))
        elapsed = time.perf_counter() - started
        state = client.get("/api/state")
        state.raise_for_status()
        latency = sorted(item["seconds"] for item in trials)
        report = {"kind": "actual-running-cluster-concurrency", "model": args.model,
                  "workers": len(state.json()["nodes"]), "requests": args.requests,
                  "concurrency": args.concurrency, "max_output_tokens": args.tokens,
                  "elapsed_seconds": elapsed, "requests_per_second": len(trials) / elapsed,
                  "p50_seconds": latency[math.ceil(len(latency) * .50) - 1],
                  "p95_seconds": latency[math.ceil(len(latency) * .95) - 1],
                  "all_parity": all(item["parity"] for item in trials),
                  "statuses": {str(code): sum(item["status"] == code for item in trials)
                               for code in {item["status"] for item in trials}},
                  "active_requests_after": state.json()["summary"]["active_requests"], "trials": trials}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))
        if not report["all_parity"] or report["active_requests_after"]:
            raise SystemExit("Concurrency validation failed; inspect the raw trials")


if __name__ == "__main__":
    main()

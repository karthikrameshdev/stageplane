"""Run a real, strictly limited heterogeneous Docker pool; retain failures.

No replacement model, precision change or swap escape is used. RAM tiers are
explicit so a retry with larger limits is recorded separately.
If Docker cannot allocate storage for 50 containers, clean up and retry with 25.
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import psutil

from scripts.cluster import free_port


def save_evidence(output, result, container_names):
    """Persist measurements before optional log collection can fail under load."""
    path = output / "result.json"
    path.write_text(json.dumps(result, indent=2))
    errors = []
    for name in container_names:
        filename = "executor" if name.endswith("-executor") else "node-" + name.rsplit("-node-", 1)[-1]
        try:
            proc = subprocess.run(["docker", "logs", name], capture_output=True, text=True, timeout=10)
            (output / (filename + ".log")).write_text(proc.stdout + proc.stderr)
            if proc.returncode:
                errors.append({"container": name, "error": proc.stderr.strip()})
        except (subprocess.TimeoutExpired, OSError) as error:
            errors.append({"container": name, "error": str(error)})
    result["log_capture_errors"] = errors
    path.write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/qwen3-4b/heterogeneous-50"))
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--cpu-cores", type=float, default=.05,
                        help="Hard per-worker CPU quota, also published to the executor")
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--load-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=600,
                        help="HTTP inference deadline; execution-layer deadlines remain separate")
    parser.add_argument("--memory-tiers-mb", default="150,250,350,400,500",
                        help="Comma-separated decimal MB limits, repeated across workers")
    parser.add_argument("--image", default="stageplane:qwen4-review")
    parser.add_argument("--cpu-kernel", choices=["native", "bounded-f32"], default="bounded-f32")
    args = parser.parse_args()
    if args.cpu_cores <= 0:
        parser.error("CPU cores must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    store = args.store.resolve()
    run = f"stageplane-review-{os.getpid()}"
    token = "local-resource-review"
    names = []
    network = run + "-network"
    tiers = [int(value) for value in args.memory_tiers_mb.split(",")]
    if not tiers or min(tiers) <= 0:
        parser.error("Memory tiers must be positive integer MB values")
    result = {"model": args.model, "requested_nodes": args.nodes, "memory_tiers_MB": tiers,
              "cpu_cores_per_node": args.cpu_cores, "swap_allowed": False, "storage_fallback": False,
              "artifact_location": str(store / args.model), "image": args.image, "cpu_kernel": args.cpu_kernel}
    result["host_memory_bytes"] = psutil.virtual_memory().total
    result.update(startup_timeout_s=args.startup_timeout, load_timeout_s=args.load_timeout,
                  request_timeout_s=args.request_timeout)

    def docker(*cmd, check=True):
        proc = subprocess.run(["docker", *cmd], capture_output=True, text=True, timeout=90)
        if check and proc.returncode:
            raise RuntimeError(proc.stderr.strip())
        return proc.stdout.strip()

    def cleanup():
        if names:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda name: docker("rm", "-f", name, check=False), names))
            names.clear()
        docker("network", "rm", network, check=False)

    try:
        result["docker_memory_bytes"] = int(docker("info", "--format", "{{.MemTotal}}"))
        for count in [args.nodes, 25] if args.nodes == 50 else [args.nodes]:
            try:
                docker("network", "create", network)
                port = free_port()
                executor = run + "-executor"
                names.append(executor)
                state = args.output.resolve() / "state"
                state.mkdir(exist_ok=True)
                docker("run", "-d", "--name", executor, "--network", network,
                       "--memory", "768000000", "--cpus", "1", "-p", f"127.0.0.1:{port}:8080",
                       "-e", f"STAGEPLANE_TOKEN={token}", "-e", "STAGEPLANE_STATE_FILE=/state/executor.sqlite",
                       "-e", "STAGEPLANE_HEARTBEAT_TIMEOUT=60", "-v", f"{store}:/models:ro",
                       "-v", f"{state}:/state", args.image, "executor", "--host", "0.0.0.0", "--store", "/models")
                client = httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": f"Bearer {token}"}, timeout=30, trust_env=False)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.5)
                else:
                    raise RuntimeError("Resource-test executor did not become healthy")
                worker_names = [run + f"-node-{i:02d}" for i in range(count)]
                names.extend(worker_names)
                def start(i):
                    name = worker_names[i]
                    memory = tiers[i % len(tiers)] * 1_000_000
                    docker("run", "-d", "--name", name, "--network", network,
                           "--memory", str(memory), "--memory-swap", str(memory), "--cpus", str(args.cpu_cores),
                           "-e", f"STAGEPLANE_TOKEN={token}", args.image, "node", "--id", f"node-{i:02d}",
                           "--executor", f"http://{executor}:8080", "--host", "0.0.0.0",
                           "--advertise-url", f"http://{name}:8101", "--max-memory", f"{tiers[i % len(tiers)]}MB",
                           "--max-cpu", str(args.cpu_cores), "--threads", "1", "--dtype", "bfloat16", "--cpu-kernel", args.cpu_kernel)
                with ThreadPoolExecutor(max_workers=5) as pool:
                    list(pool.map(start, range(count)))
                print(f"Started {count} containers with strict memory/CPU limits.", flush=True)
                deadline = time.monotonic() + args.startup_timeout
                while time.monotonic() < deadline:
                    snapshot = client.get("/api/state").json()
                    observed = {n["id"] for n in snapshot["nodes"]}
                    inspected = json.loads(docker("inspect", *worker_names))
                    completed = all(f"node-{i:02d}" in observed or not row["State"]["Running"] for i, row in enumerate(inspected))
                    if completed:
                        break
                    time.sleep(2)
                deploy = client.post("/api/deployments", json={"model": args.model, "replicas": 1, "groups": 1})
                loading_started = time.perf_counter()
                route_ready = False
                if deploy.status_code == 200:
                    deadline = time.monotonic() + args.load_timeout
                    while time.monotonic() < deadline:
                        snapshot = client.get("/api/state").json()
                        model = next(m for m in snapshot["models"] if m["key"] == args.model)
                        placement = snapshot["deployments"][args.model]
                        nodes = {n["id"]: n for n in snapshot["nodes"]}
                        route_ready = all(any(nid in nodes and nodes[nid]["active"] and
                                              args.model + ":" + stage["id"] in nodes[nid]["ready"]
                                              for nid in placement["placements"].get(stage["id"], []))
                                          for stage in model["stages"])
                        if route_ready or placement.get("capacity", {}).get("status") == "insufficient_capacity":
                            break
                        time.sleep(2)
                result.update(route_ready_before_inference=route_ready,
                              loading_wait_ms=(time.perf_counter()-loading_started)*1000)
                tick = time.perf_counter()
                response = client.post("/v1/chat/completions", json={"model": args.model,
                    "messages": [{"role": "user", "content": "What is 2 plus 2? Answer briefly."}], "max_tokens": 16, "temperature": 0}, timeout=args.request_timeout)
                snapshot = client.get("/api/state").json()
                inspected = json.loads(docker("inspect", *worker_names))
                states = [{"node": f"node-{i:02d}", "memory_bytes": row["HostConfig"]["Memory"],
                           "memory_swap_bytes": row["HostConfig"]["MemorySwap"], "nano_cpus": row["HostConfig"]["NanoCpus"],
                           "state": row["State"]} for i, row in enumerate(inspected)]
                result.update(actual_nodes=count, deploy_status=deploy.status_code, deploy_response=deploy.json(),
                              inference_status=response.status_code, inference_response=response.json(),
                              response_ms=(time.perf_counter()-tick)*1000, cluster=snapshot,
                              containers=states, oom_count=sum(row["state"]["OOMKilled"] for row in states),
                              registered_nodes=len(snapshot["nodes"]), completed_inference=response.status_code == 200)
                save_evidence(args.output, result, [executor, *worker_names])
                client.close()
                break
            except RuntimeError as error:
                if count == 50 and "no space left on device" in str(error).lower():
                    result.update(storage_fallback=True, fifty_node_storage_error=str(error))
                    cleanup()
                    continue
                raise
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("cluster", "containers")}, indent=2))
    finally:
        cleanup()


if __name__ == "__main__":
    main()

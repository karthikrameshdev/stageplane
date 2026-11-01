"""Owned local services for integration tests and reproducible benchmarks."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def submit_deployment(client, model, replicas=1, groups=1):
    """Submit once; accept a confirmed pending-capacity plan for readiness waiting."""
    response = client.post("/api/deployments", json={"model": model, "replicas": replicas, "groups": groups})
    pending_capacity = False
    if response.status_code == 503:
        # Deployment admission persists a capacity-constrained plan before
        # returning 503. Startup telemetry can settle on a later heartbeat;
        # confirm that exact pending plan before using the caller's bounded wait.
        confirmation = client.get("/api/state")
        confirmation.raise_for_status()
        deployment = confirmation.json().get("deployments", {}).get(model, {})
        pending_capacity = (deployment.get("capacity", {}).get("status") == "insufficient_capacity"
                            and deployment.get("replicas") == replicas
                            and deployment.get("groups") == groups)
    if not pending_capacity:
        response.raise_for_status()
    return response


class Cluster:
    def __init__(self, store, work, nodes=4, replicas=1, groups=1, transports=None, port=None, token="test-stageplane-token", local_artifacts=False, certs=None, dtype="float32", memory_gb=6, threads=2, cpu_kernel="native"):
        self.store, self.work = Path(store).resolve(), Path(work).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.port, self.count = port or free_port(), nodes
        self.url = f"{'https' if certs else 'http'}://127.0.0.1:{self.port}"
        self.replicas, self.groups = replicas, groups
        self.transports = transports or ["rest", "grpc", "websocket", "rest"]
        self.dtype, self.memory_gb, self.threads = dtype, memory_gb, threads
        self.cpu_kernel = cpu_kernel
        self.env = {**os.environ, "STAGEPLANE_TOKEN": token, "STAGEPLANE_HEARTBEAT_TIMEOUT": "4",
                    "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                    "STAGEPLANE_STATE_FILE": str(self.work / "executor.sqlite")}
        self.env.pop("STAGEPLANE_LOCAL_ARTIFACTS", None)
        if local_artifacts:
            self.env["STAGEPLANE_LOCAL_ARTIFACTS"] = str(self.store)
        self.certs = Path(certs).resolve() if certs else None
        tls = {}
        if self.certs:
            self.env.update(STAGEPLANE_CA=str(self.certs / "ca.crt"), STAGEPLANE_TLS_CERT=str(self.certs / "worker.crt"),
                STAGEPLANE_TLS_KEY=str(self.certs / "worker.key"), STAGEPLANE_CLIENT_CERT=str(self.certs / "worker.crt"),
                STAGEPLANE_CLIENT_KEY=str(self.certs / "worker.key"))
            import ssl
            ctx = ssl.create_default_context(cafile=str(self.certs / "ca.crt"))
            ctx.load_cert_chain(str(self.certs / "worker.crt"), str(self.certs / "worker.key"))
            tls = {"verify": ctx}
        self.http = httpx.Client(base_url=self.url, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=120, trust_env=False, **tls)
        self.processes, self.logs, self.node_processes = [], [], {}

    def spawn(self, name, args):
        log = (self.work / f"{name}.log").open("w")
        self.logs.append(log)
        p = subprocess.Popen([sys.executable, "-m", "stageplane", *args], cwd=REPO, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        self.processes.append(p)
        return p

    def wait_http(self, path="/health", timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = self.http.get(path, timeout=3)
                if r.status_code == 200:
                    return r.json()
            except httpx.HTTPError:
                pass
            if self.processes and self.processes[0].poll() is not None:
                raise RuntimeError((self.work / "executor.log").read_text())
            time.sleep(0.25)
        raise TimeoutError(f"No healthy executor; logs: {self.work}")

    def start_node(self, i):
        port, grpc = free_port(), free_port()
        mode = self.transports[i % len(self.transports)]
        p = self.spawn(f"node-{i}", ["node", "--id", f"node-{i}", "--executor", self.url,
                 "--port", str(port), "--grpc-port", str(grpc), "--transport", mode,
                 "--advertise-url", f"{'https' if self.certs else 'http'}://127.0.0.1:{port}",
                 "--group", str(i % self.groups), "--cache", str(self.work / "cache"),
                 "--memory-gb", str(self.memory_gb), "--threads", str(self.threads), "--dtype", self.dtype,
                 "--cpu-kernel", self.cpu_kernel])
        self.node_processes[f"node-{i}"] = p

    def start(self, model=None):
        self.spawn("executor", ["executor", "--store", str(self.store), "--port", str(self.port)])
        self.wait_http()
        state = self.http.get("/api/state").json()
        self.model = model or state["models"][0]["key"]
        for i in range(self.count):
            self.start_node(i)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = self.http.get("/api/state").json()
            if len(state["nodes"]) == self.count:
                break
            time.sleep(.25)
        submit_deployment(self.http, self.model, self.replicas, self.groups)
        self.wait_ready()
        return self

    def wait_ready(self, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.http.get("/api/state").json()
            m = next(m for m in state["models"] if m["key"] == self.model)
            d = state["deployments"][self.model]
            nodes = {n["id"]: n for n in state["nodes"]}
            if all(any(nodes[n].get("active") and self.model + ":" + s["id"] in nodes[n]["ready"]
                       for n in d["placements"].get(s["id"], []) if n in nodes) for s in m["stages"]):
                return state
            errors = {n["id"]: n.get("metrics", {}).get("errors") for n in state["nodes"] if n.get("metrics", {}).get("errors")}
            if errors:
                raise RuntimeError(json.dumps(errors, indent=2))
            time.sleep(.5)
        raise TimeoutError(f"Incomplete route; inspect {self.work}")

    def kill_node(self, node):
        p = self.node_processes[node]
        p.kill()
        p.wait(timeout=10)

    def close(self):
        for p in reversed(self.processes):
            if p.poll() is None:
                p.terminate()
        for p in self.processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in self.logs:
            log.close()
        self.http.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()

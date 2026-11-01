"""Prove direct local/HTTP delivery and executor fallback in isolated Docker networks."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx

from scripts.tiny_checkpoint import make
from stageplane.distribution import distribute

TOKEN = "local-artifact-validation"


def run(image, output, port=8082, keep=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="stageplane-delivery-"))
    project = "stageplane-delivery-" + uuid.uuid4().hex[:8]
    compose = work / "compose.json"
    command = ["docker", "compose", "-p", project, "-f", str(compose)]
    evidence = {"image": image, "project": project, "fixture": "tiny offline Qwen3; not a real-model benchmark", "phases": []}

    def docker(*args, capture=False):
        return subprocess.run([*command, *args], check=True, text=True,
                              stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.STDOUT if capture else None)

    def wait_for(client, predicate, label, seconds=180):
        deadline = time.monotonic() + seconds
        last = None
        while time.monotonic() < deadline:
            try:
                last = client.get("/api/state").raise_for_status().json()
                if predicate(last):
                    return last
            except (httpx.HTTPError, KeyError):
                pass
            time.sleep(.5)
        raise AssertionError(f"Timed out waiting for {label}; last state: {last}")

    def state_ready(state):
        return (len(state["nodes"]) == 3 and state["deployments"].get(model, {}).get("capacity", {}).get("readiness") == "ready")

    def infer(client, stream=False):
        body = {"model": model, "messages": [{"role": "user", "content": "hello world"}],
                "temperature": 0, "max_tokens": 4, "stream": stream}
        if not stream:
            response = client.post("/v1/chat/completions", json=body).raise_for_status().json()
            return response["choices"][0]["message"]["content"], response["id"]
        answer, request_id, finished = "", None, False
        with client.stream("POST", "/v1/chat/completions", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    finished = True
                    break
                item = json.loads(line[6:])
                assert "error" not in item, item
                request_id = item["id"]
                answer += item["choices"][0]["delta"].get("content", "") or ""
        assert finished and request_id
        return answer, request_id

    def snapshot(client, phase, request_id):
        state = wait_for(client, lambda s: state_ready(s) and s["summary"]["active_requests"] == 0
                         and all(n["metrics"].get("sessions", 0) == 0 for n in s["nodes"]), "clean ready pool")
        record = next(r for r in state["requests"] if r["id"] == request_id)
        assert record["status"] == "completed"
        assert {hop["node"] for hop in record["route"]} == {"node-local", "node-http", "node-relay"}
        assert {entry["node"] for entry in record["last_trace"]} == {"node-local", "node-http", "node-relay"}
        metrics = {n["id"]: n["metrics"]["artifact_transfers"] for n in state["nodes"]}
        evidence["phases"].append({"phase": phase, "transfers": metrics, "request": record})
        print(json.dumps({"phase": phase, "transfers": metrics}), flush=True)
        return metrics

    try:
        evidence["image_id"] = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
        make(work / "checkpoint")
        manifest = distribute(str(work / "checkpoint"), work / "models", layers_per_chunk=1)
        model = manifest["key"]
        evidence.update(model=model, manifest_sha256=manifest["manifest_sha256"])
        (work / "origin.py").write_text('''import http.server,json
class Handler(http.server.SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory="/models",**kwargs)
 def do_GET(self):
  print(json.dumps({"artifact_request":self.path,"authorization_present":"Authorization" in self.headers,"cookie_present":"Cookie" in self.headers}),flush=True)
  super().do_GET()
http.server.ThreadingHTTPServer(("0.0.0.0",8000),Handler).serve_forever()
''')
        services = {
            "central": {"image": image, "entrypoint": ["python", "-u", "/origin.py"], "command": [],
                        "volumes": [f"{work}/models:/models:ro", f"{work}/origin.py:/origin.py:ro"],
                        "networks": ["origin"], "mem_limit": "128m"},
            "executor": {"image": image, "command": ["executor", "--host", "0.0.0.0", "--store", "/executor-store",
                         "--artifact-source", "/shared-models", "--artifact-source", "http://central:8000"],
                         "environment": {"STAGEPLANE_TOKEN": TOKEN, "STAGEPLANE_STATE_FILE": "/state/executor.sqlite"},
                         "volumes": [f"{work}/models:/executor-store:ro", "state:/state"],
                         "ports": [f"127.0.0.1:{port}:8080"], "networks": ["control"], "mem_limit": "1g", "cpus": 1},
        }
        volumes = {"state": {}}
        for group, (name, mode) in enumerate((("node-local", "rest"), ("node-http", "grpc"), ("node-relay", "websocket"))):
            services[name] = {"image": image, "command": ["node", "--id", name, "--executor", "http://executor:8080",
                              "--host", "0.0.0.0", "--advertise-url", f"http://{name}:8101", "--grpc-port", "9101",
                              "--grpc-url", f"{name}:9101", "--transport", mode, "--group", str(group), "--cache", "/cache", "--threads", "1"],
                              "environment": {"STAGEPLANE_TOKEN": TOKEN}, "volumes": [name + "-cache:/cache"],
                              "networks": ["control", "origin"] if name == "node-http" else ["control"], "mem_limit": "1g", "cpus": 1}
            volumes[name + "-cache"] = {}
        services["node-local"]["volumes"].append(f"{work}/models:/shared-models:ro")
        config = {"services": services, "volumes": volumes, "networks": {"control": {}, "origin": {"internal": True}}}
        compose.write_text(json.dumps(config, indent=2))
        docker("up", "-d")
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": "Bearer " + TOKEN},
                          timeout=120, trust_env=False) as client:
            wait_for(client, lambda s: len(s["nodes"]) == 3, "three registered workers")
            client.post("/api/deployments", json={"model": model, "replicas": 1, "groups": 3}).raise_for_status()
            wait_for(client, state_ready, "initial artifact delivery")
            answer, request_id = infer(client)
            metrics = snapshot(client, "local-http-relay", request_id)
            assert metrics["node-local"]["direct_local"] > 0 and metrics["node-local"]["executor"] == 0
            assert metrics["node-http"]["direct_http"] > 0 and metrics["node-http"]["executor"] == 0
            assert metrics["node-relay"]["executor"] > 0 and metrics["node-relay"]["fallbacks"] > 0
            origin_log = docker("logs", "--no-log-prefix", "central", capture=True).stdout
            requests = [json.loads(line) for line in origin_log.splitlines() if line.startswith('{"artifact_request"')]
            assert requests and not any(r["authorization_present"] or r["cookie_present"] for r in requests)
            evidence["origin_requests"] = requests

            # The HTTP worker loses its origin and its old cache, but the executor still has the files.
            docker("stop", "central")
            volumes["http-cold-cache"] = {}
            services["node-http"]["volumes"] = ["http-cold-cache:/cache"]
            compose.write_text(json.dumps(config, indent=2))
            docker("up", "-d", "--no-deps", "node-http")
            wait_for(client, lambda s: state_ready(s) and any(n["id"] == "node-http" and
                     n["metrics"].get("artifact_transfers", {}).get("executor", 0) > 0 for n in s["nodes"]), "cold-cache origin failure fallback")
            streamed, request_id = infer(client, stream=True)
            assert streamed == answer
            metrics = snapshot(client, "origin-offline-cold-cache", request_id)
            assert metrics["node-http"]["direct_http"] == 0 and metrics["node-http"]["fallbacks"] > 0

            # A verified cache remains sufficient even with both direct sources unavailable.
            services["node-local"]["volumes"] = ["node-local-cache:/cache"]
            compose.write_text(json.dumps(config, indent=2))
            docker("up", "-d", "--no-deps", "node-local")
            wait_for(client, lambda s: state_ready(s) and any(n["id"] == "node-local" and
                     n["metrics"].get("artifact_transfers", {}).get("cache_hits", 0) > 0 for n in s["nodes"]), "verified cache reuse")
            cached, request_id = infer(client)
            assert cached == answer
            metrics = snapshot(client, "cache-without-origin", request_id)
            assert metrics["node-local"]["direct_local"] == metrics["node-local"]["executor"] == 0
            evidence.update(result="passed", output=answer, url=str(client.base_url))
    finally:
        (output / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
        if compose.exists():
            logs = subprocess.run([*command, "logs", "--no-color"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (output / "services.log").write_text(logs.stdout)
            if not keep:
                subprocess.run([*command, "down", "--volumes", "--remove-orphans"], check=True)
        if keep:
            print(f"Kept local fixture at http://127.0.0.1:{port}; token={TOKEN}; compose={compose}", flush=True)
        else:
            shutil.rmtree(work)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="stageplane:local")
    parser.add_argument("--output", type=Path, default=Path("reports/artifact-delivery"))
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--keep", action="store_true", help="Keep owned containers and fixture files after validation")
    args = parser.parse_args()
    run(args.image, args.output, args.port, args.keep)

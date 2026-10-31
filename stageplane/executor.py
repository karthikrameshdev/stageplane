from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import torch
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer

from .common import MAX_CONTEXT, ROOT, ChatRequest, DeployRequest, ImportRequest, NodeInfo, authorize, safe_path
from .distribution import distribute, manifests
from .bundles import export_bundle, import_bundle
from .placement import plan, route
from .wire import ChunkFailure, Transport, pack, unpack
from .resources import capacity, weight_budget


class Coordinator:
    def __init__(self, store: Path = ROOT, state_file: Path | None = None):
        self.store = store.resolve()
        self.models = manifests(self.store)
        self.nodes, self.deployments, self.tokenizers = {}, {}, {}
        self.events = deque(maxlen=400)
        self.requests = deque(maxlen=100)
        self.clients = set()
        self.jobs = {}
        self.import_tasks = set()
        self.bundle_locks = {}
        self.last_rebalance = 0
        self.retry_limit = int(os.environ.get("STAGEPLANE_EXECUTION_RETRIES", "3"))
        self.transport = Transport()
        self.admission = asyncio.Semaphore(int(os.environ.get("STAGEPLANE_MAX_REQUESTS", "4")))
        state_file = state_file or Path(os.environ.get("STAGEPLANE_STATE_FILE", str(self.store / "executor.sqlite")))
        state_file.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(state_file)
        self.db.execute("CREATE TABLE IF NOT EXISTS deployments(model TEXT PRIMARY KEY, body TEXT)")
        for model, body in self.db.execute("SELECT model, body FROM deployments"):
            if model in self.models:
                self.deployments[model] = json.loads(body)

    def event(self, kind, **data):
        item = {"id": str(uuid.uuid4()), "kind": kind, "time": time.time(), **data}
        self.events.append(item)
        for queue in self.clients:
            if not queue.full():
                queue.put_nowait(item)

    def resolve(self, name):
        if name in self.models:
            return name
        matches = [k for k, m in self.models.items() if m["source"] == name or name in m.get("aliases", [])]
        if len(matches) == 1:
            return matches[0]
        raise HTTPException(404, "Unknown or ambiguous model; use the immutable model key from /v1/models")

    def persist(self, model):
        self.db.execute("INSERT OR REPLACE INTO deployments VALUES (?, ?)", (model, json.dumps(self.deployments[model])))
        self.db.commit()

    def rebalance(self, changed_groups=None):
        self.last_rebalance = time.monotonic()
        for model, d in self.deployments.items():
            if model not in self.models:
                continue
            before = d.get("placements", {})
            after = plan(self.models[model], list(self.nodes.values()), d["replicas"], d["groups"], before, changed_groups)
            status = capacity(self.models[model], after, list(self.nodes.values()), d["groups"])
            old_capacity = d.get("capacity", {})
            if status["status"] == "insufficient_capacity" and old_capacity.get("message") != status["message"]:
                print(status["message"], flush=True)
                self.event("insufficient_capacity", model=model, **status)
            d["capacity"] = status
            if before != after:
                d["placements"] = after
                self.persist(model)
                changed = [sid for sid in after if after[sid] != before.get(sid)]
                self.event("placement", model=model, chunks=changed, placements=after)
            elif old_capacity != status:
                self.persist(model)

    def register(self, info: NodeInfo):
        if not info.url.startswith(("http://", "https://")):
            raise HTTPException(422, "Node URL must be HTTP(S)")
        old = self.nodes.get(info.id)
        quarantined = old.get("quarantined_until", 0) if old else 0
        node = {**info.model_dump(), "last_seen": time.time(), "active": time.time() >= quarantined,
                "quarantined_until": quarantined,
                "draining": old.get("draining", False) if old else False}
        self.nodes[info.id] = node
        if node["active"] and (old is None or not old["active"]):
            self.event("node_joined", node=info.id, group=info.group)
            self.rebalance({info.group})
        elif time.monotonic() - self.last_rebalance > 30 and not node["metrics"].get("sessions") and not node["metrics"].get("loading"):
            old_budget, budget = weight_budget(old), weight_budget(node)
            changed = abs(budget - old_budget) > max(64 * 1024**2, old_budget * .25)
            old_cpu = old.get("metrics", {}).get("available_cpu_cores", 1)
            changed |= node["metrics"].get("available_cpu_cores", 1) < old_cpu * .25
            if changed or any(d.get("capacity", {}).get("status") == "insufficient_capacity" for d in self.deployments.values()):
                self.rebalance({info.group})
        return {"assignments": self.assignments(info.id)}

    def assignments(self, nid):
        result = []
        for model, d in self.deployments.items():
            if model not in self.models:
                continue
            if d.get("capacity", {}).get("status") == "insufficient_capacity":
                continue
            for chunk, owners in d.get("placements", {}).items():
                if nid in owners:
                    result.append({"model": model, "chunk": chunk,
                                   "manifest_sha256": self.models[model]["manifest_sha256"]})
        return result

    async def watchdog(self):
        while True:
            changed = set()
            for node in self.nodes.values():
                if node["active"] and time.time() - node["last_seen"] > float(os.environ.get("STAGEPLANE_HEARTBEAT_TIMEOUT", "8")):
                    node["active"] = False
                    changed.add(node["group"])
                    self.event("node_lost", node=node["id"], group=node["group"])
            if changed:
                self.rebalance(changed)
            await asyncio.sleep(0.5)

    def state(self):
        return {"models": [{"key": k, "source": m["source"], "revision": m["revision"],
                            "architecture": m["architecture"], "stages": m["stages"],
                            "artifact_bytes": sum(a["bytes"] for a in m["artifacts"].values())}
                           for k, m in self.models.items()],
                "nodes": list(self.nodes.values()), "deployments": self.deployments,
                "events": list(self.events), "requests": list(self.requests), "time": time.time(),
                "transport": self.transport.metrics.copy()}

    def get_route(self, model, exclude=()):
        if model not in self.deployments:
            raise RuntimeError("Model has no deployment")
        if self.deployments[model].get("capacity", {}).get("status") == "insufficient_capacity":
            raise RuntimeError(self.deployments[model]["capacity"]["message"])
        return route(self.models[model], self.deployments[model].get("placements", {}),
                     {nid: n for nid, n in self.nodes.items() if nid not in exclude})

    async def ready_route(self, model, timeout=None, exclude=()):
        timeout = timeout if timeout is not None else float(os.environ.get("STAGEPLANE_RECOVERY_WAIT_SECONDS", "10"))
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.get_route(model, exclude)
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                await asyncio.sleep(0.25)

    async def cleanup(self, tenant, session, route_nodes):
        async def release(node):
            with contextlib.suppress(Exception):
                await self.transport.http.delete(node["url"] + f"/sessions/{tenant}/{session}", timeout=2)
        await asyncio.gather(*(release(n) for n in route_nodes.values()))

    async def probe_failed_route(self, path):
        async def probe(hop):
            try:
                r = await self.transport.http.get(hop["url"] + "/health", timeout=1)
                r.raise_for_status()
                return None
            except Exception:
                node = self.nodes.get(hop["node"])
                if node:
                    node["active"] = False
                    self.event("node_lost", node=node["id"], group=node["group"])
                    return node["group"]
        hops = {h["node"]: h for h in path}
        groups = {g for g in await asyncio.gather(*(probe(h) for h in hops.values())) if g is not None}
        if groups:
            self.rebalance(groups)

    async def generate(self, request: ChatRequest):
        model = self.resolve(request.model)
        manifest = self.models[model]
        if model not in self.tokenizers:
            self.tokenizers[model] = await asyncio.to_thread(AutoTokenizer.from_pretrained, self.store / model,
                                                          trust_remote_code=False, local_files_only=True)
        tokenizer = self.tokenizers[model]
        messages = [m.model_dump() for m in request.messages]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False, return_dict=False)
        if len(tokens) + request.max_tokens > MAX_CONTEXT:
            raise HTTPException(422, f"Prompt plus output exceeds {MAX_CONTEXT} token context budget")
        session = "chatcmpl-" + uuid.uuid4().hex
        tenant = "default"
        used_nodes, generated, epoch = {}, [], 0
        record = {"id": session, "model": model, "status": "running", "started": time.time(),
                  "prompt_tokens": len(tokens), "output_tokens": 0, "recoveries": 0, "token_ms": [],
                  "boundary_bytes": 0}
        self.requests.append(record)
        generator = torch.Generator().manual_seed(request.seed)
        previous_text = ""
        started = time.perf_counter()
        route_path = self.get_route(model)
        self.event("request_started", request=session, model=model, route=route_path, prompt_tokens=len(tokens))
        offset, input_tokens = 0, tokens[:]
        failures = 0
        failed_nodes = set()
        try:
            for step in range(request.max_tokens):
                token_start = time.perf_counter()
                while True:
                    used_nodes.update({h["node"]: h for h in route_path})
                    meta = {"tenant": tenant, "session": session, "model": model, "revision": manifest["revision"],
                            "manifest_sha256": manifest["manifest_sha256"], "epoch": epoch, "offset": offset,
                            "context_budget": len(tokens) + request.max_tokens,
                            "lease_until": time.time() + 120, "route": route_path, "index": 0, "trace": []}
                    try:
                        frame = pack(meta, torch.tensor([input_tokens], dtype=torch.int64))
                        data = await self.transport.send(route_path[0], frame)
                        result, logits = unpack(data)
                        if result["epoch"] != epoch or result["session"] != session or result["index"] != len(route_path):
                            raise ValueError("Inconsistent route result")
                        break
                    except ChunkFailure as error:
                        # Prefix replay restores attention AND recurrent state. No output
                        # is committed until this entire forward pass has succeeded.
                        failures += 1
                        if failures > self.retry_limit:
                            raise RuntimeError(f"Recovery exhausted: {error}") from error
                        epoch += 1
                        record["recoveries"] += 1
                        self.event("recovery", request=session, epoch=epoch, committed=len(generated), reason=error.code,
                                   chunk=error.chunk, node=error.node, attempt=failures, retry_limit=self.retry_limit)
                        if error.node in self.nodes:
                            failed_nodes.add(error.node)
                            node = self.nodes[error.node]
                            node["active"] = False
                            node["quarantined_until"] = time.time() + 30
                            self.rebalance({node["group"]})
                        else:
                            await self.probe_failed_route(route_path)
                        # A ready equivalent replica is selected without delay;
                        # prefix replay supplies its missing session state.
                        route_path = await self.ready_route(model, exclude=failed_nodes)
                        self.event("chunk_retry", request=session, epoch=epoch, chunk=error.chunk,
                                   previous_node=error.node, attempt=failures, route=route_path)
                        offset, input_tokens = 0, tokens + generated
                score = logits[0, -1].float()
                if request.temperature == 0:
                    token = int(torch.argmax(score))
                else:
                    probs = torch.softmax(score / request.temperature, dim=-1)
                    if request.top_p < 1:
                        ordered, indices = probs.sort(descending=True)
                        remove = ordered.cumsum(0) - ordered >= request.top_p
                        ordered[remove] = 0
                        probs = torch.zeros_like(probs).scatter(0, indices, ordered)
                        probs /= probs.sum()
                    token = int(torch.multinomial(probs, 1, generator=generator))
                generated.append(token)
                latency = (time.perf_counter() - token_start) * 1000
                record["token_ms"].append(latency)
                if step == 0:
                    record["ttft_ms"] = (time.perf_counter() - started) * 1000
                record["output_tokens"] = len(generated)
                trace = result["trace"]
                hidden_size = manifest["config"].get("text_config", manifest["config"])["hidden_size"]
                record["boundary_bytes"] += sum(input_tokens.__len__() * hidden_size * 4 for a, b in zip(route_path, route_path[1:]) if a["node"] != b["node"])
                self.event("token_committed", request=session, epoch=epoch, offset=step, token_id=token,
                           ms=latency, trace=trace)
                text = tokenizer.decode(generated, skip_special_tokens=True)
                # Avoid emitting a partial UTF-8 replacement character.
                if not text.endswith("\ufffd"):
                    delta = text[len(previous_text):]
                    previous_text = text
                else:
                    delta = ""
                yield {"id": session, "delta": delta, "token_id": token, "step": step,
                       "model": model, "trace": trace, "finish_reason": None}
                eos = tokenizer.eos_token_id
                if token == eos:
                    break
                offset += len(input_tokens)
                input_tokens = [token]
            record["status"] = "completed"
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            record["tokens_per_second"] = len(generated) / (record["elapsed_ms"] / 1000)
            yield {"id": session, "delta": "", "model": model,
                   "finish_reason": "stop" if generated[-1] == tokenizer.eos_token_id else "length",
                   "usage": {"prompt_tokens": len(tokens), "completion_tokens": len(generated),
                             "total_tokens": len(tokens) + len(generated)}}
            self.event("request_completed", request=session, metrics=record.copy())
        except BaseException:
            record["status"] = "cancelled" if asyncio.current_task().cancelling() else "failed"
            self.event("request_" + record["status"], request=session)
            raise
        finally:
            await self.cleanup(tenant, session, used_nodes)


def create_app(coordinator: Coordinator):
    c = coordinator

    @asynccontextmanager
    async def lifespan(app):
        watcher = asyncio.create_task(c.watchdog())
        yield
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        await c.transport.close()
        c.db.close()

    app = FastAPI(title="Stageplane executor", version="0.1.0", lifespan=lifespan)
    protected = [Depends(authorize)]

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/state", dependencies=protected)
    async def state():
        return c.state()

    @app.get("/v1/models", dependencies=protected)
    async def models():
        return {"object": "list", "data": [{"id": k, "object": "model", "created": int(m["created"]),
                 "owned_by": "stageplane", "source": m["source"]} for k, m in c.models.items()]}

    @app.post("/api/models/import", dependencies=protected, status_code=202)
    async def import_model(body: ImportRequest):
        if any(job["status"] == "running" for job in c.jobs.values()):
            raise HTTPException(409, "Another import is running")
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "running", "source": body.source, "started": time.time()}
        c.jobs[job_id] = job

        async def run():
            try:
                if body.kind == "bundle":
                    m = await asyncio.to_thread(import_bundle, body.source, c.store, body.manifest_sha256,
                                                body.revision, body.subdir)
                else:
                    m = await asyncio.to_thread(distribute, body.source, c.store, body.revision, body.layers_per_chunk)
                c.models[m["key"]] = m
                job.update(status="completed", model=m["key"], finished=time.time())
                c.event("model_imported", model=m["key"], chunks=len(m["stages"]))
            except Exception as e:
                job.update(status="failed", error=str(e), finished=time.time())
                c.event("model_import_failed", job=job_id, error=type(e).__name__)
        task = asyncio.create_task(run())
        c.import_tasks.add(task)
        task.add_done_callback(c.import_tasks.discard)
        return job

    @app.get("/api/jobs/{job_id}", dependencies=protected)
    async def get_job(job_id: str):
        if job_id not in c.jobs:
            raise HTTPException(404, "Unknown import job")
        return c.jobs[job_id]

    @app.get("/api/bundles/{model}", dependencies=protected)
    async def download_bundle(model: str):
        model = c.resolve(model)
        path = c.store / ".exports" / (c.models[model]["manifest_sha256"] + ".stageplane")
        async with c.bundle_locks.setdefault(model, asyncio.Lock()):
            if not path.exists():
                await asyncio.to_thread(export_bundle, c.store / model, path)
        return FileResponse(path, media_type="application/zip", filename=model + ".stageplane")

    @app.post("/api/deployments", dependencies=protected)
    async def deploy(body: DeployRequest):
        model = c.resolve(body.model)
        c.deployments[model] = {"replicas": body.replicas, "groups": body.groups, "placements": {}}
        c.rebalance()
        c.persist(model)
        if c.deployments[model]["capacity"]["status"] == "insufficient_capacity":
            raise HTTPException(503, c.deployments[model]["capacity"]["message"])
        return c.deployments[model]

    @app.post("/api/nodes/register", dependencies=protected)
    async def register(body: NodeInfo):
        return c.register(body)

    @app.post("/api/nodes/{node_id}/heartbeat", dependencies=protected)
    async def heartbeat(node_id: str, body: NodeInfo):
        if node_id != body.id:
            raise HTTPException(422, "Node identity mismatch")
        if node_id not in c.nodes:
            raise HTTPException(404, "Register first")
        return c.register(body)

    @app.post("/api/nodes/{node_id}/drain", dependencies=protected)
    async def drain(node_id: str):
        if node_id not in c.nodes:
            raise HTTPException(404, "Unknown node")
        c.nodes[node_id]["draining"] = True
        c.rebalance({c.nodes[node_id]["group"]})
        return {"draining": True}

    @app.get("/api/manifests/{model}", dependencies=protected)
    async def manifest(model: str):
        return c.models[c.resolve(model)]

    @app.get("/api/artifacts/{model}/{name}", dependencies=protected)
    async def artifact(model: str, name: str):
        model = c.resolve(model)
        if name not in c.models[model]["files"]:
            raise HTTPException(404, "Unknown artifact")
        return FileResponse(safe_path(c.store / model, name), media_type="application/octet-stream")

    @app.post("/v1/chat/completions", dependencies=protected)
    async def chat(body: ChatRequest, request: Request):
        model = c.resolve(body.model)
        try:
            c.get_route(model)
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e
        try:
            await asyncio.wait_for(c.admission.acquire(), timeout=0.1)
        except TimeoutError as e:
            raise HTTPException(429, "Inference admission capacity exhausted") from e

        async def chunks():
            try:
                async for item in c.generate(body):
                    if await request.is_disconnected():
                        break
                    delta = {"content": item["delta"]}
                    payload = {"id": item["id"], "object": "chat.completion.chunk", "created": int(time.time()),
                               "model": item["model"], "choices": [{"index": 0, "delta": delta,
                               "finish_reason": item["finish_reason"]}]}
                    if "usage" in item:
                        payload["usage"] = item["usage"]
                    yield "data: " + json.dumps(payload) + "\n\n"
            except Exception as e:
                yield "data: " + json.dumps({"error": {"message": str(e), "type": "inference_error"}}) + "\n\n"
            finally:
                c.admission.release()
            yield "data: [DONE]\n\n"

        if body.stream:
            return StreamingResponse(chunks(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        content, final = "", None
        try:
            async for item in c.generate(body):
                content += item["delta"]
                final = item
            return {"id": final["id"], "object": "chat.completion", "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": final["finish_reason"]}], "usage": final["usage"]}
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e
        finally:
            c.admission.release()

    @app.websocket("/ws/events")
    async def events(ws: WebSocket):
        await ws.accept()
        try:
            hello = await asyncio.wait_for(ws.receive_json(), timeout=5)
            if os.environ.get("STAGEPLANE_TOKEN") and hello.get("token") != os.environ["STAGEPLANE_TOKEN"]:
                await ws.close(1008)
                return
            queue = asyncio.Queue(maxsize=64)
            c.clients.add(queue)
            await ws.send_json({"kind": "snapshot", "state": c.state()})
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=2)
                        await ws.send_json(event)
                    except TimeoutError:
                        await ws.send_json({"kind": "snapshot", "state": c.state()})
            finally:
                c.clients.discard(queue)
        except Exception:
            pass

    ui = Path(__file__).parent / "ui"
    app.mount("/ui", StaticFiles(directory=ui), name="ui")

    @app.get("/")
    async def index():
        return FileResponse(ui / "index.html")

    return app

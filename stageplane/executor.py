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
from .placement import bucket, plan, route
from .wire import ChunkFailure, Transport, pack, unpack
from .resources import capacity, weight_budget, weight_factor
from .security import AccessPolicy, RequestBodyLimit


class Coordinator:
    def __init__(self, store: Path = ROOT, state_file: Path | None = None):
        self.store = store.resolve()
        self.models = manifests(self.store)
        self.nodes, self.deployments, self.tokenizers = {}, {}, {}
        self.events = deque(maxlen=400)
        self.requests = deque(maxlen=100)
        self.active_requests = {}
        self.request_totals = {"completed": 0, "failed": 0, "cancelled": 0}
        self.started = time.time()
        self.access = AccessPolicy()
        self.clients = set()
        self.jobs = {}
        self.import_tasks = set()
        self.cleanup_tasks = set()
        self.bundle_locks = {}
        self.assignment_index = {}
        self.last_rebalance = 0
        self.group_rebalances = {}
        self.pending_rebalance_groups = set()
        self.resource_baselines = {}
        self.rebalance_count = 0
        self.retry_limit = int(os.environ.get("STAGEPLANE_EXECUTION_RETRIES", "3"))
        self.transport = Transport()
        self.admission = asyncio.Semaphore(int(os.environ.get("STAGEPLANE_MAX_REQUESTS", "4")))
        state_file = state_file or Path(os.environ.get("STAGEPLANE_STATE_FILE", str(self.store / "executor.sqlite")))
        state_file.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(state_file)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
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

    def _reservations(self, exclude):
        memory, storage = {}, {}
        ready = {nid: set(node.get("ready", [])) for nid, node in self.nodes.items()}
        for model, deployment in self.deployments.items():
            if model == exclude or model not in self.models:
                continue
            for stage in self.models[model]["stages"]:
                for nid in self.effective_placements(deployment).get(stage["id"], []):
                    node = self.nodes.get(nid)
                    if node is None or not node.get("active"):
                        continue
                    if deployment.get("capacity", {}).get("status") == "insufficient_capacity" and model + ":" + stage["id"] not in ready[nid]:
                        continue
                    memory[nid] = memory.get(nid, 0) + stage["weight_bytes"] * weight_factor(node)
                    if model + ":" + stage["id"] not in ready[nid]:
                        storage[nid] = storage.get(nid, 0) + stage["weight_bytes"]
        return memory, storage

    @staticmethod
    def effective_placements(deployment):
        placements = {sid: list(owners) for sid, owners in deployment.get("placements", {}).items()}
        for sid, owners in deployment.get("retiring_placements", {}).items():
            placements[sid] = list(dict.fromkeys([*placements.get(sid, []), *owners]))
        return placements

    def rebuild_assignments(self):
        self.assignment_index = {}
        for model, deployment in self.deployments.items():
            if model not in self.models:
                continue
            for chunk, owners in self.effective_placements(deployment).items():
                assignment = {"model": model, "chunk": chunk, "manifest_sha256": self.models[model]["manifest_sha256"]}
                for nid in owners:
                    if deployment.get("capacity", {}).get("status") == "insufficient_capacity" and model + ":" + chunk not in self.nodes.get(nid, {}).get("ready", []):
                        continue
                    self.assignment_index.setdefault(nid, []).append(assignment)

    def settle_transitions(self):
        changed = False
        ready = {nid: set(node.get("ready", [])) for nid, node in self.nodes.items() if node.get("active")}
        pinned = {(request["model"], hop["chunk"], hop["node"])
                  for request in self.active_requests.values() for hop in request.get("route", [])
                  if "model" in request and "chunk" in hop}
        for model, deployment in self.deployments.items():
            transitions = deployment.get("retiring_placements", {})
            remaining = {}
            for sid, owners in transitions.items():
                targets = deployment.get("placements", {}).get(sid, [])
                prepared = bool(targets) and all(model + ":" + sid in ready.get(nid, set()) for nid in targets)
                keep = [nid for nid in owners if nid in ready and (
                    not prepared or (model, sid, nid) in pinned or self.nodes[nid].get("metrics", {}).get("sessions", 0))]
                if keep:
                    remaining[sid] = keep
            if remaining != transitions:
                deployment["retiring_placements"] = remaining
                self.persist(model)
                self.event("placement_settled", model=model, retiring_placements=remaining)
                changed = True
        if changed:
            self.rebuild_assignments()

    def deployment_status(self, model, deployment, ready_sets=None):
        status = dict(deployment.get("capacity", {}))
        stages = self.models[model]["stages"]
        placements = self.effective_placements(deployment)
        if ready_sets is None:
            ready_sets = {nid: set(node.get("ready", [])) for nid, node in self.nodes.items()
                          if node.get("active") and not node.get("draining")}
        ready_counts = {stage["id"]: sum(model + ":" + stage["id"] in ready_sets.get(nid, ())
                                        for nid in set(placements.get(stage["id"], []))) for stage in stages}
        ready = sum(count > 0 for count in ready_counts.values())
        target_ready = 0
        for stage in stages:
            targets = deployment.get("placements", {}).get(stage["id"], [])
            if targets and all(model + ":" + stage["id"] in ready_sets.get(nid, ()) for nid in targets):
                target_ready += 1
        under_replicated = [sid for sid, count in ready_counts.items() if count < deployment["replicas"]]
        status.update(replica_target=deployment["replicas"], under_replicated_stages=under_replicated,
                      ready_stages=ready, total_stages=len(stages), loading_stages=len(stages) - target_ready,
                      target_ready_stages=target_ready,
                      transitioning_stages=list(deployment.get("retiring_placements", {})))
        status["serving"] = ready == len(stages)
        status["readiness"] = ("blocked" if status.get("status") == "insufficient_capacity" and ready < len(stages) else
                               "rebalancing" if status.get("status") == "insufficient_capacity" else
                               "loading" if ready < len(stages) else
                               "rebalancing" if deployment.get("retiring_placements") else
                               "degraded" if under_replicated else "ready")
        return status

    def rebalance(self, changed_groups=None):
        self.last_rebalance = time.monotonic()
        self.rebalance_count += 1
        for model, d in self.deployments.items():
            if model not in self.models:
                continue
            before = d.get("placements", {})
            reserved, storage_reserved = self._reservations(model)
            resident = {}
            old_effective = self.effective_placements(d)
            ready = {nid: set(node.get("ready", [])) for nid, node in self.nodes.items() if node.get("active")}
            for sid, owners in old_effective.items():
                for nid in owners:
                    if model + ":" + sid in ready.get(nid, set()):
                        resident.setdefault(nid, set()).add(sid)
            after = plan(self.models[model], list(self.nodes.values()), d["replicas"], d["groups"], before,
                         changed_groups, reserved=reserved, storage_reserved=storage_reserved, resident=resident)
            status = capacity(self.models[model], after, list(self.nodes.values()), d["groups"])
            status["available_weight_budget_bytes"] = max(0, status["available_weight_budget_bytes"] - sum(reserved.values()))
            old_capacity = d.get("capacity", {})
            if status["status"] == "insufficient_capacity" and old_capacity.get("message") != status["message"]:
                print(status["message"], flush=True)
                self.event("insufficient_capacity", model=model, **status)
            d["capacity"] = status
            old_retiring = d.get("retiring_placements", {})
            retiring = {sid: [nid for nid in owners if nid not in after.get(sid, [])
                             and model + ":" + sid in ready.get(nid, set())]
                        for sid, owners in old_effective.items()}
            d["retiring_placements"] = {sid: owners for sid, owners in retiring.items() if owners}
            if before != after:
                d["placements"] = after
                self.persist(model)
                changed = [sid for sid in after if after[sid] != before.get(sid)]
                self.event("placement", model=model, chunks=changed, placements=after)
            elif old_capacity != status or old_retiring != d["retiring_placements"]:
                self.persist(model)
            changed = range(d["groups"]) if changed_groups is None else {g % d["groups"] for g in changed_groups}
            for group in changed:
                self.group_rebalances[(d["groups"], group)] = self.last_rebalance
        counts = {d["groups"] for d in self.deployments.values()} or {1}
        for nid, node in self.nodes.items():
            if changed_groups is None or any(node["group"] % count in {g % count for g in changed_groups} for count in counts):
                self.resource_baselines[nid] = self._resources(node)
        self.rebuild_assignments()
        self.settle_transitions()
        if changed_groups:
            for group in changed_groups:
                if any(node["active"] and node.get("metrics", {}).get("loading")
                       and any(node["group"] % count == group % count for count in counts)
                       for node in self.nodes.values()):
                    self.pending_rebalance_groups.add(group)

    def flush_pending_rebalances(self):
        counts = {d["groups"] for d in self.deployments.values()} or {1}
        settled = {group for group in self.pending_rebalance_groups
                   if not any(node["active"] and node.get("metrics", {}).get("loading")
                              and any(node["group"] % count == group % count for count in counts)
                              for node in self.nodes.values())}
        if settled:
            self.pending_rebalance_groups.difference_update(settled)
            self.rebalance(settled)

    @staticmethod
    def _resources(node):
        metrics = node.get("metrics", {})
        return (weight_budget(node), metrics.get("available_cpu_cores", 1),
                metrics.get("available_storage_bytes", 2**63))

    def register(self, info: NodeInfo):
        if not info.url.startswith(("http://", "https://")):
            raise HTTPException(422, "Node URL must be HTTP(S)")
        try:
            json.dumps(info.model_dump(), allow_nan=False)
            for key in ("available_memory_bytes", "memory_limit_bytes", "memory_used_bytes", "loaded_weight_bytes",
                        "available_storage_bytes", "rss_bytes", "available_cpu_cores", "sessions", "max_sessions"):
                value = info.metrics.get(key)
                if key in info.metrics and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
                    raise ValueError("Invalid numeric telemetry")
            timings = info.metrics.get("stage_ms", {})
            if not isinstance(timings, dict) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in timings.values()):
                raise ValueError("Invalid stage timing telemetry")
        except (ValueError, TypeError) as error:
            raise HTTPException(422, "Node telemetry must contain finite, nonnegative numeric resource values") from error
        old = self.nodes.get(info.id)
        quarantined = old.get("quarantined_until", 0) if old else 0
        node = {**info.model_dump(), "last_seen": time.time(), "active": time.time() >= quarantined,
                "quarantined_until": quarantined,
                "draining": old.get("draining", False) if old else False}
        self.nodes[info.id] = node
        changed_identity = old is not None and any(old.get(field) != node.get(field)
            for field in ("group", "url", "grpc_url", "transport", "failure_domain"))
        if node["active"] and (old is None or not old["active"] or changed_identity):
            self.event("node_joined" if old is None or not old["active"] else "node_updated", node=info.id, group=info.group)
            self.rebalance({info.group} | ({old["group"]} if old else set()))
        elif old is not None and node["active"] and not node["metrics"].get("sessions") and not node["metrics"].get("loading"):
            counts = {d["groups"] for d in self.deployments.values()} or {1}
            last = max(self.group_rebalances.get((count, info.group % count), self.last_rebalance) for count in counts)
            interval = float(os.environ.get("STAGEPLANE_REBALANCE_INTERVAL", "30"))
            baseline = self.resource_baselines.get(info.id, self._resources(old))
            current = self._resources(node)
            changed = any(abs(after - before) > max(floor, abs(before) * .25)
                          for before, after, floor in zip(baseline, current, (64 * 1024**2, .01, 64 * 1024**2)))
            incomplete = any(d.get("capacity", {}).get("status") == "insufficient_capacity"
                             for d in self.deployments.values())
            loading_peer = any(peer["active"] and peer["metrics"].get("loading")
                               and any(peer["group"] % count == info.group % count for count in counts)
                               for peer in self.nodes.values())
            if time.monotonic() - last >= interval and (changed or incomplete) and not loading_peer:
                self.rebalance({info.group})
        self.settle_transitions()
        self.flush_pending_rebalances()
        return {"assignments": self.assignments(info.id)}

    def assignments(self, nid):
        # Heartbeats scale with that worker's assignments, not the entire cluster graph.
        return [dict(item) for item in self.assignment_index.get(nid, [])]

    def sweep_nodes(self, now=None):
        now = time.time() if now is None else now
        timeout = float(os.environ.get("STAGEPLANE_HEARTBEAT_TIMEOUT", "8"))
        retention = max(timeout, float(os.environ.get("STAGEPLANE_NODE_RETENTION_SECONDS", "3600")))
        referenced = {hop["node"] for request in self.active_requests.values() for hop in request.get("route", [])}
        changed = set()
        for nid, node in list(self.nodes.items()):
            age = now - node["last_seen"]
            if node["active"] and age > timeout:
                node["active"] = False
                changed.add(node["group"])
                self.event("node_lost", node=nid, group=node["group"])
            if not node["active"] and age > retention and nid not in referenced:
                del self.nodes[nid]
                self.resource_baselines.pop(nid, None)
                self.event("node_expired", node=nid, group=node["group"])
        if changed:
            self.rebalance(changed)

    async def watchdog(self):
        while True:
            self.sweep_nodes()
            self.flush_pending_rebalances()
            await asyncio.sleep(0.5)

    def state(self):
        now = time.time()
        requests = [*self.requests, *self.active_requests.values()]
        requests = [{**record, "elapsed_ms": (now - record["started"]) * 1000}
                    if record["status"] in ("running", "recovering") else dict(record) for record in requests]
        nodes = list(self.nodes.values())
        ready_sets = {node["id"]: set(node.get("ready", [])) for node in nodes
                      if node.get("active") and not node.get("draining")}
        deployments = {model: {**d, "capacity": self.deployment_status(model, d, ready_sets)}
                       for model, d in self.deployments.items() if model in self.models}
        return {"models": [{"key": k, "source": m["source"], "revision": m["revision"],
                            "architecture": m["architecture"], "stages": m["stages"],
                            "artifact_bytes": sum(a["bytes"] for a in m["artifacts"].values())}
                           for k, m in self.models.items()],
                "nodes": nodes, "deployments": deployments,
                "events": list(self.events), "requests": requests, "jobs": list(self.jobs.values()), "time": now,
                "summary": {"active_nodes": sum(n["active"] for n in nodes), "total_nodes": len(nodes),
                            "draining_nodes": sum(bool(n.get("draining")) for n in nodes),
                            "active_requests": len(self.active_requests),
                            **{key + "_requests": value for key, value in self.request_totals.items()},
                            "uptime_seconds": now - self.started, "rebalance_count": self.rebalance_count},
                "transport": self.transport.metrics.copy()}

    def get_route(self, model, exclude=(), reuse_nodes=()):
        if model not in self.deployments:
            raise RuntimeError("Model has no deployment")
        local_sessions = {}
        for request in self.active_requests.values():
            for nid in {hop["node"] for hop in request.get("route", [])}:
                local_sessions[nid] = local_sessions.get(nid, 0) + 1
        placements = self.effective_placements(self.deployments[model])
        owners = {nid for values in placements.values() for nid in values}
        candidates = {nid: {**n, "metrics": {**n.get("metrics", {}),
                      "sessions": max(0, n.get("metrics", {}).get("sessions", 0) - int(nid in reuse_nodes), local_sessions.get(nid, 0))}}
                      for nid in owners if nid not in exclude and (n := self.nodes.get(nid)) is not None}
        try:
            return route(self.models[model], placements, candidates)
        except RuntimeError as error:
            if self.deployments[model].get("capacity", {}).get("status") == "insufficient_capacity":
                raise RuntimeError(self.deployments[model]["capacity"]["message"]) from error
            raise

    async def ready_route(self, model, timeout=None, exclude=(), reuse_nodes=()):
        timeout = timeout if timeout is not None else float(os.environ.get("STAGEPLANE_RECOVERY_WAIT_SECONDS", "10"))
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.get_route(model, exclude, reuse_nodes)
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

    async def generate(self, request: ChatRequest, tenant="default"):
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
        used_nodes, generated, epoch = {}, [], 0
        record = {"id": session, "model": model, "tenant": tenant, "status": "running", "started": time.time(),
                  "prompt_tokens": len(tokens), "output_tokens": 0, "recoveries": 0, "token_ms": [],
                  "boundary_bytes": 0, "updated": time.time(), "route": [], "last_trace": []}
        self.active_requests[session] = record
        generator = torch.Generator().manual_seed(request.seed)
        previous_text = ""
        started = time.perf_counter()
        offset, input_tokens = 0, tokens[:]
        failures = 0
        failed_nodes = set()
        resident_session_nodes = set()
        try:
            route_path = self.get_route(model)
            record["route"] = route_path
            self.event("request_started", request=session, model=model, route=route_path, prompt_tokens=len(tokens))
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
                        resident_session_nodes.update(hop["node"] for hop in route_path)
                        break
                    except ChunkFailure as error:
                        # Prefix replay restores attention AND recurrent state. No output
                        # is committed until this entire forward pass has succeeded.
                        failed_index = next((i for i, hop in enumerate(route_path)
                                             if hop["node"] == error.node and hop["chunk"] == error.chunk), None)
                        if failed_index is not None:
                            resident_session_nodes.update(hop["node"] for hop in route_path[:failed_index])
                        failures += 1
                        if failures > self.retry_limit:
                            raise RuntimeError(f"Recovery exhausted: {error}") from error
                        epoch += 1
                        record["recoveries"] += 1
                        record.update(status="recovering", updated=time.time())
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
                        record["route"] = []
                        route_path = await self.ready_route(model, exclude=failed_nodes, reuse_nodes=resident_session_nodes)
                        record.update(route=route_path, status="running", updated=time.time())
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
                record.update(last_trace=trace, updated=time.time())
                hidden_size = manifest["config"].get("text_config", manifest["config"])["hidden_size"]
                record["boundary_bytes"] += sum(input_tokens.__len__() * hidden_size * 4 for a, b in zip(route_path, route_path[1:]) if a["node"] != b["node"])
                self.event("token_committed", request=session, epoch=epoch, offset=step,
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
            self.event("request_completed", request=session, metrics=record.copy())
            yield {"id": session, "delta": "", "model": model,
                   "finish_reason": "stop" if generated[-1] == tokenizer.eos_token_id else "length",
                   "usage": {"prompt_tokens": len(tokens), "completion_tokens": len(generated),
                             "total_tokens": len(tokens) + len(generated)}}
        except BaseException as error:
            if record["status"] != "completed":
                record["status"] = "cancelled" if isinstance(error, (asyncio.CancelledError, GeneratorExit)) else "failed"
                record["error"] = type(error).__name__
                self.event("request_" + record["status"], request=session)
            raise
        finally:
            record.update(finished=time.time(), updated=time.time(), elapsed_ms=(time.perf_counter() - started) * 1000)
            record["tokens_per_second"] = len(generated) / max(record["elapsed_ms"] / 1000, 1e-9)
            if record["status"] in ("running", "recovering"):
                record["status"] = "cancelled"
            self.active_requests.pop(session, None)
            self.request_totals[record["status"]] += 1
            self.requests.append(record)
            async def release():
                await self.cleanup(tenant, session, used_nodes)
                self.settle_transitions()
            cleanup = asyncio.create_task(release())
            self.cleanup_tasks.add(cleanup)
            cleanup.add_done_callback(self.cleanup_tasks.discard)
            # ASGI cancellation may be re-delivered at every await inside an AnyIO
            # cancellation scope. The detached task must outlive that scope.
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                pass


def create_app(coordinator: Coordinator):
    c = coordinator

    @asynccontextmanager
    async def lifespan(app):
        watcher = asyncio.create_task(c.watchdog())
        yield
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        for task in c.import_tasks:
            task.cancel()
        await asyncio.gather(*c.import_tasks, return_exceptions=True)
        if c.cleanup_tasks:
            _, pending = await asyncio.wait(list(c.cleanup_tasks), timeout=5)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        await c.transport.close()
        c.db.close()

    app = FastAPI(title="Stageplane executor", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestBodyLimit)
    protected = [Depends(c.access.authorize)]

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
        while len(c.jobs) >= 100:
            oldest = next((key for key, item in c.jobs.items() if item["status"] != "running"), None)
            if oldest is None:
                break
            del c.jobs[oldest]
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
        previous = c.deployments.get(model, {})
        c.deployments[model] = {**previous, "replicas": body.replicas, "groups": body.groups,
                                "placements": previous.get("placements", {})}
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

    @app.post("/api/nodes/{node_id}/resume", dependencies=protected)
    async def resume(node_id: str):
        if node_id not in c.nodes:
            raise HTTPException(404, "Unknown node")
        node = c.nodes[node_id]
        node["draining"] = False
        now = time.time()
        node["active"] = (now - node["last_seen"] <= float(os.environ.get("STAGEPLANE_HEARTBEAT_TIMEOUT", "8"))
                          and now >= node.get("quarantined_until", 0))
        c.rebalance({node["group"]})
        c.event("node_resumed", node=node_id, active=node["active"])
        return {"draining": False}

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
        tenant = request.state.principal.tenant
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
                async with contextlib.aclosing(c.generate(body, tenant=tenant)) as generation:
                    async for item in generation:
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
            async for item in c.generate(body, tenant=tenant):
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
            try:
                c.access.websocket_principal(hello.get("token", ""))
            except HTTPException:
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

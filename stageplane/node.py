from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

import grpc
import httpx
import psutil
import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket

from .common import MAX_CONTEXT, MAX_FRAME, NodeInfo, auth_headers, authorize, authorize_ws, digest, safe_path, tls_kwargs
from .runtime import Stage, cache_budget
from .wire import ChunkFailure, Transport, error_frame, pack, unpack
from .resources import available_limits, sample

log = logging.getLogger("stageplane.node")


class Worker:
    def __init__(self, info: NodeInfo, executor: str, cache: Path, device="cpu", dtype="float32", max_sessions=8):
        self.info, self.executor, self.cache = info, executor.rstrip("/"), cache.resolve()
        total_memory, total_cpu = available_limits()
        if "memory_bytes" not in info.model_fields_set:
            self.info.memory_bytes = total_memory
        if self.info.cpu_cores is None:
            self.info.cpu_cores = total_cpu
        self.cache.mkdir(parents=True, exist_ok=True)
        self.device, self.dtype, self.max_sessions = device, dtype, max_sessions
        self.loaded, self.manifests, self.sessions, self.epochs = {}, {}, {}, {}
        self.locks, self.metrics = {}, {"stage_ms": {}, "calls": 0, "bytes_received": 0, "bytes_sent": 0}
        self.identity_touched = {}
        self.reservations = {}
        self.transport = Transport()
        self.errors = {}
        self.compute = asyncio.Semaphore(int(os.environ.get("STAGEPLANE_NODE_CONCURRENCY", "1")))
        self.process = psutil.Process()
        self.process.cpu_percent(None)
        self.background_compute = set()
        self.loading = False
        self.loading_finished = 0

    async def ensure_file(self, model, name, expected):
        root = self.cache / model
        root.mkdir(parents=True, exist_ok=True)
        path = safe_path(root, name)
        if path.exists() and await asyncio.to_thread(digest, path) == expected["sha256"]:
            return path
        # Explicit test optimization; ordinary workers fetch every assigned artifact.
        local = os.environ.get("STAGEPLANE_LOCAL_ARTIFACTS")
        if local:
            source = safe_path(Path(local), model + "/" + name)
            if source.exists() and await asyncio.to_thread(digest, source) == expected["sha256"]:
                path.unlink(missing_ok=True)
                os.link(source, path)
                return path
        tmp = path.with_suffix(".partial")
        async with self.transport.http.stream("GET", f"{self.executor}/api/artifacts/{model}/{name}") as r:
            r.raise_for_status()
            total = 0
            with tmp.open("wb") as out:
                async for block in r.aiter_bytes(1024 * 1024):
                    total += len(block)
                    if total > expected["bytes"]:
                        raise ValueError("Artifact larger than declared size")
                    out.write(block)
        if tmp.stat().st_size != expected["bytes"] or await asyncio.to_thread(digest, tmp) != expected["sha256"]:
            tmp.unlink(missing_ok=True)
            raise ValueError("Artifact integrity check failed")
        tmp.replace(path)
        return path

    async def reconcile(self, desired):
        wanted = set()
        for assignment in desired:
            model, sid = assignment["model"], assignment["chunk"]
            key = model + ":" + sid
            wanted.add(key)
            if key in self.loaded:
                continue
            self.loading = True
            try:
                if model not in self.manifests:
                    r = await self.transport.http.get(f"{self.executor}/api/manifests/{model}")
                    r.raise_for_status()
                    m = r.json()
                    if m["manifest_sha256"] != assignment["manifest_sha256"]:
                        raise ValueError("Manifest revision mismatch")
                    import hashlib
                    actual = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k != "manifest_sha256"}, sort_keys=True).encode()).hexdigest()
                    if actual != m["manifest_sha256"]:
                        raise ValueError("Manifest digest mismatch")
                    self.manifests[model] = m
                m = self.manifests[model]
                spec = next(s for s in m["stages"] if s["id"] == sid)
                for a in spec["artifacts"]:
                    name = m["artifacts"][a]["file"]
                    await self.ensure_file(model, name, m["files"][name])
                stage = await asyncio.to_thread(Stage, m, spec, self.cache / model, self.device, self.dtype)
                # A real fixture is executed before declaring readiness.
                state = stage.state()
                cfg = m["config"].get("text_config", m["config"])
                fixture = torch.tensor([[1]], dtype=torch.long) if spec["kind"] == "embedding" else torch.zeros(1, 1, cfg["hidden_size"])
                output = await asyncio.to_thread(stage, fixture, state, 0)
                expected_last = cfg["vocab_size"] if spec["kind"] == "head" else cfg["hidden_size"]
                if output.shape != (1, 1, expected_last) or not torch.isfinite(output).all():
                    raise ValueError("Readiness fixture failed")
                self.loaded[key] = stage
                self.errors.pop(key, None)
            except Exception as e:
                self.errors[key] = str(e)
                log.exception("Loading %s failed", key)
            finally:
                self.loading = False
                self.loading_finished = time.monotonic()
        busy = {k[2] for k in self.sessions}
        for key in list(self.loaded):
            if key not in wanted and key not in busy:
                del self.loaded[key]

    def stats(self):
        self.info.ready = list(self.loaded)
        parameters = {p.data_ptr(): p.numel() * p.element_size() for s in self.loaded.values() for p in s.parameters()}
        self.info.metrics = {**self.metrics, **sample(self.info.memory_bytes, self.process, self.info.cpu_cores),
                             "loaded_weight_bytes": sum(parameters.values()),
                             "available_storage_bytes": shutil.disk_usage(self.cache if self.cache.exists() else self.cache.parent).free,
                             "sessions": len({s[:2] for s in self.sessions}), "errors": self.errors,
                             "loading": self.loading or time.monotonic() - self.loading_finished < 2,
                             "cache_bytes": sum(p.stat().st_size for p in self.cache.glob("*/*.safetensors")),
                             "transport": self.transport.metrics.copy(),
                             "device": self.device, "dtype": self.dtype}
        return self.info.model_dump()

    async def heartbeat(self):
        registered = False
        loader = None
        while True:
            try:
                endpoint = "/api/nodes/register" if not registered else f"/api/nodes/{self.info.id}/heartbeat"
                r = await self.transport.http.post(self.executor + endpoint, json=self.stats())
                if r.status_code == 404:
                    registered = False
                    continue
                r.raise_for_status()
                registered = True
                if loader is None or loader.done():
                    if loader is not None:
                        loader.result()
                    loader = asyncio.create_task(self.reconcile(r.json()["assignments"]))
                expiry = time.time() - int(os.environ.get("STAGEPLANE_SESSION_TTL", "120"))
                for key, state in list(self.sessions.items()):
                    if state.touched < expiry and not self.locks.get(key[:2], asyncio.Lock()).locked():
                        del self.sessions[key]
                        self.reservations.pop(key, None)
                for identity, touched in list(self.identity_touched.items()):
                    if touched < expiry and not self.locks.get(identity, asyncio.Lock()).locked():
                        self.epochs.pop(identity, None)
                        self.locks.pop(identity, None)
                        self.identity_touched.pop(identity, None)
            except asyncio.CancelledError:
                if loader:
                    loader.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await loader
                raise
            except Exception:
                log.exception("Heartbeat/reconciliation failed")
            await asyncio.sleep(1)

    async def forward(self, data):
        meta, value = unpack(data)
        route, index = meta["route"], meta["index"]
        if not 0 <= index < len(route) <= 1024:
            raise ValueError("Invalid route position")
        hop = route[index]
        if hop["node"] != self.info.id or not time.time() < meta["lease_until"] < time.time() + 180:
            raise ValueError("Wrong node or expired route lease")
        key = meta["model"] + ":" + hop["chunk"]
        if key not in self.loaded:
            raise ValueError("Stage is not ready")
        m = self.manifests[meta["model"]]
        if meta["revision"] != m["revision"] or meta["manifest_sha256"] != m["manifest_sha256"]:
            raise ValueError("Incompatible model revision")
        offset = meta["offset"]
        if not isinstance(offset, int) or offset < 0 or value.shape[1] < 1 or offset + value.shape[1] > MAX_CONTEXT:
            raise ValueError("Context budget exceeded")
        stage = self.loaded[key]
        if stage.spec["kind"] == "embedding":
            if value.dtype != torch.int64 or value.ndim != 2 or value.min() < 0 or value.max() >= stage.cfg.vocab_size:
                raise ValueError("Invalid input token IDs")
        elif value.ndim != 3 or value.shape[-1] != stage.cfg.hidden_size or not value.is_floating_point():
            raise ValueError("Incompatible hidden tensor")
        if len(meta["session"]) > 128 or len(meta["tenant"]) > 128:
            raise ValueError("Invalid session identity")
        identity = (meta["tenant"], meta["session"])
        epoch = meta["epoch"]
        if not isinstance(epoch, int) or epoch < 0 or epoch > 2**32:
            raise ValueError("Invalid epoch")
        self.identity_touched[identity] = time.time()
        async with self.locks.setdefault(identity, asyncio.Lock()):
            old = self.epochs.get(identity, -1)
            if epoch < old:
                raise ValueError("Stale route epoch")
            if epoch > old:
                self.cleanup(*identity, keep_epoch=True)
                self.epochs[identity] = epoch
            skey = (*identity, key)
            if skey not in self.sessions:
                if offset != 0:
                    raise ValueError("Missing prefix state: replay required")
                if len({s[:2] for s in self.sessions}) >= self.max_sessions and identity not in {s[:2] for s in self.sessions}:
                    raise ValueError("Worker session capacity exhausted")
                reserve = cache_budget(stage.cfg, stage.spec, meta.get("context_budget", MAX_CONTEXT))
                if psutil.Process().memory_info().rss + sum(self.reservations.values()) + reserve > self.info.memory_bytes:
                    raise ValueError("Worker memory admission budget exhausted")
                self.reservations[skey] = reserve
                self.sessions[skey] = stage.state()
            state = self.sessions[skey]
            started = time.perf_counter()
            executing = False
            async def execute():
                nonlocal executing
                async with self.compute:
                    executing = True
                    return await asyncio.to_thread(stage, value, state, offset)
            work = asyncio.create_task(execute())
            try:
                result = await asyncio.wait_for(asyncio.shield(work), float(os.environ.get("STAGEPLANE_CHUNK_TIMEOUT", "30")))
            except TimeoutError as e:
                self.sessions.pop(skey, None)
                self.reservations.pop(skey, None)
                if not executing:
                    work.cancel()
                else:
                    # Native kernels cannot be cancelled. Keep their compute
                    # permit until completion, and discard the abandoned state.
                    self.background_compute.add(work)
                    def finish(task):
                        self.background_compute.discard(task)
                        if not task.cancelled():
                            task.exception()
                    work.add_done_callback(finish)
                raise ChunkFailure(self.info.id, hop["chunk"], "chunk_timeout") from e
            elapsed = (time.perf_counter() - started) * 1000
            state.touched = time.time()
            self.metrics["stage_ms"][hop["chunk"]] = elapsed
            self.metrics["calls"] += 1
            self.metrics["bytes_received"] += len(data)
        meta.setdefault("trace", []).append({"node": self.info.id, "chunk": hop["chunk"], "ms": elapsed,
                                             "input_shape": list(value.shape), "output_shape": list(result.shape),
                                             "offset": offset})
        meta["index"] += 1
        outgoing = pack(meta, result)
        self.metrics["bytes_sent"] += len(outgoing)
        if meta["index"] < len(route):
            nxt = route[meta["index"]]
            # Stages colocated on one node execute without a network boundary.
            if nxt["node"] == self.info.id:
                return await self.forward(outgoing)
            return await self.transport.send(nxt, outgoing)
        return outgoing

    def cleanup(self, tenant, session, keep_epoch=False):
        for key in list(self.sessions):
            if key[:2] == (tenant, session):
                del self.sessions[key]
                self.reservations.pop(key, None)
        # Retain a bounded stale-epoch tombstone until TTL cleanup.
        if not keep_epoch:
            self.epochs[(tenant, session)] = max(self.epochs.get((tenant, session), -1), 2**53)


def create_app(worker: Worker, grpc_port=0):
    @asynccontextmanager
    async def lifespan(app):
        hb = asyncio.create_task(worker.heartbeat())
        server = None
        if grpc_port:
            server = grpc.aio.server(options=[("grpc.max_receive_message_length", MAX_FRAME), ("grpc.max_send_message_length", MAX_FRAME)])

            async def rpc(data, context):
                token = os.environ.get("STAGEPLANE_TOKEN")
                if token and dict(context.invocation_metadata()).get("authorization") != f"Bearer {token}":
                    await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid bearer token")
                try:
                    return await worker.forward(data)
                except ChunkFailure as e:
                    return error_frame(e)
                except Exception as e:
                    await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))

            server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler("stageplane.Worker", {
                "Forward": grpc.unary_unary_rpc_method_handler(rpc)})])
            if os.environ.get("STAGEPLANE_TLS_CERT"):
                ca = Path(os.environ["STAGEPLANE_CA"]).read_bytes()
                credentials = grpc.ssl_server_credentials([(Path(os.environ["STAGEPLANE_TLS_KEY"]).read_bytes(),
                    Path(os.environ["STAGEPLANE_TLS_CERT"]).read_bytes())], root_certificates=ca, require_client_auth=True)
                server.add_secure_port(f"[::]:{grpc_port}", credentials)
            else:
                server.add_insecure_port(f"[::]:{grpc_port}")
            await server.start()
        yield
        hb.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hb
        if server:
            await server.stop(2)
        await worker.transport.close()

    app = FastAPI(title="Stageplane worker", lifespan=lifespan)
    protected = [Depends(authorize)]

    @app.get("/health", dependencies=protected)
    async def health():
        return worker.stats()

    @app.post("/v1/forward", dependencies=protected)
    async def forward(request: Request):
        body = bytearray()
        async for block in request.stream():
            body.extend(block)
            if len(body) > MAX_FRAME:
                raise HTTPException(413, "Frame too large")
        try:
            return Response(await worker.forward(bytes(body)), media_type="application/octet-stream")
        except ChunkFailure as e:
            return Response(error_frame(e), media_type="application/octet-stream")
        except (ValueError, KeyError, RuntimeError) as e:
            raise HTTPException(409, str(e)) from e

    @app.websocket("/ws/forward")
    async def websocket(ws: WebSocket):
        if not await authorize_ws(ws):
            return
        await ws.accept()
        try:
            data = await ws.receive_bytes()
            if len(data) > MAX_FRAME:
                raise ValueError("Frame too large")
            await ws.send_bytes(await worker.forward(data))
        except ChunkFailure as e:
            await ws.send_bytes(error_frame(e))
        except Exception as e:
            await ws.send_text(json.dumps({"error": str(e)}))
        finally:
            await ws.close()

    @app.delete("/sessions/{tenant}/{session}", dependencies=protected)
    async def cleanup(tenant: str, session: str):
        worker.cleanup(tenant, session)
        return {"released": True}

    return app

import asyncio
import time

import pytest
import torch

from scripts.tiny_checkpoint import make
from stageplane.common import ChatRequest, NodeInfo
from stageplane.distribution import distribute
from stageplane.executor import Coordinator
from stageplane.node import Worker
from stageplane.runtime import Stage
from stageplane.wire import ChunkFailure, pack


async def test_stage_deadline_releases_abandoned_session(tmp_path, monkeypatch):
    make(tmp_path / "source")
    m = distribute(str(tmp_path / "source"), tmp_path / "models")
    spec = m["stages"][1]
    stage = Stage(m, spec, tmp_path / "models" / m["key"])
    def slow(value, state, offset):
        time.sleep(.05)
        return value
    monkeypatch.setattr(stage, "forward", slow)
    monkeypatch.setenv("STAGEPLANE_CHUNK_TIMEOUT", ".001")
    worker = Worker(NodeInfo(id="slow", url="http://localhost"), "http://localhost", tmp_path / "cache")
    key = m["key"] + ":" + spec["id"]
    worker.loaded[key], worker.manifests[m["key"]] = stage, m
    meta = {"model": m["key"], "manifest_sha256": m["manifest_sha256"], "revision": m["revision"],
            "tenant": "t", "session": "s", "epoch": 0, "offset": 0, "index": 0,
            "lease_until": time.time() + 60, "route": [{"node": "slow", "chunk": spec["id"]}]}
    try:
        with pytest.raises(ChunkFailure, match="chunk_timeout") as failure:
            await worker.forward(pack(meta, torch.zeros(1, 1, 32)))
        assert failure.value.node == "slow" and failure.value.chunk == spec["id"]
        assert not worker.sessions and not worker.reservations
        await asyncio.gather(*worker.background_compute)
        assert not worker.background_compute
    finally:
        await worker.transport.close()


async def test_execution_retry_is_bounded_and_uses_another_ready_node(tmp_path, monkeypatch):
    make(tmp_path / "source")
    m = distribute(str(tmp_path / "source"), tmp_path / "models")
    c = Coordinator(tmp_path / "models")
    for i in range(4):
        c.register(NodeInfo(id=f"node-{i}", url=f"http://node-{i}", memory_bytes=10**9,
                            ready=[m["key"] + ":" + s["id"] for s in m["stages"]]))
    c.deployments[m["key"]] = {"replicas": 2, "groups": 1, "placements": {}}
    c.rebalance()
    c.retry_limit = 2
    calls = []
    async def fail(hop, data):
        calls.append(hop["node"])
        raise ChunkFailure(hop["node"], hop["chunk"], "chunk_timeout")
    monkeypatch.setattr(c.transport, "send", fail)
    # Keep all healthy replicas ready; no client-level retries participate.
    async def ready(model, timeout=None, exclude=(), reuse_nodes=()):
        return c.get_route(model, exclude, reuse_nodes)
    monkeypatch.setattr(c, "ready_route", ready)
    request = ChatRequest(model=m["key"], messages=[{"role": "user", "content": "hello"}], max_tokens=1)
    try:
        with pytest.raises(RuntimeError, match="Recovery exhausted"):
            async for _ in c.generate(request):
                pass
        assert len(calls) == 3 and len(set(calls)) == 3
        assert len([e for e in c.events if e["kind"] == "chunk_retry"]) == 2
    finally:
        c.db.close()
        await c.transport.close()

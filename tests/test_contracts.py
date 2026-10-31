import json
import time

import httpx
import pytest
import torch

from scripts.tiny_checkpoint import make
from stageplane.common import NodeInfo, safe_path
from stageplane.distribution import distribute
from stageplane.executor import Coordinator, create_app
from stageplane.node import Worker
from stageplane.runtime import Stage
from stageplane.wire import pack


async def test_worker_rejects_stale_epoch_shapes_offsets_and_leases(tmp_path):
    make(tmp_path / "source")
    m = distribute(str(tmp_path / "source"), tmp_path / "models", layers_per_chunk=1)
    info = NodeInfo(id="n", url="http://127.0.0.1:1")
    w = Worker(info, "http://127.0.0.1:1", tmp_path / "cache")
    sid = m["stages"][1]
    key = m["key"] + ":" + sid["id"]
    w.loaded[key] = Stage(m, sid, tmp_path / "models" / m["key"])
    w.manifests[m["key"]] = m
    meta = {"tenant": "a", "session": "s", "model": m["key"], "revision": m["revision"],
            "manifest_sha256": m["manifest_sha256"], "route": [{"node": "n", "chunk": sid["id"]}],
            "index": 0, "offset": 0, "epoch": 2, "lease_until": time.time() + 60}
    hidden = torch.zeros(1, 2, 32)
    try:
        await w.forward(pack(meta.copy(), hidden))
        for changes, value, message in [({"epoch": 1}, hidden, "Stale"),
            ({"offset": 1}, hidden, "Cache offset"), ({"lease_until": time.time()-1}, hidden, "expired"),
            ({"revision": "different"}, hidden, "revision"), ({}, torch.zeros(1, 2, 31), "hidden")]:
            with pytest.raises(ValueError, match=message):
                await w.forward(pack({**meta, **changes}, value))
        # Fresh epoch resets state; one session cannot read another tenant's cache.
        await w.forward(pack({**meta, "epoch": 3}, hidden))
        with pytest.raises(ValueError, match="Missing prefix"):
            await w.forward(pack({**meta, "tenant": "b", "offset": 2}, torch.zeros(1, 1, 32)))
        w.cleanup("a", "s")
        assert not any(k[:2] == ("a", "s") for k in w.sessions)
    finally:
        await w.transport.close()


async def test_artifact_integrity_and_path(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    w = Worker(NodeInfo(id="n", url="http://localhost"), "http://executor", cache)
    w.transport.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"corrupt")))
    try:
        with pytest.raises(ValueError, match="integrity"):
            await w.ensure_file("model", "blocks.safetensors", {"bytes": 7, "sha256": "0" * 64})
        assert not (cache / "model" / "blocks.safetensors").exists()
        with pytest.raises(ValueError, match="outside"):
            safe_path(cache, "../../outside")
    finally:
        await w.transport.close()


async def test_model_import_and_no_ready_route(tmp_path, monkeypatch):
    monkeypatch.delenv("STAGEPLANE_TOKEN", raising=False)
    make(tmp_path / "source")
    c = Coordinator(tmp_path / "models")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(c)), base_url="http://executor") as client:
        response = await client.post("/api/models/import", json={"source": str(tmp_path / "source"), "layers_per_chunk": 1})
        assert response.status_code == 202
        job = response.json()
        for _ in range(100):
            import asyncio
            await asyncio.sleep(.02)
            job = (await client.get("/api/jobs/" + job["id"])).json()
            if job["status"] != "running":
                break
        assert job["status"] == "completed", job
        response = await client.post("/v1/chat/completions", json={"model": job["model"], "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 503
        bad = await client.post("/v1/chat/completions", json={"model": job["model"], "messages": [], "tools": []})
        assert bad.status_code == 422
    c.db.close()
    await c.transport.close()


async def test_removed_model_in_persistent_metadata_cannot_break_registration(tmp_path):
    c = Coordinator(tmp_path / "models")
    c.db.execute("INSERT INTO deployments VALUES (?, ?)", ("removed-revision", json.dumps({"placements": {"embedding": ["n"]}})))
    c.db.commit()
    c.db.close()
    await c.transport.close()
    restored = Coordinator(tmp_path / "models")
    try:
        assert restored.register(NodeInfo(id="n", url="http://localhost")) == {"assignments": []}
    finally:
        restored.db.close()
        await restored.transport.close()

import asyncio
import json

import httpx

from stageplane.common import NodeInfo
from stageplane.node import Worker


async def test_default_worker_publishes_exact_full_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr("stageplane.node.available_limits", lambda: (8_000_000_000, 4.0))
    worker = Worker(NodeInfo(id="all-resources", url="http://worker"), "http://executor", tmp_path / "cache")
    captured = {}
    registered = asyncio.Event()
    async def handle(request):
        captured.update(json.loads(request.content))
        registered.set()
        return httpx.Response(200, json={"assignments": []})
    await worker.transport.http.aclose()
    worker.transport.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    task = asyncio.create_task(worker.heartbeat())
    try:
        await asyncio.wait_for(registered.wait(), 2)
        assert captured["memory_bytes"] == 8_000_000_000
        assert captured["cpu_cores"] == 4.0
        assert isinstance(captured["memory_bytes"], int)
        assert isinstance(captured["cpu_cores"], float)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await worker.transport.close()

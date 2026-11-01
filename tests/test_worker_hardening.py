import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import torch

from stageplane.common import NodeInfo
from stageplane.node import Worker
from stageplane.wire import pack


class ControlledStage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spec = {"id": "block", "kind": "blocks", "start": 0, "end": 1}
        self.cfg = SimpleNamespace(hidden_size=2, num_key_value_heads=1, num_attention_heads=1)
        self.started, self.release = threading.Event(), threading.Event()
    def state(self):
        return SimpleNamespace(touched=time.time(), offset=0)
    def forward(self, value, state, offset):
        self.started.set()
        assert self.release.wait(5)
        state.offset += value.shape[1]
        return value


def installed_worker(tmp_path):
    worker = Worker(NodeInfo(id="worker", url="http://localhost", memory_bytes=10**12), "http://localhost", tmp_path / "cache")
    stage = ControlledStage()
    worker.loaded["model:block"] = stage
    worker.manifests["model"] = {"revision": "revision", "manifest_sha256": "digest"}
    meta = {"model": "model", "revision": "revision", "manifest_sha256": "digest", "tenant": "tenant", "session": "session",
            "epoch": 0, "offset": 0, "context_budget": 4, "index": 0, "lease_until": time.time() + 60,
            "route": [{"node": "worker", "chunk": "block"}]}
    return worker, stage, meta


@pytest.mark.parametrize("release_session", [False, True])
async def test_cancelled_kernel_keeps_capacity_until_native_completion(tmp_path, release_session):
    worker, stage, meta = installed_worker(tmp_path)
    task = asyncio.create_task(worker.forward(pack(meta, torch.zeros(1, 1, 2))))
    try:
        for _ in range(100):
            if stage.started.is_set():
                break
            await asyncio.sleep(.005)
        assert stage.started.is_set()
        reserved = sum(worker.reservations.values())
        if release_session:
            worker.cleanup("tenant", "session")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not worker.sessions and not worker.reservations
        assert sum(worker.abandoned_reservations.values()) == reserved > 0
        assert worker.compute.locked()
        stage.release.set()
        await asyncio.gather(*worker.background_compute)
        await asyncio.sleep(0)
        assert not worker.background_compute and not worker.abandoned_reservations
        assert not worker.compute.locked()
    finally:
        stage.release.set()
        await worker.transport.close()


async def test_session_cannot_extend_unreserved_context(tmp_path):
    worker, stage, meta = installed_worker(tmp_path)
    stage.release.set()
    try:
        await worker.forward(pack(meta, torch.zeros(1, 1, 2)))
        meta.update(offset=1, context_budget=8)
        with pytest.raises(ValueError, match="reservation exceeded"):
            await worker.forward(pack(meta, torch.zeros(1, 4, 2)))
    finally:
        await worker.transport.close()


async def test_interrupted_artifact_transfer_removes_partial_file(tmp_path):
    worker, _, _ = installed_worker(tmp_path)
    await worker.transport.http.aclose()
    worker.transport.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"oversize")))
    try:
        with pytest.raises(ValueError, match="larger"):
            await worker.ensure_file("model", "weights.safetensors", {"bytes": 1, "sha256": "0" * 64})
        assert not list((worker.cache / "model").glob("*.partial"))
        with pytest.raises(ValueError, match="outside"):
            await worker.ensure_file("../../escape", "weights.safetensors", {"bytes": 1, "sha256": "0" * 64})
    finally:
        await worker.transport.close()

import httpx
import pytest
import torch

from stageplane.wire import ChunkFailure, Transport, pack


@pytest.mark.parametrize("status", [409, 429, 500, 503])
async def test_rest_worker_rejections_reach_bounded_executor_retry(status):
    transport = Transport()
    await transport.http.aclose()
    transport.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status)))
    try:
        with pytest.raises(ChunkFailure) as caught:
            await transport.send({"node": "failed-worker", "chunk": "blocks-0", "url": "http://worker"}, b"request")
        assert caught.value.node == "failed-worker"
        assert caught.value.chunk == "blocks-0"
        assert transport.metrics["calls"] == 1  # The transport never retries independently.
    finally:
        await transport.close()


async def test_rest_peer_cannot_return_unbounded_body(monkeypatch):
    monkeypatch.setattr("stageplane.wire.MAX_FRAME", 1024)
    transport = Transport()
    await transport.http.aclose()
    transport.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 1025)))
    try:
        with pytest.raises(ChunkFailure, match="oversized_response"):
            await transport.send({"node": "worker", "chunk": "block", "url": "http://worker"}, b"request")
    finally:
        await transport.close()


def test_oversized_tensor_is_rejected_before_serialization(monkeypatch):
    monkeypatch.setattr("stageplane.wire.MAX_FRAME", 1024)
    def forbidden(*args, **kwargs):
        pytest.fail("Oversized tensor must not be serialized")
    monkeypatch.setattr("stageplane.wire.save", forbidden)
    with pytest.raises(ValueError, match="frame exceeds"):
        pack({}, torch.zeros(1, 1024))

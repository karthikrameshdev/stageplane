import json

import httpx
import pytest

from stageplane.executor import Coordinator, create_app
from stageplane.security import AccessPolicy, RequestBodyLimit


@pytest.fixture
def keys(tmp_path, monkeypatch):
    entries = [
        {"token": "operator-key-00000001", "tenant": "operations", "role": "admin"},
        {"token": "observer-key-00000001", "tenant": "operations", "role": "viewer"},
        {"token": "tenant-a-key-00000001", "tenant": "organization-a", "role": "inference"},
        {"token": "tenant-b-key-00000001", "tenant": "organization-b", "role": "inference"},
    ]
    path = tmp_path / "keys.json"
    path.write_text(json.dumps(entries))
    monkeypatch.setenv("STAGEPLANE_TOKEN", "internal-worker-key")
    monkeypatch.setenv("STAGEPLANE_API_KEYS_FILE", str(path))
    return entries


async def test_roles_separate_worker_control_observation_and_inference(keys, tmp_path):
    c = Coordinator(tmp_path / "models")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(c)), base_url="http://executor") as client:
            for entry in keys:
                headers = {"Authorization": "Bearer " + entry["token"]}
                assert (await client.get("/v1/models", headers=headers)).status_code == 200
                state = await client.get("/api/state", headers=headers)
                assert state.status_code == (403 if entry["role"] == "inference" else 200)
                assert (await client.post("/api/nodes/register", headers=headers,
                    json={"id": "worker", "url": "http://127.0.0.1:9000"})).status_code == 403
                if entry["role"] != "admin":
                    assert (await client.post("/api/deployments", headers=headers, json={"model": "missing"})).status_code == 403
            assert (await client.get("/api/state", headers={"Authorization": "Bearer invalid"})).status_code == 401
            internal = {"Authorization": "Bearer internal-worker-key"}
            assert (await client.post("/api/nodes/register", headers=internal,
                json={"id": "worker", "url": "http://127.0.0.1:9000"})).status_code == 200
    finally:
        await c.transport.close()
        c.db.close()


async def test_chat_uses_authenticated_tenant_and_viewer_cannot_infer(keys, tmp_path, monkeypatch):
    c = Coordinator(tmp_path / "models")
    c.models["tiny"] = {"source": "tiny"}
    monkeypatch.setattr(c, "get_route", lambda model: [])
    observed = []
    async def generate(body, tenant="default"):
        observed.append(tenant)
        yield {"id": "request", "model": "tiny", "delta": "ok", "finish_reason": "stop",
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    monkeypatch.setattr(c, "generate", generate)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(c)), base_url="http://executor") as client:
            for entry in keys[2:]:
                response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer " + entry["token"]},
                    json={"model": "tiny", "messages": [{"role": "user", "content": "hello"}]})
                assert response.status_code == 200, response.text
            response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer " + keys[1]["token"]},
                json={"model": "tiny", "messages": [{"role": "user", "content": "hello"}]})
            assert response.status_code == 403
        assert observed == ["organization-a", "organization-b"]
    finally:
        await c.transport.close()
        c.db.close()


def test_event_credentials_and_invalid_configuration_fail_closed(keys, tmp_path, monkeypatch):
    from fastapi import HTTPException
    policy = AccessPolicy()
    assert policy.websocket_principal(keys[1]["token"]).role == "viewer"
    with pytest.raises(HTTPException) as caught:
        policy.websocket_principal(keys[2]["token"])
    assert caught.value.status_code == 403
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps([keys[0], keys[0]]))
    monkeypatch.setenv("STAGEPLANE_API_KEYS_FILE", str(path))
    with pytest.raises(ValueError, match="Invalid API key configuration") as caught:
        AccessPolicy()
    assert keys[0]["token"] not in str(caught.value)
    monkeypatch.delenv("STAGEPLANE_TOKEN")
    with pytest.raises(ValueError, match="requires STAGEPLANE_TOKEN"):
        AccessPolicy()


@pytest.mark.parametrize("declared", [True, False])
async def test_body_limit_rejects_declared_and_streamed_overflow(declared):
    called = False
    async def downstream(scope, receive, send):
        nonlocal called
        called = True
    messages = iter([{"type": "http.request", "body": b"a" * 700, "more_body": True},
                     {"type": "http.request", "body": b"b" * 700}])
    async def receive():
        return next(messages)
    sent = []
    async def send(message):
        sent.append(message)
    await RequestBodyLimit(downstream, max_bytes=1024)(
        {"type": "http", "method": "POST", "headers": [(b"content-length", b"1400")] if declared else []}, receive, send)
    assert sent[0]["status"] == 413
    assert not called

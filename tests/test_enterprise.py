import asyncio
import json

import httpx
import pytest
import torch

from scripts.cluster import Cluster
from scripts.tiny_checkpoint import make
from stageplane.common import ChatRequest
from stageplane.distribution import distribute
from stageplane.executor import Coordinator, create_app
from stageplane.observability import Histogram, Metrics
from stageplane.runtime import LocalPipeline, cache_budget, text_config


def test_chat_request_accepts_openai_fields():
    request = ChatRequest(model="m", messages=[
        {"role": "developer", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]}],
        max_completion_tokens=5, stop="END", user="u1", stream_options={"include_usage": True})
    assert request.output_limit == 5 and request.stop == ["END"]
    assert [m.template() for m in request.messages] == [{"role": "system", "content": "be brief"},
                                                         {"role": "user", "content": "hi there"}]
    with pytest.raises(ValueError):
        ChatRequest(model="m", messages=[{"role": "user", "content": "x"}], stop=["a"] * 5)
    with pytest.raises(ValueError):
        ChatRequest(model="m", messages=[{"role": "user", "content": "x"}], n=2)


def test_kv_budget_scales_with_dtype_and_bounds_prefill(tmp_path):
    make(tmp_path / "src")
    m = distribute(str(tmp_path / "src"), tmp_path / "models", layers_per_chunk=2)
    cfg, spec = text_config(m), m["stages"][1]
    full = cache_budget(cfg, spec, 8192, 4)
    half = cache_budget(cfg, spec, 8192, 2)
    chunked = cache_budget(cfg, spec, 8192, 4, prefill_chunk=512)
    assert half < full and chunked < full
    # Chunked prefill keeps the score matrix linear in context, not quadratic.
    assert cache_budget(cfg, spec, 16384, 4, 512) < 2.2 * cache_budget(cfg, spec, 8192, 4, 512)


@pytest.mark.parametrize("architecture", ["qwen3", "qwen3_5_text"])
def test_chunked_prefill_matches_single_pass(tmp_path, architecture):
    make(tmp_path / "src", architecture)
    m = distribute(str(tmp_path / "src"), tmp_path / "models", layers_per_chunk=2)
    tokens = torch.randint(0, 64, (1, 45), generator=torch.Generator().manual_seed(3))
    whole = LocalPipeline(m, tmp_path / "models" / m["key"]).forward(tokens)
    pipeline = LocalPipeline(m, tmp_path / "models" / m["key"])
    for start in range(0, 45, 7):
        part = pipeline.forward(tokens[:, start:start + 7], start)
    assert torch.allclose(whole, part, atol=1e-5)


def test_histogram_and_metrics_render():
    h = Histogram((1, 2))
    for v in (0.5, 1.5, 3):
        h.observe(v)
    lines = list(h.lines("x", 'a="b"'))
    assert 'x_bucket{a="b",le="1"} 1' in lines and 'x_bucket{a="b",le="+Inf"} 3' in lines
    metrics = Metrics()
    metrics.observe({"model": "m\"bad", "tenant": "t", "status": "completed", "output_tokens": 3,
                     "prompt_tokens": 2, "ttft_ms": 10, "token_ms": [10, 5], "elapsed_ms": 20})
    assert metrics.tokens[("m_bad", "t")] == 3


@pytest.fixture
async def executor(tmp_path, monkeypatch):
    monkeypatch.delenv("STAGEPLANE_TOKEN", raising=False)
    make(tmp_path / "source")
    distribute(str(tmp_path / "source"), tmp_path / "models", layers_per_chunk=1)
    c = Coordinator(tmp_path / "models", state_file=tmp_path / "state.sqlite")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(c)), base_url="http://executor") as client:
        yield c, client
    c.db.close()
    await c.transport.close()


async def test_operational_endpoints(executor):
    c, client = executor
    model = next(iter(c.models))
    health = await client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert health.json()["version"] and health.headers["x-request-id"] == "trace-123"
    assert len((await client.get("/health", headers={"X-Request-ID": "bad id\n"})).headers["x-request-id"]) == 32
    assert (await client.get("/ready")).status_code == 503
    card = (await client.get(f"/v1/models/{model}")).json()
    assert card["id"] == model and card["context_window"] > 0 and card["deployed"] is False
    missing = await client.get("/v1/models/nope")
    assert missing.status_code == 404 and missing.json()["error"]["type"] == "not_found_error"
    invalid = await client.post("/v1/chat/completions", json={"model": model, "messages": []})
    assert invalid.status_code == 422 and "messages" in invalid.json()["error"]["message"]
    assert isinstance(invalid.json()["detail"], list)
    deployed = await client.post("/api/deployments", json={"model": model})
    assert deployed.status_code == 503  # no workers yet; plan persisted as pending
    listed = (await client.get("/api/deployments")).json()["deployments"]
    assert listed[0]["model"] == model and listed[0]["strategy"] == "capacity"
    assert (await client.get(f"/api/deployments/{model}")).json()["routing"] == "hub"
    no_route = await client.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "x"}]})
    assert no_route.status_code == 503 and no_route.headers["retry-after"] == "5"
    assert (await client.delete(f"/api/deployments/{model}")).json() == {"model": model, "deleted": True}
    assert (await client.get(f"/api/deployments/{model}")).status_code == 404
    audit = (await client.get("/api/audit")).json()["entries"]
    assert [e["action"] for e in audit] == ["deployment.delete", "deployment.apply"]
    assert audit[1]["detail"]["strategy"] == "capacity"
    text = (await client.get("/metrics")).text
    assert "stageplane_build_info" in text and 'stageplane_http_request_duration_seconds_count{method="GET",route="/health"' in text
    assert c.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 2


async def test_tenant_quota_returns_retry_after(executor, monkeypatch):
    c, client = executor
    model = next(iter(c.models))
    c.tenant_limit = 1
    c.tenant_inflight["default"] = 1
    monkeypatch.setattr(c, "get_route", lambda *a, **k: [])
    r = await client.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 429 and r.headers["retry-after"] == "1"
    assert r.json()["error"]["type"] == "rate_limit_error"


def single_node_answer(tmp_path, prompt):
    cluster = Cluster(tmp_path / "models", tmp_path / "reference", nodes=1, transports=["rest"])
    try:
        cluster.start()
        cluster.wait_ready()
        r = cluster.http.post("/v1/chat/completions", json={"model": cluster.model, "temperature": 0, "max_tokens": 12,
                                                            "messages": [{"role": "user", "content": prompt}]})
        return r.json()["choices"][0]["message"]["content"]
    finally:
        cluster.close()


def test_live_cluster_openai_semantics_long_prompt_and_metrics(tmp_path):
    make(tmp_path / "src")
    distribute(str(tmp_path / "src"), tmp_path / "models", layers_per_chunk=1)
    cluster = Cluster(tmp_path / "models", tmp_path / "run", nodes=2, transports=["rest", "grpc"])
    cluster.env["STAGEPLANE_PREFILL_CHUNK"] = "16"
    try:
        cluster.start()
        cluster.wait_ready()
        base = {"model": cluster.model, "temperature": 0, "max_tokens": 12}
        long_prompt = " ".join(f"word{i}" for i in range(120))
        full = cluster.http.post("/v1/chat/completions", json={**base, "messages": [{"role": "user", "content": long_prompt}]})
        assert full.status_code == 200, full.text
        body = full.json()
        content = body["choices"][0]["message"]["content"]
        assert body["usage"]["prompt_tokens"] > 16 and body["system_fingerprint"].startswith("stageplane-")
        record = cluster.http.get("/api/state").json()["requests"][-1]
        assert record["prefill_chunks"] == -(-body["usage"]["prompt_tokens"] // 16)
        assert len(content) > 2
        # Chunked prefill over the network matches a single unchunked forward pass.
        assert content == single_node_answer(tmp_path, long_prompt)
        stop = content[2:4]
        stopped = cluster.http.post("/v1/chat/completions", json={**base, "stop": [stop],
                                    "messages": [{"role": "user", "content": long_prompt}]}).json()
        assert stopped["choices"][0]["finish_reason"] == "stop"
        assert stopped["choices"][0]["message"]["content"] == content[:content.find(stop)]
        chunks, usage_only = [], None
        with cluster.http.stream("POST", "/v1/chat/completions", json={
                **base, "stream": True, "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": long_prompt}]}) as response:
            for line in response.iter_lines():
                if line.startswith("data: {"):
                    item = json.loads(line[6:])
                    if item["choices"]:
                        chunks.append(item)
                    else:
                        usage_only = item
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == content
        assert usage_only["usage"]["completion_tokens"] == body["usage"]["completion_tokens"]
        metrics = cluster.http.get("/metrics").text
        assert f'stageplane_requests_total{{model="{cluster.model}",tenant="default",status="completed"}} 3' in metrics
        assert "stageplane_time_to_first_token_seconds_bucket" in metrics
        node = cluster.http.get("/api/state").json()["nodes"][0]
        worker = httpx.get(node["url"] + "/metrics", headers=cluster.http.headers, trust_env=False).text
        assert "stageplane_worker_calls" in worker
        assert httpx.get(node["url"] + "/ready", trust_env=False).json()["ready"] is True
        # Graceful shutdown deregisters immediately rather than after the heartbeat timeout.
        victim = node["id"]
        cluster.node_processes[victim].terminate()
        cluster.node_processes[victim].wait(timeout=15)
        state = cluster.http.get("/api/state").json()
        assert not next(n for n in state["nodes"] if n["id"] == victim)["active"]
        assert any(e["kind"] == "node_left" and e["node"] == victim for e in state["events"])
    finally:
        cluster.close()


async def test_tenant_quota_holds_while_waiting_for_admission(executor, monkeypatch):
    c, client = executor
    model = next(iter(c.models))
    c.tenant_limit = 1
    c.admission = asyncio.Semaphore(0)
    monkeypatch.setenv("STAGEPLANE_ADMISSION_WAIT", "0.3")
    monkeypatch.setattr(c, "get_route", lambda *a, **k: [])
    body = {"model": model, "messages": [{"role": "user", "content": "x"}]}
    first = asyncio.create_task(client.post("/v1/chat/completions", json=body))
    await asyncio.sleep(0.05)
    # The waiting request already counts against the tenant quota.
    second = await client.post("/v1/chat/completions", json=body)
    assert second.json()["detail"] == "Tenant concurrent request quota exhausted"
    assert (await first).json()["detail"] == "Inference admission capacity exhausted"
    assert c.tenant_inflight == {}


async def test_failed_readiness_fixture_returns_streamed_budget(tmp_path, monkeypatch):
    from stageplane.common import NodeInfo
    from stageplane.node import Worker
    make(tmp_path / "src")
    m = distribute(str(tmp_path / "src"), tmp_path / "models", layers_per_chunk=2)
    worker = Worker(NodeInfo(id="n", url="http://n", streaming=True, layer_cache_bytes=10**9),
                    "http://executor", tmp_path / "cache", stream_weights=True)
    worker.manifests[m["key"]] = m

    async def present(*args, **kwargs):
        return None
    monkeypatch.setattr(worker, "ensure_file", present)
    worker.cache = tmp_path / "models"
    real_isfinite = torch.isfinite
    monkeypatch.setattr(torch, "isfinite", lambda t: real_isfinite(t) & False)
    try:
        await worker.reconcile([{"model": m["key"], "chunk": "blocks-000"}])
        assert "Readiness fixture failed" in worker.errors[m["key"] + ":blocks-000"]
        assert worker.layer_budget.used == 0
    finally:
        await worker.transport.close()
        await worker.artifacts.close()


def test_placement_reserves_streaming_layer_cache():
    from stageplane.placement import plan
    reserve = 64 * 1024**2
    stages = [{"id": "embedding", "kind": "embedding", "start": 0, "end": 0, "weight_bytes": 100}]
    edge = {"id": "edge", "memory_bytes": reserve + 250, "failure_domain": "edge", "streaming": True,
            "layer_cache_bytes": 100, "metrics": {"available_storage_bytes": 10**9}}
    # 200 bytes of float32 embedding fit in 250, but not alongside a 100-byte pinned layer cache.
    assert plan({"key": "m", "stages": stages}, [edge])["embedding"] == []
    assert plan({"key": "m", "stages": stages}, [{**edge, "layer_cache_bytes": 0}])["embedding"] == ["edge"]

import time
from types import SimpleNamespace

import pytest
import torch

from stageplane.common import ChatRequest, NodeInfo
from stageplane.executor import Coordinator
from stageplane.placement import bucket, plan, route
from stageplane.wire import pack, unpack


@pytest.fixture
async def coordinator(tmp_path):
    c = Coordinator(tmp_path / "models")
    try:
        yield c
    finally:
        c.db.close()
        await c.transport.close()


def manifest(key="model", weight=400_000_000):
    return {"key": key, "source": key, "revision": "abc123", "architecture": "qwen3",
            "manifest_sha256": "0" * 64, "config": {"hidden_size": 4}, "artifacts": {},
            "stages": [{"id": "blocks-000", "weight_bytes": weight}]}


def node(nid, group=0, free=1_000_000_000):
    return NodeInfo(id=nid, url="http://" + nid, group=group, memory_bytes=2_000_000_000,
                    metrics={"available_memory_bytes": free, "available_cpu_cores": 1})


async def test_deployments_share_one_resource_budget(coordinator):
    c = coordinator
    c.register(NodeInfo(id="n1", url="http://n1", memory_bytes=1_000_000_000))
    for key in ("first", "second"):
        c.models[key] = manifest(key)
        c.deployments[key] = {"replicas": 1, "groups": 1, "placements": {}}
    c.rebalance()
    assert c.deployments["first"]["capacity"]["status"] == "planned"
    assert c.deployments["second"]["capacity"]["status"] == "insufficient_capacity"
    assert {item["model"] for item in c.assignments("n1")} == {"first"}
    c.register(NodeInfo(id="n2", url="http://n2", memory_bytes=1_000_000_000))
    assert all(d["capacity"]["status"] == "planned" for d in c.deployments.values())
    for nid in c.nodes:
        assert len(c.assignments(nid)) == 1


def test_changed_groups_use_the_same_modulo_as_worker_eligibility():
    sid = next(f"stage-{i}" for i in range(20) if bucket(f"stage-{i}", 2) == 1)
    m = {"stages": [{"id": sid, "weight_bytes": 1000}]}
    nodes = [{"id": "old", "group": 5, "memory_bytes": 10**9},
             {"id": "new", "group": 7, "memory_bytes": 10**9, "speed": 10**9}]
    assert plan(m, nodes, groups=2, previous={sid: ["old"]}, changed_groups={7})[sid] == ["new"]


async def test_group_change_rebalances_both_old_and_new_groups(coordinator, monkeypatch):
    c = coordinator
    c.register(node("moving", 0))
    calls = []
    monkeypatch.setattr(c, "rebalance", lambda groups=None: calls.append(groups))
    c.register(node("moving", 5))
    assert calls == [{0, 5}]


async def test_gradual_memory_pressure_is_compared_to_last_plan(coordinator, monkeypatch):
    c = coordinator
    monkeypatch.setenv("STAGEPLANE_REBALANCE_INTERVAL", "0")
    c.register(node("gradual"))
    calls = []
    original = c.rebalance
    def rebalance(groups=None):
        calls.append(groups)
        return original(groups)
    monkeypatch.setattr(c, "rebalance", rebalance)
    c.register(node("gradual", free=900_000_000))
    c.register(node("gradual", free=800_000_000))
    assert calls == []
    c.register(node("gradual", free=700_000_000))
    assert calls == [{0}]


async def test_rebalance_cooldowns_are_independent_between_groups(coordinator, monkeypatch):
    c = coordinator
    c.deployments["not-loaded"] = {"groups": 2, "capacity": {"status": "planned"}}
    c.register(node("first", 0))
    c.register(node("second", 1))
    now = time.monotonic()
    c.last_rebalance = now
    c.group_rebalances = {(2, 0): now - 31, (2, 1): now}
    calls = []
    monkeypatch.setattr(c, "rebalance", lambda groups=None: calls.append(groups))
    c.register(node("first", 0, free=500_000_000))
    assert calls == [{0}]


def test_ready_artifacts_do_not_consume_free_disk_twice():
    m = manifest(weight=100_000_000)
    n = {"id": "cached", "memory_bytes": 10**9, "ready": ["model:blocks-000"],
         "metrics": {"available_storage_bytes": 0}}
    assert plan(m, [n])["blocks-000"] == ["cached"]
    n["ready"] = []
    assert plan(m, [n])["blocks-000"] == []


async def test_local_request_reservations_avoid_saturated_nodes(coordinator):
    c = coordinator
    c.models["model"] = manifest()
    c.deployments["model"] = {"replicas": 2, "groups": 1,
                               "placements": {"blocks-000": ["busy", "spare"]}}
    for nid in ("busy", "spare"):
        c.nodes[nid] = {"id": nid, "url": "http://" + nid, "active": True,
                        "ready": ["model:blocks-000"], "metrics": {"sessions": 0, "max_sessions": 1}}
    c.active_requests["first"] = {"route": [{"node": "busy"}]}
    assert c.get_route("model")[0]["node"] == "spare"
    c.nodes["spare"]["draining"] = True
    with pytest.raises(RuntimeError, match="No ready replica"):
        c.get_route("model")


async def test_live_requests_are_never_evicted_by_history_limit(coordinator):
    c = coordinator
    for index in range(150):
        c.requests.append({"id": f"old-{index}", "status": "completed"})
        c.active_requests[str(index)] = {"id": str(index), "status": "running", "started": time.time() - 2}
    state = c.state()
    assert len(c.requests) == 100
    assert state["summary"]["active_requests"] == 150
    assert sum(record["status"] == "running" for record in state["requests"]) == 150
    assert all(record["elapsed_ms"] >= 2000 for record in state["requests"] if record["status"] == "running")


async def test_route_startup_failure_records_terminal_state(coordinator):
    c = coordinator
    c.models["model"] = manifest()
    c.tokenizers["model"] = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1])
    request = ChatRequest(model="model", messages=[{"role": "user", "content": "hello"}], max_tokens=1)
    with pytest.raises(RuntimeError, match="no deployment"):
        async for _ in c.generate(request, tenant="tenant-a"):
            pass
    assert not c.active_requests
    assert c.requests[-1]["status"] == "failed"
    assert c.requests[-1]["tenant"] == "tenant-a"
    assert c.requests[-1]["finished"] >= c.requests[-1]["started"]
    assert c.state()["summary"]["failed_requests"] == 1


async def test_closed_stream_cancels_request_and_releases_tenant_cache(coordinator, monkeypatch):
    c = coordinator
    c.models["model"] = manifest()
    c.tokenizers["model"] = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1],
                                           decode=lambda values, **kw: "a" * len(values), eos_token_id=0)
    path = [{"node": "n", "chunk": "blocks-000", "url": "http://n"}]
    monkeypatch.setattr(c, "get_route", lambda model: path)
    async def forward(hop, frame):
        meta, _ = unpack(frame)
        meta.update(index=1, trace=[])
        return pack(meta, torch.tensor([[[0., 0., 1., 0.]]]))
    released = []
    async def cleanup(tenant, session, used):
        released.append((tenant, session, set(used)))
    monkeypatch.setattr(c.transport, "send", forward)
    monkeypatch.setattr(c, "cleanup", cleanup)
    request = ChatRequest(model="model", messages=[{"role": "user", "content": "hello"}], max_tokens=3)
    generation = c.generate(request, tenant="tenant-a")
    await anext(generation)
    assert len(c.active_requests) == 1
    await generation.aclose()
    assert c.requests[-1]["status"] == "cancelled"
    assert not c.active_requests
    assert released[0][0] == "tenant-a" and released[0][2] == {"n"}
    assert c.request_totals["cancelled"] == 1


async def test_concurrent_requests_reserve_distinct_available_worker_slots(coordinator, monkeypatch):
    import asyncio
    c = coordinator
    c.models["model"] = manifest()
    c.deployments["model"] = {"replicas": 2, "groups": 1, "placements": {"blocks-000": ["a", "b"]}}
    for nid in ("a", "b"):
        c.nodes[nid] = {"id": nid, "url": "http://" + nid, "active": True,
                        "ready": ["model:blocks-000"], "metrics": {"max_sessions": 1}}
    c.tokenizers["model"] = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1],
                                           decode=lambda values, **kw: "a" * len(values), eos_token_id=0)
    calls = []
    async def forward(hop, frame):
        calls.append(hop["node"])
        await asyncio.sleep(.01)
        meta, _ = unpack(frame)
        meta.update(index=1, trace=[])
        return pack(meta, torch.tensor([[[0., 0., 1., 0.]]]))
    async def cleanup(*args):
        pass
    monkeypatch.setattr(c.transport, "send", forward)
    monkeypatch.setattr(c, "cleanup", cleanup)
    request = ChatRequest(model="model", messages=[{"role": "user", "content": "hello"}], max_tokens=2)
    streams = [c.generate(request, tenant=f"tenant-{i}") for i in range(2)]
    try:
        await asyncio.gather(*(anext(stream) for stream in streams))
        assert sorted(calls) == ["a", "b"]
        with pytest.raises(RuntimeError, match="No ready replica"):
            await anext(c.generate(request, tenant="third"))
    finally:
        await asyncio.gather(*(stream.aclose() for stream in streams))
    assert not c.active_requests


async def test_dead_node_retention_preserves_active_route_references(coordinator, monkeypatch):
    c = coordinator
    monkeypatch.setenv("STAGEPLANE_NODE_RETENTION_SECONDS", "60")
    c.register(node("retained"))
    c.register(node("expired"))
    c.active_requests["session"] = {"route": [{"node": "retained"}]}
    c.sweep_nodes(now=time.time() + 100)
    assert set(c.nodes) == {"retained"}
    assert not c.nodes["retained"]["active"]
    c.active_requests.clear()
    c.sweep_nodes(now=time.time() + 101)
    assert not c.nodes
    assert not c.resource_baselines


@pytest.mark.parametrize("metrics", [
    {"available_cpu_cores": "busy"}, {"available_memory_bytes": -1},
    {"available_cpu_cores": float("nan")}, {"rss_bytes": float("inf")},
    {"stage_ms": {"block": "fast"}},
])
async def test_invalid_telemetry_is_rejected_without_poisoning_registry(coordinator, metrics):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        coordinator.register(NodeInfo(id="invalid", url="http://invalid", metrics=metrics))
    assert error.value.status_code == 422
    assert not coordinator.nodes


async def test_scale_out_keeps_old_route_until_replacement_ready_and_sessions_release(coordinator):
    c = coordinator
    m = manifest(weight=100_000_000)
    c.models["model"] = m
    c.register(NodeInfo(id="old", url="http://old", memory_bytes=10**9, ready=["model:blocks-000"]))
    c.deployments["model"] = {"replicas": 1, "groups": 1, "placements": {"blocks-000": ["old"]}}
    c.rebalance()
    original = c.get_route("model")
    c.active_requests["existing"] = {"model": "model", "route": original}
    c.register(NodeInfo(id="new", url="http://new", memory_bytes=10**9, speed=10**9))
    assert c.deployments["model"]["placements"]["blocks-000"] == ["new"]
    assert c.deployments["model"]["retiring_placements"] == {"blocks-000": ["old"]}
    assert c.get_route("model")[0]["node"] == "old"
    assert c.assignments("old") and c.assignments("new")
    assert c.deployment_status("model", c.deployments["model"])["readiness"] == "rebalancing"
    c.register(NodeInfo(id="new", url="http://new", memory_bytes=10**9, speed=10**9, ready=["model:blocks-000"]))
    assert c.assignments("old"), "A live request still pins its old route"
    c.active_requests.clear()
    c.settle_transitions()
    assert not c.assignments("old")
    assert c.get_route("model")[0]["node"] == "new"
    assert c.deployment_status("model", c.deployments["model"])["readiness"] == "ready"


async def test_pending_placement_capacity_does_not_withdraw_complete_old_route(coordinator):
    c = coordinator
    sid = next(f"stage-{i}" for i in range(20) if bucket(f"stage-{i}", 2) == 1)
    m = manifest(weight=100_000_000)
    m["stages"][0]["id"] = sid
    c.models["model"] = m
    c.register(NodeInfo(id="old", url="http://old", group=0, memory_bytes=10**9,
                        ready=["model:" + sid]))
    c.deployments["model"] = {"replicas": 1, "groups": 1, "placements": {sid: ["old"]}}
    c.rebalance()
    c.deployments["model"]["groups"] = 2
    c.rebalance()
    assert c.deployments["model"]["capacity"]["status"] == "insufficient_capacity"
    assert c.get_route("model")[0]["node"] == "old"
    assert c.assignments("old")
    status = c.deployment_status("model", c.deployments["model"])
    assert status["serving"] and status["readiness"] == "rebalancing"


def test_make_before_break_accounts_for_both_resident_and_incoming_stages():
    m = {"key": "model", "stages": [{"id": "a", "weight_bytes": 300_000_000},
                                    {"id": "b", "weight_bytes": 300_000_000}]}
    n = {"id": "only", "memory_bytes": 1_000_000_000, "ready": ["model:a"]}
    placements = plan(m, [n], resident={"only": {"a"}})
    assert placements["a"] == ["only"]
    assert placements["b"] == [], "600MB resident plus 600MB incoming must not fit a 1GB node"


async def test_recovery_can_reuse_its_existing_saturated_worker_session(coordinator):
    c = coordinator
    c.models["model"] = manifest()
    c.deployments["model"] = {"replicas": 1, "groups": 1, "placements": {"blocks-000": ["busy"]}}
    c.nodes["busy"] = {"id": "busy", "url": "http://busy", "active": True,
                        "ready": ["model:blocks-000"], "metrics": {"sessions": 1, "max_sessions": 1}}
    with pytest.raises(RuntimeError, match="No ready replica"):
        c.get_route("model")
    assert (await c.ready_route("model", timeout=0, reuse_nodes={"busy"}))[0]["node"] == "busy"
    # The credit belongs only to that existing session; new requests still fail admission.
    with pytest.raises(RuntimeError, match="No ready replica"):
        c.get_route("model")


async def test_repeated_cancellation_does_not_cancel_detached_cache_cleanup(coordinator, monkeypatch):
    import asyncio
    c = coordinator
    c.models["model"] = manifest()
    c.tokenizers["model"] = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1],
                                           decode=lambda values, **kw: "a" * len(values), eos_token_id=0)
    monkeypatch.setattr(c, "get_route", lambda model: [{"node": "n", "chunk": "blocks-000", "url": "http://n"}])
    async def forward(hop, frame):
        meta, _ = unpack(frame)
        meta.update(index=1, trace=[])
        return pack(meta, torch.tensor([[[0., 0., 1., 0.]]]))
    started, completed = asyncio.Event(), asyncio.Event()
    async def cleanup(*args):
        started.set()
        await asyncio.sleep(.03)
        completed.set()
    monkeypatch.setattr(c.transport, "send", forward)
    monkeypatch.setattr(c, "cleanup", cleanup)
    request = ChatRequest(model="model", messages=[{"role": "user", "content": "hello"}], max_tokens=3)
    stream = c.generate(request)
    await anext(stream)
    closing = asyncio.create_task(stream.aclose())
    await started.wait()
    closing.cancel()
    await asyncio.gather(closing, return_exceptions=True)
    await asyncio.gather(*c.cleanup_tasks)
    assert completed.is_set()
    assert c.requests[-1]["status"] == "cancelled"


def test_http_stream_disconnect_releases_all_worker_caches(tmp_path):
    import json
    from scripts.cluster import Cluster
    from scripts.tiny_checkpoint import make
    from stageplane.distribution import distribute
    make(tmp_path / "source")
    distribute(str(tmp_path / "source"), tmp_path / "models", layers_per_chunk=1)
    cluster = Cluster(tmp_path / "models", tmp_path / "run", nodes=4)
    try:
        cluster.start()
        session = None
        with cluster.http.stream("POST", "/v1/chat/completions", json={
            "model": cluster.model, "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": 1024, "temperature": 0, "stream": True,
        }) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    assert "error" not in data
                    session = data["id"]
                    break
        assert session is not None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = cluster.http.get("/api/state").json()
            record = next((r for r in state["requests"] if r["id"] == session), None)
            if record and record["status"] == "cancelled" and all(n["metrics"].get("sessions", 0) == 0 for n in state["nodes"]):
                break
            time.sleep(.1)
        else:
            pytest.fail("Disconnected request retained worker session caches: " + json.dumps(state["requests"]))
        assert state["summary"]["active_requests"] == 0
    finally:
        cluster.close()


async def test_resume_restores_fresh_workers_but_not_stale_ones(coordinator):
    import httpx
    from stageplane.executor import create_app
    c = coordinator
    c.models["model"] = manifest(weight=1000)
    c.register(NodeInfo(id="n", url="http://n", memory_bytes=10**9, ready=["model:blocks-000"]))
    c.deployments["model"] = {"replicas": 1, "groups": 1, "placements": {"blocks-000": ["n"]}}
    c.rebalance()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(c)), base_url="http://executor") as client:
        assert (await client.post("/api/nodes/n/drain")).status_code == 200
        with pytest.raises(RuntimeError):
            c.get_route("model")
        response = await client.post("/api/nodes/n/resume")
        assert response.status_code == 200 and response.json() == {"draining": False}
        assert c.get_route("model")[0]["node"] == "n"
        c.nodes["n"]["last_seen"] = time.time() - 60
        assert (await client.post("/api/nodes/n/resume")).status_code == 200
        assert not c.nodes["n"]["active"]
        with pytest.raises(RuntimeError):
            c.get_route("model")


async def test_join_rebalances_busy_workers_without_unloading_live_copies(coordinator):
    c = coordinator
    c.models["model"] = manifest(weight=100_000_000)
    c.register(NodeInfo(id="old", url="http://old", memory_bytes=10**9,
                        ready=["model:blocks-000"], metrics={"sessions": 1}))
    c.deployments["model"] = {"replicas": 1, "groups": 1, "placements": {"blocks-000": ["old"]}}
    c.rebalance()
    c.register(NodeInfo(id="new", url="http://new", memory_bytes=10**9, speed=10**9))
    assert c.deployments["model"]["placements"]["blocks-000"] == ["new"]
    assert c.assignments("old") and c.get_route("model")[0]["node"] == "old"
    c.register(NodeInfo(id="new", url="http://new", memory_bytes=10**9, speed=10**9,
                        ready=["model:blocks-000"]))
    assert c.assignments("old"), "Worker still reports a live session"
    c.register(NodeInfo(id="old", url="http://old", memory_bytes=10**9,
                        ready=["model:blocks-000"], metrics={"sessions": 0}))
    assert not c.assignments("old")
    assert c.get_route("model")[0]["node"] == "new"


async def test_loading_join_replans_once_after_settling_without_resource_change(coordinator):
    c = coordinator
    c.models["model"] = manifest(weight=100_000_000)
    c.register(NodeInfo(id="old", url="http://old", memory_bytes=10**9,
                        ready=["model:blocks-000"], metrics={"loading": True}))
    c.deployments["model"] = {"replicas": 1, "groups": 1, "placements": {"blocks-000": ["old"]}}
    c.rebalance()
    c.register(NodeInfo(id="new", url="http://new", memory_bytes=10**9, speed=10**9))
    assert c.deployments["model"]["placements"]["blocks-000"] == ["old"]
    assert c.pending_rebalance_groups == {0}
    before = c.rebalance_count
    c.register(NodeInfo(id="old", url="http://old", memory_bytes=10**9,
                        ready=["model:blocks-000"], metrics={"loading": False}))
    assert c.rebalance_count == before + 1
    assert c.deployments["model"]["placements"]["blocks-000"] == ["new"]
    assert not c.pending_rebalance_groups
    c.register(NodeInfo(id="old", url="http://old", memory_bytes=10**9,
                        ready=["model:blocks-000"], metrics={"loading": False}))
    assert c.rebalance_count == before + 1

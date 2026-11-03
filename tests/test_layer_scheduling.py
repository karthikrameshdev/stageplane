import torch

from scripts.cluster import Cluster
from scripts.tiny_checkpoint import make
from stageplane.distribution import distribute
from stageplane.placement import plan
from stageplane.resources import capacity
from stageplane.runtime import LocalPipeline, Stage


def manifest(layers=24, weight=100):
    return {"key": "m", "stages": [{"id": f"blocks-{i:03d}", "kind": "blocks", "start": i, "end": i + 1,
                                    "weight_bytes": weight} for i in range(layers)]}


def node(name, budget, **extra):
    return {"id": name, "memory_bytes": budget, "failure_domain": name, **extra}


def test_interleaved_layers_scale_with_node_size():
    m = manifest()
    # Per layer cost is 200 bytes in float32. Budgets leave room for exactly 1, 8 and 15 layers plus reserve.
    reserve = 64 * 1024**2
    nodes = [node("tiny", reserve + 200), node("mid", reserve + 1600), node("big", reserve + 3000)]
    placed = plan(m, nodes, strategy="capacity")
    counts = {n["id"]: sum(n["id"] in v for v in placed.values()) for n in nodes}
    assert sum(counts.values()) == 24 and all(len(v) == 1 for v in placed.values())
    assert counts["tiny"] <= 1 and counts["mid"] <= 8 and counts["big"] <= 15
    assert counts["tiny"] >= 1 and counts["big"] > counts["mid"] > counts["tiny"]
    # Non contiguous ownership: the big node owns layers separated by other nodes' layers.
    owned = sorted(int(s.split("-")[1]) for s, v in placed.items() if "big" in v)
    assert any(b - a > 1 for a, b in zip(owned, owned[1:]))


def test_streaming_node_hosts_layers_without_resident_memory():
    m = manifest(weight=10**9)
    reserve = 64 * 1024**2
    small = node("edge", reserve + 3 * 10**9, streaming=True, metrics={"available_storage_bytes": 10**12})
    assert not all(plan(m, [{**small, "streaming": False}]).values())
    placed = plan(m, [small], strategy="capacity")
    assert all(v == ["edge"] for v in placed.values())
    assert capacity(m, placed, [small])["status"] == "planned"


def test_streaming_requires_disk_for_all_layers():
    m = manifest(layers=4, weight=10**9)
    small = node("edge", 3 * 10**9, streaming=True, metrics={"available_storage_bytes": 2 * 10**9})
    placed = plan(m, [small])
    assert sum(bool(v) for v in placed.values()) == 2


def test_streamed_stage_matches_resident_stage(tmp_path):
    make(tmp_path / "src")
    m = distribute(str(tmp_path / "src"), tmp_path / "models", layers_per_chunk=2)
    directory = tmp_path / "models" / m["key"]
    resident = LocalPipeline(m, directory)
    streamed = [Stage(m, s, directory, stream_budget=1 if s["kind"] == "blocks" else None) for s in m["stages"]]
    states = [s.state() for s in streamed]
    tokens = torch.tensor([[1, 5, 9, 3]])
    expected = resident.forward(tokens)
    value = tokens
    for stage, state in zip(streamed, states):
        value = stage(value, state, 0)
    assert torch.equal(value, expected)
    nxt = torch.tensor([[7]])
    expected2 = resident.forward(nxt, 4)
    value = nxt
    for stage, state in zip(streamed, states):
        value = stage(value, state, 4)
    assert torch.equal(value, expected2)
    blocks = [s for s in streamed if s.streamer]
    # A one-byte budget pins nothing: every layer is read from disk on every pass.
    assert all(s.streamer.stats["transient"] == s.streamer.stats["misses"] > 0 and s.resident_bytes() == 0 for s in blocks)


def test_shared_budget_pins_layers_until_full(tmp_path):
    from stageplane.runtime import LayerBudget
    make(tmp_path / "src")
    m = distribute(str(tmp_path / "src"), tmp_path / "models", layers_per_chunk=2)
    directory = tmp_path / "models" / m["key"]
    blocks = [s for s in m["stages"] if s["kind"] == "blocks"]
    probe = Stage(m, blocks[0], directory, stream_budget=10**9)
    probe(torch.zeros(1, 1, 32), probe.state(), 0)
    per_layer = probe.resident_bytes() // 2
    probe.close()
    budget = LayerBudget(per_layer * 3)
    stages = [Stage(m, s, directory, stream_budget=budget) for s in blocks]
    for _ in range(3):
        for stage in stages:
            stage(torch.zeros(1, 1, 32), stage.state(), 0)
    assert budget.used == per_layer * 3
    hits = sum(s.streamer.stats["hits"] for s in stages)
    misses = sum(s.streamer.stats["misses"] for s in stages)
    # Three of four layers pinned: after the first pass, 3/4 of layer reads are RAM hits.
    assert hits == 6 and misses == 6
    for stage in stages:
        stage.close()
    assert budget.used == 0


def chat(cluster, content="hello world"):
    r = cluster.http.post("/v1/chat/completions", json={"model": cluster.model, "temperature": 0, "max_tokens": 8,
                          "messages": [{"role": "user", "content": content}]})
    assert r.status_code == 200, r.text
    return r.json()["choices"][0]["message"]["content"]


def run(tmp_path, name, **kwargs):
    c = Cluster(tmp_path / "models", tmp_path / name, **kwargs)
    try:
        c.start()
        c.wait_ready()
        text = chat(c)
        state = c.http.get("/api/state").json()
        return text, state
    finally:
        c.close()


def test_streaming_node_is_overflow_capacity():
    m = manifest(layers=6, weight=100)
    reserve = 64 * 1024**2
    big = node("big", reserve + 800)
    edge = node("edge", reserve + 400, streaming=True, metrics={"available_storage_bytes": 10**9})
    placed = plan(m, [big, edge], strategy="capacity")
    assert sum("big" in v for v in placed.values()) == 4 and sum("edge" in v for v in placed.values()) == 2


def test_streaming_nodes_share_layers_by_memory_not_shared_disk():
    m = manifest(layers=12, weight=100)
    disk = {"available_storage_bytes": 10**12}
    small = node("small", 10**8, streaming=True, metrics=disk)
    large = node("large", 3 * 10**8, streaming=True, metrics=disk)
    placed = plan(m, [small, large], strategy="capacity")
    counts = {n: sum(n in v for v in placed.values()) for n in ("small", "large")}
    assert counts == {"small": 3, "large": 9}


def test_hub_interleaved_cluster_matches_chain_single_node(tmp_path):
    make(tmp_path / "src")
    distribute(str(tmp_path / "src"), tmp_path / "models", layers_per_chunk=1)
    baseline, _ = run(tmp_path, "base", nodes=1, transports=["rest"], deploy_options={"routing": "chain"})
    text, state = run(tmp_path, "hub", nodes=3, transports=["rest", "grpc", "websocket"],
                      deploy_options={"routing": "hub", "strategy": "capacity"})
    assert text == baseline
    nodes = [h["node"] for h in state["requests"][-1]["route"]]
    assert len(set(nodes)) == 3
    # Interleaved placement: a node reappears after another node has run in between.
    assert any(nodes[i] != nodes[i + 1] and nodes[i] in nodes[i + 2:] for i in range(len(nodes) - 2)), nodes


def test_streaming_cluster_node_matches_resident(tmp_path):
    make(tmp_path / "src")
    distribute(str(tmp_path / "src"), tmp_path / "models", layers_per_chunk=1)
    baseline, _ = run(tmp_path, "base", nodes=1, transports=["rest"])
    text, state = run(tmp_path, "stream", nodes=1, transports=["rest"],
                      node_args={0: ["--stream-weights", "--layer-cache", "1MiB"]})
    assert text == baseline
    metrics = state["nodes"][0]["metrics"]
    assert metrics["streaming"] is True and metrics["layer_stream"]["misses"] > 0
    assert metrics["loaded_weight_bytes"] < 1024**2

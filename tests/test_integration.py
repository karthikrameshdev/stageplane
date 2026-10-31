import json
import time

import httpx
import pytest

from scripts.cluster import Cluster
from scripts.tiny_checkpoint import make
from stageplane.distribution import distribute


def payload(model, stream=False):
    return {"model": model, "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": 8, "temperature": 0, "stream": stream}


@pytest.mark.parametrize("architecture", ["qwen3", "qwen3_5_text"])
def test_four_nodes_mixed_transports_recovery(tmp_path, architecture):
    make(tmp_path / "source", architecture)
    m = distribute(str(tmp_path / "source"), tmp_path / "models", layers_per_chunk=1)
    cluster = Cluster(tmp_path / "models", tmp_path / "run", replicas=2)
    try:
        cluster.start()
        assert httpx.get(cluster.url + "/api/state", trust_env=False).status_code == 401
        assert cluster.http.get("/health").json() == {"status": "ok"}
        assert cluster.http.post("/v1/chat/completions", json={**payload(cluster.model), "model": "missing"}).status_code == 404
        first = cluster.http.post("/v1/chat/completions", json=payload(cluster.model))
        assert first.status_code == 200, first.text
        expected = first.json()["choices"][0]["message"]["content"]
        output, killed = "", False
        with cluster.http.stream("POST", "/v1/chat/completions", json=payload(cluster.model, True)) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                item = json.loads(line[6:])
                assert "error" not in item, item
                output += item["choices"][0]["delta"].get("content", "")
                if not killed:
                    state = cluster.http.get("/api/state").json()
                    event = next(e for e in reversed(state["events"]) if e["kind"] == "request_started")
                    # Remove a node on the active route after output commitment.
                    victim = event["route"][len(event["route"]) // 2]["node"]
                    cluster.kill_node(victim)
                    killed = True
        assert output == expected
        state = cluster.http.get("/api/state").json()
        record = state["requests"][-1]
        assert record["recoveries"] >= 1
        assert record["status"] == "completed"
        assert any(e["kind"] == "recovery" for e in state["events"])
        # Rejoining node triggers only its placement group; all chunks remain valid.
        cluster.start_node(int(victim.split("-")[-1]))
        cluster.wait_ready()
    finally:
        cluster.close()


def test_single_node_all_chunks(tmp_path):
    make(tmp_path / "source")
    distribute(str(tmp_path / "source"), tmp_path / "models", layers_per_chunk=1)
    c = Cluster(tmp_path / "models", tmp_path / "run", nodes=1, transports=["rest"])
    try:
        c.start()
        state = c.wait_ready()
        assert len(state["nodes"][0]["ready"]) == 6
        result = c.http.post("/v1/chat/completions", json=payload(c.model))
        assert result.status_code == 200, result.text
        assert result.json()["usage"]["completion_tokens"] > 0
    finally:
        c.close()

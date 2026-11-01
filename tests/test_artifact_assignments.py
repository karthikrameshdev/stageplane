import sys

import httpx
import pytest

from stageplane import cli, executor
from stageplane.common import NodeInfo
from stageplane.executor import Coordinator, create_app


@pytest.fixture
async def coordinators(tmp_path, monkeypatch):
    monkeypatch.delenv("STAGEPLANE_API_KEYS_FILE", raising=False)
    monkeypatch.delenv("STAGEPLANE_STATE_FILE", raising=False)
    monkeypatch.setenv("STAGEPLANE_TOKEN", "artifact-test-token")
    instances = []

    def make(**kwargs):
        c = Coordinator(tmp_path / str(len(instances)) / "models", **kwargs)
        instances.append(c)
        return c

    yield make
    for c in instances:
        c.db.close()
        await c.transport.close()


def install_model(c):
    c.models["model"] = {
        "key": "model", "source": "model", "revision": "immutable-revision",
        "architecture": "qwen3", "manifest_sha256": "a" * 64,
        "config": {"hidden_size": 4}, "artifacts": {},
        "stages": [{"id": "blocks-000", "weight_bytes": 1000}],
    }
    c.deployments["model"] = {"replicas": 1, "groups": 1, "placements": {}}


@pytest.mark.parametrize("configured", [None, [], ["https://mirror.example/models/", "https://backup.example/models"]])
async def test_registration_and_heartbeat_advertise_sources(coordinators, configured):
    c = coordinators(artifact_sources=configured)
    install_model(c)
    expected = ([str(c.store)] if configured is None else
                [source.rstrip("/") for source in configured])
    info = NodeInfo(id="worker", url="http://worker", memory_bytes=2_000_000_000)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(c)),
                                base_url="http://executor",
                                headers={"Authorization": "Bearer artifact-test-token"}) as client:
        for endpoint in ("/api/nodes/register", "/api/nodes/worker/heartbeat"):
            response = await client.post(endpoint, json=info.model_dump())
            assert response.status_code == 200, response.text
            assert response.json()["assignments"] == [{
                "model": "model", "chunk": "blocks-000", "manifest_sha256": "a" * 64,
                "artifact_sources": expected,
            }]
    assert "artifact_sources" not in c.models["model"]
    assert c.models["model"]["manifest_sha256"] == "a" * 64


async def test_default_store_can_start_empty_and_later_supply_assignments(coordinators):
    c = coordinators()
    assert c.models == {}
    assert c.artifact_sources == [str(c.store.resolve())]
    install_model(c)
    result = c.register(NodeInfo(id="worker", url="http://worker"))
    assert result["assignments"][0]["artifact_sources"] == [str(c.store)]


async def test_source_order_normalization_and_response_mutations_are_isolated(coordinators, tmp_path):
    local = tmp_path / "shared models"
    configured = [local.as_uri(), "https://MIRROR.example/models/", str(local)]
    c = coordinators(artifact_sources=configured)
    expected = [str(local), "https://mirror.example/models"]
    configured.append("https://unexpected.example")
    install_model(c)
    response = c.register(NodeInfo(id="worker", url="http://worker"))
    assert response["assignments"][0]["artifact_sources"] == expected
    response["assignments"][0]["artifact_sources"].clear()
    assert c.assignments("worker")[0]["artifact_sources"] == expected
    assert c.artifact_sources == expected


@pytest.mark.parametrize("sources", [
    ["relative/path"], ["https://user:secret@example.com/models"],
    ["/" + "x" * 2048], [f"https://mirror{i}.example" for i in range(9)],
])
def test_invalid_sources_rejected_before_store_discovery(tmp_path, monkeypatch, sources):
    def unexpected_discovery(store):
        pytest.fail("Source validation must precede manifest discovery and startup side effects")
    monkeypatch.setattr(executor, "manifests", unexpected_discovery)
    with pytest.raises(ValueError):
        Coordinator(tmp_path / "not-created", artifact_sources=sources)
    assert not (tmp_path / "not-created").exists()


async def test_eight_advertised_sources_are_supported(coordinators):
    sources = [f"https://mirror{i}.example/models" for i in range(8)]
    c = coordinators(artifact_sources=sources)
    install_model(c)
    assert c.register(NodeInfo(id="worker", url="http://worker"))["assignments"][0]["artifact_sources"] == sources


@pytest.mark.parametrize("command", ["executor", "node"])
@pytest.mark.parametrize("explicit", [False, True])
def test_cli_passes_normalized_sources_and_preserves_omitted_none(tmp_path, monkeypatch, command, explicit):
    import torch
    import uvicorn
    from stageplane import node, resources

    monkeypatch.delenv("STAGEPLANE_TLS_CERT", raising=False)
    monkeypatch.setattr(resources, "resolve_limits", lambda *args: (1_000_000_000, 1.0))
    monkeypatch.setattr(resources, "available_limits", lambda: (1_000_000_000, 1.0))
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    captured = {}
    instance = object()
    app = object()

    def construct(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return instance

    def build_app(candidate, *args):
        assert candidate is instance
        return app

    def run(candidate, **kwargs):
        assert candidate is app
        captured["served"] = True

    monkeypatch.setattr(executor, "Coordinator", construct)
    monkeypatch.setattr(executor, "create_app", build_app)
    monkeypatch.setattr(node, "Worker", construct)
    monkeypatch.setattr(node, "create_app", build_app)
    monkeypatch.setattr(uvicorn, "run", run)
    args = ["stageplane", command]
    if command == "executor":
        args += ["--store", str(tmp_path / "models")]
    else:
        args += ["--id", "worker", "--cache", str(tmp_path / "cache")]
    sources = [str(tmp_path / "shared"), "https://mirror.example/models"]
    if explicit:
        args += ["--artifact-source", (tmp_path / "shared").as_uri(),
                 "--artifact-source", "https://mirror.example/models/"]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert captured["kwargs"]["artifact_sources"] == (sources if explicit else None)
    assert captured["served"]
    if command == "executor":
        assert captured["args"][0] == tmp_path / "models"
    else:
        assert captured["args"][0].id == "worker"
        assert captured["args"][2] == tmp_path / "cache" / "worker"


@pytest.mark.parametrize("command", ["executor", "node"])
def test_cli_invalid_source_is_parser_error_before_startup(monkeypatch, capsys, command):
    from stageplane import resources
    def unexpected_startup(*args, **kwargs):
        pytest.fail("Invalid sources must fail before starting the executor or resolving worker limits")
    monkeypatch.setattr(executor, "Coordinator", unexpected_startup)
    monkeypatch.setattr(resources, "resolve_limits", unexpected_startup)
    args = ["stageplane", command, "--artifact-source", "relative/path"]
    if command == "node":
        args += ["--id", "worker"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err

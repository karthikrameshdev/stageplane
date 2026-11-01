import json
import shutil
import struct

import httpx
import pytest
from safetensors.torch import load_file, save_file

from scripts.tiny_checkpoint import make
from stageplane.bundles import read_manifest, verify_files
from stageplane.distribution import Source, distribute, manifests


def test_failure_never_publishes_revision_and_retry_has_clean_inventory(tmp_path, monkeypatch):
    make(tmp_path / "source")
    read = Source.read
    def interrupted(self, name, start=None, end=None):
        if name.endswith(".safetensors") and start is not None and start > 8:
            raise OSError("simulated interrupted tensor read")
        return read(self, name, start, end)
    with monkeypatch.context() as patch:
        patch.setattr(Source, "read", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            distribute(str(tmp_path / "source"), tmp_path / "models")
    assert manifests(tmp_path / "models") == {}
    assert list((tmp_path / "models").iterdir()) == []
    result = distribute(str(tmp_path / "source"), tmp_path / "models")
    root = tmp_path / "models" / result["key"]
    verify_files(root, read_manifest(root / "manifest.json"))
    assert not any(name.endswith(".partial") for name in result["files"])


@pytest.mark.parametrize("damage", ["duplicate", "shape", "rank", "dimension", "missing-layer", "truncated"])
def test_malformed_checkpoints_are_rejected_before_publication(tmp_path, damage):
    make(tmp_path / "source")
    source = tmp_path / "source" / "model.safetensors"
    if damage == "duplicate":
        shutil.copyfile(source, source.with_name("duplicate.safetensors"))
    elif damage == "missing-layer":
        weights = {key: value for key, value in load_file(source).items() if ".layers.1." not in key}
        save_file(weights, source)
    elif damage == "truncated":
        source.write_bytes(source.read_bytes()[:-1])
    else:
        data = source.read_bytes()
        size = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8:8 + size])
        tensor = next(value for key, value in header.items() if key != "__metadata__")
        tensor["shape"] = [1] * 33 if damage == "rank" else [2**64] if damage == "dimension" else [999999]
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + size:])
    with pytest.raises(ValueError):
        distribute(str(tmp_path / "source"), tmp_path / "models")
    assert manifests(tmp_path / "models") == {}


def test_hugging_face_credential_is_never_sent_to_arbitrary_sources(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fixture-hf-secret")
    seen = []
    def respond(request):
        seen.append(request)
        if request.url.host == "huggingface.co":
            return httpx.Response(200, json={"sha": "a" * 40, "siblings": []})
        return httpx.Response(200, json={"revision": "a" * 40, "files": []})
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    for source in ["https://models.example/public", "organization/model"]:
        handle = Source(source)
        handle.close()
    assert "authorization" not in seen[0].headers
    assert seen[1].headers["authorization"] == "Bearer fixture-hf-secret"


def test_local_ranges_must_have_exact_declared_length(tmp_path):
    (tmp_path / "short.bin").write_bytes(b"abc")
    source = Source(str(tmp_path))
    try:
        with pytest.raises(ValueError, match="Truncated"):
            source.read("short.bin", 0, 10)
    finally:
        source.close()


@pytest.mark.parametrize("budget", ["headers", "bundle", "storage"])
def test_resource_budgets_reject_before_tensor_downloads(tmp_path, monkeypatch, budget):
    import stageplane.distribution as distribution
    make(tmp_path / "source")
    read, bulk_reads = Source.read, []
    def tracked(self, name, start=None, end=None):
        if name.endswith(".safetensors") and start is not None and start > 8:
            bulk_reads.append((name, start))
        return read(self, name, start, end)
    monkeypatch.setattr(Source, "read", tracked)
    if budget == "headers":
        monkeypatch.setattr(distribution, "MAX_TOTAL_HEADER_BYTES", 1)
    elif budget == "bundle":
        monkeypatch.setattr("stageplane.bundles.MAX_BUNDLE", 1)
    else:
        from types import SimpleNamespace
        monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(ValueError):
        distribute(str(tmp_path / "source"), tmp_path / "models")
    assert not bulk_reads
    assert manifests(tmp_path / "models") == {}
    assert list((tmp_path / "models").iterdir()) == []

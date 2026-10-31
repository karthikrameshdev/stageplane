import asyncio
import functools
import http.server
import json
import subprocess
import threading
import zipfile
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import torch

from scripts.tiny_checkpoint import make
from stageplane.bundles import download, export_bundle, import_bundle, read_manifest
from stageplane.common import NodeInfo
from stageplane.distribution import distribute, rechunk
from stageplane.executor import Coordinator, create_app
from stageplane.resources import MEMORY_CAUTION
from stageplane.runtime import LocalPipeline, load_reference


@contextmanager
def serve(directory):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.fixture
def bundle(tmp_path):
    make(tmp_path / "checkpoint")
    m = distribute(str(tmp_path / "checkpoint"), tmp_path / "producer")
    path = tmp_path / "public" / "model.stageplane"
    export_bundle(tmp_path / "producer" / m["key"], path)
    return m, path


def test_local_http_directory_catalog_and_git_roundtrip(bundle, tmp_path):
    m, archive = bundle
    for source in [str(archive), str(tmp_path / "producer" / m["key"] / "manifest.json"),
                   (tmp_path / "producer" / m["key"]).as_uri()]:
        assert import_bundle(source, tmp_path / "local", m["manifest_sha256"])["key"] == m["key"]
    export_bundle(tmp_path / "producer" / m["key"], archive.parent / "directory", "directory")
    with serve(archive.parent) as url:
        for source in [url + "/model.stageplane", url + "/directory/", url + "/directory/manifest.json"]:
            assert import_bundle(source, tmp_path / "remote")["manifest_sha256"] == m["manifest_sha256"]
        catalog = {"format": "stageplane-catalog-v1", "manifest": url + "/directory/manifest.json",
                   "manifest_sha256": m["manifest_sha256"],
                   "files": {name: url + "/directory/" + name for name in m["files"]}}
        (archive.parent / "catalog.json").write_text(json.dumps(catalog))
        assert import_bundle(url + "/catalog.json", tmp_path / "catalog")["key"] == m["key"]
    repo = archive.parent / "directory"
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "bundle"], check=True, capture_output=True)
    result = import_bundle("git+" + repo.as_uri(), tmp_path / "git")
    assert result["manifest_sha256"] == m["manifest_sha256"]


def test_corruption_bad_pin_and_archive_path_never_publish(bundle, tmp_path):
    m, archive = bundle
    with pytest.raises(ValueError, match="SHA-256"):
        import_bundle(str(archive), tmp_path / "bad-pin", "0" * 64)
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(tmp_path / "corrupt.stageplane", "w") as out:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "embedding.safetensors":
                data = bytes([data[0] ^ 1]) + data[1:]
            out.writestr(item, data)
    with pytest.raises(ValueError, match="integrity"):
        import_bundle(str(tmp_path / "corrupt.stageplane"), tmp_path / "corrupt")
    assert not (tmp_path / "corrupt" / m["key"]).exists()
    with zipfile.ZipFile(tmp_path / "evil.stageplane", "w") as out:
        out.writestr("../outside", "no")
    with pytest.raises(ValueError, match="Unsafe"):
        import_bundle(str(tmp_path / "evil.stageplane"), tmp_path / "unsafe")
    assert not (tmp_path / "outside").exists()


def test_drive_download_adapter_reuses_same_verified_import(bundle, tmp_path, monkeypatch):
    import gdown
    m, archive = bundle
    def fake_download(**kwargs):
        assert "fuzzy" not in kwargs and not kwargs["use_cookies"]
        kwargs["output"].write(archive.read_bytes())
        return kwargs["output"]
    monkeypatch.setattr(gdown, "download", fake_download)
    imported = import_bundle("https://drive.google.com/file/d/public-fixture/view", tmp_path / "drive")
    assert imported["manifest_sha256"] == m["manifest_sha256"]


def test_drive_real_downloader_preserves_verified_import_and_limits(bundle, tmp_path, monkeypatch):
    m, archive = bundle
    # Exercise gdown itself with a loopback HTTP fixture instead of public Drive.
    # A fake gdown.download would miss removed arguments and stream requirements.
    monkeypatch.setattr("stageplane.bundles.is_drive", lambda url: True)
    with serve(archive.parent) as url:
        source = url + "/model.stageplane"
        imported = import_bundle(source, tmp_path / "drive-real")
        assert imported["manifest_sha256"] == m["manifest_sha256"]
        with pytest.raises(ValueError, match="size or disk budget"):
            download(source, tmp_path / "too-large.stageplane", limit=1)
    assert (tmp_path / "too-large.stageplane").stat().st_size == 0


def test_drive_download_failure_remains_an_import_error(tmp_path, monkeypatch):
    import gdown
    def unavailable(**kwargs):
        raise gdown.DownloadError("Download quota exceeded")
    monkeypatch.setattr(gdown, "download", unavailable)
    with pytest.raises(ValueError, match="Public Google Drive download failed"):
        download("https://drive.google.com/file/d/public-fixture/view", tmp_path / "failed.stageplane")


def test_bf16_cached_decode_and_rechunk_parity(bundle, tmp_path):
    torch.set_num_threads(1)
    m, _ = bundle
    m = rechunk(tmp_path / "producer" / m["key"], tmp_path / "single-layer", 1)
    assert len(m["stages"]) == 6
    path = tmp_path / "single-layer" / m["key"]
    read_manifest(path / "manifest.json")
    pipeline = LocalPipeline(m, path, dtype="bfloat16")
    native = load_reference(m, path, dtype="bfloat16")
    cache, tokens = None, torch.tensor([[6, 7, 8]])
    for offset in [0, 3, 4]:
        with torch.inference_mode():
            reference = native(tokens, past_key_values=cache, use_cache=True)
            cache = reference.past_key_values
            actual = pipeline.forward(tokens, offset)
        torch.testing.assert_close(actual[:, -1], reference.logits[:, -1].float(), atol=.005, rtol=.005)
        assert actual[:, -1].argmax(-1).tolist() == reference.logits[:, -1].argmax(-1).tolist()
        tokens = reference.logits[:, -1].argmax(-1).unsqueeze(1)


async def test_bundle_api_and_exact_capacity_message(bundle, tmp_path, capsys):
    m, archive = bundle
    c = Coordinator(tmp_path / "executor")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(c)), base_url="http://executor") as client:
        r = await client.post("/api/models/import", json={"kind": "bundle", "source": str(archive)})
        assert r.status_code == 202
        while c.jobs[r.json()["id"]]["status"] == "running":
            await asyncio.sleep(.02)
        assert c.jobs[r.json()["id"]]["status"] == "completed"
        response = await client.post("/api/deployments", json={"model": m["key"]})
        assert response.status_code == 503 and response.json()["detail"] == MEMORY_CAUTION
        assert MEMORY_CAUTION in capsys.readouterr().out
        c.register(NodeInfo(id="tiny", url="http://tiny", memory_bytes=1))
        assert c.assignments("tiny") == []
        c.register(NodeInfo(id="large", url="http://large", memory_bytes=10**9))
        assert c.assignments("large")
        response = await client.get("/api/bundles/" + m["key"])
        assert response.status_code == 200 and response.content.startswith(b"PK")
    c.db.close()
    await c.transport.close()

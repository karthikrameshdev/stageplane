import asyncio
import contextlib
import errno
import hashlib
import http.server
import threading
import time
from pathlib import Path

import httpx
import pytest

import stageplane.artifacts as artifact_module
from stageplane.artifacts import ArtifactFetcher, BLOCK_SIZE, normalize_sources
from stageplane.common import NodeInfo
from stageplane.node import Worker


def expected(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def artifact(root, data=b"verified weights", name="weights.safetensors"):
    path = root / "model" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
async def workers(tmp_path, monkeypatch):
    monkeypatch.delenv("STAGEPLANE_LOCAL_ARTIFACTS", raising=False)
    created = []
    async def make(sources=None, body=b"verified weights", token=None):
        worker = Worker(NodeInfo(id=f"w{len(created)}", url="http://worker"), "http://executor",
                        tmp_path / f"cache{len(created)}", artifact_sources=sources)
        await worker.transport.http.aclose()
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(200, content=body)
        worker.transport.http = httpx.AsyncClient(transport=httpx.MockTransport(handle),
                                                  headers={"Authorization": f"Bearer {token}"} if token else {})
        created.append(worker)
        return worker, calls
    yield make
    for worker in created:
        await worker.artifacts.close()
        await worker.transport.close()


@contextlib.contextmanager
def origin(routes):
    requests = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            status, headers, body, delay = routes.get(self.path, (404, {}, b"missing", 0))
            if delay:
                time.sleep(delay)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_normalize_sources_preserves_priority_and_bounds(tmp_path):
    assert normalize_sources([str(tmp_path), tmp_path.as_uri(), "https://EXAMPLE.com/models/",
                              "https://example.com/models"]) == [str(tmp_path), "https://example.com/models"]
    assert normalize_sources(["file://localhost" + str(tmp_path)]) == [str(tmp_path)]
    assert normalize_sources(None) == normalize_sources([]) == []
    with pytest.raises(ValueError, match="8"):
        normalize_sources([str(tmp_path)] * 9)


@pytest.mark.parametrize("value", ["relative/path", "s3://bucket", "file://remote/path", "file:relative",
                                      "https://user:secret@example.com", "https://example.com/?token=secret",
                                      "https://example.com/#fragment", "https://example.com:99999", "https://a b/path",
                                      "https://example.com/\npath", "file:///a%00b", "", "x" * 2049])
def test_normalize_sources_rejects_unsafe_or_unsupported_roots(value):
    with pytest.raises(ValueError):
        normalize_sources([value])
    with pytest.raises(ValueError):
        normalize_sources(value)


async def test_local_copy_then_cache_precedes_all_sources(tmp_path, workers):
    source = artifact(tmp_path / "source")
    worker, executor = await workers([str(tmp_path / "source")])
    cached = await worker.ensure_file("model", source.name, expected(source.read_bytes()), [str(tmp_path / "missing")])
    assert cached.read_bytes() == source.read_bytes()
    assert cached.stat().st_ino != source.stat().st_ino
    source.write_bytes(b"changed source")
    assert await worker.ensure_file("model", source.name, expected(b"verified weights")) == cached
    assert not executor
    assert worker.artifacts.metrics == {"cache_hits": 1, "direct_local": 1, "direct_http": 0, "executor": 0,
        "fallbacks": 0, "direct_failures": 0, "bytes_direct": 16, "bytes_executor": 0}
    assert worker.stats()["metrics"]["artifact_transfers"]["direct_local"] == 1


async def test_corrupt_worker_root_falls_back_to_assignment_root(tmp_path, workers):
    artifact(tmp_path / "bad", b"corrupt! weights!")
    artifact(tmp_path / "shared")
    worker, calls = await workers([str(tmp_path / "bad")])
    path = await worker.ensure_file("model", "weights.safetensors", expected(b"verified weights"),
                                    [str(tmp_path / "shared")])
    assert path.read_bytes() == b"verified weights" and not calls
    assert worker.artifacts.metrics["direct_failures"] == 1
    assert not list(path.parent.glob("*.partial"))


async def test_corrupt_cache_is_repaired_from_verified_source(tmp_path, workers):
    artifact(tmp_path / "source")
    worker, calls = await workers([str(tmp_path / "source")])
    artifact(worker.cache, b"corrupt! weights!")
    path = await worker.ensure_file("model", "weights.safetensors", expected(b"verified weights"))
    assert path.read_bytes() == b"verified weights" and not calls
    assert worker.artifacts.metrics["cache_hits"] == 0
    assert worker.artifacts.metrics["direct_local"] == 1


async def test_real_http_download_has_no_executor_credentials(tmp_path, workers, monkeypatch):
    body = b"weights!" * (BLOCK_SIZE // 8 + 1)
    worker, calls = await workers(body=body, token="executor-only-secret")
    # Loading a worker certificate here would fail. Mirrors must use an entirely
    # separate TLS configuration, even when the executor uses mutual TLS.
    monkeypatch.setenv("STAGEPLANE_CLIENT_CERT", str(tmp_path / "nonexistent-cert"))
    monkeypatch.setenv("STAGEPLANE_CLIENT_KEY", str(tmp_path / "nonexistent-key"))
    monkeypatch.setenv("STAGEPLANE_TOKEN", "executor-only-secret")
    with origin({"/models/model/weights.safetensors": (200, {}, body, 0)}) as (url, seen):
        path = await worker.ensure_file("model", "weights.safetensors", expected(body), [url + "/models"])
    assert path.read_bytes() == body and not calls
    assert len(seen) == 1
    assert not any(k.lower() in ("authorization", "cookie") for k in seen[0][1])
    assert worker.artifacts.metrics["direct_http"] == 1
    assert worker.artifacts.metrics["bytes_direct"] == len(body)


@pytest.mark.parametrize("response", [(404, {}, b"missing", 0), (403, {}, b"denied", 0),
                                     (200, {}, b"corrupt! weights!", 0),
                                     (200, {}, b"verified weights plus extra", 0),
                                     (200, {"Content-Encoding": "gzip"}, b"not compressed", 0)])
async def test_direct_http_failure_uses_authenticated_executor(workers, response):
    worker, calls = await workers(token="executor-secret")
    with origin({"/model/weights.safetensors": response}) as (url, seen):
        path = await worker.ensure_file("model", "weights.safetensors", expected(b"verified weights"), [url])
    assert path.read_bytes() == b"verified weights"
    assert len(seen) == len(calls) == 1
    assert calls[0].headers["authorization"] == "Bearer executor-secret"
    assert calls[0].url.path == "/api/artifacts/model/weights.safetensors"
    assert worker.artifacts.metrics["executor"] == worker.artifacts.metrics["fallbacks"] == 1
    assert worker.artifacts.metrics["direct_failures"] == 1
    assert worker.artifacts.metrics["bytes_executor"] == 16
    assert not list(path.parent.glob("*.partial"))


async def test_real_read_timeout_cools_origin_for_next_chunk(workers):
    worker, calls = await workers()
    worker.artifacts.http = httpx.AsyncClient(timeout=.03, trust_env=False)
    with origin({"/model/a.safetensors": (200, {}, b"verified weights", .15)}) as (url, seen):
        for name in ("a.safetensors", "b.safetensors"):
            await worker.ensure_file("model", name, expected(b"verified weights"), [url])
    assert len(seen) == 1 and len(calls) == 2
    assert worker.artifacts.metrics["direct_failures"] == 1
    assert worker.artifacts.metrics["fallbacks"] == 2


async def test_cancellation_cleans_partial_and_never_falls_back(workers):
    worker, calls = await workers()
    started = asyncio.Event()
    class Stalled(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * BLOCK_SIZE
            started.set()
            await asyncio.Event().wait()
    worker.artifacts.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=Stalled())))
    task = asyncio.create_task(worker.ensure_file("model", "weights.safetensors",
        {"bytes": BLOCK_SIZE * 2, "sha256": "0" * 64}, ["https://mirror.example"]))
    await asyncio.wait_for(started.wait(), 2)
    assert list(worker.cache.rglob("*.partial"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not calls and not list(worker.cache.rglob("*.partial"))
    assert not (worker.cache / "model" / "weights.safetensors").exists()
    assert worker.artifacts.metrics["fallbacks"] == 0


@pytest.mark.parametrize("slow_operation", ["read", "write"])
async def test_slow_filesystem_keeps_loop_responsive_and_cancellation_waits_for_io(tmp_path, workers, monkeypatch, slow_operation):
    artifact(tmp_path / "source")
    worker, calls = await workers([str(tmp_path / "source")])
    entered, release, finished, closed = (threading.Event() for _ in range(4))
    observed = {}
    class SlowHandle:
        def __init__(self, handle):
            self.handle = handle
        def read(self, count):
            return self.slow(self.handle.read, count)
        def write(self, data):
            return self.slow(self.handle.write, data)
        def slow(self, operation, value):
            entered.set()
            assert release.wait(3), "Event loop failed to release the simulated mount I/O"
            result = operation(value)
            finished.set()
            return result
        def close(self):
            observed["closed_after_io"] = finished.is_set()
            self.handle.close()
            closed.set()
    if slow_operation == "read":
        original = artifact_module._open_local
        monkeypatch.setattr(artifact_module, "_open_local", lambda *args: SlowHandle(original(*args)))
    else:
        original = artifact_module._temporary
        def temporary(*args):
            output, path = original(*args)
            return SlowHandle(output), path
        monkeypatch.setattr(artifact_module, "_temporary", temporary)
    transfer = asyncio.create_task(worker.ensure_file("model", "weights.safetensors", expected(b"verified weights")))
    ticks = 0
    try:
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(.005)
        assert entered.is_set() and not finished.is_set()
        # These event-loop ticks stand in for heartbeats while filesystem I/O is
        # deliberately stalled in another thread.
        for _ in range(3):
            await asyncio.sleep(.005)
            ticks += 1
        transfer.cancel()
        await asyncio.sleep(.01)
        assert ticks == 3 and not transfer.done() and not closed.is_set()
        assert list(worker.cache.rglob("*.partial"))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(transfer, 2)
        assert finished.is_set() and closed.is_set() and observed["closed_after_io"]
        assert not list(worker.cache.rglob("*.partial")) and not calls
        assert not (worker.cache / "model" / "weights.safetensors").exists()
    finally:
        release.set()
        if not transfer.done():
            transfer.cancel()
        await asyncio.gather(transfer, return_exceptions=True)


async def test_destination_disk_full_is_not_masked_by_fallback(tmp_path, workers, monkeypatch):
    artifact(tmp_path / "source")
    worker, calls = await workers([str(tmp_path / "source")])
    original = Path.replace
    def fail_publish(path, destination):
        if path.name.endswith(".partial"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return original(path, destination)
    monkeypatch.setattr(Path, "replace", fail_publish)
    with pytest.raises(OSError) as error:
        await worker.ensure_file("model", "weights.safetensors", expected(b"verified weights"))
    assert error.value.errno == errno.ENOSPC
    assert not calls and not list(worker.cache.rglob("*.partial"))


async def test_teardown_failure_cannot_replace_destination_disk_error(tmp_path, monkeypatch):
    fetcher = ArtifactFetcher([])
    class FailingClose:
        def __aiter__(self):
            async def blocks():
                yield b"verified weights"
            return blocks()
        async def aclose(self):
            raise httpx.ReadError("Source teardown failed")
    def no_space(*args):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(Path, "replace", no_space)
    with pytest.raises(OSError) as error:
        await fetcher._publish(tmp_path / "weights.safetensors", expected(b"verified weights"), FailingClose())
    assert error.value.errno == errno.ENOSPC
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("model,name", [("../escape", "weights.safetensors"), ("model", "../escape"),
                                         ("model", "%2e%2e%2fsecret"), ("model", "/tmp/escape")])
async def test_artifact_names_cannot_escape_cache_or_url(workers, model, name):
    worker, calls = await workers()
    with pytest.raises(ValueError, match="outside"):
        await worker.ensure_file(model, name, expected(b"verified weights"))
    assert not calls


async def test_source_symlink_escape_falls_back(tmp_path, workers):
    outside = artifact(tmp_path / "outside")
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "model").symlink_to(outside.parent, target_is_directory=True)
    worker, calls = await workers([str(tmp_path / "source")])
    await worker.ensure_file("model", "weights.safetensors", expected(b"verified weights"))
    assert len(calls) == 1
    assert worker.artifacts.metrics["direct_failures"] == 1


async def test_concurrent_downloads_publish_unique_verified_files(workers):
    worker, calls = await workers()
    with origin({"/model/weights.safetensors": (200, {}, b"verified weights", .03)}) as (url, seen):
        paths = await asyncio.gather(*(worker.ensure_file("model", "weights.safetensors",
            expected(b"verified weights"), [url]) for _ in range(3)))
    assert all(path.read_bytes() == b"verified weights" for path in paths)
    assert len(seen) == 3 and not calls and not list(worker.cache.rglob("*.partial"))


async def test_explicit_sources_disable_legacy_hardlinks(tmp_path, workers, monkeypatch):
    source = artifact(tmp_path / "legacy")
    monkeypatch.setenv("STAGEPLANE_LOCAL_ARTIFACTS", str(tmp_path / "legacy"))
    worker, calls = await workers([])
    path = await worker.ensure_file("model", source.name, expected(source.read_bytes()))
    assert len(calls) == 1 and path.stat().st_ino != source.stat().st_ino
    legacy, calls = await workers(None)
    path = await legacy.ensure_file("model", source.name, expected(source.read_bytes()))
    assert not calls and path.stat().st_ino == source.stat().st_ino


def test_cooldown_and_metric_state_remains_bounded():
    fetcher = ArtifactFetcher([])
    for index in range(100):
        fetcher._cooldown(f"https://mirror-{index}.example")
    assert len(fetcher.cooldowns) == 64
    fetcher._count("bytes_direct", 2**65)
    assert fetcher.metrics["bytes_direct"] == 2**63 - 1


def test_invalid_worker_sources_fail_before_creating_cache(tmp_path):
    with pytest.raises(ValueError):
        Worker(NodeInfo(id="invalid", url="http://worker"), "http://executor", tmp_path / "cache",
               artifact_sources=["relative/path"])
    assert not (tmp_path / "cache").exists()

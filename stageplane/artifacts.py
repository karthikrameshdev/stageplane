"""Verified worker artifact retrieval from local mounts, mirrors, and executor."""
from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx

from .common import safe_path

MAX_SOURCES = 8
MAX_SOURCE_LENGTH = 2048
BLOCK_SIZE = 1024 * 1024
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
SHA = re.compile(r"^[0-9a-f]{64}$")
DISK_FULL = {errno.ENOSPC, errno.EDQUOT}


def normalize_sources(values=None):
    """Canonical, ordered roots; credentials and non-local file URLs are refused."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)) or len(values) > MAX_SOURCES:
        raise ValueError("Artifact sources must be a list of at most 8 roots")
    roots = []
    for value in values:
        if not isinstance(value, str) or not 1 <= len(value) <= MAX_SOURCE_LENGTH:
            raise ValueError("Artifact source must contain 1 to 2048 characters")
        value = value.strip()
        if not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("Invalid artifact source")
        if Path(value).is_absolute():
            root = str(Path(value).resolve())
        else:
            parsed = urlsplit(value)
            if parsed.query or parsed.fragment or parsed.username is not None or parsed.password is not None:
                raise ValueError("Artifact source cannot contain credentials, query, or fragment")
            if parsed.scheme == "file":
                path = Path(unquote(parsed.path))
                if parsed.netloc not in ("", "localhost") or not path.is_absolute():
                    raise ValueError("Artifact file source must be an absolute local path")
                root = str(path.resolve())
            elif parsed.scheme in ("http", "https") and parsed.hostname:
                if any(c.isspace() for c in parsed.netloc) or "\\" in value:
                    raise ValueError("Invalid artifact source URL")
                # Parsing the port validates malformed or out-of-range values.
                parsed.port
                root = urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))
            else:
                raise ValueError("Artifact source must be an absolute path or an HTTP(S) root URL")
        if any(ord(c) < 32 or ord(c) == 127 for c in root) or len(root) > MAX_SOURCE_LENGTH:
            raise ValueError("Invalid normalized artifact source")
        if root not in roots:
            roots.append(root)
    return roots


class SourceUnavailable(Exception):
    """A source failed without implicating the destination cache filesystem."""


class ArtifactIntegrityError(ValueError):
    pass


async def _file_io(operation, *args, cancel_cleanup=None):
    """Keep mount I/O off the event loop and finish it before cancellation cleanup.

    A filesystem syscall cannot be interrupted safely by cancelling its Python
    future. Unresponsive filesystems remain subject to OS/mount timeout policy.
    """
    work = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not work.cancelled() and work.exception() is None and cancel_cleanup is not None:
            # Opening a handle or tempfile may have succeeded after cancellation.
            with contextlib.suppress(BaseException):
                await _file_io(cancel_cleanup, work.result())
        raise


def _temporary(path):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        return os.fdopen(fd, "wb"), Path(temporary)
    except BaseException:
        os.close(fd)
        Path(temporary).unlink(missing_ok=True)
        raise


def _close_temporary(resource):
    output, temporary = resource
    try:
        output.close()
    finally:
        temporary.unlink(missing_ok=True)


def _destination(cache, model, name):
    root = safe_path(cache, model)
    root.mkdir(parents=True, exist_ok=True)
    return safe_path(root, name)


def _open_local(path, expected):
    try:
        if not path.is_file():
            raise SourceUnavailable("Artifact source is not a regular file")
        source = os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb")
        try:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise SourceUnavailable("Artifact source is not a regular file")
            if info.st_size != expected["bytes"]:
                raise ArtifactIntegrityError("Artifact integrity check failed: source size")
            return source
        except BaseException:
            source.close()
            raise
    except OSError as error:
        if error.errno in DISK_FULL:
            raise
        raise SourceUnavailable("Cannot open or inspect artifact source") from error


def _matches(path, expected):
    """Hash at most the declared bytes plus one, even if a file grows mid-read."""
    try:
        if not path.is_file():
            return False
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected["bytes"]:
                return False
            total, digest = 0, hashlib.sha256()
            while block := source.read(min(BLOCK_SIZE, expected["bytes"] - total + 1)):
                total += len(block)
                if total > expected["bytes"]:
                    return False
                digest.update(block)
            return total == expected["bytes"] and digest.hexdigest() == expected["sha256"]
    except OSError as error:
        if error.errno in DISK_FULL:
            raise
        return False


class ArtifactFetcher:
    def __init__(self, sources=None):
        self.sources = normalize_sources(sources)
        # This opt-in predates mirrors and is used by local benchmark fixtures.
        # An explicit source list, including [], disables the legacy shortcut.
        self.legacy_root = os.environ.get("STAGEPLANE_LOCAL_ARTIFACTS") if sources is None else None
        self.http = None
        self.cooldowns = {}
        self.metrics = {key: 0 for key in ("cache_hits", "direct_local", "direct_http", "executor", "fallbacks",
                                        "direct_failures", "bytes_direct", "bytes_executor")}

    def _count(self, key, amount=1):
        self.metrics[key] = min(2**63 - 1, self.metrics[key] + amount)

    def _direct_http(self):
        if self.http is None:
            # Deliberately independent of the authenticated executor transport:
            # no bearer headers, client certificate, environment proxy or shared cookie jar.
            self.http = httpx.AsyncClient(timeout=httpx.Timeout(10, connect=3, pool=3), trust_env=False,
                                          follow_redirects=True, max_redirects=3,
                                          headers={"Accept-Encoding": "identity"})
        return self.http

    def _cooldown(self, source):
        origin = urlsplit(source)
        key = (origin.scheme, origin.netloc)
        self.cooldowns.pop(key, None)
        if len(self.cooldowns) >= 64:
            self.cooldowns.pop(next(iter(self.cooldowns)))
        self.cooldowns[key] = time.monotonic() + 30

    def _cooling(self, source):
        origin = urlsplit(source)
        return self.cooldowns.get((origin.scheme, origin.netloc), 0) > time.monotonic()

    async def _local_blocks(self, path, expected):
        source = await _file_io(_open_local, path, expected, cancel_cleanup=lambda handle: handle.close())
        try:
            total = 0
            while True:
                try:
                    block = await _file_io(source.read, min(BLOCK_SIZE, expected["bytes"] - total + 1))
                except OSError as error:
                    if error.errno in DISK_FULL:
                        raise
                    raise SourceUnavailable("Cannot read artifact source") from error
                if not block:
                    return
                total += len(block)
                yield block
        finally:
            await _file_io(source.close)

    async def _http_blocks(self, client, url):
        async with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            # Refuse compressed responses so decompression cannot evade the byte
            # budget before the bounded block iterator sees the expanded payload.
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise ArtifactIntegrityError("Artifact integrity check failed: unexpected content encoding")
            async for block in response.aiter_bytes(BLOCK_SIZE):
                yield block

    async def _publish(self, path, expected, blocks):
        output, temporary = await _file_io(_temporary, path, cancel_cleanup=_close_temporary)
        failed = False
        try:
            total, digest = 0, hashlib.sha256()
            try:
                async for block in blocks:
                    total += len(block)
                    if total > expected["bytes"]:
                        raise ArtifactIntegrityError("Artifact larger than declared size")
                    await _file_io(output.write, block)
                    digest.update(block)
            except BaseException:
                with contextlib.suppress(BaseException):
                    await _file_io(output.close)
                raise
            else:
                await _file_io(output.close)
            if total != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ArtifactIntegrityError("Artifact integrity check failed")
            await _file_io(temporary.replace, path)
        except BaseException:
            failed = True
            raise
        finally:
            if failed:
                # Teardown failures must not replace cancellation or ENOSPC with
                # a retryable mirror error and accidentally trigger fallback.
                with contextlib.suppress(BaseException):
                    await blocks.aclose()
                with contextlib.suppress(BaseException):
                    await _file_io(temporary.unlink, True)
            else:
                try:
                    await blocks.aclose()
                finally:
                    await _file_io(temporary.unlink, True)

    async def _legacy_link(self, path, model, name, expected):
        if not self.legacy_root:
            return False
        try:
            source = await _file_io(safe_path, Path(self.legacy_root), model + "/" + name)
        except ValueError:
            return False
        if not await _file_io(_matches, source, expected):
            return False
        def link():
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
            os.close(fd)
            temporary = Path(temporary)
            try:
                temporary.unlink()
                try:
                    os.link(source, temporary)
                except OSError as error:
                    if error.errno in DISK_FULL:
                        raise
                    return False
                temporary.replace(path)
                return True
            finally:
                temporary.unlink(missing_ok=True)
        if await _file_io(link):
            self._count("direct_local")
            return True
        return False

    async def ensure(self, cache, model, name, expected, executor, executor_http, sources=()):
        if not isinstance(model, str) or not NAME.fullmatch(model) or not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError("Path outside artifact directory or invalid artifact name")
        if (not isinstance(expected, dict) or type(expected.get("bytes")) is not int
                or not 0 <= expected["bytes"] < 2**63 or not SHA.fullmatch(str(expected.get("sha256", "")))):
            raise ValueError("Invalid artifact size or SHA-256")
        advertised = await _file_io(normalize_sources, sources)
        roots = list(dict.fromkeys([*self.sources, *advertised]))
        path = await _file_io(_destination, cache, model, name)
        if await _file_io(_matches, path, expected):
            self._count("cache_hits")
            return path
        if await self._legacy_link(path, model, name, expected):
            return path
        for source in roots:
            remote = source.startswith(("http://", "https://"))
            if remote and self._cooling(source):
                continue
            try:
                if remote:
                    blocks = self._http_blocks(self._direct_http(), f"{source}/{model}/{name}")
                else:
                    try:
                        source_path = await _file_io(safe_path, Path(source), model + "/" + name)
                    except ValueError as error:
                        raise SourceUnavailable("Artifact source path escapes root") from error
                    blocks = self._local_blocks(source_path, expected)
                await self._publish(path, expected, blocks)
            except (SourceUnavailable, ArtifactIntegrityError, httpx.HTTPError) as error:
                self._count("direct_failures")
                if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
                    self._cooldown(source)
                continue
            self._count("direct_http" if remote else "direct_local")
            self._count("bytes_direct", expected["bytes"])
            return path
        if roots or self.legacy_root:
            self._count("fallbacks")
        await self._publish(path, expected, self._http_blocks(executor_http, f"{executor}/api/artifacts/{model}/{name}"))
        self._count("executor")
        self._count("bytes_executor", expected["bytes"])
        return path

    async def close(self):
        if self.http is not None:
            await self.http.aclose()

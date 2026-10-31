"""Portable, verified model bundles: filesystem, HTTP, Git/LFS and public Drive.

Publishing is deliberately separate from importing: users choose their host and
upload an archive or directory, then any executor can import the same manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx

from .common import ROOT, atomic_json, digest, safe_path
from .distribution import SUPPORTED

MAX_MANIFEST = 4 * 1024**2
MAX_BUNDLE = int(os.environ.get("STAGEPLANE_BUNDLE_MAX_BYTES", str(256 * 1024**3)))
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
SHA = re.compile(r"^[0-9a-f]{64}$")


def validate_manifest(m: dict, expected_sha256=None):
    if not isinstance(m, dict) or m.get("schema") != 1:
        raise ValueError("Unsupported Stageplane manifest schema")
    actual = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k != "manifest_sha256"}, sort_keys=True).encode()).hexdigest()
    if actual != m.get("manifest_sha256") or expected_sha256 and actual != expected_sha256:
        raise ValueError("Manifest SHA-256 mismatch")
    if not NAME.fullmatch(m.get("key", "")) or m.get("architecture") not in SUPPORTED:
        raise ValueError("Unsupported model key or architecture")
    cfg = m["config"].get("text_config", m["config"])
    if cfg.get("model_type") != m["architecture"]:
        raise ValueError("Architecture does not match model configuration")
    if not isinstance(cfg.get("num_hidden_layers"), int) or not 1 <= cfg["num_hidden_layers"] <= 4096:
        raise ValueError("Invalid language layer count")
    files = m.get("files", {})
    if not isinstance(files, dict) or not 1 <= len(files) <= 20000:
        raise ValueError("Invalid file inventory")
    for name, info in files.items():
        if not NAME.fullmatch(name) or name == "manifest.json" or not (name == "LICENSE" or name.endswith((".json", ".safetensors", ".txt", ".md", ".jinja"))):
            raise ValueError("Unsafe or unsupported bundle filename")
        if not isinstance(info.get("bytes"), int) or not 0 <= info["bytes"] <= MAX_BUNDLE or not SHA.fullmatch(info.get("sha256", "")):
            raise ValueError("Invalid artifact size or hash")
    total = sum(f["bytes"] for f in files.values())
    if total > MAX_BUNDLE or not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= files.keys():
        raise ValueError("Incomplete or oversized bundle")
    for artifact in m["artifacts"].values():
        info = files.get(artifact["file"])
        if not info or (info["bytes"], info["sha256"]) != (artifact["bytes"], artifact["sha256"]):
            raise ValueError("Artifact inventory mismatch")
    stages = m["stages"]
    if not 3 <= len(stages) <= 4098 or len({s["id"] for s in stages}) != len(stages):
        raise ValueError("Invalid execution stages")
    if stages[0]["kind"] != "embedding" or stages[-1]["kind"] != "head":
        raise ValueError("Incomplete execution path")
    end = 0
    for stage in stages:
        if stage["kind"] not in ("embedding", "blocks", "head"):
            raise ValueError("Unsupported stage kind")
        if not NAME.fullmatch(stage["id"]) or any(a not in m["artifacts"] for a in stage["artifacts"]):
            raise ValueError("Invalid stage artifacts")
        if stage["weight_bytes"] != sum(m["artifacts"][a]["bytes"] for a in stage["artifacts"]):
            raise ValueError("Stage size mismatch")
        if stage["kind"] == "blocks":
            if stage["start"] != end or stage["end"] <= end:
                raise ValueError("Layer intervals are not contiguous")
            end = stage["end"]
    if end != cfg["num_hidden_layers"]:
        raise ValueError("Missing language layers")
    return m


def read_manifest(path: Path, expected=None):
    if path.stat().st_size > MAX_MANIFEST:
        raise ValueError("Manifest too large")
    return validate_manifest(json.loads(path.read_text()), expected)


def verify_files(directory: Path, manifest):
    for name, info in manifest["files"].items():
        path = safe_path(directory, name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Artifact integrity check failed: {name}")
    if json.loads((directory / "config.json").read_text()) != manifest["config"]:
        raise ValueError("Model configuration differs from manifest")


def export_bundle(directory: Path, output: Path, format="archive"):
    directory, output = directory.resolve(), output.resolve()
    manifest = read_manifest(directory / "manifest.json")
    verify_files(directory, manifest)
    if output.exists():
        raise ValueError("Export destination already exists")
    if output.is_relative_to(directory):
        raise ValueError("Export must be outside the model directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ["manifest.json", *sorted(manifest["files"])]
    if format == "directory":
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".export-") as tmp:
            stage = Path(tmp) / "bundle"
            stage.mkdir()
            for name in names:
                shutil.copyfile(directory / name, stage / name)
            stage.rename(output)
    else:
        tmp = output.with_suffix(output.suffix + ".partial")
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for name in names:
                    archive.write(directory / name, name)
            tmp.rename(output)
        finally:
            tmp.unlink(missing_ok=True)
    return {"path": str(output), "model": manifest["key"], "manifest_sha256": manifest["manifest_sha256"]}


def is_drive(url):
    return urlparse(url).hostname in ("drive.google.com", "drive.usercontent.google.com")


def download(url, path: Path, limit=MAX_BUNDLE, expected_bytes=None):
    if is_drive(url):
        import gdown
        class BoundedOutput:
            def __init__(self, file):
                self.file, self.total = file, 0
            def write(self, data):
                self.total += len(data)
                if self.total > limit or shutil.disk_usage(path.parent).free < len(data) + 64 * 1024**2:
                    raise ValueError("Drive download exceeds size or disk budget")
                return self.file.write(data)
            def tell(self):
                return self.file.tell()
            def flush(self):
                return self.file.flush()
        try:
            with path.open("wb") as file:
                result = gdown.download(url=url, output=BoundedOutput(file), quiet=True, use_cookies=False)
        except gdown.DownloadError as error:
            raise ValueError("Public Google Drive download failed; check sharing and download quota") from error
        if not result:
            raise ValueError("Public Google Drive download failed; check sharing and download quota")
        if path.stat().st_size > limit:
            raise ValueError("Drive file exceeds import limit")
    else:
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError("Only HTTP(S) artifact URLs are supported")
        with httpx.Client(follow_redirects=True, timeout=120, trust_env=False) as client:
            with client.stream("GET", url) as r:
                r.raise_for_status()
                if int(r.headers.get("content-length", "0")) > limit:
                    raise ValueError("Remote file exceeds import limit")
                total = 0
                with path.open("wb") as out:
                    for block in r.iter_bytes(1024**2):
                        total += len(block)
                        if total > limit or shutil.disk_usage(path.parent).free < len(block) + 64 * 1024**2:
                            raise ValueError("Download exceeds size or disk budget")
                        out.write(block)
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError("Downloaded artifact size mismatch")


@contextmanager
def git_checkout(source: str, revision="HEAD", subdir="."):
    repo = source.removeprefix("git+")
    with tempfile.TemporaryDirectory(prefix="stageplane-git-") as tmp:
        checkout = Path(tmp) / "repo"
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
        def git(*args):
            return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.ext.allow=never", *args],
                                  env=env, check=True, capture_output=True, text=True, timeout=180)
        git("clone", "--no-checkout", "--", repo, str(checkout))
        git("-C", str(checkout), "checkout", "--detach", revision)
        if (checkout / ".gitattributes").exists() and "filter=lfs" in (checkout / ".gitattributes").read_text():
            git("-C", str(checkout), "lfs", "pull")
        root = safe_path(checkout, subdir)
        if root.is_symlink():
            raise ValueError("Git bundle directory cannot be a symlink")
        yield root


def import_bundle(source: str, store: Path = ROOT, expected_sha256=None, revision="HEAD", subdir="."):
    store = store.resolve()
    store.mkdir(parents=True, exist_ok=True)
    if source.startswith("git+") or source.endswith(".git"):
        with git_checkout(source, revision, subdir) as root:
            document = root / "catalog.json" if root.is_dir() and (root / "catalog.json").exists() else root
            return import_bundle(str(document), store, expected_sha256)
    if source.startswith("file://"):
        source = unquote(urlparse(source).path)
    with tempfile.TemporaryDirectory(dir=store, prefix=".import-") as tmp:
        temporary = Path(tmp)
        local = Path(source).expanduser()
        base_url = None
        if local.is_dir():
            document = local / "manifest.json"
        elif local.is_file():
            document = local
        else:
            parsed = urlparse(source)
            if parsed.scheme not in ("http", "https"):
                raise ValueError("Bundle source does not exist or is not a supported URL")
            url = source
            if not is_drive(source) and (parsed.path.endswith("/") or not Path(parsed.path).suffix):
                url = source.rstrip("/") + "/manifest.json"
            base_url = url
            document = temporary / "download"
            download(url, document)
        stage = temporary / "verified"
        stage.mkdir()
        if zipfile.is_zipfile(document):
            with zipfile.ZipFile(document) as archive:
                infos = archive.infolist()
                if len(infos) > 20001 or len({i.filename for i in infos}) != len(infos):
                    raise ValueError("Invalid ZIP file inventory")
                for info in infos:
                    if not NAME.fullmatch(info.filename) or info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                        raise ValueError("Unsafe ZIP member")
                header = archive.getinfo("manifest.json")
                if header.file_size > MAX_MANIFEST:
                    raise ValueError("Manifest too large")
                m = validate_manifest(json.loads(archive.read(header)), expected_sha256)
                if {i.filename for i in infos} != {"manifest.json", *m["files"]}:
                    raise ValueError("ZIP files differ from manifest inventory")
                for info in infos:
                    if info.filename != "manifest.json" and info.file_size != m["files"][info.filename]["bytes"]:
                        raise ValueError("ZIP artifact size mismatch")
                if shutil.disk_usage(store).free < sum(i.file_size for i in infos) + 64 * 1024**2:
                    raise ValueError("Not enough disk space to import bundle")
                for info in infos:
                    with archive.open(info) as inp, (stage / info.filename).open("wb") as out:
                        shutil.copyfileobj(inp, out, 1024**2)
        else:
            if document.stat().st_size > MAX_MANIFEST:
                raise ValueError("Expected a Stageplane ZIP archive or JSON manifest/catalog")
            raw = json.loads(document.read_text())
            overrides = {}
            if raw.get("format") == "stageplane-catalog-v1":
                overrides = raw.get("files", {})
                manifest_url = urljoin(base_url, raw["manifest"]) if base_url else raw["manifest"]
                expected_sha256 = expected_sha256 or raw.get("manifest_sha256")
                if urlparse(manifest_url).scheme in ("http", "https"):
                    manifest_file = temporary / "remote-manifest.json"
                    download(manifest_url, manifest_file, MAX_MANIFEST)
                    base_url = manifest_url
                else:
                    manifest_file = safe_path(document.parent, manifest_url)
                m = read_manifest(manifest_file, expected_sha256)
                local = manifest_file.parent
            else:
                m = validate_manifest(raw, expected_sha256)
                local = document.parent
            if shutil.disk_usage(store).free < sum(f["bytes"] for f in m["files"].values()) + 64 * 1024**2:
                raise ValueError("Not enough disk space to import bundle")
            for name, info in m["files"].items():
                destination = stage / name
                url = overrides.get(name)
                if url:
                    download(urljoin(base_url, url) if base_url else url, destination, info["bytes"], info["bytes"])
                elif base_url:
                    download(urljoin(base_url, name), destination, info["bytes"], info["bytes"])
                else:
                    candidate = safe_path(local, name)
                    if candidate.is_symlink():
                        raise ValueError("Bundle artifacts cannot be symlinks")
                    shutil.copyfile(candidate, destination)
            atomic_json(stage / "manifest.json", m)
        verify_files(stage, m)
        target = store / m["key"]
        if target.exists():
            old = read_manifest(target / "manifest.json")
            if old["manifest_sha256"] != m["manifest_sha256"]:
                raise ValueError("Immutable model key already has a different manifest")
            verify_files(target, m)
            return old
        stage.rename(target)
        return m

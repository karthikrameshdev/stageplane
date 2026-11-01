"""Download an immutable checkpoint straight into independently loadable stages.

HTTP range reads avoid keeping a second, complete checkpoint on disk. Every tensor
is retained, including auxiliary weights that the text-only route does not use.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
import shutil
import tempfile
from itertools import islice
import re
import struct
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .common import ROOT, atomic_json, digest, model_key, safe_path

MAX_TOTAL_HEADER_BYTES = int(os.environ.get("STAGEPLANE_MAX_TOTAL_HEADER_BYTES", str(128 * 1024**2)))

SUPPORTED = {"qwen2", "qwen3", "qwen3_5_text", "llama", "mistral"}
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
               "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def resolve_source(source):
    if source.startswith("https://huggingface.co/"):
        repo = source.removeprefix("https://huggingface.co/").rstrip("/")
        if len(repo.split("/")) == 2:
            return repo
    if source.startswith("ollama:"):
        tag = source.removeprefix("ollama:").removeprefix("library/").lower()
        # Explicit, verified dense-family aliases. Do not guess a source for
        # private tags, fine-tunes, MoE models, or arbitrary GGUF conversions.
        aliases = {f"qwen3.5:{size}b": f"Qwen/Qwen3.5-{size}B" for size in ("0.8", "2", "4", "9", "27")}
        if tag in aliases:
            return aliases[tag]
        raise ValueError("Cannot resolve this Ollama GGUF tag. Use --hf-source with the exact original HF repository; quantized tensors cannot be reinterpreted as dense weights.")
    return source


class Source:
    def __init__(self, source: str, revision: str = "main"):
        self.local = Path(source).expanduser().is_dir()
        self.source = source
        self.http = httpx.Client(follow_redirects=True, timeout=120, trust_env=False,
            headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") and not source.startswith(("http://", "https://")) and not self.local else {})
        if self.local:
            self.root = Path(source).expanduser().resolve()
            self.files = [str(p.relative_to(self.root)) for p in self.root.iterdir() if p.is_file()]
            identity = {name: digest(self.root / name) for name in sorted(self.files)
                        if name.endswith((".safetensors", ".json", ".jinja"))}
            self.revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        elif source.startswith(("http://", "https://")):
            # A trusted static HF-format directory with a files.json inventory.
            self.root = source.rstrip("/")
            inventory = self.json("files.json")
            self.revision = inventory["revision"]
            self.files = inventory["files"]
        else:
            r = self.http.get(f"https://huggingface.co/api/models/{source}/revision/{revision}")
            r.raise_for_status()
            info = r.json()
            self.revision = info["sha"]
            self.root = f"https://huggingface.co/{source}/resolve/{self.revision}"
            self.files = [f["rfilename"] for f in info["siblings"]]

    def read(self, name: str, start: int | None = None, end: int | None = None):
        if self.local:
            with safe_path(self.root, name).open("rb") as f:
                f.seek(start or 0)
                limit = end - (start or 0) + 1 if end is not None else 128 * 1024**2
                data = f.read(limit if end is not None else limit + 1)
                if end is not None and len(data) != limit:
                    raise ValueError("Truncated tensor range")
                if len(data) > limit:
                    raise ValueError("Source metadata exceeds size limit")
                return data
        for attempt in range(4):
            try:
                headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"} if start is not None else {}
                # Unique query prevents intermediaries reusing an unrelated cached range.
                url = f"{self.root}/{name}"
                if start is not None:
                    url += f"?stageplane_range={start}-{end}"
                with self.http.stream("GET", url, headers=headers) as r:
                    r.raise_for_status()
                    if start is not None:
                        if r.status_code != 206 or not r.headers.get("content-range", "").startswith(f"bytes {start}-{end}/"):
                            raise ValueError("Source must support exact HTTP ranges; refusing a full duplicate download")
                    limit = end - start + 1 if start is not None else 128 * 1024**2
                    pieces, total = [], 0
                    for piece in r.iter_bytes(1024**2):
                        total += len(piece)
                        if total > limit:
                            raise ValueError("Source response exceeds expected size")
                        pieces.append(piece)
                    data = b"".join(pieces)
                if start is not None and len(data) != end - start + 1:
                    raise ValueError("Truncated tensor range")
                return data
            except (httpx.HTTPError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    def json(self, name):
        return json.loads(self.read(name))

    def close(self):
        self.http.close()


def classify(key: str, layer_count: int, layers_per_chunk: int):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
    # Vision layers belong to the auxiliary artifact, never the language path.
    if "visual." in key or "vision" in key or key.startswith("mtp."):
        return "auxiliary"
    if match:
        layer = int(match[1])
        if layer >= layer_count:
            return "auxiliary"
        return f"blocks-{layer // layers_per_chunk:03d}"
    if "embed_tokens." in key:
        return "embedding"
    if re.search(r"(?:^|\.)norm\.(weight|bias)$", key) or key.startswith("lm_head."):
        return "head"
    return "auxiliary"


def distribute(source: str, store: Path = ROOT, revision="main", layers_per_chunk=2):
    if layers_per_chunk < 1:
        raise ValueError("layers_per_chunk must be positive")
    requested_source = source
    source = resolve_source(source)
    src = Source(source, revision)
    staging = None
    try:
        config = src.json("config.json")
        text = config.get("text_config", config)
        if text["model_type"] not in SUPPORTED:
            raise ValueError(f"Unsupported architecture {text['model_type']}; add and verify an adapter first")
        key = model_key(source, src.revision)
        target = store / key
        if (target / "manifest.json").exists():
            raise ValueError("Immutable revision already exists. Reuse it or choose a different store for a new chunk layout.")
        destination = target
        store.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".building-", dir=store))
        target = staging
        atomic_json(target / "config.json", config)
        count = text["num_hidden_layers"]
        if type(count) is not int or not 1 <= count <= 4096:
            raise ValueError("Invalid language layer count")
        seen, language_layers = set(), set()
        groups = defaultdict(list)
        headers = {}
        total_header_bytes = 0
        total_tensor_bytes = 0
        for file in sorted(f for f in src.files if f.endswith(".safetensors")):
            n = struct.unpack("<Q", src.read(file, 0, 7))[0]
            if n > 16 * 1024 * 1024:
                raise ValueError("Unreasonable safetensors header")
            total_header_bytes += n
            if total_header_bytes > MAX_TOTAL_HEADER_BYTES:
                raise ValueError("Checkpoint headers exceed configured metadata budget")
            def unique_object(pairs):
                obj = {}
                for name, value in pairs:
                    if name in obj:
                        raise ValueError("Duplicate Safetensors header key")
                    obj[name] = value
                return obj
            head = json.loads(src.read(file, 8, 7 + n), object_pairs_hook=unique_object)
            if not isinstance(head, dict):
                raise ValueError("Invalid Safetensors header")
            expected_offset = 0
            headers[file] = head
            entries = [(name, info) for name, info in head.items() if name != "__metadata__"]
            for name, info in entries:
                if not isinstance(info, dict) or not isinstance(info.get("shape"), list):
                    raise ValueError("Invalid tensor descriptor")
                if len(info["shape"]) > 32 or any(type(dimension) is not int or not 0 <= dimension < 2**63 for dimension in info["shape"]):
                    raise ValueError("Invalid tensor shape")
                offsets = info.get("data_offsets")
                if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int or not 0 <= x < 2**63 for x in offsets):
                    raise ValueError("Invalid tensor offsets")
                if info.get("dtype") not in DTYPE_BYTES or offsets[1] - offsets[0] != math.prod(info["shape"]) * DTYPE_BYTES[info["dtype"]]:
                    raise ValueError("Tensor shape, dtype and byte length disagree")
            for name, info in sorted(entries, key=lambda item: (item[1]["data_offsets"][0], item[1]["data_offsets"][1])):
                if info["data_offsets"][0] != expected_offset:
                    raise ValueError("Safetensors offsets overlap or contain gaps")
                expected_offset = info["data_offsets"][1]
                if name == "__metadata__":
                    continue
                if info["dtype"] not in DTYPE_BYTES:
                    raise ValueError(f"Unsupported stored dtype {info['dtype']}")
                if name in seen:
                    raise ValueError("Duplicate tensor across checkpoint shards")
                seen.add(name)
                group = classify(name, count, layers_per_chunk)
                if group.startswith("blocks-"):
                    language_layers.add(int(re.search(r"(?:^|\.)layers\.(\d+)\.", name)[1]))
                groups[group].append((file, name, info, n + 8))
                total_tensor_bytes += info["data_offsets"][1] - info["data_offsets"][0]
            if src.local and (src.root / file).stat().st_size != n + 8 + expected_offset:
                raise ValueError("Safetensors file length does not match header")
        if not groups or not groups.get("embedding") or not groups.get("head"):
            raise ValueError("Checkpoint lacks complete language model tensors")
        if language_layers != set(range(count)):
            raise ValueError("Checkpoint is missing language layers")
        from .bundles import MAX_BUNDLE
        from .resources import CAUTION
        if total_tensor_bytes + total_header_bytes > MAX_BUNDLE:
            raise ValueError("Checkpoint exceeds STAGEPLANE_BUNDLE_MAX_BYTES; increase the configured bundle byte budget")
        if shutil.disk_usage(store).free < total_tensor_bytes + total_header_bytes + 64 * 1024**2:
            raise ValueError(CAUTION)
        artifacts = {}
        for group, tensors in sorted(groups.items()):
            name = f"{group}.safetensors"
            path = target / name
            offset, header = 0, {"__metadata__": {"format": "pt"}}
            ordered = sorted(tensors, key=lambda t: (t[0], t[2]["data_offsets"][0]))
            for _, tensor, info, _ in ordered:
                length = info["data_offsets"][1] - info["data_offsets"][0]
                header[tensor] = {**info, "data_offsets": [offset, offset + length]}
                offset += length
            raw = json.dumps(header, separators=(",", ":")).encode()
            raw += b" " * (-len(raw) % 8)
            tmp = path.with_suffix(".partial")
            with tmp.open("wb") as out:
                out.write(struct.pack("<Q", len(raw)))
                out.write(raw)
                # Merge adjacent tensors into one download, capped to 32 MiB.
                runs = []
                for file, _, info, base in ordered:
                    start, stop = [base + x for x in info["data_offsets"]]
                    if runs and runs[-1][0] == file and runs[-1][2] == start and stop - runs[-1][1] <= 32 * 1024**2:
                        runs[-1][2] = stop
                    else:
                        runs.append([file, start, stop])
                segments = ((file, pos, min(stop, pos + 16 * 1024**2) - 1)
                            for file, start, stop in runs for pos in range(start, stop, 16 * 1024**2))
                # Four bounded range reads overlap link and dispatch latency. Keep
                # at most four results resident, even on a multi-terabyte model.
                with ThreadPoolExecutor(max_workers=4) as pool:
                    while batch := list(islice(segments, 4)):
                        futures = [pool.submit(src.read, *segment) for segment in batch]
                        for future in futures:
                            block = future.result()
                            if shutil.disk_usage(target).free < len(block) + 64 * 1024**2:
                                raise ValueError(CAUTION)
                            out.write(block)
                        print(f"{group}: downloading {out.tell() / 1024**2:.1f} / {offset / 1024**2:.1f} MiB", flush=True)
            tmp.replace(path)
            artifacts[group] = {"file": name, "sha256": digest(path), "bytes": path.stat().st_size,
                                "tensor_count": len(ordered)}
            print(f"{group}: {path.stat().st_size / 1024**2:.1f} MiB", flush=True)
        for name in src.files:
            if "/" not in name and (name.startswith(("tokenizer", "vocab", "merges", "chat_template", "generation_config")) or name in ("LICENSE", "README.md")):
                content = src.read(name)
                if shutil.disk_usage(target).free < len(content) + 64 * 1024**2:
                    raise ValueError(CAUTION)
                (target / name).write_bytes(content)
        stages = [{"id": "embedding", "kind": "embedding", "start": 0, "end": 0, "artifacts": ["embedding"]}]
        for start in range(0, count, layers_per_chunk):
            sid = f"blocks-{start // layers_per_chunk:03d}"
            if sid not in artifacts:
                raise ValueError(f"Missing layers at {start}")
            stages.append({"id": sid, "kind": "blocks", "start": start,
                           "end": min(count, start + layers_per_chunk), "artifacts": [sid]})
        head_artifacts = ["head"]
        if text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
            head_artifacts.append("embedding")
        stages.append({"id": "head", "kind": "head", "start": count, "end": count, "artifacts": head_artifacts})
        for s in stages:
            s["weight_bytes"] = sum(artifacts[a]["bytes"] for a in s["artifacts"])
        files = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in target.iterdir() if p.is_file()}
        manifest = {"schema": 1, "key": key, "source": source, "revision": src.revision,
                    "aliases": [requested_source] if requested_source != source else [],
                    "architecture": text["model_type"], "config": config,
                    "layers_per_chunk": layers_per_chunk, "stages": stages, "artifacts": artifacts,
                    "files": files, "header_sha256": hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest(),
                    "tokenizer_sha256": files.get("tokenizer.json", {}).get("sha256"),
                    "activation_dtype": "float32", "layout": "BSH", "text_only": True,
                    "created": time.time()}
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        atomic_json(target / "manifest.json", manifest)
        # Verify the final inventory before exposing a discoverable revision.
        from .bundles import validate_manifest
        validate_manifest(manifest)
        if destination.exists():
            raise ValueError("Immutable revision destination already exists")
        target.rename(destination)
        staging = None
        return manifest
    finally:
        src.close()
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def manifests(store: Path = ROOT):
    return {m["key"]: m for p in store.glob("*/manifest.json") if not p.parent.name.startswith(".") and (m := json.loads(p.read_text()))}


def rechunk(directory: Path, store: Path, layers_per_chunk=1):
    """Repartition verified original tensors without another checkpoint download."""
    from .bundles import read_manifest, verify_files
    original = read_manifest(directory / "manifest.json")
    verify_files(directory, original)
    store.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rechunk-", dir=store) as staging:
        m = distribute(str(directory.resolve()), Path(staging), layers_per_chunk=layers_per_chunk)
        temporary = Path(staging) / m["key"]
        m.update(key=model_key(original["source"], original["revision"]) + f"-l{layers_per_chunk}",
                 source=original["source"], revision=original["revision"],
                 derived_from_manifest=original["manifest_sha256"])
        m.pop("manifest_sha256")
        m["manifest_sha256"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
        destination = store / m["key"]
        if destination.exists():
            raise ValueError("Rechunk destination already exists")
        atomic_json(temporary / "manifest.json", m)
        temporary.rename(destination)
        return m

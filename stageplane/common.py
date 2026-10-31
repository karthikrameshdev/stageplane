from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, Request, WebSocket
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(os.environ.get("STAGEPLANE_STORE", "data/models")).resolve()
MAX_FRAME = 64 * 1024 * 1024
MAX_CONTEXT = int(os.environ.get("STAGEPLANE_MAX_CONTEXT", "2048"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def model_key(source: str, revision: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "--", source)[-100:] + "-" + revision[:12]


def safe_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path outside artifact directory")
    return path


def auth_headers():
    token = os.environ.get("STAGEPLANE_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def tls_kwargs():
    import ssl
    context = ssl.create_default_context(cafile=os.environ.get("STAGEPLANE_CA"))
    if os.environ.get("STAGEPLANE_CLIENT_CERT"):
        context.load_cert_chain(os.environ["STAGEPLANE_CLIENT_CERT"], os.environ["STAGEPLANE_CLIENT_KEY"])
    return {"verify": context, "trust_env": False}


async def authorize(request: Request):
    token = os.environ.get("STAGEPLANE_TOKEN")
    if token:
        import secrets
        if not secrets.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
            raise HTTPException(401, "Invalid bearer token")


async def authorize_ws(ws: WebSocket):
    token = os.environ.get("STAGEPLANE_TOKEN")
    if token and ws.headers.get("authorization") != f"Bearer {token}":
        await ws.close(code=1008)
        return False
    return True


class NodeInfo(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    url: str
    grpc_url: str = ""
    transport: Literal["rest", "grpc", "websocket"] = "rest"
    group: int = Field(default=0, ge=0)
    memory_bytes: int = Field(default=4 * 1024**3, gt=0)
    cpu_cores: float | None = Field(default=None, gt=0)
    speed: float = Field(default=1, ge=0.01)
    failure_domain: str = ""
    ready: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)


class DeployRequest(BaseModel):
    model: str
    replicas: int = Field(default=1, ge=1, le=32)
    groups: int = Field(default=1, ge=1, le=100)


class ImportRequest(BaseModel):
    source: str = Field(min_length=1, max_length=1024)
    revision: str = Field(default="main", max_length=128)
    layers_per_chunk: int = Field(default=2, ge=1, le=256)
    kind: Literal["checkpoint", "bundle"] = "checkpoint"
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    subdir: str = "."


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    stream: bool = False
    seed: int = 0


def now():
    return time.time()

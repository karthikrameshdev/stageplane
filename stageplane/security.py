"""Executor credentials, role boundaries, and bounded JSON request ingestion."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse


@dataclass(frozen=True)
class Principal:
    tenant: str
    role: str
    internal: bool = False


class AccessPolicy:
    def __init__(self):
        self.internal_token = os.environ.get("STAGEPLANE_TOKEN", "")
        self.keys = {}
        filename = os.environ.get("STAGEPLANE_API_KEYS_FILE")
        if filename:
            if not self.internal_token:
                raise ValueError("STAGEPLANE_API_KEYS_FILE requires STAGEPLANE_TOKEN for worker authentication")
            try:
                with Path(filename).open("rb") as source:
                    raw = source.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError
                entries = json.loads(raw)
                if not isinstance(entries, list) or not entries or len(entries) > 10000:
                    raise ValueError
                for item in entries:
                    if set(item) != {"token", "tenant", "role"}:
                        raise ValueError
                    token, tenant, role = item["token"], item["tenant"], item["role"]
                    if not isinstance(token, str) or not 16 <= len(token) <= 512 or not token.isascii():
                        raise ValueError
                    if not isinstance(tenant, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tenant):
                        raise ValueError
                    if role not in ("admin", "viewer", "inference") or token == self.internal_token:
                        raise ValueError
                    digest = hashlib.sha256(token.encode()).digest()
                    if digest in self.keys:
                        raise ValueError
                    self.keys[digest] = Principal(tenant, role)
            except (OSError, ValueError, TypeError, KeyError):
                raise ValueError("Invalid API key configuration; expected unique token/tenant/role entries") from None

    def principal(self, token: str):
        if not self.internal_token and not self.keys:
            return Principal("default", "admin", True)
        if not isinstance(token, str) or len(token) > 512 or not token.isascii():
            raise HTTPException(401, "Invalid bearer token")
        if self.internal_token and secrets.compare_digest(token, self.internal_token):
            return Principal("default", "admin", True)
        result = self.keys.get(hashlib.sha256(token.encode()).digest())
        if result is None:
            raise HTTPException(401, "Invalid bearer token")
        return result

    async def authorize(self, request: Request):
        header = request.headers.get("authorization", "")
        principal = self.principal(header[7:] if header.startswith("Bearer ") else "")
        path, method = request.url.path, request.method
        worker_operation = (path == "/api/nodes/register" or path.endswith("/heartbeat")
                            or path.startswith("/api/artifacts/") or path.startswith("/api/manifests/"))
        if worker_operation:
            allowed = principal.internal
        elif path == "/v1/chat/completions":
            allowed = principal.role in ("admin", "inference")
        elif path == "/v1/models":
            allowed = True
        elif method == "GET":
            # A viewer can observe operations, but cannot export model weights.
            allowed = principal.role == "admin" or (principal.role == "viewer" and not path.startswith("/api/bundles/"))
        else:
            allowed = principal.role == "admin"
        if not allowed:
            raise HTTPException(403, "Credential role does not permit this operation")
        request.state.principal = principal
        return principal

    def websocket_principal(self, token):
        principal = self.principal(token)
        if principal.role not in ("admin", "viewer"):
            raise HTTPException(403, "Credential role does not permit operational events")
        return principal


class RequestBodyLimit:
    """Bound memory before JSON parsing, including clients without Content-Length."""
    def __init__(self, app, max_bytes=None):
        self.app = app
        self.max_bytes = int(max_bytes or os.environ.get("STAGEPLANE_MAX_REQUEST_BYTES", 2 * 1024**2))
        if not 1024 <= self.max_bytes <= 64 * 1024**2:
            raise ValueError("STAGEPLANE_MAX_REQUEST_BYTES must be between 1024 and 67108864")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        length = next((v for k, v in scope.get("headers", []) if k == b"content-length"), None)
        try:
            if length is not None and (int(length) < 0 or int(length) > self.max_bytes):
                return await JSONResponse({"detail": "Request body exceeds configured byte limit"}, status_code=413)(scope, receive, send)
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
        body, total = bytearray(), 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_bytes:
                return await JSONResponse({"detail": "Request body exceeds configured byte limit"}, status_code=413)(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        body = bytes(body)
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        await self.app(scope, replay, send)

"""Bounded JSON envelope + Safetensors payload; never pickle remote objects."""
from __future__ import annotations

import json
import asyncio
import os
import ssl
import struct

import grpc
import httpx
import torch
import websockets
from safetensors.torch import load, save

from .common import MAX_FRAME, auth_headers, tls_kwargs


class ChunkFailure(RuntimeError):
    def __init__(self, node, chunk, code="transport_error"):
        self.node, self.chunk, self.code = node, chunk, code
        super().__init__(f"{code}: {chunk} on {node}")


def error_frame(error):
    return b"STAGEPLANE_ERROR\n" + json.dumps({"node": error.node, "chunk": error.chunk, "code": error.code}).encode()


def pack(meta: dict, tensor: torch.Tensor):
    header = json.dumps(meta, separators=(",", ":")).encode()
    if len(header) > 65536:
        raise ValueError("Oversized envelope")
    data = struct.pack("<I", len(header)) + header + save({"value": tensor.contiguous()})
    if len(data) > MAX_FRAME:
        raise ValueError("Tensor frame exceeds limit")
    return data


def unpack(data: bytes):
    if len(data) < 12 or len(data) > MAX_FRAME:
        raise ValueError("Invalid frame size")
    size = struct.unpack("<I", data[:4])[0]
    if size > 65536 or 4 + size >= len(data):
        raise ValueError("Invalid envelope size")
    meta = json.loads(data[4:4 + size])
    tensor = load(data[4 + size:])["value"]
    if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16, torch.int64):
        raise ValueError("Unsupported tensor dtype")
    if tensor.ndim not in (2, 3) or tensor.shape[0] != 1:
        raise ValueError("Only batch-one tensors are supported")
    return meta, tensor


class Transport:
    def __init__(self):
        self.http = httpx.AsyncClient(timeout=60, headers=auth_headers(), **tls_kwargs())
        self.channels = {}
        self.metrics = {"request_payload_bytes": 0, "response_payload_bytes": 0, "calls": 0}

    async def send(self, node: dict, data: bytes):
        self.metrics["request_payload_bytes"] += len(data)
        self.metrics["calls"] += 1
        try:
            response = await asyncio.wait_for(self._send(node, data), float(os.environ.get("STAGEPLANE_ROUTE_RPC_TIMEOUT", "180")))
        except ChunkFailure:
            raise
        except (TimeoutError, httpx.TransportError, grpc.aio.AioRpcError, OSError, websockets.exceptions.WebSocketException) as e:
            raise ChunkFailure(node.get("node", node.get("id", "unknown")), node.get("chunk", "unknown"), "transport_timeout" if isinstance(e, (TimeoutError, httpx.TimeoutException)) else "transport_error") from e
        self.metrics["response_payload_bytes"] += len(response)
        if response.startswith(b"STAGEPLANE_ERROR\n"):
            info = json.loads(response.split(b"\n", 1)[1])
            raise ChunkFailure(info["node"], info["chunk"], info["code"])
        return response

    async def _send(self, node: dict, data: bytes):
        mode = node.get("transport", "rest")
        if mode == "grpc":
            target = node["grpc_url"]
            if target not in self.channels:
                options = [("grpc.max_receive_message_length", MAX_FRAME), ("grpc.max_send_message_length", MAX_FRAME)]
                if os.environ.get("STAGEPLANE_CA"):
                    from pathlib import Path
                    ca = Path(os.environ["STAGEPLANE_CA"]).read_bytes()
                    cert = Path(os.environ["STAGEPLANE_CLIENT_CERT"]).read_bytes() if os.environ.get("STAGEPLANE_CLIENT_CERT") else None
                    key = Path(os.environ["STAGEPLANE_CLIENT_KEY"]).read_bytes() if cert else None
                    self.channels[target] = grpc.aio.secure_channel(target, grpc.ssl_channel_credentials(ca, key, cert), options=options)
                else:
                    self.channels[target] = grpc.aio.insecure_channel(target, options=options)
            call = self.channels[target].unary_unary("/stageplane.Worker/Forward")
            return await call(data, timeout=float(os.environ.get("STAGEPLANE_ROUTE_RPC_TIMEOUT", "180")), metadata=[(k.lower(), v) for k, v in auth_headers().items()])
        if mode == "websocket":
            url = node["url"].replace("https://", "wss://").replace("http://", "ws://") + "/ws/forward"
            kwargs = {}
            if url.startswith("wss:"):
                ctx = ssl.create_default_context(cafile=os.environ.get("STAGEPLANE_CA"))
                if os.environ.get("STAGEPLANE_CLIENT_CERT"):
                    ctx.load_cert_chain(os.environ["STAGEPLANE_CLIENT_CERT"], os.environ["STAGEPLANE_CLIENT_KEY"])
                kwargs["ssl"] = ctx
            async with websockets.connect(url, additional_headers=auth_headers(), max_size=MAX_FRAME,
                                          open_timeout=10, close_timeout=2, **kwargs) as ws:
                await ws.send(data)
                result = await ws.recv()
                if not isinstance(result, bytes):
                    raise ValueError(result)
                return result
        r = await self.http.post(node["url"] + "/v1/forward", content=data,
                                 headers={"Content-Type": "application/octet-stream"}, timeout=float(os.environ.get("STAGEPLANE_ROUTE_RPC_TIMEOUT", "180")))
        if r.status_code >= 500:
            raise ChunkFailure(node.get("node", "unknown"), node.get("chunk", "unknown"), "worker_error")
        r.raise_for_status()
        return r.content

    async def close(self):
        await self.http.aclose()
        for c in self.channels.values():
            await c.close()

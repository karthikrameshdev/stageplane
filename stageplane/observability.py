"""Prometheus text exposition, structured logs, and request correlation without extra dependencies."""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import threading
import time
import uuid

REQUEST_ID = contextvars.ContextVar("stageplane_request_id", default="")
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_LABEL = re.compile(r"[^a-zA-Z0-9_.:/-]")


def label(value):
    return _LABEL.sub("_", str(value))[:128]


class Histogram:
    def __init__(self, buckets=LATENCY_BUCKETS):
        self.buckets, self.counts, self.total, self.count = buckets, [0] * len(buckets), 0.0, 0

    def observe(self, value):
        self.total += value
        self.count += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1

    def lines(self, name, labels=""):
        sep = "," if labels else ""
        for bound, count in zip(self.buckets, self.counts):
            yield f'{name}_bucket{{{labels}{sep}le="{bound}"}} {count}'
        yield f'{name}_bucket{{{labels}{sep}le="+Inf"}} {self.count}'
        yield f"{name}_sum{{{labels}}} {self.total:.6f}"
        yield f"{name}_count{{{labels}}} {self.count}"


class Metrics:
    """Executor counters keyed by model and tenant; label values are sanitized and bounded."""
    def __init__(self):
        self.lock = threading.Lock()
        self.requests, self.tokens, self.prompt_tokens, self.recoveries = {}, {}, {}, {}
        self.ttft, self.token_latency, self.duration = {}, {}, {}
        self.http = {}

    def observe(self, record):
        key = (label(record.get("model", "")), label(record.get("tenant", "")))
        with self.lock:
            status = (*key, record.get("status", "unknown"))
            self.requests[status] = self.requests.get(status, 0) + 1
            self.tokens[key] = self.tokens.get(key, 0) + record.get("output_tokens", 0)
            self.prompt_tokens[key] = self.prompt_tokens.get(key, 0) + record.get("prompt_tokens", 0)
            self.recoveries[key] = self.recoveries.get(key, 0) + record.get("recoveries", 0)
            if record.get("ttft_ms") is not None:
                self.ttft.setdefault(key[0], Histogram()).observe(record["ttft_ms"] / 1000)
            for ms in record.get("token_ms", [])[1:]:
                self.token_latency.setdefault(key[0], Histogram()).observe(ms / 1000)
            if record.get("elapsed_ms") is not None:
                self.duration.setdefault(key[0], Histogram()).observe(record["elapsed_ms"] / 1000)

    def http_request(self, method, route, status, seconds):
        key = (method, route, str(status // 100) + "xx")
        with self.lock:
            self.http.setdefault(key, Histogram()).observe(seconds)

    def render(self, coordinator):
        out = []

        def family(name, kind, help_text):
            out.append(f"# HELP {name} {help_text}")
            out.append(f"# TYPE {name} {kind}")

        with self.lock:
            family("stageplane_requests_total", "counter", "Completed, failed, or cancelled chat requests.")
            for (model, tenant, status), value in sorted(self.requests.items()):
                out.append(f'stageplane_requests_total{{model="{model}",tenant="{tenant}",status="{status}"}} {value}')
            for name, data, text in (("stageplane_completion_tokens_total", self.tokens, "Generated tokens."),
                                     ("stageplane_prompt_tokens_total", self.prompt_tokens, "Prompt tokens."),
                                     ("stageplane_recoveries_total", self.recoveries, "Prefix-replay recoveries.")):
                family(name, "counter", text)
                for (model, tenant), value in sorted(data.items()):
                    out.append(f'{name}{{model="{model}",tenant="{tenant}"}} {value}')
            for name, data, text in (("stageplane_time_to_first_token_seconds", self.ttft, "Time to first token."),
                                     ("stageplane_inter_token_latency_seconds", self.token_latency, "Decode step latency."),
                                     ("stageplane_request_duration_seconds", self.duration, "End-to-end request time.")):
                family(name, "histogram", text)
                for model, histogram in sorted(data.items()):
                    out.extend(histogram.lines(name, f'model="{model}"'))
            family("stageplane_http_request_duration_seconds", "histogram", "Executor HTTP handling time.")
            for (method, route, status), histogram in sorted(self.http.items()):
                out.extend(histogram.lines("stageplane_http_request_duration_seconds",
                                           f'method="{method}",route="{label(route)}",status="{status}"'))
        nodes = list(coordinator.nodes.values())
        family("stageplane_nodes", "gauge", "Registered workers by state.")
        for state in ("active", "draining", "inactive"):
            count = sum((n.get("draining") and state == "draining") or (not n.get("draining") and (
                n.get("active") if state == "active" else not n.get("active") and state == "inactive")) for n in nodes)
            out.append(f'stageplane_nodes{{state="{state}"}} {count}')
        family("stageplane_active_requests", "gauge", "Requests currently generating.")
        out.append(f"stageplane_active_requests {len(coordinator.active_requests)}")
        family("stageplane_node_loaded_weight_bytes", "gauge", "Weights resident in worker memory.")
        family("stageplane_node_available_memory_bytes", "gauge", "Worker-reported free memory.")
        family("stageplane_node_sessions", "gauge", "Worker resident session caches.")
        for n in nodes:
            m, nid = n.get("metrics", {}), label(n["id"])
            out.append(f'stageplane_node_loaded_weight_bytes{{node="{nid}"}} {m.get("loaded_weight_bytes", 0)}')
            out.append(f'stageplane_node_available_memory_bytes{{node="{nid}"}} {m.get("available_memory_bytes", 0)}')
            out.append(f'stageplane_node_sessions{{node="{nid}"}} {m.get("sessions", 0)}')
        family("stageplane_deployment_ready_stages", "gauge", "Stages with at least one ready replica.")
        family("stageplane_deployment_total_stages", "gauge", "Stages in the deployed manifest.")
        for model, d in coordinator.deployments.items():
            if model not in coordinator.models:
                continue
            status = coordinator.deployment_status(model, d)
            out.append(f'stageplane_deployment_ready_stages{{model="{label(model)}"}} {status["ready_stages"]}')
            out.append(f'stageplane_deployment_total_stages{{model="{label(model)}"}} {status["total_stages"]}')
        family("stageplane_transport_payload_bytes_total", "counter", "Executor tensor bytes sent and received.")
        t = coordinator.transport.metrics
        out.append(f'stageplane_transport_payload_bytes_total{{direction="sent"}} {t["request_payload_bytes"]}')
        out.append(f'stageplane_transport_payload_bytes_total{{direction="received"}} {t["response_payload_bytes"]}')
        family("stageplane_build_info", "gauge", "Executor version.")
        from . import __version__
        out.append(f'stageplane_build_info{{version="{__version__}"}} 1')
        return "\n".join(out) + "\n"


def worker_metrics(stats):
    m, nid = stats.get("metrics", {}), label(stats.get("id", ""))
    lines = []
    for key in ("loaded_weight_bytes", "rss_bytes", "available_memory_bytes", "reserved_session_bytes",
                "cache_bytes", "sessions", "max_sessions", "calls", "bytes_received", "bytes_sent"):
        name = f"stageplane_worker_{key}"
        lines += [f"# TYPE {name} {'counter' if key in ('calls', 'bytes_received', 'bytes_sent') else 'gauge'}",
                  f'{name}{{node="{nid}"}} {m.get(key, 0)}']
    for chunk, ms in sorted(m.get("stage_ms", {}).items()):
        lines.append(f'stageplane_worker_stage_last_seconds{{node="{nid}",chunk="{label(chunk)}"}} {ms / 1000:.6f}')
    stream = m.get("layer_stream", {})
    for key, value in sorted(stream.items()):
        lines.append(f'stageplane_worker_layer_stream_{key}{{node="{nid}"}} {value}')
    return "\n".join(lines) + "\n"


class JsonFormatter(logging.Formatter):
    def format(self, record):
        item = {"time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
                "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        if REQUEST_ID.get():
            item["request_id"] = REQUEST_ID.get()
        for key in ("event", "node", "model", "tenant", "status"):
            if hasattr(record, key):
                item[key] = getattr(record, key)
        if record.exc_info:
            item["exception"] = self.formatException(record.exc_info)
        return json.dumps(item, default=str)


def configure_logging(level=None, fmt=None):
    level = (level or os.environ.get("STAGEPLANE_LOG_LEVEL", "INFO")).upper()
    fmt = fmt or os.environ.get("STAGEPLANE_LOG_FORMAT", "text")
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if fmt == "json" else
                         logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger("stageplane")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False


_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class RequestContext:
    """Propagates or assigns X-Request-ID and records per-route latency."""
    def __init__(self, app, metrics=None):
        self.app, self.metrics = app, metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"x-request-id"), "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = REQUEST_ID.set(request_id)
        scope.setdefault("state", {})["request_id"] = request_id
        started, status = time.perf_counter(), [500]

        async def wrapped(message):
            if message["type"] == "http.response.start":
                status[0] = message["status"]
                message["headers"] = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
            await send(message)
        try:
            await self.app(scope, receive, wrapped)
        finally:
            if self.metrics is not None:
                route = scope.get("route")
                path = getattr(route, "path", None) or ("/ui" if scope["path"].startswith("/ui/") else "unmatched")
                self.metrics.http_request(scope["method"], path, status[0], time.perf_counter() - started)
            REQUEST_ID.reset(token)

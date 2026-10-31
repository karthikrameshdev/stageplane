"""Capacity-aware rendezvous placement with group-scoped membership changes."""
from __future__ import annotations

import hashlib
import math
from .resources import MIN_CPU_CORES, weight_budget, weight_factor


def bucket(value, groups):
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % groups


def score(chunk, node):
    u = (int(hashlib.sha256(f"{chunk}:{node['id']}".encode()).hexdigest()[:16], 16) + 1) / (2**64 + 1)
    # Weighted rendezvous is stable as unrelated workers join or leave.
    speed = max(0.01, node.get("speed", 1))
    speed *= max(.005, min(2, node.get("metrics", {}).get("available_cpu_cores", 1)))
    latency = node.get("metrics", {}).get("stage_ms", {}).get(chunk)
    if latency:
        speed /= max(latency / 100, 0.1)
    return -math.log(u) / speed


def plan(manifest, nodes, replicas=1, groups=1, previous=None, changed_groups=None):
    previous = previous or {}
    active = {n["id"]: n for n in nodes if not n.get("draining") and n.get("active", True)
              and (n.get("metrics", {}).get("available_cpu_cores", 1) >= MIN_CPU_CORES
                   or n.get("metrics", {}).get("loading") or n.get("metrics", {}).get("sessions"))}
    assigned = {nid: 0 for nid in active}
    stored = {nid: 0 for nid in active}
    result = {}
    # Retain healthy assignments outside membership groups being rebalanced.
    for stage in manifest["stages"]:
        sid = stage["id"]
        group = bucket(sid, groups)
        unchanged = changed_groups is not None and group not in changed_groups
        retained = [n for n in previous.get(sid, []) if n in active
                    and active[n].get("group", 0) % groups == group
                    and (unchanged or active[n].get("metrics", {}).get("loading")
                         or active[n].get("metrics", {}).get("sessions"))]
        if retained:
            result[sid] = retained
            for nid in retained:
                assigned[nid] += stage["weight_bytes"] * weight_factor(active[nid])
                stored[nid] += stage["weight_bytes"]
    for stage in sorted(manifest["stages"], key=lambda s: -s["weight_bytes"]):
        sid = stage["id"]
        if sid in result and len(result[sid]) >= replicas:
            continue
        group = bucket(sid, groups)
        candidates = [n for n in active.values() if n.get("group", 0) % groups == group]
        # No cross-group fallback: it would turn a local join into global movement.
        candidates.sort(key=lambda n: score(sid, n))
        selected = result.get(sid, [])
        domains = {active[n].get("failure_domain") or n for n in selected}
        for node in candidates:
            nid = node["id"]
            if node.get("metrics", {}).get("available_cpu_cores", 1) < MIN_CPU_CORES:
                continue
            domain = node.get("failure_domain") or nid
            # Conservative CPU F32 expansion of BF16 weights plus working reserve.
            needed = stage["weight_bytes"] * weight_factor(node) + 64 * 1024**2
            if nid in selected or domain in domains or assigned[nid] + needed > weight_budget(node):
                continue
            if stored[nid] + stage["weight_bytes"] > node.get("metrics", {}).get("available_storage_bytes", 2**63):
                continue
            selected.append(nid)
            domains.add(domain)
            assigned[nid] += stage["weight_bytes"] * weight_factor(node)
            stored[nid] += stage["weight_bytes"]
            if len(selected) >= replicas:
                break
        result[sid] = selected
    return result


def route(manifest, placements, nodes, previous=None):
    chosen, last = [], None
    for stage in manifest["stages"]:
        aid = manifest["key"] + ":" + stage["id"]
        ready = [nodes[nid] for nid in placements.get(stage["id"], [])
                 if nid in nodes and nodes[nid].get("active") and aid in nodes[nid].get("ready", [])]
        if not ready:
            raise RuntimeError(f"No ready replica for {stage['id']}")
        node = min(ready, key=lambda n: (n["id"] != last, n.get("metrics", {}).get("sessions", 0),
                                       score(stage["id"], n)))
        chosen.append({"chunk": stage["id"], "node": node["id"], "url": node["url"],
                       "grpc_url": node.get("grpc_url", ""), "transport": node.get("transport", "rest")})
        last = node["id"]
    return chosen

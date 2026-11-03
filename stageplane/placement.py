"""Capacity-aware rendezvous placement with group-scoped membership changes."""
from __future__ import annotations

import hashlib
import math
from .resources import MIN_CPU_CORES, cache_reserve, resident_cost, weight_budget, working_set


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


def plan(manifest, nodes, replicas=1, groups=1, previous=None, changed_groups=None,
         reserved=None, storage_reserved=None, resident=None, strategy="rendezvous"):
    previous = previous or {}
    changed_groups = {g % groups for g in changed_groups} if changed_groups is not None else None
    reserved, storage_reserved = reserved or {}, storage_reserved or {}
    resident = resident or {}
    active = {n["id"]: n for n in nodes if not n.get("draining") and n.get("active", True)
              and (n.get("metrics", {}).get("available_cpu_cores", 1) >= MIN_CPU_CORES
                   or n.get("metrics", {}).get("loading") or n.get("metrics", {}).get("sessions"))}
    kinds = {stage["id"]: stage for stage in manifest["stages"]}
    # Pinned streamed layers occupy up to layer_cache_bytes and are also reported in
    # loaded_weight_bytes, which weight_budget adds back as reclaimable; reserve it here.
    assigned = {nid: reserved.get(nid, 0) + cache_reserve(active[nid]) + sum(resident_cost(active[nid], kinds[sid])
                    for sid in resident.get(nid, set()) if sid in kinds) for nid in active}
    stored = {nid: storage_reserved.get(nid, 0) for nid in active}
    streamed = dict.fromkeys(active, 0)
    candidates_by_group = {}
    for node in active.values():
        candidates_by_group.setdefault(node.get("group", 0) % groups, []).append(node)
    ready_sets = {nid: set(node.get("ready", [])) for nid, node in active.items()}

    def memory_cost(stage, nid):
        return 0 if stage["id"] in resident.get(nid, set()) else resident_cost(active[nid], stage)

    def fill(nid, stage):
        # Fraction of the node's own capacity used after taking this stage, so small and
        # large nodes fill evenly and layers interleave across nodes in proportion to size.
        node = active[nid]
        if node.get("streaming"):
            # Streamed layers live on disk, but speed depends on how many fit in RAM, so share
            # is still proportional to the node's memory. Streaming nodes rank behind resident
            # nodes and absorb overflow.
            return 1 + (streamed[nid] + stage["weight_bytes"]) / max(1, weight_budget(node))
        return (assigned[nid] + memory_cost(stage, nid)) / max(1, weight_budget(node))

    def disk_cost(stage, nid):
        # Free disk excludes files that the worker has already verified and loaded.
        aid = manifest.get("key", "") + ":" + stage["id"]
        return 0 if aid in ready_sets[nid] else stage["weight_bytes"]
    result = {}
    # Retain healthy assignments outside membership groups being rebalanced.
    for stage in manifest["stages"]:
        sid = stage["id"]
        group = bucket(sid, groups)
        unchanged = changed_groups is not None and group not in changed_groups
        retained = [n for n in previous.get(sid, []) if n in active
                    and active[n].get("group", 0) % groups == group
                    and (unchanged or active[n].get("metrics", {}).get("loading"))]
        retained = retained[:replicas]
        for nid in retained:
            cost = memory_cost(stage, nid)
            if assigned[nid] + max(cost, working_set(active[nid], stage)) + 64 * 1024**2 > weight_budget(active[nid]):
                continue
            if stored[nid] + disk_cost(stage, nid) > active[nid].get("metrics", {}).get("available_storage_bytes", 2**63):
                continue
            result.setdefault(sid, []).append(nid)
            assigned[nid] += cost
            stored[nid] += disk_cost(stage, nid)
            streamed[nid] += stage["weight_bytes"]
    for stage in sorted(manifest["stages"], key=lambda s: -s["weight_bytes"]):
        sid = stage["id"]
        if sid in result and len(result[sid]) >= replicas:
            continue
        group = bucket(sid, groups)
        candidates = list(candidates_by_group.get(group, []))
        # No cross-group fallback: it would turn a local join into global movement.
        if strategy == "capacity":
            candidates.sort(key=lambda n: (fill(n["id"], stage), score(sid, n)))
        else:
            candidates.sort(key=lambda n: score(sid, n))
        selected = result.get(sid, [])
        domains = {active[n].get("failure_domain") or n for n in selected}
        for node in candidates:
            nid = node["id"]
            if node.get("metrics", {}).get("available_cpu_cores", 1) < MIN_CPU_CORES:
                continue
            domain = node.get("failure_domain") or nid
            # Conservative CPU F32 expansion of BF16 weights plus working reserve.
            needed = max(memory_cost(stage, nid), working_set(node, stage)) + 64 * 1024**2
            if nid in selected or domain in domains or assigned[nid] + needed > weight_budget(node):
                continue
            if stored[nid] + disk_cost(stage, nid) > node.get("metrics", {}).get("available_storage_bytes", 2**63):
                continue
            selected.append(nid)
            domains.add(domain)
            assigned[nid] += memory_cost(stage, nid)
            stored[nid] += disk_cost(stage, nid)
            streamed[nid] += stage["weight_bytes"]
            if len(selected) >= replicas:
                break
        result[sid] = selected
    return result


def route(manifest, placements, nodes, previous=None):
    chosen, last = [], None
    ready_sets = {}
    for stage in manifest["stages"]:
        aid = manifest["key"] + ":" + stage["id"]
        ready = []
        for nid in placements.get(stage["id"], []):
            node = nodes.get(nid)
            if not node or not node.get("active") or node.get("draining"):
                continue
            metrics = node.get("metrics", {})
            if metrics.get("sessions", 0) >= metrics.get("max_sessions", float("inf")):
                continue
            if nid not in ready_sets:
                ready_sets[nid] = set(node.get("ready", []))
            if aid in ready_sets[nid]:
                ready.append(node)
        if not ready:
            raise RuntimeError(f"No ready replica for {stage['id']}")
        node = min(ready, key=lambda n: (n["id"] != last, n.get("metrics", {}).get("sessions", 0),
                                       score(stage["id"], n)))
        chosen.append({"chunk": stage["id"], "node": node["id"], "url": node["url"],
                       "grpc_url": node.get("grpc_url", ""), "transport": node.get("transport", "rest")})
        last = node["id"]
    return chosen

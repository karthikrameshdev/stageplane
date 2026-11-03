"""Current worker headroom, including Docker cgroup limits."""
from pathlib import Path
import math
import os
import re
import psutil

CAUTION = "Not enough space available to run this model. Kindly add more nodes."
MEMORY_CAUTION = "Not enough memory available to run this model. Kindly add more nodes."
COMPUTE_CAUTION = "Not enough compute available to run this model. Kindly add more nodes."
MIN_CPU_CORES = float(os.environ.get("STAGEPLANE_MIN_CPU_CORES", ".01"))


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def available_limits():
    host = psutil.virtual_memory()
    raw = _read("/sys/fs/cgroup/memory.max")
    memory = min(int(raw) if raw and raw != "max" else host.total, host.total)
    cores = float(psutil.cpu_count() or 1)
    try:
        cores = min(cores, len(psutil.Process().cpu_affinity()))
    except (AttributeError, NotImplementedError, psutil.AccessDenied):
        pass
    raw_cpu = _read("/sys/fs/cgroup/cpu.max")
    if raw_cpu:
        quota, period = raw_cpu.split()
        if quota != "max":
            cores = min(cores, int(quota) / int(period))
    return memory, cores


def resolve_limits(memory="100%", cpu="100%"):
    total, cores = available_limits()
    if memory.endswith("%"):
        fraction = float(memory[:-1]) / 100
        if not 0 < fraction <= 1:
            raise ValueError("Memory percentage must be between 0 and 100")
        memory_bytes = int(total * fraction)
    else:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB|KiB|MiB|GiB)?", memory, re.IGNORECASE)
        if not match:
            raise ValueError("Memory must be bytes, MB, MiB, GB, GiB, or a percentage")
        units = {"b": 1, "kb": 1000, "mb": 10**6, "gb": 10**9, "kib": 1024, "mib": 1024**2, "gib": 1024**3}
        memory_bytes = int(float(match[1]) * units[(match[2] or "B").lower()])
    if cpu.endswith("%"):
        fraction = float(cpu[:-1]) / 100
        if not 0 < fraction <= 1:
            raise ValueError("CPU percentage must be between 0 and 100")
        cpu_cores = cores * fraction
    else:
        cpu_cores = float(cpu[:-1]) / 1000 if cpu.endswith("m") else float(cpu)
    if memory_bytes <= 0 or not math.isfinite(cpu_cores) or cpu_cores <= 0:
        raise ValueError("Resource limits must be positive")
    return min(total, memory_bytes), min(cores, cpu_cores)


def sample(declared_bytes, process, declared_cpu=None):
    host = psutil.virtual_memory()
    memory, cores = available_limits()
    limit = min(declared_bytes, memory)
    cores = min(cores, declared_cpu) if declared_cpu is not None else cores
    raw_current = _read("/sys/fs/cgroup/memory.current")
    rss = process.memory_info().rss
    used = int(raw_current) if raw_current else rss
    process_cpu = process.cpu_percent(None)
    host_idle = (psutil.cpu_count() or 1) * (1 - psutil.cpu_percent(None) / 100)
    return {"rss_bytes": rss, "memory_limit_bytes": limit, "memory_used_bytes": used,
            "available_memory_bytes": max(0, min(host.available, limit - used)),
            "cpu_limit_cores": cores, "cpu_usage_percent": process_cpu,
            "available_cpu_cores": max(0, min(cores - process_cpu / 100, host_idle))}


def weight_factor(node):
    return 1 if node.get("metrics", {}).get("dtype") in ("bfloat16", "float16") else 2


def working_set(node, stage):
    """Peak transient memory a streaming node needs for a stage: the executing layer plus the
    prefetched next layer. Zero for resident nodes, which account the whole stage instead."""
    if not node.get("streaming") or stage.get("kind") != "blocks":
        return 0
    layers = max(1, stage.get("end", 1) - stage.get("start", 0))
    return stage["weight_bytes"] * weight_factor(node) * min(2, layers) // layers


def resident_cost(node, stage):
    """Memory a placed stage permanently holds; streamed layers live on disk and in a shared per-node cache."""
    if node.get("streaming") and stage.get("kind") == "blocks":
        return 0
    return stage["weight_bytes"] * weight_factor(node)


def cache_reserve(node):
    """RAM a streaming node permanently dedicates to pinned layers (its --layer-cache)."""
    return node.get("layer_cache_bytes", 0) if node.get("streaming") else 0


def weight_budget(node):
    metrics = node.get("metrics", {})
    total = min(node["memory_bytes"], metrics.get("memory_limit_bytes", node["memory_bytes"]))
    if "available_memory_bytes" not in metrics:
        return total
    weights = metrics.get("loaded_weight_bytes", 0)
    reserve = max(96 * 1024**2, metrics.get("rss_bytes", 0) - weights)
    return max(0, min(total - reserve, metrics["available_memory_bytes"] + weights))


def capacity(manifest, placements, nodes, groups=1):
    from .placement import bucket
    missing = [s["id"] for s in manifest["stages"] if not placements.get(s["id"])]
    active = [n for n in nodes if n.get("active", True) and not n.get("draining")]
    reason = "memory"
    reserved = {n["id"]: cache_reserve(n) for n in active}
    stored = {n["id"]: 0 for n in active}
    by_id = {n["id"]: n for n in active}
    for stage in manifest["stages"]:
        for nid in placements.get(stage["id"], []):
            if nid in by_id:
                reserved[nid] += resident_cost(by_id[nid], stage)
                stored[nid] += stage["weight_bytes"]
    for stage in manifest["stages"]:
        if stage["id"] not in missing:
            continue
        candidates = [n for n in active if n.get("group", 0) % groups == bucket(stage["id"], groups)]
        cpu_nodes = [n for n in candidates if n.get("metrics", {}).get("available_cpu_cores", 1) >= MIN_CPU_CORES]
        if candidates and not cpu_nodes:
            reason = "compute"
            break
        memory_nodes = [n for n in cpu_nodes if reserved[n["id"]] + max(resident_cost(n, stage), working_set(n, stage))
                        + 64 * 1024**2 <= weight_budget(n)]
        if memory_nodes and all(stored[n["id"]] + stage["weight_bytes"] >
                                n.get("metrics", {}).get("available_storage_bytes", MAX_STORAGE) for n in memory_nodes):
            reason = "storage"
            break
    message = {"compute": COMPUTE_CAUTION, "memory": MEMORY_CAUTION, "storage": CAUTION}[reason]
    return {"status": "insufficient_capacity" if missing else "planned",
            "message": message if missing else "Model placement fits the available nodes.",
            "reason": reason if missing else None,
            "unassigned_stages": missing,
            "available_weight_budget_bytes": sum(weight_budget(n) for n in nodes if n.get("active", True) and not n.get("draining")),
            "required_stage_storage_bytes": sum(s["weight_bytes"] for s in manifest["stages"])}


MAX_STORAGE = 2**63

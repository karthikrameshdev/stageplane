"""Reproducible placement stress test; allocates metadata, never model weights."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path

from stageplane.placement import bucket, plan


def scenario(parameters, stages, workers, memory_gib, groups, replicas=2):
    total_bytes = parameters * 2
    stage_bytes = total_bytes // stages
    manifest = {"key": "synthetic", "stages": [
        {"id": f"blocks-{i:04d}", "weight_bytes": stage_bytes + (total_bytes % stages if i == 0 else 0)}
        for i in range(stages)]}
    nodes = [{"id": f"worker-{i}", "group": i % groups, "active": True,
              "failure_domain": f"host-{i}", "memory_bytes": memory_gib * 1024**3,
              "metrics": {"dtype": "bfloat16", "available_cpu_cores": 8,
                          "available_storage_bytes": memory_gib * 2 * 1024**3}}
             for i in range(workers)]
    begin = time.perf_counter()
    placements = plan(manifest, nodes, replicas=replicas, groups=groups)
    planning = time.perf_counter() - begin
    assert all(len(owners) == replicas for owners in placements.values()), "Insufficient synthetic capacity"
    memory = {n["id"]: 0 for n in nodes}
    for stage in manifest["stages"]:
        for owner in placements[stage["id"]]:
            memory[owner] += stage["weight_bytes"]
    assert max(memory.values()) + 64 * 1024**2 <= memory_gib * 1024**3
    victim = next(iter(placements.values()))[0]
    failed_group = next(n["group"] for n in nodes if n["id"] == victim)
    remaining = [n for n in nodes if n["id"] != victim]
    begin = time.perf_counter()
    recovered = plan(manifest, remaining, replicas=replicas, groups=groups,
                     previous=placements, changed_groups={failed_group})
    recovery = time.perf_counter() - begin
    assert all(len(owners) == replicas and victim not in owners for owners in recovered.values())
    outside = [sid for sid in placements if bucket(sid, groups) != failed_group]
    moved_outside = sum(placements[sid] != recovered[sid] for sid in outside)
    assert moved_outside == 0, "Failure moved an unrelated placement group"
    return {"synthetic_parameters": parameters, "bf16_weight_bytes": total_bytes,
            "stages": stages, "workers": workers, "worker_memory_gib": memory_gib,
            "groups": groups, "replicas": replicas, "placement_seconds": planning,
            "failure_rebalance_seconds": recovery, "unaffected_group_changes": moved_outside,
            "max_assigned_weight_bytes_per_worker": max(memory.values()),
            "stage_bytes": stage_bytes, "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/hardening/scale.json"))
    args = parser.parse_args()
    results = [scenario(*case) for case in [
        (100_000_000_000, 80, 32, 64, 4),
        (1_000_000_000_000, 128, 128, 96, 8),
        (10_000_000_000_000, 256, 1024, 256, 16),
        (10_000_000_000_000, 256, 10_000, 256, 32),
    ]]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"kind": "synthetic-placement-only", "platform": platform.platform(),
              "peak_process_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
              "assumptions": "BF16, equal whole-stage weights, unique failure domains, idle workers; no tensor execution, KV cache, network or gateway throughput measured.",
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

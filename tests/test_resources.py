from types import SimpleNamespace

from stageplane import resources
from stageplane.placement import plan, score


def test_cgroup_memory_and_cpu_are_used(monkeypatch):
    values = {"/sys/fs/cgroup/memory.max": "100000000", "/sys/fs/cgroup/memory.current": "70000000",
              "/sys/fs/cgroup/cpu.max": "5000 100000"}
    monkeypatch.setattr(resources, "_read", values.get)
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=20_000_000), cpu_percent=lambda _: 1)
    metrics = resources.sample(1024**3, process)
    assert metrics["memory_limit_bytes"] == 100_000_000
    assert metrics["available_memory_bytes"] <= 30_000_000
    assert metrics["cpu_limit_cores"] == .05
    assert metrics["available_cpu_cores"] <= .04


def test_current_headroom_and_cpu_affect_placement():
    manifest = {"stages": [{"id": "blocks-0", "weight_bytes": 200_000_000}]}
    nodes = [{"id": "busy", "memory_bytes": 500_000_000,
              "metrics": {"dtype": "bfloat16", "available_memory_bytes": 20_000_000,
                          "rss_bytes": 480_000_000, "available_cpu_cores": .001}},
             {"id": "free", "memory_bytes": 500_000_000,
              "metrics": {"dtype": "bfloat16", "available_memory_bytes": 400_000_000,
                          "rss_bytes": 100_000_000, "available_cpu_cores": .05}}]
    placement = plan(manifest, nodes)
    assert placement["blocks-0"] == ["free"]
    for n in nodes:
        n["metrics"]["available_memory_bytes"] = 1
    assert resources.capacity(manifest, plan(manifest, nodes), nodes)["message"] == resources.MEMORY_CAUTION


def test_default_and_explicit_allocations_publish_numbers(monkeypatch):
    monkeypatch.setattr(resources, "available_limits", lambda: (16_000_000_000, 8.0))
    assert resources.resolve_limits() == (16_000_000_000, 8.0)
    assert resources.resolve_limits("500MB", "50m") == (500_000_000, .05)
    assert resources.resolve_limits("25%", "50%") == (4_000_000_000, 4.0)


def test_distinct_compute_memory_and_storage_cautions():
    manifest = {"stages": [{"id": "blocks-0", "weight_bytes": 1000}]}
    def status(metrics):
        nodes = [{"id": "n", "memory_bytes": 10**9, "metrics": metrics}]
        return resources.capacity(manifest, plan(manifest, nodes), nodes)
    assert status({"available_cpu_cores": 0})["message"] == resources.COMPUTE_CAUTION
    assert status({"available_memory_bytes": 0})["message"] == resources.MEMORY_CAUTION
    assert status({"available_storage_bytes": 0})["message"] == resources.CAUTION


def test_storage_caution_identifies_a_large_stage_that_does_not_fit():
    # The small stage fits disk, while the larger stage has enough RAM but not disk.
    # Looking only at the smallest stage would incorrectly report a RAM shortage.
    manifest = {"stages": [{"id": "embedding", "weight_bytes": 800_000_000},
                           {"id": "blocks-0", "weight_bytes": 200_000_000}]}
    nodes = [{"id": "n", "memory_bytes": 4_000_000_000,
              "metrics": {"dtype": "bfloat16", "available_storage_bytes": 600_000_000}}]
    placements = plan(manifest, nodes)
    assert placements["blocks-0"] == ["n"] and not placements["embedding"]
    assert resources.capacity(manifest, placements, nodes)["message"] == resources.CAUTION


def test_loading_assignments_survive_transient_cpu_exhaustion():
    manifest = {"stages": [{"id": "blocks-0", "weight_bytes": 200_000_000},
                           {"id": "blocks-1", "weight_bytes": 200_000_000}]}
    nodes = [{"id": "loading", "memory_bytes": 2_000_000_000,
              "metrics": {"loading": True, "available_cpu_cores": 0}},
             {"id": "idle", "memory_bytes": 2_000_000_000,
              "metrics": {"available_cpu_cores": .05}}]
    previous = {"blocks-0": ["loading"]}
    placement = plan(manifest, nodes, previous=previous, changed_groups={0})
    assert placement == {"blocks-0": ["loading"], "blocks-1": ["idle"]}
    # Busy nodes retain their reservations, but receive no new work without CPU.
    nodes[0]["metrics"]["loading"] = False
    assert plan(manifest, nodes, previous=previous)["blocks-0"] == ["idle"]

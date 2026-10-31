"""Run a local four-worker pool until Ctrl-C. Never downloads a model."""
import argparse
import signal
from pathlib import Path

from scripts.cluster import Cluster

p = argparse.ArgumentParser()
p.add_argument("--store", type=Path, default=Path("data/models"))
p.add_argument("--work", type=Path, default=Path("data/dev"))
p.add_argument("--nodes", type=int, default=4)
p.add_argument("--replicas", type=int, default=1)
p.add_argument("--groups", type=int, default=1)
p.add_argument("--port", type=int, default=8080)
p.add_argument("--local-artifacts", action="store_true", help="Hardlink existing artifacts instead of downloading worker copies")
p.add_argument("--token", default="local-stageplane-development")
p.add_argument("--model", help="Select a specific immutable model key")
p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
p.add_argument("--memory-gb", type=float, default=6)
p.add_argument("--cpu-kernel", choices=["native", "bounded-f32"], default="native")
a = p.parse_args()
c = Cluster(a.store, a.work, a.nodes, a.replicas, a.groups, port=a.port, token=a.token, local_artifacts=a.local_artifacts, dtype=a.dtype, memory_gb=a.memory_gb, cpu_kernel=a.cpu_kernel)
try:
    c.start(a.model)
    print(f"Dashboard: {c.url}\nModel: {c.model}\nUse --token value in the dashboard access dialog.", flush=True)
    signal.pause()
except KeyboardInterrupt:
    pass
finally:
    c.close()

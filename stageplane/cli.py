from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="stageplane")
    commands = parser.add_subparsers(dest="command", required=True)
    dist = commands.add_parser("distribute", help="Download and partition a compatible checkpoint")
    dist.add_argument("source", help="HF repo, local checkpoint, static HTTP directory, or ollama:<tag> with --hf-source")
    dist.add_argument("--hf-source", help="Exact original Safetensors repository for an Ollama model")
    dist.add_argument("--revision", default="main")
    dist.add_argument("--layers-per-chunk", type=int, default=2)
    dist.add_argument("--store", type=Path, default=Path("data/models"), help="Output root for versioned model manifests and chunk files")
    rec = commands.add_parser("rechunk", help="Change block grouping using existing verified model tensors")
    rec.add_argument("directory", type=Path)
    rec.add_argument("--store", type=Path, default=Path("data/models"), help="Output root for the new chunk layout")
    rec.add_argument("--layers-per-chunk", type=int, default=1)
    export = commands.add_parser("export", help="Package a verified model for public hosting")
    export.add_argument("model", help="Immutable model key")
    export.add_argument("--store", type=Path, default=Path("data/models"), help="Input root containing the model to export")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--format", choices=["archive", "directory"], default="archive")
    bundle = commands.add_parser("import-bundle", help="Import published stages without downloading the original checkpoint")
    bundle.add_argument("source", help="Local path, HTTP URL, git+URL, or public Google Drive file URL")
    bundle.add_argument("--store", type=Path, default=Path("data/models"), help="Output root for verified imported model artifacts")
    bundle.add_argument("--manifest-sha256")
    bundle.add_argument("--revision", default="HEAD")
    bundle.add_argument("--subdir", default=".")
    exe = commands.add_parser("executor")
    exe.add_argument("--store", type=Path, default=Path("data/models"),
                     help="Model root produced by distribute/import-bundle; reads ROOT/MODEL_KEY/manifest.json")
    exe.add_argument("--artifact-source", action="append", default=None, metavar="ROOT",
                     help="Worker-readable artifact root: absolute path, local file:// URL, or HTTP(S) URL; repeat up to 8. Defaults to --store")
    exe.add_argument("--host", default="127.0.0.1")
    exe.add_argument("--port", type=int, default=8080)
    exe.add_argument("--bundle", action="append", default=[], help="Import a local/remote published bundle before serving; repeatable")
    node = commands.add_parser("node")
    node.add_argument("--id", required=True)
    node.add_argument("--executor", default="http://127.0.0.1:8080")
    node.add_argument("--host", default="127.0.0.1")
    node.add_argument("--port", type=int, default=8101)
    node.add_argument("--advertise-url")
    node.add_argument("--grpc-port", type=int, default=0)
    node.add_argument("--grpc-url", default="")
    node.add_argument("--transport", choices=["rest", "grpc", "websocket"], default="rest")
    node.add_argument("--group", type=int, default=0)
    node.add_argument("--failure-domain", default="")
    node.add_argument("--memory-gb", type=float, help="Legacy GiB serving budget; prefer --max-memory")
    node.add_argument("--max-memory", default="100%", help="Examples: 500MB, 2GiB, 50%%; default 100%% of node/container capacity")
    node.add_argument("--max-cpu", "--max-compute", default="100%", help="Cores, millicores (50m), or percent; default 100%% of node/container capacity")
    node.add_argument("--speed", type=float, default=1)
    node.add_argument("--cache", type=Path, default=Path("data/nodes"), help="Local worker cache root; files are stored below NODE_ID")
    node.add_argument("--artifact-source", action="append", default=None, metavar="ROOT",
                      help="Node-local artifact root tried before executor-advertised roots; absolute path, local file:// URL, or HTTP(S); repeat up to 8")
    node.add_argument("--device", default="cpu")
    node.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    node.add_argument("--cpu-kernel", choices=["native", "bounded-f32"], default="native")
    node.add_argument("--threads", type=int, help="Native compute threads, bounded by the allocated CPU cores")
    node.add_argument("--max-sessions", type=int, default=8)
    args = parser.parse_args()
    artifact_sources = None
    if args.command in ("executor", "node"):
        from .artifacts import normalize_sources
        try:
            if args.artifact_source is not None:
                artifact_sources = normalize_sources(args.artifact_source)
        except ValueError as error:
            parser.error(str(error))
    if args.command == "rechunk":
        from .distribution import rechunk
        print(rechunk(args.directory, args.store, args.layers_per_chunk)["key"])
        return
    if args.command == "distribute":
        from .distribution import distribute
        m = distribute(args.hf_source or args.source, args.store.resolve(), args.revision, args.layers_per_chunk)
        print(f"Model: {m['key']}\nStages: {len(m['stages'])}\nManifest: {args.store / m['key'] / 'manifest.json'}")
        return
    if args.command in ("export", "import-bundle"):
        from .bundles import export_bundle, import_bundle
        import json
        if args.command == "export":
            from .common import safe_path
            result = export_bundle(safe_path(args.store.resolve(), args.model), args.output, args.format)
        else:
            m = import_bundle(args.source, args.store, args.manifest_sha256, args.revision, args.subdir)
            result = {"model": m["key"], "manifest_sha256": m["manifest_sha256"], "location": str(args.store.resolve() / m["key"])}
        print(json.dumps(result, indent=2))
        return
    if args.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("STAGEPLANE_TOKEN"):
        parser.error("Set STAGEPLANE_TOKEN before binding to a non-loopback interface")
    if args.command == "node":
        os.environ["STAGEPLANE_CPU_KERNEL"] = args.cpu_kernel
        import json
        import math
        from .resources import available_limits, resolve_limits
        try:
            memory_bytes, cpu_cores = resolve_limits(str(args.memory_gb) + "GiB" if args.memory_gb is not None else args.max_memory, args.max_cpu)
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps({"node": args.id, "max_memory_bytes": memory_bytes, "max_cpu_cores": cpu_cores}), flush=True)
        if cpu_cores < available_limits()[1] - .000001:
            import atexit
            import subprocess
            import sys
            governor = subprocess.Popen([sys.executable, "-m", "stageplane.cpu_governor", str(os.getpid()), str(cpu_cores)],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            atexit.register(governor.terminate)
    import uvicorn
    tls = {}
    if os.environ.get("STAGEPLANE_TLS_CERT"):
        import ssl
        tls = {"ssl_certfile": os.environ["STAGEPLANE_TLS_CERT"], "ssl_keyfile": os.environ["STAGEPLANE_TLS_KEY"],
               "ssl_ca_certs": os.environ["STAGEPLANE_CA"], "ssl_cert_reqs": ssl.CERT_REQUIRED}
    if args.command == "executor":
        from .bundles import import_bundle
        for source in args.bundle:
            import_bundle(source, args.store)
        from .executor import Coordinator, create_app
        app = create_app(Coordinator(args.store.resolve(), artifact_sources=artifact_sources))
    else:
        import torch
        from .common import NodeInfo
        from .node import Worker, create_app
        torch.set_num_threads(min(args.threads or max(1, math.ceil(cpu_cores)), max(1, math.ceil(cpu_cores))))
        info = NodeInfo(id=args.id, url=args.advertise_url or f"http://127.0.0.1:{args.port}",
                        grpc_url=args.grpc_url or f"127.0.0.1:{args.grpc_port}", transport=args.transport,
                        group=args.group, memory_bytes=memory_bytes, cpu_cores=cpu_cores, speed=args.speed,
                        failure_domain=args.failure_domain or args.id)
        app = create_app(Worker(info, args.executor, args.cache / args.id, args.device, args.dtype, args.max_sessions,
                               artifact_sources=artifact_sources), args.grpc_port)
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=64 * 1024**2, **tls)


if __name__ == "__main__":
    main()

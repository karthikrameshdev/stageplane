import subprocess

import pytest
import httpx

from scripts.cluster import Cluster
from scripts.tiny_checkpoint import make
from stageplane.distribution import distribute


def test_mutual_tls_all_transports_and_openai_sdk(tmp_path):
    from openai import OpenAI
    certs = tmp_path / "certs"
    subprocess.run(["sh", "scripts/dev_certs.sh", str(certs)], check=True, capture_output=True)
    make(tmp_path / "source")
    distribute(str(tmp_path / "source"), tmp_path / "models", layers_per_chunk=1)
    c = Cluster(tmp_path / "models", tmp_path / "run", nodes=4, replicas=1, certs=certs)
    try:
        c.start()
        # The fixture places work across REST, gRPC and WebSocket workers.
        with pytest.raises(httpx.HTTPError):
            import ssl
            httpx.get(c.url + "/health", verify=ssl.create_default_context(cafile=str(certs / "ca.crt")), trust_env=False, timeout=3)
        sdk = OpenAI(api_key="test-stageplane-token", base_url=c.url + "/v1", http_client=c.http)
        result = sdk.chat.completions.create(model=c.model,
            messages=[{"role": "user", "content": "hello world"}], max_tokens=4, temperature=0)
        assert result.choices[0].message.role == "assistant"
        chunks = list(sdk.chat.completions.create(model=c.model,
            messages=[{"role": "user", "content": "hello world"}], max_tokens=4, stream=True, temperature=0))
        assert chunks[-1].choices[0].finish_reason in ("stop", "length")
        assert "".join(x.choices[0].delta.content or "" for x in chunks) == result.choices[0].message.content
        events = c.http.get("/api/state").json()["events"]
        trace_nodes = {t["node"] for e in events if e["kind"] == "token_committed" for t in e["trace"]}
        assert {"node-1", "node-2"} <= trace_nodes
    finally:
        c.close()

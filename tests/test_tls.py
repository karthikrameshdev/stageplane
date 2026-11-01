import subprocess

import pytest
import httpx

from scripts.cluster import Cluster
from scripts.tiny_checkpoint import make
from stageplane.distribution import distribute
from stageplane.placement import bucket


def test_mutual_tls_all_transports_and_openai_sdk(tmp_path):
    from openai import OpenAI
    certs = tmp_path / "certs"
    subprocess.run(["sh", "scripts/dev_certs.sh", str(certs)], check=True, capture_output=True)
    make(tmp_path / "source")
    manifest = distribute(str(tmp_path / "source"), tmp_path / "models", layers_per_chunk=1)
    # One transport per placement group forces real network transitions even
    # when live resource telemetry makes a single worker otherwise preferable.
    assert {bucket(stage["id"], 3) for stage in manifest["stages"]} == {0, 1, 2}
    c = Cluster(tmp_path / "models", tmp_path / "run", nodes=4, replicas=1, groups=3,
                transports=["rest", "grpc", "websocket", "rest"], certs=certs)
    try:
        c.start()
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
        for request_id in (result.id, chunks[-1].id):
            started = next(event for event in events
                           if event["kind"] == "request_started" and event["request"] == request_id)
            route = started["route"]
            network_hops = [route[0], *(successor for previous, successor in zip(route, route[1:])
                                       if previous["node"] != successor["node"])]
            assert {hop["transport"] for hop in network_hops} == {"rest", "grpc", "websocket"}
            committed = [event for event in events
                         if event["kind"] == "token_committed" and event["request"] == request_id]
            assert committed
            for event in committed:
                trace_nodes = {entry["node"] for entry in event["trace"]}
                assert {hop["node"] for hop in network_hops} <= trace_nodes
    finally:
        c.close()

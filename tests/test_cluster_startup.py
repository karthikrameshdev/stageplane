from unittest.mock import Mock

import httpx
import pytest

from scripts.cluster import Cluster


@pytest.fixture
def startup(tmp_path, monkeypatch):
    clusters = []

    def make(status, deployment=None, confirmation_status=200):
        cluster = Cluster(tmp_path / "models", tmp_path / "work", nodes=1, port=18880)
        cluster.http.close()
        requests = []
        posted = False

        def handle(request):
            nonlocal posted
            requests.append((request.method, request.url.path))
            if request.method == "POST":
                posted = True
                assert request.url.path == "/api/deployments"
                return httpx.Response(status, json={"detail": "admission result"})
            assert request.url.path == "/api/state"
            return httpx.Response(confirmation_status if posted else 200, json={
                "models": [{"key": "model"}], "nodes": [{"id": "node-0"}],
                "deployments": {"model": deployment} if deployment is not None else {},
            })

        cluster.http = httpx.Client(base_url=cluster.url, transport=httpx.MockTransport(handle))
        monkeypatch.setattr(cluster, "spawn", Mock())
        monkeypatch.setattr(cluster, "wait_http", Mock())
        monkeypatch.setattr(cluster, "start_node", Mock())
        monkeypatch.setattr(cluster, "wait_ready", Mock())
        clusters.append(cluster)
        return cluster, requests

    yield make
    for cluster in clusters:
        cluster.close()


def pending(**changes):
    return {"replicas": 1, "groups": 1, "capacity": {"status": "insufficient_capacity"}, **changes}


def test_persisted_capacity_503_uses_existing_bounded_readiness_wait(startup):
    cluster, requests = startup(503, pending())
    assert cluster.start() is cluster
    cluster.wait_ready.assert_called_once_with()
    assert requests.count(("POST", "/api/deployments")) == 1
    assert requests[-1] == ("GET", "/api/state")


@pytest.mark.parametrize("deployment", [None, pending(capacity={"status": "planned"}),
                                         pending(replicas=2), pending(groups=2)])
def test_unconfirmed_503_still_raises(startup, deployment):
    cluster, requests = startup(503, deployment)
    with pytest.raises(httpx.HTTPStatusError) as error:
        cluster.start()
    assert error.value.response.status_code == 503
    cluster.wait_ready.assert_not_called()
    assert requests.count(("POST", "/api/deployments")) == 1


def test_server_500_raises_without_capacity_confirmation(startup):
    cluster, requests = startup(500, pending())
    with pytest.raises(httpx.HTTPStatusError) as error:
        cluster.start()
    assert error.value.response.status_code == 500
    assert requests[-1] == ("POST", "/api/deployments")
    cluster.wait_ready.assert_not_called()


def test_successful_admission_waits_for_readiness(startup):
    cluster, requests = startup(200)
    assert cluster.start() is cluster
    cluster.wait_ready.assert_called_once_with()
    assert requests[-1] == ("POST", "/api/deployments")


def test_capacity_confirmation_http_error_is_not_ignored(startup):
    cluster, _ = startup(503, pending(), confirmation_status=500)
    with pytest.raises(httpx.HTTPStatusError) as error:
        cluster.start()
    assert error.value.response.status_code == 500
    cluster.wait_ready.assert_not_called()


def test_pending_deployment_does_not_hide_readiness_timeout(startup):
    cluster, _ = startup(503, pending())
    cluster.wait_ready.side_effect = TimeoutError("Incomplete route")
    with pytest.raises(TimeoutError, match="Incomplete route"):
        cluster.start()
    cluster.wait_ready.assert_called_once_with()

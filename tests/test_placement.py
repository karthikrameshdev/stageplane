from stageplane.placement import bucket, plan


def fixture():
    m = {"stages": [{"id": f"blocks-{i:03d}", "weight_bytes": 1024} for i in range(24)]}
    nodes = [{"id": f"n{i}", "memory_bytes": 10**9, "group": i % 2, "failure_domain": f"host{i}"} for i in range(4)]
    return m, nodes


def test_node_count_independent_of_chunks():
    m, nodes = fixture()
    for count in [1, 4, 100]:
        pool = [{**nodes[0], "id": f"n{i}", "failure_domain": f"h{i}"} for i in range(count)]
        p = plan(m, pool)
        assert all(len(owners) == 1 for owners in p.values())
        assert len(p) == 24
    p = plan(m, [{**nodes[0], "memory_bytes": 1}])
    assert all(not v for v in p.values())


def test_join_leave_only_moves_affected_group():
    m, nodes = fixture()
    p = plan(m, nodes, groups=2)
    joined = nodes + [{**nodes[0], "id": "new", "failure_domain": "new"}]
    q = plan(m, joined, groups=2, previous=p, changed_groups={0})
    for sid in p:
        if bucket(sid, 2) == 1:
            assert p[sid] == q[sid]
    gone = nodes[0]["id"]
    q = plan(m, nodes[1:], groups=2, previous=p, changed_groups={0})
    assert all(gone not in owners for owners in q.values())


def test_replication_failure_domains():
    m, nodes = fixture()
    nodes[1]["failure_domain"] = nodes[0]["failure_domain"]
    p = plan(m, nodes, replicas=4)
    assert all(len(owners) == 3 for owners in p.values())

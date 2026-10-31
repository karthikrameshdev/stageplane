import json
import subprocess

from scripts.resource_profile import save_evidence


def test_log_timeout_cannot_discard_completed_measurements(tmp_path, monkeypatch):
    result = {"registered_nodes": 50, "oom_count": 5, "completed_inference": False}
    def timeout(*args, **kwargs):
        # The primary measurements must already be durable before logs are read.
        assert json.loads((tmp_path / "result.json").read_text())["oom_count"] == 5
        raise subprocess.TimeoutExpired(args[0], 10)
    monkeypatch.setattr(subprocess, "run", timeout)
    save_evidence(tmp_path, result, ["stageplane-review-executor"])
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["registered_nodes"] == 50 and saved["oom_count"] == 5
    assert len(saved["log_capture_errors"]) == 1

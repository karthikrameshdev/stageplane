import pytest

from stageplane.bundles import NAME
from stageplane.common import model_key


@pytest.mark.parametrize("source", [
    "/tmp/checkpoint",
    "./checkpoint",
    "_checkpoint",
    "---",
    "/" + "a" * 100 + "_" * 99 + "x",
])
def test_generated_local_keys_are_accepted_by_bundle_validation(source):
    key = model_key(source, "0123456789abcdef")
    assert NAME.fullmatch(key)
    assert key.endswith("-0123456789ab")
    assert len(key) <= 113


def test_existing_hugging_face_key_format_is_preserved():
    assert model_key("Qwen/Qwen3-4B-Instruct-2507", "0123456789abcdef") == (
        "Qwen--Qwen3-4B-Instruct-2507-0123456789ab"
    )

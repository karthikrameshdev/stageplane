import json

import pytest
import torch

from scripts.tiny_checkpoint import make
from stageplane.distribution import distribute, resolve_source
from stageplane.runtime import LocalPipeline, load_reference
from stageplane.wire import pack, unpack


@pytest.mark.parametrize("architecture", ["qwen3", "qwen2", "llama", "mistral", "qwen3_5_text"])
def test_native_prefill_decode_parity(tmp_path, architecture):
    torch.set_num_threads(1)
    original = make(tmp_path / "original", architecture).eval()
    manifest = distribute(str(tmp_path / "original"), tmp_path / "store", layers_per_chunk=1)
    assert len(manifest["stages"]) == 6
    pipeline = LocalPipeline(manifest, tmp_path / "store" / manifest["key"])
    assert pipeline.stages[0].embed_tokens.weight.data_ptr() == pipeline.stages[-1].lm_head.weight.data_ptr()
    native = load_reference(manifest, tmp_path / "store" / manifest["key"])
    tokens = torch.tensor([[6, 7, 8]])
    cache = None
    for offset in [0, 3, 4]:
        with torch.inference_mode():
            result = native(tokens, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            distributed = pipeline.forward(tokens, offset)
        torch.testing.assert_close(distributed[:, -1], result.logits[:, -1], rtol=1e-5, atol=1e-5)
        if offset == 0:
            with torch.inference_mode():
                torch.testing.assert_close(result.logits, original(tokens).logits, rtol=1e-5, atol=1e-5)
        tokens = result.logits[:, -1].argmax(-1).unsqueeze(1)


def test_frame_roundtrip_and_bounds():
    tensor = torch.randn(1, 2, 16)
    meta, value = unpack(pack({"session": "test"}, tensor))
    assert meta["session"] == "test"
    torch.testing.assert_close(value, tensor)
    for bad in [b"", b"\xff" * 20, pack({}, torch.zeros(2, 3))]:
        with pytest.raises(ValueError):
            unpack(bad)


def test_immutable_artifacts_and_unsupported(tmp_path):
    make(tmp_path / "original")
    manifest = distribute(str(tmp_path / "original"), tmp_path / "store")
    with pytest.raises(ValueError, match="Immutable"):
        distribute(str(tmp_path / "original"), tmp_path / "store")
    with pytest.raises(ValueError, match="GGUF"):
        distribute("ollama:qwen3", tmp_path / "store")


def test_model_source_aliases():
    assert resolve_source("ollama:qwen3.5:0.8b") == "Qwen/Qwen3.5-0.8B"
    assert resolve_source("https://huggingface.co/Qwen/Qwen3.5-0.8B/") == "Qwen/Qwen3.5-0.8B"
    with pytest.raises(ValueError, match="exact original"):
        resolve_source("ollama:private-finetune:latest")

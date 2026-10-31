"""Native Transformers block kernels, with only the assigned weights materialized."""
from __future__ import annotations

import re
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel
from transformers.cache_utils import DynamicCache
from .kernels import configure_cpu_linears

_TENSORS = weakref.WeakValueDictionary()


def text_config(manifest):
    raw = manifest["config"].get("text_config", manifest["config"])
    cfg = AutoConfig.for_model(**raw)
    cfg._attn_implementation = "eager"
    return cfg


def canonical(name):
    for prefix in ("model.language_model.", "language_model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def read_weights(directory: Path, manifest, stage, dtype):
    result = {}
    for artifact in stage["artifacts"]:
        with safe_open(str(directory / manifest["artifacts"][artifact]["file"]), framework="pt", device="cpu") as f:
            for key in f.keys():
                name = canonical(key)
                if name.startswith(("layers.", "embed_tokens.", "norm.", "lm_head.")):
                    cache_key = (str(directory.resolve()), artifact, key, dtype)
                    value = _TENSORS.get(cache_key)
                    if value is None:
                        value = f.get_tensor(key).to(dtype=dtype, copy=True)
                        _TENSORS[cache_key] = value
                    result[name] = value
    return result


@dataclass
class StageState:
    cache: DynamicCache
    offset: int = 0
    touched: float = 0


class Stage(torch.nn.Module):
    def __init__(self, manifest: dict, stage: dict, directory: Path, device="cpu", dtype="float32"):
        super().__init__()
        self.spec = stage
        self.cfg = text_config(manifest)
        self.device_name = device
        self.dtype = getattr(torch, dtype)
        # Meta construction allocates no full-model parameter storage.
        with torch.device("meta"):
            skeleton = AutoModel.from_config(self.cfg, trust_remote_code=False)
        kind = stage["kind"]
        if kind == "embedding":
            self.embed_tokens = skeleton.embed_tokens
        elif kind == "head":
            self.norm = skeleton.norm
            self.lm_head = torch.nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False, device="meta")
        else:
            self.layers = torch.nn.ModuleDict({str(i): skeleton.layers[i] for i in range(stage["start"], stage["end"])})
            # Rotary buffers are computed, not serialized checkpoint parameters.
            self.rotary_emb = type(skeleton.rotary_emb)(config=self.cfg)
        del skeleton
        weights = read_weights(directory, manifest, stage, self.dtype)
        if kind == "head" and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["embed_tokens.weight"]
        expected = set(dict(self.named_parameters()))
        selected = {k: v for k, v in weights.items() if k in expected}
        if expected - selected.keys():
            raise ValueError(f"Incomplete {stage['id']}: {expected - selected.keys()}")
        self.load_state_dict(selected, strict=False, assign=True)
        self.to(device=device, dtype=self.dtype)
        if kind == "blocks":
            # Native rotary kernels retain FP32 frequencies at BF16 precision.
            self.rotary_emb = type(self.rotary_emb)(config=self.cfg).to(device=device)
        # Parameters wrap tensor storage without retaining the original Tensor
        # object. Keep CPU sources alive so the weak cache can share tied weights.
        self._shared_source_tensors = selected if device == "cpu" else {}
        configure_cpu_linears(self)
        self.eval()

    def state(self):
        return StageState(DynamicCache(config=self.cfg), touched=time.time())

    @torch.inference_mode()
    def forward(self, value: torch.Tensor, state: StageState, offset: int):
        kind = self.spec["kind"]
        value = value.to(self.device_name)
        if kind == "embedding":
            return self.embed_tokens(value.long()).cpu().contiguous()
        if kind == "head":
            return self.lm_head(self.norm(value[:, -1:, :].to(self.dtype))).float().cpu().contiguous()
        if offset != state.offset:
            raise ValueError(f"Cache offset {state.offset} does not match requested offset {offset}")
        value = value.to(self.dtype)
        length = value.shape[1]
        positions = torch.arange(offset, offset + length, device=self.device_name).unsqueeze(0)
        if self.cfg.model_type == "qwen3_5_text":
            rotary_positions = positions.unsqueeze(0).expand(3, -1, -1)
        else:
            rotary_positions = positions
        embeddings = self.rotary_emb(value, rotary_positions)
        keys = torch.arange(offset + length, device=self.device_name)
        allow = keys[None, :] <= positions[0, :, None]
        window = getattr(self.cfg, "sliding_window", None)
        if window:
            allow &= keys[None, :] > positions[0, :, None] - window
        mask = torch.zeros((1, 1, length, offset + length), dtype=self.dtype, device=self.device_name)
        mask.masked_fill_(~allow[None, None, :, :], torch.finfo(self.dtype).min)
        for _, layer in self.layers.items():
            linear = getattr(layer, "block_type", None) == "linear_attention"
            result = layer(value, attention_mask=None if linear else mask,
                           position_ids=positions, past_key_values=state.cache,
                           position_embeddings=embeddings, use_cache=True)
            value = result[0] if isinstance(result, tuple) else result
        state.offset += length
        state.touched = time.time()
        return value.cpu().contiguous()


class LocalPipeline:
    """An in-process staged reference. Benchmark direct uses the native full model."""
    def __init__(self, manifest, directory, **kwargs):
        self.stages = [Stage(manifest, s, directory, **kwargs) for s in manifest["stages"]]
        self.states = [s.state() for s in self.stages]

    def forward(self, tokens, offset=0):
        value = tokens
        for s, state in zip(self.stages, self.states):
            value = s(value, state, offset)
        return value


def load_reference(manifest, directory: Path, dtype="float32"):
    """Native unpartitioned model, loaded from the identical immutable artifacts."""
    from transformers import AutoModelForCausalLM
    cfg = text_config(manifest)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False)
    target = dict(model.named_parameters(remove_duplicate=False))
    weights = {}
    for a in manifest["artifacts"].values():
        with safe_open(str(directory / a["file"]), framework="pt", device="cpu") as f:
            for k in f.keys():
                name = canonical(k)
                mapped = name if name.startswith("lm_head.") else "model." + name
                if mapped in target:
                    weights[mapped] = f.get_tensor(k).to(getattr(torch, dtype), copy=True)
    if "lm_head.weight" not in weights and cfg.tie_word_embeddings:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    if target.keys() - weights.keys():
        raise ValueError(f"Reference missing {target.keys() - weights.keys()}")
    model.load_state_dict(weights, strict=False, assign=True)
    # Meta rotary buffers need a concrete CPU initialization.
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    model.eval()
    configure_cpu_linears(model)
    return model


def cache_budget(cfg, spec, tokens, bytes_per_element=4):
    """Conservative session allocation estimate, including hybrid recurrent state."""
    size = 0
    types = getattr(cfg, "layer_types", []) or []
    for i in range(spec["start"], spec["end"]):
        if types and types[i] == "linear_attention":
            heads = cfg.linear_num_value_heads
            size += heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim * 4
            conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + heads * cfg.linear_value_head_dim
            size += conv_dim * cfg.linear_conv_kernel_dim * bytes_per_element
        else:
            size += 2 * tokens * cfg.num_key_value_heads * getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads) * bytes_per_element
    return size + tokens * cfg.hidden_size * bytes_per_element * 4

"""Native Transformers block kernels, with only the assigned weights materialized."""
from __future__ import annotations

import copy
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def layer_index(directory: Path, manifest, stage):
    """Map layer number -> [(file, tensor key)] without reading any tensor data."""
    index = {}
    for artifact in stage["artifacts"]:
        path = str(directory / manifest["artifacts"][artifact]["file"])
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                match = re.match(r"layers\.(\d+)\.", canonical(key))
                if match:
                    index.setdefault(int(match[1]), []).append((path, key))
    return index


class LayerBudget:
    """Node-wide RAM budget shared by every streamed stage on a worker."""
    def __init__(self, limit_bytes):
        self.limit = max(0, int(limit_bytes))
        self.used = 0
        self.lock = threading.Lock()

    def reserve(self, size):
        with self.lock:
            if self.used + size > self.limit:
                return False
            self.used += size
            return True

    def release(self, size):
        with self.lock:
            self.used = max(0, self.used - size)


class LayerStreamer:
    """Keeps a stage's layers on disk and pins as many materialized layers as the node budget allows.

    Every token sweeps the same layers in order, which defeats LRU (a cyclic scan larger than
    the cache never hits). Pinning the layers that fit and streaming the rest transiently gives
    a hit rate of budget/working-set instead. Placement only changes speed: each layer computes
    with the identical tensors a resident stage would use, and the next layer is read while the
    current one computes.
    """
    def __init__(self, protos, index, dtype, budget, device):
        self.protos, self.index, self.dtype, self.device = protos, index, dtype, device
        self.budget = budget if isinstance(budget, LayerBudget) else LayerBudget(budget)
        self.pinned = {}
        self.bytes = 0
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="stageplane-prefetch")
        self.pending = {}
        self.stats = {"hits": 0, "misses": 0, "loaded_bytes": 0, "transient": 0}

    def _read(self, i):
        layer = copy.deepcopy(self.protos[str(i)])
        weights = {}
        for path, key in self.index[i]:
            with safe_open(path, framework="pt", device="cpu") as f:
                weights[canonical(key)[len(f"layers.{i}."):]] = f.get_tensor(key).to(dtype=self.dtype, copy=True)
        expected = set(dict(layer.named_parameters()))
        if expected - weights.keys():
            raise ValueError(f"Incomplete layer {i}: {expected - weights.keys()}")
        layer.load_state_dict({k: v for k, v in weights.items() if k in expected}, strict=False, assign=True)
        layer.to(device=self.device, dtype=self.dtype).eval()
        # Streamed layers must use the same linear kernel as resident ones.
        configure_cpu_linears(layer)
        size = sum(p.numel() * p.element_size() for p in layer.parameters())
        return layer, size

    def prefetch(self, i):
        with self.lock:
            if i in self.pinned or i in self.pending or str(i) not in self.protos:
                return
            self.pending[i] = self.pool.submit(self._read, i)

    def get(self, i):
        with self.lock:
            if i in self.pinned:
                self.stats["hits"] += 1
                return self.pinned[i]
            future = self.pending.pop(i, None)
        self.stats["misses"] += 1
        layer, size = (future or self.pool.submit(self._read, i)).result()
        with self.lock:
            self.stats["loaded_bytes"] += size
            if i not in self.pinned and self.budget.reserve(size):
                self.pinned[i] = layer
                self.bytes += size
            elif i not in self.pinned:
                self.stats["transient"] += 1
        return layer

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        with self.lock:
            self.budget.release(self.bytes)
            self.pinned.clear()
            self.bytes = 0


@dataclass
class StageState:
    cache: DynamicCache
    offset: int = 0
    touched: float = 0


class Stage(torch.nn.Module):
    def __init__(self, manifest: dict, stage: dict, directory: Path, device="cpu", dtype="float32",
                 stream_budget=None):
        super().__init__()
        self.streamer = None
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
        if kind == "blocks" and stream_budget is not None:
            protos, self.layers = self.layers, torch.nn.ModuleDict()
            self.streamer = LayerStreamer(protos, layer_index(directory, manifest, stage), self.dtype,
                                          stream_budget, device)
            self.rotary_emb = self.rotary_emb.to(device=device)
            self.eval()
            return
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

    def close(self):
        if self.streamer:
            self.streamer.close()

    def resident_bytes(self):
        if self.streamer:
            return self.streamer.bytes
        return sum({p.data_ptr(): p.numel() * p.element_size() for p in self.parameters()}.values())

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
        if self.streamer:
            order = list(range(self.spec["start"], self.spec["end"]))
            layers = []
            self.streamer.prefetch(order[0])
        else:
            layers = list(self.layers.items())
        for n in range(len(order) if self.streamer else len(layers)):
            if self.streamer:
                layer = self.streamer.get(order[n])
                if n + 1 < len(order):
                    self.streamer.prefetch(order[n + 1])
            else:
                layer = layers[n][1]
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


def cache_budget(cfg, spec, tokens, bytes_per_element=4, prefill_chunk=None):
    """Conservative session allocation estimate, including hybrid recurrent state.

    KV entries use the stage's element size. Activations and eager attention scores are
    bounded by the prefill chunk, because prompts never execute in larger pieces.
    """
    chunk = min(tokens, prefill_chunk or tokens)
    size = 0
    attention = False
    types = getattr(cfg, "layer_types", []) or []
    for i in range(spec["start"], spec["end"]):
        if types and types[i] == "linear_attention":
            heads = cfg.linear_num_value_heads
            size += heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim * 4
            conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + heads * cfg.linear_value_head_dim
            size += conv_dim * cfg.linear_conv_kernel_dim * bytes_per_element
        else:
            attention = True
            size += 2 * tokens * cfg.num_key_value_heads * getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads) * bytes_per_element
    if attention:
        # One layer's scores and mask exist at a time; softmax runs in float32.
        size += chunk * tokens * (cfg.num_attention_heads * 4 + bytes_per_element)
    return size + chunk * cfg.hidden_size * 4 * 4

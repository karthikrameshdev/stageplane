"""Deterministic offline checkpoint for correctness and fault integration tests."""
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast


def make(path: Path, architecture="qwen3"):
    torch.manual_seed(7)
    kwargs = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                  max_position_embeddings=2048, tie_word_embeddings=True)
    if architecture == "qwen3_5_text":
        kwargs.update(linear_key_head_dim=8, linear_value_head_dim=8,
                      linear_num_key_heads=2, linear_num_value_heads=4,
                      full_attention_interval=2, layer_types=["linear_attention", "full_attention"] * 2,
                      rope_parameters={"rope_type": "default", "rope_theta": 10000,
                                       "partial_rotary_factor": 1, "mrope_section": [1, 1, 2]})
    config = AutoConfig.for_model(architecture, **kwargs)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(path, safe_serialization=True)
    words = ["[UNK]", "[PAD]", "[EOS]", "system", "user", "assistant", "hello", "world", "what", "is", "two", "plus", "one"]
    words += [f"token{i}" for i in range(64 - len(words))]
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tok.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tok.save_pretrained(path)
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--architecture", default="qwen3")
    a = p.parse_args()
    make(a.path, a.architecture)

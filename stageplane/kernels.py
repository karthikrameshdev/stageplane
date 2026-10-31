"""Optional bounded FP32 accumulation for CPUs with slow BF16 dot kernels.

Weights remain in their original half precision. Each FP32 weight tile is at
most 16 MiB, avoiding a second full FP32 model. Both reference and stages use
the identical explicitly selected kernel for a fair execution comparison.
"""
import os
import types

import torch
from torch.nn import functional as F


@torch.inference_mode()
def bounded_linear(module, value):
    source_dtype = value.dtype
    inputs = value.float()
    rows = max(1, 16 * 1024**2 // (module.in_features * 4))
    result = torch.empty((*value.shape[:-1], module.out_features), device=value.device, dtype=source_dtype)
    for start in range(0, module.out_features, rows):
        end = min(module.out_features, start + rows)
        weight = module.weight[start:end].float()
        bias = module.bias[start:end].float() if module.bias is not None else None
        result[..., start:end] = F.linear(inputs, weight, bias).to(source_dtype)
    return result


def configure_cpu_linears(model):
    if os.environ.get("STAGEPLANE_CPU_KERNEL", "native") != "bounded-f32":
        return
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.device.type == "cpu" and module.weight.dtype in (torch.bfloat16, torch.float16):
            module.forward = types.MethodType(bounded_linear, module)

import torch

from stageplane.kernels import configure_cpu_linears


def test_bounded_cpu_linear_matches_bf16_and_limits_tiles(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    layer = torch.nn.Linear(128, 40000, dtype=torch.bfloat16).eval()
    inputs = torch.randn(1, 128, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = layer(inputs)
    seen = []
    original = torch.nn.functional.linear
    def record(value, weight, bias):
        seen.append(weight.numel() * weight.element_size())
        return original(value, weight, bias)
    monkeypatch.setenv("STAGEPLANE_CPU_KERNEL", "bounded-f32")
    monkeypatch.setattr(torch.nn.functional, "linear", record)
    configure_cpu_linears(layer)
    actual = layer(inputs)
    assert actual.dtype == torch.bfloat16
    assert len(seen) > 1 and max(seen) <= 16 * 1024**2
    torch.testing.assert_close(actual, expected, atol=.01, rtol=.01)

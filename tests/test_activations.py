"""Experiment 0 checks for ProxyReLU. The expected probe gradient is literal."""

import torch
import pytest

from src.activations import (
    DTYPES,
    PROBE_SLOPES,
    ProxyReLU,
    analytical_proxy_derivative,
    bertoin_emulated_relu,
    expected_probe_gradient,
    probe_tensor,
    proxy_gradient,
)
from src.determinism import s0_twin_control


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("s", PROBE_SLOPES)
def test_probe_gradient_matches_specified_pattern(dtype, s):
    gradient = proxy_gradient(probe_tensor(dtype), s)
    expected = expected_probe_gradient(s, dtype)
    assert torch.equal(gradient, expected)
    # Both signed zeros sit in the s slots of [-2, -1e-7, -0, +0, +1e-7, +2].
    assert torch.equal(gradient[2:4], expected[2:4])
    assert torch.all(probe_tensor(dtype)[2:4] == 0)


@pytest.mark.parametrize("s", PROBE_SLOPES)
def test_expected_probe_pattern_is_literal_in_fp64(s):
    expected = expected_probe_gradient(s, torch.float64)
    literal = torch.tensor([0.0, 0.0, s, s, 1.0, 1.0], dtype=torch.float64)
    assert torch.equal(expected, literal)
    assert torch.equal(analytical_proxy_derivative(probe_tensor(torch.float64), s), literal)


@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_matches_relu(dtype):
    x = probe_tensor(dtype)
    assert torch.equal(ProxyReLU(0.5)(x), torch.relu(x))


def test_emulation_matches_relu_exactly_in_fp64():
    x = torch.linspace(-5, 5, 2001, dtype=torch.float64)
    x = x[x != 0]
    for s in (0.1, 0.25, 0.3, 0.5, 0.75, 1.0):
        assert torch.equal(ProxyReLU(s)(x), bertoin_emulated_relu(x, s))
        leaf = x.detach().requires_grad_(True)
        grad, = torch.autograd.grad(bertoin_emulated_relu(leaf, s).sum(), leaf)
        assert torch.equal(grad, analytical_proxy_derivative(x, s))


def test_s0_twin_stays_at_zero_l1():
    result = s0_twin_control(steps=20, seed=0)
    assert result["final_l1"] == 0.0
    assert result["max_l1"] == 0.0
    assert result["parameters_equal"] is True

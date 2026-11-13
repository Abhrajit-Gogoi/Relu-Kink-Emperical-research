"""Proxy derivative at the ReLU kink, and the Bertoin algebraic emulation."""

from __future__ import annotations

import torch
import torch.nn as nn

PROBE_VALUES = (-2.0, -1e-7, -0.0, 0.0, 1e-7, 2.0)
PROBE_SLOPES = (0.0, 0.25, 0.5, 0.75, 1.0)
EMULATION_SLOPES = (0.1, 0.25, 0.3, 0.5, 0.75, 1.0)
DTYPES = (torch.float64, torch.float32, torch.float16, torch.bfloat16)
EMULATION_DTYPES = (torch.float32, torch.bfloat16)


class CustomReLUFunction(torch.autograd.Function):
    """Forward max(0, x). Backward slope s only where x compares equal to 0."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, s: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.s = float(s)
        return torch.clamp(x, min=0.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        s_value = torch.tensor(ctx.s, dtype=grad_output.dtype, device=grad_output.device)
        zeros = torch.zeros_like(grad_output)
        at_zero = grad_output * s_value
        grad_x = torch.where(x > 0, grad_output, torch.where(x == 0, at_zero, zeros))
        return grad_x, None


class ProxyReLU(nn.Module):
    def __init__(self, s: float = 0.0) -> None:
        super().__init__()
        self.s = float(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return CustomReLUFunction.apply(x, self.s)


def bertoin_emulated_relu(x: torch.Tensor, s: float) -> torch.Tensor:
    """f_s(x) = ReLU(x) + s (ReLU(-x) - ReLU(x) + x), with PyTorch's ReLU'(0) = 0."""
    s_value = torch.tensor(float(s), dtype=x.dtype, device=x.device)
    relu_x = torch.relu(x)
    relu_neg = torch.relu(-x)
    return relu_x + s_value * (relu_neg - relu_x + x)


def analytical_proxy_derivative(x: torch.Tensor, s: float) -> torch.Tensor:
    """ReLU'_s in ``x.dtype``: 1 if x > 0, s if x == 0, else 0."""
    s_value = torch.tensor(float(s), dtype=x.dtype, device=x.device)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    return torch.where(x > 0, ones, torch.where(x == 0, s_value, zeros))


def probe_tensor(dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(PROBE_VALUES, dtype=dtype, device=device)


def expected_probe_gradient(s: float, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, float(s), float(s), 1.0, 1.0], dtype=dtype, device=device)


def proxy_gradient(x: torch.Tensor, s: float) -> torch.Tensor:
    """Elementwise derivative of ProxyReLU(s), via a unit upstream gradient."""
    leaf = x.detach().requires_grad_(True)
    ProxyReLU(s)(leaf).sum().backward()
    return leaf.grad.detach()


def probe_passes(dtype: torch.dtype, s: float) -> bool:
    gradient = proxy_gradient(probe_tensor(dtype), s)
    return torch.equal(gradient, expected_probe_gradient(s, dtype))

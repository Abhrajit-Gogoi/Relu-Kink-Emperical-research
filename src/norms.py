"""Feature normalizations whose reductions stay in a chosen dtype."""

from __future__ import annotations

import torch
import torch.nn as nn


class ExplicitLayerNorm(nn.Module):
    """LayerNorm over the last dimension. Scale and shift are separate switches.

    ``keep_native_dtype=True`` runs the mean and variance in ``x.dtype``.
    ``False`` upcasts the reduction to FP32 and casts the normalized values back.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        scale: bool = True,
        shift: bool = True,
        keep_native_dtype: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.keep_native_dtype = keep_native_dtype
        if scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)
        if shift:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        work = x if self.keep_native_dtype else x.to(torch.float32)
        mu = work.mean(dim=-1, keepdim=True)
        centered = work - mu
        var = (centered * centered).mean(dim=-1, keepdim=True)
        hat = centered / torch.sqrt(var + self.eps)
        if not self.keep_native_dtype:
            hat = hat.to(orig_dtype)
        if self.weight is not None:
            hat = hat * self.weight
        if self.bias is not None:
            hat = hat + self.bias
        return hat


class ExplicitRMSNorm(nn.Module):
    """RMSNorm over the last dimension. There is no mean subtraction and no shift."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        scale: bool = True,
        keep_native_dtype: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.keep_native_dtype = keep_native_dtype
        if scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        work = x if self.keep_native_dtype else x.to(torch.float32)
        rms = torch.sqrt((work * work).mean(dim=-1, keepdim=True) + self.eps)
        bar = work / rms
        if not self.keep_native_dtype:
            bar = bar.to(orig_dtype)
        if self.weight is not None:
            bar = bar * self.weight
        return bar


class ExplicitBatchNorm1d(nn.Module):
    """Batch mean and variance over dim 0, per feature. A pure function of the batch.

    Running averages are not kept, so a later proxy slope cannot inherit statistics
    from an earlier slope.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        scale: bool = True,
        shift: bool = True,
        keep_native_dtype: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.keep_native_dtype = keep_native_dtype
        if scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)
        if shift:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        work = x if self.keep_native_dtype else x.to(torch.float32)
        mu = work.mean(dim=0, keepdim=True)
        centered = work - mu
        var = (centered * centered).mean(dim=0, keepdim=True)
        hat = centered / torch.sqrt(var + self.eps)
        if not self.keep_native_dtype:
            hat = hat.to(orig_dtype)
        if self.weight is not None:
            hat = hat * self.weight
        if self.bias is not None:
            hat = hat + self.bias
        return hat


def make_norm(name: str, dim: int, with_bias_and_shift: bool, keep_native_dtype: bool = True) -> nn.Module:
    """Build the norm named by the study matrix.

    LayerNorm and BatchNorm keep a learnable scale in both bias arms. The false
    arm removes only the shift. RMSNorm has no shift; the false arm removes its scale.
    """
    key = name.lower()
    if key in {"nonorm", "none", "identity"}:
        return nn.Identity()
    if key == "layernorm":
        return ExplicitLayerNorm(dim, scale=True, shift=with_bias_and_shift, keep_native_dtype=keep_native_dtype)
    if key == "rmsnorm":
        return ExplicitRMSNorm(dim, scale=with_bias_and_shift, keep_native_dtype=keep_native_dtype)
    if key in {"batchnorm1d", "batchnorm"}:
        return ExplicitBatchNorm1d(dim, scale=True, shift=with_bias_and_shift, keep_native_dtype=keep_native_dtype)
    raise ValueError(f"unknown norm {name}")

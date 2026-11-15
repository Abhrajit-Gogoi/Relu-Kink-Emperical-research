"""Networks used by the study. Stage 2 runs the plain MLP."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.activations import ProxyReLU
from src.determinism import seed_everything
from src.norms import make_norm

MLP_WIDTHS = (784, 256, 256, 256, 10)


def set_slope(model: nn.Module, slope: float) -> None:
    for module in model.modules():
        if isinstance(module, ProxyReLU):
            module.s = float(slope)


def initialize_kaiming_(model: nn.Module) -> None:
    """Draw weights in the model's current dtype. Call this on an FP32 module."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=0.0, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                bound = 1.0 / math.sqrt(module.in_features)
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


class ConfigurableMLP(nn.Module):
    """Topology A: a_l = ReLU(Norm(W_l a_{l-1} + b_l)). The last linear is the classifier."""

    def __init__(
        self,
        widths: tuple[int, ...] = MLP_WIDTHS,
        s: float = 0.0,
        norm: str = "NoNorm",
        with_bias_and_shift: bool = True,
        keep_native_dtype: bool = True,
    ) -> None:
        super().__init__()
        if len(widths) < 3:
            raise ValueError("widths must include input, one hidden layer, and output")
        self.widths = tuple(widths)
        self.norm_name = norm
        self.with_bias_and_shift = with_bias_and_shift
        self.keep_native_dtype = keep_native_dtype
        self.n_hidden = len(widths) - 2
        for index in range(self.n_hidden):
            self.add_module(
                f"fc{index + 1}",
                nn.Linear(widths[index], widths[index + 1], bias=with_bias_and_shift),
            )
            self.add_module(
                f"norm{index + 1}",
                make_norm(norm, widths[index + 1], with_bias_and_shift, keep_native_dtype),
            )
            self.add_module(f"act{index + 1}", ProxyReLU(s))
        self.add_module(
            f"fc{self.n_hidden + 1}",
            nn.Linear(widths[-2], widths[-1], bias=with_bias_and_shift),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for index in range(1, self.n_hidden + 1):
            hidden = getattr(self, f"fc{index}")(hidden)
            hidden = getattr(self, f"norm{index}")(hidden)
            hidden = getattr(self, f"act{index}")(hidden)
        return getattr(self, f"fc{self.n_hidden + 1}")(hidden)


class _InsideBranchBlock(nn.Module):
    """Topology C block. The skip does not pass through the kink."""

    def __init__(self, width: int, s: float, norm: str, with_bias_and_shift: bool, keep_native_dtype: bool) -> None:
        super().__init__()
        self.norm = make_norm(norm, width, with_bias_and_shift, keep_native_dtype)
        self.fc1 = nn.Linear(width, width, bias=with_bias_and_shift)
        self.act = ProxyReLU(s)
        self.fc2 = nn.Linear(width, width, bias=with_bias_and_shift)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(self.norm(residual))
        return residual + self.fc2(self.act(hidden))


class _HighwayBlock(nn.Module):
    """Topology B block. The second ReLU sits on the residual sum."""

    def __init__(self, width: int, s: float, norm: str, with_bias_and_shift: bool, keep_native_dtype: bool) -> None:
        super().__init__()
        self.norm = make_norm(norm, width, with_bias_and_shift, keep_native_dtype)
        self.fc1 = nn.Linear(width, width, bias=with_bias_and_shift)
        self.act_branch = ProxyReLU(s)
        self.fc2 = nn.Linear(width, width, bias=False)
        self.act_highway = ProxyReLU(s)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        branch = self.act_branch(self.norm(self.fc1(residual)))
        return self.act_highway(residual + self.fc2(branch))


class ResidualFFNNet(nn.Module):
    """Three residual blocks at a matched width, after a linear stem from the input."""

    def __init__(
        self,
        topology: str = "C",
        in_features: int = 784,
        width: int = 256,
        n_classes: int = 10,
        depth: int = 3,
        s: float = 0.0,
        norm: str = "NoNorm",
        with_bias_and_shift: bool = True,
        keep_native_dtype: bool = True,
    ) -> None:
        super().__init__()
        key = topology.upper()
        if key not in {"B", "C"}:
            raise ValueError("topology must be B or C")
        self.topology = key
        block_cls = _HighwayBlock if key == "B" else _InsideBranchBlock
        self.stem = nn.Linear(in_features, width, bias=with_bias_and_shift)
        self.blocks = nn.ModuleList(
            [
                block_cls(width, s, norm, with_bias_and_shift, keep_native_dtype)
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(width, n_classes, bias=with_bias_and_shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self._split(self.w_q(tokens))
        key = self._split(self.w_k(tokens))
        value = self._split(self.w_v(tokens))
        scores = query @ key.transpose(-2, -1) * (self.d_k ** -0.5)
        weights = torch.softmax(scores, dim=-1)
        mixed = (weights @ value).transpose(1, 2).contiguous()
        batch, length, heads, dim = mixed.shape
        mixed = mixed.view(batch, length, heads * dim)
        return self.w_o(mixed)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.n_heads, self.d_k).transpose(1, 2)


class MicroTransformer(nn.Module):
    """One Pre-Norm encoder block on 4x4 patches of a 28x28 image."""

    def __init__(
        self,
        s: float = 0.0,
        norm: str = "LayerNorm",
        with_bias_and_shift: bool = True,
        keep_native_dtype: bool = True,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_classes: int = 10,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        bias = with_bias_and_shift
        self.patch = nn.Linear(16, d_model, bias=bias)
        self.pos = nn.Parameter(torch.zeros(1, 49, d_model))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm1": make_norm(norm, d_model, with_bias_and_shift, keep_native_dtype),
                        "attn": MultiHeadSelfAttention(d_model, n_heads, bias),
                        "norm2": make_norm(norm, d_model, with_bias_and_shift, keep_native_dtype),
                        "fc1": nn.Linear(d_model, d_ff, bias=bias),
                        "act": ProxyReLU(s),
                        "fc2": nn.Linear(d_ff, d_model, bias=bias),
                    }
                )
            )
        self.final_norm = make_norm(norm, d_model, with_bias_and_shift, keep_native_dtype)
        self.head = nn.Linear(d_model, n_classes, bias=bias)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(x) + self.pos
        for block in self.blocks:
            tokens = tokens + block["attn"](block["norm1"](tokens))
            hidden = block["act"](block["fc1"](block["norm2"](tokens)))
            tokens = tokens + block["fc2"](hidden)
        pooled = self.final_norm(tokens.mean(dim=1))
        return self.head(pooled)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        patches = x.unfold(2, 4, 4).unfold(3, 4, 4)
        patches = patches.contiguous().view(x.shape[0], 49, 16)
        return self.patch(patches)


def build_mlp(
    seed: int,
    dtype: torch.dtype,
    s: float = 0.0,
    widths: tuple[int, ...] = MLP_WIDTHS,
    norm: str = "NoNorm",
    with_bias_and_shift: bool = True,
    keep_native_dtype: bool = True,
) -> ConfigurableMLP:
    """Seed, draw an FP32 initialization, then cast the state dict to ``dtype``."""
    seed_everything(seed)
    model = ConfigurableMLP(
        widths=widths,
        s=s,
        norm=norm,
        with_bias_and_shift=with_bias_and_shift,
        keep_native_dtype=keep_native_dtype,
    )
    initialize_kaiming_(model)
    if dtype != torch.float32:
        model.to(dtype)
    set_slope(model, s)
    return model

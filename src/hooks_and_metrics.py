"""Kink counts, confidence intervals, and gradient distances."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.activations import ProxyReLU

LOG_BIN_EDGES = (-12.0, -8.0, -5.0, -3.0, -1.0)
HIST_COLUMNS = (
    "hist_exact_zero",
    "hist_le_m12",
    "hist_m12_m8",
    "hist_m8_m5",
    "hist_m5_m3",
    "hist_m3_m1",
    "hist_gt_m1",
)
MC_COLUMNS = (
    "precision",
    "norm",
    "with_bias_and_shift",
    "s",
    "keep_native_dtype",
    "exact_zero_rate",
    *HIST_COLUMNS,
    "bifurcation_weight_prop",
    "hoeffding_ci_95",
    "bifurcation_batch_prop",
    "mcdiarmid_ci_95",
    "rel_l2_q1",
    "rel_l2_q2",
    "rel_l2_q3",
    "grad_norm_ratio_q1",
    "grad_norm_ratio_q2",
    "grad_norm_ratio_q3",
    "grad_cosine_l1",
    "grad_cosine_l2",
    "grad_cosine_l3",
    "grad_cosine_l4",
    "n_impacted_pairs",
)
TRAJ_COLUMNS = (
    "record_type",
    "seed",
    "topology",
    "precision",
    "norm",
    "s",
    "step_k",
    "min_abs_preact",
    "exact_zero_count_k",
    "traj_l1_dist",
    "traj_l2_dist",
    "weight_cosine_sim",
    "grad_cosine_sim_step",
    "twin_l1_dist",
    "first_bifurcation_step",
)
SUMMARY_COLUMNS = (
    "precision",
    "norm",
    "topology",
    "s",
    "step_k",
    "n_seeds",
    "l1_q1",
    "l1_median",
    "l1_q3",
    "l1_iqr",
    "l2_q1",
    "l2_median",
    "l2_q3",
    "grad_cosine_median",
    "kstar_median",
)


def hoeffding_ci_95(m: int) -> float:
    return math.sqrt(math.log(2.0 / 0.05) / (2.0 * m))


def mcdiarmid_ci_95(m: int, q: int) -> float:
    return math.sqrt(0.5 * ((1.0 / m) + (1.0 / q)) * math.log(40.0))


def assign_log_bins(log_abs: torch.Tensor) -> torch.Tensor:
    """Bin log10 magnitudes into (-inf, -12], (-12, -8], ..., (-1, inf)."""
    edges = torch.tensor(LOG_BIN_EDGES, dtype=log_abs.dtype, device=log_abs.device)
    # side='left' places x <= -12 in bin 0 and keeps each later edge in the right-closed bin.
    return torch.bucketize(log_abs, edges, right=False)


class KinkCollisionTracker:
    """Counts exact zeros and log-spaced magnitudes of pre-activations."""

    def __init__(self) -> None:
        self.recording = True
        self.reset()

    def reset(self) -> None:
        self.total_elements = 0
        self.exact_zeros = 0
        self.min_abs = float("inf")
        self.bins = [0, 0, 0, 0, 0, 0]

    def observe(self, x: torch.Tensor) -> None:
        if not self.recording:
            return
        flat = x.detach().reshape(-1)
        n = int(flat.numel())
        if n == 0:
            return
        self.total_elements += n
        exact_mask = flat == 0
        n_exact = int(exact_mask.sum().item())
        self.exact_zeros += n_exact
        amin = float(flat.abs().min().to(torch.float64).item())
        if amin < self.min_abs:
            self.min_abs = amin
        if n_exact == n:
            return
        logs = flat[~exact_mask].abs().to(torch.float64).log10()
        counts = torch.bincount(assign_log_bins(logs), minlength=6)
        for i in range(6):
            self.bins[i] += int(counts[i].item())

    def snapshot(self) -> dict:
        return {
            "total": self.total_elements,
            "exact": self.exact_zeros,
            "min_abs": None if self.min_abs == float("inf") else self.min_abs,
            "bins": tuple(self.bins),
            "zero_rate": self.zero_rate,
        }

    @property
    def zero_rate(self) -> float:
        return self.exact_zeros / max(1, self.total_elements)


def attach_kink_hooks(model: nn.Module, tracker: KinkCollisionTracker) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []
    for module in model.modules():
        if isinstance(module, ProxyReLU):
            handles.append(module.register_forward_hook(lambda mod, inputs, output: tracker.observe(inputs[0])))
    return handles


def quartiles(values: list[float]) -> tuple[float, float, float] | None:
    if not values:
        return None
    q1, q2, q3 = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75], method="linear")
    return float(q1), float(q2), float(q3)


def _flat(parts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([part.reshape(-1) for part in parts])


@dataclass
class GradPack:
    layers: list[torch.Tensor]
    nonfinite: bool = False

    def flat(self) -> torch.Tensor:
        return _flat(self.layers)


def packs_equal(left: GradPack, right: GradPack) -> bool:
    if len(left.layers) != len(right.layers):
        return False
    return all(torch.equal(a, b) for a, b in zip(left.layers, right.layers))


def promote(vector: torch.Tensor) -> torch.Tensor:
    return vector.detach().to(torch.float64).reshape(-1)


def relative_l2(gs: torch.Tensor, g0: torch.Tensor) -> float:
    left = promote(gs)
    right = promote(g0)
    return float(torch.linalg.vector_norm(left - right).item() / (torch.linalg.vector_norm(right).item() + 1e-30))


def norm_ratio(gs: torch.Tensor, g0: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(promote(gs)).item() / (torch.linalg.vector_norm(promote(g0)).item() + 1e-30))


def cosine(gs: torch.Tensor, g0: torch.Tensor) -> float:
    left = promote(gs)
    right = promote(g0)
    denom = torch.linalg.vector_norm(left).item() * torch.linalg.vector_norm(right).item() + 1e-30
    return float(torch.dot(left, right).item() / denom)


def parameter_flat(model: nn.Module) -> torch.Tensor:
    pieces = [p.detach().reshape(-1).to(torch.float64) for p in model.parameters()]
    return torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.float64)


def parameter_l1(left: nn.Module, right: nn.Module) -> float:
    return float(torch.abs(parameter_flat(left) - parameter_flat(right)).sum().item())


def parameter_l2(left: nn.Module, right: nn.Module) -> float:
    return float(torch.linalg.vector_norm(parameter_flat(left) - parameter_flat(right)).item())


def parameter_cosine(left: nn.Module, right: nn.Module) -> float:
    return cosine(parameter_flat(left), parameter_flat(right))


def parameters_identical(left: nn.Module, right: nn.Module) -> bool:
    return all(torch.equal(a, b) for a, b in zip(left.parameters(), right.parameters()))


def linear_modules(model: nn.Module) -> list[nn.Linear]:
    found = []
    index = 1
    while hasattr(model, f"fc{index}"):
        found.append(getattr(model, f"fc{index}"))
        index += 1
    return found


def capture_grads(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, slope: float) -> GradPack:
    """One native-dtype forward and backward. Returns cloned per-layer gradients."""
    from src.models import set_slope

    set_slope(model, slope)
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    if not torch.isfinite(loss.detach()):
        return GradPack(layers=[], nonfinite=True)
    loss.backward()
    layers: list[torch.Tensor] = []
    nonfinite = False
    for linear in linear_modules(model):
        parts = []
        for tensor in (linear.weight, linear.bias):
            if tensor is None:
                continue
            grad = tensor.grad
            if grad is None:
                grad = torch.zeros_like(tensor)
            if not torch.isfinite(grad).all():
                nonfinite = True
            parts.append(grad.detach().reshape(-1).clone())
        layers.append(torch.cat(parts))
    return GradPack(layers=layers, nonfinite=nonfinite)


@dataclass
class CellStats:
    any_diff: bool = False
    n_pairs: int = 0
    n_impacted: int = 0
    n_nonfinite: int = 0
    rel_l2: list[float] = field(default_factory=list)
    norm_ratios: list[float] = field(default_factory=list)
    layer_cosines: list[list[float]] = field(default_factory=lambda: [[], [], [], []])


def empty_cell_stats() -> CellStats:
    return CellStats(layer_cosines=[[], [], [], []])


def update_pair(stats: CellStats, baseline: GradPack, other: GradPack) -> None:
    stats.n_pairs += 1
    if baseline.nonfinite or other.nonfinite or not packs_equal(baseline, other):
        stats.n_impacted += 1
        stats.any_diff = True
    if baseline.nonfinite or other.nonfinite or not baseline.layers or not other.layers:
        if baseline.nonfinite or other.nonfinite:
            stats.n_nonfinite += 1
        return
    if packs_equal(baseline, other):
        return
    stats.rel_l2.append(relative_l2(other.flat(), baseline.flat()))
    stats.norm_ratios.append(norm_ratio(other.flat(), baseline.flat()))
    for index, (gs, g0) in enumerate(zip(other.layers, baseline.layers)):
        if index < 4:
            stats.layer_cosines[index].append(cosine(gs, g0))


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.17g}"


def fmt_quartiles(values: list[float]) -> tuple[str, str, str]:
    found = quartiles(values)
    if found is None:
        return "", "", ""
    return tuple(fmt_float(item) for item in found)  # type: ignore[return-value]


def median_cosine(values: list[float]) -> str:
    found = quartiles(values)
    if found is None:
        return ""
    return fmt_float(found[1])

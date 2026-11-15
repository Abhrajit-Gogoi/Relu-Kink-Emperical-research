"""Seed locking, and the Stage 1 s = 0 twin control."""

from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.activations import ProxyReLU


def preferred_device() -> torch.device:
    """Use the laptop GPU for every later stage.

    Stage 2 ran on the CPU wheel (``torch 2.15.0+cpu``). The FP16 Monte Carlo
    alone took about three hours. This laptop is an RTX 4060, and the CUDA
    build installed for later stages is ``torch==2.14.1+cu130``. When CUDA is
    visible, return that device and put models and batches on it. CPU is only
    the fallback for a build that cannot see a GPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> dict:
    """Seed Python, NumPy, and PyTorch. Request deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    info = {"seed": int(seed), "deterministic": True, "deterministic_error": None}
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # pragma: no cover - depends on the build
        info["deterministic"] = False
        info["deterministic_error"] = str(exc)
    return info


def parameter_l1(left: nn.Module, right: nn.Module) -> float:
    """L1 distance between parameters after promotion to FP64."""
    total = 0.0
    with torch.no_grad():
        for a, b in zip(left.parameters(), right.parameters()):
            if not torch.equal(a, b):
                total += (a.detach().to(torch.float64) - b.detach().to(torch.float64)).abs().sum().item()
    return float(total)


def parameters_equal(left: nn.Module, right: nn.Module) -> bool:
    with torch.no_grad():
        return all(torch.equal(a, b) for a, b in zip(left.parameters(), right.parameters()))


class _TwinMLP(nn.Module):
    """One hidden layer, used only for the s = 0 determinism control."""

    def __init__(self, in_features: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=True)
        self.act = ProxyReLU(0.0)
        self.fc2 = nn.Linear(hidden, n_classes, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_uniform_(layer.weight, a=0.0, mode="fan_in", nonlinearity="relu")
            bound = 1.0 / math.sqrt(layer.in_features)
            nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def s0_twin_control(
    steps: int = 20,
    seed: int = 0,
    batch_size: int = 128,
    in_features: int = 16,
    hidden: int = 32,
    n_classes: int = 4,
    lr: float = 0.01,
    batch_path: Path | None = None,
) -> dict:
    """Train two s = 0 copies on the same batches. Return the FP64 L1 distance.

    Batches are drawn with a local generator. Weights are drawn in FP32 from the
    global generator after ``seed_everything(seed)``, then copied.
    """
    seed_info = seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    inputs = []
    targets = []
    for _ in range(steps):
        inputs.append(torch.rand(batch_size, in_features, generator=generator))
        targets.append(torch.randint(0, n_classes, (batch_size,), generator=generator))

    seed_everything(seed)
    model_a = _TwinMLP(in_features, hidden, n_classes)
    model_b = _TwinMLP(in_features, hidden, n_classes)
    model_b.load_state_dict(copy.deepcopy(model_a.state_dict()))
    opt_a = torch.optim.SGD(model_a.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)

    max_l1 = 0.0
    final_l1 = 0.0
    for x, y in zip(inputs, targets):
        for model, opt in ((model_a, opt_a), (model_b, opt_b)):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        final_l1 = parameter_l1(model_a, model_b)
        max_l1 = max(max_l1, final_l1)

    saved = None
    if batch_path is not None:
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            batch_path,
            inputs=torch.stack(inputs).numpy(),
            targets=torch.stack(targets).numpy(),
        )
        saved = str(batch_path)

    return {
        "seed_info": seed_info,
        "steps": steps,
        "seed": seed,
        "batch_size": batch_size,
        "in_features": in_features,
        "hidden": hidden,
        "n_classes": n_classes,
        "lr": lr,
        "final_l1": final_l1,
        "max_l1": max_l1,
        "parameters_equal": parameters_equal(model_a, model_b),
        "batch_file": saved,
    }

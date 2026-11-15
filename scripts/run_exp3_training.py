"""Experiment 3: 20-epoch training stability with MLP and Micro-Transformer.

Matrix:
- Architectures: Residual MLP (Topology C) and 1-block Micro-Transformer
- Norms: LayerNorm, RMSNorm (plus NoNorm on MLP)
- Precisions: FP32, BF16
- Activations: ProxyReLU(s=0), ProxyReLU(s=0.5), ProxyReLU(s=1), nn.GELU()
- 5 paired seeds, 20 epochs, SGD lr=0.01, batch size 128
- Datasets: MNIST (required), Fashion-MNIST (after MNIST complete)
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.determinism import preferred_device, seed_everything  # noqa: E402
from src.hooks_and_metrics import (  # noqa: E402
    KinkCollisionTracker,
    attach_kink_hooks,
    fmt_float,
    parameter_l1,
)
from src.models import (  # noqa: E402
    MicroTransformer,
    MLP_WIDTHS,
    ResidualFFNNet,
    build_mlp,
    initialize_kaiming_,
    set_slope,
)

RESULTS = ROOT / "results"
BATCH_SIZE = 128
LR = 0.01
EPOCHS = 20
N_SEEDS = 5

# Architecture configs
MLP_CONFIG = {
    "topology": "C",
    "in_features": 784,
    "width": 256,
    "n_classes": 10,
    "depth": 3,
}

TRANSFORMER_CONFIG = {
    "d_model": 64,
    "n_heads": 4,
    "d_ff": 256,
    "n_classes": 10,
    "depth": 1,
}

# Experimental matrix
MLP_NORMS = ("NoNorm", "LayerNorm", "RMSNorm")
TRANSFORMER_NORMS = ("LayerNorm", "RMSNorm")
PRECISIONS = ("float32", "bfloat16")

# Activation configurations: (name, slope_or_None)
# slope=None means use nn.GELU()
ACTIVATIONS = [
    ("relu_0.0", 0.0),
    ("relu_0.5", 0.5),
    ("relu_1.0", 1.0),
    ("gelu", None),
]

DTYPE_OF = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def load_mnist(root: Path, train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision import datasets

    dataset = datasets.MNIST(str(root), train=train, download=True)
    images = dataset.data.to(torch.float32).div(255.0).view(-1, 784).contiguous()
    labels = dataset.targets.to(torch.long).contiguous()
    return images, labels


def load_fashion_mnist(root: Path, train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision import datasets

    dataset = datasets.FashionMNIST(str(root), train=train, download=True)
    images = dataset.data.to(torch.float32).div(255.0).view(-1, 784).contiguous()
    labels = dataset.targets.to(torch.long).contiguous()
    return images, labels


def make_batch_indices(n_items: int, n_batches: int, batch_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(n_items, generator=generator)
    needed = n_batches * batch_size
    if needed > n_items:
        repeats = (needed + n_items - 1) // n_items
        permutation = permutation.repeat(repeats)
    return permutation[:needed].view(n_batches, batch_size).to(torch.long)


def build_model(
    arch: str,
    seed: int,
    dtype: torch.dtype,
    norm: str,
    activation_name: str,
    slope: float | None,
    with_bias_and_shift: bool,
    keep_native_dtype: bool,
    device: torch.device,
) -> nn.Module:
    """Build model with given architecture and activation."""
    seed_everything(seed)

    if arch == "mlp":
        model = ResidualFFNNet(
            topology="C",
            in_features=MLP_CONFIG["in_features"],
            width=MLP_CONFIG["width"],
            n_classes=MLP_CONFIG["n_classes"],
            depth=MLP_CONFIG["depth"],
            s=0.0 if slope is not None else 0.0,  # placeholder, will be set below
            norm=norm,
            with_bias_and_shift=with_bias_and_shift,
            keep_native_dtype=keep_native_dtype,
        )
    elif arch == "transformer":
        model = MicroTransformer(
            s=0.0 if slope is not None else 0.0,  # placeholder
            norm=norm,
            with_bias_and_shift=with_bias_and_shift,
            keep_native_dtype=keep_native_dtype,
            d_model=TRANSFORMER_CONFIG["d_model"],
            n_heads=TRANSFORMER_CONFIG["n_heads"],
            d_ff=TRANSFORMER_CONFIG["d_ff"],
            n_classes=TRANSFORMER_CONFIG["n_classes"],
            depth=TRANSFORMER_CONFIG["depth"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    initialize_kaiming_(model)

    if dtype != torch.float32 or device.type != "cpu":
        model.to(device=device, dtype=dtype)

    # Set activation
    if slope is not None:
        set_slope(model, slope)
    else:
        # Replace ProxyReLU with GELU
        for module in model.modules():
            if isinstance(module, torch.nn.Module) and module.__class__.__name__ == "ProxyReLU":
                # We need to replace the module. This is tricky with nn.Module.
                # Instead, we'll use a different approach: build with a flag.
                pass
        # For GELU, we need a different model builder. Let's handle this separately.
        raise NotImplementedError("GELU replacement not yet implemented in this builder")

    return model


def build_model_with_activation(
    arch: str,
    seed: int,
    dtype: torch.dtype,
    norm: str,
    activation_name: str,
    slope: float | None,
    with_bias_and_shift: bool,
    keep_native_dtype: bool,
    device: torch.device,
) -> nn.Module:
    """Build model with proper activation (ProxyReLU or GELU)."""
    seed_everything(seed)

    if arch == "mlp":
        # Build MLP with ProxyReLU, then optionally replace with GELU
        model = ResidualFFNNet(
            topology="C",
            in_features=MLP_CONFIG["in_features"],
            width=MLP_CONFIG["width"],
            n_classes=MLP_CONFIG["n_classes"],
            depth=MLP_CONFIG["depth"],
            s=0.0,  # placeholder
            norm=norm,
            with_bias_and_shift=with_bias_and_shift,
            keep_native_dtype=keep_native_dtype,
        )
        initialize_kaiming_(model)
        if dtype != torch.float32 or device.type != "cpu":
            model.to(device=device, dtype=dtype)

        if slope is not None:
            set_slope(model, slope)
        else:
            # Replace all ProxyReLU with GELU
            for name, module in model.named_modules():
                if module.__class__.__name__ == "ProxyReLU":
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    setattr(parent, child_name, nn.GELU())

    elif arch == "transformer":
        model = MicroTransformer(
            s=0.0,  # placeholder
            norm=norm,
            with_bias_and_shift=with_bias_and_shift,
            keep_native_dtype=keep_native_dtype,
            d_model=TRANSFORMER_CONFIG["d_model"],
            n_heads=TRANSFORMER_CONFIG["n_heads"],
            d_ff=TRANSFORMER_CONFIG["d_ff"],
            n_classes=TRANSFORMER_CONFIG["n_classes"],
            depth=TRANSFORMER_CONFIG["depth"],
        )
        initialize_kaiming_(model)
        if dtype != torch.float32 or device.type != "cpu":
            model.to(device=device, dtype=dtype)

        if slope is not None:
            set_slope(model, slope)
        else:
            # Replace all ProxyReLU with GELU
            for name, module in model.named_modules():
                if module.__class__.__name__ == "ProxyReLU":
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    setattr(parent, child_name, nn.GELU())
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    batch_indices: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    tracker: KinkCollisionTracker,
    epoch: int,
    steps_per_epoch: int,
    progress_every: int = 50,
) -> tuple[float, float, int]:
    """Train for one epoch. Returns (mean_loss, accuracy_pct, exact_zero_count)."""
    model.train()
    tracker.reset()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Slice batch_indices to get only the batches for this epoch
    start_idx = (epoch - 1) * steps_per_epoch
    end_idx = epoch * steps_per_epoch
    epoch_batches = batch_indices[start_idx:end_idx]
    n_batches = epoch_batches.shape[0]

    for batch_idx, batch in enumerate(epoch_batches):
        inputs = train_images.index_select(0, batch).to(device=device, dtype=dtype)
        targets = train_labels.index_select(0, batch).to(device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += inputs.size(0)

        # Progress reporting
        if (batch_idx + 1) % progress_every == 0 or (batch_idx + 1) == n_batches:
            pct = 100.0 * (batch_idx + 1) / n_batches
            current_loss = total_loss / total_samples
            current_acc = 100.0 * total_correct / total_samples
            print(f"    Epoch {epoch} [{batch_idx+1}/{n_batches}] ({pct:.1f}%) loss={current_loss:.4f} acc={current_acc:.2f}%", flush=True)

    mean_loss = total_loss / total_samples
    accuracy_pct = 100.0 * total_correct / total_samples
    return mean_loss, accuracy_pct, tracker.exact_zeros


def evaluate(
    model: nn.Module,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on test set. Returns (loss, accuracy_pct)."""
    model.eval()
    with torch.no_grad():
        inputs = test_images.to(device=device, dtype=dtype)
        targets = test_labels.to(device=device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets).item()
        accuracy_pct = 100.0 * (logits.argmax(dim=1) == targets).sum().item() / test_labels.size(0)
    return loss, accuracy_pct


def run_training(
    arch: str,
    norm: str,
    precision: str,
    activation_name: str,
    slope: float | None,
    with_bias_and_shift: bool,
    seed: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    batch_indices: torch.Tensor,
    device: torch.device,
    epochs: int = EPOCHS,
    steps_per_epoch: int = 468,
    progress_every: int = 50,
) -> dict:
    """Run full training for one configuration. Returns dict with all metrics."""
    dtype = DTYPE_OF[precision]
    keep_native_dtype = True

    model = build_model_with_activation(
        arch=arch,
        seed=seed,
        dtype=dtype,
        norm=norm,
        activation_name=activation_name,
        slope=slope,
        with_bias_and_shift=with_bias_and_shift,
        keep_native_dtype=keep_native_dtype,
        device=device,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0, weight_decay=0.0)
    tracker = KinkCollisionTracker()
    attach_kink_hooks(model, tracker)

    epoch_logs = []
    step_losses = []  # For TV calculation
    step_grad_cosines = []  # For mean_step_grad_cosine

    prev_grad_flat = None

    for epoch in range(1, epochs + 1):
        # Training epoch
        mean_loss, train_acc, exact_zeros = train_one_epoch(
            model, optimizer, train_images, train_labels, batch_indices, dtype, device, tracker,
            epoch=epoch, steps_per_epoch=steps_per_epoch, progress_every=progress_every
        )

        # Compute step-level gradient cosine for this epoch's last step
        # (We capture gradient after the last batch of the epoch)
        model.train()  # Need train mode for gradients
        sample_batch = batch_indices[-1]
        inputs = train_images.index_select(0, sample_batch).to(device=device, dtype=dtype)
        targets = train_labels.index_select(0, sample_batch).to(device=device)
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        # Flatten gradients
        grad_parts = []
        for p in model.parameters():
            if p.grad is not None:
                grad_parts.append(p.grad.detach().reshape(-1).to(torch.float64))
        if grad_parts:
            grad_flat = torch.cat(grad_parts)
            if prev_grad_flat is not None and grad_flat.numel() == prev_grad_flat.numel():
                # Cosine similarity
                denom = torch.norm(grad_flat) * torch.norm(prev_grad_flat) + 1e-30
                cos = torch.dot(grad_flat, prev_grad_flat) / denom
                step_grad_cosines.append(cos.item())
            prev_grad_flat = grad_flat

        # Test evaluation
        test_loss, test_acc = evaluate(model, test_images, test_labels, dtype, device)

        # Exact zero rate for this epoch
        epoch_exact_zero_rate = tracker.exact_zeros / max(1, tracker.total_elements)

        epoch_logs.append({
            "epoch": epoch,
            "train_loss_mean": mean_loss,
            "train_acc_pct": train_acc,
            "test_loss": test_loss,
            "test_acc_pct": test_acc,
            "epoch_exact_zero_rate": epoch_exact_zero_rate,
        })

        step_losses.append(mean_loss)

        # Epoch summary
        print(f"    Epoch {epoch} complete: train_loss={mean_loss:.4f} train_acc={train_acc:.2f}% "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% exact_zero_rate={epoch_exact_zero_rate:.2e}", flush=True)

    # Compute summary metrics
    # Loss total variation
    if len(step_losses) > 1:
        tv = float(np.mean(np.abs(np.diff(step_losses))))
    else:
        tv = 0.0

    # Mean step gradient cosine
    mean_step_grad_cosine = float(np.mean(step_grad_cosines)) if step_grad_cosines else 0.0

    # For transformer: ffn_vs_attn_grad_diff_ratio
    ffn_vs_attn_grad_diff_ratio = ""
    if arch == "transformer":
        # This would require storing gradients from attention and FFN separately
        # Simplified: compute ratio of gradient norms in attention vs FFN
        # For now, leave blank - would need more detailed gradient capture
        pass

    return {
        "epoch_logs": epoch_logs,
        "loss_total_variation": tv,
        "mean_step_grad_cosine": mean_step_grad_cosine,
        "ffn_vs_attn_grad_diff_ratio": ffn_vs_attn_grad_diff_ratio,
        "final_test_acc_pct": epoch_logs[-1]["test_acc_pct"],
        "final_train_acc_pct": epoch_logs[-1]["train_acc_pct"],
    }


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        # Read existing data to preserve it
        existing_rows = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
        # Combine existing + new, avoiding duplicates by (record_type, architecture, norm, precision, activation, seed, epoch)
        # For simplicity, we'll just append new rows that aren't already in the file
        existing_keys = set()
        for r in existing_rows:
            key = (r.get("record_type"), r.get("architecture"), r.get("norm"), r.get("precision"), 
                   r.get("activation"), r.get("seed"), r.get("epoch"))
            existing_keys.add(key)
        
        new_rows = []
        for r in rows:
            key = (r.get("record_type"), r.get("architecture"), r.get("norm"), r.get("precision"),
                   r.get("activation"), r.get("seed"), r.get("epoch"))
            if key not in existing_keys:
                new_rows.append(r)
        
        if not new_rows:
            return  # Nothing new to add
        
        all_rows = existing_rows + new_rows
    else:
        all_rows = rows
    
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)


def checkpoint_results(
    rows_epoch: list[dict],
    rows_run: list[dict],
    rows_wilcoxon: list[dict],
    csv_path: Path,
    columns: tuple[str, ...],
) -> None:
    """Save intermediate results to disk."""
    all_rows = rows_epoch + rows_run + rows_wilcoxon
    if all_rows:
        write_csv(csv_path, all_rows, columns, append=True)
        print(f"  [Checkpoint] Saved {len(all_rows)} rows to {csv_path.name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mnist", "fashion"), default="mnist")
    parser.add_argument("--arch", choices=("mlp", "transformer", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--batch-seed", type=int, default=0)
    parser.add_argument("--precision", choices=PRECISIONS, default=None)
    parser.add_argument("--norm", default=None)
    parser.add_argument("--activation", choices=[a[0] for a in ACTIVATIONS], default=None)
    args = parser.parse_args()

    device = preferred_device()
    print(f"Device: {device}", flush=True)

    # Load data
    if args.dataset == "mnist":
        train_images, train_labels = load_mnist(ROOT / "data", train=True)
        test_images, test_labels = load_mnist(ROOT / "data", train=False)
    else:
        train_images, train_labels = load_fashion_mnist(ROOT / "data", train=True)
        test_images, test_labels = load_fashion_mnist(ROOT / "data", train=False)

    print(f"Dataset: {args.dataset}, train={train_images.shape[0]}, test={test_images.shape[0]}", flush=True)

    # Prepare batch indices (same for all runs)
    steps_per_epoch = train_images.shape[0] // BATCH_SIZE
    total_steps = steps_per_epoch * args.epochs
    batch_index_path = RESULTS / f"{args.dataset}_batch_indices.npy"
    if batch_index_path.exists():
        import numpy as np
        stored = torch.from_numpy(np.load(batch_index_path)).to(torch.long)
        if stored.shape[0] >= total_steps:
            batch_indices = stored[:total_steps]
        else:
            batch_indices = make_batch_indices(train_images.shape[0], total_steps, BATCH_SIZE, args.batch_seed)
            np.save(batch_index_path, batch_indices.numpy())
    else:
        batch_indices = make_batch_indices(train_images.shape[0], total_steps, BATCH_SIZE, args.batch_seed)
        batch_index_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        np.save(batch_index_path, batch_indices.numpy())

    print(f"Total steps: {total_steps} ({steps_per_epoch} per epoch)", flush=True)

    # Build experiment matrix
    archs = [args.arch] if args.arch != "both" else ["mlp", "transformer"]
    precisions = [args.precision] if args.precision else PRECISIONS
    activations = [a for a in ACTIVATIONS if args.activation is None or a[0] == args.activation]

    # Calculate total configurations for progress tracking
    total_configs = 0
    for arch in archs:
        norms = MLP_NORMS if arch == "mlp" else TRANSFORMER_NORMS
        if args.norm:
            norms = [args.norm]
        total_configs += len(norms) * len(precisions) * len(activations)

    config_counter = 0

    rows_epoch = []
    rows_run = []
    rows_wilcoxon = []

    # Column definitions for checkpointing
    EPOCH_COLUMNS = (
        "record_type", "dataset", "architecture", "norm", "precision", "activation",
        "seed", "with_bias_and_shift", "epoch", "train_loss_mean", "train_acc_pct",
        "test_loss", "test_acc_pct", "epoch_exact_zero_rate",
    )
    RUN_COLUMNS = (
        "record_type", "dataset", "architecture", "norm", "precision", "activation",
        "seed", "with_bias_and_shift", "loss_total_variation", "mean_step_grad_cosine",
        "ffn_vs_attn_grad_diff_ratio", "final_test_acc_pct", "final_train_acc_pct",
    )
    WILCOXON_COLUMNS = (
        "record_type", "dataset", "architecture", "norm", "precision", "activation",
        "baseline_activation", "metric", "wilcoxon_p_val", "paired_median_diff", "n_pairs",
        "error",
    )
    ALL_COLUMNS = list(set(EPOCH_COLUMNS + RUN_COLUMNS + WILCOXON_COLUMNS))

    for arch in archs:
        norms = MLP_NORMS if arch == "mlp" else TRANSFORMER_NORMS
        if args.norm:
            norms = [args.norm]

        for norm in norms:
            with_bias_and_shift = True  # As per protocol, norms keep scale, bias/shift controlled by flag
            # For Stage 5, protocol says: "LayerNorm and RMSNorm, plus NoNorm on MLP"
            # with_bias_and_shift=True enables biases and LayerNorm/RMSNorm shift
            # We'll run with_bias_and_shift=True as the default (matching Stage 3/4)

            for precision in precisions:
                for activation_name, slope in activations:
                    config_counter += 1
                    print(f"\n=== [{config_counter}/{total_configs}] {arch} | {norm} | {precision} | {activation_name} ===", flush=True)

                    # Run paired seeds
                    seed_results = []
                    for seed in range(1, args.seeds + 1):
                        print(f"  Seed {seed}/{args.seeds}...", flush=True)
                        result = run_training(
                            arch=arch,
                            norm=norm,
                            precision=precision,
                            activation_name=activation_name,
                            slope=slope,
                            with_bias_and_shift=with_bias_and_shift,
                            seed=seed,
                            train_images=train_images,
                            train_labels=train_labels,
                            test_images=test_images,
                            test_labels=test_labels,
                            batch_indices=batch_indices,
                            device=device,
                            epochs=args.epochs,
                            steps_per_epoch=steps_per_epoch,
                            progress_every=50,
                        )
                        seed_results.append(result)

                        # Epoch rows
                        for epoch_log in result["epoch_logs"]:
                            rows_epoch.append({
                                "record_type": "epoch",
                                "dataset": args.dataset,
                                "architecture": arch,
                                "norm": norm,
                                "precision": precision,
                                "activation": activation_name,
                                "seed": seed,
                                "with_bias_and_shift": str(with_bias_and_shift).lower(),
                                **epoch_log,
                            })

                    # Run summary rows (one per seed)
                    for seed_idx, result in enumerate(seed_results):
                        seed = seed_idx + 1
                        rows_run.append({
                            "record_type": "run",
                            "dataset": args.dataset,
                            "architecture": arch,
                            "norm": norm,
                            "precision": precision,
                            "activation": activation_name,
                            "seed": seed,
                            "with_bias_and_shift": str(with_bias_and_shift).lower(),
                            "loss_total_variation": fmt_float(result["loss_total_variation"]),
                            "mean_step_grad_cosine": fmt_float(result["mean_step_grad_cosine"]),
                            "ffn_vs_attn_grad_diff_ratio": result["ffn_vs_attn_grad_diff_ratio"],
                            "final_test_acc_pct": fmt_float(result["final_test_acc_pct"]),
                            "final_train_acc_pct": fmt_float(result["final_train_acc_pct"]),
                        })

                    # Checkpoint after each configuration completes (append to combined CSV)
                    checkpoint_path = RESULTS / f"exp3_training_{args.dataset}.csv"
                    checkpoint_results(rows_epoch, rows_run, rows_wilcoxon, checkpoint_path, ALL_COLUMNS)

    # Wilcoxon tests: compare each activation against relu_0.0 (s=0)
    # For each (arch, norm, precision, dataset, metric), compare paired seeds
    from scipy.stats import wilcoxon

    metrics_to_test = ["final_test_acc_pct", "loss_total_variation"]
    baseline_activation = "relu_0.0"
    comparison_activations = ["relu_0.5", "relu_1.0", "gelu"]

    # Only run Wilcoxon if we have at least 2 seeds
    run_wilcoxon = args.seeds >= 2

    if run_wilcoxon:
        for arch in archs:
            norms = MLP_NORMS if arch == "mlp" else TRANSFORMER_NORMS
            if args.norm:
                norms = [args.norm]

            for norm in norms:
                for precision in precisions:
                    for metric in metrics_to_test:
                        for comp_activation in comparison_activations:
                            # Collect paired values
                            baseline_vals = []
                            comp_vals = []
                            for seed in range(1, args.seeds + 1):
                                # Find baseline
                                baseline_row = next(
                                    (r for r in rows_run
                                     if r["architecture"] == arch and r["norm"] == norm
                                     and r["precision"] == precision and r["activation"] == baseline_activation
                                     and r["seed"] == seed and r["dataset"] == args.dataset), None)
                                comp_row = next(
                                    (r for r in rows_run
                                     if r["architecture"] == arch and r["norm"] == norm
                                     and r["precision"] == precision and r["activation"] == comp_activation
                                     and r["seed"] == seed and r["dataset"] == args.dataset), None)

                                if baseline_row and comp_row:
                                    baseline_vals.append(float(baseline_row[metric]))
                                    comp_vals.append(float(comp_row[metric]))

                            if len(baseline_vals) == args.seeds and len(comp_vals) == args.seeds:
                                try:
                                    stat, p_val = wilcoxon(comp_vals, baseline_vals, alternative="two-sided")
                                    median_diff = float(np.median(np.array(comp_vals) - np.array(baseline_vals)))
                                    rows_wilcoxon.append({
                                        "record_type": "wilcoxon",
                                        "dataset": args.dataset,
                                        "architecture": arch,
                                        "norm": norm,
                                        "precision": precision,
                                        "activation": comp_activation,
                                        "baseline_activation": baseline_activation,
                                        "metric": metric,
                                        "wilcoxon_p_val": fmt_float(p_val),
                                        "paired_median_diff": fmt_float(median_diff),
                                        "n_pairs": len(baseline_vals),
                                    })
                                    print(f"  Wilcoxon {arch} {norm} {precision} {comp_activation} vs {baseline_activation} "
                                          f"({metric}): p={p_val:.4g}, median_diff={median_diff:.4g}", flush=True)
                                except Exception as e:
                                    print(f"  Wilcoxon failed: {e}", flush=True)
                                    rows_wilcoxon.append({
                                        "record_type": "wilcoxon",
                                        "dataset": args.dataset,
                                        "architecture": arch,
                                        "norm": norm,
                                        "precision": precision,
                                        "activation": comp_activation,
                                        "baseline_activation": baseline_activation,
                                        "metric": metric,
                                        "wilcoxon_p_val": "",
                                        "paired_median_diff": "",
                                        "n_pairs": len(baseline_vals),
                                        "error": str(e),
                                    })

    if not run_wilcoxon:
        print(f"  Skipping Wilcoxon tests (need at least 2 seeds, got {args.seeds})", flush=True)

    # Write CSVs - always append to combined CSV
    csv_name = f"exp3_training_{args.dataset}.csv"
    write_csv(RESULTS / csv_name, rows_epoch + rows_run + rows_wilcoxon, ALL_COLUMNS, append=True)

    # Meta
    meta = {
        "script": "run_exp3_training.py",
        "git_commit": None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "dataset": args.dataset,
        "architectures": archs,
        "precisions": precisions,
        "norms": norms if 'norms' in locals() else (list(MLP_NORMS) + list(TRANSFORMER_NORMS)),
        "activations": [a[0] for a in activations],
        "epochs": args.epochs,
        "seeds": list(range(1, args.seeds + 1)),
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "batch_index_file": str(batch_index_path.relative_to(ROOT)),
        "csv": f"results/{csv_name}",
    }
    meta_path = RESULTS / f"run_exp3_training_{args.dataset}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {csv_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
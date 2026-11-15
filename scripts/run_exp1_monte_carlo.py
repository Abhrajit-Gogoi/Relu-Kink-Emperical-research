"""Static Monte Carlo bifurcation volume.

``--mode reproduction`` is Stage 2. ``--mode full`` is Stage 3: 4 precisions,
4 norms, and 2 bias settings, 32 cells, each with slopes 0.5 and 1 against 0.
Stage 3 is prepared here and is not started by importing this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hooks_and_metrics import (  # noqa: E402
    MC_COLUMNS,
    KinkCollisionTracker,
    attach_kink_hooks,
    capture_grads,
    empty_cell_stats,
    fmt_float,
    fmt_quartiles,
    hoeffding_ci_95,
    mcdiarmid_ci_95,
    median_cosine,
    update_pair,
)
from src.models import build_mlp  # noqa: E402

RESULTS = ROOT / "results"
BATCH_SIZE = 128
REPRODUCTION_PRECISIONS = ("float64", "float32", "float16")
REPRODUCTION_SLOPES = (0.5, 1.0)
STAGE3_PRECISIONS = ("float64", "float32", "float16", "bfloat16")
STAGE3_NORMS = ("NoNorm", "BatchNorm1d", "LayerNorm", "RMSNorm")
DTYPE_OF = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_mnist_train(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision import datasets

    dataset = datasets.MNIST(str(root), train=True, download=True)
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


def reproduction_jobs() -> list[dict]:
    return [
        {
            "precision": precision,
            "norm": "NoNorm",
            "with_bias_and_shift": True,
            "keep_native_dtype": True,
        }
        for precision in REPRODUCTION_PRECISIONS
    ]


def result_paths(mode: str, precision: str | None, official_sample: bool) -> tuple[str, str]:
    """CSV and meta paths relative to ``results/``.

    An official per-precision Stage 3 run writes under ``parts/`` so four
    precisions can run together. The Stage 2 meta file is a different name
    from the Stage 3 meta file.
    """
    if mode == "reproduction":
        if precision is not None and official_sample:
            return (
                f"parts/reproduction_mc_{precision}.csv",
                f"parts/run_exp1_monte_carlo_meta_{precision}.json",
            )
        csv_name = "reproduction_mc.csv" if official_sample and precision is None else "reproduction_mc_partial.csv"
        return csv_name, "run_exp1_monte_carlo_meta.json"
    if precision is not None and official_sample:
        return (
            f"parts/exp1_monte_carlo_{precision}.csv",
            f"parts/exp1_monte_carlo_meta_{precision}.json",
        )
    csv_name = "exp1_monte_carlo.csv" if official_sample and precision is None else "exp1_monte_carlo_partial.csv"
    return csv_name, "exp1_monte_carlo_meta.json"


def load_saved_batches(path: Path, n_batches: int, batch_size: int, n_items: int, seed: int) -> torch.Tensor:
    """Reuse the Stage 2 index file when it is already on disk.

    Regenerating the permutation after a PyTorch upgrade can change the
    batches. Stage 3 reads the first ``n_batches`` rows of the saved file.
    """
    import numpy as np

    if path.exists():
        stored = torch.from_numpy(np.load(path)).to(torch.long)
        if stored.ndim != 2 or stored.shape[1] != batch_size or stored.shape[0] < n_batches:
            raise ValueError(f"{path} has shape {tuple(stored.shape)}, need at least ({n_batches}, {batch_size})")
        return stored[:n_batches].contiguous()
    made = make_batch_indices(n_items, n_batches, batch_size, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, made.numpy())
    return made


def failure_rows(job: dict, slopes: tuple[float, ...], m_inits: int, q_batches: int) -> list[dict]:
    """Identity columns for a cell whose dtype could not run. Metrics stay blank."""
    rows = []
    for slope in slopes:
        row = {column: "" for column in MC_COLUMNS}
        row["precision"] = job["precision"]
        row["norm"] = job["norm"]
        row["with_bias_and_shift"] = str(job["with_bias_and_shift"]).lower()
        row["s"] = _slope_label(slope)
        row["keep_native_dtype"] = str(job["keep_native_dtype"]).lower()
        row["hoeffding_ci_95"] = fmt_float(hoeffding_ci_95(m_inits))
        row["mcdiarmid_ci_95"] = fmt_float(mcdiarmid_ci_95(m_inits, q_batches))
        rows.append(row)
    return rows


def stage3_jobs() -> list[dict]:
    jobs = []
    for precision in STAGE3_PRECISIONS:
        for norm in STAGE3_NORMS:
            for with_bias in (True, False):
                jobs.append(
                    {
                        "precision": precision,
                        "norm": norm,
                        "with_bias_and_shift": with_bias,
                        "keep_native_dtype": True,
                    }
                )
    return jobs


def run_cell_counts(
    job: dict,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_index: torch.Tensor,
    m_inits: int,
    slopes: tuple[float, ...],
    init_seed0: int,
    widths: tuple[int, ...] = (784, 256, 256, 256, 10),
    device: torch.device | None = None,
) -> tuple[list[dict], dict]:
    from src.determinism import preferred_device

    dtype = DTYPE_OF[job["precision"]]
    target = preferred_device() if device is None else device
    tracker = KinkCollisionTracker()
    stats = {slope: empty_cell_stats() for slope in slopes}
    weights_hit = {slope: 0 for slope in slopes}
    for draw in range(m_inits):
        model = build_mlp(
            seed=init_seed0 + draw,
            dtype=dtype,
            s=0.0,
            widths=widths,
            norm=job["norm"],
            with_bias_and_shift=job["with_bias_and_shift"],
            keep_native_dtype=job["keep_native_dtype"],
            device=target,
        )
        handles = attach_kink_hooks(model, tracker)
        hit = {slope: False for slope in slopes}
        for batch in batch_index:
            inputs = images.index_select(0, batch).to(device=target, dtype=dtype)
            targets = labels.index_select(0, batch).to(device=target)
            tracker.recording = True
            baseline = capture_grads(model, inputs, targets, 0.0)
            tracker.recording = False
            for slope in slopes:
                other = capture_grads(model, inputs, targets, slope)
                impacted_before = stats[slope].n_impacted
                update_pair(stats[slope], baseline, other)
                if stats[slope].n_impacted > impacted_before:
                    hit[slope] = True
        for slope in slopes:
            weights_hit[slope] += int(hit[slope])
        for handle in handles:
            handle.remove()
        if (draw + 1) % 10 == 0 or draw + 1 == m_inits:
            print(f"{job['precision']} {job['norm']} bias={job['with_bias_and_shift']} init {draw + 1}/{m_inits}", flush=True)

    snap = tracker.snapshot()
    hist_values = [snap["exact"], *snap["bins"]]
    rows = []
    q_batches = int(batch_index.shape[0])
    for slope in slopes:
        cell = stats[slope]
        rel = fmt_quartiles(cell.rel_l2)
        ratio = fmt_quartiles(cell.norm_ratios)
        row = {
            "precision": job["precision"],
            "norm": job["norm"],
            "with_bias_and_shift": str(job["with_bias_and_shift"]).lower(),
            "s": _slope_label(slope),
            "keep_native_dtype": str(job["keep_native_dtype"]).lower(),
            "exact_zero_rate": fmt_float(snap["zero_rate"]),
            "bifurcation_weight_prop": fmt_float(weights_hit[slope] / m_inits),
            "hoeffding_ci_95": fmt_float(hoeffding_ci_95(m_inits)),
            "bifurcation_batch_prop": fmt_float(cell.n_impacted / max(1, cell.n_pairs)),
            "mcdiarmid_ci_95": fmt_float(mcdiarmid_ci_95(m_inits, q_batches)),
            "rel_l2_q1": rel[0],
            "rel_l2_q2": rel[1],
            "rel_l2_q3": rel[2],
            "grad_norm_ratio_q1": ratio[0],
            "grad_norm_ratio_q2": ratio[1],
            "grad_norm_ratio_q3": ratio[2],
            "grad_cosine_l1": median_cosine(cell.layer_cosines[0]),
            "grad_cosine_l2": median_cosine(cell.layer_cosines[1]),
            "grad_cosine_l3": median_cosine(cell.layer_cosines[2]),
            "grad_cosine_l4": median_cosine(cell.layer_cosines[3]),
            "n_impacted_pairs": str(cell.n_impacted),
        }
        for column, value in zip(
            (
                "hist_exact_zero",
                "hist_le_m12",
                "hist_m12_m8",
                "hist_m8_m5",
                "hist_m5_m3",
                "hist_m3_m1",
                "hist_gt_m1",
            ),
            hist_values,
        ):
            row[column] = str(int(value))
        rows.append(row)
    info = {
        "zero_rate": snap["zero_rate"],
        "total_preactivations": snap["total"],
        "n_nonfinite": {_slope_label(slope): stats[slope].n_nonfinite for slope in slopes},
        "n_pairs": {_slope_label(slope): stats[slope].n_pairs for slope in slopes},
        "weights_hit": {_slope_label(slope): weights_hit[slope] for slope in slopes},
    }
    return rows, info


def _slope_label(slope: float) -> str:
    return "0.5" if slope == 0.5 else f"{slope:.1f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reproduction", "full"), default="reproduction")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--init-seed0", type=int, default=0)
    parser.add_argument("--batch-seed", type=int, default=0)
    parser.add_argument("--precision", choices=tuple(DTYPE_OF), default=None)
    args = parser.parse_args()
    from src.determinism import preferred_device

    print(f"device {preferred_device()}", flush=True)
    m_inits = args.m if args.m is not None else 200
    q_batches = args.q if args.q is not None else 100
    jobs = reproduction_jobs() if args.mode == "reproduction" else stage3_jobs()
    if args.precision is not None:
        jobs = [job for job in jobs if job["precision"] == args.precision]
    images, labels = load_mnist_train(ROOT / "data")
    # The on-disk file has 200 batches from Stage 2. Read it; do not draw a new permutation.
    index_path = RESULTS / "reproduction_batch_indices.npy"
    requested = q_batches if index_path.exists() else max(q_batches, 200)
    used = load_saved_batches(index_path, requested, BATCH_SIZE, images.shape[0], args.batch_seed)
    used = used[:q_batches]
    rows: list[dict] = []
    cell_info = []
    for job in jobs:
        print(
            f"{args.mode} {job['precision']} {job['norm']} bias={job['with_bias_and_shift']}",
            flush=True,
        )
        try:
            job_rows, info = run_cell_counts(job, images, labels, used, m_inits, REPRODUCTION_SLOPES, args.init_seed0)
            cell_info.append({"job": job, "status": "ok", **info})
        except RuntimeError as exc:
            job_rows = failure_rows(job, REPRODUCTION_SLOPES, m_inits, q_batches)
            info = {"status": "failed", "exception": f"{type(exc).__name__}: {exc}"}
            cell_info.append({"job": job, **info})
            print(f"failed cell {job['precision']} {job['norm']} bias={job['with_bias_and_shift']}: {exc}", flush=True)
        rows.extend(job_rows)
    official_sample = args.m is None and args.q is None
    out_name, meta_name = result_paths(args.mode, args.precision, official_sample)
    write_csv(RESULTS / out_name, rows)
    device = preferred_device()
    meta = {
        "script": "run_exp1_monte_carlo.py",
        "mode": args.mode,
        "git_commit": None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "m": m_inits,
        "q": q_batches,
        "batch_size": BATCH_SIZE,
        "init_seed0": args.init_seed0,
        "batch_seed": args.batch_seed,
        "batch_index_file": "results/reproduction_batch_indices.npy",
        "slopes": ["0.5", "1.0"],
        "csv": f"results/{out_name}",
        "cells": cell_info,
    }
    meta_path = RESULTS / meta_name
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote results/{out_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

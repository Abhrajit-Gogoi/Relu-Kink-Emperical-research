"""Static Monte Carlo bifurcation volume. Stage 2 runs ``--mode reproduction``."""

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
) -> tuple[list[dict], dict]:
    dtype = DTYPE_OF[job["precision"]]
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
        )
        handles = attach_kink_hooks(model, tracker)
        hit = {slope: False for slope in slopes}
        for batch in batch_index:
            inputs = images.index_select(0, batch).to(dtype)
            targets = labels.index_select(0, batch)
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
    m_inits = args.m if args.m is not None else 200
    q_batches = args.q if args.q is not None else 100
    jobs = reproduction_jobs() if args.mode == "reproduction" else stage3_jobs()
    if args.precision is not None:
        jobs = [job for job in jobs if job["precision"] == args.precision]
    images, labels = load_mnist_train(ROOT / "data")
    # Trajectory stage uses 200 batches. Keep at least that many indices on disk.
    n_batches = max(q_batches, 200)
    indices = make_batch_indices(images.shape[0], n_batches, BATCH_SIZE, args.batch_seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    import numpy as np

    index_path = RESULTS / "reproduction_batch_indices.npy"
    if not index_path.exists():
        np.save(index_path, indices.numpy())
    used = indices[:q_batches]
    rows: list[dict] = []
    cell_info = []
    for job in jobs:
        job_rows, info = run_cell_counts(job, images, labels, used, m_inits, REPRODUCTION_SLOPES, args.init_seed0)
        rows.extend(job_rows)
        cell_info.append({"job": job, **info})
    official = args.m is None and args.q is None and args.precision is None
    if args.precision is not None and args.m is None and args.q is None and args.mode == "reproduction":
        out_name = f"parts/reproduction_mc_{args.precision}.csv"
    elif args.mode == "reproduction":
        out_name = "reproduction_mc.csv" if official else "reproduction_mc_partial.csv"
    else:
        out_name = "exp1_monte_carlo.csv" if official else "exp1_monte_carlo_partial.csv"
    write_csv(RESULTS / out_name, rows)
    meta = {
        "script": "run_exp1_monte_carlo.py",
        "mode": args.mode,
        "git_commit": None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "m": m_inits,
        "q": q_batches,
        "batch_size": BATCH_SIZE,
        "init_seed0": args.init_seed0,
        "batch_seed": args.batch_seed,
        "batch_index_file": "results/reproduction_batch_indices.npy",
        "slopes": list(REPRODUCTION_SLOPES),
        "csv": f"results/{out_name}",
        "cells": cell_info,
    }
    if args.precision is not None and args.m is None and args.q is None and args.mode == "reproduction":
        meta_path = RESULTS / "parts" / f"run_exp1_monte_carlo_meta_{args.precision}.json"
    else:
        meta_path = RESULTS / "run_exp1_monte_carlo_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote results/{out_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

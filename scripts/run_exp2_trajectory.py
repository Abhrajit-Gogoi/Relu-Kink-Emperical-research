"""Coupled SGD trajectories. Stage 2 runs ``--mode reproduction`` on topology A."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exp1_monte_carlo import (  # noqa: E402
    BATCH_SIZE,
    DTYPE_OF,
    REPRODUCTION_PRECISIONS,
    load_mnist_train,
    make_batch_indices,
)
from src.determinism import seed_everything  # noqa: E402
from src.hooks_and_metrics import (  # noqa: E402
    SUMMARY_COLUMNS,
    TRAJ_COLUMNS,
    KinkCollisionTracker,
    attach_kink_hooks,
    capture_grads,
    cosine,
    fmt_float,
    parameter_cosine,
    parameter_l1,
    parameter_l2,
    quartiles,
)
from src.models import ConfigurableMLP, MLP_WIDTHS, initialize_kaiming_, set_slope  # noqa: E402

RESULTS = ROOT / "results"
LR = 0.01
SLOPE_NAMES = (("A", 0.0), ("B", 0.0), ("s05", 0.5), ("s1", 1.0))


def spawn_twins(seed: int, dtype: torch.dtype, norm: str, widths: tuple[int, ...]) -> tuple[dict, dict]:
    seed_everything(seed)
    template = ConfigurableMLP(widths=widths, s=0.0, norm=norm, with_bias_and_shift=True, keep_native_dtype=True)
    initialize_kaiming_(template)
    state = copy.deepcopy(template.state_dict())
    models = {}
    optimizers = {}
    for name, slope in SLOPE_NAMES:
        model = ConfigurableMLP(widths=widths, s=slope, norm=norm, with_bias_and_shift=True, keep_native_dtype=True)
        model.load_state_dict(state)
        if dtype != torch.float32:
            model.to(dtype)
        set_slope(model, slope)
        models[name] = model
        optimizers[name] = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0, weight_decay=0.0)
    return models, optimizers


def run_seed(
    seed: int,
    precision: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_index: torch.Tensor,
    norm: str = "NoNorm",
    topology: str = "A",
    widths: tuple[int, ...] = MLP_WIDTHS,
) -> list[dict]:
    dtype = DTYPE_OF[precision]
    models, optimizers = spawn_twins(seed, dtype, norm, widths)
    tracker = KinkCollisionTracker()
    attach_kink_hooks(models["A"], tracker)
    kstar = {0.5: None, 1.0: None}
    pending = []
    for step, batch in enumerate(batch_index, start=1):
        inputs = images.index_select(0, batch).to(dtype)
        targets = labels.index_select(0, batch)
        tracker.reset()
        grads = {}
        for name, slope in SLOPE_NAMES:
            grads[name] = capture_grads(models[name], inputs, targets, slope)
        for name, opt in optimizers.items():
            opt.step()
        snap = tracker.snapshot()
        twin_l1 = parameter_l1(models["A"], models["B"])
        for name, slope in (("s05", 0.5), ("s1", 1.0)):
            distance = parameter_l1(models[name], models["A"])
            if distance > 0.0 and kstar[slope] is None:
                kstar[slope] = step
            grad_cos = ""
            if not grads[name].nonfinite and not grads["A"].nonfinite and grads[name].layers:
                grad_cos = cosine(grads[name].flat(), grads["A"].flat())
            pending.append(
                {
                    "record_type": "step",
                    "seed": seed,
                    "topology": topology,
                    "precision": precision,
                    "norm": norm,
                    "s": slope,
                    "step_k": step,
                    "min_abs_preact": snap["min_abs"],
                    "exact_zero_count_k": snap["exact"],
                    "traj_l1_dist": distance,
                    "traj_l2_dist": parameter_l2(models[name], models["A"]),
                    "weight_cosine_sim": parameter_cosine(models[name], models["A"]),
                    "grad_cosine_sim_step": grad_cos,
                    "twin_l1_dist": twin_l1,
                }
            )
    k_steps = int(batch_index.shape[0])
    rows = []
    for record in pending:
        filled = kstar[record["s"]]
        record["first_bifurcation_step"] = k_steps + 1 if filled is None else filled
        rows.append(record)
    return rows


def summarize_trajectories(records: list[dict]) -> list[dict]:
    checkpoints = (50, 100, 200)
    buckets: dict[tuple, list[dict]] = {}
    kstars: dict[tuple, list[int]] = {}
    for record in records:
        identity = (record["precision"], record["norm"], record["topology"], record["s"])
        kstars.setdefault((*identity, record["seed"]), record["first_bifurcation_step"])
        if record["step_k"] in checkpoints:
            buckets.setdefault((*identity, record["step_k"]), []).append(record)
    kstar_by_cell: dict[tuple, list[int]] = {}
    for (precision, norm, topology, slope, _seed), value in kstars.items():
        kstar_by_cell.setdefault((precision, norm, topology, slope), []).append(value)
    summary = []
    for (precision, norm, topology, slope, step), group in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][3], item[0][4])):
        l1 = quartiles([float(row["traj_l1_dist"]) for row in group])
        l2 = quartiles([float(row["traj_l2_dist"]) for row in group])
        cosines = [float(row["grad_cosine_sim_step"]) for row in group if row["grad_cosine_sim_step"] != ""]
        cosine_q = quartiles(cosines)
        k_values = kstar_by_cell[(precision, norm, topology, slope)]
        k_q = quartiles([float(value) for value in k_values])
        summary.append(
            {
                "precision": precision,
                "norm": norm,
                "topology": topology,
                "s": slope,
                "step_k": step,
                "n_seeds": len(group),
                "l1_q1": None if l1 is None else l1[0],
                "l1_median": None if l1 is None else l1[1],
                "l1_q3": None if l1 is None else l1[2],
                "l1_iqr": None if l1 is None else l1[2] - l1[0],
                "l2_q1": None if l2 is None else l2[0],
                "l2_median": None if l2 is None else l2[1],
                "l2_q3": None if l2 is None else l2[2],
                "grad_cosine_median": None if cosine_q is None else cosine_q[1],
                "kstar_median": None if k_q is None else k_q[1],
            }
        )
    return summary


def _write(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt_float(row[key]) if isinstance(row[key], float) else row[key] for key in columns})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reproduction",), default="reproduction")
    parser.add_argument("--seeds", type=int, default=None, help="Use seeds 1..SEEDS. Default 10.")
    parser.add_argument("--k", type=int, default=None, help="SGD steps. Default 200.")
    parser.add_argument("--batch-seed", type=int, default=0)
    parser.add_argument("--precision", choices=REPRODUCTION_PRECISIONS, default=None)
    args = parser.parse_args()
    n_seeds = 10 if args.seeds is None else args.seeds
    k_steps = 200 if args.k is None else args.k
    images, labels = load_mnist_train(ROOT / "data")
    indices = make_batch_indices(images.shape[0], k_steps, BATCH_SIZE, args.batch_seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    index_path = RESULTS / "reproduction_batch_indices.npy"
    if not index_path.exists():
        np.save(index_path, indices.numpy())
    precisions = (args.precision,) if args.precision is not None else REPRODUCTION_PRECISIONS
    records: list[dict] = []
    for precision in precisions:
        for seed in range(1, n_seeds + 1):
            print(f"trajectory {precision} seed {seed}/{n_seeds}", flush=True)
            records.extend(
                run_seed(seed, precision, images, labels, indices, norm="NoNorm", topology="A", widths=MLP_WIDTHS)
            )
    official = args.seeds is None and args.k is None and args.precision is None
    if args.precision is not None and args.seeds is None and args.k is None:
        traj_name = f"parts/reproduction_traj_{args.precision}.csv"
        summary_name = f"parts/reproduction_summary_{args.precision}.csv"
    else:
        traj_name = "reproduction_traj.csv" if official else "reproduction_traj_partial.csv"
        summary_name = "reproduction_summary.csv" if official else "reproduction_summary_partial.csv"
    _write(RESULTS / traj_name, TRAJ_COLUMNS, records)
    _write(RESULTS / summary_name, SUMMARY_COLUMNS, summarize_trajectories(records))
    meta = {
        "script": "run_exp2_trajectory.py",
        "mode": args.mode,
        "git_commit": None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seeds": list(range(1, n_seeds + 1)),
        "k": k_steps,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "precisions": list(precisions),
        "topology": "A",
        "norm": "NoNorm",
        "batch_index_file": "results/reproduction_batch_indices.npy",
        "csv": f"results/{traj_name}",
        "summary_csv": f"results/{summary_name}",
        "max_twin_l1": max(float(row["twin_l1_dist"]) for row in records),
    }
    if args.precision is not None and args.seeds is None and args.k is None:
        meta_path = RESULTS / "parts" / f"run_exp2_trajectory_meta_{args.precision}.json"
    else:
        meta_path = RESULTS / "run_exp2_trajectory_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote results/{traj_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

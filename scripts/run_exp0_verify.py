"""Experiment 0: probe ProxyReLU, compare the Bertoin emulation, run the s = 0 twin."""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.activations import (  # noqa: E402
    DTYPES,
    EMULATION_DTYPES,
    EMULATION_SLOPES,
    PROBE_SLOPES,
    ProxyReLU,
    analytical_proxy_derivative,
    bertoin_emulated_relu,
    probe_passes,
)
from src.determinism import s0_twin_control, seed_everything  # noqa: E402

RESULTS = ROOT / "results"
N_SAMPLES = 1_000_000
DRAW_SEED = 0
CSV_COLUMNS = (
    "dtype",
    "s",
    "grad_probe_pass",
    "max_abs_forward_err_nonzero",
    "frac_nonzero_grad_mismatch",
    "n_nonzero_samples",
)


_S_LABELS = {
    0.0: "0.0",
    0.1: "0.1",
    0.25: "0.25",
    0.3: "0.3",
    0.5: "0.5",
    0.75: "0.75",
    1.0: "1.0",
}


def _format_s(s: float) -> str:
    return _S_LABELS[s]


def _blank_row(dtype_name: str, s: float) -> dict:
    return {
        "dtype": dtype_name,
        "s": _format_s(s),
        "grad_probe_pass": "",
        "max_abs_forward_err_nonzero": "",
        "frac_nonzero_grad_mismatch": "",
        "n_nonzero_samples": "",
    }


def draw_nonzero(dtype: torch.dtype, n: int, seed: int) -> torch.Tensor:
    """Draw n uniforms in FP32, cast into ``dtype``, and drop exact zeros."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    draws = torch.empty(n, dtype=torch.float32)
    draws.uniform_(-5.0, 5.0, generator=generator)
    native = draws.to(dtype)
    return native[native != 0]


def emulation_stats(x: torch.Tensor, s: float) -> tuple[float, float]:
    leaf = x.detach().requires_grad_(True)
    emulated = bertoin_emulated_relu(leaf, s)
    grad, = torch.autograd.grad(emulated.sum(), leaf)
    with torch.no_grad():
        proxy = ProxyReLU(s)(x)
        forward_err = (proxy - emulated.detach()).abs().to(torch.float64).max().item()
        reference = analytical_proxy_derivative(x, s)
        n_mismatch = int((grad != reference).sum().item())
    return float(forward_err), n_mismatch / x.numel()


def rounded_epsilons() -> dict:
    values = {}
    for dtype in DTYPES:
        values[str(dtype).replace("torch.", "")] = float(torch.tensor(1e-5, dtype=dtype))
    return values


def main() -> int:
    seed_everything(DRAW_SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: dict[tuple[str, str], dict] = {}

    def row_for(dtype_name: str, s: float) -> dict:
        key = (dtype_name, _format_s(s))
        if key not in rows:
            rows[key] = _blank_row(dtype_name, s)
            rows[key]["s"] = _format_s(s)
        return rows[key]

    probe_ok = True
    for dtype in DTYPES:
        dtype_name = str(dtype).replace("torch.", "")
        for s in PROBE_SLOPES:
            passed = probe_passes(dtype, s)
            probe_ok = probe_ok and passed
            row_for(dtype_name, s)["grad_probe_pass"] = "true" if passed else "false"

    sample_counts = {}
    for dtype in EMULATION_DTYPES:
        dtype_name = str(dtype).replace("torch.", "")
        x = draw_nonzero(dtype, N_SAMPLES, DRAW_SEED)
        sample_counts[dtype_name] = int(x.numel())
        for s in EMULATION_SLOPES:
            forward_err, frac = emulation_stats(x, s)
            record = row_for(dtype_name, s)
            record["max_abs_forward_err_nonzero"] = f"{forward_err:.17g}"
            record["frac_nonzero_grad_mismatch"] = f"{frac:.17g}"
            record["n_nonzero_samples"] = str(x.numel())

    twin = s0_twin_control(
        steps=20,
        seed=DRAW_SEED,
        batch_path=RESULTS / "exp0_twin_batches.npz",
    )
    twin_ok = twin["final_l1"] == 0.0 and twin["max_l1"] == 0.0 and twin["parameters_equal"]

    csv_path = RESULTS / "exp0_verification.csv"
    ordered = sorted(rows.values(), key=lambda record: (record["dtype"], float(record["s"])))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = {
        "script": "run_exp0_verify.py",
        "git_commit": None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": device,
        "dtypes": [str(dtype).replace("torch.", "") for dtype in DTYPES],
        "seed_list": [DRAW_SEED],
        "rounded_epsilon": rounded_epsilons(),
        "batch_index_file": "results/exp0_twin_batches.npz",
        "deterministic_algorithms": twin["seed_info"]["deterministic"],
        "deterministic_error": twin["seed_info"]["deterministic_error"],
        "n_uniform_draws": N_SAMPLES,
        "uniform_interval": [-5.0, 5.0],
        "draw_procedure": "float32 uniform_, cast to the working dtype, drop x == 0",
        "n_nonzero_samples": sample_counts,
        "probe_all_passed": probe_ok,
        "twin": {
            "final_l1": twin["final_l1"],
            "max_l1": twin["max_l1"],
            "parameters_equal": twin["parameters_equal"],
            "steps": twin["steps"],
            "seed": twin["seed"],
            "batch_size": twin["batch_size"],
            "lr": twin["lr"],
            "architecture": "Linear(16, 32) -> ProxyReLU(0) -> Linear(32, 4)",
        },
        "gate_passed": bool(probe_ok and twin_ok),
    }
    meta_path = RESULTS / "run_exp0_verify_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"probe_all_passed={probe_ok} twin_l1={twin['final_l1']} gate_passed={meta['gate_passed']}")
    return 0 if meta["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

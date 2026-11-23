#!/usr/bin/env python3
"""Stage 6 — Tables and figures.

Reads *only* the CSV logs under ``relu-kink/results/`` and writes:

* LaTeX ``booktabs`` / ``longtable`` tables to ``relu-kink/paper/tables/``
* Vector PDF figures to ``relu-kink/figures/``

Every number is sourced from a CSV. Nothing is hand-typed into a table or
plotted from a literal value. Per the Stage 6 gate, the script raises
(SystemExit) if a required CSV column is missing.

Run from ``relu-kink/``:

    python scripts/build_tables_and_figures.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Matplotlib with a non-interactive backend so the script can run headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS = Path("results")
TABLES = Path("paper/tables")
FIGURES = Path("figures")

CSV_EXP0 = "exp0_verification.csv"
CSV_EXP1 = "exp1_monte_carlo.csv"
CSV_EXP2_STEP = "exp2_trajectories.csv"
CSV_EXP2_SUMMARY = "exp2_summary.csv"
CSV_EXP3 = "exp3_training_mnist.csv"

# ---------------------------------------------------------------------------
# Required-column validation (Stage 6 gate)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: dict[str, list[str]] = {
    CSV_EXP0: [
        "dtype", "s", "grad_probe_pass",
        "max_abs_forward_err_nonzero", "frac_nonzero_grad_mismatch",
    ],
    CSV_EXP1: [
        "precision", "norm", "with_bias_and_shift", "s", "keep_native_dtype",
        "exact_zero_rate", "bifurcation_weight_prop", "hoeffding_ci_95",
        "bifurcation_batch_prop", "mcdiarmid_ci_95",
        "rel_l2_q1", "rel_l2_q2", "rel_l2_q3",
        "grad_cosine_l1", "grad_cosine_l2", "grad_cosine_l3", "grad_cosine_l4",
        "hist_exact_zero", "hist_le_m12", "hist_m12_m8", "hist_m8_m5",
        "hist_m5_m3", "hist_m3_m1", "hist_gt_m1",
    ],
    CSV_EXP2_STEP: [
        "record_type", "seed", "topology", "precision", "norm", "s", "step_k",
        "min_abs_preact", "traj_l1_dist", "first_bifurcation_step",
    ],
    CSV_EXP2_SUMMARY: [
        "precision", "norm", "topology", "s", "step_k",
        "l1_median", "l1_q1", "l1_q3", "kstar_median", "grad_cosine_median",
    ],
    CSV_EXP3: [
        "record_type", "dataset", "architecture", "norm", "precision",
        "activation", "final_test_acc_pct", "loss_total_variation",
        "baseline_activation", "wilcoxon_p_val", "paired_median_diff",
    ],
}

PREC_LABEL = {"float64": "FP64", "float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}
PREC_ORDER = ["float64", "float32", "float16", "bfloat16"]
NORM_ORDER = ["NoNorm", "BatchNorm1d", "LayerNorm", "RMSNorm"]

ACT_LABEL = {
    "relu_0.0": "ReLU ($s=0$)",
    "relu_0.5": "ReLU ($s=0.5$)",
    "relu_1.0": "ReLU ($s=1$)",
    "gelu": "GELU",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(
            f"Stage 6 gate failed: required CSV not found: {path}"
        )
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = [c for c in REQUIRED_COLUMNS[name] if c not in rows[0].keys()] if rows else []
    if missing:
        raise SystemExit(
            f"Stage 6 gate failed: {name} is missing required columns: {missing}"
        )
    return rows


def _f(val: str, fmt: str = ".4f", default: str = "--") -> str:
    """Format a CSV cell; return ``default`` when empty / NaN."""
    if val is None or val == "" or val.lower() == "nan":
        return default
    try:
        return format(float(val), fmt)
    except (ValueError, TypeError):
        return default


def _tex_escape(text: str) -> str:
    s = str(text)
    s = s.replace("\\", r"\textbackslash{}")
    return (
        s.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _write_table(path: Path, caption: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="")
    print(f"[table] wrote {path}  ({caption})")


def _to_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Table 1 — Experiment 0 (operator validity)
# ---------------------------------------------------------------------------

def build_table1(rows: list[dict]) -> str:
    body_lines = [
        "\\begin{tabular}{l c c c c}",
        "\\toprule",
        "dtype & $s$ & probe pass & $\\max |\\Delta f|$ & frac nonzero grad mismatch \\\\",
        "\\midrule",
    ]
    for r in rows:
        s_val = _tex_escape(r["s"])
        probe = str(r["grad_probe_pass"]).lower()
        if probe == "true":
            probe_txt = "yes"
            fwd_err = _f(r["max_abs_forward_err_nonzero"], ".2e")
            mismatch = _f(r["frac_nonzero_grad_mismatch"], ".2e")
        else:
            # s in {0.1, 0.3} were only run for the Bertoin emulation audit,
            # not the six-point unit probe. Emulation results are 0.00e+00.
            probe_txt = "---"
            fwd_err_val = r.get("max_abs_forward_err_nonzero", "") or ""
            mismatch_val = r.get("frac_nonzero_grad_mismatch", "") or ""
            fwd_err = _f(fwd_err_val, ".2e") if fwd_err_val else "---"
            mismatch = _f(mismatch_val, ".2e") if mismatch_val else "---"
        body_lines.append(
            f"{_tex_escape(r['dtype'])} & "
            f"{s_val} & "
            f"{probe_txt} & "
            f"{fwd_err} & "
            f"{mismatch} \\\\"
        )
    body_lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Table 2 — Experiment 1 (H1, H2)
# ---------------------------------------------------------------------------

def build_table2(rows: list[dict]) -> str:
    body_lines = []
    for r in rows:
        hoff_ci = f"({_f(r['hoeffding_ci_95'], '.3f')})"
        mcd_ci = f"({_f(r['mcdiarmid_ci_95'], '.3f')})"
        rel_l2_parts = []
        for c in ("rel_l2_q1", "rel_l2_q2", "rel_l2_q3"):
            v = _f(r[c])
            rel_l2_parts.append(v if v != "—" else "—")
        rel_l2 = " / ".join(rel_l2_parts)
        if rel_l2 == "— / — / —":
            rel_l2 = "—"
        body_lines.append(
            f"{_tex_escape(r['precision'])} & "
            f"{_tex_escape(r['norm'])} & "
            f"{'on' if str(r['with_bias_and_shift']).lower() == 'true' else 'off'} & "
            f"{_tex_escape(r['s'])} & "
            f"{_f(r['exact_zero_rate'], '.2e')} & "
            f"{_f(r['bifurcation_weight_prop'], '.4f')} {hoff_ci} & "
            f"{_f(r['bifurcation_batch_prop'], '.5f')} {mcd_ci} & "
            f"{rel_l2} \\\\"
        )
    return "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Table 3 — Experiment 2 summary (H3), compact: s=1.0 only at step 200
# ---------------------------------------------------------------------------

def build_table3(rows: list[dict]) -> str:
    sub = [r for r in rows if r["step_k"] == "200" and float(r["s"]) == 1.0]
    body_lines = [
        "\\begin{tabular}{l l l c c c c}",
        "\\toprule",
        "precision & norm & topology & $k^\\star$ (median) & "
        "$D_1^{(200)}$ med (q1/q3) & cosine med \\\\",
        "\\midrule",
    ]
    for r in sub:
        l1_med = _f(r['l1_median'], '.4f')
        l1_q1 = _f(r['l1_q1'], '.4f')
        l1_q3 = _f(r['l1_q3'], '.4f')
        if l1_med == "0.0000" and l1_q1 == "0.0000" and l1_q3 == "0.0000":
            l1_str = "0"
        else:
            l1_str = f"{l1_med} ({l1_q1}/{l1_q3})"
        body_lines.append(
            f"{_tex_escape(r['precision'])} & "
            f"{_tex_escape(r['norm'])} & "
            f"{_tex_escape(r['topology'])} & "
            f"{_f(r['kstar_median'], '.0f')} & "
            f"{l1_str} & "
            f"{_f(r['grad_cosine_median'], '.4f')} \\\\"
        )
    body_lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Table 4 — Experiment 3 (H2, H4)
# ---------------------------------------------------------------------------

def build_table4(rows: list[dict]) -> str:
    run_rows = [r for r in rows if r["record_type"] == "run"]
    wil_rows = [r for r in rows if r["record_type"] == "wilcoxon"]

    wil_lookup: dict[tuple, str] = {}
    for r in wil_rows:
        key = (r["norm"], r["precision"], r["activation"], r["metric"])
        wil_lookup[key] = _f(r["wilcoxon_p_val"], ".4f")

    # Aggregate run rows to one row per (norm, precision, activation) cell.
    cells: dict[tuple, list[float]] = {}
    for r in run_rows:
        key = (r["norm"], r["precision"], r["activation"])
        cells.setdefault(key, []).append(
            (_to_float(r["final_test_acc_pct"]), _to_float(r["loss_total_variation"]))
        )

    body_lines = [
        "\\begin{tabular}{l l l c c c}",
        "\\toprule",
        "norm & precision & activation & test acc (\\%) & $\\mathrm{TV}_\\ell$ & Wilcoxon $p$ \\\\",
        "\\midrule",
    ]
    for key in sorted(cells):
        norm, precision, activation = key
        vals = cells[key]
        accs = [v[0] for v in vals if v[0] is not None]
        tvs = [v[1] for v in vals if v[1] is not None]
        mean_acc = float(np.mean(accs)) if accs else np.nan
        mean_tv = float(np.mean(tvs)) if tvs else np.nan
        pval = wil_lookup.get((norm, precision, activation, "final_test_acc_pct"), "--")
        body_lines.append(
            f"{_tex_escape(norm)} & "
            f"{_tex_escape(precision)} & "
            f"{ACT_LABEL.get(activation, activation)} & "
            f"{mean_acc:.2f} & "
            f"{mean_tv:.6f} & "
            f"{pval} \\\\"
        )
    body_lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _maybe_log(ax, values: list[float]) -> None:
    """Use log scale only when every value is strictly positive and finite."""
    clean = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    if clean:
        ax.set_yscale("log")


# ---------------------------------------------------------------------------
# Figure 1 — H3: representative single seed, topology A, NoNorm, s=1 vs s=0
# ---------------------------------------------------------------------------

def figure1(rows: list[dict]) -> None:
    sub = [r for r in rows if r["record_type"] == "step"
           and r["topology"] == "A" and r["norm"] == "NoNorm"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharex=True)
    # Plot a single representative seed (seed 1) to show the exact bifurcation step.
    seed = "1"
    for prec in ["float64", "float32", "bfloat16"]:
        cell = [r for r in sub if r["precision"] == prec
                and r["seed"] == seed and float(r["s"]) == 1.0]
        cell.sort(key=lambda r: int(r["step_k"]))
        steps = [int(r["step_k"]) for r in cell]
        l1 = [_to_float(r["traj_l1_dist"]) for r in cell]
        zmin = [_to_float(r["min_abs_preact"]) for r in cell]
        axes[0].plot(steps, l1, label=PREC_LABEL[prec], lw=1.6)
        axes[1].plot(steps, zmin, label=PREC_LABEL[prec], lw=1.6)

    axes[0].set_ylabel("$D_1^{(k)}(s{=}1)$  (seed 1)")
    axes[1].set_ylabel("$z_{\\min}^{(k)}$  (seed 1)")
    for ax in axes:
        _style_axes(ax, "step $k$", ax.get_ylabel())
    axes[0].legend(title="precision", fontsize=8, title_fontsize=9)
    axes[1].legend(title="precision", fontsize=8, title_fontsize=9)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "figure1.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — H1, H2: bifurcation proportion bar chart (weight and batch)
# ---------------------------------------------------------------------------

def figure_bar_chart(rows: list[dict]) -> None:
    sub = [r for r in rows
           if float(r["s"]) == 1.0 and str(r["with_bias_and_shift"]).lower() == "true"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    x = np.arange(len(PREC_ORDER))
    width = 0.2
    norm_offsets = {"NoNorm": -1.5 * width, "BatchNorm1d": -0.5 * width,
                    "LayerNorm": 0.5 * width, "RMSNorm": 1.5 * width}

    for i, ax in enumerate(axes):
        metric = "bifurcation_weight_prop" if i == 0 else "bifurcation_batch_prop"
        ci = "hoeffding_ci_95" if i == 0 else "mcdiarmid_ci_95"
        ylabel = "$\\hat V_{0,1}$" if i == 0 else "$\\hat B_{0,1}$"
        for norm in NORM_ORDER:
            means, errs_lo, errs_hi = [], [], []
            for prec in PREC_ORDER:
                match = [r for r in sub if r["precision"] == prec and r["norm"] == norm]
                if match:
                    val = _f(match[0][metric], ".6f", "nan")
                    ci_val = _f(match[0][ci], ".6f", "0")
                    means.append(float(val))
                    lo = max(0.0, float(val) - float(ci_val))
                    hi = min(1.0, float(val) + float(ci_val))
                    errs_lo.append(float(val) - lo)
                    errs_hi.append(hi - float(val))
                else:
                    means.append(np.nan); errs_lo.append(0.0); errs_hi.append(0.0)
            means = np.array(means, dtype=float)
            ax.bar(x + norm_offsets[norm], means, width, label=norm,
                   yerr=[errs_lo, errs_hi], capsize=2,
                   error_kw=dict(lw=0.6, capthick=0.6))
        ax.set_xticks(x)
        ax.set_xticklabels([PREC_LABEL[p] for p in PREC_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        _style_axes(ax, "precision", ylabel)
    axes[0].legend(title="norm", fontsize=8, title_fontsize=9, ncol=2)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "figure1.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — H2: near-zero pre-activation histogram, LayerNorm vs RMSNorm
# ---------------------------------------------------------------------------

def figure2(rows: list[dict]) -> None:
    sub = [r for r in rows
           if r["norm"] in ("LayerNorm", "RMSNorm")
           and float(r["s"]) == 1.0
           and str(r["with_bias_and_shift"]).lower() == "true"]
    bins = ["hist_exact_zero", "hist_le_m12", "hist_m12_m8", "hist_m8_m5",
            "hist_m5_m3", "hist_m3_m1", "hist_gt_m1"]
    bin_labels = ["exact 0", "(-inf,-12]", "(-12,-8]", "(-8,-5]",
                  "(-5,-3]", "(-3,-1]", "(-1,inf)"]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.0), sharey=True)
    for i, prec in enumerate(PREC_ORDER):
        ax = axes[i]
        all_vals: list[float] = []
        for j, norm in enumerate(["LayerNorm", "RMSNorm"]):
            match = [r for r in sub if r["precision"] == prec and r["norm"] == norm]
            if not match:
                continue
            vals = [float(match[0][b]) for b in bins]
            all_vals.extend(vals)
            ax.bar(np.arange(len(bins)) + (j - 0.5) * 0.35, vals, 0.35, label=norm)
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(PREC_LABEL[prec], fontsize=9)
        _style_axes(ax, "bin", "count")
        _maybe_log(ax, all_vals)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "figure2.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — H3: bifurcation trajectory, BF16 LayerNorm topology A, mean over 10 seeds
# ---------------------------------------------------------------------------

def figure3(rows: list[dict]) -> None:
    sub = [r for r in rows
           if r["record_type"] == "step"
           and r["topology"] == "A"
           and r["norm"] == "LayerNorm"
           and r["precision"] == "bfloat16"
           and float(r["s"]) == 1.0]

    steps = sorted({int(r["step_k"]) for r in sub})
    mean_l1 = []
    mean_zmin = []
    for step in steps:
        l1_vals = [_to_float(r["traj_l1_dist"]) for r in sub if int(r["step_k"]) == step]
        z_vals = [_to_float(r["min_abs_preact"]) for r in sub if int(r["step_k"]) == step]
        mean_l1.append(float(np.mean(l1_vals)) if l1_vals else 0.0)
        mean_zmin.append(float(np.mean(z_vals)) if z_vals else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharex=True)
    axes[0].plot(steps, mean_l1, label="BF16 LayerNorm topology A", lw=1.6)
    axes[0].set_ylabel("$D_1^{(k)}(s{=}1)$  (mean over 10 seeds)")
    axes[1].plot(steps, mean_zmin, label="BF16 LayerNorm topology A", lw=1.6)
    axes[1].set_ylabel("$z_{\\min}^{(k)}$  (mean over 10 seeds)")
    for ax in axes:
        _style_axes(ax, "step $k$", ax.get_ylabel())
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "figure3.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — H3: BF16 only at step 200, zoom gradient cosine
# ---------------------------------------------------------------------------

def figure4(rows: list[dict]) -> None:
    sub = [r for r in rows if r["step_k"] == "200" and r["precision"] == "bfloat16"]
    norms_present = sorted({r["norm"] for r in sub})

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    x = np.arange(3)  # topologies A, B, C
    width = 0.25
    offsets = {n: -width + i * width for i, n in enumerate(norms_present)}

    # Panel 1: D_1^(200)
    for norm in norms_present:
        means = []
        for topo in ("A", "B", "C"):
            match = [r for r in sub if r["topology"] == topo and r["norm"] == norm]
            val = _to_float(match[0]["l1_median"]) if match else np.nan
            means.append(val)
        ax = axes[0]
        ax.bar(x + offsets[norm], means, width, label=norm)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(("A", "B", "C"))
    axes[0].set_ylabel("$D_1^{(200)}$ median (BF16)")
    _style_axes(axes[0], "topology", axes[0].get_ylabel())

    # Panel 2: gradient cosine, zoomed to [0.97, 1.002]
    for norm in norms_present:
        means = []
        for topo in ("A", "B", "C"):
            match = [r for r in sub if r["topology"] == topo and r["norm"] == norm]
            val = _to_float(match[0]["grad_cosine_median"]) if match else np.nan
            means.append(val)
        ax = axes[1]
        ax.bar(x + offsets[norm], means, width, label=norm)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(("A", "B", "C"))
    axes[1].set_ylabel("grad cosine median (BF16)")
    axes[1].set_ylim(0.97, 1.002)
    _style_axes(axes[1], "topology", axes[1].get_ylabel())

    axes[0].legend(title="norm", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — H4: TV_l and test accuracy, ReLU slopes vs GELU
# ---------------------------------------------------------------------------

def figure5(rows: list[dict]) -> None:
    run_rows = [r for r in rows if r["record_type"] == "run"]
    for r in run_rows:
        r["_acc"] = _to_float(r["final_test_acc_pct"])
        r["_tv"] = _to_float(r["loss_total_variation"])

    act_order = ["relu_0.0", "relu_0.5", "relu_1.0", "gelu"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    x = np.arange(len(act_order))
    width = 0.35
    offsets = {"float32": -width / 2, "bfloat16": width / 2}

    for i, ax in enumerate(axes):
        metric = "_tv" if i == 0 else "_acc"
        ylabel = r"$\mathrm{TV}_\ell$" if i == 0 else "test accuracy (%)"
        for prec in ("float32", "bfloat16"):
            means = []
            for act in act_order:
                vals = [r[metric] for r in run_rows
                        if r["precision"] == prec and r["activation"] == act and r[metric] is not None]
                means.append(float(np.mean(vals)) if vals else np.nan)
            ax.bar(x + offsets[prec], means, width, label=PREC_LABEL[prec])
        ax.set_xticks(x)
        ax.set_xticklabels([ACT_LABEL[a] for a in act_order], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        _style_axes(ax, "activation", ylabel)
    axes[0].legend(title="precision", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6: build tables and figures from CSV logs."
    )
    parser.add_argument("--results", default="results", help="results directory")
    parser.add_argument("--tables", default="paper/tables", help="tables output directory")
    parser.add_argument("--figures", default="figures", help="figures output directory")
    args = parser.parse_args()

    global RESULTS, TABLES, FIGURES
    RESULTS = Path(args.results)
    TABLES = Path(args.tables)
    FIGURES = Path(args.figures)

    # Load + validate all CSVs (gate: fail fast if a required column is missing).
    df0 = _load(CSV_EXP0)
    df1 = _load(CSV_EXP1)
    df2step = _load(CSV_EXP2_STEP)
    df2sum = _load(CSV_EXP2_SUMMARY)
    df3 = _load(CSV_EXP3)

    # Tables.
    _write_table(TABLES / "table1.tex", "Experiment 0 probe", build_table1(df0))
    _write_table(TABLES / "table2.tex", "Experiment 1 proportions", build_table2(df1))
    _write_table(TABLES / "table3.tex", "Experiment 2 summary", build_table3(df2sum))
    _write_table(TABLES / "table4.tex", "Experiment 3 accuracy/TV/Wilcoxon", build_table4(df3))

    # Figures.
    figure_bar_chart(df1)
    print("[figure] wrote figures/figure1.pdf  (H1, H2)")
    figure2(df1)
    print("[figure] wrote figures/figure2.pdf  (H2)")
    figure3(df2step)
    print("[figure] wrote figures/figure3.pdf  (H3)")
    figure4(df2sum)
    print("[figure] wrote figures/figure4.pdf  (H3)")
    figure5(df3)
    print("[figure] wrote figures/figure5.pdf  (H4)")

    print("Stage 6 complete: tables in paper/tables/, figures in figures/.")


if __name__ == "__main__":
    main()

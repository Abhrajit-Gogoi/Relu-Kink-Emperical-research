# Finite-Precision Proxy Derivatives at the ReLU Kink: BFloat16 Arithmetic, LayerNorm versus RMSNorm, and Residual Topology

**Authors:**  Abhrajit Gogoi, Atharva Chavan
**Document:** Independent Research Manuscript (`paper/main.pdf`, November 2026)

Empirical investigation of how the proxy derivative $s = \mathrm{ReLU}'(0) \in \{0, 0.5, 1\}$ interacts with IEEE-754 floating-point precision ($\mathrm{FP64}$, $\mathrm{FP32}$, $\mathrm{FP16}$, $\mathrm{BF16}$), feature normalization ($\mathrm{LayerNorm}$ versus $\mathrm{RMSNorm}$), additive bias, and residual topology (plain MLP, on-highway residual, and inside-branch Pre-Norm residual FFN) on MNIST under vanilla SGD ($\eta = 0.01$, batch size $128$).

Over the real numbers $\mathbb{R}$, the bifurcation zone—parameters at which backpropagation depends on $s = \mathrm{ReLU}'(0)$—has Lebesgue measure zero ([Bolte & Pauwels, 2021](https://arxiv.org/abs/1909.10300); [Bertoin et al., 2021](https://arxiv.org/abs/2106.12915)). On discrete IEEE-754 grids, rounding and subtractive cancellation create exact zeros ($z = 0.0$), giving the bifurcation zone positive discrete volume. This repository contains the complete deterministic code, CSV logs, vector figures, and LaTeX source for our manuscript.

## Summary of Hypotheses and Findings

| ID | Hypothesis | Outcome | Key Quantitative Finding |
| :--- | :--- | :--- | :--- |
| **H1** | Exact-zero collisions and bifurcation volume rise as the mantissa narrows; $\mathrm{BF16}$ ($p = 7, e = 8$) exceeds $\mathrm{FP32}$ and $\mathrm{FP16}$ despite sharing $\mathrm{FP32}$'s exponent range. | **Supported** | On `NoNorm` (`bias=on`, $s = 1$ vs. $s = 0$): $\mathrm{BF16}$ ($\hat{p}_0 = 2.99 \times 10^{-5}$, $\hat{B} = 0.9436$) $>$ $\mathrm{FP16}$ ($3.80 \times 10^{-6}$, $\hat{B} = 0.3133$) $\gg$ $\mathrm{FP32}$ ($2.54 \times 10^{-9}$, $\hat{B} = 0.00025$) $>$ $\mathrm{FP64}$ ($0$). |
| **H2** | $\mathrm{LayerNorm}$ raises the exact-zero rate relative to $\mathrm{RMSNorm}$ via mean subtraction $(x_i - \mu)$; both norms buffer test accuracy versus unnormalized networks. | **Supported** | In $\mathrm{BF16}$, $\mathrm{LayerNorm}$ yields $\hat{p}_0 = 1.12 \times 10^{-4}$ (`bias=on`) and $1.13 \times 10^{-4}$ (`bias=off`), whereas $\mathrm{RMSNorm}$ drops from $2.99 \times 10^{-5}$ (`bias=on`) to $1.83 \times 10^{-8}$ (`bias=off`), confirming centering drives exact-zero collisions. |
| **H3** | Twin SGD runs stay bitwise-equal until the first exact zero; an inside-branch Pre-Norm residual FFN (Topology C) resists bifurcation longer than an on-highway residual $\mathrm{ReLU}$ (Topology B). | **Supported** | All $s = 0$ determinism twins maintain $D_1 = 0$ at every step. In $\mathrm{BF16}$, Topology B splits immediately (median $k^\star = 1$, $D_1^{(200)} = 41.97$–$53.55$), while Topology C's identity skip resists bifurcation (median $k^\star = 201$, $D_1^{(200)} = 0$). |
| **H4** | Smooth $\mathrm{GELU}$ ($\mathrm{GELU}'(0) = 0.5$) yields lower step-to-step loss total variation ($\mathrm{TV}_\ell$) than $\mathrm{ReLU}$ at matching slope $s = 0.5$. | **Directionally supported (underpowered)** | $\mathrm{GELU}$ achieves strictly lower $\mathrm{TV}_\ell$ and strictly higher final test accuracy across all 6 precision $\times$ norm cells, but with $R = 5$ paired seeds the minimum possible two-sided Wilcoxon $p$-value is $2 \cdot (1/2)^5 = 0.0625$, above $\alpha = 0.05$. |

**Baseline Reproduction Note:** Our Stage-2 NeurIPS 2021 $\mathrm{FP32}$ baseline check measured $\hat{V}_{0,1} = 0.06$ ($12/200$ initializations) versus $0.025$ ($5/200$) in Experiment 1 (`FP32 NoNorm bias-on`), below [Bertoin et al. (2021)](https://arxiv.org/abs/2106.12915)'s published $\sim 35$–$45\%$ proportion. This discrepancy is documented transparently in Section 5.5 of the manuscript rather than retuned away.

## Experiments Implemented

1. **Experiment 0 — Operator Audit (`scripts/run_exp0_verify.py`, `tests/test_activations.py`):** Verifies `ProxyReLU` on $x = [-2, -10^{-7}, -0.0, +0.0, +10^{-7}, +2]$ across $\mathrm{FP64}, \mathrm{FP32}, \mathrm{FP16}, \mathrm{BF16}$ for $s \in \{0, 0.25, 0.5, 0.75, 1\}$, audits algebraic emulation $f_s(x)$ on $10^6$ non-zero draws, and runs a 20-step $s = 0$ determinism twin check.
2. **Experiment 1 — Static Monte Carlo Bifurcation Volume (`scripts/run_exp1_monte_carlo.py`):** Evaluates a 3-hidden-layer MLP ($784 \times 256 \times 256 \times 256 \times 10$, $P = 335{,}114$ with biases) across 4 precisions $\times$ 4 normalizations (`NoNorm`, `BatchNorm1d`, `LayerNorm`, `RMSNorm`) $\times$ 2 bias settings $\times$ 2 slopes ($s \in \{0.5, 1.0\}$ vs. $s = 0$) over $M = 200$ initializations and $Q = 100$ mini-batches of $128$ (64 rows).
3. **Experiment 2 — Coupled Trajectory Bifurcation & Residual Topology (`scripts/run_exp2_trajectory.py`):** Runs $R = 10$ seeds $\times$ 200 shared mini-batches of coupled SGD ($s = 0$ twin A, $s = 0$ twin B, $s = 0.5$, $s = 1$) across 27 cells: Topologies A (plain MLP), B (on-highway residual), and C (inside-branch Pre-Norm residual FFN) $\times$ $\{\mathrm{FP64}, \mathrm{FP32}, \mathrm{BF16}\}$ $\times$ $\{\mathrm{NoNorm}, \mathrm{LayerNorm}, \mathrm{RMSNorm}\}$.
4. **Experiment 3 — Multi-Epoch Training Stability & $\mathrm{GELU}$ Control (`scripts/run_exp3_training.py`):** Trains Topology C for 10 epochs on MNIST across $\{\mathrm{FP32}, \mathrm{BF16}\}$ $\times$ $\{\mathrm{NoNorm}, \mathrm{LayerNorm}, \mathrm{RMSNorm}\}$ $\times$ $\{\mathrm{ReLU}\ (s = 0, 0.5, 1), \mathrm{GELU}\}$ over $R = 5$ paired seeds.

## Repository Layout

```text
src/            determinism, ProxyReLU, ExplicitLayerNorm/RMSNorm/BatchNorm1d, models, metrics
tests/          Experiment 0 unit tests (test_activations.py)
scripts/        experiment runners (run_exp0..3) and table/figure builder (build_tables_and_figures.py)
results/        tracked CSV logs from all experiments
figures/        generated vector PDF figures (Figures 1–5)
paper/          LaTeX manuscript (main.tex, main.bbl, main.pdf)
docs/protocol.md
references.bib
```

`ProxyReLU` is implemented as a custom `torch.autograd.Function`: the forward pass computes $\max(0, x)$, and the backward pass multiplies the upstream gradient by $s$ only where the saved pre-activation compares equal to zero (`x == 0`) in its native dtype.

## Environment & Reproduction

Python 3.10+. Experiments 0, 1, and 3 were executed on an RTX 4060 laptop GPU (`torch==2.14.1+cu130`, `torchvision==0.29.1+cu130`), Experiment 2 on a Google Colab T4 GPU, and the Stage-2 baseline reproduction check on single-threaded CPU (`torch==2.14.1+cu130`).

```bash
pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

For CPU-only machines:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Running the Pipeline

MNIST downloads automatically into `data/` (gitignored) on the first run. To reproduce all experiments, tables, figures, and the manuscript:

```bash
# 1. Experiment 0: operator audit & unit tests
pytest tests/test_activations.py
python scripts/run_exp0_verify.py

# 2. Experiment 1: static Monte Carlo bifurcation volume
python scripts/run_exp1_monte_carlo.py --mode reproduction
python scripts/run_exp1_monte_carlo.py --mode full

# 3. Experiment 2: 200-step coupled trajectories (Topologies A, B, C)
python scripts/run_exp2_trajectory.py --mode stage4

# 4. Experiment 3: 10-epoch training stability & GELU control (Topology C)
python scripts/run_exp3_training.py

# 5. Regenerate all figures and LaTeX tables directly from results/*.csv
python scripts/build_tables_and_figures.py

# 6. Compile the manuscript
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Reproducibility Guarantees

- Exact zeros are counted with `(z == 0)` in the tensor's native dtype (`float64`, `float32`, `float16`, or `bfloat16`).
- Initial weights are drawn in `float32` using Kaiming uniform initialization (`a=0, mode="fan_in", nonlinearity="relu"`) and then cast to the target dtype so every precision of a seed shares the same underlying draw.
- Every proxy slope reloads identical initial parameters and normalization buffers.
- `ExplicitLayerNorm`, `ExplicitRMSNorm`, and `ExplicitBatchNorm1d` perform reductions in the input dtype (`keep_native_dtype=True`, $\varepsilon = 10^{-5}$).
- Gradient equality is tested via `torch.equal` on native tensors; $\ell_1$, $\ell_2$, and cosine metrics are promoted to `float64`.
- All manuscript numbers, tables, and figures are generated directly from `results/*.csv`.

## Citation & References

All primary references cited in the manuscript are cataloged in [`references.bib`](references.bib) (including [Bertoin et al., 2021](https://arxiv.org/abs/2106.12915); [Bolte & Pauwels, 2021](https://arxiv.org/abs/1909.10300); [Lee et al., 2020](https://arxiv.org/abs/2006.06903); [Lee, Park, & Aiken, 2023](https://arxiv.org/abs/2301.13370); [Park, Chun, & Lee, 2024](https://openreview.net/forum?id=8vKknbgXxf); [Hückelheim et al., 2023](https://arxiv.org/abs/2305.07546); [Yang et al., 2023](https://arxiv.org/abs/2302.04351); [Martorella et al., 2023](https://zenodo.org/records/8173692); [Ba et al., 2016](https://arxiv.org/abs/1607.06450); [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467); [Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415); and [Salmani et al., 2024](https://arxiv.org/abs/2410.10553)). Third-party papers are not vendored in this repository.

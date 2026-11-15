# Finite-precision proxy derivatives at the ReLU kink

Empirical study of how the proxy slope \(s = \mathrm{ReLU}'(0)\) interacts with IEEE-754 precision, LayerNorm versus RMSNorm, and residual topology in small feed-forward and transformer networks.

Over the reals, the parameters at which backpropagation depends on \(s\) form a measure-zero set. On FP64, FP32, FP16, and BF16 that set has positive discrete volume, because rounding creates exact zeros. This repository will measure that volume, the step at which coupled SGD trajectories split, and whether a smooth GELU control (derivative 0.5 at the origin) is more stable than ReLU with \(s = 0.5\).

The methods are specified in [`docs/protocol.md`](docs/protocol.md). Citations are in [`references.bib`](references.bib). Experiments have not been run yet. Tables and the manuscript will be generated from the CSV files this protocol names, once those files exist.

## Hypotheses

| ID | Claim |
| :--- | :--- |
| H1 | Exact-zero collisions and bifurcation volume increase as the mantissa narrows. BF16, with a 7-bit mantissa and an 8-bit exponent, exceeds both FP32 and FP16. |
| H2 | LayerNorm raises the exact-zero rate relative to RMSNorm by centering pre-activations at 0. Both norms buffer test accuracy relative to an unnormalized network. |
| H3 | Twin SGD runs stay identical until the first exact zero. A Pre-Norm branch, whose skip path avoids the kink, diverges more slowly than a ReLU placed on the residual highway. |
| H4 | GELU yields lower step-to-step loss variation than ReLU at the matching slope \(s = 0.5\). |

## Layout

```text
src/            determinism, ProxyReLU, explicit norms, models, metrics
tests/          Experiment 0
scripts/        one runner per experiment, plus the table and figure builder
results/        CSV logs and run metadata
figures/        vector PDFs
paper/          LaTeX manuscript
docs/protocol.md
references.bib
```

`ProxyReLU` is a custom `torch.autograd.Function`: the forward pass is \(\max(0, x)\), and the backward pass multiplies the upstream gradient by \(s\) only when the saved pre-activation compares equal to zero in its own dtype.

## Setup

Python 3.10 or newer. This laptop has an RTX 4060. Later stages use it. The installed build is `torch==2.14.1+cu130` and `torchvision==0.29.1+cu130`:

```bash
pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

The CPU wheel is only for a machine with no GPU. Stage 2 was run on `torch 2.15.0+cpu`.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

MNIST is downloaded into `data/` on the first run (that directory is gitignored). Training is vanilla SGD with learning rate 0.01 and batch size 128. The dataset of record is MNIST. Fashion-MNIST is the second dataset in the 20-epoch experiment, as specified in the protocol.

## Reproducibility rules that the code has to keep

- Count exact zeros with `(z == 0)` in the native dtype.
- Draw initial weights in FP32 and cast, so every precision of a seed shares the same draw.
- Reload the same parameters and norm buffers before each proxy slope.
- Compute LayerNorm, RMSNorm, and BatchNorm reductions in the input dtype (`keep_native_dtype=True`).
- Treat two tensors as different when `torch.equal` is false, and promote them to FP64 before reporting an L1, L2, or cosine.
- Quote manuscript numbers only from files in `results/`.

## Status

The protocol, bibliography, and directory skeleton are in place. Stage 1 (the operator audit) is the next implementation step. Third-party papers are not vendored in this repository; cite them from `references.bib`.

No code license has been chosen yet.

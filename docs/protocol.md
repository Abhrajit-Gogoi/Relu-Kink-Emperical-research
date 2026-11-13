# Study protocol

Methods for the experiments in this repository. Runners, CSV names, and stage gates below are the contract for the code. A number in the manuscript has to be a number in `results/`.

## Question

How does \(s = \mathrm{ReLU}'(0) \in \{0, 0.5, 1\}\) interact with precision in \(\{\mathrm{FP64}, \mathrm{FP32}, \mathrm{FP16}, \mathrm{BF16}\}\) and with LayerNorm versus RMSNorm, in (a) the fraction of weights and minibatches for which one backpropagation step depends on \(s\), and (b) the divergence of coupled SGD trajectories on a feed-forward net, a residual net, and a one-block micro-transformer?

## Shared measurement rules

1. Exact zeros are `(z == 0)` in `z.dtype`. The count is cast to a Python integer after the comparison.
2. Two native tensors differ when `torch.equal` is false. L1, L2, cosine, and relative L2 are computed after promotion to FP64.
3. Initial weights are drawn with `kaiming_uniform_(a=0, mode="fan_in", nonlinearity="relu")` in FP32, then the state dict is cast to the run dtype. Enabled biases are \(\mathcal{U}(-1/\sqrt{\mathrm{fan\_in}}, 1/\sqrt{\mathrm{fan\_in}})\).
4. Every proxy reloads the same parameters and the same normalization buffers.
5. Explicit LayerNorm, RMSNorm, and BatchNorm reduce in the input dtype. `eps = 1e-5`. Run metadata stores the value of that epsilon after rounding to the run dtype.
6. Pixels come from torchvision MNIST in \([0, 1]\), with no augmentation and no further standardization, then cast to the run dtype. Batch indices are saved.
7. Loss is cross-entropy. Optimizer is SGD with learning rate 0.01, momentum 0, weight decay 0, batch size 128.
8. `with_bias_and_shift=True` enables linear biases and the LayerNorm or BatchNorm shift, and leaves scales learnable. `False` removes linear biases and the shift. RMSNorm has no shift; under `False` its learnable scale is removed as well.
9. Python, NumPy, and PyTorch seeds are fixed, and `torch.use_deterministic_algorithms(True)` is requested. A twin pair of \(s = 0\) models is the empirical check: their L1 distance stays 0.
10. Each runner writes `results/<script>_meta.json` with the commit, torch version, device, dtypes, seeds, rounded epsilon, and the batch-index file.

Relative gradient discrepancy, on impacted pairs only:

\[
\delta_{\mathrm{rel}}(g_s, g_0) = \frac{\|g_s - g_0\|_2}{\|g_0\|_2 + 10^{-30}}.
\]

## Checkpoints

The work proceeds in seven checkpoints. Later checkpoints wait until the current gate holds.

| Checkpoint | What is produced | Gate |
| :--- | :--- | :--- |
| 1. Experiment 0 | `results/exp0_verification.csv` | Probe gradients match in FP64, FP32, FP16, and BF16. The \(s = 0\) twin has L1 distance 0. |
| 2. Bertoin et al. (2021) baseline | `results/reproduction_mc.csv`, `results/reproduction_traj.csv` | FP64 bifurcation weight proportion is 0. FP32 falls near 35–45% and the trajectory jumps at an exact zero. |
| 3. Experiment 1 | `results/exp1_monte_carlo.csv` | All 32 precision \(\times\) norm \(\times\) bias cells, with both confidence intervals. |
| 4. Experiment 2 | `results/exp2_trajectories.csv` | \(k^\star\) recorded for every seed, twin distance 0, no NaNs. |
| 5. Experiment 3 | `results/exp3_training.csv` | Complete 20-epoch MNIST logs and Wilcoxon tests. Design freezes at the end of this checkpoint. |
| 6. Tables and figures | `figures/*.pdf`, `paper/tables/` | Each of H1–H4 maps to a table row and a figure panel. |
| 7. Manuscript | `paper/main.pdf` | Every quoted number matches a CSV. Citations resolve. |

Auxiliary factors (CIFAR-10, Adam, learning-rate searches) are dropped if they take more than four hours of debugging. If the micro-transformer attention block is still not training correctly late in checkpoint 5, the transformer arm is replaced by topology C below, and the manuscript says the attention contrast was not run.

## Experiment 0 — operator audit

`ProxyReLU` has forward \(\max(0, x)\) and backward slope \(s\) at exact zero, 1 on the positive side, and 0 on the negative side. The algebraic emulation used as a comparison is

\[
f_s(x) = \mathrm{ReLU}(x) + s\big(\mathrm{ReLU}(-x) - \mathrm{ReLU}(x) + x\big).
\]

Probe \(x = [-2,\ -10^{-7},\ -0,\ +0,\ +10^{-7},\ +2]\) in all four dtypes, at \(s \in \{0, 0.25, 0.5, 0.75, 1\}\). The expected gradient is \([0, 0, s, s, 1, 1]\).

Then draw \(10^6\) nonzero uniforms on \([-5, 5]\) in FP32 and BF16, at \(s \in \{0.1, 0.25, 0.3, 0.5, 0.75, 1\}\). Log `max_abs_forward_err_nonzero` and `frac_nonzero_grad_mismatch`, the fraction of nonzero inputs whose emulation backward disagrees with the analytical proxy derivative.

## Experiment 1 — static Monte Carlo

MLP widths \(784 \times 256 \times 256 \times 256 \times 10\) (335,114 parameters with biases).

- Precisions: FP64, FP32, FP16, BF16.
- Norms: none, BatchNorm1d, LayerNorm, RMSNorm.
- Bias and shift: on or off, as defined above.
- Slopes \(s \in \{0.5, 1\}\) differenced against \(s = 0\).
- \(M = 200\) initializations, \(Q = 100\) minibatches of size 128.

Checkpoint 2 runs the `NoNorm`, bias-on subset in FP64, FP32, and FP16 before the rest of the matrix.

Weight proportion and its Hoeffding radius:

\[
\hat{V}_{0,s} = \frac{1}{M}\sum_{m=1}^{M}\mathbf{1}\{\exists q:\ g_s \neq g_0\},
\qquad
\varepsilon_{\mathrm{Hoeff}} = \sqrt{\frac{\ln(40)}{2M}}.
\]

Batch proportion and its McDiarmid radius:

\[
\hat{B}_{0,s} = \frac{1}{MQ}\sum_{m,q}\mathbf{1}\{g_s \neq g_0\},
\qquad
\varepsilon_{\mathrm{McD}} = \sqrt{\tfrac12\big(\tfrac1M + \tfrac1Q\big)\ln 40}.
\]

Also log the exact-zero rate \(\hat{p}_0\), the quartiles of \(\delta_{\mathrm{rel}}\) and of the gradient-norm ratio on impacted pairs, layerwise gradient cosines, and a histogram of \(\log_{10}|z|\) on nonzero pre-activations with edges at \(-12, -8, -5, -3, -1\). Zeros occupy their own bin.

The header of `results/exp1_monte_carlo.csv` is: cell identifiers (`precision`, `norm`, `with_bias_and_shift`, `s`, `keep_native_dtype`), `exact_zero_rate`, seven histogram counts, both proportions, both radii, three quartiles of relative L2, three quartiles of the norm ratio, four layer cosines, and `n_impacted_pairs`.

## Experiment 2 — coupled trajectories

Three topologies, width 256, three blocks. \(R = 10\) seeds, \(K = 200\) shared minibatches, precisions FP64, FP32, and BF16, norms none, LayerNorm, and RMSNorm.

- **A.** \(a_l = \mathrm{ReLU}_s(\mathrm{Norm}_l(W_l a_{l-1} + b_l))\).
- **B, kink on the highway.** \(a_l = \mathrm{ReLU}_s\big(a_{l-1} + W_{l,2}\mathrm{ReLU}_s(\mathrm{Norm}_l(W_{l,1}a_{l-1}+b_{l,1}))\big)\).
- **C, kink inside the branch.** \(a_l = a_{l-1} + W_{l,2}\mathrm{ReLU}_s(W_{l,1}\mathrm{Norm}_l(a_{l-1})+b_{l,1}) + b_{l,2}\).

Each seed runs two \(s = 0\) copies, one \(s = 0.5\) copy, and one \(s = 1\) copy. The first bifurcation step is

\[
k^\star(s) = \inf\{k \ge 1 : \|\theta_{k,s} - \theta_{k,0}\|_1 > 0\},
\]

stored as \(K+1\) when the copies never split. Each step logs minimum absolute pre-activation, exact-zero count, L1 and L2 parameter distance, weight cosine, gradient cosine, and the distance between the two \(s = 0\) twins.

## Experiment 3 — twenty epochs

- **Residual MLP.** Topology C, three blocks, width 256, input 784, output 10. Norms: none, LayerNorm, RMSNorm.
- **Micro-transformer.** 49 patches of \(4\times 4\), embedding width 64, one Pre-Norm block, 4 heads of dimension 16, feed-forward width 256, mean-pool, final norm, linear classifier to 10 classes. Norms: LayerNorm and RMSNorm. An optional second block is allowed only after the one-block matrix is complete.
- Precisions: FP32 and BF16.
- Activations: ReLU at \(s \in \{0, 0.5, 1\}\), and `nn.GELU()`.
- Five paired seeds, 20 epochs.

MNIST is required. Fashion-MNIST repeats the same matrix after MNIST is complete. If finishing Fashion-MNIST would cut MNIST short, MNIST is the reported dataset and Fashion-MNIST is marked incomplete.

Per epoch, log train loss, train accuracy, test loss, test accuracy, and the epoch exact-zero rate. Per run, log loss total variation

\[
\mathrm{TV}_\ell = \frac{1}{K-1}\sum_{k=2}^{K}|\ell_k - \ell_{k-1}|
\]

and the mean cosine between successive gradients. On the transformer, also log the ratio of relative gradient change in the attention projections to relative gradient change in the feed-forward weights. Compare each of \(s = 0.5\), \(s = 1\), and GELU with \(s = 0\) by a two-sided paired Wilcoxon signed-rank test on test accuracy and on \(\mathrm{TV}_\ell\).

## Figures and tables

| Artifact | Content | Hypothesis |
| :--- | :--- | :--- |
| Table 1 | Experiment 0 probe, forward error, emulation mismatch | operator |
| Table 2 | \(\hat{V}\), \(\hat{B}\), radii, \(\hat{p}_0\), \(\delta_{\mathrm{rel}}\) quartiles | H1, H2 |
| Figure 1 | L1 distance and minimum \(\lvert z\rvert\) versus step | H3 |
| Figure 2 | Bifurcation proportions by precision and norm | H1, H2 |
| Figure 3 | Near-zero histogram, LayerNorm versus RMSNorm | H2 |
| Figure 4 | Gradient cosine and final L1 distance for topologies A–C | H3 |
| Figure 5 | Loss total variation and test accuracy, ReLU versus GELU | H4 |
| Table 3 | \(k^\star\) and final L1 distance, median and interquartile range | H3 |
| Table 4 | Accuracy, total variation, Wilcoxon \(p\) against \(s = 0\) | H2, H4 |

## What a contradicted hypothesis looks like

H1 is read from Experiment 1 on the unnormalized, bias-on MLP, then checked under the other norms. H2 is read from the LayerNorm versus RMSNorm histogram and from Experiment 3 accuracy against `NoNorm`. H3 is read from alignment of \(k^\star\) with an exact zero, and from topology B versus topology C. H4 is read from \(\mathrm{TV}_\ell\) of GELU versus ReLU at \(s = 0.5\). A CSV that points the other way is reported that way. Checkpoints are not retuned to reverse it.

# Stage 2 gate

The sample is the Stage 2 protocol: MLP 784×256×256×256×10, NoNorm, biases on, Kaiming uniform with a=0 drawn in FP32 and then cast, MNIST in [0, 1], batch 128, vanilla SGD with learning rate 0.01. Monte Carlo uses M=200 and Q=100. Trajectories use seeds 1 through 10 and K=200. Both arms share `results/reproduction_batch_indices.npy` (batch seed 0, 200 batches; the Monte Carlo arm uses the first 100). Device is CPU. PyTorch is 2.15.0+cpu. Python is 3.14.3. No gradient in the Monte Carlo arm was nonfinite.

**Outcome.** The gate does not pass. Stage 3 stays closed. Depth, width, learning rate, M, Q, K, and R were left as specified.

## Monte Carlo

Counts are in `results/reproduction_mc.csv`. Each cell sees 1,966,080,000 pre-activations. The Hoeffding radius at M=200 is 0.0960. The McDiarmid radius at M=200 and Q=100 is 0.1663.

| Precision | s | Exact-zero rate | Weight proportion | Batch proportion | Impacted pairs | Relative L2 (q1, q2, q3) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP64 | 0.5 and 1 | 0 | 0 | 0 | 0 | blank |
| FP32 | 0.5 | 6.104e-9 | 0.06 | 0.0006 | 12 | 0.000733, 0.001104, 0.001745 |
| FP32 | 1 | 6.104e-9 | 0.06 | 0.0006 | 12 | 0.001466, 0.002209, 0.003491 |
| FP16 | 0.5 | 6.612e-8 | 0.47 | 0.00645 | 129 | 0.000457, 0.001062, 0.002239 |
| FP16 | 1 | 6.612e-8 | 0.47 | 0.00645 | 129 | 0.000915, 0.002110, 0.004468 |

FP64 has zero exact collisions and weight proportion 0. That item of the gate holds.

FP32 weight proportion is 0.06 (12 of 200 initializations, 12 of 20,000 pairs, 12 exact zeros). The Hoeffding interval is about [0, 0.156]. The published comparison band is about 35–45 percent (Bertoin, Bolte, Gerchinovitz, and Pauwels, NeurIPS 2021, Table 1, 40 percent in 32 bits; the corrigendum table prints 43 percent and says the proportions are unchanged). 0.06 lies outside that band. This is the item that blocks Stage 3.

The 12 FP32 disagreements are the same 12 exact zeros. The sourced comparison with Bertoin et al. is in the section below. M, Q, width, and the estimator were left as specified.

FP16 weight proportion is 0.47 (94 of 200 initializations). The histogram records 130 exact zeros and 129 impacted pairs, so one exact zero left the native gradient bitwise equal. FP16 is part of the Stage 2 sample. The 35–45 percent check is the FP32 check.

## Trajectories

`results/reproduction_traj.csv` has 12,000 step rows. The maximum s=0 twin L1 distance is 0 in FP64, FP32, and FP16, at every seed and every step.

FP64 never records an exact zero. Parameter L1 between s=0.5 and s=0, and between s=1 and s=0, stays 0 for all 200 steps. Median k* is 201.

FP32 records one exact zero: seed 5, step 160, count 1, `min_abs_preact` 0. L1 at step 159 is 0. At step 160 it is 1.135e-4 for s=0.5 and 2.270e-4 for s=1. By step 200 that seed's L1 is about 0.149. The other nine seeds never hit an exact zero and stay at L1 0, so the median L1 at steps 50, 100, and 200 is 0 and the median k* is 201. The jump that does occur is on the exact-zero step.

FP16 first jumps, for both slopes, are seed 3 step 7, seed 2 step 35, seed 6 step 38, seed 7 step 50, seed 8 step 70, and seed 9 step 92. Each of those steps has `min_abs_preact` 0 and exact-zero count 1. Seeds 1, 4, 5, and 10 never leave L1 0. Median k* is 81. Median L1 at step 200 is about 3.66 for s=1.

Every first positive L1 in this sample sits on a step whose pre-activation minimum is an exact 0, and the twin distance on that step is 0.

## Why the published FP32 volume is not recovered

The local file is `literature/pdfs/bertoin2021_relu_prime_zero.pdf`, the 3 November 2023 erratum (arXiv:2106.12915v4). The erratum occupies the first pages and then reprints the original paper. Page numbers below are pages of that file.

### What the paper reports

The erratum (page 2) says the backward used in the original experiments assigned the proxy slope instead of multiplying it by the incoming gradient: `grad_input[input == 0] = 0.5` in place of `grad_input[input == 0] *= 0.5`. It says the proportions in its Table 1 and Figure 2 are unchanged up to non-significant variation, and that the relative L2 amplitudes fall by a factor of about 10^5.

Erratum Table 1 (page 2), columns 16 / 32 / 64 bits:

- Weight proportion: 100%, 43%, 0%.
- Impacted mini-batches: 0.005%, 0.0002%, 0%.
- Relative L2 quartiles: (1e-3, 2e-3, 3e-3), (1e-3, 2e-3, 3e-3), (0, 0, 0).

Original Table 1 (page 12), same architecture, 784×256×256×256×10:

- Weight proportion, with the printed Hoeffding margin "CI ± 5%": 100%, 40%, 0%.
- Impacted mini-batches, with the printed McDiarmid margin "CI ± 13%": 0.05%, 0.0002%, 0%.
- Relative L2 quartiles: (98, 117, 137), (19, 25, 47), (0, 0, 0).

The 32-bit weight figure is the 35–45 percent band in the stage file. The 32-bit batch figure is 0.0002% in both tables. The 16-bit batch figure moves from 0.05% to 0.005%.

The Monte Carlo procedure is on pages 12 and 22. M = 1000. Weights of each affine map are drawn independently and uniformly on [-α, α] with α = sqrt(6) / sqrt(m), where m is the fan-in. The text calls this "the default weight initialization scheme in PyTorch (Kaiming-Uniform)". Each draw is then scored on the whole MNIST training set. Appendix C writes both estimators on the sum of per-example losses. The Hoeffding margin is sqrt(ln(40) / (2M)). The McDiarmid margin is sqrt((1/2)(1/M + 1/Q) ln 40). Page 12 says these Monte Carlo experiments were run in PyTorch on GPU. The same page says the influence of mini-batch size is not significant. Figure 2 (page 13) varies training-set size, width, depth, and batch size. Its first three panels use batch size 256. The first panel is a 4-hidden-layer net, so it is not the Table 1 architecture. The plotted curves are not stored as numbers in the text layer of the PDF.

Figure 1 is a different experiment (page 11). The network is 784×2000×128×10, the mini-batch size is 128, α_k = 1, γ is drawn from [0.01, 0.012], the run is on CPU, and the L1 distance jumps at iteration 65 on an exact zero. That is the trajectory illustration. It is not Table 1.

The paper's Section 3.2 states the weight law and does not state a bias law. Appendix A.2 (page 20) includes biases in the parameter count, P = sum_h d_h (d_{h-1} + 1).

### What this run did

`initialize_kaiming_` calls `kaiming_uniform_(a=0, mode="fan_in", nonlinearity="relu")`. For fan-in 784 that bound equals sqrt(6/784) = 0.08748177652797065, the same value as the printed α. PyTorch's `nn.Linear` default is a different call, `kaiming_uniform_(a=sqrt(5))`, whose bound is 1/sqrt(fan_in) = 0.03571428571428571. The sentence "default" and the displayed α refer to two different draws. This run followed the displayed α, which is also the draw required by `RESEARCH_STAGES.md`. Biases are uniform on [-1/sqrt(fan_in), 1/sqrt(fan_in)], the `nn.Linear` bias default. The paper never states that interval.

`capture_grads` uses `F.cross_entropy`, whose default reduction is the mean. Appendix C differentiates the sum. The volume is `torch.equal` on every linear weight gradient and every linear bias gradient (`packs_equal`). The ReLU hook counts exact zeros. It does not decide whether two gradients differ. On the FP32 cell the hook count and the disagreement count are the same number, 12, so the weight proportion is the collision count.

A one-batch constructed check, the same zero-preactivation network as `test_identical_slope_grads_match_and_a_zero_preactivation_does_not`, compared s = 0 with s = 1 under both reductions. `torch.equal` was false for the mean and false for the sum. The mean reduction does not turn that kink into a bitwise match. The official `capture_grads` path on the same network also returned unequal packs. This check is not a new Monte Carlo sample.

The run meta records CPU, PyTorch 2.15.0+cpu, and deterministic algorithms with one thread. Page 12 puts the volume experiment on GPU. Page 11 puts the Figure 1 trajectory on CPU.

### Which differences can move 0.06 to about 0.40

Exposure accounts for the measured 0.06. The FP32 pair rate is 12/20,000 = 0.0006. At that rate, Q = 100 gives 1 - (1 - 0.0006)^100 = 0.0583, and the run produced 0.06, with one impacted batch on each of 12 initializations. The same rate over a full epoch of 469 batches of 128 gives 1 - (1 - 0.0006)^469 = 0.245. That is still outside 35–45 percent. Figure 2's first panel shows the weight proportion rising with the number of training images, on a deeper net and at batch 256, which is the same direction. The stage file fixes M = 200 and Q = 100, so the official cell was not enlarged.

The chain-rule bug does not account for the low volume. The erratum says the proportions survive the correction, and the change is the L2 amplitude. Our s = 1 quartiles are 0.001466, 0.002209, and 0.003491. Those sit on the erratum's (1e-3, 2e-3, 3e-3), and they sit far from the original (19, 25, 47). This run multiplies the incoming gradient by the proxy slope.

Table 1's two proportion lines do not generate each other. At M = 1000 the Hoeffding formula is 0.0429, which rounds to the printed ±5 percent. The McDiarmid formula equals 0.130 at Q = 122 and 0.133 at Q = 117, which is the printed ±13 percent if that margin is an absolute proportion. A full epoch at batch 128 is Q = 469 and the same formula equals 0.076. Batch 512 gives Q = 117. Table 1 does not name its batch size. Forty percent of M = 1000 is 400 initializations, so there are at least 400 impacted pairs. That requires a pair rate of at least 400/(1000·122) = 0.00328 at Q = 122, and at least 0.000855 at Q = 469. The printed 0.0002 percent is the fraction 2e-6. At Q = 122 that fraction produces about 0.24 impacted pairs in the whole Monte Carlo, and an expected weight proportion near 2e-4. Reading 0.0002 as a fraction rather than a percent still gives an expected weight proportion near 0.024 at Q = 122. Under Appendix C, line 2 of Table 1 does not produce line 1 at any Q that also produces the printed ±13 percent margin. Our own lines do produce each other: 12 pairs, 12 weights, pair rate 0.0006, predicted weight proportion 0.058, observed 0.06.

Three procedural differences remain, and none of them was used to edit the sample. The paper scores the whole training set on GPU; this cell scores 12,800 images on CPU. The paper does not state the bias draw; this cell uses the `nn.Linear` bias default. The paper's "default" sentence points at `nn.Linear`, while the displayed α is the ReLU Kaiming bound this cell uses. A full-epoch extrapolation of the measured CPU pair rate is still about 0.25, so those differences would have to change the pair rate itself before the published 40 percent would appear. The stage file does not allow that change.

## What blocks Stage 3

The FP32 bifurcation weight proportion is 0.06, and the 95 percent Hoeffding interval around it does not reach 35 percent. The section above is the cause: Q = 100 at the measured pair rate produces 0.06, a full epoch of that rate is about 0.25, and Table 1's own batch line does not generate its 40 percent weight line. The trajectory conditions in this sample hold: the twin stays at L1 0, and each observed split starts on an exact zero. Stage 3 does not start until that volume disagreement is accepted as the result of this protocol.

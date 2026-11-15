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

The 12 FP32 disagreements are the same 12 exact zeros. For s=1 the relative-L2 quartiles are 0.00147, 0.00221, and 0.00349, in line with the corrigendum's corrected Table 1 amplitudes, about (1e-3, 2e-3, 3e-3). The original Table 1 amplitudes, (19, 25, 47), came from a backward that assigned the proxy slope instead of multiplying it by the incoming gradient. This run uses the multiplication.

The published 40 percent used M=1000 and a pass over the whole MNIST training set. This stage fixes M=200 and Q=100 batches of 128, which is 12,800 images. The measured FP32 pair rate is 0.0006. At that rate the expected weight proportion for Q=100 is about 0.058, and the run produced 0.06. Extending the same pair rate to one epoch of 469 batches of 128 gives a weight proportion of about 0.25, which is still outside 35–45 percent. The stage file does not allow a larger M or Q in order to land inside the band.

FP16 weight proportion is 0.47 (94 of 200 initializations). The histogram records 130 exact zeros and 129 impacted pairs, so one exact zero left the native gradient bitwise equal. FP16 is part of the Stage 2 sample. The 35–45 percent check is the FP32 check.

## Trajectories

`results/reproduction_traj.csv` has 12,000 step rows. The maximum s=0 twin L1 distance is 0 in FP64, FP32, and FP16, at every seed and every step.

FP64 never records an exact zero. Parameter L1 between s=0.5 and s=0, and between s=1 and s=0, stays 0 for all 200 steps. Median k* is 201.

FP32 records one exact zero: seed 5, step 160, count 1, `min_abs_preact` 0. L1 at step 159 is 0. At step 160 it is 1.135e-4 for s=0.5 and 2.270e-4 for s=1. By step 200 that seed's L1 is about 0.149. The other nine seeds never hit an exact zero and stay at L1 0, so the median L1 at steps 50, 100, and 200 is 0 and the median k* is 201. The jump that does occur is on the exact-zero step.

FP16 first jumps, for both slopes, are seed 3 step 7, seed 2 step 35, seed 6 step 38, seed 7 step 50, seed 8 step 70, and seed 9 step 92. Each of those steps has `min_abs_preact` 0 and exact-zero count 1. Seeds 1, 4, 5, and 10 never leave L1 0. Median k* is 81. Median L1 at step 200 is about 3.66 for s=1.

Every first positive L1 in this sample sits on a step whose pre-activation minimum is an exact 0, and the twin distance on that step is 0.

## What blocks Stage 3

The FP32 bifurcation weight proportion is 0.06, and the 95 percent Hoeffding interval around it does not reach 35 percent. The trajectory conditions in this sample hold: the twin stays at L1 0, and each observed split starts on an exact zero. Stage 3 does not start until that volume disagreement is accepted as the result of this protocol.

"""Stage 2 model, metric, and reproduction-loop checks. They do not run the full matrix."""

import torch
import torch.nn.functional as F

from scripts.run_exp1_monte_carlo import reproduction_jobs, run_cell_counts, stage3_jobs
from scripts.run_exp2_trajectory import run_seed, summarize_trajectories
from src.hooks_and_metrics import (
    KinkCollisionTracker,
    assign_log_bins,
    hoeffding_ci_95,
    mcdiarmid_ci_95,
)
from src.models import (
    MLP_WIDTHS,
    ConfigurableMLP,
    MicroTransformer,
    ResidualFFNNet,
    build_mlp,
    initialize_kaiming_,
)
from src.norms import ExplicitLayerNorm, ExplicitRMSNorm, ExplicitBatchNorm1d


def test_mlp_parameter_count_matches_the_protocol():
    model = ConfigurableMLP(MLP_WIDTHS, norm="NoNorm", with_bias_and_shift=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 335_114


def test_removing_bias_drops_bias_parameters_only():
    full = sum(p.numel() for p in ConfigurableMLP(with_bias_and_shift=True).parameters())
    bare = sum(p.numel() for p in ConfigurableMLP(with_bias_and_shift=False).parameters())
    assert full - bare == 256 + 256 + 256 + 10


def test_layernorm_centers_and_rmsnorm_does_not():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    centered = ExplicitLayerNorm(4)(values)
    scaled = ExplicitRMSNorm(4)(values)
    assert torch.allclose(centered.mean(dim=-1), torch.zeros(1), atol=1e-6)
    assert float(scaled.detach().mean()) > 0.1
    assert torch.all(scaled > 0)


def test_batchnorm_uses_the_batch_axis_and_keeps_fp16():
    layer = ExplicitBatchNorm1d(3)
    batch = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    normalized = layer(batch)
    assert torch.allclose(normalized.mean(dim=0), torch.zeros(3), atol=1e-5)
    half = ExplicitLayerNorm(4, keep_native_dtype=True).half()
    assert half(torch.randn(6, 4, dtype=torch.float16)).dtype == torch.float16


def test_log_bins_are_right_closed():
    edges_input = torch.tensor([-12.0, -11.0, -8.0, -6.0, -4.0, -2.0, 1.0])
    bins = assign_log_bins(edges_input).tolist()
    assert bins == [0, 1, 1, 2, 3, 4, 5]


def test_tracker_separates_exact_zeros_from_the_log_bins():
    tracker = KinkCollisionTracker()
    tracker.observe(torch.tensor([0.0, 1e-13, 1e-10, 1e-6, 1e-4, 1e-2, 10.0], dtype=torch.float64))
    snap = tracker.snapshot()
    assert snap["exact"] == 1
    assert snap["bins"] == (1, 1, 1, 1, 1, 1)
    assert snap["zero_rate"] == 1 / 7


def test_confidence_intervals_match_the_formulas():
    assert abs(hoeffding_ci_95(200) - (1.3581 / (200 ** 0.5))) < 1e-4
    radius = mcdiarmid_ci_95(200, 100)
    assert radius > 0


def test_identical_slope_grads_match_and_a_zero_preactivation_does_not():
    model = ConfigurableMLP(widths=(4, 3, 2), s=0.0, norm="NoNorm")
    initialize_kaiming_(model)
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    from src.hooks_and_metrics import capture_grads, packs_equal

    first = capture_grads(model, inputs, targets, 0.0)
    second = capture_grads(model, inputs, targets, 0.0)
    assert packs_equal(first, second)

    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.bias.zero_()
        # A single logit depends on one hidden unit. Ones would cancel because
        # the class gradients of cross-entropy sum to zero.
        model.fc2.weight.zero_()
        model.fc2.weight[0, 0] = 1.0
        model.fc2.bias.zero_()
    blocked = capture_grads(model, inputs, targets, 0.0)
    passed = capture_grads(model, inputs, targets, 1.0)
    assert not packs_equal(blocked, passed)
    assert torch.count_nonzero(blocked.layers[0]) == 0
    assert torch.count_nonzero(passed.layers[0]) > 0


def test_fp16_backward_stays_in_fp16():
    model = build_mlp(seed=1, dtype=torch.float16, widths=(16, 8, 4))
    logits = model(torch.randn(8, 16, dtype=torch.float16))
    loss = F.cross_entropy(logits, torch.randint(0, 4, (8,)))
    loss.backward()
    assert model.fc1.weight.grad is not None
    assert model.fc1.weight.grad.dtype == torch.float16


def test_stage_cell_lists():
    assert len(reproduction_jobs()) == 3
    assert len(stage3_jobs()) == 32


def test_monte_carlo_cell_on_a_tiny_net():
    images = torch.rand(32, 8)
    labels = torch.randint(0, 3, (32,))
    index = torch.arange(16).view(2, 8)
    rows, info = run_cell_counts(
        {"precision": "float32", "norm": "NoNorm", "with_bias_and_shift": True, "keep_native_dtype": True},
        images,
        labels,
        index,
        m_inits=2,
        slopes=(0.5, 1.0),
        init_seed0=0,
        widths=(8, 4, 3),
    )
    assert len(rows) == 2
    assert {row["s"] for row in rows} == {"0.5", "1.0"}
    assert info["n_pairs"]["0.5"] == 4
    assert rows[0]["grad_cosine_l4"] == ""


def test_coupled_twin_stays_together_and_summary_groups_checkpoints():
    images = torch.rand(40, 8)
    labels = torch.randint(0, 3, (40,))
    index = torch.arange(16).view(2, 8)
    records = run_seed(1, "float32", images, labels, index, widths=(8, 4, 3))
    assert records
    assert max(float(row["twin_l1_dist"]) for row in records) == 0.0
    assert {row["s"] for row in records} == {0.5, 1.0}
    fake = []
    for seed in range(1, 5):
        for step in (50, 100):
            fake.append(
                {
                    "precision": "float32",
                    "norm": "NoNorm",
                    "topology": "A",
                    "s": 1.0,
                    "seed": seed,
                    "step_k": step,
                    "traj_l1_dist": float(seed),
                    "traj_l2_dist": float(seed) / 2,
                    "grad_cosine_sim_step": 1.0,
                    "first_bifurcation_step": 3,
                }
            )
    summary = summarize_trajectories(fake)
    assert [row["step_k"] for row in summary] == [50, 100]
    assert summary[0]["l1_median"] == 2.5
    assert summary[0]["kstar_median"] == 3.0


def test_residual_skip_and_transformer_shapes():
    block_net = ResidualFFNNet(topology="C", in_features=16, width=16, n_classes=4, depth=1, norm="NoNorm")
    for parameter in block_net.parameters():
        parameter.data.zero_()
    residual = torch.randn(2, 16)
    assert torch.equal(block_net.blocks[0](residual), residual)

    highway = ResidualFFNNet(topology="B", in_features=16, width=16, n_classes=4, depth=1, norm="NoNorm")
    for parameter in highway.parameters():
        parameter.data.zero_()
    assert torch.equal(highway.blocks[0](residual), torch.relu(residual))

    transformer = MicroTransformer(norm="LayerNorm", depth=1)
    assert transformer(torch.rand(2, 1, 28, 28)).shape == (2, 10)
    assert transformer(torch.rand(2, 784)).shape == (2, 10)

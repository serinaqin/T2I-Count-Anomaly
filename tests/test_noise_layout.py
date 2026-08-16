import torch
from src.noise_layout import grid_boxes, count_aware_latent


def test_grid_boxes_count_and_bounds():
    boxes = grid_boxes(4, 100, 100)
    assert len(boxes) == 4
    for (y0, y1, x0, x1) in boxes:
        assert 0 <= y0 < y1 <= 100 and 0 <= x0 < x1 <= 100


def test_gaussian_raises_center():
    base = torch.zeros(1, 4, 64, 64)
    out = count_aware_latent(base, 1, scheme="gaussian", omega=0.3)
    assert out.shape == base.shape
    assert out[0, 0, 32, 32] > 0.1                     # strong center bump
    assert out[0, 0, 0, 0] < 0.05                      # weak (much smaller) at corner
    assert out[0, 0, 32, 32] > 10 * out[0, 0, 0, 0]    # center >> corner


def test_fixed_and_uniform_scaled():
    base = torch.ones(1, 4, 64, 64)
    f = count_aware_latent(base, 1, scheme="fixed", fixed_value=0.0)
    (y0, y1, x0, x1) = grid_boxes(1, 64, 64)[0]
    assert torch.allclose(f[0, 0, y0:y1, x0:x1], torch.zeros(y1 - y0, x1 - x0))
    u = count_aware_latent(base, 1, scheme="uniform_scaled", gamma=0.1)
    assert torch.allclose(u[0, 0, y0:y1, x0:x1], 0.1 * torch.ones(y1 - y0, x1 - x0))


def test_noise_boost_amplifies_box():
    base = torch.ones(1, 4, 64, 64)
    out = count_aware_latent(base, 1, scheme="noise_boost", beta=1.6)
    (y0, y1, x0, x1) = grid_boxes(1, 64, 64)[0]
    assert torch.allclose(out[0, 0, y0:y1, x0:x1],
                          torch.full((y1 - y0, x1 - x0), 1.6))


def test_gaussian_noise_localized_and_artifact_free():
    base = torch.zeros(1, 4, 64, 64)
    out = count_aware_latent(base, 1, scheme="gaussian_noise", omega=0.3, noise_seed=0)
    assert out.shape == base.shape
    (y0, y1, x0, x1) = grid_boxes(1, 64, 64)[0]
    inbox = out[:, :, y0:y1, x0:x1].abs().mean()
    corner = out[:, :, :8, :8].abs().mean()
    assert inbox > corner                                   # energy in the box
    # artifact-free: no strong positive DC shared across channels at the center
    assert out[0, :, 32, 32].mean().abs() < out[:, :, y0:y1, x0:x1].abs().mean()


def test_gaussian_gray_channel_weighted():
    base = torch.zeros(1, 4, 64, 64)
    w = torch.tensor([1.0, -1.0, 0.0, 2.0])
    out = count_aware_latent(base, 1, scheme="gaussian_gray", omega=0.3,
                             channel_weights=w)
    assert torch.allclose(out[0, 2], base[0, 2])          # weight 0 -> unchanged
    c0, c3 = out[0, 0, 32, 32], out[0, 3, 32, 32]
    assert abs(c3 - 2 * c0) < 1e-4                          # ch3 = 2 * ch0 (weights)
    assert torch.allclose(out[0, 1, 32, 32], -c0, atol=1e-4)  # ch1 = -ch0

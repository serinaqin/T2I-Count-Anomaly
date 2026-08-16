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

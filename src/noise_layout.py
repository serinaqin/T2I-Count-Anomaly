import math
import torch


def grid_boxes(n, H, W, fill=0.6):
    """N non-overlapping boxes on a near-square grid; each fills `fill` of its
    cell. Returns [(y0, y1, x0, x1), ...] in latent-grid coordinates."""
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    ch, cw = H / rows, W / cols
    boxes = []
    for i in range(n):
        r, c = divmod(i, cols)
        cy, cx = (r + 0.5) * ch, (c + 0.5) * cw
        hy, hx = fill * ch / 2, fill * cw / 2
        y0, y1 = int(round(cy - hy)), int(round(cy + hy))
        x0, x1 = int(round(cx - hx)), int(round(cx + hx))
        boxes.append((max(0, y0), min(H, max(y0 + 1, y1)),
                      max(0, x0), min(W, max(x0 + 1, x1))))
    return boxes


def _bump_envelope(boxes, H, W, omega, alpha):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32),
                            indexing="ij")
    bump = torch.zeros((H, W), dtype=torch.float32)
    for (y0, y1, x0, x1) in boxes:
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        sy = alpha * (y1 - y0) / 2 + 1e-6
        sx = alpha * (x1 - x0) / 2 + 1e-6
        bump += omega * torch.exp(-(((yy - cy) / sy) ** 2 +
                                    ((xx - cx) / sx) ** 2) / 2)
    return bump


def count_aware_latent(base, n, scheme="gaussian", gamma=0.1, omega=0.3,
                       alpha=0.8, fixed_value=0.0, fill=0.6, beta=1.6,
                       noise_seed=0):
    """Inject a count-aware layout into the initial latent: partition into N
    boxes and perturb the in-box noise. Schemes:
      - uniform_scaled: scale in-box noise by gamma
      - fixed: set in-box to fixed_value
      - gaussian: add a positive Gaussian bump (DC -> can tint color)
      - noise_boost: multiply in-box noise by beta>1 (artifact-free, zero-mean)
      - gaussian_noise: add a Gaussian-enveloped fresh noise (artifact-free)
    `base` is (1, C, H, W); returns a modified copy."""
    lat = base.clone()
    _, C, H, W = lat.shape
    boxes = grid_boxes(n, H, W, fill)
    if scheme == "uniform_scaled":
        for (y0, y1, x0, x1) in boxes:
            lat[:, :, y0:y1, x0:x1] *= gamma
    elif scheme == "fixed":
        for (y0, y1, x0, x1) in boxes:
            lat[:, :, y0:y1, x0:x1] = fixed_value
    elif scheme == "noise_boost":
        for (y0, y1, x0, x1) in boxes:
            lat[:, :, y0:y1, x0:x1] *= beta
    elif scheme == "gaussian":
        bump = _bump_envelope(boxes, H, W, omega, alpha)
        lat = lat + bump.to(lat.dtype).to(lat.device)[None, None]
    elif scheme == "gaussian_noise":
        bump = _bump_envelope(boxes, H, W, omega, alpha)
        g = torch.Generator(device="cpu").manual_seed(noise_seed)
        noise = torch.randn((1, C, H, W), generator=g)
        extra = bump[None, None] * noise
        lat = lat + extra.to(lat.dtype).to(lat.device)
    else:
        raise ValueError(f"unknown scheme {scheme}")
    return lat

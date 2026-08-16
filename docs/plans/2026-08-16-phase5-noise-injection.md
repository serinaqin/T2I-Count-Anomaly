# Phase 5: Count-Aware Noise Injection (mitigation)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** Since the count is noise-bound and set spatially in the early window (Phases 2c/4/4b), mitigate at the **initial latent**: partition it into N non-overlapping regions and perturb the in-region noise so the model realizes N objects. Measure exact-count accuracy vs baseline. This is Demystifying's training-free remedy — applied because we *proved* the mechanism it targets on SDXL.

**Architecture:** Build the seed's base latent, apply a count-aware layout (Uniform-Scaled / Fixed / Gaussian) for the requested N, pass it as `latents=` to the pipeline, score the output. Compare to the unmodified baseline.

**Tech Stack:** Phase 0 stack.

## Global Constraints
- Counts 1–7; detector-oracle scoring; exact-count accuracy + MAE.
- Modify a UNIT-variance base latent (the pipeline scales by init_noise_sigma internally).
- Three schemes with the paper's defaults: Uniform-Scaled γ=0.1, Gaussian ω=0.3/α=0.8, Fixed.
- Compare baseline vs injected on the SAME base latent (apples-to-apples).

---

### Task 1: Noise-layout module (`src/noise_layout.py`)

**Files:** Create `src/noise_layout.py`, `tests/test_noise_layout.py`.

**Interfaces (Produces):**
- `grid_boxes(n, H, W, fill=0.6) -> list[(y0,y1,x0,x1)]` — N non-overlapping boxes on a near-square grid, each filling `fill` of its cell.
- `count_aware_latent(base, n, scheme="gaussian", gamma=0.1, omega=0.3, alpha=0.8, fixed_value=0.0, fill=0.6) -> Tensor` — copy of `base` (shape (1,C,H,W)) with the in-box noise modified per scheme.

- [ ] **Step 1: failing tests** — `tests/test_noise_layout.py`:
```python
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
    assert out[0, 0, 32, 32] > 0.1          # bump at the single-box center
    assert torch.allclose(out[0, 0, 0, 0], torch.tensor(0.0), atol=1e-4)  # corner ~untouched

def test_fixed_and_uniform_scaled():
    base = torch.ones(1, 4, 64, 64)
    f = count_aware_latent(base, 1, scheme="fixed", fixed_value=0.0)
    (y0, y1, x0, x1) = grid_boxes(1, 64, 64)[0]
    assert torch.allclose(f[0, 0, y0:y1, x0:x1], torch.zeros(y1 - y0, x1 - x0))
    u = count_aware_latent(base, 1, scheme="uniform_scaled", gamma=0.1)
    assert torch.allclose(u[0, 0, y0:y1, x0:x1], 0.1 * torch.ones(y1 - y0, x1 - x0))
```

- [ ] **Step 2: run to fail** — `pytest tests/test_noise_layout.py -q`.

- [ ] **Step 3: implement** — `src/noise_layout.py`:
```python
import math
import torch


def grid_boxes(n, H, W, fill=0.6):
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


def count_aware_latent(base, n, scheme="gaussian", gamma=0.1, omega=0.3,
                       alpha=0.8, fixed_value=0.0, fill=0.6):
    lat = base.clone()
    _, _, H, W = lat.shape
    boxes = grid_boxes(n, H, W, fill)
    if scheme == "uniform_scaled":
        for (y0, y1, x0, x1) in boxes:
            lat[:, :, y0:y1, x0:x1] *= gamma
    elif scheme == "fixed":
        for (y0, y1, x0, x1) in boxes:
            lat[:, :, y0:y1, x0:x1] = fixed_value
    elif scheme == "gaussian":
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
        lat = lat + bump.to(lat.dtype).to(lat.device)[None, None]
    else:
        raise ValueError(f"unknown scheme {scheme}")
    return lat
```

- [ ] **Step 4: run to pass** — `pytest -q`.
- [ ] **Step 5: commit** — `git commit -am "feat: count-aware noise-layout injection (Phase 5 Task 1)"`

---

### Task 2: Config (`configs/phase5.yaml`)

- [ ] Create:
```yaml
counts: [1, 2, 3, 4, 5]
objects: [cat]
seeds: [0, 1, 2, 3, 4]
num_inference_steps: 30
score_threshold: 0.3
schemes: [uniform_scaled, fixed, gaussian]
gamma: 0.1
omega: 0.3
alpha: 0.8
box_fill: 0.6
```
- [ ] Commit — `git commit -am "feat: phase-5 config (Task 2)"`

---

### Task 3: Notebook (`notebooks/05_noise_injection.ipynb`)

Flow:
1. clone/install; imports (`count_aware_latent`, `build_prompt`, detector, `load_sdxl`).
2. helpers: `base_latent(seed)` -> `torch.randn((1,4,128,128), generator=cpu_seed).to(pipe dtype/device)`; `gen_latent(prompt, lat)` -> `pipe(prompt, latents=lat, num_inference_steps=...).images[0]`.
3. loop over requested count N × object × seed: `base`; baseline = `gen_latent(build_prompt(N,obj), base.clone())`; for each scheme: `lat = count_aware_latent(base, N, scheme, ...)`, injected image, detector count. Record exact-match (count==N) and error.
4. save `results/phase5_counts.csv`.
5. **Accuracy plot:** exact-count accuracy (and MAE) baseline vs each scheme, overall and per requested count. Bar chart.
6. **Eyeball:** for a few (N, seed), baseline | best-scheme injected, with counts — do the injected images show N in coherent scenes?
7. **Verdict md:** a scheme with clearly higher exact-accuracy than baseline = a working, training-free, causally-motivated mitigation -> the full arc closes (interpret -> localize -> causal -> characterize -> mitigate). No gain -> the SDXL noise->count coupling needs a stronger layout prior (bigger perturbation / boxes) or fine-tuning (out of scope); report honestly.

- [ ] Build `notebooks/_build_05.py`, emit, `nbformat.validate`, commit — `git commit -am "feat: phase-5 noise-injection notebook (Task 3)"`.

---

## Self-Review
- **Mitigates at the proven level (noise/early spatial):** count-aware latent (Tasks 1, 3). ✅
- **Decisive metric:** exact-count accuracy vs baseline, per scheme (Task 3). ✅
- **Apples-to-apples:** same base latent for baseline and injected (Task 3). ✅
- **Three schemes, paper defaults** (Global Constraints, Task 2). ✅
- **Testable core:** grid_boxes + count_aware_latent unit-tested; generation GPU-smoke (Task 1). ✅
- **Types:** base latent (1,4,128,128); count_aware_latent preserves shape; passed as latents= (Task 3). ✅

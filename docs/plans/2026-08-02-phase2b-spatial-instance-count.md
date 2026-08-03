# Phase 2b: Spatial Instance-Count Readout

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** Phase 2 (pooled probe) was blind to spatial count. Here we read the count *spatially* — cluster the up-block self-attention feature maps into object blobs — and find WHERE/WHEN the internal instance-count first diverges from the requested number. This is "where the model loses the connection between the known number and the spatial layout."

**Architecture:** Reuse Phase 0–2. Capture the *un-pooled* conditional self-attention feature map at up-block `attn1` sites at several timesteps, convert each to a per-pixel saliency map, threshold + connected-components to get an **internal instance count**, and compare that count to the requested and rendered counts across sites × timesteps.

**Tech Stack:** Phase 0–2 stack + `scipy.ndimage` (connected components).

## Global Constraints
- Counts 1–7 only; detector-oracle rendered count.
- Read the count **spatially** (blob count), never pooled — pooling is what failed in Phase 2.
- Self-attention (`attn1`) only — image-side realization; up-blocks (the localized region).
- Saliency threshold is tunable and **validated by eyeballing** saliency maps against images before trusting the counts.

---

### Task 1: Spatial readout helpers (`src/spatial.py`)

**Files:** Create `src/spatial.py`, `tests/test_spatial.py`.

**Interfaces (Produces):**
- `featuremap_saliency(act, cond_index=1) -> np.ndarray` — (B,T,C)→ cond half (T,C) → per-pixel L2 norm over C → reshape to (H,W) with H=W=round(sqrt(T)) → min-max normalize to [0,1].
- `saliency_to_instance_count(sal, thresh=0.5, min_size=1) -> int` — threshold `sal >= thresh`, count 4-connected components of size ≥ `min_size`.

- [ ] **Step 1: failing tests** — `tests/test_spatial.py`:
```python
import numpy as np
from src.spatial import featuremap_saliency, saliency_to_instance_count

def test_featuremap_saliency_shape_and_peak():
    # (B=2, T=4, C=3): cond half has one strong pixel
    cond = np.zeros((4, 3)); cond[2] = [3.0, 4.0, 0.0]   # norm 5 at token 2
    act = np.stack([np.zeros((4, 3)), cond])
    sal = featuremap_saliency(act, cond_index=1)
    assert sal.shape == (2, 2)
    assert sal.max() == 1.0
    assert np.unravel_index(np.argmax(sal), sal.shape) == (1, 0)  # token 2 -> row1,col0

def test_saliency_to_instance_count_two_blobs():
    sal = np.zeros((5, 5))
    sal[0, 0] = 1.0            # blob 1
    sal[4, 4] = 1.0; sal[4, 3] = 1.0   # blob 2 (2 px)
    assert saliency_to_instance_count(sal, thresh=0.5) == 2

def test_saliency_min_size_filters_specks():
    sal = np.zeros((5, 5)); sal[0, 0] = 1.0; sal[4, 4] = 1.0; sal[4, 3] = 1.0
    assert saliency_to_instance_count(sal, thresh=0.5, min_size=2) == 1  # drops 1-px blob
```

- [ ] **Step 2: run to fail** — `pytest tests/test_spatial.py -q`.

- [ ] **Step 3: implement** — `src/spatial.py`:
```python
import numpy as np
from scipy import ndimage


def featuremap_saliency(act, cond_index=1):
    a = act.detach().cpu().float().numpy() if hasattr(act, "detach") \
        else np.asarray(act, float)
    if a.ndim == 3:
        a = a[cond_index] if a.shape[0] > cond_index else a[-1]   # (T, C)
    norm = np.linalg.norm(a, axis=-1)                             # (T,)
    h = int(round(np.sqrt(norm.shape[0])))
    sal = norm[: h * h].reshape(h, h)
    lo, hi = sal.min(), sal.max()
    return (sal - lo) / (hi - lo) if hi > lo else np.zeros_like(sal)


def saliency_to_instance_count(sal, thresh=0.5, min_size=1) -> int:
    mask = np.asarray(sal) >= thresh
    labeled, n = ndimage.label(mask)
    if n == 0:
        return 0
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    return int((sizes >= min_size).sum())
```

- [ ] **Step 4: run to pass** — `pytest -q` (full suite).
- [ ] **Step 5: commit** — `git commit -am "feat: spatial saliency + instance-count readout (Phase 2b Task 1)"`

---

### Task 2: Reducer hook in capture (`src/pipeline.py`) + config

**Files:** Modify `src/pipeline.py`; create `configs/phase2b.yaml`.

**Interfaces:** `generate_and_capture(..., reducer=None)` — if `reducer` is None use `pool_activation`; otherwise store `reducer(act, cond_index)` per site (lets us store a saliency map instead of a pooled vector).

- [ ] **Step 1: implement** — change `generate_and_capture` signature to add `reducer=None`, default to `pool_activation` inside, and apply it in the snapshot dict comprehension (replace `pool_activation(cap.acts[s], cond_index)` with `reducer(cap.acts[s], cond_index)`).
- [ ] **Step 2:** `configs/phase2b.yaml`:
```yaml
# Phase 2b spatial instance-count: up-block self-attn maps across timesteps
counts: [1, 2, 3, 4, 5]
objects: [cat, dog]
seeds: [0, 1, 2, 3, 4, 5, 6, 7]
num_inference_steps: 30
capture_steps: [3, 6, 10, 15, 22, 29]
score_threshold: 0.3
saliency_thresh: 0.5
```
  (`saliency_thresh` read via raw yaml; not an ExperimentConfig field.)
- [ ] **Step 3:** `pytest -q` → PASS (existing Phase 2 notebook still uses the default reducer; no break).
- [ ] **Step 4: commit** — `git commit -am "feat: pluggable reducer in generate_and_capture + phase2b config (Task 2)"`

---

### Task 3: Phase-2b notebook (`notebooks/02b_spatial_instance_count.ipynb`)

**Files:** Create `notebooks/_build_02b.py` + emitted notebook.

Flow:
1. clone/install (`pip install scipy` is already a dep; ensure present).
2. imports (+ `featuremap_saliency`, `saliency_to_instance_count`).
3. `cfg = load_config('configs/phase2b.yaml')`; `raw = yaml.safe_load(...)`; `thr_s = raw['saliency_thresh']`.
4. `pipe, det`; `sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet)) if 'up_blocks' in s and s.endswith('attn1')]` (the up-block image-side sites).
5. generation loop with `reducer=featuremap_saliency`: for each image store, per (step, site), the saliency map; keep the PIL image and rendered count. Save `results/phase2b_counts.csv`.
6. compute internal **instance count** = `saliency_to_instance_count(sal, thr_s)` per (image, site, step).
7. **Divergence analysis** — for each (site, step): correlation of the internal instance-count with **requested** and with **rendered**; and mean internal count vs mean requested. A step where internal-count correlates with *requested* = the number is still honored there; where it stops correlating with requested and starts matching *rendered* = where the count is lost. Plot both correlations vs timestep (per site), and mean |internal − requested| heatmap over (site × step).
8. **Eyeball cell** — for ~6 images, show the image + detector count + the saliency map at the localized site/step with its internal instance count overlaid, so the saliency segmentation is validated against reality (and the threshold tuned).
9. Verdict md: (a) does the internal instance-count ever match the requested count, and at which step does it diverge? (b) is it wrong from the earliest captured step (count set at/near the noise) or does it drift later (dynamic failure)? (c) which up-block site tracks the rendered count best → the Phase 2c causal-patch target.

- [ ] **Step 1:** write `notebooks/_build_02b.py` with all plotting verbatim; **Step 2:** build + `nbformat.validate`; **Step 3: commit** `git commit -am "feat: phase-2b spatial instance-count notebook (Task 3)"`.

---

## Self-Review
- **Spatial (not pooled) readout:** saliency + connected components → internal instance count (Task 1). ✅
- **Where/when it diverges:** correlation-with-requested-vs-rendered across site×step + divergence heatmap (Task 3). ✅
- **Set-early vs drift:** capture_steps span 3→29; verdict reads the trajectory (Task 3). ✅
- **Validation:** eyeball saliency vs image before trusting counts; threshold tunable (Task 3, config). ✅
- **No break to prior phases:** reducer defaults to pool_activation (Task 2). ✅
- **Placeholders/types:** helper signatures `(act, cond_index)` match reducer contract; scipy label usage concrete. ✅

# Phase 2: Trace the Count Signal (layer × timestep)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** Find WHERE (layer/attention type) and WHEN (timestep) SDXL commits to an object count, and whether the failure is on the **image side**, the **text→image matching** (cross-attention), or **distributed** — building on Phase 1's finding that *text drives the count but realizes it unreliably (over-generation)*.

**Architecture:** Reuse Phase 0/1. During generation, snapshot pooled activations at a curated set of attention sites at several timesteps (one generation, multiple snapshots via a step-end callback). For each image, label the **requested** count (prompt) and the **rendered** count (detector). Then, at every (site, timestep), measure how linearly decodable each count is (probe R²). Compare **attn1** (self-attention = image-side, no text leak) vs **attn2** (cross-attention = the matching junction) vs the **text embedding**.

**Tech Stack:** Same as Phase 0/1.

## Global Constraints
- Counts 1–7 only; template "{number-word} {object}"; detector-oracle scoring.
- **Text-leak control (crux):** the requested number is a token in the text conditioning and enters via cross-attention (attn2). To claim the *image side* represents the count, probe **attn1 / self-attention** (image↔image); treat attn2 decodability as the matching junction, not proof of image-side count.
- Pool activations as: take the **conditional half** of the CFG batch (index 1) and **mean over tokens** → one channel-vector per (image, site, step).
- All pure logic in `src/`, unit-tested off-GPU; SDXL/probe-capture run only on Colab.

---

### Task 1: Pure helpers (`src/pipeline.py`, `src/probes.py`)

**Files:** Modify `src/pipeline.py`, `src/probes.py`, `tests/test_pipeline.py`, `tests/test_probes.py`.

**Interfaces (Produces):**
- `select_probe_sites(sites: list[str]) -> list[str]` — keep only `...transformer_blocks.0.attn1|attn2` (one representative block per attentions module → ~22 sites, depth-ordered).
- `pool_activation(act, cond_index=1) -> np.ndarray` — (B,T,C)→cond half, mean over T →(C,); (T,C)→mean→(C,); else ravel. Accepts torch or numpy.
- `decodability_map(features: dict[str, np.ndarray], y, mode="magnitude") -> dict[str, float]` — per site, R² (magnitude) or balanced acc (classify) of decoding y from that site's (n×C) feature matrix.

- [ ] **Step 1: failing tests** — append to `tests/test_pipeline.py`:
```python
import numpy as np
from src.pipeline import select_probe_sites, pool_activation

def test_select_probe_sites():
    sites = [
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2",
        "down_blocks.1.attentions.0.transformer_blocks.1.attn1",  # block 1 -> dropped
        "mid_block.attentions.0.transformer_blocks.0.attn1",
    ]
    got = select_probe_sites(sites)
    assert got == [sites[0], sites[1], sites[3]]

def test_pool_activation_cond_half_mean():
    # (B=2, T=3, C=2): uncond all 0, cond all 5 -> pooled = [5,5]
    act = np.stack([np.zeros((3, 2)), np.full((3, 2), 5.0)])
    out = pool_activation(act, cond_index=1)
    assert out.shape == (2,)
    assert np.allclose(out, [5.0, 5.0])
```
and append to `tests/test_probes.py`:
```python
from src.probes import decodability_map

def test_decodability_map_ranks_sites():
    rng = np.random.default_rng(0)
    y = rng.integers(1, 6, 120).astype(float)
    good = y[:, None] * np.array([1.0, 0, 0]) + rng.normal(0, 0.05, (120, 3))
    bad = rng.normal(0, 1, (120, 3))
    m = decodability_map({"good": good, "bad": bad}, y, mode="magnitude")
    assert m["good"] > 0.9
    assert m["bad"] < 0.3
```

- [ ] **Step 2: run to fail** — `pytest tests/test_pipeline.py tests/test_probes.py -q` → FAIL.

- [ ] **Step 3: implement** — add to `src/pipeline.py`:
```python
import numpy as np


def select_probe_sites(sites):
    return [s for s in sites
            if "transformer_blocks.0." in s
            and (s.endswith("attn1") or s.endswith("attn2"))]


def pool_activation(act, cond_index=1):
    a = act.detach().cpu().float().numpy() if hasattr(act, "detach") else np.asarray(act, float)
    if a.ndim == 3:
        b = a[cond_index] if a.shape[0] > cond_index else a[-1]
        return b.mean(axis=0)
    if a.ndim == 2:
        return a.mean(axis=0)
    return a.ravel()
```
and add to `src/probes.py`:
```python
def decodability_map(features, y, mode="magnitude") -> dict:
    out = {}
    for site, X in features.items():
        if mode == "classify":
            out[site] = fit_eval_classifier(X, y)["bal_acc"]
        else:
            out[site] = fit_eval_magnitude(X, y)["r2"]
    return out
```

- [ ] **Step 4: run to pass** — `pytest -q` (full suite) → PASS.
- [ ] **Step 5: commit** — `git commit -am "feat: phase-2 probe-site selection, pooling, decodability map"`

---

### Task 2: Capture-during-generation (`src/pipeline.py`) + config

**Files:** Modify `src/pipeline.py`; create `configs/phase2.yaml`.

**Interfaces (Produces):**
- `generate_and_capture(pipe, prompt, seed, sites, capture_steps, num_inference_steps=30, cond_index=1) -> (PIL.Image, dict[int, dict[str, np.ndarray]])` — one generation; snapshots pooled activations at each step in `capture_steps`. GPU-only; validated on Colab.

- [ ] **Step 1: implement** — add to `src/pipeline.py` (GPU-only, not unit-tested; smoke-validated in the notebook):
```python
def generate_and_capture(pipe, prompt, seed, sites, capture_steps,
                         num_inference_steps=30, cond_index=1):
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    store = {}
    targets = set(capture_steps)
    with ActivationCapture(pipe.unet, sites) as cap:
        def cb(p, step, timestep, kw):
            if step in targets:
                store[step] = {s: pool_activation(cap.acts[s], cond_index)
                               for s in cap.acts}
            return kw
        img = pipe(prompt, generator=g,
                   num_inference_steps=num_inference_steps,
                   callback_on_step_end=cb).images[0]
    return img, store
```
- [ ] **Step 2:** create `configs/phase2.yaml`:
```yaml
# Phase 2 signal-trace: probe layers x timesteps
counts: [1, 2, 3, 4, 5]
objects: [cat, dog]
seeds: [0, 1, 2, 3, 4, 5, 6, 7]   # 5 x 2 x 8 = 80 images
num_inference_steps: 30
capture_steps: [6, 15, 29]         # early / mid / late
score_threshold: 0.3
```
- [ ] **Step 3:** confirm the module imports cleanly: `pytest -q` → PASS (no new tests; ensures no syntax/import break).
- [ ] **Step 4: commit** — `git commit -am "feat: generate_and_capture + phase-2 config"`

---

### Task 3: Phase-2 notebook (`notebooks/02_signal_trace.ipynb`)

**Files:** Create `notebooks/_build_02.py` and emitted `notebooks/02_signal_trace.ipynb`.

Notebook flow (thin driver):
1. clone/install; imports (`generate_grid`, `load_sdxl`, `catalog_attention_sites`, `select_probe_sites`, `generate_and_capture`, `Detector`, `count_from_detections`, `decodability_map`, `load_config`).
2. `cfg = load_config('configs/phase2.yaml')`; note `capture_steps` read via a raw yaml load (ExperimentConfig has no such field) — load with `import yaml; raw = yaml.safe_load(open('configs/phase2.yaml')); capture_steps = raw['capture_steps']`.
3. `pipe = load_sdxl(); sites = select_probe_sites(catalog_attention_sites(pipe.unet)); print(len(sites), 'probe sites')`.
4. generation+capture loop over `generate_grid(...)`: `img, snaps = generate_and_capture(pipe, p.text, p.seed, sites, capture_steps, cfg.num_inference_steps)`; `rendered = count_from_detections(det.detect(img,[p.obj]), p.obj, thr)`; append `requested=p.count`, `rendered`, and `snaps` (dict step->site->vec). Save counts to `results/phase2_counts.csv`.
5. assemble, per (step, site), a feature matrix `X` (n_images × C) and probe both **requested** and **rendered** count with `decodability_map(..., mode="magnitude")`.
6. **also** probe the fused text embedding for requested count (H3) — defensive `try/except` around `pipe.encode_prompt`, mean-pool tokens; if the API differs, print a note and skip.
7. **Plots:**
   - Heatmap A: requested-count R² over (site rows × step cols).
   - Heatmap B: rendered-count R² over (site rows × step cols).
   - The GAP (A − B): where the number is present but the eventual count is not yet realized.
   Split attn1 vs attn2 rows visually (or annotate) so image-side vs matching is readable.
8. **Verdict md:**
   - Rendered-count R² rising at a specific (attn1 site, step) = the **image-side commitment** point.
   - Requested decodable from **attn1** early but rendered only later/downstream ⇒ image builds the number over steps (realization).
   - If requested is decodable mainly from **attn2** (text leak) but weakly from attn1, and rendered stays low until late ⇒ the number lives in the text/matching channel and is realized late ⇒ **matching/realization is the break** (Phase-1-consistent).
   - Where rendered-count R² first becomes high = the site/step to target in Phase 4 mitigation.

- [ ] **Step 1:** write `notebooks/_build_02.py` emitting the above (put all plotting code in verbatim; use `matplotlib` `imshow` for heatmaps with site labels on the y-axis and capture_steps on the x-axis).
- [ ] **Step 2:** `python notebooks/_build_02.py`; validate with nbformat; confirm cell count.
- [ ] **Step 3: commit** — `git commit -am "feat: phase-2 signal-trace notebook"`

---

## Self-Review
- **Where/when covered:** layer×timestep R² map (Task 3 heatmaps) from captured snapshots (Task 2) at curated sites (Task 1). ✅
- **Image vs matching vs text:** attn1 (image) vs attn2 (matching) split + text-embedding probe. ✅
- **Text-leak control:** attn1 is the load-bearing image-side signal; attn2/text treated as the text channel — stated in Global Constraints and the verdict. ✅
- **Robust to messy rendered counts:** magnitude R² (not class accuracy) for the maps. ✅
- **Placeholders:** capture callback + pooling + plots all given verbatim; encode_prompt wrapped defensively. ✅
- **Types:** `pool_activation`/`select_probe_sites`/`generate_and_capture`/`decodability_map` signatures consistent across pipeline↔probes↔notebook; feature dict shape (n×C) matches `decodability_map`. ✅

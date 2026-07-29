# Phase 1: Noise Seed-Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Measure how much of SDXL's realized object count is driven by the **initial noise (seed)** versus the **text prompt** — the Phase 1 gate for the whole investigation.

**Architecture:** Reuse the Phase 0 pipeline. Generate a grid where the SAME set of seeds is crossed against every (count, object), so each seed's initial latent is reused across prompts (a fixed seed integer → identical initial noise at fixed resolution). Score with the detector oracle. Then decompose the realized count into noise vs text contributions with new, unit-tested analysis functions, and visualize.

**Tech Stack:** Same as Phase 0 (diffusers/SDXL, GroundingDINO, sklearn, pandas, matplotlib, pytest).

## Global Constraints
- Counts 1–7 only; template "{number-word} {object}"; detector-oracle scoring (no VLM).
- Same seed list reused across ALL prompts (this is what makes the swap valid).
- All logic in `src/`, unit-tested off-GPU; SDXL/detector run only on Colab.
- Report robust stats (MAE, off-by-one) alongside exact-match; flag degenerate collage images rather than deleting them.

---

### Task 1: Phase-1 analysis functions (`src/analysis.py`)

**Files:** Modify `src/analysis.py`; modify `tests/test_analysis.py`.

**Interfaces (Produces):**
- `text_responsiveness(df, realized="realized_count", requested="count") -> dict` → `{"slope": float, "r2": float}` (OLS realized ~ requested; slope≈1 means text controls count, slope≈0 means text ignored).
- `count_variance_decomposition(df, value="realized_count", factors=("seed","count","obj")) -> dict` → marginal η² per factor (fraction of realized-count variance aligning with each; not orthogonal — documented).
- `per_seed_summary(df, value="realized_count", seed="seed") -> pandas.DataFrame` → index=seed, columns `mean`,`std` (a seed with low std across prompts has a persistent "preferred count").
- `flag_degenerate(df, max_requested, factor=3, value="realized_count") -> pandas.DataFrame` → copy of df with bool column `degenerate` = `value > factor*max_requested` (e.g. the collage 49s).

- [ ] **Step 1: Write failing tests** — append to `tests/test_analysis.py`:
```python
import pandas as pd
from src.analysis import (text_responsiveness, count_variance_decomposition,
                          per_seed_summary, flag_degenerate)

def test_text_responsiveness_perfect():
    df = pd.DataFrame({"realized_count": [1,2,3,4], "count": [1,2,3,4]})
    out = text_responsiveness(df)
    assert abs(out["slope"] - 1.0) < 1e-6
    assert out["r2"] > 0.99

def test_text_responsiveness_ignored():
    # realized count independent of requested -> slope ~ 0
    df = pd.DataFrame({"realized_count": [3,3,3,3,3,3],
                       "count": [1,2,3,4,5,6]})
    out = text_responsiveness(df)
    assert abs(out["slope"]) < 1e-6

def test_count_variance_decomposition_keys_and_noise_dominant():
    # realized determined ONLY by seed
    rows = []
    for seed, val in {0:2, 1:5}.items():
        for c in [1,2,3]:
            rows.append({"realized_count": val, "seed": seed, "count": c, "obj":"cat"})
    df = pd.DataFrame(rows)
    out = count_variance_decomposition(df)
    assert set(out) == {"seed","count","obj"}
    assert out["seed"] > 0.99 and out["count"] < 0.01

def test_per_seed_summary():
    df = pd.DataFrame({"realized_count":[2,2,5,5], "seed":[0,0,1,1], "count":[1,2,1,2]})
    s = per_seed_summary(df)
    assert list(s.index) == [0,1]
    assert s.loc[0,"mean"] == 2 and s.loc[0,"std"] == 0
    assert s.loc[1,"mean"] == 5

def test_flag_degenerate():
    df = pd.DataFrame({"realized_count":[1,2,49], "count":[1,2,1]})
    out = flag_degenerate(df, max_requested=5, factor=3)  # threshold = 15
    assert out["degenerate"].tolist() == [False, False, True]
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_analysis.py -v` → FAIL (imports missing).

- [ ] **Step 3: Implement** — append to `src/analysis.py`:
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def text_responsiveness(df, realized="realized_count", requested="count") -> dict:
    x = df[[requested]].to_numpy(float)
    y = df[realized].to_numpy(float)
    reg = LinearRegression().fit(x, y)
    return {"slope": float(reg.coef_[0]),
            "r2": float(r2_score(y, reg.predict(x)))}


def count_variance_decomposition(df, value="realized_count",
                                 factors=("seed", "count", "obj")) -> dict:
    return {f: variance_explained(df, value, f) for f in factors}


def per_seed_summary(df, value="realized_count", seed="seed") -> pd.DataFrame:
    g = df.groupby(seed)[value]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0)})


def flag_degenerate(df, max_requested, factor=3, value="realized_count"):
    out = df.copy()
    out["degenerate"] = out[value] > factor * max_requested
    return out
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_analysis.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add src/analysis.py tests/test_analysis.py && git commit -m "feat: phase-1 noise-vs-text analysis functions"`

---

### Task 2: Phase-1 experiment config (`configs/phase1.yaml`)

**Files:** Create `configs/phase1.yaml`.

- [ ] **Step 1:** Create `configs/phase1.yaml` (moderate grid; shrink seeds for a first pilot):
```yaml
# Phase 1 seed-swap: same seeds crossed against every (count, object)
counts: [1, 2, 3, 4, 5]
objects: [cat, dog, apple, car]
seeds: [0, 1, 2, 3, 4, 5, 6, 7]   # 5 x 4 x 8 = 160 images
num_inference_steps: 30
score_threshold: 0.3
```
- [ ] **Step 2:** Sanity-check it loads: `python -c "from src.config import load_config; print(load_config('configs/phase1.yaml'))"` (run from repo root with `sys.path` = repo). Expected: prints an ExperimentConfig with 5 counts, 4 objects, 8 seeds.
- [ ] **Step 3: Commit** — `git add configs/phase1.yaml && git commit -m "feat: phase-1 seed-swap config"`

---

### Task 3: Phase-1 notebook (`notebooks/01_noise_seed_swap.ipynb`)

**Files:** Create `notebooks/_build_01.py` (generator) and the emitted `notebooks/01_noise_seed_swap.ipynb`.

The notebook (thin driver) does: clone/install → import src → build the seed-swap grid → generate + detector-score (store detections) → save `results/phase1_counts.csv` → run the three analyses → produce the money plots → print a verdict.

- [ ] **Step 1:** Write `notebooks/_build_01.py` that emits the notebook with these cells:
  1. md: title + what Phase 1 tests.
  2. code: clone/install (skip clone if `src` exists); `pip install pytest groundingdino-py`.
  3. code: imports (`generate_grid`, `load_sdxl`, `generate`, `Detector`, `count_from_detections`, `load_config`, and the four analysis fns).
  4. code: `cfg = load_config('configs/phase1.yaml'); grid = generate_grid(cfg.counts, cfg.objects, cfg.seeds); print(len(grid))`.
  5. code: `pipe = load_sdxl(); det = Detector()`.
  6. code: generation loop — for each spec: `img=generate(...)`, `dets=det.detect(img,[obj])`, `n=count_from_detections(dets,obj,thr)`; append row `{obj,count,seed,realized_count}`; `df=pd.DataFrame(rows); df.to_csv('results/phase1_counts.csv')`; show df.head().
  7. code: `df = flag_degenerate(df, max_requested=max(cfg.counts))`; print how many degenerate; make a `df_clean = df[~df.degenerate]`.
  8. code — **verdict metrics:** `print(text_responsiveness(df_clean)); print(count_variance_decomposition(df_clean)); per_seed_summary(df_clean)`.
  9. code — **money plot A (text responsiveness):** per-seed line plot, x=requested count, y=mean realized count for that (seed,count), one line per seed; overlay the diagonal y=x. Flat lines ⇒ noise-fixed; diagonal-tracking ⇒ text-controlled.
  10. code — **money plot B (variance decomposition):** bar chart of η² for seed vs count vs obj.
  11. md — how to read the verdict (flat lines + seed η² ≫ count η² + slope≈0 ⇒ noise dominates ⇒ Phase 2 localizes where noise's count signal is read; diagonal + count η² ≫ seed η² + slope≈1 ⇒ text controls ⇒ pivot Phase 2 to the text/matching pathway).

  Plot A code (put in the builder verbatim):
```python
import matplotlib.pyplot as plt
piv = df_clean.groupby(['seed','count'])['realized_count'].mean().reset_index()
fig, ax = plt.subplots(figsize=(6,5))
for s, g in piv.groupby('seed'):
    ax.plot(g['count'], g['realized_count'], marker='o', alpha=0.6, label=f'seed {s}')
lims = [min(cfg.counts), max(cfg.counts)]
ax.plot(lims, lims, 'k--', label='y=x (perfect text control)')
ax.set_xlabel('requested count'); ax.set_ylabel('mean realized count')
ax.set_title('Per-seed count response (flat = noise-fixed, diagonal = text-controlled)')
ax.legend(fontsize=7); plt.tight_layout()
plt.savefig('results/phase1_perseed.png', dpi=90, bbox_inches='tight'); plt.show()
```
  Plot B code:
```python
dec = count_variance_decomposition(df_clean)
fig, ax = plt.subplots(figsize=(4,4))
ax.bar(list(dec.keys()), list(dec.values()))
ax.set_ylabel('variance explained (eta^2)'); ax.set_ylim(0,1)
ax.set_title('What drives realized count?')
plt.tight_layout(); plt.savefig('results/phase1_variance.png', dpi=90, bbox_inches='tight'); plt.show()
```

- [ ] **Step 2:** `python notebooks/_build_01.py` then validate with nbformat (`nbformat.validate`), confirm cell count.
- [ ] **Step 3: Commit** — `git add notebooks/_build_01.py notebooks/01_noise_seed_swap.ipynb && git commit -m "feat: phase-1 seed-swap notebook"`

---

## Self-Review
- **Gate question covered:** noise-vs-text quantified by (a) text slope, (b) η² decomposition, (c) per-seed preference plot → Task 1 + Task 3. ✅
- **Same-seed-across-prompts design:** `generate_grid` reuses the seed list for every (count,obj); config supplies one shared seed list → Task 2. ✅
- **Robustness to collage outliers:** `flag_degenerate` + `df_clean` → Task 1 + Task 3 cell 7. ✅
- **Placeholder scan:** plot code given verbatim; no TBDs. ✅
- **Type consistency:** `variance_explained` reused by `count_variance_decomposition`; df columns `realized_count/seed/count/obj` consistent across analysis fns, config, and notebook. ✅

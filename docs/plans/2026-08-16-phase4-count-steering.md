# Phase 4: Count-Direction Steering (mitigation)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** Turn the Phase 2c causal finding (up-block self-attention, early steps, controls the count) into a **donor-free control knob**: learn a "more-objects" direction in that activation space and add it during generation. Test whether it monotonically moves the rendered count (dose-response) — the interpretability-native mitigation. Unlike the project's prior failed steering (wrong site, mid/late, cross-attn, correlational), this steers the **proven causal site at the right time**.

**Architecture:** (1) Capture pooled early self-attention activations across a count grid + rendered counts. (2) Estimate a per-(step,site) count direction = mean(high-count) − mean(low-count). (3) During generation, ADD α·direction at the early steps and sweep α to measure the dose-response of the rendered count.

**Tech Stack:** Phase 0–2c stack.

## Global Constraints
- Steer only the causal site/window: `up_blocks.0` self-attention (`attn1`), early steps (0–5).
- ADD a direction (not replace) — a donor-free steering vector.
- Score the final image with the detector; validate with an eyeball (coherent count change, not degradation).
- Direction is the raw mean-difference (natural scale); α is a multiplier (α=±1 ≈ one high−low step).

---

### Task 1: Steering primitives + direction estimator

**Files:** Modify `src/pipeline.py`, `src/probes.py`, `tests/test_pipeline.py`, `tests/test_probes.py`.

**Interfaces (Produces):**
- `make_steer_hook(site, directions, alpha, steps, state) -> hook` — at a patch step, returns `out + alpha * directions[step][site]` (broadcast (C,) over (...,C)), else None.
- `generate_with_steer(pipe, prompt, seed, directions, alpha, steps, num_inference_steps=30) -> Image` — GPU; adds the direction at `steps`.
- `count_direction(X, y, ) -> np.ndarray` (in probes) — mean(X[y>median(y)]) − mean(X[y<=median(y)]); zeros if a side is empty.

- [ ] **Step 1: failing tests** — append to `tests/test_pipeline.py`:
```python
def test_make_steer_hook_adds_direction():
    from src.pipeline import make_steer_hook
    d = torch.tensor([1.0, 2.0, 3.0])
    state = {"step": 2}
    hook = make_steer_hook("x", {2: {"x": d}}, alpha=2.0, steps={2}, state=state)
    out = torch.zeros(1, 4, 3)
    res = hook(None, (out,), out)
    assert torch.allclose(res, (2.0 * d).expand(1, 4, 3))
    state["step"] = 1
    assert hook(None, (out,), out) is None
```
and to `tests/test_probes.py`:
```python
def test_count_direction_points_to_high():
    rng = np.random.default_rng(0)
    y = np.array([1, 1, 5, 5], float)
    X = np.zeros((4, 3)); X[y > 3, 0] = 5.0     # high-count samples shifted on ch0
    from src.probes import count_direction
    d = count_direction(X, y)
    assert d[0] > 4 and abs(d[1]) < 1e-6
```

- [ ] **Step 2: run to fail** — `pytest tests/test_pipeline.py tests/test_probes.py -q`.

- [ ] **Step 3: implement** — add to `src/pipeline.py`:
```python
def make_steer_hook(site, directions, alpha, steps, state):
    def hook(_m, _i, out):
        st = state["step"]
        if st in steps and st in directions and site in directions[st]:
            d = directions[st][site].to(out.dtype).to(out.device)
            return out + alpha * d
        return None
    return hook


def generate_with_steer(pipe, prompt, seed, directions, alpha, steps,
                        num_inference_steps=30):
    steps = set(steps)
    sites = {s for st in directions for s in directions[st]}
    modmap = dict(pipe.unet.named_modules())
    state = {"step": 0}
    handles = [modmap[s].register_forward_hook(
        make_steer_hook(s, directions, alpha, steps, state)) for s in sites]
    g = torch.Generator(device=pipe.device).manual_seed(seed)

    def cb(_p, step, _t, kw):
        state["step"] = step + 1
        return kw
    try:
        img = pipe(prompt, generator=g, num_inference_steps=num_inference_steps,
                   callback_on_step_end=cb).images[0]
    finally:
        for h in handles:
            h.remove()
    return img
```
and to `src/probes.py`:
```python
def count_direction(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    thr = np.median(y)
    hi, lo = y > thr, y <= thr
    if hi.sum() == 0 or lo.sum() == 0:
        return np.zeros(X.shape[1])
    return X[hi].mean(0) - X[lo].mean(0)
```

- [ ] **Step 4: run to pass** — `pytest -q`.
- [ ] **Step 5: commit** — `git commit -am "feat: count-steering hook + direction estimator (Phase 4 Task 1)"`

---

### Task 2: Config (`configs/phase4.yaml`)

- [ ] Create:
```yaml
counts: [1, 2, 3, 4, 5]
objects: [cat]
seeds: [0, 1, 2, 3, 4, 5]      # training seeds for direction estimation
num_inference_steps: 30
score_threshold: 0.3
patch_block: up_blocks.0
patch_attn: attn1
steer_steps: [0, 1, 2, 3, 4, 5]
alphas: [-2, -1, 0, 1, 2]       # dose-response multipliers
eval_counts: [2, 3]
eval_seeds: [6, 7, 8, 9]        # held-out seeds for the dose-response
```
- [ ] Commit — `git commit -am "feat: phase-4 config (Task 2)"`

---

### Task 3: Notebook (`notebooks/04_count_steering.ipynb`)

Flow:
1. clone/install; imports (`generate_and_capture` with default pool reducer, `count_direction`, `generate_with_steer`, `generate`, detector, sites).
2. **Train:** over the count grid × training seeds, `generate_and_capture(..., reducer=pool_activation)` at steer_steps → pooled (C,) per (step, site) + rendered count. Build per-(step,site) feature matrices.
3. **Direction:** `directions[step][site] = torch.tensor(count_direction(X, rendered))`.
4. **Dose-response:** for each α in alphas, for each eval prompt (eval_counts × eval_seeds), `generate_with_steer(..., alpha=α, steps=steer_steps)`, detector-count. Plot **mean rendered count vs α** (should rise monotonically if the direction is a real count knob). Save csv.
5. **Eyeball:** one eval prompt at α = min, 0, max → images with counts (coherent add/remove?).
6. **Verdict md:** monotonic rising count-vs-α = donor-free causal control confirmed → a usable mitigation knob (calibrate α per requested count next). Flat = the learned pooled direction isn't sufficient (count needs spatial structure) → fall back to template/donor-bank injection.

- [ ] Build `notebooks/_build_04.py`, emit, `nbformat.validate`, commit — `git commit -am "feat: phase-4 count-steering notebook (Task 3)"`.

---

## Self-Review
- **Donor-free control:** learned direction added at the causal site/window (Tasks 1, 3). ✅
- **Decisive metric:** dose-response of rendered count vs α (Task 3). ✅
- **Not prior failed steering:** proven causal site (up0 attn1) + right time (early), ADD a learned count direction (Global Constraints). ✅
- **Coherence check:** eyeball at α extremes (Task 3). ✅
- **Testable primitives:** steer hook + direction estimator unit-tested; generate_with_steer GPU-smoke (Task 1). ✅
- **Types:** directions {step:{site:tensor}} consistent between count_direction output (wrapped to tensor) and make_steer_hook input. ✅

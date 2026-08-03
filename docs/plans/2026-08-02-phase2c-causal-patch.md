# Phase 2c: Causal Activation Patch

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** Test whether the up-block self-attention site (localized in Phase 2/2b) *causes* the count. Inject the attn1 activations from a "five {obj}" run into a "two {obj}" run at the commitment steps, and measure whether the **output** count moves toward five (scored by the detector — no reliance on the noisy internal counter). This is the correlational→causal upgrade the project's prior steering work lacked.

**Architecture:** For each seed, run a **donor** prompt and capture its raw attn1 tensors at the patch steps; run the **recipient** prompt unpatched (baseline) and patched (donor tensors injected via forward hooks at those steps). Compare recipient output counts. Do it **both directions** (2→inject5 should raise the count; 5→inject2 should lower it) — a causal site moves counts both ways.

**Tech Stack:** Phase 0–2b stack.

## Global Constraints
- Same seed for donor, baseline, and patched runs (aligned trajectories).
- Score the **final image** count with the detector (reliable), not the internal counter.
- Patch the up-block attn1 sites at the commitment steps found in Phase 2/2b (~6–15).
- Symmetric control (both patch directions) — asymmetric or null effect = not causal.

---

### Task 1: Patch primitives (`src/pipeline.py`)

**Files:** Modify `src/pipeline.py`, `tests/test_pipeline.py`.

**Interfaces (Produces):**
- `raw_reducer(act, cond_index=1) -> Tensor` — returns `act.detach().cpu()` (full tensor, for injection).
- `make_patch_hook(site, patch_map, state) -> hook` — forward hook that, when `state["step"] in patch_map and site in patch_map[state["step"]]`, returns the donor tensor (matched to output dtype/device), else None.
- `generate_with_patch(pipe, prompt, seed, patch_map, num_inference_steps=30) -> PIL.Image` — registers patch hooks on the sites in `patch_map`, tracks the step via `callback_on_step_end`, injects donor activations at the target steps. GPU-only.

- [ ] **Step 1: failing tests** — append to `tests/test_pipeline.py`:
```python
def test_raw_reducer_returns_full_tensor():
    from src.pipeline import raw_reducer
    a = torch.arange(6).reshape(1, 2, 3).float()
    assert raw_reducer(a).shape == (1, 2, 3)

def test_make_patch_hook_replaces_at_target_step():
    from src.pipeline import make_patch_hook
    donor = torch.ones(1, 3)
    state = {"step": 5}
    hook = make_patch_hook("x", {5: {"x": donor}}, state)
    out = torch.zeros(1, 3)
    assert torch.equal(hook(None, (out,), out), donor)     # patched at step 5
    state["step"] = 4
    assert hook(None, (out,), out) is None                 # untouched otherwise
```

- [ ] **Step 2: run to fail** — `pytest tests/test_pipeline.py -q`.

- [ ] **Step 3: implement** — add to `src/pipeline.py`:
```python
def raw_reducer(act, cond_index=1):
    return act.detach().cpu()


def make_patch_hook(site, patch_map, state):
    def hook(_m, _i, out):
        st = state["step"]
        if st in patch_map and site in patch_map[st]:
            return patch_map[st][site].to(out.dtype).to(out.device)
        return None
    return hook


def generate_with_patch(pipe, prompt, seed, patch_map, num_inference_steps=30):
    sites = {s for d in patch_map.values() for s in d}
    modmap = dict(pipe.unet.named_modules())
    state = {"step": 0}
    handles = [modmap[s].register_forward_hook(make_patch_hook(s, patch_map, state))
               for s in sites]
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

- [ ] **Step 4: run to pass** — `pytest -q`.
- [ ] **Step 5: commit** — `git commit -am "feat: causal activation-patch primitives (Phase 2c Task 1)"`

---

### Task 2: Config (`configs/phase2c.yaml`)

- [ ] Create:
```yaml
counts: [2, 5]                 # source/donor counts (schema placeholder)
objects: [cat]
seeds: [0, 1, 2, 3, 4, 5, 6, 7]
num_inference_steps: 30
score_threshold: 0.3
pairs: [[2, 5], [5, 2]]        # (source, donor): inject donor's attn1 into source
patch_steps: [6, 10, 15]       # commitment window from Phase 2/2b
patch_block: up_blocks.0       # patch this block's attn1 sites (image-side)
```
- [ ] Commit — `git commit -am "feat: phase-2c config (Task 2)"`

---

### Task 3: Notebook (`notebooks/02c_causal_patch.ipynb`)

**Files:** `notebooks/_build_02c.py` + emitted notebook.

Flow:
1. clone/install; imports (`load_sdxl`, `generate`, `catalog_attention_sites`, `select_probe_sites`, `generate_and_capture`, `raw_reducer`, `generate_with_patch`, `Detector`, `count_from_detections`, `load_config`, `yaml`).
2. read cfg + raw (`pairs`, `patch_steps`, `patch_block`).
3. `sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet)) if raw['patch_block'] in s and s.endswith('attn1')]`.
4. For each `(src, dnr)` in pairs, for each seed:
   - donor: `img_d, snaps = generate_and_capture(pipe, f'{word[dnr]} {obj}s', seed, sites, patch_steps, steps, reducer=raw_reducer)`; `c_donor = detect count`.
   - baseline: `img_b = generate(pipe, f'{word[src]} {obj}s', seed, steps)`; `c_base`.
   - patched: `img_p = generate_with_patch(pipe, f'{word[src]} {obj}s', seed, snaps, steps)`; `c_patch`.
   - record `src, dnr, seed, c_donor, c_base, c_patch`.
   (Use `src.prompts.build_prompt` for correct number words/pluralization.)
5. Save `results/phase2c_patch.csv`.
6. **Analysis:** per direction, `delta = c_patch - c_base`. Up-patch (src<dnr) expects delta>0; down-patch (src>dnr) expects delta<0. Print mean delta per direction + a paired summary; simple sign test (fraction of seeds moving the expected way).
7. **Plot:** paired scatter/line per seed (baseline → patched), colored by direction, with donor level marked. And a bar of mean delta per direction.
8. **Eyeball:** for 3 seeds, show baseline | patched | donor images with their counts.
9. **Verdict md:** delta significantly >0 for up and <0 for down → the up-block attn1 site **causally controls** the count → the realization mechanism is found; Phase 4 mitigation intervenes here. Delta ≈ 0 → correlational (like prior steering) → the count is set elsewhere (e.g., distributed / earlier noise) and we widen the patch (more steps/sites) or move upstream.

- [ ] Build, `nbformat.validate`, commit — `git commit -am "feat: phase-2c causal-patch notebook (Task 3)"`.

---

## Self-Review
- **Causal (not correlational):** injection + output-count scoring, both directions (Tasks 1, 3). ✅
- **Robust to the noisy internal counter:** scores the final image (Task 3). ✅
- **Aligned trajectories:** same seed for donor/baseline/patched (Task 3). ✅
- **Targets the localized site/steps:** up_blocks.0 attn1 at steps 6–15 (Task 2). ✅
- **Testable primitives:** raw_reducer + make_patch_hook unit-tested; generate_with_patch GPU-smoke (Task 1). ✅
- **Placeholders/types:** patch_map shape {step:{site:tensor}} consistent between generate_and_capture(raw_reducer) output and generate_with_patch input. ✅

# Design: SDXL Count-Anomaly — Root-Cause Investigation & Mitigation

**Date:** 2026-07-28
**Status:** Approved (brainstorming) — pending spec review
**Repo:** `sdxl-count-anomaly/` (remote: `T2I-Count-Anomaly`)

---

## 1. Goal & posture

Find out **why** SDXL generates the wrong number of objects for a counting prompt — with an **open, cautious, hypothesis-neutral** stance — then, only once we have a proven root cause or hard clues, design mitigation.

**No one knows the real cause yet.** This is genuine exploration. We explicitly do **not** assume the answer is the noise prior (from *Demystifying Numerosity*, arXiv 2510.11117); that paper gave us a useful *clue*, not a conclusion, and its evidence is on DiT models (FLUX/SD-3.5), not SDXL's U-Net. We treat noise, text↔image matching, text-side encoding, and "no single cause / distributed" as **competing hypotheses to discriminate between**, not a thesis to confirm.

Prior work in this project produced good initial guesses but not proof. That is the gap this cycle closes.

## 2. The organizing idea: *trace where the count signal breaks*

We reframe the problem as three yes/no questions asked at the point where the image commits to a layout:

1. **Text side** — does the fused text embedding encode the correct number N?
2. **Image side** — does the latent/U-Net side *independently* commit to N objects' worth of structure? (the crux — see the trap below)
3. **Matching / binding** — does text-N actually govern how many distinct object regions the image forms (cross-attention at the number token)?

The answers localize the cause:
- text=N, image=N, output wrong → **matching / realization failure** (the "if both understand it, it's the matching" hypothesis).
- text=N, image≠N (image locked elsewhere) → **image/noise committed without listening to text** (noise-prior story).
- text≠N → **text-side failure** (prior work argues against this: requested-N was decodable).

**The load-bearing trap (question 2).** "Requested count is decodable from mid-block activations" may be *cheating*: the number word is literally a token in the text conditioning, so a probe may read the text leak rather than a genuine image-side count. The single most important measurement in the project is showing whether the **image side itself** has (or has not) constructed N, *independent* of that text token. If the image genuinely builds N and pixels still come out wrong → matching/realization is proven, and no one has shown that for counting.

## 3. Candidate causes (co-equal until evidence decides)

| # | Hypothesis | Where | How we test it |
|---|-----------|-------|----------------|
| H1 | **Noise prior** — initial latent biases count regardless of text | image-side, upstream, early steps | seed-swap; fixed-noise/varying-text; layout stability |
| H2 | **Matching / binding** — number understood but mis-bound to spatial regions | cross-attention (attn2), instance separation | number-token attention vs #regions; attention overlap of identical instances |
| H3 | **Text-side** — number under-encoded or lost in fusion | dual text encoders + fusion | probe fused embedding for N; ablate/patch text-N |
| H4 | **Distributed / no single layer** — count emerges from layer×step interactions | across U-Net | layer×timestep probe map; is there a single break or a gradient? |
| — | Secondary/open | timing (which step commits), *identical*-object competition, VAE decode | folded into H1–H4 probes; VAE low-priority |

The truth is likely an **interaction** (e.g. weak text→image binding → image defaults to noise prior). The design is built to reveal interactions, not force a single winner.

## 4. Literature grounding

See `docs/LITERATURE_REVIEW.md` for the full synthesis + citations. Load-bearing points:
- Noise-prior is the strongest *published* causal story but DiT-only and never localized → SDXL localization is open.
- **CountGen (Make It Count, CVPR 2025)** localizes per-instance identity to SDXL **up-block self-attention ~t=500** — a concrete anchor for H2/H4 and corroborates prior Track A.
- A **matching camp exists** (Attend-and-Excite catastrophic neglect; attention-overlap → entity missing) but only for *missing objects*, never proven as the root of *miscounting* → H2 is credible and open.
- Representation format contested (continuous magnitude/subitizing vs discrete slots) → lead with monotonic/linear probes, SAE exploratory.
- Evaluate with a **detector oracle** (CountGD/GroundingDINO), not a VLM.

## 5. Scope

**In:** SDXL base U-Net; controlled single-object pure-count prompts (counts 1–7); architecture mapping; behavioral + probing + causal experiments discriminating H1–H4; a cause-targeted mitigation test. Start every phase on a **small dataset**.

**Out (this cycle):** LoRA/RLHF fine-tuning; multi-object compositional prompts; DiT models except as cited comparison; VAE cause treated as low-priority/open.

## 6. Plan — architecture-first, then discriminate; every phase starts small

Each phase gates the next scientifically, and each starts with a **small pilot** before any full run (see §8).

### Phase 0 — Understand the architecture & build shared infrastructure
- **Map SDXL end-to-end:** dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG) and how embeddings fuse; U-Net down/mid/up blocks; where **cross-attention (attn2 = text→image, the matching junction)** vs **self-attention (attn1 = image↔image)** live; noise schedule / timesteps; VAE. Output: a concise architecture reference (in README/docs) + a **catalog of candidate probe sites** (every `{block, attn-type, dim, grid, timestep}` we might read).
- **Build infrastructure in `src/`:** controlled prompt grid, SDXL generation + activation/attention capture, detector scorer, probe utilities. Unit-tested off-GPU; validated with a tiny Colab smoke run.
- **Gate:** we have a site catalog and a working, scored pipeline on a small pilot.

### Phase 1 — Behavioral characterization: how much does text move the count at all?
- **Seed-swap:** same seed × counts 1–7 × K nouns × M seeds (start small, e.g. 3×2×3). Variance-decompose *realized* (detector-scored) count into **noise-driven vs text-driven**; measure layout stability of object centers under fixed noise (replicate Demystifying's design *on SDXL*).
- **Gate / decision:** quantifies H1's strength. If count is largely noise-fixed → H1 in play; if text meaningfully moves it → H2/H3 in play. Feeds the tracing in P2. (Publishable either way: first SDXL test of the DiT-only claim.)

### Phase 2 — Trace the signal: text? image? matching? (the heart)
- **2a Text side (H3):** probe the fused text embedding for N; does number info survive fusion?
- **2b Image side (H1/H4, the crux):** across **layers × timesteps**, probe where *requested* N is present vs where *rendered* count emerges vs where they diverge — **controlling for text-token leak** (design controls so a positive read means the image side, not the text token). Find the commitment step and the break site.
- **2c Matching (H2):** number-token cross-attention maps vs the number of distinct object regions formed; instance/attention overlap for *identical* objects.
- **Gate / decision:** attributes the break to a cause (or an interaction) and pins its `{layer, timestep}`.

### Phase 3 — Representation geometry at the break
- At the localized site: is count a **continuous magnitude** (monotonic direction) or **discrete slots** (separable instance clusters)? Monotonic/linear probes + DBSCAN-style instance analysis lead; **SAE exploratory only**.
- **Gate:** determines whether mitigation should target a direction or instance features.

### Phase 4 — Cause-targeted mitigation (only after a cause is found)
- Choose the intervention to match the proven cause: noise-side edit (if H1), attention guidance/binding fix (if H2), embedding intervention (if H3). Include the training-free noise-injection idea from Demystifying **only if H1 wins**, with the caveat it needs target boxes (an oracle-layout condition, reported as such). Symmetric causal control (preserve non-count content).
- **Gate:** does the cause-matched intervention improve exact-count accuracy on SDXL without fine-tuning?

## 7. Evaluation methodology

- **Prompts (primary):** controlled grid, template `"{number-word} {object}"`, counts 1–7, K nouns (default ~10), M seeds. One category per prompt (clean confounds).
- **Scorer:** **detector oracle** — CountGD primary, GroundingDINO/OWL-ViT cross-check, human-verified subset (GeckoNum mode-of-5). **No VLM-only scoring.**
- **Metrics:** exact-count accuracy (primary), MAE, off-by-one/tolerance, per count level.
- **External anchors:** GeckoNum (1–10), GenEval-counting (2–4).

## 8. Engineering design

### Repo layout
```
sdxl-count-anomaly/
├── README.md
├── docs/{LITERATURE_REVIEW.md, specs/, plans/}
├── src/{prompts,pipeline,scoring,probes,analysis}.py
├── notebooks/  # thin Colab per phase, import src
├── configs/    # counts, nouns, seeds, layers, steps
├── results/    # summary CSVs + figures (raw artifacts gitignored)
├── tests/      # pure-logic unit tests (off-GPU)
├── requirements.txt
└── .gitignore
```

### Notebook architecture
**Thin notebooks import `src/`** (clone repo → pip install → import). All logic in `src/`, unit-testable off-GPU, identical across pilot and full runs. All SDXL runs on **Colab GPU**.

### Staged-run rule (every phase)
1. **Logic tests** local (off-GPU) — grid, math, config.
2. **Small Colab pilot** — minimal grid; confirm pipeline runs AND whether the effect appears.
3. **Full Colab run** — only after the pilot shows signal. Effort-gate on top of the scientific gates: never burn a full run on a dead hypothesis.

## 9. Risks
- **Text-leak confound (P2b):** the crux measurement; design explicit controls or the result is uninterpretable.
- **Magnitude-vs-slot:** SAE may mislead if count is continuous → probes lead.
- **Cause may be an interaction / distributed:** plan measures a layer×step map rather than assuming one culprit.
- **Noise-prior may not transfer to SDXL:** a publishable negative result; P1 detects it.
- **Detector scorer errors:** second detector + human subset.
- **Compute:** staged-run rule caps waste.

## 10. Deliverables
- Repo with one runnable Colab per phase + a short results write-up per phase.
- An evidence-backed answer to: **where does the count signal break in SDXL, and why** — and whether a cause-matched, training-free intervention helps.

## 11. Immediate next step
Invoke writing-plans for **Phase 0 (architecture + infrastructure)** first — we cannot localize a cause in a system we haven't mapped, and P0 builds the pipeline every later phase imports.

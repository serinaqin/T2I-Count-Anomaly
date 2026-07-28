# sdxl-count-anomaly

Interpreting and mitigating the **count/numerosity anomaly** in SDXL: why the number of objects in a generated image doesn't match the number requested in the prompt.

## Central question
**Why does SDXL generate the wrong number of objects?** No one knows yet. We investigate with an open, cautious, hypothesis-neutral stance — the noise-prior claim from *Demystifying Numerosity* (arXiv 2510.11117) is a **clue, not a conclusion** (and its evidence is DiT-only, not SDXL). We treat noise, text↔image **matching**, text-side encoding, and "no single cause" as **competing hypotheses to discriminate**, by tracing **where the count signal breaks** in the architecture.

Key framing: if both the text side *and* the image side understand the number but the output is still wrong, the culprit is the **matching/binding** — but we prove it, not assume it.

## Approach — architecture-first, then discriminate among causes
| Phase | Question | Method | Notebook |
|------|----------|--------|----------|
| **P0 — Architecture** | Where *could* count live/break? | Map SDXL (encoders, U-Net blocks, attn2/attn1, timesteps, VAE) + build pipeline | `notebooks/00_setup_smoke.ipynb` |
| **P1 — Behavioral** | How much does text move the count at all? | Seed-swap + variance decomposition (noise vs text) | `notebooks/01_noise_seed_swap.ipynb` |
| **P2 — Trace the signal** | Text? Image? Matching? | Probe text vs image vs cross-attn across layers × timesteps | `notebooks/02_localization.ipynb` |
| **P3 — Geometry** | Magnitude or discrete slots? | Probe geometry; SAE exploratory | `notebooks/03_geometry.ipynb` |
| **P4 — Mitigation** | Can we fix the *proven* cause? | Cause-matched intervention (noise / attention / embedding) | `notebooks/04_crossattn_mitigation.ipynb` |

Each phase gates the next, and each starts on a **small dataset** before scaling. See `docs/specs/2026-07-28-sdxl-count-anomaly-design.md` for the full design and `docs/LITERATURE_REVIEW.md` for the grounding literature.

## How it runs
- **Logic** lives in `src/` and is unit-tested off-GPU (`tests/`).
- **Notebooks** are thin drivers: clone repo → `pip install -r requirements.txt` → import `src`. All SDXL runs happen on **Colab GPU**.
- **Staged runs:** local logic tests → small Colab pilot → full Colab run (only after the pilot shows signal).

## Evaluation
- Primary benchmark: controlled grid `"{number-word} {object}"`, counts 1–7.
- Scorer: **detector oracle** (CountGD / GroundingDINO) + human-verified subset. No VLM-only counting.
- Metrics: exact-count accuracy, MAE, off-by-one. External anchors: GeckoNum, GenEval-counting.

## Status
Design approved 2026-07-28. Next: Phase 1 implementation plan.

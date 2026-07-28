# Count/Numerosity Anomaly in Text-to-Image Diffusion Models — Literature Synthesis

_Compiled 2026-07-28 for the SDXL count-anomaly interpretability project. Sources gathered via three parallel research passes (mechanisms / mitigations / benchmarks) + direct read of the Demystifying paper (arXiv 2510.11117v1)._

---

## 0. The empirical shape of the failure (what any mechanism must explain)
Accuracy is high at count=1, declines monotonically, collapses past ~3–4 objects — roughly the human **subitizing** limit. No SOTA model (SDXL, SD3.5, FLUX, DALL·E 3, GPT-4o) exceeds ~50% exact-count accuracy on clean benchmarks. Scaling model/data does NOT fix it. Prompt refinement does NOT help (T2ICountBench). This matches the project's own clean single-object run (baseline ~30%, monotonic decline 1→7).

## 1. Explanation A — the initial NOISE PRIOR determines count (strongest causal story)
- **Demystifying Numerosity (arXiv 2510.11117, PKU+MSRA, 2025, under review).** Claim: "noise determines layout, text activates locations." Fixed-noise/varying-text experiment (100 noise × 3000 prompts = 300k imgs on FLUX): a given noise vector produces near-identical positions and often the same count regardless of prompt; K-means object centers >90% stable across requested counts. Scaling (1K→500K data, 2.5B→12B params) plateaus <23% exact. Remedy = inject count-aware layout into noise inside N bounding boxes (Uniform-Scaled γ=0.1 / Fixed / Gaussian ω=0.3,α=0.8) + LoRA fine-tune (rank16, 512px, 10k steps). GrayCount250 20.0%→85.3% exact (MAE 4.30→0.19); NaturalCount6 74.8%→86.3%. **CAUSAL.**
  - **CRITICAL for us:** evidence is on **FLUX.1-dev + SD-3.5 = DiT models, NOT SDXL's U-Net.** Transfer to SDXL is an assumption to test → seed-swap Phase 1 = novel SDXL replication. Remedy is **NOT training-free** (needs boxes at inference + LoRA). Paper does **NOT** localize to any layer/timestep (App. K: failed to remove noise bias via training/inference tweaks) → **layer/timestep localization on SDXL is wide open.**
- **Corroboration:** Be Decisive (ECCV 2025, noise-induced layouts); Reliable Random Seeds (arXiv 2411.18810, seed alone swings numeric-composition 28.3%→41.0%); IMAGHarmony (init-noise sensitivity). Four independent groups converge.

## 2. Explanation B — cross-attention / where object identity lives
- **Make It Count / CountGen (CVPR 2025, arXiv 2406.10210) — MOST RELEVANT.** On **SDXL**: PCA over self-attention features shows most layers do NOT separate instances, but **up-block layer "l52_up" self-attention at timestep t=500** produces distinct per-instance features; DBSCAN-cluster them to count/localize mid-denoising, then correct. Count = **discrete separable object-identity feature in a specific mid-network up-block layer at a mid/early denoise step.** Base model SDXL, open code. → Directly corroborates the project's Track A up-block cross/self-attn localization and the "capture activations early" pivot.
- **Attend-and-Excite (SIGGRAPH 2023):** cross-attention activation strength gates whether an object renders ("catastrophic neglect") — underlies undercounting. CAUSAL.
- **Attention Overlap = entity missing (arXiv 2410.20972):** overlapping cross-attn maps suppress the weaker instance → identical instances compete for the same region (undercount).
- **CountSteer (arXiv 2511.11253, 2025):** diffusion models carry an INTERNAL cross-attention signal that shifts with count-correctness; steering → +~4%. Evidence a count-correctness signal exists internally (complicates pure "noise decides all").
- **ATHENA (arXiv 2603.19676, 2026):** estimates count from intermediate reps during sampling, applies count-aware noise corrections EARLY. Endorses "count set early."
- **eDiff-I (2022):** layout/semantics form at high-noise EARLY steps; text has ~no effect in last ~7%. Canonical timing evidence.
- **DAAM / Prompt-to-Prompt:** cross-attention = word→region attribution; manipulating it causally controls layout.

## 3. Explanation C — text encoder (CLIP) cannot represent number (competing camp)
- **Teaching CLIP to Count to Ten (ICCV 2023):** CLIP fails to encode counting; counting-contrastive fine-tune + conditioning Imagen on it improves count fidelity. CAUSAL. Introduces CountBench.
- Can CLIP Count Stars? (EMNLP-F 2024, quantity bias); Kamath (EMNLP 2023, text-encoder bottleneck); BoW-cross-modal debate (arXiv 2502.03566, says issue is cross-modal alignment not encoder alone).
- **CONFLICT (A vs C):** noise-prior camp says bottleneck is downstream of text; CLIP camp says encoder fails. Plausible unproven synthesis: weak text numeric signal → U-Net defaults count to noise prior. Project's own result (requested-N decodable at mid-block ~1.0) already argues comprehension is NOT the bottleneck → leans toward A over C.

## 4. Explanation D — representation format: continuous MAGNITUDE vs discrete SLOTS (contested; key for SAE choice)
- **Continuous-magnitude/ANS + subitizing:** Counting Circuits (arXiv 2603.18523, LVLM activation patching, quantity in continuous latent magnitude, only 5.5% heads counting-critical); Nasr 2019/2021 & Kim 2021 (numerosity neurons, log/Weber tuning, emerge even untrained).
- **Discrete-slot/object-file:** Make It Count (separable instance features).
- **Skeptics:** Zhang & Wu 2020 and Kajić & Nematzadeh 2023 (selectivity is small-sample artifact; fails to generalize OOD).
- **Implication for us:** if count is continuous magnitude, a **sparse one-hot-ish SAE feature may be the wrong tool** (a linear/monotonic probe direction fits better) — matches the project's prior "slot NOT a linear signal; ~20 distributed latents; hotspot-vs-N r=0.41 driven only by count=7." Choose probe geometry accordingly. The universal collapse past 3–4 = subitizing signature.

## 5. Explanation E — VAE / decoder / resolution
Unestablished. No paper isolates VAE/resolution as a primary cause. Treat as open, low-priority.

## 6. Explanation F — interpretability tooling (SAEs) touching count
**No published SAE work has isolated a validated count/numerosity feature in a diffusion model or CLIP.** One-Step-is-Enough (SDXL-Turbo SAEs, steerable, no count feature); SAeUron (unlearning, no count); CLIP/ViT SAE line finds shape/color/style/semantic ("trio","three" as semantic concepts, explicitly NOT cardinality). → **The SAE-count intersection is genuinely open** = the project's contribution opportunity AND its central methodological risk (see §4).

## 7. Mitigations ranked for a Colab-scale SDXL project
1. **CountGen / Make It Count** — CVPR 2025, SDXL-native, open code, 29%→48%. Best reproduction target + interpretability hook (self-attn instance features).
2. **Demystifying noise-injection** — isolate the training-free noise component; test on SDXL as ablation (headline gains lean on LoRA + synthetic data).
3. **D2D (arXiv 2510.19278)** — training-free detector-to-differentiable critic optimizing initial noise; +13.7%; turbo-model-friendly.
4. Counting Guidance (WACV 2025), CountCluster (training-free, weak evidence), Iterative Count Optimization (WACV 2026).
5. Layout/box conditioning (GLIGEN/R&B) = oracle upper bound (changes the problem).
6. RLHF/DPO/fine-tune counting reward = heaviest, poor Colab fit, related-work only.
- Prompt refinement = confirmed negative baseline.

## 8. Benchmarks & evaluation
- **No off-the-shelf benchmark fits single-object pure-count 1–7.** GenEval-counting 2–4; CountBench/CoCoCount start at 2 (no "1"); T2I-CompBench & GeckoNum mix multi-object/attribute. **Count=1 absent from every automated split.**
- **Recommendation:** custom controlled grid ({number-word}{object}, counts 1–7, K nouns × M seeds, single template) = PRIMARY instrument (clean confounds for interpretability). Borrow scoring: **CountGD** open-world detector as oracle (~99.8% exact on clean synthetic) + GroundingDINO/OWL-ViT cross-check + **human-verified subset (GeckoNum mode-of-5 protocol)**. **Avoid VLM-only counting** (VLMs weak/prior-biased past ~4) — CHANGE from prior Qwen-VL scorer. Metrics: exact-match (primary), MAE, off-by-one/tolerance.
- **External validation anchors:** GeckoNum (1–10, human-verified), GenEval-counting (2–4).

## 9. Net implications for the project
1. Phase-1 seed-swap is a **novel SDXL replication** of a DiT-only finding — publishable either way.
2. **Localization is wide open** — the paper explicitly didn't do it; CountGen gives a concrete SDXL anchor (up-block self-attn, t≈500 / early step).
3. Capture activations **early/high-noise** (eDiff-I, ATHENA, CountGen t=500) — matches the prior pivot.
4. **Rethink SAE-vs-probe:** count may be continuous magnitude → prefer monotonic/linear probes + magnitude analysis over one-hot SAE latents; SAE stays exploratory.
5. **Swap scorer to a detector (CountGD)**, not a VLM.
6. There IS an internal count-correctness signal (CountSteer) — steering isn't hopeless but is weak (~+4%); noise-side interventions are the strong lever.
7. SAE-count feature discovery = genuine open contribution (with magnitude-vs-slot risk noted).

---

## Source list (title — venue/year — URL)

**Noise-prior**
- Demystifying Numerosity in Diffusion Models — arXiv 2025 — https://arxiv.org/abs/2510.11117
- Be Decisive: Noise-Induced Layouts for Multi-Subject Generation — ECCV 2025 — https://arxiv.org/abs/2505.21488
- Enhancing Compositional T2I with Reliable Random Seeds — arXiv 2024 — https://arxiv.org/abs/2411.18810
- IMAGHarmony — arXiv 2025 — https://arxiv.org/abs/2506.01949

**Cross-attention / counting methods / localization**
- Attend-and-Excite — SIGGRAPH 2023 — https://arxiv.org/abs/2301.13826
- Attention Overlap Is Responsible for the Entity Missing Problem — arXiv 2024 — https://arxiv.org/abs/2410.20972
- Make It Count (CountGen) — CVPR 2025 — https://arxiv.org/abs/2406.10210 (code: https://github.com/Litalby1/make-it-count)
- CountCluster — arXiv 2025 — https://arxiv.org/abs/2508.10710
- CountSteer — arXiv 2025 — https://arxiv.org/abs/2511.11253
- ATHENA — arXiv 2026 — https://arxiv.org/abs/2603.19676
- Iterative Object Count Optimization — WACV 2026 — https://arxiv.org/abs/2408.11721
- Counting Guidance — WACV 2025 — https://arxiv.org/abs/2306.17567
- D2D: Detector-to-Differentiable Critic — arXiv 2025 — https://arxiv.org/abs/2510.19278
- eDiff-I — arXiv 2022 — https://arxiv.org/abs/2211.01324
- DAAM — EMNLP 2023 — https://arxiv.org/abs/2210.04885

**Text encoder / CLIP numerosity**
- Teaching CLIP to Count to Ten (+CountBench) — ICCV 2023 — https://arxiv.org/abs/2302.12066
- Can CLIP Count Stars? — Findings of EMNLP 2024 — https://arxiv.org/abs/2409.15035
- Text encoders bottleneck compositionality in contrastive VLMs — EMNLP 2023 — https://arxiv.org/abs/2305.14897
- CLIP Behaves like a Bag-of-Words Cross-modally — arXiv 2025 — https://arxiv.org/abs/2502.03566

**Magnitude vs slots / numerosity neurons**
- Counting Circuits (LVLM) — arXiv 2026 — https://arxiv.org/abs/2603.18523
- Number detectors spontaneously emerge (Nasr) — Science Advances 2019 — https://www.science.org/doi/10.1126/sciadv.aav7903
- Numerosity zero (Nasr & Nieder) — iScience 2021 — https://pmc.ncbi.nlm.nih.gov/articles/PMC8571726/
- Visual number sense in untrained DNNs (Kim) — Science Advances 2021 — https://www.science.org/doi/full/10.1126/sciadv.abd6127
- On Numerosity of DNNs (critique) — arXiv 2020 — https://arxiv.org/abs/2011.08674
- Evaluating Visual Number Discrimination in DNNs (critique) — arXiv 2023 — https://arxiv.org/abs/2303.07172

**Benchmarks**
- GeckoNum — NeurIPS 2024 — https://arxiv.org/abs/2406.14774 (repo: https://github.com/google-deepmind/geckonum_benchmark_t2i)
- T2ICountBench — arXiv 2025 — https://arxiv.org/abs/2503.06884
- Your VLM Can't Even Count to 20 — arXiv 2025 — https://arxiv.org/abs/2510.04401
- GenEval — NeurIPS 2023 — https://arxiv.org/abs/2310.11513 (repo: https://github.com/djghosh13/geneval)
- T2I-CompBench / ++ — NeurIPS 2023 / TPAMI 2025 — https://arxiv.org/abs/2307.06350

**SAE / interpretability tooling**
- One-Step is Enough: SAEs for T2I Diffusion (SDXL Turbo) — arXiv 2024/25 — https://arxiv.org/abs/2410.22366
- SAeUron — arXiv 2025 — https://arxiv.org/abs/2501.18052
- Steering CLIP's ViT with SAEs — arXiv 2025 — https://arxiv.org/abs/2504.08729
- Interpreting CLIP with Hierarchical (Matryoshka) SAEs — arXiv 2025 — https://arxiv.org/abs/2502.20578
- Causal Interpretation of SAE Features in Vision (CaFE) — arXiv 2025 — https://arxiv.org/abs/2509.00749
- PatchSAE — ICLR 2025 — https://arxiv.org/abs/2412.05276

_Caveat: several 2026-dated arXiv IDs (Counting Circuits, ATHENA, CountSteer) are very recent preprints; verify quantitative details against final PDFs before citing verbatim. Demystifying hyperparameters (γ/ω/α, LoRA config) read from HTML — spot-check PDF._

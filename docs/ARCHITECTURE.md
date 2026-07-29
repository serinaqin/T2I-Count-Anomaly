# SDXL Architecture Reference (count-signal candidate sites)

_Working map of where the object-count signal could live or break in SDXL. Filled in progressively; the site catalog is pasted from the Phase 0 smoke run._

## Text pathway
- Two text encoders: **CLIP ViT-L/14** (`text_encoder`) + **OpenCLIP ViT-bigG/14** (`text_encoder_2`).
- Their penultimate hidden states are **concatenated** → the conditioning fed to U-Net cross-attention. `text_encoder_2` also provides a **pooled** embedding (added, with time/size embeddings, to the timestep embedding).
- Candidate cause **H3 (text-side):** does the number survive encoding + fusion? Probe the fused embedding for N.

## U-Net (where text meets image)
- Blocks: `down_blocks[0..2]`, `mid_block`, `up_blocks[0..2]`.
- Each transformer block carries **`attn1`** (self-attention, image↔image — per CountGen this is where per-instance identity separates) and **`attn2`** (cross-attention, text→image — **the matching junction**, candidate cause **H2**).
- Prior Track-A localization (this project): `up_blocks[0].attentions[1]` cross-attn (1280-dim, 32×32 grid).
- CountGen (Make It Count, CVPR 2025) anchor: an **up-block self-attention** layer at timestep **t≈500** produces DBSCAN-separable per-instance features.
- Candidate cause **H4 (distributed):** count may emerge from layer×step interactions rather than one site — the Phase 2 probe map tests this.

## Timesteps
- Layout/semantics form **early / high-noise** (eDiff-I: text has ~no effect in the last ~7% of steps). Capture emphasis: first ~20% of denoising.
- Candidate cause **H1 (noise prior):** initial latent may fix layout/count before text acts.

## VAE
- Latent 128×128×4 → 1024×1024 RGB decode. Deprioritized as a cause (no literature support); revisit only if evidence points here.

## Candidate site catalog (filled from smoke run)
Run `notebooks/00_setup_smoke.ipynb` cell 5; it prints `catalog_attention_sites(pipe.unet)` and writes `results_sites.txt`. Paste the full list here (expected ~140 `...attn1` / `...attn2` module names across down/mid/up blocks):

```
<paste catalog_attention_sites(pipe.unet) output here after the Colab smoke run>
```

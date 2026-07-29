"""Generate notebooks/01_noise_seed_swap.ipynb (Phase 1 — seed-swap gate).

Measures how much of SDXL's realized object count is driven by the initial
noise (seed) vs the text prompt. Thin Colab driver over src/.

Run: python notebooks/_build_01.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Phase 1 — Noise Seed-Swap (the gate)\n"
    "**Question:** does the initial *noise* (seed) or the *text* prompt decide "
    "how many objects SDXL generates?\n\n"
    "We cross the SAME seeds against every (count, object) prompt, score with "
    "the detector, then decompose the realized count into noise vs text.\n\n"
    "**Runtime:** GPU."))

c.append(nbf.v4.new_code_cell(
    "import os\n"
    "if not os.path.exists('src'):\n"
    "    !git clone https://github.com/serinaqin/T2I-Count-Anomaly.git\n"
    "    %cd T2I-Count-Anomaly\n"
    "!pip install -q -r requirements.txt\n"
    "!pip install -q pytest groundingdino-py"))

c.append(nbf.v4.new_code_cell(
    "import sys; sys.path.insert(0, '.')\n"
    "import pandas as pd, os\n"
    "from src.prompts import generate_grid\n"
    "from src.pipeline import load_sdxl, generate\n"
    "from src.detector import Detector\n"
    "from src.scoring import count_from_detections\n"
    "from src.config import load_config\n"
    "from src.analysis import (text_responsiveness, count_variance_decomposition,\n"
    "                          per_seed_summary, flag_degenerate)"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/phase1.yaml')\n"
    "grid = generate_grid(cfg.counts, cfg.objects, cfg.seeds)\n"
    "print(len(grid), 'images:', len(cfg.counts), 'counts x',\n"
    "      len(cfg.objects), 'objects x', len(cfg.seeds), 'seeds')"))

c.append(nbf.v4.new_code_cell(
    "pipe = load_sdxl()\n"
    "det = Detector()"))

c.append(nbf.v4.new_code_cell(
    "# Generate the full seed-swap grid and score each image with the detector.\n"
    "rows = []\n"
    "for i, p in enumerate(grid):\n"
    "    img = generate(pipe, p.text, p.seed, cfg.num_inference_steps)\n"
    "    dets = det.detect(img, [p.obj])\n"
    "    n = count_from_detections(dets, p.obj, cfg.score_threshold)\n"
    "    rows.append({'obj': p.obj, 'count': p.count, 'seed': p.seed,\n"
    "                 'realized_count': n})\n"
    "    if (i + 1) % 20 == 0:\n"
    "        print(f'{i+1}/{len(grid)}')\n"
    "df = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/phase1_counts.csv', index=False)\n"
    "df.head()"))

c.append(nbf.v4.new_code_cell(
    "# Flag collage/degenerate blow-ups (correct counts, pathological images)\n"
    "df = flag_degenerate(df, max_requested=max(cfg.counts))\n"
    "print('degenerate images:', int(df['degenerate'].sum()), 'of', len(df))\n"
    "df_clean = df[~df['degenerate']].copy()"))

c.append(nbf.v4.new_markdown_cell(
    "## Verdict metrics\n"
    "- **text slope** ~1 = text controls the count; ~0 = text ignored.\n"
    "- **variance decomposition** (eta^2): how much of the realized count "
    "aligns with seed vs count vs object.\n"
    "- **per-seed summary**: a seed with low std across prompts has a fixed "
    "'preferred count' (noise-driven)."))

c.append(nbf.v4.new_code_cell(
    "print('text responsiveness:', text_responsiveness(df_clean))\n"
    "print('variance decomposition:', count_variance_decomposition(df_clean))\n"
    "per_seed_summary(df_clean)"))

c.append(nbf.v4.new_code_cell(
    "# Money plot A: per-seed count response\n"
    "import matplotlib.pyplot as plt\n"
    "piv = df_clean.groupby(['seed', 'count'])['realized_count'].mean().reset_index()\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "for s, g in piv.groupby('seed'):\n"
    "    ax.plot(g['count'], g['realized_count'], marker='o', alpha=0.6,\n"
    "            label=f'seed {s}')\n"
    "lims = [min(cfg.counts), max(cfg.counts)]\n"
    "ax.plot(lims, lims, 'k--', label='y=x (perfect text control)')\n"
    "ax.set_xlabel('requested count'); ax.set_ylabel('mean realized count')\n"
    "ax.set_title('Per-seed response (flat = noise-fixed, diagonal = text-controlled)')\n"
    "ax.legend(fontsize=7); plt.tight_layout()\n"
    "plt.savefig('results/phase1_perseed.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_code_cell(
    "# Money plot B: variance decomposition\n"
    "dec = count_variance_decomposition(df_clean)\n"
    "fig, ax = plt.subplots(figsize=(4, 4))\n"
    "ax.bar(list(dec.keys()), list(dec.values()))\n"
    "ax.set_ylabel('variance explained (eta^2)'); ax.set_ylim(0, 1)\n"
    "ax.set_title('What drives the realized count?')\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase1_variance.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **Flat per-seed lines + seed eta^2 >> count eta^2 + slope ~ 0** -> the "
    "NOISE decides the count on SDXL. Phase 2 then localizes *where/when* the "
    "noise's count signal is read out in the U-Net.\n"
    "- **Diagonal lines + count eta^2 >> seed eta^2 + slope ~ 1** -> the TEXT "
    "controls the count and the noise-prior story does NOT transfer to SDXL; "
    "Phase 2 pivots to the text/cross-attention (matching) pathway.\n"
    "- **In between** -> both matter; the relative eta^2 tells us how to weight "
    "the Phase 2 probes."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "01_noise_seed_swap.ipynb")
nbf.write(nb, out)
print("wrote", out)

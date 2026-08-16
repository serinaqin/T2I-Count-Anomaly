"""Generate notebooks/04_count_steering.ipynb (Phase 4 - mitigation).

Learns a donor-free 'more-objects' direction in the early up-block self-attention
(the Phase 2c causal site) and adds it during generation. Dose-response of the
rendered count vs steering strength = a usable count knob. Thin Colab driver.

Run: python notebooks/_build_04.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Phase 4 - Count-direction steering (mitigation)\n"
    "Phase 2c proved up_blocks.0 self-attention at early steps causally sets the "
    "count. Here we learn a **donor-free** 'more-objects' direction there "
    "(mean high-count - mean low-count activations) and ADD it during "
    "generation, sweeping the strength alpha. A monotonic rise of the rendered "
    "count vs alpha = a usable control knob.\n\n"
    "This is NOT the project's prior failed steering (that was mid/late, "
    "cross-attn, correlational) - it steers the proven causal site.\n\n"
    "**Runtime:** GPU (~20 min)."))

c.append(nbf.v4.new_code_cell(
    "import os\n"
    "if not os.path.exists('src'):\n"
    "    !git clone https://github.com/serinaqin/T2I-Count-Anomaly.git\n"
    "    %cd T2I-Count-Anomaly\n"
    "!pip install -q -r requirements.txt\n"
    "!pip install -q pytest groundingdino-py"))

c.append(nbf.v4.new_code_cell(
    "import sys; sys.path.insert(0, '.')\n"
    "import numpy as np, pandas as pd, os, yaml, torch\n"
    "import matplotlib.pyplot as plt\n"
    "from src.prompts import build_prompt, generate_grid\n"
    "from src.pipeline import (load_sdxl, generate, catalog_attention_sites,\n"
    "                          select_probe_sites, generate_and_capture,\n"
    "                          generate_with_steer)\n"
    "from src.probes import count_direction\n"
    "from src.detector import Detector\n"
    "from src.scoring import count_from_detections\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/phase4.yaml')\n"
    "raw = yaml.safe_load(open('configs/phase4.yaml'))\n"
    "steer_steps, alphas = raw['steer_steps'], raw['alphas']\n"
    "block, attn = raw['patch_block'], raw['patch_attn']\n"
    "obj = cfg.objects[0]\n"
    "pipe = load_sdxl(); det = Detector()\n"
    "sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet))\n"
    "         if block in s and s.endswith(attn)]\n"
    "def cnt(img):\n"
    "    return count_from_detections(det.detect(img, [obj]), obj, cfg.score_threshold)\n"
    "print('steer sites:', sites)"))

c.append(nbf.v4.new_code_cell(
    "# TRAIN: capture pooled early self-attn + rendered count across the grid.\n"
    "grid = generate_grid(cfg.counts, cfg.objects, cfg.seeds)\n"
    "feats = {st: {s: [] for s in sites} for st in steer_steps}\n"
    "rendered = []\n"
    "for i, p in enumerate(grid):\n"
    "    img, snaps = generate_and_capture(pipe, p.text, p.seed, sites, steer_steps,\n"
    "                                      cfg.num_inference_steps)  # default pool reducer\n"
    "    rendered.append(cnt(img))\n"
    "    for st in steer_steps:\n"
    "        for s in sites:\n"
    "            feats[st][s].append(snaps.get(st, {}).get(s))\n"
    "    if (i + 1) % 10 == 0: print(f'{i+1}/{len(grid)}')\n"
    "rendered = np.array(rendered)\n"
    "print('captured', len(grid), 'train images; rendered range', rendered.min(), rendered.max())"))

c.append(nbf.v4.new_code_cell(
    "# DIRECTION: mean(high-count) - mean(low-count) per (step, site).\n"
    "directions = {}\n"
    "for st in steer_steps:\n"
    "    directions[st] = {}\n"
    "    for s in sites:\n"
    "        X = np.array([v for v in feats[st][s] if v is not None])\n"
    "        d = count_direction(X, rendered)\n"
    "        directions[st][s] = torch.tensor(d, dtype=torch.float32)\n"
    "print('built directions for', len(steer_steps), 'steps x', len(sites), 'sites')"))

c.append(nbf.v4.new_code_cell(
    "# DOSE-RESPONSE: sweep alpha on held-out eval prompts.\n"
    "rows = []\n"
    "for a in alphas:\n"
    "    for cnt_req in raw['eval_counts']:\n"
    "        prompt = build_prompt(cnt_req, obj)\n"
    "        for seed in raw['eval_seeds']:\n"
    "            img = generate_with_steer(pipe, prompt, seed, directions, float(a),\n"
    "                                      steer_steps, cfg.num_inference_steps)\n"
    "            rows.append({'alpha': a, 'req': cnt_req, 'seed': seed, 'rendered': cnt(img)})\n"
    "    print('alpha', a, 'done')\n"
    "dfd = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "dfd.to_csv('results/phase4_dose.csv', index=False)\n"
    "dfd.groupby('alpha')['rendered'].mean()"))

c.append(nbf.v4.new_code_cell(
    "# Plot mean rendered count vs alpha (monotonic rise = a real count knob).\n"
    "m = dfd.groupby('alpha')['rendered'].mean()\n"
    "se = dfd.groupby('alpha')['rendered'].sem()\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "ax.errorbar(m.index, m.values, yerr=se.values, marker='o', capsize=3)\n"
    "ax.set_xlabel('steering strength alpha'); ax.set_ylabel('mean rendered count')\n"
    "ax.set_title('Dose-response: does the count direction control the count?')\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase4_dose.png', dpi=100, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_code_cell(
    "# Eyeball: one eval prompt at alpha = min, 0, max.\n"
    "seed0, req0 = raw['eval_seeds'][0], raw['eval_counts'][0]\n"
    "prompt0 = build_prompt(req0, obj)\n"
    "trio = [min(alphas), 0, max(alphas)]\n"
    "fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))\n"
    "for j, a in enumerate(trio):\n"
    "    img = generate_with_steer(pipe, prompt0, seed0, directions, float(a),\n"
    "                              steer_steps, cfg.num_inference_steps)\n"
    "    axes[j].imshow(img); axes[j].axis('off')\n"
    "    axes[j].set_title(f\"'{prompt0}' | alpha={a} | count={cnt(img)}\", fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase4_eyeball.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **Mean rendered count rises monotonically with alpha** = the learned "
    "direction is a real, donor-free **count knob** -> we can steer the count at "
    "the causal site without a donor. Mitigation then = calibrate alpha per "
    "requested count (next step). And the eyeball should show coherently more "
    "animals as alpha grows.\n"
    "- **Flat / non-monotonic** = a single pooled channel-direction isn't enough "
    "(the count needs spatial structure, which pooling discarded). Fallback: "
    "inject a per-count spatial TEMPLATE (averaged donor activations) instead of "
    "a pooled vector - the donor patch worked, so a template bank should too.\n"
    "- Either outcome is informative: it tells us whether the count is a linear "
    "direction (steerable) or a spatial pattern (template-only) at the causal site."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "04_count_steering.ipynb")
nbf.write(nb, out)
print("wrote", out)

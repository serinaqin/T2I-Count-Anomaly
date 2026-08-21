"""Generate notebooks/exp_spatialprobe.ipynb (spatial vs pooled probe, fixed).

Does the rendered count decode better from a SPATIAL readout of the up-block
self-attention than from a global/pooled one? We compare, all with
cross-validated ridge R^2 (stable at small n): (a) pooled channel vector, (b)
grid-pooled raw energy at increasing resolution, and (c) the nonlinear
peak-count spatial readout. Thin Colab driver.

Run: python notebooks/_build_exp_spatialprobe.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Spatial vs pooled decoding of the rendered count\n"
    "Is the count carried by the **spatial layout** of up0 self-attention rather "
    "than a global summary? We decode the rendered count three ways, all with "
    "**5-fold cross-validated ridge $R^2$** (stable at small $n$; ~0 when "
    "uninformative, not wildly negative):\n"
    "1. **pooled** channel vector (mean over tokens) - global/linear baseline;\n"
    "2. **grid** raw-energy at resolution $g$ (spatial, linear);\n"
    "3. **peak-count** - number of saliency peaks (spatial, nonlinear readout).\n\n"
    "**Runtime:** GPU (~15 min)."))

c.append(nbf.v4.new_code_cell(
    "import os\n"
    "if not os.path.exists('src'):\n"
    "    !git clone https://github.com/serinaqin/T2I-Count-Anomaly.git\n"
    "    %cd T2I-Count-Anomaly\n"
    "else:\n"
    "    !git pull   # get latest src fixes (NOTE: still Restart Runtime to reload imported modules)\n"
    "!pip install -q -r requirements.txt\n"
    "!pip install -q pytest groundingdino-py"))

c.append(nbf.v4.new_code_cell(
    "import sys; sys.path.insert(0, '.')\n"
    "import numpy as np, pandas as pd, os, yaml\n"
    "import matplotlib.pyplot as plt\n"
    "from src.prompts import generate_grid\n"
    "from src.pipeline import (load_sdxl, catalog_attention_sites,\n"
    "                          select_probe_sites, generate_and_capture,\n"
    "                          raw_reducer, pool_activation)\n"
    "from src.spatial import featuremap_saliency, grid_pool_2d, count_peaks\n"
    "from src.probes import cv_r2\n"
    "from src.detector import Detector\n"
    "from src.scoring import count_from_detections\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/exp_spatialprobe.yaml')\n"
    "raw = yaml.safe_load(open('configs/exp_spatialprobe.yaml'))\n"
    "step = raw['capture_step']; grid_sizes = raw['grid_sizes']\n"
    "block, attn = raw['patch_block'], raw['patch_attn']\n"
    "pipe = load_sdxl(); det = Detector()\n"
    "sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet))\n"
    "         if block in s and s.endswith(attn)]\n"
    "def cnt(img, obj):\n"
    "    return count_from_detections(det.detect(img, [obj]), obj, cfg.score_threshold)\n"
    "print('sites:', sites, '| capture step', step)"))

c.append(nbf.v4.new_code_cell(
    "# Capture RAW activations; derive both a pooled channel vector and a raw\n"
    "# (non-normalized) energy map per image, averaged over the 3 up0 attn1 sites.\n"
    "grid = generate_grid(cfg.counts, cfg.objects, cfg.seeds)\n"
    "pooled_list, sal_list, rows = [], [], []\n"
    "for i, p in enumerate(grid):\n"
    "    img, snaps = generate_and_capture(pipe, p.text, p.seed, sites, [step],\n"
    "                                      cfg.num_inference_steps, reducer=raw_reducer)\n"
    "    acts = [snaps[step][s] for s in sites if s in snaps.get(step, {})]\n"
    "    pooled_list.append(np.mean([pool_activation(a) for a in acts], axis=0))          # (C,)\n"
    "    sal_list.append(np.mean([featuremap_saliency(a, normalize=False) for a in acts], axis=0))  # (H,W) raw\n"
    "    rows.append({'obj': p.obj, 'count': p.count, 'seed': p.seed, 'rendered': cnt(img, p.obj)})\n"
    "    if (i + 1) % 20 == 0: print(f'{i+1}/{len(grid)}')\n"
    "df = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/exp_spatialprobe_counts.csv', index=False)\n"
    "print('captured', len(grid), 'maps of shape', sal_list[0].shape)"))

c.append(nbf.v4.new_code_cell(
    "# PRIMARY (robust): Spearman rank correlation of a GLOBAL vs a SPATIAL scalar\n"
    "# readout with the rendered count. Rank-based -> immune to car-style count\n"
    "# outliers; no CV folds / no overfitting.\n"
    "from scipy.stats import spearmanr\n"
    "y = df['rendered'].to_numpy(float)\n"
    "energy = np.array([s.mean() for s in sal_list])                    # global (amount)\n"
    "pc = np.array([count_peaks(s, thresh_rel=0.5) for s in sal_list])  # spatial (multiplicity)\n"
    "rho_energy = float(spearmanr(energy, y).correlation)\n"
    "rho_peak = float(spearmanr(pc, y).correlation)\n"
    "print(f'GLOBAL energy   Spearman rho = {rho_energy:.3f}')\n"
    "print(f'SPATIAL peak-ct Spearman rho = {rho_peak:.3f}')\n"
    "# SECONDARY: shuffled-CV linear probes (sensitive to count outliers; report with care)\n"
    "pooled_r2 = cv_r2(np.array(pooled_list), y)\n"
    "grid_r2 = {g: cv_r2(np.array([grid_pool_2d(s, g) for s in sal_list]), y) for g in grid_sizes}\n"
    "print('(secondary) pooled cv_R2 =', round(pooled_r2, 3), '| grid cv_R2 =',\n"
    "      {g: round(v, 3) for g, v in grid_r2.items()})\n"
    "pd.DataFrame([{'readout': 'global energy', 'spearman_rho': rho_energy},\n"
    "              {'readout': 'spatial peak-count', 'spearman_rho': rho_peak}]\n"
    "             ).to_csv('results/exp_spatialprobe_rho.csv', index=False)"))

c.append(nbf.v4.new_code_cell(
    "# PRIMARY plot: spatial vs global readout (Spearman rho).\n"
    "fig, ax = plt.subplots(figsize=(5.5, 4.5))\n"
    "ax.bar(['global\\nenergy', 'spatial\\npeak-count'], [rho_energy, rho_peak],\n"
    "       color=['gray', 'C2'])\n"
    "ax.axhline(0, color='k', lw=0.6); ax.set_ylim(-0.2, 1.0)\n"
    "ax.set_ylabel('Spearman rho with rendered count')\n"
    "ax.set_title('Is the count in the spatial layout or just the amount?')\n"
    "for i, v in enumerate([rho_energy, rho_peak]):\n"
    "    ax.text(i, v + 0.02, f'{v:.2f}', ha='center')\n"
    "plt.tight_layout(); plt.savefig('results/exp_spatialprobe_rho.png', dpi=110, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "The **Spearman rho bars are the load-bearing result** (rank-based -> robust "
    "to the car-style count outliers and to fold/overfitting artifacts that broke "
    "the linear R^2 in earlier versions).\n"
    "- **peak-count rho notably > energy rho** = the count is carried by spatial "
    "*multiplicity* (how many blobs), beyond mere total activation *amount* -> "
    "supports the spatial object-layout framing.\n"
    "- **energy rho comparable/higher** = the count tracks mostly the *amount* of "
    "activation, not a distinct spatial-multiplicity code; report honestly.\n"
    "- **Both rho small (e.g., <0.3)** = the rendered count is only weakly "
    "decodable here at all -> this is the point: decodability is weak while the "
    "causal patch (Exp #2) is strong, i.e. *decodability != causality*.\n"
    "- The secondary cv_R2 numbers are informative but count-outlier sensitive; do "
    "not over-read them."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "exp_spatialprobe.ipynb")
nbf.write(nb, out)
print("wrote", out)

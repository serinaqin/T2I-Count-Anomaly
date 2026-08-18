"""Generate notebooks/exp_spatialprobe.ipynb (spatial vs pooled probe).

Positive evidence for the 'spatial object-layout' claim: decode the rendered
count from the up-block self-attention map at increasing spatial grid resolution.
If R^2 is ~0 when globally pooled (g=1) but rises with a spatial grid (g=4,8),
the count lives in spatial structure -> resolves the pooled-probe contradiction.
Thin Colab driver.

Run: python notebooks/_build_exp_spatialprobe.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Spatial probe: the count is in the spatial layout\n"
    "We decode the **rendered count** from the up0 self-attention saliency map at "
    "increasing grid resolution $g$ (average-pool into $g\\times g$ cells). "
    "$g{=}1$ is a global, spatially blind summary; larger $g$ preserves layout. "
    "If $R^2$ rises sharply with $g$, the count is encoded **spatially** - the "
    "positive result that the earlier pooled probe could not show.\n\n"
    "**Runtime:** GPU (~15 min)."))

c.append(nbf.v4.new_code_cell(
    "import os\n"
    "if not os.path.exists('src'):\n"
    "    !git clone https://github.com/serinaqin/T2I-Count-Anomaly.git\n"
    "    %cd T2I-Count-Anomaly\n"
    "!pip install -q -r requirements.txt\n"
    "!pip install -q pytest groundingdino-py"))

c.append(nbf.v4.new_code_cell(
    "import sys; sys.path.insert(0, '.')\n"
    "import numpy as np, pandas as pd, os, yaml\n"
    "import matplotlib.pyplot as plt\n"
    "from src.prompts import generate_grid\n"
    "from src.pipeline import (load_sdxl, catalog_attention_sites,\n"
    "                          select_probe_sites, generate_and_capture)\n"
    "from src.spatial import featuremap_saliency, grid_pool_2d\n"
    "from src.probes import fit_eval_magnitude\n"
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
    "# Capture the up0 self-attn saliency map (averaged over the 3 sites) + rendered count.\n"
    "grid = generate_grid(cfg.counts, cfg.objects, cfg.seeds)\n"
    "sal_maps, rows = [], []\n"
    "for i, p in enumerate(grid):\n"
    "    img, snaps = generate_and_capture(pipe, p.text, p.seed, sites, [step],\n"
    "                                      cfg.num_inference_steps, reducer=featuremap_saliency)\n"
    "    maps = [snaps.get(step, {}).get(s) for s in sites]\n"
    "    maps = [m for m in maps if m is not None]\n"
    "    sal_maps.append(np.mean(maps, axis=0) if maps else None)\n"
    "    rows.append({'obj': p.obj, 'count': p.count, 'seed': p.seed, 'rendered': cnt(img, p.obj)})\n"
    "    if (i + 1) % 20 == 0: print(f'{i+1}/{len(grid)}')\n"
    "df = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/exp_spatialprobe_counts.csv', index=False)\n"
    "print('captured', len(grid), 'maps of shape', sal_maps[0].shape)"))

c.append(nbf.v4.new_code_cell(
    "# Decode rendered count from g x g grid-pooled features, two ways:\n"
    "#  raw          = amount + arrangement (g=1 is total activation 'amount')\n"
    "#  arrangement  = sum-normalized per image (amount removed; g=1 is constant->R2~0)\n"
    "y_ren = df['rendered'].to_numpy(float)\n"
    "def feats(s, g, norm):\n"
    "    v = grid_pool_2d(s, g)\n"
    "    if norm:\n"
    "        t = v.sum(); v = v / t if t > 0 else v\n"
    "    return v\n"
    "res = []\n"
    "for g in grid_sizes:\n"
    "    Xr = np.array([feats(s, g, False) for s in sal_maps])\n"
    "    Xn = np.array([feats(s, g, True) for s in sal_maps])\n"
    "    res.append({'g': g, 'R2_raw': fit_eval_magnitude(Xr, y_ren)['r2'],\n"
    "                'R2_arrangement': fit_eval_magnitude(Xn, y_ren)['r2']})\n"
    "res = pd.DataFrame(res)\n"
    "res.to_csv('results/exp_spatialprobe_r2.csv', index=False)\n"
    "print(res.round(3))"))

c.append(nbf.v4.new_code_cell(
    "# R^2 vs grid resolution. The 'arrangement' curve isolates SPATIAL layout\n"
    "# (amount removed): if it rises above ~0 with g, the count is in the layout.\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "ax.plot(res['g'], res['R2_raw'], 'o-', label='raw (amount + arrangement)')\n"
    "ax.plot(res['g'], res['R2_arrangement'], 's-', label='arrangement only (amount removed)')\n"
    "ax.set_xscale('log', base=2); ax.set_xticks(res['g']); ax.set_xticklabels(res['g'])\n"
    "ax.set_xlabel('spatial grid resolution g (g=1 = global summary)')\n"
    "ax.set_ylabel('rendered-count probe R^2'); ax.set_ylim(-0.1, 1.0)\n"
    "ax.axhline(0, color='gray', lw=0.6)\n"
    "ax.set_title('Rendered count decodes from SPATIAL layout, not a global summary')\n"
    "ax.legend(); plt.tight_layout()\n"
    "plt.savefig('results/exp_spatialprobe_r2.png', dpi=110, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **`arrangement` (amount removed) starts ~0 at g=1 and rises with g** (e.g., "
    ">0.4 at g=4/8) = the rendered count is carried by the **spatial arrangement** "
    "of up-block self-attention, independent of total activation amount. This is "
    "the clean positive demonstration that the count is spatial, and it replaces "
    "the earlier 'pooled probe is weak, so we infer spatial' argument.\n"
    "- **`raw` high already at g=1** = total activation *amount* also tracks count "
    "(more objects -> more energy); informative but confounded, which is exactly "
    "why the amount-removed `arrangement` curve is the load-bearing one.\n"
    "- **Both flat/low at all g** = the saliency map does not linearly carry the "
    "count even spatially; we fall back to the unsupervised peak-count readout "
    "(Phase 2b, corr~0.5-0.65) and temper the 'spatial code' claim."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "exp_spatialprobe.ipynb")
nbf.write(nb, out)
print("wrote", out)

"""Generate notebooks/02b_spatial_instance_count.ipynb (Phase 2b).

Reads the count SPATIALLY: clusters up-block self-attention feature maps into
object blobs (an internal instance count), and finds where/when that internal
count diverges from the requested number. Thin Colab driver over src/.

Run: python notebooks/_build_02b.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Phase 2b — Spatial instance-count readout\n"
    "Phase 2's pooled probe was blind to spatial count. Here we read the count "
    "*spatially*: turn each up-block **self-attention** feature map into a "
    "saliency map, threshold + count blobs = an **internal instance count**, "
    "and see where/when it diverges from the requested number.\n\n"
    "**Question:** is the count wrong from the earliest step (set by the noise "
    "layout) or does it drift later (dynamic failure)?\n\n**Runtime:** GPU."))

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
    "from src.prompts import generate_grid\n"
    "from src.pipeline import (load_sdxl, catalog_attention_sites,\n"
    "                          select_probe_sites, generate_and_capture)\n"
    "from src.spatial import featuremap_saliency, count_peaks, find_peaks\n"
    "from src.detector import Detector\n"
    "from src.scoring import count_from_detections\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/phase2b.yaml')\n"
    "raw = yaml.safe_load(open('configs/phase2b.yaml'))\n"
    "PK = dict(sigma=raw['peak_sigma'], min_distance=raw['peak_min_distance'],\n"
    "          thresh_rel=raw['peak_thresh_rel'])\n"
    "grid = generate_grid(cfg.counts, cfg.objects, cfg.seeds)\n"
    "print(len(grid), 'images; steps', cfg.capture_steps, '; peak params', PK)"))

c.append(nbf.v4.new_code_cell(
    "pipe = load_sdxl()\n"
    "det = Detector()\n"
    "sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet))\n"
    "         if 'up_blocks' in s and s.endswith('attn1')]\n"
    "print(len(sites), 'up-block self-attention (image-side) sites:')\n"
    "print(sites)"))

c.append(nbf.v4.new_code_cell(
    "# Capture SALIENCY MAPS (not pooled) at each up-block attn1 site x step.\n"
    "sal_store = {st: {s: [] for s in sites} for st in cfg.capture_steps}\n"
    "images, rows = [], []\n"
    "for i, p in enumerate(grid):\n"
    "    img, snaps = generate_and_capture(pipe, p.text, p.seed, sites,\n"
    "                                      cfg.capture_steps, cfg.num_inference_steps,\n"
    "                                      reducer=featuremap_saliency)\n"
    "    rendered = count_from_detections(det.detect(img, [p.obj]), p.obj,\n"
    "                                     cfg.score_threshold)\n"
    "    images.append(img)\n"
    "    rows.append({'obj': p.obj, 'count': p.count, 'seed': p.seed, 'rendered': rendered})\n"
    "    for st in cfg.capture_steps:\n"
    "        for s in sites:\n"
    "            sal_store[st][s].append(snaps.get(st, {}).get(s))\n"
    "    if (i + 1) % 20 == 0:\n"
    "        print(f'{i+1}/{len(grid)}')\n"
    "df = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/phase2b_counts.csv', index=False)\n"
    "df.head()"))

c.append(nbf.v4.new_code_cell(
    "# Internal instance count = mean PEAK-count over up-block sites, per step.\n"
    "req = df['count'].to_numpy(float); ren = df['rendered'].to_numpy(float)\n"
    "def inst_at(st, s):\n"
    "    return np.array([count_peaks(m, **PK) if m is not None\n"
    "                     else np.nan for m in sal_store[st][s]])\n"
    "inst = {st: np.nanmean(np.vstack([inst_at(st, s) for s in sites]), axis=0)\n"
    "        for st in cfg.capture_steps}\n"
    "def safecorr(a, b):\n"
    "    a, b = np.asarray(a, float), np.asarray(b, float)\n"
    "    m = ~(np.isnan(a) | np.isnan(b))\n"
    "    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0: return np.nan\n"
    "    return float(np.corrcoef(a[m], b[m])[0, 1])\n"
    "print('step :  corr(internal, requested) | corr(internal, rendered) | mean internal')\n"
    "for st in cfg.capture_steps:\n"
    "    print(f'{st:>4} : {safecorr(inst[st], req):>25.2f} | '\n"
    "          f'{safecorr(inst[st], ren):>18.2f} | {np.nanmean(inst[st]):.2f}')"))

c.append(nbf.v4.new_code_cell(
    "# Money plot: mean internal instance count vs requested count, per step.\n"
    "import matplotlib.pyplot as plt\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "for st in cfg.capture_steps:\n"
    "    means = [np.nanmean(inst[st][req == c]) for c in cfg.counts]\n"
    "    ax.plot(cfg.counts, means, marker='o', label=f'step {st}')\n"
    "ax.plot(cfg.counts, cfg.counts, 'k--', label='y=x (internal = requested)')\n"
    "ax.set_xlabel('requested count'); ax.set_ylabel('mean internal instance count')\n"
    "ax.set_title('Does the internal spatial count track the request?')\n"
    "ax.legend(fontsize=8); plt.tight_layout()\n"
    "plt.savefig('results/phase2b_internal_vs_requested.png', dpi=100, bbox_inches='tight')\n"
    "plt.show()"))

c.append(nbf.v4.new_code_cell(
    "# Eyeball: validate that the saliency segments objects. image | saliency.\n"
    "site0, st0 = sites[0], (15 if 15 in cfg.capture_steps else cfg.capture_steps[len(cfg.capture_steps)//2])\n"
    "order = np.argsort(df['count'].to_numpy())\n"
    "pick = order[np.linspace(0, len(order) - 1, 6).astype(int)]\n"
    "fig, axes = plt.subplots(len(pick), 2, figsize=(7, 3 * len(pick)))\n"
    "for row, idx in enumerate(pick):\n"
    "    sal = sal_store[st0][site0][idx]\n"
    "    pk = find_peaks(sal, **PK) if sal is not None else np.empty((0, 2))\n"
    "    axes[row, 0].imshow(images[idx]); axes[row, 0].axis('off')\n"
    "    axes[row, 0].set_title(f\"asked {df['count'][idx]} rendered {df['rendered'][idx]}\", fontsize=9)\n"
    "    axes[row, 1].imshow(sal, cmap='magma'); axes[row, 1].axis('off')\n"
    "    if len(pk):\n"
    "        axes[row, 1].scatter(pk[:, 1], pk[:, 0], c='cyan', s=60, marker='x')\n"
    "    axes[row, 1].set_title(f'saliency+peaks @ {site0.split(\".\")[1]} step {st0} | peaks={len(pk)}', fontsize=8)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase2b_eyeball.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **corr(internal, requested) high at early steps, then falling** = the "
    "layout starts honoring the number and loses it as denoising proceeds -> "
    "the count is lost DURING allocation; the step where it drops is where.\n"
    "- **corr(internal, requested) low from the earliest step** = the spatial "
    "count is wrong from the start -> set by the initial noise layout, not "
    "drift.\n"
    "- **corr(internal, rendered) rising toward late steps** = the internal "
    "blob-count converges to the eventual (wrong) output -> our readout is real "
    "and the error crystallizes by that step.\n"
    "- **Money plot:** lines on the diagonal at some step = internal count "
    "matches the request there; lines flattening ABOVE the diagonal = "
    "over-allocation (too many blobs), matching Phase 1's over-generation.\n"
    "- **Eyeball first!** If the saliency maps don't land on the objects, raise/"
    "lower `saliency_thresh` in `configs/phase2b.yaml` before trusting the "
    "counts. The best-tracking site+step becomes the Phase 2c causal-patch target."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "02b_spatial_instance_count.ipynb")
nbf.write(nb, out)
print("wrote", out)

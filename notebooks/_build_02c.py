"""Generate notebooks/02c_causal_patch.ipynb (Phase 2c — causal patch).

Injects a donor count's up-block self-attention activations into a source-count
run at the commitment steps and measures whether the OUTPUT count moves. Tests
CAUSATION (both directions), scored by the detector. Thin Colab driver.

Run: python notebooks/_build_02c.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Phase 2c — Causal activation patch\n"
    "Does the up-block **self-attention** site *cause* the count? For each seed "
    "we capture the attn1 activations from a **donor** prompt (e.g. 'five cats') "
    "and inject them into a **source** run ('two cats') at the commitment steps, "
    "then check whether the OUTPUT count moves toward the donor.\n\n"
    "Both directions (2->inject5 should raise; 5->inject2 should lower). Effect "
    "in both directions = causal; ~0 = correlational (like prior steering).\n\n"
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
    "import numpy as np, pandas as pd, os, yaml\n"
    "from src.prompts import build_prompt\n"
    "from src.pipeline import (load_sdxl, generate, catalog_attention_sites,\n"
    "                          select_probe_sites, generate_and_capture,\n"
    "                          raw_reducer, generate_with_patch)\n"
    "from src.detector import Detector\n"
    "from src.scoring import count_from_detections\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/phase2c.yaml')\n"
    "raw = yaml.safe_load(open('configs/phase2c.yaml'))\n"
    "pairs, patch_steps, block = raw['pairs'], raw['patch_steps'], raw['patch_block']\n"
    "obj = cfg.objects[0]\n"
    "print('pairs', pairs, '| patch steps', patch_steps, '| block', block, '| obj', obj)"))

c.append(nbf.v4.new_code_cell(
    "pipe = load_sdxl()\n"
    "det = Detector()\n"
    "sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet))\n"
    "         if block in s and s.endswith('attn1')]\n"
    "print(len(sites), 'patch sites:', sites)\n"
    "def cnt(img):\n"
    "    return count_from_detections(det.detect(img, [obj]), obj, cfg.score_threshold)"))

c.append(nbf.v4.new_code_cell(
    "# For each (source, donor) pair and seed: donor-capture, baseline, patched.\n"
    "rows, gallery = [], []\n"
    "for src, dnr in pairs:\n"
    "    sp, dp = build_prompt(src, obj), build_prompt(dnr, obj)\n"
    "    for seed in cfg.seeds:\n"
    "        img_d, snaps = generate_and_capture(pipe, dp, seed, sites, patch_steps,\n"
    "                                            cfg.num_inference_steps, reducer=raw_reducer)\n"
    "        img_b = generate(pipe, sp, seed, cfg.num_inference_steps)\n"
    "        img_p = generate_with_patch(pipe, sp, seed, snaps, cfg.num_inference_steps)\n"
    "        cd, cb, cp = cnt(img_d), cnt(img_b), cnt(img_p)\n"
    "        direction = 'up' if src < dnr else 'down'\n"
    "        rows.append({'src': src, 'dnr': dnr, 'seed': seed, 'direction': direction,\n"
    "                     'c_donor': cd, 'c_base': cb, 'c_patch': cp})\n"
    "        if direction == 'up':\n"
    "            gallery.append((seed, img_b, img_p, img_d, cb, cp, cd))\n"
    "df = pd.DataFrame(rows)\n"
    "df['delta'] = df['c_patch'] - df['c_base']\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/phase2c_patch.csv', index=False)\n"
    "df"))

c.append(nbf.v4.new_code_cell(
    "# Effect per direction: does patching move the count the expected way?\n"
    "for direction, g in df.groupby('direction'):\n"
    "    md = g['delta'].mean()\n"
    "    frac = (g['delta'] > 0).mean() if direction == 'up' else (g['delta'] < 0).mean()\n"
    "    print(f'{direction:>4}: mean(patched-baseline) = {md:+.2f} | '\n"
    "          f'baseline={g.c_base.mean():.2f} patched={g.c_patch.mean():.2f} '\n"
    "          f'donor={g.c_donor.mean():.2f} | frac moved expected way = {frac:.2f}')"))

c.append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 5))\n"
    "col = {'up': 'C0', 'down': 'C3'}\n"
    "for direction, g in df.groupby('direction'):\n"
    "    for _, r in g.iterrows():\n"
    "        axes[0].plot([0, 1], [r.c_base, r.c_patch], color=col[direction],\n"
    "                     alpha=0.5, marker='o')\n"
    "axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['baseline', 'patched'])\n"
    "axes[0].set_ylabel('output count')\n"
    "axes[0].set_title('Per-seed: does injecting the donor move the count?')\n"
    "means = df.groupby('direction')['delta'].mean()\n"
    "axes[1].bar(means.index, means.values, color=[col[d] for d in means.index])\n"
    "axes[1].axhline(0, color='k', lw=0.8)\n"
    "axes[1].set_ylabel('mean (patched - baseline) count')\n"
    "axes[1].set_title('Causal effect (up should be +, down should be -)')\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase2c_effect.png', dpi=100, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_code_cell(
    "# Eyeball: baseline | patched | donor for a few seeds (up direction).\n"
    "show = gallery[:3]\n"
    "fig, axes = plt.subplots(len(show), 3, figsize=(9, 3 * len(show)))\n"
    "if len(show) == 1: axes = axes[None, :]\n"
    "for row, (seed, ib, ip, idn, cb, cp, cd) in enumerate(show):\n"
    "    for col_i, (im, ti) in enumerate([(ib, f'baseline={cb}'), (ip, f'patched={cp}'),\n"
    "                                      (idn, f'donor={cd}')]):\n"
    "        axes[row, col_i].imshow(im); axes[row, col_i].axis('off')\n"
    "        axes[row, col_i].set_title(f'seed {seed} | {ti}', fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase2c_eyeball.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **up mean delta clearly > 0 AND down mean delta clearly < 0** = injecting "
    "the donor's up-block self-attention *moves the output count toward the "
    "donor* -> this site **causally controls** the count. That's the realization "
    "mechanism; Phase 4 mitigation intervenes here.\n"
    "- **delta ~ 0 (or only one direction works)** = correlational, not causal "
    "(the same result as the project's earlier steering). The count is set "
    "elsewhere -> widen the patch (more steps/sites), move to a different block, "
    "or upstream toward the initial noise.\n"
    "- **Eyeball:** in the up rows, does the patched image actually show more "
    "animals than the baseline (toward the donor)? Counts only mean something if "
    "the images visibly change."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "02c_causal_patch.ipynb")
nbf.write(nb, out)
print("wrote", out)

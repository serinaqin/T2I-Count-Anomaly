"""Generate notebooks/02c3_causal_confirm_timing.ipynb (Phase 2c-v3).

Confirms the causal site found in 2c-v2 (up_blocks.0 self-attn, early steps):
(A) EYEBALL the patched images (coherent count change vs degradation) + a second
object for robustness; (B) FINE TIMING - cumulative early windows to pinpoint
when the count locks. Reuses existing patch primitives. Thin Colab driver.

Run: python notebooks/_build_02c3.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Phase 2c-v3 - Confirm the causal site + pinpoint the timing\n"
    "2c-v2 found: patching **up_blocks.0 self-attention at early steps (0-5)** "
    "moves the output count toward the donor (~45%, both directions). Here we "
    "**(A)** eyeball the patched images (real count change, not mush?) + test a "
    "second object, and **(B)** sweep cumulative early windows to find *when* the "
    "count locks.\n\n**Runtime:** GPU (~20 min)."))

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
    "from src.prompts import build_prompt\n"
    "from src.pipeline import (load_sdxl, generate, catalog_attention_sites,\n"
    "                          select_probe_sites, generate_and_capture,\n"
    "                          raw_reducer, generate_with_patch)\n"
    "from src.detector import Detector\n"
    "from src.scoring import count_from_detections\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/phase2c3.yaml')\n"
    "raw = yaml.safe_load(open('configs/phase2c3.yaml'))\n"
    "pairs = raw['pairs']; cap_steps = raw['capture_steps']\n"
    "block, attn = raw['patch_block'], raw['patch_attn']\n"
    "pipe = load_sdxl(); det = Detector()\n"
    "sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet))\n"
    "         if block in s and s.endswith(attn)]\n"
    "print('patch sites:', sites)\n"
    "def cnt(img, label):\n"
    "    return count_from_detections(det.detect(img, [label]), label, cfg.score_threshold)\n"
    "def donor(pipe, prompt, seed):\n"
    "    return generate_and_capture(pipe, prompt, seed, sites, cap_steps,\n"
    "                                cfg.num_inference_steps, reducer=raw_reducer)"))

c.append(nbf.v4.new_markdown_cell(
    "## Part A - Eyeball + robustness (full early window 0-5, cat & dog)"))

c.append(nbf.v4.new_code_cell(
    "rowsA, gallery = [], []\n"
    "for obj in raw['objects']:\n"
    "    for src, dnr in pairs:\n"
    "        direction = 'up' if src < dnr else 'down'\n"
    "        sp, dp = build_prompt(src, obj), build_prompt(dnr, obj)\n"
    "        for seed in raw['eyeball_seeds']:\n"
    "            img_d, snaps = donor(pipe, dp, seed)\n"
    "            img_b = generate(pipe, sp, seed, cfg.num_inference_steps)\n"
    "            pm = {st: snaps[st] for st in cap_steps if st in snaps}\n"
    "            img_p = generate_with_patch(pipe, sp, seed, pm, cfg.num_inference_steps)\n"
    "            cb, cp, cd = cnt(img_b, obj), cnt(img_p, obj), cnt(img_d, obj)\n"
    "            rowsA.append({'obj': obj, 'direction': direction, 'seed': seed,\n"
    "                          'c_base': cb, 'c_patch': cp, 'c_donor': cd})\n"
    "            gallery.append((obj, direction, seed, img_b, img_p, img_d, cb, cp, cd))\n"
    "dfA = pd.DataFrame(rowsA)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "dfA.to_csv('results/phase2c3_partA.csv', index=False)\n"
    "for (obj, direction), g in dfA.groupby(['obj', 'direction']):\n"
    "    sd = (g.c_patch - g.c_base) if direction == 'up' else (g.c_base - g.c_patch)\n"
    "    print(f'{obj:>4} {direction:>4}: base={g.c_base.mean():.2f} patch={g.c_patch.mean():.2f} '\n"
    "          f'donor={g.c_donor.mean():.2f} | donor-directed delta={sd.mean():+.2f}')"))

c.append(nbf.v4.new_code_cell(
    "# Eyeball: baseline | patched | donor for one seed of each obj x direction.\n"
    "seen, show = set(), []\n"
    "for g in gallery:\n"
    "    key = (g[0], g[1])\n"
    "    if key not in seen:\n"
    "        seen.add(key); show.append(g)\n"
    "fig, axes = plt.subplots(len(show), 3, figsize=(9, 3 * len(show)))\n"
    "for row, (obj, direction, seed, ib, ip, idn, cb, cp, cd) in enumerate(show):\n"
    "    for j, (im, ti) in enumerate([(ib, f'base={cb}'), (ip, f'patched={cp}'), (idn, f'donor={cd}')]):\n"
    "        axes[row, j].imshow(im); axes[row, j].axis('off')\n"
    "        axes[row, j].set_title(f'{obj} {direction} s{seed} | {ti}', fontsize=8)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase2c3_eyeball.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## Part B - Fine timing: cumulative early windows (when does the count lock?)"))

c.append(nbf.v4.new_code_cell(
    "obj = raw['timing_object']; windows = raw['cumulative_windows']\n"
    "rowsB = []\n"
    "for src, dnr in pairs:\n"
    "    direction = 'up' if src < dnr else 'down'\n"
    "    sp, dp = build_prompt(src, obj), build_prompt(dnr, obj)\n"
    "    for seed in raw['timing_seeds']:\n"
    "        img_d, snaps = donor(pipe, dp, seed)\n"
    "        cb = cnt(generate(pipe, sp, seed, cfg.num_inference_steps), obj)\n"
    "        for w in windows:\n"
    "            pm = {st: snaps[st] for st in w if st in snaps}\n"
    "            cp = cnt(generate_with_patch(pipe, sp, seed, pm, cfg.num_inference_steps), obj)\n"
    "            sd = (cp - cb) if direction == 'up' else (cb - cp)\n"
    "            rowsB.append({'direction': direction, 'seed': seed,\n"
    "                          'last_step': max(w), 'signed_delta': sd})\n"
    "    print(direction, 'done')\n"
    "dfB = pd.DataFrame(rowsB)\n"
    "dfB.to_csv('results/phase2c3_partB.csv', index=False)\n"
    "dfB.groupby(['direction', 'last_step'])['signed_delta'].mean().unstack(0)"))

c.append(nbf.v4.new_code_cell(
    "# How the effect grows as more early steps are included.\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "for direction, g in dfB.groupby('direction'):\n"
    "    m = g.groupby('last_step')['signed_delta'].mean()\n"
    "    ax.plot(m.index, m.values, marker='o', label=direction)\n"
    "ax.axhline(0, color='k', lw=0.8)\n"
    "ax.set_xlabel('last early step included in the patch')\n"
    "ax.set_ylabel('donor-directed delta')\n"
    "ax.set_title('When does the count lock? (effect vs how many early steps patched)')\n"
    "ax.legend(); plt.tight_layout()\n"
    "plt.savefig('results/phase2c3_timing.png', dpi=100, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "**Part A (eyeball):** in the patched column, do you SEE more/fewer animals "
    "than baseline (toward the donor), in coherent scenes? If yes -> the causal "
    "effect is real. If the patched images are degraded/mushy -> the count change "
    "was an artifact and we reconsider. Robustness: does dog show the same "
    "donor-directed effect as cat?\n\n"
    "**Part B (timing):** where the curve rises then flattens tells us the count "
    "is locked by that step. If it saturates after including step ~2-3, the count "
    "commits in the first 2-3 denoising steps (very early / noise-driven). If it "
    "keeps rising through step 5, the window is broader.\n\n"
    "**Next:** with a confirmed causal site + window, Phase 4 mitigation steers "
    "this early up-block self-attention toward the REQUESTED count (no donor)."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "02c3_causal_confirm_timing.ipynb")
nbf.write(nb, out)
print("wrote", out)

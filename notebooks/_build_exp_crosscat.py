"""Generate notebooks/exp_crosscat_causal.ipynb (cross-category causal patching).

The crux control for "count code" vs "scene/layout template": inject a donor's
early up0 self-attention into a "two cats" source and score BOTH categories.
If a "five dogs" donor makes 5 CATS -> count transfers without category (abstract
count/layout code). If dogs appear -> category leaks (scene template). Includes
same-count and cross-category donor controls. Thin Colab driver.

Run: python notebooks/_build_exp_crosscat.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Cross-category causal patching + controls\n"
    "Source = **'two cats'**. For a fixed seed we inject the early up0 "
    "self-attention of four donors and score the output for BOTH cats and dogs.\n\n"
    "- `same-cat_cross-count` (five cats): positive control - count should rise, cats.\n"
    "- `KEY_cross-cat_cross-count` (five dogs): **the crux** - if output = ~5 CATS, "
    "count transfers without category (abstract count/layout code); if dogs appear, "
    "category leaks (scene template).\n"
    "- `ctrl_*_same-count` (two cats / two dogs): count should NOT move (isolates the "
    "donor-count effect from mere patch perturbation; and tests category leakage "
    "without a count change).\n\n**Runtime:** GPU (~12 min)."))

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
    "cfg = load_config('configs/exp_crosscat.yaml')\n"
    "raw = yaml.safe_load(open('configs/exp_crosscat.yaml'))\n"
    "psteps = raw['patch_steps']; block, attn = raw['patch_block'], raw['patch_attn']\n"
    "src_cat, src_count, donors = raw['source_cat'], raw['source_count'], raw['donors']\n"
    "cats = raw['objects']  # ['cat','dog'] - the two labels we always score\n"
    "pipe = load_sdxl(); det = Detector()\n"
    "sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet))\n"
    "         if block in s and s.endswith(attn)]\n"
    "def counts_both(img):\n"
    "    dets = det.detect(img, cats)\n"
    "    return {a: count_from_detections(dets, a, cfg.score_threshold) for a in cats}\n"
    "print('sites:', sites, '| labels scored:', cats)"))

c.append(nbf.v4.new_code_cell(
    "rows = []\n"
    "for seed in cfg.seeds:\n"
    "    sp = build_prompt(src_count, src_cat)          # 'two cats'\n"
    "    b = generate(pipe, sp, seed, cfg.num_inference_steps)\n"
    "    cb = counts_both(b)\n"
    "    rows.append({'seed': seed, 'cond': 'baseline', 'donor_count': src_count,\n"
    "                 'donor_cat': src_cat, **{f'out_{a}': cb[a] for a in cats}})\n"
    "    for dcount, dcat, label in donors:\n"
    "        dp = build_prompt(dcount, dcat)\n"
    "        _, snaps = generate_and_capture(pipe, dp, seed, sites, psteps,\n"
    "                                        cfg.num_inference_steps, reducer=raw_reducer)\n"
    "        p = generate_with_patch(pipe, sp, seed, snaps, cfg.num_inference_steps)\n"
    "        cp = counts_both(p)\n"
    "        rows.append({'seed': seed, 'cond': label, 'donor_count': dcount,\n"
    "                     'donor_cat': dcat, **{f'out_{a}': cp[a] for a in cats}})\n"
    "    print('seed', seed, 'done')\n"
    "df = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/exp_crosscat.csv', index=False)\n"
    "df.groupby('cond')[[f'out_{a}' for a in cats]].mean().round(2)"))

c.append(nbf.v4.new_code_cell(
    "# Summary: for each condition, mean source-cat count and mean donor-cat (dog) count.\n"
    "summ = df.groupby(['cond', 'donor_cat', 'donor_count'])[[f'out_{a}' for a in cats]].mean().round(2)\n"
    "print(summ)\n"
    "base = df[df.cond == 'baseline'][[f'out_{a}' for a in cats]].mean()\n"
    "print('\\nbaseline (two cats):', {a: round(base[f'out_{a}'], 2) for a in cats})"))

c.append(nbf.v4.new_code_cell(
    "# Bar: output cat-count vs dog-count per condition (the crux is cross-cat/cross-count).\n"
    "m = df.groupby('cond')[[f'out_{a}' for a in cats]].mean()\n"
    "order = ['baseline', 'ctrl_same-cat_same-count', 'same-cat_cross-count',\n"
    "         'ctrl_cross-cat_same-count', 'KEY_cross-cat_cross-count']\n"
    "m = m.reindex([o for o in order if o in m.index])\n"
    "ax = m.plot(kind='bar', figsize=(10, 5))\n"
    "ax.set_ylabel('mean output count'); ax.set_title('Cross-category patch: does COUNT transfer without CATEGORY?')\n"
    "ax.legend(title='detected as'); plt.xticks(rotation=20, ha='right')\n"
    "plt.tight_layout(); plt.savefig('results/exp_crosscat_bars.png', dpi=100, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_code_cell(
    "# Eyeball: baseline vs each donor for one seed. Are the added animals cats or dogs?\n"
    "seed0 = cfg.seeds[0]; sp = build_prompt(src_count, src_cat)\n"
    "panels = [('baseline', generate(pipe, sp, seed0, cfg.num_inference_steps))]\n"
    "for dcount, dcat, label in donors:\n"
    "    _, snaps = generate_and_capture(pipe, build_prompt(dcount, dcat), seed0, sites,\n"
    "                                    psteps, cfg.num_inference_steps, reducer=raw_reducer)\n"
    "    panels.append((f'{dcount} {dcat}', generate_with_patch(pipe, sp, seed0, snaps, cfg.num_inference_steps)))\n"
    "fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.3))\n"
    "for ax, (name, im) in zip(axes, panels):\n"
    "    cc = counts_both(im)\n"
    "    ax.imshow(im); ax.axis('off')\n"
    "    ax.set_title(f'{name}\\ncat={cc[cats[0]]} dog={cc[cats[1]]}', fontsize=8)\n"
    "plt.tight_layout(); plt.savefig('results/exp_crosscat_eyeball.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "Focus on **KEY_cross-cat_cross-count** (donor = five dogs, source = two cats):\n"
    "- **Output ~5 CATS, ~0 dogs** -> the count/layout transfers but the category "
    "does NOT -> an abstract, category-invariant **count/layout code**. Strong result; "
    "'count code' is earned.\n"
    "- **Output shows DOGS (or a cat/dog mix scaling with the donor)** -> the donor "
    "transplanted a *scene/instance layout* tied to its content -> reframe from 'count "
    "code' to **object-layout code** (count is emergent). Also a clean, publishable "
    "conclusion - just a different one.\n\n"
    "**Controls:** `ctrl_*_same-count` (two cats / two dogs donor) should leave the "
    "cat-count near the baseline (~2): if it does, the count shift in the cross-count "
    "conditions is driven by the donor's COUNT, not by mere patch perturbation. If the "
    "'two dogs' control makes dogs appear without changing the count, category leaks "
    "spatially even without a count change (informative for the layout-vs-count question)."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "exp_crosscat_causal.ipynb")
nbf.write(nb, out)
print("wrote", out)
